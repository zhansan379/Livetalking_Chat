###############################################################################
#  Tracer：contextvars 驱动的请求级 trace——无需在各函数签名间手动传 ID。
#
#  一次用户请求 = 一个 async task（server/routes.stream_llm_chat 由
#  asyncio.create_task 触发，整条链路同属该 task）。在 task 开头由
#  begin_trace 设置的 contextvars 会自动向下传播到每个子 await，因此
#  infra_ai/_invoke_with_retry、agent/tool_loop 只需调用 emit() 打事件，
#  tracer 从当前 ctx 自动补上 trace_id/parent_id 完成嵌套。
#
#  上下文变量：
#    _TRACE_ID    当前请求 root id（uuid hex）
#    _PARENT_ID   当前 span 的父 id；round_span 进入时改为 round span id，
#                 期间发出的 llm_call/tool_call 自动挂到该轮下
#    _SESSION_ID  会话 id
#    _KIND        trace 类型（chat / summary）
#    _START_MS    该 trace 的 monotonic 起点（用于计算响应耗时）
#    _ROUND_CTR   本 trace 内「真正调了工具」的轮数
#    _LLM_CTR     本 trace 内的 llm_call 事件数
###############################################################################

import contextlib
import contextvars
import datetime
import threading
import time
import uuid

from .config import is_enabled
from .writer import JsonlWriter

_EPOCH = time.monotonic()  # 进程级基准，recorder 与 query 共用 now_ms()


def now_ms() -> float:
    return (time.monotonic() - _EPOCH) * 1000.0


_TRACE_ID: contextvars.ContextVar = contextvars.ContextVar("obs_trace_id", default=None)
_PARENT_ID: contextvars.ContextVar = contextvars.ContextVar("obs_parent_id", default=None)
_SESSION_ID: contextvars.ContextVar = contextvars.ContextVar("obs_session_id", default=None)
_KIND: contextvars.ContextVar = contextvars.ContextVar("obs_kind", default="chat")
_START_MS: contextvars.ContextVar = contextvars.ContextVar("obs_start_ms", default=None)
_ROUND_CTR: contextvars.ContextVar = contextvars.ContextVar("obs_round_ctr", default=0)
_LLM_CTR: contextvars.ContextVar = contextvars.ContextVar("obs_llm_ctr", default=0)


class _Round:
    """round_span 返回句柄，用于在主体内回填本轮的元数据。"""

    n_tool_calls: int = 0


class Tracer:
    def __init__(self) -> None:
        self._writer = JsonlWriter()
        self._seq = 0
        self._seq_lock = threading.Lock()

    # ─── 事件写入 ──────────────────────────────────────────────────────────
    def _next_seq(self) -> int:
        with self._seq_lock:
            self._seq += 1
            return self._seq

    def emit(self, event: dict) -> None:
        """供 obs_hook 与工具循环调用的公开入口：自动打信封并落盘。"""
        if not is_enabled():
            return
        # 信封：seq/ts/ms/trace_id/session_id/parent_id 由 ctx 提供，
        # 其余字段（type/span_id/kind/业务字段）由调用方传入。
        ev: dict = {
            "seq": self._next_seq(),
            "ts": _now_iso(),
            "ms": round(now_ms(), 3),
            "trace_id": _TRACE_ID.get(),
            "session_id": _SESSION_ID.get(),
            "parent_id": _PARENT_ID.get(),
        }
        for k, v in event.items():
            if k not in ("seq", "ts", "ms"):
                ev[k] = v
        if "kind" not in ev:
            ev["kind"] = _KIND.get()
        if ev.get("type") == "llm_call":
            _LLM_CTR.set(_LLM_CTR.get() + 1)
        self._writer.append(ev)

    # 供 infra_ai/obs_hook 回调使用（与 emit 同义，避免语义耦合命名）
    def ingest(self, event: dict) -> None:
        self.emit(event)

    # ─── 请求级 trace ──────────────────────────────────────────────────────
    def begin_trace(self, session_id: str, msg_preview: str | None,
                    tool_mode: bool | None = None, kind: str = "chat") -> str | None:
        if not is_enabled():
            return None
        trace_id = uuid.uuid4().hex
        _TRACE_ID.set(trace_id)
        _SESSION_ID.set(session_id)
        _KIND.set(kind)
        _START_MS.set(now_ms())
        _ROUND_CTR.set(0)
        _LLM_CTR.set(0)
        # 根事件：span_id = trace_id，parent_id = None
        self.emit({
            "type": "trace_start",
            "span_id": trace_id,
            "parent_id": None,
            "kind": kind,
            "msg_preview": msg_preview,
            "tool_mode": bool(tool_mode),
        })
        _PARENT_ID.set(trace_id)  # 之后的事件都挂到该 trace 下
        return trace_id

    def end_trace(self, success: bool, fail_reason: str | None = None,
                  text_len: int = 0, circuit_open: bool = False,
                  tool_rounds: int | None = None,
                  llm_calls: int | None = None) -> None:
        if not is_enabled():
            return
        start = _START_MS.get() or now_ms()
        self.emit({
            "type": "trace_end",
            "span_id": _TRACE_ID.get(),
            "elapsed_ms": round(now_ms() - start, 1),   # = 响应耗时
            "success": bool(success),
            "fail_reason": fail_reason,
            "tool_rounds": tool_rounds if tool_rounds is not None else _ROUND_CTR.get(),
            "llm_calls": llm_calls if llm_calls is not None else _LLM_CTR.get(),
            "text_len": int(text_len or 0),
            "circuit_open": bool(circuit_open),
        })

    # ─── 工具轮 span ───────────────────────────────────────────────────────
    @contextlib.asynccontextmanager
    async def round_span(self, round_idx: int):
        r = _Round()
        parent_before = _PARENT_ID.get()
        span_id = f"round-{round_idx}-{uuid.uuid4().hex[:6]}"
        _PARENT_ID.set(span_id)
        try:
            yield r
        finally:
            root = _TRACE_ID.get()
            # 只有当本轮真的调了工具才算一轮工具调用
            if r.n_tool_calls > 0:
                _ROUND_CTR.set(_ROUND_CTR.get() + 1)
            self.emit({
                "type": "tool_round",
                "span_id": span_id,
                "parent_id": root,
                "round": round_idx,
                "n_tool_calls": r.n_tool_calls,
                "answered": r.n_tool_calls <= 0,
            })
            _PARENT_ID.set(parent_before)

    # ─── 独立后台 trace（压缩摘要用，不污染用户请求统计）───────────────
    @contextlib.contextmanager
    def new_trace(self, session_id: str, kind: str = "summary"):
        if not is_enabled():
            yield
            return
        saved = (
            _TRACE_ID.get(), _PARENT_ID.get(), _SESSION_ID.get(),
            _KIND.get(), _START_MS.get(),
        )
        _TRACE_ID.set(None)
        ok = True
        try:
            trace_id = uuid.uuid4().hex
            _TRACE_ID.set(trace_id)
            _SESSION_ID.set(session_id)
            _KIND.set(kind)
            _START_MS.set(now_ms())
            _ROUND_CTR.set(0)
            _LLM_CTR.set(0)
            self.emit({
                "type": "trace_start",
                "span_id": trace_id,
                "parent_id": None,
                "kind": kind,
                "msg_preview": None,
                "tool_mode": False,
            })
            _PARENT_ID.set(trace_id)
            try:
                yield
            except Exception:
                ok = False
                raise
            finally:
                self.end_trace(success=ok, text_len=0)
        finally:
            _TRACE_ID.set(saved[0])
            _PARENT_ID.set(saved[1])
            _SESSION_ID.set(saved[2])
            _KIND.set(saved[3])
            _START_MS.set(saved[4])


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="milliseconds")