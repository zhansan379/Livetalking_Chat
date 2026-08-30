###############################################################################
#  观测平台公开 API。
#
#  外部用法：
#    - server/routes.stream_llm_chat   : begin_trace(...) → try/finally → end_trace(...)
#    - agent/tool_loop.run_tool_loop    : async with round_span(i) as r: ...  + emit(tool_call)
#    - agent/agent._call_summarize      : with new_trace(sid, kind="summary"):
#    - infra_ai (经 obs_hook)           : 自动 emit(llm_call)，无需改调用方
#    - server/routes.setup_routes       : obs.setup_routes(app)
#    - app.py main()                    : obs.install()
###############################################################################

from .config import is_enabled
from .recorder import Tracer

_tracer: Tracer | None = None


def _get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def install() -> None:
    """启动时调用：连接 infra_ai 的 obs_hook，开始收集 LLM 调用事件。"""
    global _tracer
    _tracer = Tracer()
    if not is_enabled():
        return
    try:
        from infra_ai.obs_hook import set_obs
        set_obs(_tracer.ingest)
    except Exception:  # noqa: BLE001 - 观测挂钩失败不影响主流程
        pass


# ─── 请求级 trace ────────────────────────────────────────────────────────
def begin_trace(session_id: str, msg_preview: str | None,
                tool_mode: bool | None = None, kind: str = "chat",
                trace_id: str | None = None, bound_tools: list[str] | None = None):
    return _get_tracer().begin_trace(session_id, msg_preview, tool_mode, kind,
                                     trace_id=trace_id, bound_tools=bound_tools)


def end_trace(success: bool, fail_reason: str | None = None, text_len: int = 0,
              circuit_open: bool = False, tool_rounds: int | None = None,
              llm_calls: int | None = None) -> None:
    _get_tracer().end_trace(success, fail_reason, text_len, circuit_open,
                            tool_rounds, llm_calls)


def emit(event: dict) -> None:
    _get_tracer().emit(event)


def emit_explicit(event: dict, *, trace_id: str | None, session_id: str | None,
                  parent_id: str | None = None, kind: str = "chat") -> None:
    """跨线程显式 ID 的 emit（base_tts 的 TTS 工作线程用它挂回聊天 trace）。"""
    _get_tracer().emit_explicit(event, trace_id=trace_id, session_id=session_id,
                                parent_id=parent_id, kind=kind)


def round_span(round_idx: int):
    return _get_tracer().round_span(round_idx)


def new_trace(session_id: str, kind: str = "summary"):
    return _get_tracer().new_trace(session_id, kind)


def setup_routes(app) -> None:
    from .routes import register
    register(app)


__all__ = [
    "install", "is_enabled", "begin_trace", "end_trace", "emit",
    "emit_explicit", "round_span", "new_trace", "setup_routes",
]