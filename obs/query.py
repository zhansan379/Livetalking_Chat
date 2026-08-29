###############################################################################
#  query：从 JSONL 事件文件读取并聚合，供 /api/obs/* 与面板使用。
#  纯函数，不依赖 aiohttp；窗口过滤用 monotonic ms（与 recorder.now_ms 同基准）。
###############################################################################

import json

from .config import query_limit, query_window
from .recorder import now_ms
from .writer import iter_event_files

# 事件读取限制：文件再大也最多扫描前 max_read 事件，防止失控崩服务
_MAX_READ = 200_000

# 后台维护类 trace：不参与用户请求统计/列表/时间线。新增后台任务
# （如长期记忆提取/整理）需把 kind 纳入此处，否则会污染图表与指标。
_BACKGROUND_KINDS = ("summary", "longterm_extract", "longterm_consolidate")


def _read_events() -> list[dict]:
    events: list[dict] = []
    for path in iter_event_files():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 半行/损坏行跳过
                    events.append(ev)
                    if len(events) >= _MAX_READ:
                        return events
        except OSError:
            continue
    events.sort(key=lambda e: e.get("seq", 0))
    return events


def _in_window(ev: dict, window: int | None) -> bool:
    if window is None:
        return True
    try:
        return (now_ms() - float(ev.get("ms", 0))) <= window * 1000.0
    except (TypeError, ValueError):
        return True


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = int((len(sorted_vals) - 1) * pct)
    return round(sorted_vals[idx], 1)


def _pipeline_bounds(events: list[dict]) -> dict[str, dict]:
    """对每条 chat 聊天 trace，算出全链路的起止 monotonic 时间点。

    单条全链路 trace 已把 ASR/LLM/工具/TTS 拼在一起（asr_call 也可能异步晚于
    trace_end 的 tts_call 都在同一 trace_id 下）。这里收集：
      - chat_tids：拥有 kind!=asr/summary 的 root trace_start 的 tid（排除独立
        asr trace——它没有 chat trace_start，天然不会进入全链路耗时）。
      - 每 trace 的 min（首事件；asr_call 按其 emit 时刻减去推理耗时作为近似起点）
        与 max（末事件，可能是异步 TTS 合成完成，晚于 trace_end）。
    返回 {tid: {"min": float, "max": float}}，供 summary/requests 拼 pipeline_ms。
    """
    chat_tids: set[str] = set()
    for ev in events:
        if ev.get("type") == "trace_start" and ev.get("kind") not in ("asr", *_BACKGROUND_KINDS):
            chat_tids.add(ev.get("trace_id"))
    bounds: dict[str, dict] = {}
    for ev in events:
        tid = ev.get("trace_id")
        if not tid or tid not in chat_tids:
            continue
        ms = float(ev.get("ms", 0) or 0)
        lo = ms
        # asr_call 的 emit 时刻 = 推理结束；以 emit 时刻减推理耗时近似语料起点
        if ev.get("type") == "asr_call":
            lo = ms - float(ev.get("elapsed_ms", 0) or 0)
        b = bounds.setdefault(tid, {"min": None, "max": None})
        b["min"] = lo if b["min"] is None else min(b["min"], lo)
        b["max"] = max(b["max"] or 0.0, ms)
    for tid in bounds:
        if bounds[tid]["min"] is None:
            bounds[tid]["min"] = bounds[tid]["max"] or 0.0
    return bounds


def summary(window: int | None = None) -> dict:
    # 显式传 None 表示全量；否则走配置默认
    window = query_window() if window is None else window

    traces = total_llm_calls = total_tool_calls = tool_rounds = 0
    success_count = 0
    in_tok = out_tok = total_tok = 0
    response_samples: list[float] = []      # 聊天段（trace_start→end）
    pipeline_samples: list[float] = []      # 全链路（ASR 起→最后一个 TTS 完成）

    per_model: dict[str, dict] = {}
    tool_counts: dict[str, int] = {}

    # ASR / TTS 独立聚合（不参与聊天统计）
    asr_calls = asr_ok = 0
    asr_ms = asr_audio = 0.0
    asr_rtfs: list[float] = []
    tts_calls = tts_ok = tts_retries = tts_trunc = 0
    tts_ms = 0.0

    _events = _read_events()
    _bounds = _pipeline_bounds(_events)  # 全链路耗时：跨 ASR/LLM/TTS 同 trace 聚合
    for ev in _events:
        if not _in_window(ev, window):
            continue
        if ev.get("kind") == "asr":
            # ASR 独立 trace：只喂 asr 聚合，不污染聊天统计
            if ev.get("type") == "asr_call":
                asr_calls += 1
                if ev.get("success"):
                    asr_ok += 1
                asr_ms += float(ev.get("inference_ms", 0) or 0)
                asr_audio += float(ev.get("audio_ms", 0) or 0)
                if float(ev.get("inference_ms", 0) or 0):
                    asr_rtfs.append(float(ev.get("rtf", 0) or 0))
            continue
        if ev.get("kind") in _BACKGROUND_KINDS:
            continue  # 后台维护 trace（压缩/长期记忆等）不参与用户请求统计
        t = ev.get("type")
        if t == "trace_end":
            traces += 1
            if ev.get("success"):
                success_count += 1
            try:
                response_samples.append(float(ev.get("elapsed_ms", 0)))
            except (TypeError, ValueError):
                pass
            # 全链路耗时段（ASR 起始 → 最后一段 TTS 完成）：对同一 trace 提取，
            # 若含 ASR/TTS 则覆盖聊天段之外的部分；纯聊天 trace 退化为 elapsed_ms。
            try:
                b = _bounds.get(ev.get("trace_id"))
                if b and b.get("min") is not None:
                    pipeline_samples.append(b["max"] - b["min"])
            except (TypeError, ValueError):
                pass
        elif t == "llm_call":
            total_llm_calls += 1
            in_tok += int(ev.get("input_tokens", 0) or 0)
            out_tok += int(ev.get("output_tokens", 0) or 0)
            total_tok += int(ev.get("total_tokens", 0) or 0)
            key = ev.get("model") or ev.get("route") or "?"
            m = per_model.setdefault(key, {
                "model": key, "calls": 0, "fail": 0,
                "avg_elapsed_ms": 0.0, "avg_tokens": 0,
            })
            m["calls"] += 1
            if not ev.get("success"):
                m["fail"] += 1
            elapsed = float(ev.get("elapsed_ms", 0) or 0)
            tokens = int(ev.get("total_tokens", 0) or 0)
            m["avg_elapsed_ms"] += elapsed
            m["avg_tokens"] += tokens
        elif t == "tool_round":
            tool_rounds += 1
        elif t == "tool_call":
            total_tool_calls += 1
            tool = ev.get("tool", "?")
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
        elif t == "tts_call":
            # TTS 合成段：kind="chat" 的子事件（非 trace_end），只喂 tts 聚合
            tts_calls += 1
            if ev.get("success"):
                tts_ok += 1
            tts_ms += float(ev.get("elapsed_ms", 0) or 0)
            if ev.get("retried"):
                tts_retries += 1
            if ev.get("truncated"):
                tts_trunc += 1

    per_model_list: list[dict] = []
    for m in per_model.values():
        if m["calls"]:
            m["avg_elapsed_ms"] = round(m["avg_elapsed_ms"] / m["calls"], 1)
            m["avg_tokens"] = int(m["avg_tokens"] / m["calls"])
        per_model_list.append(m)
    per_model_list.sort(key=lambda x: -x["calls"])

    response_sorted = sorted(response_samples)
    n = len(response_sorted)

    def _pstats(samples: list[float]):
        s = sorted(samples)
        k = len(s)
        return {
            "avg": round(sum(s) / k, 1) if k else 0.0,
            "p50": _percentile(s, 0.5),
            "p90": _percentile(s, 0.9),
        }

    return {
        "traces": traces,
        "success": success_count,
        "success_rate": round(success_count / traces, 4) if traces else 0.0,
        "response_time": _pstats(response_samples),
        "pipeline": _pstats(pipeline_samples),
        "total_llm_calls": total_llm_calls,
        "total_tokens": {"input": in_tok, "output": out_tok, "total": total_tok},
        "per_model": per_model_list,
        "tool_call_counts": tool_counts,
        "total_tool_calls": total_tool_calls,
        "tool_rounds": tool_rounds,
        "asr": {
            "calls": asr_calls,
            "success_rate": round(asr_ok / asr_calls, 4) if asr_calls else 0.0,
            "avg_ms": round(asr_ms / asr_calls, 1) if asr_calls else 0.0,
            "total_audio_ms": round(asr_audio, 1),
            "avg_rtf": round(sum(asr_rtfs) / len(asr_rtfs), 4) if asr_rtfs else 0.0,
        },
        "tts": {
            "calls": tts_calls,
            "success_rate": round(tts_ok / tts_calls, 4) if tts_calls else 0.0,
            "avg_ms": round(tts_ms / tts_calls, 1) if tts_calls else 0.0,
            "retry_count": tts_retries,
            "truncation_count": tts_trunc,
        },
    }


def requests(limit: int | None = None) -> list[dict]:
    limit = query_limit() if limit is None else limit
    events = _read_events()
    bounds = _pipeline_bounds(events)  # 全链路起止，供每行算 pipeline_ms
    by_trace: dict[str, dict] = {}
    for ev in events:
        etype = ev.get("type")
        tid = ev.get("trace_id")
        # ASR/后台维护 trace 都不进聊天请求列表（ASR 走 pipeline 单独看）
        if not tid or ev.get("kind") in ("asr", *_BACKGROUND_KINDS):
            continue
        if etype == "trace_start":
            if tid not in by_trace:
                by_trace[tid] = {"start": ev, "end": None}
        elif etype == "trace_end" and tid in by_trace:
            by_trace[tid]["end"] = ev

    rows = []
    for tid, pair in by_trace.items():
        s, e = pair["start"], pair["end"]
        b = bounds.get(tid)
        pipeline_ms = None
        if b and b["min"] is not None:
            pipeline_ms = round(b["max"] - b["min"], 1)
        rows.append({
            "trace_id": tid,
            "ts": (e or s).get("ts"),
            "session_id": (e or s).get("session_id"),
            "kind": (s.get("kind") if s else None),   # 供前端区分后台/聊天 trace
            "msg_preview": s.get("msg_preview") if s else "",
            "elapsed_ms": (e.get("elapsed_ms") if e else None),
            "pipeline_ms": pipeline_ms,   # 全链路：ASR 起始 → 最后一段 TTS 完成
            "success": bool(e.get("success")) if e else None,
            "tool_rounds": (e.get("tool_rounds")) if e else None,
            "llm_calls": (e.get("llm_calls")) if e else None,
        })
    rows.sort(key=lambda r: r.get("ts") or "", reverse=True)
    return rows[:max(0, limit)]


def request(trace_id: str) -> list[dict]:
    """返回某 trace 的全部事件，按 seq 排序（用于展开时间线）。"""
    out = []
    for ev in _read_events():
        if ev.get("trace_id") == trace_id:
            out.append(ev)
    out.sort(key=lambda e: e.get("seq", 0))
    return out


def pipeline(session_id: str, limit: int = 20) -> list[dict]:
    """按 session_id 聚合该会话的全链路 trace 组（ASR / chat）。

    面板据此把 ASR→LLM/工具→TTS 拼成同一会话下的时间线。每个 trace 组结构与
    request() 一致（events 按 seq 升序），供前端复用同一时间线渲染。跳过
    kind=="summary" 的压缩 trace。按结束时间倒序、cap limit。
    """
    events = _read_events()
    seen: dict[str, dict] = {}  # trace_id -> {trace_end, ts, kind}
    order: list[str] = []
    for ev in events:
        if ev.get("session_id") != session_id:
            continue
        if ev.get("kind") in _BACKGROUND_KINDS:
            continue
        tid = ev.get("trace_id")
        if not tid:
            continue
        if tid not in seen:
            seen[tid] = {"end": None}
            order.append(tid)
        if ev.get("type") == "trace_end":
            seen[tid]["end"] = ev

    groups: list[dict] = []
    for tid in order:
        g = seen[tid]
        kind = None
        evs = []
        for ev in events:
            if ev.get("trace_id") != tid:
                continue
            # 整条 trace 的 kind 取根事件（trace_start）为准：ASR→LLM/工具→TTS
            # 合并成一条 trace 时，首事件常是 kind="asr" 的 asr_call，不能据此当 asr 类。
            if ev.get("type") == "trace_start":
                kind = ev.get("kind")
            evs.append(ev)
        if kind is None and evs:
            kind = evs[0].get("kind")
        end = g["end"]
        first = evs[0] if evs else None
        stamp = (end or first)
        groups.append({
            "trace_id": tid,
            "kind": kind,
            "session_id": session_id,
            "ts": stamp.get("ts") if stamp else "",
            "success": bool(end.get("success")) if end else None,
            "events": evs,
        })

    def _sort_key(g: dict) -> str:
        return g.get("ts") or ""
    groups.sort(key=_sort_key, reverse=True)
    return groups[:max(0, limit)]