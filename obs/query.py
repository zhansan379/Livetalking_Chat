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


def summary(window: int | None = None) -> dict:
    # 显式传 None 表示全量；否则走配置默认
    window = query_window() if window is None else window

    traces = total_llm_calls = total_tool_calls = tool_rounds = 0
    success_count = 0
    in_tok = out_tok = total_tok = 0
    response_samples: list[float] = []

    per_model: dict[str, dict] = {}
    tool_counts: dict[str, int] = {}

    for ev in _read_events():
        if not _in_window(ev, window):
            continue
        if ev.get("kind") == "summary":
            continue  # 压缩摘要是维护成本，不参与用户请求统计
        t = ev.get("type")
        if t == "trace_end":
            traces += 1
            if ev.get("success"):
                success_count += 1
            try:
                response_samples.append(float(ev.get("elapsed_ms", 0)))
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

    per_model_list: list[dict] = []
    for m in per_model.values():
        if m["calls"]:
            m["avg_elapsed_ms"] = round(m["avg_elapsed_ms"] / m["calls"], 1)
            m["avg_tokens"] = int(m["avg_tokens"] / m["calls"])
        per_model_list.append(m)
    per_model_list.sort(key=lambda x: -x["calls"])

    response_sorted = sorted(response_samples)
    n = len(response_sorted)
    return {
        "traces": traces,
        "success": success_count,
        "success_rate": round(success_count / traces, 4) if traces else 0.0,
        "response_time": {
            "avg": round(sum(response_samples) / n, 1) if n else 0.0,
            "p50": _percentile(response_sorted, 0.5),
            "p90": _percentile(response_sorted, 0.9),
        },
        "total_llm_calls": total_llm_calls,
        "total_tokens": {"input": in_tok, "output": out_tok, "total": total_tok},
        "per_model": per_model_list,
        "tool_call_counts": tool_counts,
        "total_tool_calls": total_tool_calls,
        "tool_rounds": tool_rounds,
    }


def requests(limit: int | None = None) -> list[dict]:
    limit = query_limit() if limit is None else limit
    events = _read_events()
    by_trace: dict[str, dict] = {}
    for ev in events:
        etype = ev.get("type")
        tid = ev.get("trace_id")
        if not tid or ev.get("kind") == "summary":
            continue
        if etype == "trace_start":
            if tid not in by_trace:
                by_trace[tid] = {"start": ev, "end": None}
        elif etype == "trace_end" and tid in by_trace:
            by_trace[tid]["end"] = ev

    rows = []
    for tid, pair in by_trace.items():
        s, e = pair["start"], pair["end"]
        rows.append({
            "trace_id": tid,
            "ts": (e or s).get("ts"),
            "session_id": (e or s).get("session_id"),
            "msg_preview": s.get("msg_preview") if s else "",
            "elapsed_ms": (e.get("elapsed_ms") if e else None),
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