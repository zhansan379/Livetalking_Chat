###############################################################################
#  /api/obs/* — 观测数据 REST 端点。注册进 server/routes.setup_routes。
#  统一响应助手来自 server/responses.py（别名保留，调用点零改动）。
###############################################################################

from server.responses import json_ok as _json_ok, json_error as _json_error

from . import query


async def obs_summary(request):
    try:
        raw = request.query.get("window")
        window = int(raw) if raw else None
        return _json_ok(query.summary(window=window))
    except Exception as e:  # noqa: BLE001
        return _json_error(str(e))


async def obs_requests(request):
    try:
        raw = request.query.get("limit")
        limit = int(raw) if raw else None
        return _json_ok({"requests": query.requests(limit=limit)})
    except Exception as e:  # noqa: BLE001
        return _json_error(str(e))


async def obs_request(request):
    try:
        trace_id = request.match_info.get("trace_id", "")
        if not trace_id:
            return _json_error("trace_id required")
        return _json_ok({"events": query.request(trace_id), "trace_id": trace_id})
    except Exception as e:  # noqa: BLE001
        return _json_error(str(e))


async def obs_pipeline(request):
    """按 session_id 返回该会话的全链路 trace 组（ASR → LLM/工具 → TTS）。"""
    try:
        sid = request.query.get("session_id", "")
        if not sid:
            return _json_error("session_id required")
        raw = request.query.get("limit")
        limit = int(raw) if raw else 20
        return _json_ok({"session_id": sid, "traces": query.pipeline(sid, limit=limit)})
    except Exception as e:  # noqa: BLE001
        return _json_error(str(e))


def register(app):
    app.router.add_get("/api/obs/summary", obs_summary)
    app.router.add_get("/api/obs/requests", obs_requests)
    app.router.add_get("/api/obs/request/{trace_id}", obs_request)
    app.router.add_get("/api/obs/pipeline", obs_pipeline)