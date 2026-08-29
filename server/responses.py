###############################################################################
#  统一 JSON 响应助手 — 全项目唯一出口（routes / obs / avatar_routes 等共用）
#  消除重复实现；统一 ensure_ascii=False，保留中文原文便于调试。
###############################################################################

import json

from aiohttp import web


def json_ok(data=None):
    """返回成功 JSON 响应"""
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body, ensure_ascii=False),
    )


def json_error(msg: str, code: int = -1):
    """返回错误 JSON 响应"""
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}, ensure_ascii=False),
    )