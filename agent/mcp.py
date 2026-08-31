###############################################################################
#  通用 MCP（Model Context Protocol）工具接入模块
#
#  它不在主循环里加任何分支，而是把配置的每台 MCP 服务器（stdio / sse / http）
#  的三把工具以「mcp_<server>_<tool>」命名注册进 tool_loop.TOOL_REGISTRY，
#  config_flag 统一盖为 tool_mcp_enabled。之后 session_tools → build_tools →
#  run_tool_loop 的既有链路全自动接管：模型像调内置工具一样调 MCP 工具，
#  handler 里往对应服务器的 ClientSession 发 tools/call。观测/耗时即走 run_tool_loop。
#  设计上 MCP 是「通用工具」不是能力域（agent/ 下，不进 capabilities/）。
#
#  生命周期：start_mcp_servers 在服务启动时拉每台服务器建连+注册；
#  close_mcp_servers 在服务关闭时 cancel 常驻连接 task、清掉注册表里 mcp_ 前缀 entry。
#  mcp 包未装 / 服务器配置失败 → 告警跳过，绝不影响服务启动。
###############################################################################

import asyncio
import json

from functools import partial
from utils.logger import logger

_MCP_PREFIX = "mcp_"          # 注册表命名前缀 + 关闭清理的依据
_MCP_FLAG = "tool_mcp_enabled"  # 统一 config_flag：与 agent_config.yaml tools.mcp.enabled 联动

# 每台服务器已连接的 ClientSession：server 名 → session（由 _serve_server 常量持有）
_SESSIONS: dict[str, object] = {}
# 每台服务器的常驻连接 task（持有 stdio/sse/http 连接的 context，防止连接被 GC 关断）
_GUARDS: dict[str, asyncio.Task] = {}

# ── mcp 包可用性：缺失时优雅降级，不拉崩服务 ──────────────────────────────
try:  # noqa: C901
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamable_http_client

    _MCP_OK = True
except Exception:  # noqa: BLE001 - 缺依赖只影响 MCP 功能本身
    ClientSession = StdioServerParameters = None
    stdio_client = sse_client = streamable_http_client = None
    _MCP_OK = False


def _get_text(block) -> str | None:
    """从 MCP 结果块里取文本；支持 pandas/pydantic 两种 data 风格。"""
    try:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return None


def _format_result(res) -> str:
    """把 CallToolResult 归一化成一段文本，供 run_tool_loop 喂回模型。"""
    parts: list[str] = []
    for blk in getattr(res, "content", None) or []:
        txt = _get_text(blk)
        if txt:
            parts.append(txt)
        else:
            tip = getattr(blk, "type", "资源")
            parts.append(f"(MCP {tip} 数据块，无法内联展示)")
    text = "\n".join(p for p in parts if p).strip()

    # 纯结构化内容（无 text 块时兜底 dump 出来）
    if not text:
        sc = getattr(res, "structured_content", None)
        if sc is None:
            sc = getattr(res, "structuredContent", None)
        if sc is not None:
            try:
                text = json.dumps(sc, ensure_ascii=False)
            except Exception:  # noqa: BLE001
                text = str(sc)

    is_err = getattr(res, "is_error", None)
    if is_err is None:
        is_err = getattr(res, "isError", False)
    if is_err:
        text = f"(MCP 工具返回错误) {text}".strip()
    return text or "(MCP 工具无输出)"


async def _mcp_call(args: dict, cfg, ctx=None, *, server: str, tool: str) -> str:
    """MCP 工具的统一 handler：server/tool 由注册时的 partial 绑定。

    签名与内置工具一致：async def handler(args, cfg, ctx=None) -> str，
    这样 run_tool_loop 无需感知 MCP 的存在。
    """
    if not _MCP_OK:
        return f"(MCP 不可用：未安装 mcp 包，请 pip install mcp) <{server}.{tool}>"
    session = _SESSIONS.get(server)
    if session is None:
        return f"(MCP 服务器 <{server}> 未连接，无法调用 <{tool}>)"
    try:
        res = await session.call_tool(tool, arguments=args or {})
    except Exception as e:  # noqa: BLE001 - 单个工具失败不中断工具循环
        logger.warning("mcp tool %s.%s failed: %s", server, tool, e)
        return f"(MCP 工具 <{tool}> 调用失败：{e})"
    return _format_result(res)


def _tool_desc(server: str, t) -> str:
    desc = ""
    try:
        desc = str(getattr(t, "description", "") or "")
    except Exception:  # noqa: BLE001
        pass
    return desc.strip() or f"MCP({server}) 提供的外部工具"


def _input_schema(t) -> dict:
    schema = getattr(t, "input_schema", None)
    if schema is None:
        schema = getattr(t, "inputSchema", None)
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    return schema


def _build_http_client(headers: dict | None):
    """http 传输若要带 headers，需把它烤进客户端工厂；无法构造则退 None。"""
    if not headers:
        return None
    for mod in ("httpx2", "httpx"):
        try:
            _httpx = __import__(mod)
            return _httpx.AsyncClient(headers=headers, timeout=30.0)
        except Exception:  # noqa: BLE001
            continue
    return None


async def _serve_server(name: str, spec: dict, cfg, started: "asyncio.Event") -> None:
    """常驻连接一台 MCP 服务器并在注册表登记其工具，直到被 cancel。

    stdio/sse/http 三个传输在 mcp SDK 里都是「异步上下文管理器，解包成
    (read, write) 流」的统一协议，因此只差选哪个 transport 的工厂。
    """
    # 惰性 import 注册表，避免配置构造期拉起 agent.tool_loop 依赖链
    from agent import tool_loop

    transport = str(spec.get("transport") or "stdio").lower()
    headers = dict(spec.get("headers") or {}) or None
    url = str(spec.get("url") or "").strip()

    try:
        if transport == "stdio":
            ctx = stdio_client(
                StdioServerParameters(
                    command=str(spec.get("command") or ""),
                    args=[str(a) for a in (spec.get("args") or [])],
                    env=dict(spec.get("env") or {}) or None,
                )
            )
        elif transport == "sse":
            if not url:
                raise ValueError("sse 传输需要 url")
            ctx = sse_client(url, headers=headers)
        else:  # http / streamable
            if not url:
                raise ValueError("http 传输需要 url")
            ctx = streamable_http_client(url, http_client=_build_http_client(headers))

        async with ctx as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                _SESSIONS[name] = session
                res = await session.list_tools()
                tools = getattr(res, "tools", None)
                if tools is None:
                    tools = res  # 兼容直接把 list 当结果的版本
                n = 0
                for t in tools or []:
                    tname = getattr(t, "name", None)
                    if not tname:
                        continue
                    reg_name = f"{_MCP_PREFIX}{name}_{tname}"
                    tool_loop.TOOL_REGISTRY[reg_name] = {
                        "description": _tool_desc(name, t),
                        "parameters": _input_schema(t),
                        "handler": partial(_mcp_call, server=name, tool=tname),
                        "config_flag": _MCP_FLAG,
                    }
                    n += 1
                started.set()
                logger.info(
                    "MCP server '%s' connected: %d tools registered (transport=%s)",
                    name, n, transport,
                )
                # 保活：连接由本 task 持有（释放后 stdio 子进程会被关断）
                try:
                    await asyncio.Future()
                finally:
                    logger.info("MCP server '%s' closing", name)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - 单台服务器失败不阻塞其它
        logger.warning("MCP server '%s' start failed: %s", name, e)
        started.set()


async def start_mcp_servers(cfg) -> list[str]:
    """连接配置里每台启用的 MCP 服务器并注册其工具。返回已连上的 server 名。"""
    if not getattr(cfg, "tool_mcp_enabled", False):
        return []
    if not _MCP_OK:
        logger.warning("mcp 包未安装，跳过所有 MCP 服务器（pip install mcp）")
        return []

    started_names: list[str] = []
    for name, spec in getattr(cfg, "mcp_servers", []) or []:
        if not spec or name in _GUARDS:
            continue
        if not spec.get("enabled", True):
            continue
        started = asyncio.Event()
        _GUARDS[name] = asyncio.create_task(_serve_server(name, spec, cfg, started))
        try:
            await asyncio.wait_for(started.wait(), timeout=cfg.mcp_connect_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "MCP server '%s' connect timeout after %ss", name, cfg.mcp_connect_timeout
            )
            continue
        if not _GUARDS[name].done():
            started_names.append(name)
    return started_names


async def close_mcp_servers() -> None:
    """关闭所有 MCP 连接、清掉注册表里 mcp_ 前缀的工具，幂等可重复调用。"""
    from agent import tool_loop

    for reg in list(tool_loop.TOOL_REGISTRY):
        if reg.startswith(_MCP_PREFIX):
            tool_loop.TOOL_REGISTRY.pop(reg, None)
    _SESSIONS.clear()
    guards = list(_GUARDS.values())
    _GUARDS.clear()
    for t in guards:
        if t and not t.done():
            t.cancel()
    if guards:
        await asyncio.gather(*guards, return_exceptions=True)