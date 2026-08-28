###############################################################################
#  工具循环：TOOL_REGISTRY 注册表 + web_search 执行 + while 工具循环
###############################################################################
#  设计要点（与 agent/agent.py 的接口约定）：
#  - 一个工具 = 一条注册表 entry，同时管住「schema（给模型引用）」和「handler（执行）」，
#    避免两者不同步。新增工具只需加一个 entry + 在 agent_config.yaml 补 tools.<name>.*。
#  - run_tool_loop 的唯一职责：把模型的 tool_calls 解析成一句最终文本答案。
#    它不做「思考循环」——OpenAI/Qwen 函数调用里，模型想用工具时返回的是 tool_calls 而非答案，
#    必须把执行结果以 {"role":"tool"} 回填再问一次，直到模型给出纯文本答案（对应 s01 的
#    while stop_reason == "tool_use"）。
#  - 契约：拿到真答案返回 str，触顶/异常返回 None（交给调用方决定说辞），绝不伪造模型回复。
###############################################################################

import asyncio
import json

from utils.logger import logger


# ─── 网络搜索执行（DuckDuckGo）─────────────────────────────────────────────
async def web_search(args: dict, cfg) -> str:
    """执行一次网络搜索，返回若干条「标题+链接+摘要」拼成的文本。"""
    query = (args or {}).get("query", "")
    if not query:
        return "（搜索关键词为空）"
    max_sources = getattr(cfg, "tool_web_search_max_sources", 5)

    def _run():
        # ddgs 是新包名；兼容仍是 duckduckgo_search 的老安装
        try:
            from ddgs import DDGS
        except ImportError:  # noqa: F841
            from duckduckgo_search import DDGS
        return list(DDGS().text(query, max_results=max_sources))

    try:
        rows = await asyncio.to_thread(_run)
    except Exception as e:  # noqa: BLE001 - 搜索失败要让模型自行处理，不能中断对话
        logger.warning("web_search failed: %s", e)
        return f"（网络搜索失败：{e}）"

    lines = []
    for row in rows or []:
        title = (row.get("title") or "").strip()
        href = (row.get("href") or "").strip()
        body = (row.get("body") or "").strip()
        if title:
            lines.append(f"标题：{title}")
        if href:
            lines.append(f"链接：{href}")
        if body:
            lines.append(f"摘要：{body}")
        lines.append("---")

    text = "\n".join(lines).strip()
    if not text:
        return "（无搜索结果）"
    return text


# ─── 工具注册表：一个工具 = 一条 (schema + handler) ─────────────────────────
TOOL_REGISTRY: dict[str, dict] = {
    "web_search": {
        "description": (
            "联网搜索互联网获取实时/最新信息，返回若干条标题、链接和摘要。"
            "当用户问到时效性强、或需要联网确认/查询的事实性问题时调用。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
        "handler": web_search,
    },
    # 以后新增工具：在这里加一个 entry，并在 agent_config.yaml 里加 tools.<name>.enabled
}


def list_enabled_tools(cfg) -> list[str]:
    """返回配置里已启用的工具名（约定：配置字段 tool_<name>_enabled）。"""
    return [
        name for name in TOOL_REGISTRY
        if getattr(cfg, f"tool_{name}_enabled", False)
    ]


def build_tools(enabled: list[str]) -> list[dict]:
    """由注册表把启用的工具名转成 OpenAI function calling 的 tools 列表。"""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_REGISTRY[name]["description"],
                "parameters": TOOL_REGISTRY[name]["parameters"],
            },
        }
        for name in enabled
    ]


# ─── 工具循环 ──────────────────────────────────────────────────────────────
def _assistant_turn(resp, tool_calls) -> dict:
    """把原生 assistant message（含 tool_calls）转成可继续喂给下一次调用的 dict。"""
    return {
        "role": "assistant",
        "content": getattr(resp, "content", None) or None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ],
    }


async def run_tool_loop(agent_messages: list, tools: list[dict], cfg) -> str | None:
    """
    while 工具循环：把模型可能打出的 tool_calls 解析成一句最终文本答案。

    :param agent_messages: 已含 system + 历史 + 当前 user 消息的上下文
    :param tools: OpenAI function calling 的 tools 列表（由 build_tools 生成）
    :param cfg: AgentConfig（提供 tool_max_rounds 与各工具参数）
    :return: 模型最终文本答案；循环触顶返回 None（不伪造回复，交给调用方说辞）
    """
    from infra_ai import async_call_llm_with_tools

    msgs = list(agent_messages)
    for _ in range(cfg.tool_max_rounds):
        try:
            resp = await async_call_llm_with_tools(msgs, tools)
        except Exception as e:  # noqa: BLE001 - LLM 调用失败：让上层走降级话术
            logger.exception("run_tool_loop LLM call failed: %s", e)
            return None

        tool_calls = getattr(resp, "tool_calls", None)
        if not tool_calls:
            # 模型决定不再调工具 → 这就是最终答案，循环唯一出口
            return (getattr(resp, "content", None) or "").strip()

        msgs.append(_assistant_turn(resp, tool_calls))
        for tc in tool_calls:
            handler = TOOL_REGISTRY.get(tc.function.name, {}).get("handler")
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if handler is None:
                result = f"（未知工具：{tc.function.name}）"
            else:
                try:
                    result = await handler(args, cfg) or "（工具无输出）"
                except Exception as e:  # noqa: BLE001 - 单个工具失败不中断循环
                    logger.exception("tool %s handler failed: %s", tc.function.name, e)
                    result = f"（工具 <{tc.function.name}> 执行失败：{e}）"
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    logger.error("tool loop hit %d rounds without a text answer", cfg.tool_max_rounds)
    return None