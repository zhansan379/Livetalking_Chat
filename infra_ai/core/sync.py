"""
同步封装（向后兼容）。

核心思想：检测当前是否在 asyncio 事件循环中，适配任何调用上下文
（同步脚本 / FastAPI / Jupyter）。具体逻辑见 infra_ai._async_utils.run_async_in_sync。
"""

from typing import Any

from infra_ai.core._async_utils import run_async_in_sync as _run_async_in_sync
from infra_ai.inference import async_call_llm, async_call_vlm


def call_llm(
    messages: list[dict[str, Any]],
    use_json: bool = True,
    *,
    model_name: str | None = None,
) -> str:
    """
    调用文本大模型（同步封装）。

    在任何上下文中安全使用（同步脚本 / FastAPI / Jupyter）。

    :param messages: 消息列表
    :param use_json: 是否以 JSON 格式输出
    :param model_name: 指定模型名称（如 "Qwen/Qwen3.6-27B"），不传则使用默认路由
    """
    return _run_async_in_sync(async_call_llm(messages, use_json, model_name=model_name))


def call_vlm(
    messages: list[dict[str, Any]],
    use_json: bool = True,
    images: list[str] | None = None,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """
    调用视觉大模型 VLM（同步封装）。

    在任何上下文中安全使用（同步脚本 / FastAPI / Jupyter）。

    :param messages: 消息列表
    :param use_json: 是否以 JSON 格式输出
    :param images: 可选，附加图片 URL/base64 列表
    :param extra: 附加信息，失败时写入错误日志
    """
    return _run_async_in_sync(
        async_call_vlm(messages, use_json, images, extra=extra)
    )