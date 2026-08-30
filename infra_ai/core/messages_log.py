###############################################################################
#  messages_log：LLM 调用的消息序列化 —— 供推理 / 流式两条观测路径复用。
#
#  obs 事件需要把「本次调用发了什么、模型回了什么」完整记录进 JSONL。
#  放在 core/ 中性子模块（而非 inference.py）是为了让 inference / streaming
#  都引用而不引入循环依赖。
#
#  口径（与旧 _messages_for_file_log 一致）：保留完整文本与完整 URL，
#  仅剥离 base64 图片数据体（太大无意义，记成占位 + 长度）。
###############################################################################

from typing import Any


def serialize_for_obs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """序列化消息列表用于观测事件：保 role、保完整文本/URL、剥 base64 图片体。"""
    result = []
    for m in messages or []:
        entry: dict[str, Any] = {"role": m.get("role", "?")}
        content = m.get("content", "")
        if isinstance(content, str):
            entry["content"] = content  # 完整文本
        elif isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    raw_url = str(item.get("image_url", {}).get("url", ""))
                    # 剥离 base64 数据体，仅保留前缀 + 长度
                    if "base64," in raw_url:
                        header, b64data = raw_url.split("base64,", 1)
                        parts.append({
                            "type": "image_url",
                            "url": f"{header}base64,<{len(b64data)} chars>",
                        })
                    else:
                        parts.append({"type": "image_url", "url": raw_url})
            entry["content"] = parts
        result.append(entry)
    return result


def output_snippet(message) -> Any:
    """从 OpenAI 返回 message 提取观测用的返回文本。

    content 为空（典型是工具调用场景：模型只回 tool_calls）时，回退为
    tool 函数名列表，保证观测里能看出模型"要调什么"。
    """
    if message is None:
        return None
    try:
        content = getattr(message, "content", None)
    except Exception:  # noqa: BLE001 - 元数据提取失败不致命
        content = None
    if content:
        return content if isinstance(content, str) else str(content)
    try:
        tcs = getattr(message, "tool_calls", None)
    except Exception:  # noqa: BLE001
        tcs = None
    if tcs:
        return {"tool_calls": [tc.function.name for tc in tcs if tc.function]}
    return content if content is not None else ""