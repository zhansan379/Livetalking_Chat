"""
流式调用路径：单个模型的 SSE token 流 + 多模型故障转移。

与推理路径复用同一套熔断、错误分类与健康标记逻辑；
token 用量优先取真实 usage（流末 chunk.usage），字符估算仅作兜底。
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from infra_ai.inference import (
    _CircuitOpenError,
    _get_model_for_target,
    _get_rate_limiter,
    _get_text_llm,
    _get_text_model_name,
    _get_text_rate_limiter,
    logger,
)
from infra_ai.core.stats import _extract_token_usage_from_text, _text_stats, _vision_stats
from infra_ai.core.router import ModelTarget
from infra_ai.obs_hook import emit_obs

# 日志通道复用 inference 的 logger（infra_ai.inference），保持单一通道。


def _real_stream_usage(chunk) -> dict | None:
    """
    尝试从流式 chunk 提取真实 token 用量。

    OpenAI SDK 流在最后一个 chunk 带 use (chunk.usage)，类型为 CompletionUsage。
    """
    usage = getattr(chunk, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
            "output_tokens": int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
    return {
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


async def _stream_single_model(
    target: "ModelTarget",
    messages: list[dict[str, Any]],
    use_json: bool,
    label: str,
) -> "AsyncIterator[str]":
    """单个模型的流式调用（内部实现）。"""
    from infra_ai.core.health_store import get_health_store

    health_store = get_health_store()
    is_routed = target.candidate is not None and target.model_id != "default"

    if target.candidate:
        model = _get_model_for_target(target, use_json)
        rate_limiter = _get_rate_limiter(target.model_id, target.model_name)
    else:
        llm = _get_text_llm()
        model = llm
        if use_json:
            model = llm.bind(response_format={"type": "json_object"})
        rate_limiter = _get_text_rate_limiter()

    model_name = getattr(model, 'model_name', None) or str(model)
    if is_routed and not health_store.allow_call(target.model_id):
        raise _CircuitOpenError(target.model_id)

    try:
        await rate_limiter.acquire(max_wait=60.0)
    except TimeoutError as e:
        logger.warning("[stream] RateLimiter 等待超时: %s", e)
        raise

    t0 = time.monotonic()
    full_response = ""
    usage_meta: dict | None = None
    stats = _text_stats if label in ("text", "chat") else _vision_stats
    t0 = time.monotonic()
    full_response = ""
    usage_meta: dict | None = None
    stats = _text_stats if label in ("text", "chat") else _vision_stats
    succeeded = True
    try:
        async for chunk in model.astream(messages):
            # 记录流末真实 usage（若有）
            candidate_usage = _real_stream_usage(chunk)
            if candidate_usage is not None:
                usage_meta = candidate_usage

            # OpenAI SDK 流: chunk.choices[0].delta.content
            content = ""
            try:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content or ""
            except (IndexError, AttributeError):
                content = ""
            if not content:
                # 兜底（部分兼容实现直接给 .content / dict）
                if isinstance(chunk, str):
                    content = chunk
                elif hasattr(chunk, 'content') and chunk.content:
                    content = chunk.content
                elif isinstance(chunk, dict):
                    content = chunk.get("content") or ""
            if content:
                full_response += content
                yield content

        # 正常结束 → 健康标记成功（统一身份 = 路由用 model_id）
        health_key = target.model_id if is_routed else model_name
        health_store.mark_success(health_key)
    except asyncio.CancelledError:
        # 上游中断（故障转移切换到下一个候选）——不计成败
        raise
    except Exception as e:
        # 临时性错误 → 标记失败触发熔断；致命错误不触发（问题在配置）
        from infra_ai.core.errors import classify_error
        err_type = classify_error(e)
        succeeded = False
        health_key = target.model_id if is_routed else model_name
        if err_type.should_retry():
            health_store.mark_failure(health_key)
        logger.warning("[stream] 模型 %s 流式失败 (%s): %s",
                       target.display if target.candidate else model_name,
                       err_type.value, e)
        raise
    finally:
        elapsed = time.monotonic() - t0
        # 真实 usage 优先，字符估算仅作兜底
        if usage_meta:
            usage = usage_meta
        else:
            usage = _extract_token_usage_from_text(len(full_response))
        stats.record(elapsed, usage.get('input_tokens', 0), usage.get('output_tokens', 0))
        emit_obs({
            "type": "llm_call",
            "mode": "stream",
            "model": model_name,
            "route": locals().get('health_key', model_name),
            "has_tools": False,
            "attempts": 1,
            "elapsed_ms": round(elapsed * 1000, 1),
            "input_tokens": usage.get('input_tokens', 0),
            "output_tokens": usage.get('output_tokens', 0),
            "total_tokens": usage.get('total_tokens', 0),
            "success": succeeded,
            "fail_reason": None,
            "err_type": None,
        })
        rate_limiter.record_tokens(usage.get('total_tokens', 0))
        rate_limiter.observe(usage.get('total_tokens', 0), elapsed)
        rate_limiter.release()
        logger.debug(
            "[stream] %.2fs | model=%s | total_chars=%d | rpm=%d tpm=%d",
            elapsed, model_name, len(full_response),
            rate_limiter.current_rpm, rate_limiter.current_tpm,
        )


async def async_stream_call_llm(
    messages: list[dict[str, Any]],
    use_json: bool = False,
) -> "AsyncIterator[str]":
    """
    流式调用文本 LLM，逐个 token yield。支持多模型故障转移。

    多候选时复用 stream_with_fallback 的 Probe-and-Commit 探活路由：
    对每个候选模型探活首包，成功则 commit 并 yield token，失败切下一个候选。

    :param messages: 消息列表
    :param use_json: 是否绑定 JSON 输出（流式通常为 False）
    :yields: 每个 token 字符串
    """
    # 尝试多模型路由（仅当有多个候选才启用探活-回退，单候选走下方单模型回退）
    try:
        from infra_ai.core.router import get_router
        from infra_ai.core.stream_bridge import stream_with_fallback
        from infra_ai.inference import _get_first_choice

        router = get_router()
        first_choice_id = _get_first_choice("chat")
        targets = router.selector.select("chat", first_choice_id=first_choice_id)
        if len(targets) > 1:

            async def _produce(target, bridge):
                try:
                    async for token in _stream_single_model(target, messages, use_json, "chat"):
                        bridge.on_token(token)
                    bridge.on_complete()
                except Exception as e:
                    bridge.on_error(e)

            async for token in stream_with_fallback(
                router.selector, "chat", _produce,
                first_choice_id=first_choice_id, timeout=60,
            ):
                yield token
            return
    except ImportError:
        # 路由模块不可用 → 单模型回退（RuntimeError 向上抛出）
        pass

    # 单模型回退
    async for token in _stream_single_model(
        ModelTarget(model_id="default", candidate=None, model_name=_get_text_model_name()),
        messages, use_json, "text",
    ):
        yield token