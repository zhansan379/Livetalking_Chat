"""
LLM 推理核心：客户端构建、熔断/限速/重试调用、多模型路由、批量接口。

依赖 stats/rate_limiter/errors/health_store/router 等子模块。
"""

import asyncio
import base64
import copy
import json
import logging
import mimetypes
import os
import time
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

from infra_ai.core.config_loader import get_config as _get_config
_cfg = _get_config()
LLM_REQUEST_TIMEOUT = _cfg.LLM_REQUEST_TIMEOUT
LLM_TIMEOUT_SCALE = _cfg.LLM_TIMEOUT_SCALE
LLM_MAX_RETRIES = _cfg.LLM_MAX_RETRIES
LLM_RETRY_BACKOFF = _cfg.LLM_RETRY_BACKOFF
DEFAULT_RATE_LIMIT = _cfg.DEFAULT_RATE_LIMIT
LLM_RATE_LIMITS = _cfg.LLM_RATE_LIMITS
AUTO_TUNE_CONCURRENCY = _cfg.AUTO_TUNE_CONCURRENCY
LLM_ERROR_LOG = _cfg.LLM_ERROR_LOG
from infra_ai.core.rate_limiter import RateLimiter
from infra_ai.core.stats import (
    _CallStats,
    _extract_token_usage,
    _text_stats,
    _vision_stats,
)
from infra_ai.obs_hook import emit_obs
logger = logging.getLogger(__name__)

# 注：日志配置交由宿主导入方完成；库代码不设置级别/handler（避免污染调用进程）。


class _CircuitOpenError(Exception):
    """模型熔断中，应触发调用方进行故障转移。"""
    def __init__(self, model_name: str):
        super().__init__(f"模型 {model_name} 熔断中，请使用候选模型")
        self.model_name = model_name


# ============================================================
# OpenAI 兼容客户端封装（替代 langchain_openai.ChatOpenAI）
# ============================================================

# 所有已创建 AsyncOpenAI 的注册表（WeakSet，避免强引用泄漏每个临时路由客户端）。
# openai v3 的 AsyncHttpxClientWrapper.__del__ 只在“有运行中事件循环”时才调度关闭底层
# httpx2 连接池；若宿主在 asyncio.run 结束后才让客户端被 GC，连接池的 httpcore2 异步生成器
# 会在无存活 loop 时收尾，抛出 “generator didn't stop after athrow()” 噪音。
# 托管方应在主流程结束、事件循环关闭前调用 aclose_all_clients() 显式关闭它们。
_ASYNC_CLIENTS: "weakref.WeakSet[AsyncOpenAI]" = weakref.WeakSet()


class _OpenAIClient:
    """
    OpenAI 兼容客户端的轻量封装，统一推理 / 流式 / 工具三条路径。

    使用原生 openai SDK（OpenAI + AsyncOpenAI），仅暴露推理路径所需的最小面：
    `.model_name` / `.bind()` / `acomplete()` / `astream()`。

    厂商专有参数（如 reasoning_effort / extra_body / thinking）可经 bind()
    固定，或每条调用通过 acomplete/astream 的 extra_kwargs 运行时传入。
    """

    def __init__(self, model_name: str, api_key: str | None = None,
                 base_url: str | None = None, temperature: float | None = 0.7,
                 timeout: float = LLM_REQUEST_TIMEOUT, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self._kwargs: dict[str, Any] = dict(kwargs)  # 固定 create() 参数
        # openai SDK 不允许 api_key 为空；缺 key 时用占位符保证懒构造不崩溃，
        # 真实缺口在请求期以 401 暴露（与旧 ChatOpenAI 的延迟错误契约一致）。
        api_key = api_key or "sk-placeholder-not-configured"
        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            timeout=timeout, max_retries=0,
        )
        self._aclient = AsyncOpenAI(
            api_key=api_key, base_url=base_url,
            timeout=timeout, max_retries=0,
        )
        _ASYNC_CLIENTS.add(self._aclient)

    async def aclose(self):
        """关闭本客户端持有的 AsyncOpenAI（连带 httpx2 连接池），幂等。"""
        aclient, self._aclient = self._aclient, None
        if aclient is not None:
            try:
                await aclient.close()
            except Exception:
                pass
            _ASYNC_CLIENTS.discard(aclient)

    def close(self):
        """同步关闭本客户端持有的 OpenAI（阻塞 httpx），幂等。"""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def bind(self, **kwargs):
        """返回带额外 create() 参数的新实例（不改动共享单例）。"""
        clone = copy.copy(self)
        clone._kwargs = {**self._kwargs, **kwargs}
        return clone

    def bind_tools(self, tools):
        return self.bind(tools=tools)

    def _payload(self, messages, *, tools=None, extra_kwargs=None) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            **self._kwargs,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if tools:
            payload["tools"] = tools
        if extra_kwargs:
            payload.update(extra_kwargs)
        return payload

    async def acomplete(self, messages, *, tools=None, extra_kwargs=None) -> Any:
        """非流式调用，返回 openai ChatCompletion 响应。"""
        return await self._aclient.chat.completions.create(
            **self._payload(messages, tools=tools, extra_kwargs=extra_kwargs))

    async def astream(self, messages, *, tools=None, extra_kwargs=None):
        """流式调用，逐块 yield openai ChatCompletionChunk。"""
        payload = self._payload(messages, tools=tools, extra_kwargs=extra_kwargs)
        payload["stream"] = True
        stream = await self._aclient.chat.completions.create(**payload)
        try:
            async for chunk in stream:
                yield chunk
        finally:
            # openai AsyncStream 在读到 [DONE] 后即 break，不会自动关闭 httpx2
            # 响应体；若放任其被 GC/客户端关闭时收尾，httpcore2 的
            # PoolByteStream.__aiter__ 会在无存活循环处被 athrow 中断，
            # 抛 “generator didn't stop after athrow()” 关停噪音。
            # 这里显式走 stream.close() → response.aclose() 的干净关闭路径，
            # 直接把底层网络流归还/关闭，避免中途 abort。
            await stream.close()


def _build_openai(model_name: str, api_key: str | None = None,
                  base_url: str | None = None, temperature: float = 0.7,
                  use_json: bool = False) -> _OpenAIClient:
    """按模型名构造 _OpenAIClient；use_json 时绑定 JSON 输出格式。"""
    kw: dict[str, Any] = {}
    if use_json:
        kw["response_format"] = {"type": "json_object"}
    return _OpenAIClient(
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kw,
    )


# ============================================================
# 资源清理：显式关闭客户端连接池
# ============================================================

async def aclose_all_clients() -> None:
    """
    关闭所有已创建的 AsyncOpenAI 客户端及其 httpx2 连接池（幂等）。

    必须在宿主应用的 `await asyncio.run(main())` 主流程结束、事件循环关闭**之前**
    调用，否则底层 httpcore2 连接池会在 GC 阶段（此时已无存活事件循环）收尾，
    打印 “RuntimeError: generator didn't stop after athrow()” 关停噪音。

    典型用法：

        asyncio.run(main())
        # ...（此处 loop 已关闭，不能再 await）
        asyncio.run(aclose_all_clients())  # 干净关闭，无噪音

    或把 aclose_all_clients() 放到 main() 返回值所在协程内部调用。

    反复调用安全。
    """
    for client in list(_ASYNC_CLIENTS):
        if client is None:
            continue
        try:
            await client.close()  # openai AsyncOpenAI.close() → httpx2 AsyncClient.aclose()
        except Exception:
            pass
    _ASYNC_CLIENTS.clear()


# ============================================================
# LLM 客户端初始化
# ============================================================

def _create_llm_from_routing(capability: str, temperature: float):
    """从 LLM_ROUTING 配置中取最高优先级启用候选，创建 _OpenAIClient（单模型回退用）。"""
    try:
        cap = _cfg.LLM_ROUTING.get(capability, {})
        candidates = cap.get("candidates", [])
        if candidates:
            c = candidates[0]
            return _OpenAIClient(
                model_name=c["model"],
                temperature=temperature,
                api_key=c.get("api_key") or None,
                base_url=c.get("base_url") or None,
            )
    except Exception:
        pass
    # fallback: 从 env 读取
    model = os.getenv(f"SF_{'CHAT_MODEL' if capability == 'chat' else 'VISION_MODEL'}", "")
    return _OpenAIClient(
        model_name=model or "Qwen/Qwen2.5-72B-Instruct-128K",
        temperature=temperature,
        api_key=os.getenv("SF_API_KEY"),
        base_url=os.getenv("SF_BASE_URL", "https://api.siliconflow.cn/v1"),
    )


# ============================================================
# 单模型回退 LLM 客户端 + 速率限制器（懒加载单例）
# 避免 import 阶段建链 / 读环境变量，由使用方首次调用时按需初始化。
# 通过 PEP 562 的模块 __getattr__ 保持这些符号可被直接引用，且惰性求值。
# ============================================================

_llm_singleton = None
_vision_llm_singleton = None
_text_rl_singleton: RateLimiter | None = None
_vision_rl_singleton: RateLimiter | None = None


def _get_text_llm():
    """文本 LLM 单例（单模型回退用，懒初始化）。"""
    global _llm_singleton
    if _llm_singleton is None:
        _llm_singleton = _create_llm_from_routing("chat", temperature=0.7)
    return _llm_singleton


def _get_vision_llm():
    """视觉 VLM 单例（单模型回退用，懒初始化）。"""
    global _vision_llm_singleton
    if _vision_llm_singleton is None:
        _vision_llm_singleton = _create_llm_from_routing("vision", temperature=0.3)
    return _vision_llm_singleton


def _get_text_model_name() -> str:
    m = _get_text_llm()
    return getattr(m, 'model_name', None) or str(m)


def _get_vision_model_name() -> str:
    m = _get_vision_llm()
    return getattr(m, 'model_name', None) or str(m)


def _build_rate_limiter(model_name: str) -> RateLimiter:
    """从 config 读取模型对应的速率限制配置（RPM + TPM + 并发 + 自适应）。"""
    limits = LLM_RATE_LIMITS.get(model_name, DEFAULT_RATE_LIMIT)
    return RateLimiter(
        rpm=limits["rpm"],
        tpm=limits.get("tpm", 0),
        max_concurrent=limits["max_concurrent"],
        auto_tune=AUTO_TUNE_CONCURRENCY,
    )


def _get_text_rate_limiter() -> RateLimiter:
    global _text_rl_singleton
    if _text_rl_singleton is None:
        _text_rl_singleton = _build_rate_limiter(_get_text_model_name())
    return _text_rl_singleton


def _get_vision_rate_limiter() -> RateLimiter:
    global _vision_rl_singleton
    if _vision_rl_singleton is None:
        _vision_rl_singleton = _build_rate_limiter(_get_vision_model_name())
    return _vision_rl_singleton


# 路由模型的速率限制器缓存：按模型身份（model_id）共享，避免每次调用新建导致限速失效。
_rate_limiters: dict[str, RateLimiter] = {}


def _get_rate_limiter(model_key: str, model_name: str | None = None) -> RateLimiter:
    """
    获取指定模型的 rate limiter（按身份缓存，跨调用共享）。

    :param model_key:  熔断/缓存统一身份（路由路径用 model_id）
    :param model_name: config 查限速参数时用的模型名（缺省回退到 model_key）
    """
    limiter = _rate_limiters.get(model_key)
    if limiter is None:
        limiter = _build_rate_limiter(model_name or model_key)
        _rate_limiters[model_key] = limiter
    return limiter


def __getattr__(name: str):
    """PEP 562 模块惰性符号：兼容旧 `llm`/`vision_llm`/`_text_rate_limiter` 等直接引用。"""
    lazy = {
        "llm": _get_text_llm,
        "vision_llm": _get_vision_llm,
        "_text_model_name": _get_text_model_name,
        "_VISION_MODEL": _get_vision_model_name,
        "_text_rate_limiter": _get_text_rate_limiter,
        "_vision_rate_limiter": _get_vision_rate_limiter,
    }
    factory = lazy.get(name)
    if factory is not None:
        return factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---- 路由辅助函数 ----

def _get_model_for_target(target, use_json: bool = False):
    """根据路由目标创建 _OpenAIClient 实例（复用已有配置）。"""
    candidate = target.candidate
    model = _OpenAIClient(
        model_name=candidate.model,
        temperature=0.7 if "vision" not in candidate.id.lower() else 0.3,
        base_url=candidate.base_url or os.getenv("SF_BASE_URL", ""),
        api_key=candidate.api_key or os.getenv("SF_API_KEY", ""),
    )
    if use_json:
        model = model.bind(response_format={"type": "json_object"})
    return model


def _get_first_choice(capability: str) -> str:
    """获取配置中的首选模型 ID（未配置时返回空串）。"""
    return _cfg.LLM_ROUTING.get(capability, {}).get("default_model", "")


# ============================================================
# 带超时 & 重试的调用核心
# ============================================================

def _messages_for_file_log(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    序列化消息列表用于文件日志。
    保留完整文本和完整 URL，仅剥离 base64 图片数据（太大无意义）。
    """
    result = []
    for m in messages:
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


def _messages_preview_for_console(messages: list[dict[str, Any]]) -> str:
    """提取消息摘要用于控制台输出（截断）。"""
    for m in messages:
        if m.get("role") == "system":
            c = m.get("content", "")
            if isinstance(c, str):
                return c[:120]
            return str(c)[:120]
    return "(无 system prompt)"


def _write_error_log(
    label: str,
    model_name: str,
    messages: list[dict[str, Any]],
    fail_reason: str,
    max_retries: int,
    last_error: Exception | None,
    stats: _CallStats,
    rate_limiter: RateLimiter,
    timeouts_used: list[float],
    extra: dict[str, Any] | None = None,
) -> None:
    """将耗尽重试的失败调用详情写入日志文件和控制台。"""
    error_entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "label": label,
        "model": model_name,
        "fail_reason": fail_reason,
        "total_attempts": max_retries,
        "timeouts_per_attempt": timeouts_used,
        "error": str(last_error) if last_error else "unknown",
        "extra": extra or {},
        "stats": {
            "calls": stats.total_calls,
            "failures": stats.total_failures,
            "total_elapsed_s": round(stats.total_elapsed, 2),
            "total_input_tokens": stats.total_input_tokens,
            "total_output_tokens": stats.total_output_tokens,
        },
        "rate_limiter": {
            "rpm": f"{rate_limiter.current_rpm}/{rate_limiter._rpm}",
            "tpm": f"{rate_limiter.current_tpm}/{rate_limiter._tpm}",
            "max_concurrent": rate_limiter._max_concurrent,
            "active": rate_limiter._active,
        },
    }

    # ---- 控制台详细输出 ----
    # 构建 extra 信息行
    extra_lines = ""
    if extra:
        for k, v in extra.items():
            extra_lines += f"\n  {k}:  {v}"

    logger.error(
        "=" * 60 + "\n"
        "  LLM 调用最终失败\n"
        "  label:       %s\n"
        "  model:       %s\n"
        "  fail_reason: %s\n"
        "  attempts:    %d  (timeouts: %s)\n"
        "  tpm/rpm:     %d/%d  |  %d/%d\n"
        "  concurrent:  active=%d  max=%d\n"
        "  stats:       %s\n"
        "  error:       %s"
        "%s\n"
        "  system_preview: %s\n"
        + "=" * 60,
        label, model_name, fail_reason, max_retries,
        ", ".join(f"{t:.0f}s" for t in timeouts_used),
        rate_limiter.current_tpm, rate_limiter._tpm,
        rate_limiter.current_rpm, rate_limiter._rpm,
        rate_limiter._active, rate_limiter._max_concurrent,
        stats.summary(),
        last_error,
        extra_lines,
        _messages_preview_for_console(messages),
    )

    # ---- 文件日志（完整内容） ----
    error_entry["messages"] = _messages_for_file_log(messages)
    try:
        log_path = Path(LLM_ERROR_LOG)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(error_entry, ensure_ascii=False) + "\n")
        logger.info("失败详情已写入 %s", log_path)
    except Exception as e:
        logger.warning("无法写入错误日志文件: %s", e)


async def _invoke_with_retry(
    model,
    messages: list[dict[str, Any]],
    rate_limiter: RateLimiter,
    stats: _CallStats,
    label: str,
    *,
    timeout: float = LLM_REQUEST_TIMEOUT,
    timeout_scale: float = LLM_TIMEOUT_SCALE,
    max_retries: int = LLM_MAX_RETRIES,
    backoff: float = LLM_RETRY_BACKOFF,
    extra: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    model_key: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
):
    """
    带超时和重试的 LLM 调用。

    超时线性递增: timeout_n = timeout * (1 + scale * n)
    例: base=180s, scale=0.4 → attempt 0=180s, 1=252s, 2=324s
    致命错误（is_fatal(): 401/403/4xx 认证/配置类）立即终止，不重试。
    可重试错误（限速/5xx/网络）耗尽全部尝试后记录完整上下文到日志文件与控制台。

    :param extra: 附加信息（如图片路径），失败时写入错误日志
    :param tools: 可选工具列表（OpenAI function calling 格式），传入后透传给 create(tools=...)
    :param model_key: 熔断/健康标记身份（路由路径传 model_id，缺省用 model_name）
    :param model_kwargs: 透传给 create() 的厂商专有参数（如 reasoning_effort / extra_body）
    :return: 无 tools 时返回 str（content），有 tools 时返回原生 message（含 .content/.tool_calls）
    """
    import random as _random

    from infra_ai.core.errors import ModelClientErrorType, classify_error
    from infra_ai.core.health_store import get_health_store

    last_error: Exception | None = None
    fail_reason: str = "unknown"
    timeouts_used: list[float] = []

    # 获取 model name（用于错误日志与健康标记）
    model_name = getattr(model, 'model_name', None) or str(model)
    health_key = model_key or model_name  # 熔断/健康标记统一身份

    for attempt in range(max_retries):
        # 线性递增超时
        current_timeout = timeout * (1 + timeout_scale * attempt)
        timeouts_used.append(current_timeout)

        # --- acquire 槽位 ---
        try:
            await rate_limiter.acquire()
        except TimeoutError as e:
            fail_reason = "ratelimit_acquire"
            last_error = e
            logger.warning(
                "[%s] RateLimiter 等待超时 (attempt %d/%d): %s",
                label, attempt + 1, max_retries, e,
            )
            if attempt < max_retries - 1:
                jitter = _random.uniform(-0.3, 0.3) * backoff * (attempt + 1)
                delay = max(0.5, backoff * (attempt + 1) + jitter)
                logger.debug("[%s] %.1fs 后重试（等待限速缓解）...", label, delay)
                await asyncio.sleep(delay)
            continue

        # --- API 调用 ---
        t0 = time.monotonic()

        try:
            # 非流式调用；tools / model_kwargs 透传 create()
            response = await asyncio.wait_for(
                model.acomplete(messages, tools=tools, extra_kwargs=model_kwargs),
                timeout=current_timeout,
            )
            elapsed = time.monotonic() - t0

            usage = _extract_token_usage(response)
            stats.record(elapsed, usage.get('input_tokens', 0), usage.get('output_tokens', 0))
            rate_limiter.record_tokens(usage.get('total_tokens', 0))
            rate_limiter.observe(usage.get('total_tokens', 0), elapsed)

            logger.debug(
                "[%s] %.2fs | model=%s | in=%d out=%d total=%d | rpm=%d tpm=%d | %s",
                label,
                elapsed,
                model_name,
                usage.get('input_tokens', 0),
                usage.get('output_tokens', 0),
                usage.get('total_tokens', 0),
                rate_limiter.current_rpm,
                rate_limiter.current_tpm,
                stats.summary(),
            )

            # 熔断器：标记成功（统一身份）
            get_health_store().mark_success(health_key)

            emit_obs({
                "type": "llm_call",
                "mode": "nonstream",
                "model": model_name,
                "route": health_key,
                "has_tools": bool(tools),
                "attempts": attempt + 1,
                "elapsed_ms": round(elapsed * 1000, 1),
                "input_tokens": usage.get('input_tokens', 0),
                "output_tokens": usage.get('output_tokens', 0),
                "total_tokens": usage.get('total_tokens', 0),
                "success": True,
                "fail_reason": None,
                "err_type": None,
            })

            message = response.choices[0].message
            # 有工具时返回完整 message（调用方需要 .tool_calls）
            if tools:
                return message
            return message.content if message.content is not None else ""

        except TimeoutError:
            elapsed = time.monotonic() - t0
            fail_reason = "api_timeout"
            last_error = TimeoutError(
                f"API 调用超时 {elapsed:.0f}s (limit={current_timeout:.0f}s)"
            )
            logger.warning(
                "[%s] API超时 %.0fs/%.0fs (attempt %d/%d) | rpm=%d tpm=%d",
                label, elapsed, current_timeout, attempt + 1, max_retries,
                rate_limiter.current_rpm, rate_limiter.current_tpm,
            )

        except Exception as e:
            elapsed = time.monotonic() - t0
            fail_reason = "api_error"
            last_error = e
            logger.warning(
                "[%s] 调用失败 (attempt %d/%d, %.1fs): %s",
                label, attempt + 1, max_retries, elapsed, e,
            )

            # P2.2: 致命错误（认证/权限/4xx 配置类）立即终止，不进入退避重试。
            err_type = classify_error(e)
            if err_type.is_fatal():
                fail_reason = f"api_error/{err_type.value}"
                logger.error("[%s] 致命错误 %s，终止重试: %s", label, err_type.value, e)
                break

        finally:
            rate_limiter.release()

        # --- 退避重试 ---
        if attempt < max_retries - 1:
            jitter = _random.uniform(-0.3, 0.3) * backoff * (attempt + 1)
            delay = max(0.5, backoff * (attempt + 1) + jitter)
            logger.debug(
                "[%s] %.1fs 后重试 (reason=%s, attempt=%d/%d, next_timeout=%.0fs)...",
                label, delay, fail_reason, attempt + 1, max_retries,
                timeout * (1 + timeout_scale * (attempt + 1)),
            )
            await asyncio.sleep(delay)

    # ---- 重试耗尽（或致命错误）：记录失败、写详细错误日志 ----
    # 错误分类 + 熔断器标记（统一身份）
    err_type = classify_error(last_error) if last_error else ModelClientErrorType.UNKNOWN
    if err_type.should_retry():
        # 临时性错误（限速/服务端/网络）→ 标记失败，触发熔断
        get_health_store().mark_failure(health_key)
    elif err_type.is_fatal():
        logger.error("[%s] 致命错误 %s: %s", label, err_type.value, last_error)
        # 致命错误不触发熔断（问题在配置，不在模型）
    fail_reason = f"{fail_reason}/{err_type.value}"
    stats.record_failure()
    emit_obs({
        "type": "llm_call",
        "mode": "nonstream",
        "model": model_name,
        "route": health_key,
        "has_tools": bool(tools),
        "attempts": attempt + 1,
        "elapsed_ms": round(elapsed * 1000, 1),
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "success": False,
        "fail_reason": fail_reason,
        "err_type": err_type.value,
    })
    _write_error_log(
        label=label,
        model_name=model_name,
        messages=messages,
        fail_reason=fail_reason,
        max_retries=max_retries,
        last_error=last_error,
        stats=stats,
        rate_limiter=rate_limiter,
        timeouts_used=timeouts_used,
        extra=extra,
    )

    raise RuntimeError(
        f"[{label}] 重试 {max_retries} 次后仍失败 "
        f"(最后原因: {fail_reason}): {last_error}"
    ) from last_error


# ============================================================
# 异步调用（公开 API）
# ============================================================

async def async_call_llm(
    messages: list[dict[str, Any]],
    use_json: bool = True,
    *,
    extra: dict[str, Any] | None = None,
    model_name: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> str:
    """
    异步调用文本大模型，带多模型路由、熔断器、速率限制和自动重试。

    :param messages: 消息列表
    :param use_json: 是否绑定 JSON 输出格式
    :param extra: 附加信息，失败时写入错误日志
    :param model_name: 指定模型名称（如 "Qwen/Qwen3.6-27B"），不传则使用默认路由
    :param model_kwargs: 透传给 create() 的厂商专有参数（如 reasoning_effort / extra_body）
    :return: 模型响应文本
    """
    # 如果指定了模型名称，创建临时模型实例
    if model_name:
        temp_model = _build_openai(model_name, use_json=use_json, temperature=0.7)

        return await _invoke_with_retry(
            temp_model, messages,
            rate_limiter=_get_text_rate_limiter(),
            stats=_text_stats,
            label="text",
            extra=extra,
            model_kwargs=model_kwargs,
        )

    # 尝试多模型路由
    try:
        from infra_ai.core.health_store import get_health_store
        from infra_ai.core.router import get_router
        router = get_router()

        async def _call_with_target(target) -> str:
            model = _get_model_for_target(target, use_json)
            if not get_health_store().allow_call(target.model_id):
                raise _CircuitOpenError(target.model_id)
            return await _invoke_with_retry(
                model, messages,
                rate_limiter=_get_rate_limiter(target.model_id, target.model_name),
                stats=_text_stats,
                label="text",
                extra=extra,
                model_key=target.model_id,
                model_kwargs=model_kwargs,
            )

        return await router.execute("chat", _call_with_target,
                                     first_choice_id=_get_first_choice("chat"))
    except ImportError:
        # 路由模块不可用（异常安装态）→ 单模型回退。
        # 注意：router.execute 抛出的 RuntimeError（所有候选失败）不在此拦截，
        # 避免静默回退复用同一失败的端点，掩盖真实错误。
        pass

    # 单模型回退
    llm = _get_text_llm()
    model = llm
    if use_json:
        model = llm.bind(response_format={"type": "json_object"})

    from infra_ai.core.health_store import get_health_store
    model_name = getattr(model, 'model_name', None) or str(llm.model_name)
    if not get_health_store().allow_call(model_name):
        raise _CircuitOpenError(model_name)

    return await _invoke_with_retry(
        model, messages,
        rate_limiter=_get_text_rate_limiter(),
        stats=_text_stats,
        label="text",
        extra=extra,
        model_kwargs=model_kwargs,
    )


async def async_call_llm_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    extra: dict[str, Any] | None = None,
    model_name: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
):
    """
    异步调用文本大模型（带原生工具调用），透传 OpenAI tools。

    复用与 async_call_llm 相同的多模型路由、熔断器、速率限制和自动重试。

    :param messages: 消息列表
    :param tools: 工具 schema 列表（OpenAI function calling 格式）
    :param extra: 附加信息，失败时写入错误日志
    :param model_name: 指定模型名称，不传则使用默认路由
    :param model_kwargs: 透传给 create() 的厂商专有参数
    :return: 原生 message 对象（含 .content 和 .tool_calls 属性）
    """
    # 如果指定了模型名称，创建临时模型实例
    if model_name:
        temp_model = _build_openai(model_name, temperature=0.7)
        return await _invoke_with_retry(
            temp_model, messages,
            rate_limiter=_get_text_rate_limiter(),
            stats=_text_stats,
            label="text+tools",
            extra=extra,
            tools=tools,
            model_kwargs=model_kwargs,
        )

    # 尝试多模型路由
    try:
        from infra_ai.core.health_store import get_health_store
        from infra_ai.core.router import get_router
        router = get_router()

        async def _call_with_target(target):
            model = _get_model_for_target(target, use_json=False)
            if not get_health_store().allow_call(target.model_id):
                raise _CircuitOpenError(target.model_id)
            return await _invoke_with_retry(
                model, messages,
                rate_limiter=_get_rate_limiter(target.model_id, target.model_name),
                stats=_text_stats,
                label="text+tools",
                extra=extra,
                tools=tools,
                model_key=target.model_id,
                model_kwargs=model_kwargs,
            )

        return await router.execute("chat", _call_with_target,
                                     first_choice_id=_get_first_choice("chat"))
    except ImportError:
        pass  # 路由模块不可用 → 单模型回退（RuntimeError 向上抛出，见 async_call_llm）

    # 单模型回退
    llm = _get_text_llm()
    model = llm
    from infra_ai.core.health_store import get_health_store
    model_name_str = getattr(model, 'model_name', None) or str(llm.model_name)
    if not get_health_store().allow_call(model_name_str):
        raise _CircuitOpenError(model_name_str)

    return await _invoke_with_retry(
        model, messages,
        rate_limiter=_get_text_rate_limiter(),
        stats=_text_stats,
        label="text+tools",
        extra=extra,
        tools=tools,
        model_kwargs=model_kwargs,
    )


async def async_call_vlm(
    messages: list[dict[str, Any]],
    use_json: bool = True,
    images: list[str] | None = None,
    *,
    extra: dict[str, Any] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> str:
    """
    异步调用视觉大模型（VLM），带速率限制、超时和自动重试。

    :param messages: 消息列表
    :param use_json: 是否绑定 JSON 输出格式
    :param images: 可选图片 URL/base64 列表
    :param extra: 附加信息（如图片路径），失败时写入错误日志
    :param model_kwargs: 透传给 create() 的厂商专有参数
    :return: 模型响应文本
    """
    vision_llm = _get_vision_llm()
    model = vision_llm
    if use_json:
        model = vision_llm.bind(response_format={"type": "json_object"})

    # 如果提供了 images 参数，追加到 messages 中
    if images:
        msgs = [dict(m) for m in messages]  # shallow copy
        last_user = None
        for m in reversed(msgs):
            if m.get("role") == "user":
                last_user = m
                break
        if last_user is None:
            last_user = {"role": "user", "content": []}
            msgs.append(last_user)

        content = last_user.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})
        last_user["content"] = content
        messages = msgs

    # 尝试多模型路由
    try:
        from infra_ai.core.health_store import get_health_store
        from infra_ai.core.router import get_router
        router = get_router()

        async def _call_with_target(target) -> str:
            m = _get_model_for_target(target, use_json)
            if not get_health_store().allow_call(target.model_id):
                raise _CircuitOpenError(target.model_id)
            return await _invoke_with_retry(
                m, messages,
                rate_limiter=_get_rate_limiter(target.model_id, target.model_name),
                stats=_vision_stats,
                label="vision",
                extra=extra,
                model_key=target.model_id,
                model_kwargs=model_kwargs,
            )

        return await router.execute("vision", _call_with_target,
                                     first_choice_id=_get_first_choice("vision"))
    except ImportError:
        pass  # 路由模块不可用 → 单模型回退（RuntimeError 向上抛出）

    # 单模型回退
    from infra_ai.core.health_store import get_health_store
    model_name = getattr(model, 'model_name', None) or str(vision_llm.model_name)
    if not get_health_store().allow_call(model_name):
        raise _CircuitOpenError(model_name)

    return await _invoke_with_retry(
        model, messages,
        rate_limiter=_get_vision_rate_limiter(),
        stats=_vision_stats,
        label="vision",
        extra=extra,
        model_kwargs=model_kwargs,
    )


async def async_call_llm_batch(
    requests: list[tuple[list[dict[str, Any]], bool]],
    *,
    return_exceptions: bool = False,
) -> list:
    """
    并发调用文本 LLM（受速率限制器约束）。

    :param requests: [(messages, use_json), ...] 列表
    :param return_exceptions: True 时单点失败不连坐，异常作为结果项返回原地
    :return: 与输入顺序对应的响应列表
    """
    tasks = [async_call_llm(msgs, use_json) for msgs, use_json in requests]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


async def async_call_vlm_batch(
    requests: list[tuple[list[dict[str, Any]], bool, list[str] | None]],
    *,
    return_exceptions: bool = False,
) -> list:
    """
    并发调用 VLM（受速率限制器约束）。

    :param requests: [(messages, use_json, images), ...] 列表
    :param return_exceptions: True 时单点失败不连坐，异常作为结果项返回原地
    :return: 与输入顺序对应的响应列表
    """
    tasks = [async_call_vlm(msgs, use_json, imgs) for msgs, use_json, imgs in requests]
    return await asyncio.gather(*tasks, return_exceptions=return_exceptions)


# ============================================================
# 工具函数
# ============================================================

def local_image_to_data_url(image_path: str) -> str:
    """将本地图片转为 base64 data URL，供 VLM 使用。"""
    file_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(file_path.name)
    if not mime_type:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"