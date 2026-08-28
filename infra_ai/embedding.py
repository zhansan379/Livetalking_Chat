"""
Embedding 服务客户端（带模型路由和降级策略）。

支持文本嵌入(1536d)和多模态VL嵌入(768d)，复用 infra_ai 熔断器和错误分类。

用法:
    from infra_ai.embedding import get_embedding_client
    client = get_embedding_client()
    vectors = client.embed_batch(texts, dimensions=1536)
"""

import asyncio
import base64
import logging
import os
import time
from collections.abc import Sequence
from pathlib import Path as _Path
from typing import Any, Union

import requests

from infra_ai.core._async_utils import run_async_in_sync
from infra_ai.core.config_loader import get_config as _get_config
from infra_ai.core.rate_limiter import RateLimiter
from infra_ai.core.router import ModelCandidate, iterate_candidates
from infra_ai.core.stats import _CallStats, _snapshot, get_stat

logger = logging.getLogger(__name__)

EmbeddingInput = Union[str, Sequence[str]]
MultimodalInput = list[str | dict[str, str]]

# VL Embedding 模型常量
VL_EMBEDDING_DIM = 768

_IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def _resolve_image_input(path_or_url: str) -> str:
    """本地路径 → base64 data URL；URL/data URL → 原样返回。"""
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    p = _Path(path_or_url)
    if p.is_file():
        ext = p.suffix.lower()
        mime = _IMAGE_MIME.get(ext, "image/jpeg")
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return path_or_url


# ----------------------------------------------------------------
# Embedding 候选配置
# ----------------------------------------------------------------

def _get_embedding_candidates(capability: str = "text") -> list[ModelCandidate]:
    """
    从 config.EMBEDDING_ROUTING 加载 embedding 候选模型列表（类型化 ModelCandidate）。

    配置在 config.yaml 的 embedding.routing[capability].candidates（经 config_loader 读取）中，
    支持 text 和 vl 两种能力，每项含 id/provider/model/base_url/api_key/dimensions/priority/enabled。
    """
    EMBEDDING_ROUTING = _get_config().EMBEDDING_ROUTING
    cap = EMBEDDING_ROUTING.get(capability, {})
    result: list[ModelCandidate] = []
    for c in cap.get("candidates", []):
        if not c.get("enabled", True):
            continue
        result.append(ModelCandidate(
            id=c.get("id", ""),
            provider=c.get("provider", ""),
            model=c.get("model", ""),
            priority=c.get("priority", 100),
            api_key=c.get("api_key", ""),
            base_url=c.get("base_url", ""),
            dimensions=c.get("dimensions", 0),
        ))
    return result


def _build_embedding_rate_limiter(capability: str) -> RateLimiter:
    """从 config.EMBEDDING_RATE_LIMITS 构建指定能力的限速器。"""
    EMBEDDING_RATE_LIMITS = _get_config().EMBEDDING_RATE_LIMITS
    limits = EMBEDDING_RATE_LIMITS.get(capability, EMBEDDING_RATE_LIMITS.get("default", {}))
    return RateLimiter(
        rpm=limits.get("rpm", 60),
        tpm=limits.get("tpm", 0),
        max_concurrent=limits.get("max_concurrent", 10),
    )


# ----------------------------------------------------------------
# Embedding Client
# ----------------------------------------------------------------

class EmbeddingClient:
    """Embedding 客户端（单 provider，带重试）。"""

    def __init__(self, candidate: ModelCandidate, timeout: int = 120, max_retries: int = 3,
                 stats: _CallStats | None = None, rate_limiter: RateLimiter | None = None):
        self._id = candidate.id
        self._provider = candidate.provider
        self._model = candidate.model
        self._vl_model = candidate.model
        self._api_key = candidate.api_key
        self._base_url = candidate.base_url.rstrip("/")
        self._url = f"{self._base_url}/embeddings"
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self.timeout_scale = _get_config().EMBEDDING_TIMEOUT_SCALE
        self.max_retries = max_retries
        self.stats = stats
        self.rate_limiter = rate_limiter
        self.last_elapsed = 0.0

    def _record_usage(self, data: dict[str, Any]) -> None:
        """解析 OpenAI 兼容 usage 并累计到统计累加器 / 限速器（复用 simplify后一行对齐 inference 路径）。"""
        usage = data.get("usage", {}) or {}
        total = int(usage.get("total_tokens", 0) or 0)
        in_tok = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        out_tok = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
        if self.stats:
            self.stats.record(self.last_elapsed, in_tok, out_tok)
        if self.rate_limiter:
            self.rate_limiter.record_tokens(total)
            self.rate_limiter.observe(total, self.last_elapsed)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """带超时和指数退避的 POST 请求。"""
        last_error: Exception | None = None
        backoff = 2.0

        for attempt in range(self.max_retries):
            current_timeout = self.timeout * (1 + self.timeout_scale * attempt)
            t0 = time.monotonic()
            try:
                resp = requests.post(self._url, json=payload, headers=self._headers,
                                     timeout=current_timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(
                        f"Embedding[{self._id}] status={resp.status_code}: {resp.text[:300]}")
                self.last_elapsed = time.monotonic() - t0
                return resp.json()
            except requests.exceptions.Timeout as e:
                last_error = e
            except requests.exceptions.ConnectionError as e:
                last_error = e
            except Exception as e:
                last_error = e

            if attempt < self.max_retries - 1:
                time.sleep(backoff ** (attempt + 1))

        raise RuntimeError(
            f"Embedding[{self._id}] 重试 {self.max_retries} 次后仍失败: {last_error}"
        ) from last_error

    def embed_batch(self, texts: Sequence[str], dimensions: int | None = 1536) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self._model, "input": list(texts),
                                     "encoding_format": "float"}
        if dimensions:
            payload["dimensions"] = dimensions
        data = self._post(payload)
        self._record_usage(data)
        return [[float(v) for v in item["embedding"]] for item in data.get("data", [])]

    def embed_image_with_context(self, image_url_or_base64: str, context_text: str,
                                  dimensions: int = VL_EMBEDDING_DIM) -> list[float]:
        if not self._vl_model:
            raise RuntimeError(f"Provider {self._provider} 不支持 VL embedding")
        img = _resolve_image_input(image_url_or_base64)
        inputs = [{"text": context_text}, {"image": img}]
        payload = {"model": self._vl_model, "input": inputs, "encoding_format": "float"}
        if dimensions:
            payload["dimensions"] = dimensions
        data = self._post(payload)
        self._record_usage(data)
        return [float(v) for v in data["data"][0]["embedding"]]

    def embed_text_vl(self, text: str, dimensions: int = VL_EMBEDDING_DIM) -> list[float]:
        """使用 VL embedding 模型对纯文本编码（用于跨模态检索）。"""
        if not self._vl_model:
            raise RuntimeError(f"Provider {self._provider} 不支持 VL embedding")
        payload = {"model": self._vl_model, "input": [{"text": text}],
                     "encoding_format": "float"}
        if dimensions:
            payload["dimensions"] = dimensions
        data = self._post(payload)
        self._record_usage(data)
        return [float(v) for v in data["data"][0]["embedding"]]


# ----------------------------------------------------------------
# 路由 Embedding Client（带熔断和降级）
# ----------------------------------------------------------------

class RoutedEmbeddingClient:
    """
    带多模型路由和降级策略的 Embedding 客户端。

    从 config.EMBEDDING_ROUTING 读取候选，复用 infra_ai 熔断器。
    text 和 vl 两种能力分别路由。
    """

    def __init__(self):
        text_limiter = _build_embedding_rate_limiter("text")
        text_stats = get_stat("embedding_text")
        self._text_clients: dict[str, EmbeddingClient] = {}
        text_cands = _get_embedding_candidates("text")
        for c in text_cands:
            self._text_clients[c.id] = EmbeddingClient(
                c, stats=text_stats, rate_limiter=text_limiter)
        self._text_candidates = sorted(text_cands, key=lambda c: c.priority)

        vl_limiter = _build_embedding_rate_limiter("vl")
        vl_stats = get_stat("embedding_vl")
        self._vl_clients: dict[str, EmbeddingClient] = {}
        vl_cands = _get_embedding_candidates("vl")
        for c in vl_cands:
            self._vl_clients[c.id] = EmbeddingClient(
                c, stats=vl_stats, rate_limiter=vl_limiter)
        self._vl_candidates = sorted(vl_cands, key=lambda c: c.priority)

    async def _route_async(self, candidates: list[ModelCandidate],
                           clients: dict[str, EmbeddingClient],
                           method: str, *args, **kwargs) -> Any:
        """异步路由调用，委托共享驱动 iterate_candidates；同步叶子经 to_thread 包成 async。"""
        from infra_ai.core.errors import classify_error
        from infra_ai.core.health_store import get_health_store

        health = get_health_store()

        async def attempt(c: ModelCandidate):
            client = clients[c.id]
            await client.rate_limiter.acquire()
            try:
                result = await asyncio.to_thread(
                    getattr(client, method), *args, **kwargs
                )
                health.mark_success(c.id)
                return result
            except Exception as e:
                if classify_error(e).should_retry():
                    health.mark_failure(c.id)
                raise
            finally:
                client.rate_limiter.release()

        return await iterate_candidates(candidates, attempt, label="embedding")

    def embed(self, text: str, dimensions: int | None = 1536) -> list[float]:
        result = self.embed_batch([text], dimensions)
        return result[0] if result else []

    def embed_batch(self, texts: Sequence[str], dimensions: int | None = 1536) -> list[list[float]]:
        return run_async_in_sync(self._route_async(
            self._text_candidates, self._text_clients,
            "embed_batch", texts, dimensions))

    def embed_image_with_context(self, image_url_or_base64: str, context_text: str,
                                  dimensions: int = VL_EMBEDDING_DIM) -> list[float]:
        return run_async_in_sync(self._route_async(
            self._vl_candidates, self._vl_clients,
            "embed_image_with_context", image_url_or_base64, context_text, dimensions))

    def embed_text_vl(self, text: str, dimensions: int = VL_EMBEDDING_DIM) -> list[float]:
        return run_async_in_sync(self._route_async(
            self._vl_candidates, self._vl_clients,
            "embed_text_vl", text, dimensions))


# ----------------------------------------------------------------
# 全局单例
# ----------------------------------------------------------------

_client: RoutedEmbeddingClient | None = None


def get_embedding_client() -> RoutedEmbeddingClient:
    global _client
    if _client is None:
        _client = RoutedEmbeddingClient()
    return _client


def get_embedding_stats() -> dict:
    """返回 Embedding（text / vl）的累计调用统计。"""
    return {
        "text": _snapshot(get_stat("embedding_text")),
        "vl": _snapshot(get_stat("embedding_vl")),
    }
