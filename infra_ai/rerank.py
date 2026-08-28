"""
Rerank 服务客户端（带模型路由和降级策略）。

支持文本重排序，复用 infra_ai 熔断器和错误分类。

用法:
    from infra_ai.rerank import rerank, async_rerank
    results = await async_rerank(query, documents, top_n=5)
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

from infra_ai.core._async_utils import run_async_in_sync
from infra_ai.core.config_loader import get_config as _get_config
_cfg = _get_config()
RERANK_RATE_LIMITS = _cfg.RERANK_RATE_LIMITS
RERANK_ROUTING = _cfg.RERANK_ROUTING
from infra_ai.core.errors import classify_error
from infra_ai.core.health_store import get_health_store
from infra_ai.core.rate_limiter import RateLimiter
from infra_ai.core.router import ModelCandidate, iterate_candidates
from infra_ai.core.stats import _CallStats, _snapshot, get_stat

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# 数据模型
# ----------------------------------------------------------------

@dataclass
class RerankResult:
    """单个文档的重排序结果。"""
    index: int
    relevance_score: float
    document: str | None = None


@dataclass
class RerankResponse:
    """Rerank 完整响应。"""
    results: list[RerankResult]
    total_tokens: int = 0
    input_tokens: int = 0


# ----------------------------------------------------------------
# RerankClient（单 Provider）
# ----------------------------------------------------------------

class RerankClient:
    """单 Provider Rerank 客户端（带重试）。"""

    def __init__(self, candidate: ModelCandidate, timeout: int = 60, max_retries: int = 3,
                 stats: _CallStats | None = None):
        self._id = candidate.id
        self._provider = candidate.provider
        self._model = candidate.model
        self._api_key = candidate.api_key
        self._base_url = candidate.base_url.rstrip("/")
        self._url = f"{self._base_url}/rerank"
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = 2.0
        self.stats = stats

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
        instruction: str | None = None,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
    ) -> RerankResponse:
        """
        执行重排序调用。

        :param query: 查询文本
        :param documents: 待重排序的文档列表
        :param top_n: 返回前 N 个结果
        :param return_documents: 是否返回文档文本
        :param instruction: reranker 指令（仅 Qwen3-Reranker 支持）
        :param max_chunks_per_doc: 每个文档最大分块数
        :param overlap_tokens: 分块重叠 token 数
        :return: RerankResponse
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "return_documents": return_documents,
        }
        if top_n is not None:
            payload["top_n"] = top_n
        if instruction is not None:
            payload["instruction"] = instruction
        if max_chunks_per_doc is not None:
            payload["max_chunks_per_doc"] = max_chunks_per_doc
        if overlap_tokens is not None:
            payload["overlap_tokens"] = overlap_tokens

        last_error: Exception | None = None
        t_start = time.monotonic()

        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(self._url, json=payload, headers=self._headers)
                    resp.raise_for_status()
                    data = resp.json()

                    results = []
                    for item in data.get("results", []):
                        results.append(RerankResult(
                            index=item["index"],
                            relevance_score=item["relevance_score"],
                            document=item.get("document", {}).get("text") if return_documents else None,
                        ))

                    meta = data.get("meta", {})
                    tokens = meta.get("tokens", {})

                    if self.stats:
                        in_tok = tokens.get("input_tokens", 0)
                        total_tok = tokens.get("total_tokens", 0)
                        self.stats.record(time.monotonic() - t_start, in_tok,
                                          max(0, total_tok - in_tok))

                    return RerankResponse(
                        results=results,
                        total_tokens=tokens.get("total_tokens", 0),
                        input_tokens=tokens.get("input_tokens", 0),
                    )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.backoff * (attempt + 1)
                    logger.warning("Rerank %s 失败 (attempt %d/%d): %s, %.1fs 后重试",
                                   self._id, attempt + 1, self.max_retries, e, delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Rerank %s 最终失败: %s", self._id, e)

        raise RuntimeError(f"Rerank {self._id} 重试 {self.max_retries} 次后仍失败") from last_error


# ----------------------------------------------------------------
# 路由辅助函数
# ----------------------------------------------------------------

def _get_rerank_candidates(capability: str = "text") -> list[ModelCandidate]:
    """从 config 加载 rerank 候选模型列表（类型化 ModelCandidate）。"""
    cap = RERANK_ROUTING.get(capability, {})
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
        ))
    return result


def _build_rerank_rate_limiter() -> RateLimiter:
    """构建 rerank 专用限流器。"""
    limits = RERANK_RATE_LIMITS.get("default", {"rpm": 60, "max_concurrent": 10})
    return RateLimiter(
        rpm=limits.get("rpm", 60),
        tpm=limits.get("tpm", 0),
        max_concurrent=limits.get("max_concurrent", 10),
    )


# ----------------------------------------------------------------
# RoutedRerankClient（多模型路由 + 降级）
# ----------------------------------------------------------------

class RoutedRerankClient:
    """
    带多模型路由和降级策略的 Rerank 客户端。

    复用 infra_ai 熔断器、错误分类、路由选择器。
    """

    def __init__(self):
        self._clients: dict[str, RerankClient] = {}
        candidates = _get_rerank_candidates("text")
        stats = get_stat("rerank")
        for c in candidates:
            self._clients[c.id] = RerankClient(c, stats=stats)
        self._sorted = sorted(candidates, key=lambda c: c.priority)
        self._rate_limiter = _build_rerank_rate_limiter()

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
        return_documents: bool = False,
        instruction: str | None = None,
        max_chunks_per_doc: int | None = None,
        overlap_tokens: int | None = None,
    ) -> RerankResponse:
        """
        带降级的重排序调用。委托共享路由-回退驱动 iterate_candidates。

        按优先级遍历候选模型，成功则返回，所有候选失败则抛异常。
        """
        health = get_health_store()

        async def attempt(c: ModelCandidate):
            client = self._clients[c.id]
            # acquire 在 try 之外：限速等待超时时不触发 release（与原先一致）
            await self._rate_limiter.acquire()
            try:
                result = await client.rerank(
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                    instruction=instruction,
                    max_chunks_per_doc=max_chunks_per_doc,
                    overlap_tokens=overlap_tokens,
                )
                health.mark_success(c.id)
                return result
            except Exception as e:
                if classify_error(e).should_retry():
                    health.mark_failure(c.id)
                raise
            finally:
                self._rate_limiter.release()

        return await iterate_candidates(self._sorted, attempt, label="rerank")


# ----------------------------------------------------------------
# 全局单例
# ----------------------------------------------------------------

_client: RoutedRerankClient | None = None


def get_rerank_client() -> RoutedRerankClient:
    """获取全局 Rerank 客户端单例。"""
    global _client
    if _client is None:
        _client = RoutedRerankClient()
    return _client


def get_rerank_stats() -> dict:
    """返回 Rerank 的累计调用统计。"""
    return {"rerank": _snapshot(get_stat("rerank"))}


# ----------------------------------------------------------------
# 公开接口
# ----------------------------------------------------------------

async def async_rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    return_documents: bool = False,
    model_name: str | None = None,
    **kwargs,
) -> RerankResponse:
    """
    异步重排序入口。

    :param query: 查询文本
    :param documents: 待重排序的文档列表
    :param top_n: 返回前 N 个结果
    :param return_documents: 是否返回文档文本
    :param model_name: 指定模型名称（当前版本未使用，预留）
    :return: RerankResponse
    """
    client = get_rerank_client()
    return await client.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=return_documents,
        **kwargs,
    )


def rerank(
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
    return_documents: bool = False,
    model_name: str | None = None,
    **kwargs,
) -> RerankResponse:
    """
    同步重排序入口（自动适配同步/异步上下文）。

    :param query: 查询文本
    :param documents: 待重排序的文档列表
    :param top_n: 返回前 N 个结果
    :param return_documents: 是否返回文档文本
    :param model_name: 指定模型名称（当前版本未使用，预留）
    :return: RerankResponse
    """
    return run_async_in_sync(
        async_rerank(query, documents, top_n=top_n,
                     return_documents=return_documents, **kwargs)
    )
