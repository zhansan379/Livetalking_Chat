"""
三态熔断器 (CLOSED → OPEN → HALF_OPEN → CLOSED)。

基于 t_rag_trace_node / t_rag_trace_run 的设计理念，为每个模型独立
跟踪健康状态。连续失败达到阈值即熔断，冷却期后进入半开状态探活。

用法:
    from infra_ai.core.health_store import get_health_store

    store = get_health_store()
    if store.allow_call("qwen2.5-72b"):
        try:
            result = do_llm_call()
            store.mark_success("qwen2.5-72b")
        except Exception:
            store.mark_failure("qwen2.5-72b")
    else:
        # 熔断中，选择下一个候选模型
        fallback()
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"        # 正常
    OPEN = "OPEN"            # 熔断
    HALF_OPEN = "HALF_OPEN"  # 探活


@dataclass
class ModelHealth:
    """单个模型的健康状态。"""
    consecutive_failures: int = 0
    open_until: float = 0.0       # 熔断到期时间（time.monotonic）
    half_open_inflight: bool = False  # 半开状态是否已有探活请求在进行

    @property
    def state(self) -> CircuitState:
        if self.open_until > 0 and time.monotonic() < self.open_until:
            return CircuitState.OPEN
        if self.open_until > 0 and self.half_open_inflight:
            return CircuitState.HALF_OPEN
        return CircuitState.CLOSED


class ModelHealthStore:
    """
    全局模型健康状态存储。

    线程安全：使用 threading.Lock 保护读写。
    """

    def __init__(self, failure_threshold: int = 2, open_duration_sec: float = 30.0):
        self._lock = threading.Lock()
        self._health: dict[str, ModelHealth] = {}
        self.failure_threshold = failure_threshold
        self.open_duration_sec = open_duration_sec

    def allow_call(self, model_id: str) -> bool:
        """
        检查是否允许调用该模型。

        返回 True 表示可以调用，False 表示熔断中应跳过。
        HALF_OPEN 状态下只允许一个探活请求通过。
        """
        with self._lock:
            health = self._health.get(model_id)
            if health is None:
                return True  # 新模型，默认健康

            state = health.state
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.OPEN:
                remaining = int(health.open_until - time.monotonic())
                logger.debug("模型 %s 熔断中 (剩余 %ds)", model_id, remaining)
                return False
            if state == CircuitState.HALF_OPEN:
                if health.half_open_inflight:
                    return False
                health.half_open_inflight = True
                logger.info("模型 %s 进入探活请求", model_id)
                return True
            return True

    def mark_success(self, model_id: str):
        """标记调用成功，重置健康状态。"""
        with self._lock:
            health = self._health.get(model_id)
            if health is not None:
                old_state = health.state
                health.consecutive_failures = 0
                health.open_until = 0.0
                health.half_open_inflight = False
                if old_state == CircuitState.HALF_OPEN:
                    logger.info("模型 %s 探活成功，恢复为 CLOSED", model_id)

    def mark_failure(self, model_id: str):
        """
        标记调用失败。

        CLOSED 下递增失败计数，达到阈值进入 OPEN。
        HALF_OPEN 下直接进入 OPEN。
        """
        with self._lock:
            health = self._health.setdefault(model_id, ModelHealth())
            old_state = health.state

            health.consecutive_failures += 1
            if old_state == CircuitState.HALF_OPEN or health.consecutive_failures >= self.failure_threshold:
                health.open_until = time.monotonic() + self.open_duration_sec
                health.half_open_inflight = False
                if old_state == CircuitState.HALF_OPEN:
                    logger.warning("模型 %s 探活失败，重新熔断 %.0fs", model_id, self.open_duration_sec)
                else:
                    logger.warning("模型 %s 连续失败 %d 次，进入熔断 %.0fs",
                                   model_id, health.consecutive_failures, self.open_duration_sec)

    def get_state(self, model_id: str) -> CircuitState:
        """获取模型当前状态（用于监控/日志）。"""
        health = self._health.get(model_id)
        return health.state if health else CircuitState.CLOSED

    def get_stats(self) -> dict[str, dict]:
        """获取所有模型的健康统计。"""
        with self._lock:
            return {
                mid: {
                    "state": h.state.value,
                    "failures": h.consecutive_failures,
                    "open_remaining": max(0, int(h.open_until - time.monotonic())) if h.open_until else 0,
                }
                for mid, h in self._health.items()
            }


# 全局单例
_store: ModelHealthStore | None = None


def get_health_store() -> ModelHealthStore:
    """获取全局熔断器单例（延迟初始化，读取 config 配置）。"""
    global _store
    if _store is None:
        from infra_ai.core.config_loader import get_config
        LLM_CIRCUIT_BREAKER = get_config().LLM_CIRCUIT_BREAKER
        threshold = LLM_CIRCUIT_BREAKER.get("failure_threshold", 2)
        open_sec = LLM_CIRCUIT_BREAKER.get("open_duration_sec", 30)
        _store = ModelHealthStore(
            failure_threshold=threshold,
            open_duration_sec=open_sec,
        )
    return _store
