"""
三态熔断器 (CLOSED → OPEN → HALF_OPEN → CLOSED)，按候选 id 独立跟踪健康状态。

设计对齐 infra_ai/core/health_store.py，但轻量且自包含——不 import infra_ai，
保证 asr 包在 infra_ai 缺失时仍可独立运行。key 取候选 id（config.yaml 的
routing.candidates[].id，如 "sensevoice"），而非模型名。

用法:
    breaker = HealthStore(failure_threshold=2, open_duration_sec=30)
    if breaker.allow_call(cand.id):
        try:
            ...engine.transcribe(...)
            breaker.mark_success(cand.id)
        except:
            breaker.mark_failure(cand.id)
    else:
        # 熔断中，跳过候选去下一个
        continue
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
class CandidateHealth:
    """单个候选的健康状态。"""
    consecutive_failures: int = 0
    open_until: float = 0.0        # 熔断到期时间（time.monotonic）
    half_open_inflight: bool = False  # 半开状态是否已有探活请求在进行

    @property
    def state(self) -> CircuitState:
        if self.open_until > 0 and time.monotonic() < self.open_until:
            return CircuitState.OPEN
        if self.open_until > 0 and self.half_open_inflight:
            return CircuitState.HALF_OPEN
        return CircuitState.CLOSED


class HealthStore:
    """线程安全（threading.Lock）的候选健康状态存储。"""

    def __init__(self, failure_threshold: int = 2, open_duration_sec: float = 30.0):
        self._lock = threading.Lock()
        self._health: dict[str, CandidateHealth] = {}
        self.failure_threshold = failure_threshold
        self.open_duration_sec = open_duration_sec

    def allow_call(self, candidate_id: str) -> bool:
        """
        是否允许调用该候选。返回 True 可调用，False 表示熔断中应跳过。
        HALF_OPEN 状态只放一个探活请求通过。
        """
        with self._lock:
            health = self._health.get(candidate_id)
            if health is None:
                return True  # 新候选，默认健康

            state = health.state
            if state == CircuitState.CLOSED:
                return True
            if state == CircuitState.OPEN:
                logger.debug("候选 %s 熔断中 (剩余 %ds)", candidate_id,
                             int(health.open_until - time.monotonic()))
                return False
            if state == CircuitState.HALF_OPEN:
                if health.half_open_inflight:
                    return False
                health.half_open_inflight = True
                logger.info("候选 %s 进入探活请求", candidate_id)
                return True
            return True

    def mark_success(self, candidate_id: str):
        """标记调用成功，重置健康状态。"""
        with self._lock:
            health = self._health.get(candidate_id)
            if health is not None:
                old_state = health.state
                health.consecutive_failures = 0
                health.open_until = 0.0
                health.half_open_inflight = False
                if old_state == CircuitState.HALF_OPEN:
                    logger.info("候选 %s 探活成功，恢复为 CLOSED", candidate_id)

    def mark_failure(self, candidate_id: str):
        """
        标记调用失败。CLOSED 下递增失败计数达阈值进入 OPEN；HALF_OPEN 下直接进入 OPEN。
        """
        with self._lock:
            health = self._health.setdefault(candidate_id, CandidateHealth())
            old_state = health.state

            health.consecutive_failures += 1
            if (old_state == CircuitState.HALF_OPEN
                    or health.consecutive_failures >= self.failure_threshold):
                health.open_until = time.monotonic() + self.open_duration_sec
                health.half_open_inflight = False
                if old_state == CircuitState.HALF_OPEN:
                    logger.warning("候选 %s 探活失败，重新熔断 %.0fs",
                                   candidate_id, self.open_duration_sec)
                else:
                    logger.warning("候选 %s 连续失败 %d 次，进入熔断 %.0fs",
                                   candidate_id, health.consecutive_failures,
                                   self.open_duration_sec)

    def get_state(self, candidate_id: str) -> CircuitState:
        """获取候选当前状态（用于监控/日志）。"""
        health = self._health.get(candidate_id)
        return health.state if health else CircuitState.CLOSED

    def get_stats(self) -> dict[str, dict]:
        """获取所有候选的健康统计。"""
        with self._lock:
            return {
                cid: {
                    "state": h.state.value,
                    "failures": h.consecutive_failures,
                    "open_remaining": (max(0, int(h.open_until - time.monotonic()))
                                       if h.open_until else 0),
                }
                for cid, h in self._health.items()
            }