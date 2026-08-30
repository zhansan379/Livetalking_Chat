"""
ASR 候选池 + 熔断回退驱动。

设计对齐 infra_ai/core/router.py 的 `iterate_candidates()` failover 语义：
- 候选按 config.yaml 的 routing.candidates 声明，priority 小优先；
- `transcribe` 顺序遍历 enabled 候选 → 熔断过滤 → 引擎懒实例化+可用性检查 →
  调用引擎 `_transcribe_observed`；成功（无 error）立即返回，异常 mark_failure 继续下一候选；
- 全候选失败返回携带 last_error 的失败结果（不抛，WS 层决定回包语义）。

回退粒度 = 一次转录（整段音频已有后遍历），非跨请求流式切。
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field

from registry import create as registry_create
from utils.logger import logger
from .base import BaseASR, TranscriptionResult
from .health_store import HealthStore


@dataclass
class ASRCandidate:
    """单个候选：对应 registry "stt" 槽的一个引擎及其启用/优先级/专属参数。"""

    id: str
    engine: str
    enabled: bool = True
    priority: int = 100
    params: dict = field(default_factory=dict)


class ASRPool:
    """候选池：构建候选列表 + 熔断 + 引擎懒实例化 + 回退驱动。"""

    def __init__(self, routing_cfg: dict, breaker_cfg: dict):
        self._health = HealthStore(
            failure_threshold=breaker_cfg.get("failure_threshold", 2),
            open_duration_sec=breaker_cfg.get("open_duration_sec", 30),
        )
        cands = routing_cfg.get("candidates", []) or []
        self._candidates = [
            ASRCandidate(
                id=c.get("id"),
                engine=c.get("engine"),
                enabled=c.get("enabled", True),
                priority=c.get("priority", 100),
                params=c.get("params") or {},
            )
            for c in cands
            if c.get("id") and c.get("engine")
        ]
        self._candidates.sort(key=lambda c: c.priority)
        self._engines: dict[str, BaseASR] = {}  # engine 名 → 实例（懒）
        self.default_engine = routing_cfg.get("default_engine")

    # ── 引擎懒实例化 ──
    def _get_engine(self, cand: ASRCandidate) -> BaseASR:
        if cand.engine not in self._engines:
            # importlib 触发 @register("stt", engine)；引擎名 = 模块名 = registry 名。
            importlib.import_module(f"asr.{cand.engine}")
            self._engines[cand.engine] = registry_create(
                "stt", cand.engine, opt={"params": cand.params},
            )
        return self._engines[cand.engine]

    # ── 挂载 gate：是否存在 ≥1 个 enabled 且依赖可用的候选 ──
    def available(self) -> bool:
        for cand in self._candidates:
            if not cand.enabled:
                continue
            if not self._health.allow_call(cand.id):
                continue
            try:
                engine = self._get_engine(cand)
            except Exception as e:  # noqa: BLE001
                logger.warning("[ASR] 候选 %s 引擎加载失败: %s", cand.id, e)
                continue
            try:
                if engine.is_available():
                    return True
            except Exception as e:  # noqa: BLE001
                logger.warning("[ASR] 候选 %s 可用性检查异常: %s", cand.id, e)
        return False

    def list_stats(self) -> dict:
        """候选健康统计（供监控/日志）。"""
        return self._health.get_stats()

    # ── 回退驱动（阻塞；由 WS handler 放 run_in_executor 在线程执行）──
    def transcribe(self, audio_float32, sample_rate: int, use_itn: bool, *,
                   trace_id: str, session_id: str) -> TranscriptionResult:
        last_error: Exception | None = None
        tried = 0
        for cand in self._candidates:
            if not cand.enabled:
                continue
            if not self._health.allow_call(cand.id):
                logger.info("[ASR] 候选 %s 熔断中，跳过", cand.id)
                continue
            # 引擎可用性（依赖可导入）
            try:
                engine = self._get_engine(cand)
                if not engine.is_available():
                    raise ImportError(f"engine '{cand.engine}' 依赖不可用")
            except Exception as e:  # noqa: BLE001
                self._health.mark_failure(cand.id)
                last_error = e
                logger.warning("[ASR] 候选 %s 不可用: %s", cand.id, e)
                continue

            tried += 1
            logger.info("[ASR] 尝试候选 %s (attempt #%d)", cand.id, tried)
            res = engine._transcribe_observed(
                audio_float32, sample_rate, use_itn,
                trace_id=trace_id, session_id=session_id,
            )
            if res.error is None:
                res.attempts = tried
                res.retried = tried > 1
                self._health.mark_success(cand.id)
                if res.retried:
                    logger.info("[ASR] 候选 %s 回退成功（放弃前 %d 个失败候选）", cand.id, tried - 1)
                return res
            self._health.mark_failure(cand.id)
            last_error = res.error
            logger.warning("[ASR] 候选 %s 转录失败: %s", cand.id, res.error)

        # 全候选失败：携带 last_error 返回，不抛（WS 层决定回包）。
        err = last_error or RuntimeError("all ASR candidates failed")
        return TranscriptionResult(
            text="", engine_id="", error=err, empty_text=False,
            attempts=tried, retried=tried > 1,
            fail_reason="all_asr_candidates_failed", err_type=err.__class__.__name__,
        )


# 全局单例（仿 infra_ai get_router）
_pool: ASRPool | None = None
_pool_lock = __import__("threading").Lock()


def get_pool() -> ASRPool:
    """获取全局候选池单例（懒加载，读取 asr 配置）。"""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                from .config_loader import get_config
                cfg = get_config()
                _pool = ASRPool(cfg.ROUTING, cfg.CIRCUIT_BREAKER)
    return _pool


def reset_pool():
    """重置单例（调试/测试用）。"""
    global _pool
    _pool = None