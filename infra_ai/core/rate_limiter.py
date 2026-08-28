"""
速率限制器：RPM + TPM 滑动窗口 + 动态并发控制 + EMA 自适应调优。

供推理与流式路径复用。
"""

import asyncio
import logging
import time

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    异步速率限制器：控制 RPM、TPM 和最大并发数。

    - RPM:   60s 滑动窗口内的请求次数上限
    - TPM:   60s 滑动窗口内的 token 消耗上限（0 = 不限制）
    - 并发:  动态可调上限，基于 EMA 观测值自适应推荐
    - auto_tune: True=自动应用推荐值，False=仅日志建议（默认）
    """

    _TPM_POLL_INTERVAL = 10.0  # TPM 短轮询间隔（秒）

    def __init__(self, rpm: int, tpm: int = 0, max_concurrent: int = 10,
                 auto_tune: bool = False):
        self._rpm = max(rpm, 1)
        self._tpm = max(tpm, 0)
        self._max_concurrent = max_concurrent
        self._auto_tune = auto_tune

        # ---- 并发控制：Event + 计数器（支持动态调整） ----
        self._active: int = 0
        self._slot_free = asyncio.Event()
        self._slot_free.set()  # 初始有可用槽位

        # ---- RPM / TPM 滑动窗口 ----
        self._request_times: list[float] = []
        self._token_records: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()
        self._waiters: int = 0

        # ---- 自适应推荐（EMA 平滑） ----
        self._ema_alpha: float = 0.08
        self._token_ema: float = 0.0
        self._duration_ema: float = 0.0
        self._obs_count: int = 0
        self._last_recommend_log: float = 0.0

    # ------------------------------------------------------------
    # 并发槽位（替代 asyncio.Semaphore，支持 resize）
    # ------------------------------------------------------------

    async def _acquire_slot(self) -> None:
        """等待并占用一个并发槽位。"""
        while True:
            if self._active < self._max_concurrent:
                self._active += 1
                if self._active >= self._max_concurrent:
                    self._slot_free.clear()
                return
            await self._slot_free.wait()

    def _release_slot(self) -> None:
        """释放一个并发槽位（同步安全：asyncio 单线程，无 await 点间切换）。"""
        self._active -= 1
        self._slot_free.set()

    def _resize_concurrency(self, new_max: int) -> None:
        """动态调整并发上限。上调立即生效，下调自然衰减。"""
        if new_max == self._max_concurrent or new_max < 1:
            return
        old = self._max_concurrent
        self._max_concurrent = new_max
        if new_max > old:
            self._slot_free.set()  # 唤醒等待者
        # new_max < old: 已持槽不受影响，新 acquire 会被 _active < new_max 拦截

    # ------------------------------------------------------------
    # acquire / release
    # ------------------------------------------------------------

    async def acquire(self, max_wait: float = 300.0) -> None:
        """
        获取一个请求槽位（并发 + RPM + TPM 三重检查）。
        max_wait 秒后仍未获取到则抛出 asyncio.TimeoutError。
        """
        deadline = time.monotonic() + max_wait
        await self._acquire_slot()
        try:
            while True:
                async with self._lock:
                    now = time.monotonic()
                    cutoff = now - 60.0
                    self._request_times = [t for t in self._request_times if t > cutoff]
                    self._token_records = [(ts, n) for ts, n in self._token_records if ts > cutoff]

                    wait = 0.0
                    wait_reason = ""

                    if len(self._request_times) >= self._rpm:
                        rpm_wait = self._request_times[0] - cutoff + 0.05
                        if rpm_wait > wait:
                            wait = rpm_wait
                            wait_reason = "rpm"

                    if self._tpm > 0 and self._token_records:
                        current_tokens = sum(n for _, n in self._token_records)
                        if current_tokens >= self._tpm:
                            tpm_wait = self._token_records[0][0] - cutoff + 0.05
                            tpm_wait = min(tpm_wait, self._TPM_POLL_INTERVAL)
                            if tpm_wait > wait:
                                wait = tpm_wait
                                wait_reason = "tpm"

                    if wait == 0.0:
                        self._request_times.append(now)
                        return

                    remaining = deadline - now
                    if wait > remaining:
                        self._release_slot()
                        raise TimeoutError(
                            f"RateLimiter[{wait_reason}] 等待超时 ({max_wait:.0f}s): "
                            f"rpm={self.current_rpm}/{self._rpm}, "
                            f"tpm={self.current_tpm}/{self._tpm}, "
                            f"need_wait={wait:.1f}s > remaining={remaining:.1f}s"
                        )

                    if wait_reason == "tpm":
                        logger.debug(
                            "RateLimiter[tpm] waiting %.1fs (tpm=%d/%d, rpm=%d/%d, "
                            "waiters=%d, max_concurrent=%d)",
                            wait, self.current_tpm, self._tpm,
                            self.current_rpm, self._rpm, self._waiters,
                            self._max_concurrent,
                        )

                self._release_slot()
                self._waiters += 1
                try:
                    await asyncio.sleep(wait)
                finally:
                    self._waiters -= 1
                    await self._acquire_slot()
        except BaseException:
            self._release_slot()
            raise

    def record_tokens(self, token_count: int) -> None:
        """记录一次调用消耗的 token 数。"""
        if self._tpm > 0 and token_count > 0:
            self._token_records.append((time.monotonic(), token_count))

    def release(self) -> None:
        """释放并发槽位（调用完成后必须调用）。"""
        self._release_slot()

    # ------------------------------------------------------------
    # 自适应推荐
    # ------------------------------------------------------------

    def observe(self, tokens: int, duration: float) -> None:
        """
        记录一次调用的观测值，用于自适应推荐 max_concurrent。
        若 auto_tune=True 且推荐值变化 ≥2，自动生效。
        """
        if tokens <= 0:
            return
        if self._token_ema == 0.0:
            self._token_ema = float(tokens)
            self._duration_ema = duration
        else:
            self._token_ema = (
                self._ema_alpha * tokens + (1 - self._ema_alpha) * self._token_ema
            )
            self._duration_ema = (
                self._ema_alpha * duration + (1 - self._ema_alpha) * self._duration_ema
            )
        self._obs_count += 1

        rec = self.recommend_max_concurrent()
        now = time.monotonic()

        # 每 ~30s 或首次达到稳定观察数时输出日志
        if (now - self._last_recommend_log > 30.0
                and self._obs_count >= 5
                and rec != self._max_concurrent):
            self._last_recommend_log = now
            logger.info(
                "自适应推荐 max_concurrent=%d (当前=%d) | "
                "avg_tokens=%.0f avg_duration=%.1fs tpm=%d | obs=%d",
                rec, self._max_concurrent,
                self._token_ema, self._duration_ema, self._tpm,
                self._obs_count,
            )

        # auto_tune 模式：推荐值偏差 ≥2 时自动调整
        if self._auto_tune and abs(rec - self._max_concurrent) >= 2:
            self._resize_concurrency(rec)
            logger.info(
                "auto-tune: max_concurrent %d → %d (avg_tokens=%.0f, "
                "avg_duration=%.1fs, tpm=%d)",
                self._max_concurrent, rec,
                self._token_ema, self._duration_ema, self._tpm,
            )

    def recommend_max_concurrent(self) -> int:
        """
        基于 EMA 观测数据推荐安全的 max_concurrent。

        两个公式取较保守值 × 安全系数 0.8：
          - 稳态: TPM × avg_duration / (60 × avg_tokens)
          - 防爆: TPM / avg_tokens
        """
        if self._tpm <= 0 or self._token_ema <= 0:
            return self._max_concurrent

        steady = self._tpm * self._duration_ema / (60.0 * self._token_ema)
        burst_safe = self._tpm / self._token_ema
        return max(1, int(min(steady, burst_safe) * 0.8))

    @property
    def current_rpm(self) -> int:
        """当前 60s 窗口内的请求数（近似值）。"""
        cutoff = time.monotonic() - 60.0
        return sum(1 for t in self._request_times if t > cutoff)

    @property
    def current_tpm(self) -> int:
        """当前 60s 窗口内的 token 消耗量（近似值）。"""
        cutoff = time.monotonic() - 60.0
        return sum(n for ts, n in self._token_records if ts > cutoff)