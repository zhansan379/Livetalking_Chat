"""TTS 路由降级 + 熔断：共享 utils 熔断器 + TTSPool 句内回退。

用假引擎（不碰网络/registry 真实模块）直接构造 TTSPool 并注入候选，验证：
1. utils/health_store 三态熔断（CLOSED→OPEN→HALF_OPEN→CLOSED）。
2. utils/config_load 的 ${ENV} 占位解析。
3. TTSPool 句内回退：A 失败 B 成功 → provider=B、attempts=2、retried=True。
4. 熔断跳过：A 连续失败到熔断后 → 只走 B（attempts=1）。
5. 全候选失败 → fail_reason == all_tts_candidates_failed。
6. min_audio_ms 截断判定：真成功但音频过短也触发回退。
"""

from types import SimpleNamespace

import pytest

from tts.base_tts import State
from tts.executor import TTSPool
from utils.cand_pool import Candidate, build_candidates
from utils.config_load import resolve_env
from utils.health_store import CandidateHealth, CircuitState, HealthStore


# ── 假引擎：不走真实注册/网络，只复刻 txt_to_audio 的成败登记语义 ───────────────
class FakeEngine:
    def __init__(self, success: bool = True, audio_ms: float = 500.0,
                 fail_reason: str = "session_failed"):
        self.state = State.RUNNING
        self.last_tts = None
        self.calls = 0
        self._success = success
        self._audio_ms = audio_ms
        self._fail_reason = fail_reason

    def txt_to_audio(self, msg):  # noqa: ARG002 - 与 BaseTTS txt_to_audio 签名对齐
        self.calls += 1
        if self._success:
            self.last_tts = {
                "success": True, "fail_reason": None, "audio_ms": self._audio_ms,
                "attempts": 1, "retried": False, "truncated": False,
            }
        else:
            self.last_tts = {
                "success": False, "fail_reason": self._fail_reason, "audio_ms": 0,
                "attempts": 1, "retried": False, "truncated": False,
            }


class FakeParent:
    """记录 put_audio_frame 的最小父对象（avatar 播放管线入口）。"""

    def __init__(self):
        self.frames = []

    def put_audio_frame(self, chunk, datainfo):  # noqa: ARG002
        self.frames.append(datainfo.copy())


def _make_pool(cands: list[dict], failed_first=True, ok_second=True,
               min_audio_ms=300, make_second=None):
    parent = FakeParent()
    pool = TTSPool(SimpleNamespace(fps=25, tts="doubao", REF_FILE="V"), parent)
    pool._health = HealthStore(failure_threshold=2, open_duration_sec=30)
    pool._candidates = build_candidates(cands)
    pool._min_audio_ms = min_audio_ms

    ok_conf = dict(success=ok_second) if make_second is None else make_second
    pool._engines = {
        cands[0]["id"]: FakeEngine(success=not failed_first),
        cands[1]["id"]: FakeEngine(**ok_conf),
    }
    return pool


# ── 1) utils/health_store 三态熔断 ──────────────────────────────────────────
def test_health_store_trip_and_recover():
    hs = HealthStore(failure_threshold=2, open_duration_sec=5)
    cid = "engineA"

    assert hs.allow_call(cid) is True and hs.get_state(cid) == CircuitState.CLOSED
    hs.mark_failure(cid)
    assert hs.get_state(cid) == CircuitState.CLOSED  # 未达阈值仍 CLOSED
    hs.mark_failure(cid)
    assert hs.get_state(cid) == CircuitState.OPEN  # 连续失败达阈值 → OPEN
    assert hs.allow_call(cid) is False  # 熔断中拒调

    import time
    hs._health[cid].open_until = time.monotonic() - 1  # 冷却到期 → 放行
    assert hs.allow_call(cid) is True
    hs.mark_success(cid)  # 这一句成功 → 复位
    assert hs.get_state(cid) == CircuitState.CLOSED
    assert hs.get_stats()[cid]["failures"] == 0


def test_health_store_fail_after_cease_ln2_reopens():
    hs = HealthStore(failure_threshold=2, open_duration_sec=30)
    cid = "a"
    hs.mark_failure(cid)
    hs.mark_failure(cid)
    assert hs.get_state(cid) == CircuitState.OPEN
    import time
    hs._health[cid].open_until = time.monotonic() - 1  # 冷却到期
    assert hs.allow_call(cid) is True  # 放行（探活语义）
    hs.mark_failure(cid)  # 探活这句也失败 → 重新熔断
    assert hs.get_state(cid) == CircuitState.OPEN


def test_health_store_state_enum():
    # 单测 CandidateHealth.state 的枚举语义（Open/HalfOpen 判定）
    import time
    now = time.monotonic()
    h = CandidateHealth()
    assert h.state == CircuitState.CLOSED
    h.open_until = now + 10  # 未到期
    assert h.state == CircuitState.OPEN  # 未到期 → OPEN（无论是否探活在途）
    h.open_until = now - 1  # 已到期但 open_until 仍 >0
    h.half_open_inflight = True
    assert h.state == CircuitState.HALF_OPEN  # 到期+探活在途 → HALF_OPEN
    h.half_open_inflight = False
    assert h.state == CircuitState.CLOSED  # 到期+非探活 → CLOSED


# ── 2) utils/config_load ${ENV} 占位解析 ───────────────────────────────────
def test_resolve_env(monkeypatch):
    monkeypatch.setenv("TTS_TEST_A", "hello")
    monkeypatch.delenv("TTS_TEST_MISSING", raising=False)
    assert resolve_env("${TTS_TEST_A}") == "hello"
    assert resolve_env("${TTS_TEST_MISSING}") == ""
    assert resolve_env("${TTS_TEST_MISSING:-fallback}") == "fallback"
    assert resolve_env({"a": ["${TTS_TEST_A}", "${X:-d}"], "b": 1}) == {
        "a": ["hello", "d"], "b": 1}
    assert resolve_env("plain") == "plain"
    assert resolve_env({"list": ["${Y:-default}"]}) == {"list": ["default"]}


# ── 3) TTSPool 句内回退 ────────────────────────────────────────────────────
def test_failover_to_second():
    pool = _make_pool([
        {"id": "failA", "engine": "failA", "priority": 1, "enabled": True, "params": {}},
        {"id": "okB", "engine": "okB", "priority": 2, "enabled": True, "params": {}},
    ], failed_first=True, ok_second=True)

    pool.txt_to_audio(("你好", {}))

    assert pool.last_tts["success"] is True
    assert pool.last_tts["provider"] == "okB"      # 胜出 = 兜底引擎
    assert pool.last_tts["attempts"] == 2
    assert pool.last_tts["retried"] is True
    assert pool._engines["failA"].calls == 1
    assert pool._engines["okB"].calls == 1


def test_all_candidates_failed():
    pool = _make_pool([
        {"id": "failA", "engine": "failA", "priority": 1, "enabled": True, "params": {}},
        {"id": "failB", "engine": "failB", "priority": 2, "enabled": True, "params": {}},
    ], failed_first=True, ok_second=False)

    pool.txt_to_audio(("你好", {}))

    assert pool.last_tts["success"] is False
    assert pool.last_tts["fail_reason"] == "all_tts_candidates_failed"
    assert pool.last_tts["attempts"] == 2


def test_min_audio_truncation_triggers_fallback():
    # 真成功但音频过短(<min_audio_ms) → 视为截断/失败，触发回退
    pool = _make_pool([
        {"id": "shortA", "engine": "shortA", "priority": 1, "enabled": True, "params": {}},
        {"id": "okB", "engine": "okB", "priority": 2, "enabled": True, "params": {}},
    ], failed_first=False, ok_second=True, min_audio_ms=300,
        make_second=dict(success=True, audio_ms=500))
    pool._engines["shortA"]._success = True
    pool._engines["shortA"]._audio_ms = 200  # < 300

    pool.txt_to_audio(("嗯", {}))

    assert pool.last_tts["provider"] == "okB"
    assert pool.last_tts["retried"] is True


# ── 4) 熔断跳过 ────────────────────────────────────────────────────────────
def test_circuit_open_skips_primary():
    pool = _make_pool([
        {"id": "failA", "engine": "failA", "priority": 1, "enabled": True, "params": {}},
        {"id": "okB", "engine": "okB", "priority": 2, "enabled": True, "params": {}},
    ])
    fail_eng = pool._engines["failA"]

    # 句1、句2 都由 failA 起手但失败（回退 okB）→ 连败 2 次，failA 熔断
    pool.txt_to_audio(("第一句", {}))
    pool.txt_to_audio(("第二句", {}))
    assert pool._health.get_state("failA") == CircuitState.OPEN
    assert fail_eng.calls == 2

    # 句3：failA 熔断中跳过 → 直接 okB，attempts=1、retried=False
    pool.txt_to_audio(("第三句", {}))
    assert fail_eng.calls == 2  # 熔断后不再调
    assert pool.last_tts["provider"] == "okB"
    assert pool.last_tts["attempts"] == 1
    assert pool.last_tts["retried"] is False


# ── 5) 编排：候选池排序 ─ ───────────────────────────────────────────────────
def test_build_candidates_sorted_by_priority():
    rows = [
        {"id": "c", "engine": "e3", "priority": 3, "params": {}},
        {"id": "a", "engine": "e1", "priority": 1, "params": {}},
        {"id": "bad"},  # 缺 engine → 忽略
        {"id": "b", "engine": "e2", "priority": 2, "params": {}},
    ]
    cands = build_candidates(rows)
    assert [c.id for c in cands] == ["a", "b", "c"]
    assert all(isinstance(c, Candidate) for c in cands)


# ── 6) 端到端：x 失败者仍在候选池中但熔断后不影响最终出声（组合场景）────── 已覆盖于 3/4