"""TTS 观测链修复的回归：池层逐候选事件 + 内部 retried 保留 + truncated 救回 + 熔断跳过。

用假引擎（不碰网络/registry）构造 TTSPool，并打桩 obs.emit_explicit 捕获观测事件，验证：
1. 失败候选发 tts_candidate 事件（含 fail_reason / engine / attempts）。
2. 引擎内部 retried/attempts 不再被池层覆盖抹掉（WS-1.2）。
3. 成功路径 truncated 透传——「断流截断→重试救回」能计数（WS-1.3）。
4. 候选被熔断跳过也发 tts_candidate(fail_reason=circuit_open)（WS-1.4 数据源）。
"""

from types import SimpleNamespace

import pytest

from tts.base_tts import State
from tts.executor import TTSPool
from utils.cand_pool import build_candidates
from utils.health_store import HealthStore, CircuitState


class _Parent:
    def __init__(self):
        self.frames = 0

    def put_audio_frame(self, chunk, datainfo):  # noqa: ARG002
        self.frames += 1


class _CallingEngine:
    """复刻「合成过程中登记 last_tts」语义的假引擎。

    注意：TTSPool 在调用前会把 eng.last_tts 归零（真正停嘴/归零语义），
    因此预设值必须在 txt_to_audio 内部写入，才与真实引擎一致。
    """

    def __init__(self, last_tts):
        self.state = State.RUNNING
        self.last_tts = None
        self.calls = 0
        self._result = last_tts

    def txt_to_audio(self, msg):  # noqa: ARG002
        self.calls += 1
        self.last_tts = self._result


def _make_pool(cands, engines, min_audio_ms=300, monkeypatch=None, captured=None):
    parent = _Parent()
    pool = TTSPool(SimpleNamespace(fps=25, tts="doubao", REF_FILE="V"), parent)
    pool._health = HealthStore(failure_threshold=2, open_duration_sec=30)
    pool._candidates = build_candidates(cands)
    pool._min_audio_ms = min_audio_ms
    pool._engines = {c["id"]: e for c, e in zip(cands, engines)}
    if monkeypatch is not None:
        monkeypatch.setattr(
            "obs.emit_explicit",
            lambda ev, **kw: captured.append((ev, kw)),
        )
    return pool


_OBS = {"trace_id": "t1", "session_id": "s1", "parent_id": "p1", "enqueued_ms": 0}


def _cand(cid, engine, priority=1):
    return {"id": cid, "engine": engine, "priority": priority, "enabled": True, "params": {}}


# ── 1) 失败候选发 tts_candidate ─────────────────────────────────────────────
def test_failed_candidate_emits_tts_candidate(monkeypatch):
    captured = []
    cands = [_cand("failA", "failA", 1), _cand("okB", "okB", 2)]
    ok_lt = {"success": True, "fail_reason": None, "audio_ms": 500.0,
             "attempts": 1, "retried": False, "truncated": False}
    pool = _make_pool(
        cands,
        [_CallingEngine({**ok_lt, "success": False, "fail_reason": "session_failed",
                         "audio_ms": 0, "err_type": "RuntimeError"}),
         _CallingEngine(ok_lt)],
        monkeypatch=monkeypatch, captured=captured,
    )
    pool.txt_to_audio(("你好", {"_obs": _OBS}))

    assert pool.last_tts["success"] is True and pool.last_tts["provider"] == "okB"
    # 失败候选 A 发了一条 tts_candidate 诊断事件
    cand_events = [e for e, _ in captured if e["type"] == "tts_candidate"]
    assert len(cand_events) == 1
    ce = cand_events[0]
    assert ce["engine"] == "failA"
    assert ce["success"] is False
    assert ce["fail_reason"] == "session_failed"
    assert ce["err_type"] == "RuntimeError"
    assert ce["attempts"] == 1
    # trace 身份经 emit_explicit 的显式 kw 传递（跨线程挂回聊天 trace）
    _, kw = captured[0]
    assert kw["trace_id"] == "t1" and kw["session_id"] == "s1" and kw["parent_id"] == "p1"


# ── 2) 引擎内部 retried/attempts 保留（WS-1.2）───────────────────────────────
def test_pool_preserves_internal_retry(monkeypatch):
    captured = []
    # 单候选成功，但引擎内部已重试：attempts=2, retried=True
    lt = {"success": True, "fail_reason": None, "audio_ms": 800.0,
          "attempts": 2, "retried": True, "truncated": False}
    pool = _make_pool([_cand("a", "a")], [_CallingEngine(lt)],
                      monkeypatch=monkeypatch, captured=captured)
    pool.txt_to_audio(("嗯", {"_obs": _OBS}))

    assert pool.last_tts["success"] is True
    assert pool.last_tts["attempts"] == 2   # 保留引擎真实尝试次数
    assert pool.last_tts["retried"] is True  # 内部重试不被池层抹成 False
    assert pool._engines["a"].calls == 1


def test_internal_retry_exhausted_surfaces_retried(monkeypatch):
    # 引擎内部重试到底仍失败：池层全失败也应保留 retried=True、attempts=实际次数
    captured = []
    lt = {"success": False, "fail_reason": "session_failed", "audio_ms": 0,
          "attempts": 3, "retried": True, "truncated": False}
    pool = _make_pool([_cand("a", "a")], [_CallingEngine(lt)],
                      monkeypatch=monkeypatch, captured=captured)
    pool.txt_to_audio(("嗯", {"_obs": _OBS}))

    assert pool.last_tts["success"] is False
    assert pool.last_tts["fail_reason"] == "all_tts_candidates_failed"
    assert pool.last_tts["attempts"] == 3      # 引擎真实尝试次数透传
    assert pool.last_tts["retried"] is True    # 内部重试透传


# ── 3) 成功路径 truncated 透传（WS-1.3）─────────────────────────────────────
def test_truncated_rescue_counts(monkeypatch):
    # 引擎「断流截断→重试救回」成功：池层 winner 必须带上 truncated=True，不丢计数
    captured = []
    lt = {"success": True, "fail_reason": None, "audio_ms": 500.0,
          "attempts": 2, "retried": True, "truncated": True}
    pool = _make_pool([_cand("a", "a")], [_CallingEngine(lt)],
                      monkeypatch=monkeypatch, captured=captured)
    pool.txt_to_audio(("嗯", {"_obs": _OBS}))

    assert pool.last_tts["success"] is True
    assert pool.last_tts["truncated"] is True
    assert pool.last_tts["retried"] is True


# ── 4) 熔断跳过发 tts_candidate(circuit_open)（WS-1.4 数据源）──────────────
def test_circuit_skip_emits_candidate(monkeypatch):
    captured = []
    cands = [_cand("failA", "failA", 1), _cand("okB", "okB", 2)]
    fail_lt = {"success": False, "fail_reason": "session_failed", "audio_ms": 0,
               "err_type": None, "attempts": 1, "retried": False, "truncated": False}
    ok_lt = {"success": True, "fail_reason": None, "audio_ms": 500.0,
             "attempts": 1, "retried": False, "truncated": False}
    pool = _make_pool(
        cands,
        [_CallingEngine(fail_lt), _CallingEngine(ok_lt)],
        monkeypatch=monkeypatch, captured=captured,
    )

    # 句1、句2 连败 failA 两次 → failA 熔断
    pool.txt_to_audio(("第一句", {"_obs": _OBS}))
    pool.txt_to_audio(("第二句", {"_obs": _OBS}))
    assert pool._health.get_state("failA") == CircuitState.OPEN

    # 句3：failA 熔断跳过 → 只发一条 circuit_open 候选事件，成功走 okB
    captured.clear()
    pool.txt_to_audio(("第三句", {"_obs": _OBS}))
    circ = [e for e, _ in captured if e["type"] == "tts_candidate"
            and e["fail_reason"] == "circuit_open"]
    assert len(circ) == 1
    assert circ[0]["engine"] == "failA"
    assert pool.last_tts["provider"] == "okB"
    assert pool._engines["failA"].calls == 2  # 熔断后不再真实调用