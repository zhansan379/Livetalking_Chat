"""DoubaoTTS 会话级重建重试：session_failed / no_audio → 重建 WS 对本句再合成。

用打桩的 _run_session（不碰真实火山 wss）验证：
1. 会话异常后重试成功 → attempts=2, retried=True, success=True。
2. 重试也挂 → attempts=2, retried=True, fail=session_failed。
3. doubao_retry=0（关闭）→ 首次失败即放弃，attempts=1, retried=False（兼容旧行为）。
4. 首次即成功 → attempts=1, retried=False。
5. _send_end（end 标记）任何路径都只发一次。
"""

import asyncio
import os
import types
from types import SimpleNamespace

import pytest

from tts.doubao import DoubaoTTS

BASE_OPT = dict(
    REF_FILE="v",
    fps=25,
    doubao_resource_id=None,
    doubao_audio_format="pcm",
    doubao_sample_rate=16000,
    doubao_tone=None,
)


class _Parent:
    """最小 fake parent：只收集 put_audio_frame 的 eventpoint status。"""

    def __init__(self):
        self.statuses: list[str] = []

    def put_audio_frame(self, data, eventpoint):
        self.statuses.append(eventpoint.get("status"))
        if data is not None:
            self.statuses.append(None)


def _make(opt, behavior):
    p = _Parent()
    t = DoubaoTTS(opt, p)

    runs = {"n": 0}

    async def behavior_session(self):
        runs["n"] += 1
        n = runs["n"]
        if behavior == "always_fail" or (behavior == "fail1_then_ok" and n == 1):
            raise RuntimeError("session boom")
        self._got_audio = True
        self._audio_ms = 1000.0

    async def stub_run(self, text, req, rid, msg):
        await behavior_session(self)

    t._run_session = types.MethodType(stub_run, t)
    return t, p, runs


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")


def test_retry_recovers_after_session_failure():
    t, p, runs = _make(SimpleNamespace(**BASE_OPT, doubao_retry=1), "fail1_then_ok")
    t.txt_to_audio(("你好测试", {"tts": {}}))
    lt = t.last_tts
    assert runs["n"] == 2
    assert lt["success"] is True and lt["attempts"] == 2 and lt["retried"] is True
    assert p.statuses.count("end") == 1  # 失败重试不重复发结束标记


def test_retry_exhausted_surfaces_session_failed():
    t, p, runs = _make(SimpleNamespace(**BASE_OPT, doubao_retry=1), "always_fail")
    t.txt_to_audio(("你好测试", {"tts": {}}))
    lt = t.last_tts
    assert runs["n"] == 2
    assert lt["success"] is False and lt["fail_reason"] == "session_failed"
    assert lt["attempts"] == 2 and lt["retried"] is True
    assert p.statuses.count("end") == 1


def test_retry_zero_preserves_legacy_behavior():
    t, p, runs = _make(SimpleNamespace(**BASE_OPT, doubao_retry=0), "always_fail")
    t.txt_to_audio(("你好测试", {"tts": {}}))
    lt = t.last_tts
    assert runs["n"] == 1
    assert lt["success"] is False and lt["fail_reason"] == "session_failed"
    assert lt["attempts"] == 1 and lt["retried"] is False
    assert p.statuses.count("end") == 1


def test_first_attempt_success_no_retry():
    t, p, runs = _make(SimpleNamespace(**BASE_OPT, doubao_retry=1), "ok_first")
    t.txt_to_audio(("你好测试", {"tts": {}}))
    lt = t.last_tts
    assert runs["n"] == 1
    assert lt["success"] is True and lt["attempts"] == 1 and lt["retried"] is False
    assert p.statuses.count("end") == 1