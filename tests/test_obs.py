###############################################################################
#  obs 观测平台单元测试（仅用 stdlib，可 python -m pytest tests/ 或直接 python 运行）
#
#  覆盖：事件写盘 / trace 嵌套 / 聚合统计(success_rate·per_model·tool 计数) /
#       summary 与 chat 分离 / requests 列表 / 只读写入不抛异常 /开关切换。
###############################################################################

import asyncio
import os
import tempfile
import unittest


class ObsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="obs_test_")
        os.environ["OBS_ENABLED"] = "1"
        os.environ["OBS_DIR"] = self._tmp

    def tearDown(self):
        for k in ("OBS_ENABLED", "OBS_DIR", "OBS_QUERY_WINDOW", "OBS_QUERY_LIMIT"):
            os.environ.pop(k, None)

    # ── 工具函数：打一条 chat trace ────────────────────────────────────────
    def _write_chat_trace(self, session_id="s1", msg="北京天气如何?",
                          tool_mode=True, success=True, fail_reason=None):
        from obs import begin_trace, emit, end_trace, round_span
        tid = begin_trace(session_id, msg, tool_mode=tool_mode)
        async def _body():
            await asyncio.sleep(0.02)  # 模拟真实 LLM 延迟，保证响应耗时 > 0
            async with round_span(0) as rd:
                emit({"type": "llm_call", "mode": "nonstream", "model": "qwen",
                      "route": "bailian", "has_tools": True, "attempts": 1,
                      "elapsed_ms": 500.0, "input_tokens": 10, "output_tokens": 30,
                      "total_tokens": 40, "success": True, "fail_reason": None,
                      "err_type": None})
                rd.n_tool_calls = 1
                emit({"type": "tool_call", "round": 0, "tool": "weather",
                      "args": '{"city": "北京"}', "result_snippet": "北京 24℃",
                      "elapsed_ms": 100.0, "success": True, "error": None})
            emit({"type": "llm_call", "mode": "nonstream", "model": "qwen",
                  "route": "bailian", "has_tools": False, "attempts": 1,
                  "elapsed_ms": 300.0, "input_tokens": 5, "output_tokens": 20,
                  "total_tokens": 25, "success": True, "fail_reason": None,
                  "err_type": None})
            end_trace(success=success, fail_reason=fail_reason, text_len=100)
        asyncio.run(_body())
        return tid

    def _read_lines(self):
        path = os.path.join(self._tmp, "events.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        return [l for l in text.splitlines() if l]

    # ── 用例 ──────────────────────────────────────────────────────────────
    def test_trace_events_and_nesting(self):
        from obs import query
        tid = self._write_chat_trace()
        lines = self._read_lines()
        # trace_start / llm_call x2 / tool_round / tool_call / trace_end = 6
        self.assertEqual(len(lines), 6, "chat trace 应恰好产生 6 条事件")

        types = [__import__("json").loads(l)["type"] for l in lines]
        # 顺序与嵌套：第一轮 LLM 与 tool_call 父都应是 round span；第二个 LLM 父是 trace
        ev = [__import__("json").loads(l) for l in lines]
        by_type = {}
        for e in ev:
            by_type.setdefault(e["type"], []).append(e)
        llm1, llm2 = by_type["llm_call"]
        tool_round = by_type["tool_round"][0]
        tool_call = by_type["tool_call"][0]
        start = by_type["trace_start"][0]
        end = by_type["trace_end"][0]

        self.assertEqual(start["parent_id"], None)
        self.assertEqual(start["span_id"], tid)
        # 两个 LLM 调用、一个工具调用，都嵌套在 round span 下；round span 父是 trace
        round_span_id = tool_round["span_id"]
        self.assertEqual(llm1["parent_id"], round_span_id)
        self.assertEqual(tool_call["parent_id"], round_span_id)
        self.assertEqual(tool_round["parent_id"], tid)
        self.assertEqual(llm2["parent_id"], tid)  # 末轮直接给答案，父回到 trace
        self.assertEqual(end["span_id"], tid)
        # 响应耗时应在 trace 维度上自洽（>0）
        self.assertGreater(end["elapsed_ms"], 0)

        # request(trace_id) 按序返回全部事件
        evs = query.request(tid)
        self.assertEqual(len(evs), 6)
        self.assertEqual([e["type"] for e in evs], types)

    def test_summary_aggregation_and_chat_separate(self):
        from obs import query
        self._write_chat_trace()
        self._write_chat_trace(session_id="s2", msg="上海天气?", tool_mode=True)

        # 一条 summary 压缩 trace，不应污染 chat 统计
        from obs import emit, new_trace
        with new_trace("s1", kind="summary"):
            emit({"type": "llm_call", "mode": "nonstream", "model": "qwen",
                  "route": "bailian", "has_tools": False, "attempts": 1,
                  "elapsed_ms": 200.0, "input_tokens": 50, "output_tokens": 10,
                  "total_tokens": 60, "success": True, "fail_reason": None,
                  "err_type": None})

        s = query.summary(window=None)
        self.assertEqual(s["traces"], 2)              # 仅 2 条 chat trace
        self.assertEqual(s["success"], 2)
        self.assertEqual(s["success_rate"], 1.0)
        self.assertEqual(s["total_llm_calls"], 4)     # 每条 chat 2 次调用，不含 summary
        self.assertEqual(s["total_tool_calls"], 2)
        self.assertEqual(s["tool_rounds"], 2)
        self.assertEqual(s["tool_call_counts"], {"weather": 2})
        self.assertEqual(s["total_tokens"]["total"], (40 + 25) * 2)  # 不含 summary 的 60
        self.assertEqual(len(s["per_model"]), 1)
        self.assertEqual(s["per_model"][0]["calls"], 4)

        # requests 列表只含 chat，不含 summary
        rows = query.requests(limit=10)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertNotIn("summary", (r.get("session_id") or ""))

    def test_mixed_success_via_new_trace_is_isolated(self):
        # 即使 summary 的 status 变更也不影响 chat 成功率
        from obs import query
        self._write_chat_trace()
        s = query.summary(window=None)
        self.assertEqual(s["success"], 1)

    def test_is_enabled_toggle(self):
        from obs.config import is_enabled
        self.assertTrue(is_enabled())
        os.environ["OBS_ENABLED"] = "0"
        self.assertFalse(is_enabled())
        os.environ["OBS_ENABLED"] = "false"
        self.assertFalse(is_enabled())
        os.environ["OBS_ENABLED"] = "1"
        self.assertTrue(is_enabled())

    def test_write_failure_is_silent(self):
        # OBS_DIR 指向一个普通文件 → 打开子路径失败（OSError），不应抛异常、不污染数据
        import json
        from obs import emit, begin_trace, end_trace
        blob = os.path.join(self._tmp, "blocker.txt")
        with open(blob, "w", encoding="utf-8") as f:
            f.write("x")
        os.environ["OBS_DIR"] = blob
        begin_trace("bad", "hi")
        emit({"type": "llm_call", "success": True})
        end_trace(True)  # 不抛即可
        # 真实目录仍可写
        os.environ["OBS_DIR"] = self._tmp
        begin_trace("ok", "hi2")
        emit({"type": "llm_call", "success": True})
        end_trace(True)
        lines = self._read_lines()
        self.assertEqual(len(lines), 3)  # 仅 ok trace 的 3 条（bad 写失败被静默跳过）

    # ── 全链路：ASR / TTS ────────────────────────────────────────────────
    def _write_asr_trace(self, session_id="s-asr", text="你好广州",
                         inference_ms=200.0, success=True):
        """独立 kind="asr" 的 trace（模拟 server/asr_server.py 的埋点）。"""
        from obs import begin_trace, emit, end_trace
        tid = begin_trace(session_id, "", tool_mode=None, kind="asr")
        emit({"type": "asr_call", "span_id": tid, "parent_id": tid,
              "audio_ms": 1500.0, "audio_len_s": 1.5,
              "inference_ms": inference_ms, "elapsed_ms": inference_ms + 30.0,
              "rtf": inference_ms / 1000.0 / 1.5,
              "text": (text or "")[:40], "text_len": len(text or ""),
              "empty": not success, "success": success,
              "fail_reason": None if success else "inference_exception",
              "err_type": None if success else "RuntimeError"})
        end_trace(success=success, text_len=len(text or ""))
        return tid

    def test_asr_trace_isolated_and_aggregated(self):
        from obs import query
        self._write_chat_trace()            # 一条 chat
        self._write_asr_trace()             # 一条独立 asr
        self._write_asr_trace(text="失败", inference_ms=0.0, success=False)

        s = query.summary(window=None)
        # asr trace 不进聊天统计
        self.assertEqual(s["traces"], 1)
        self.assertEqual(s["success"], 1)
        # 但进入 asr 聚合
        self.assertEqual(s["asr"]["calls"], 2)
        # 新口径：空转写段(empty)单独计数，成功率按非空样本算 = 1/(2-1)
        self.assertEqual(s["asr"]["success_rate"], 1.0)
        self.assertEqual(s["asr"]["empty"], 1)
        self.assertEqual(s["asr"]["avg_ms"], 100.0)   # (200+0)/2
        self.assertEqual(s["asr"]["total_audio_ms"], 3000.0)
        self.assertGreater(s["asr"]["avg_rtf"], 0.0)
        # requests 列表不含 asr trace
        rows = query.requests(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertNotIn("s-asr", [r.get("session_id") for r in rows])

    def test_tts_call_emit_explicit_nests_and_counts(self):
        import threading
        from obs import query, emit_explicit
        tid = self._write_chat_trace()

        # 模拟 base_tts 的 TTS 工作线程：contextvars 拿不到，改用 emit_explicit 显式挂回父 trace
        def _tts_worker():
            emit_explicit({
                "type": "tts_call", "provider": "edgetts",
                "text": "今天天气不错", "text_len": 6, "attempts": 1,
                "elapsed_ms": 120.0, "queue_ms": 8.0, "audio_ms": 2600.0,
                "success": True, "fail_reason": None, "err_type": None,
                "retried": False, "truncated": False,
            }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")
        th = threading.Thread(target=_tts_worker)
        th.start()
        th.join()  # 证明：即便从独立线程写，seq 仍单调、事件正常落盘

        evs = query.request(tid)
        tts = [e for e in evs if e["type"] == "tts_call"]
        self.assertEqual(len(tts), 1)
        self.assertEqual(tts[0]["parent_id"], tid)   # 挂回聊天 trace 下
        self.assertEqual(tts[0]["trace_id"], tid)
        self.assertEqual(tts[0]["provider"], "edgetts")

        s = query.summary(window=None)
        self.assertEqual(s["tts"]["calls"], 1)
        self.assertEqual(s["tts"]["success_rate"], 1.0)
        self.assertEqual(s["tts"]["avg_ms"], 120.0)
        # 层1：audio_ms 归位——合成音频毫秒进聚合
        self.assertEqual(s["tts"]["audio_ms"], 2600.0)
        self.assertEqual(s["tts"]["avg_audio_ms"], 2600.0)
        # tts_call 是 chat 子事件：不新增 trace_end，聊天 trace 数不变
        self.assertEqual(s["traces"], 1)

    def test_tts_end_serve_rate_partial(self):
        # 层2：合成了但端标记没送抵 → success_rate 仍 100%，end_serve_rate 跌到 50%。
        # 这正是「合成全成功、观众只听到一半」时 OBS 能看到的判别信号。
        from obs import query, emit_explicit
        tid = self._write_chat_trace()

        for i, delivered in enumerate((True, False)):
            emit_explicit({
                "type": "tts_call", "provider": "edgetts",
                "text": f"句子{i}", "text_len": 3, "attempts": 1,
                "elapsed_ms": 100.0, "queue_ms": 5.0, "audio_ms": 2000.0,
                "success": True, "fail_reason": None, "err_type": None,
                "retried": False, "truncated": False,
            }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")
            if delivered:
                emit_explicit({
                    "type": "tts_playback", "text": f"句子{i}", "text_len": 3,
                    "status": "end",
                }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")

        s = query.summary(window=None)
        self.assertEqual(s["tts"]["calls"], 2)
        self.assertEqual(s["tts"]["success_rate"], 1.0)   # 合成全成功
        self.assertEqual(s["tts"]["ends_served"], 1)      # 但只有一句送抵
        self.assertEqual(s["tts"]["end_serve_rate"], 0.5)

    def test_tts_playback_is_chat_subevent(self):
        # tts_playback 不新增聊天 trace，也不影响聊天统计
        from obs import query, emit_explicit
        tid = self._write_chat_trace()
        emit_explicit({
            "type": "tts_call", "provider": "edgetts", "text": "句子",
            "text_len": 2, "attempts": 1, "elapsed_ms": 100.0, "queue_ms": 5.0,
            "audio_ms": 2000.0, "success": True, "fail_reason": None,
            "err_type": None, "retried": False, "truncated": False,
        }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")
        emit_explicit({
            "type": "tts_playback", "text": "句子", "text_len": 2, "status": "end",
        }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")

        s = query.summary(window=None)
        self.assertEqual(s["traces"], 1)
        self.assertEqual(s["tts"]["ends_served"], 1)
        self.assertEqual(s["tts"]["end_serve_rate"], 1.0)  # 1 句合成 → 1 句送达

    def test_doubao_audio_ms_accumulated(self):
        # 层1源码侧：_consume_pcm 累加实际送入播放的样本 → _audio_ms 正确
        import numpy as np
        from types import SimpleNamespace
        from tts.doubao import DoubaoTTS
        from tts.base_tts import State

        class _Parent:
            def __init__(self):
                self.frames = 0
            def put_audio_frame(self, chunk, ep):
                self.frames += 1

        parent = _Parent()
        opt = SimpleNamespace(fps=25, REF_FILE="v", tts="doubao")
        t = DoubaoTTS(opt, parent)
        t.state = State.RUNNING

        # 5 个 uint16 chunk = 5×20ms = 100ms；payload 用非零样本（真实音频）
        data = (np.zeros(t.chunk * 5, dtype=np.int16) + 1000).tobytes()
        t._consume_pcm(data, "文本", {})
        self.assertEqual(parent.frames, 5)
        self.assertAlmostEqual(t._audio_ms, 100.0, delta=0.01)

        # 再喂不足一 chunk 的残余：不该累计进时长、也不该按整帧发
        t._consume_pcm((np.zeros(t.chunk // 2, dtype=np.int16) + 1000).tobytes(),
                       "文本", {})
        self.assertEqual(parent.frames, 5)             # 无新增整帧
        self.assertAlmostEqual(t._audio_ms, 100.0, delta=0.01)

    def test_tts_attempts_strength_and_circuit_aggregation(self):
        # 强度聚合：每句真实合成尝试次数 avg/max（含引擎内部重试 + 跨候选回退），
        # 以及候选被熔断跳过（circuit_open）的次数，都进 summary["tts"]。
        from obs import query, emit_explicit
        tid = self._write_chat_trace()

        # 两句：一句 attempt=2（内部重试），另一句 attempt=1
        for att, retried in ((2, True), (1, False)):
            emit_explicit({
                "type": "tts_call", "provider": "edgetts",
                "text": "句子", "text_len": 2, "attempts": att,
                "elapsed_ms": 100.0, "queue_ms": 5.0, "audio_ms": 2000.0,
                "success": not retried, "fail_reason": None if not retried else "first_failed",
                "err_type": None, "retried": retried, "truncated": False,
            }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")

        # 熔断跳过：候选诊断事件（fail_reason=circuit_open）
        emit_explicit({
            "type": "tts_candidate", "engine": "edgetts", "success": False,
            "fail_reason": "circuit_open", "err_type": None, "attempts": 0,
            "retried": False, "audio_ms": 0.0, "elapsed_ms": 0.0,
        }, trace_id=tid, session_id="s1", parent_id=tid, kind="chat")

        s = query.summary(window=None)
        tts = s["tts"]
        self.assertEqual(tts["avg_attempts"], 1.5)   # (2+1)/2
        self.assertEqual(tts["max_attempts"], 2)
        self.assertEqual(tts["circuit_skip"], 1)
        # 真实失败但仍计入调用次数（attempt=2 那句 success=False）
        self.assertEqual(tts["calls"], 2)

    def test_merged_single_trace_asr_llm_tts(self):
        # 全链路合并成一条 trace：ASR(asr_call) → chat(trace_start/llm_call/tts_call/trace_end)。
        # 模拟新流程：ASR 服务端生成回合 id → emit_explicit asr_call → 浏览器 echo →
        # begin_trace(trace_id=同 id)。请求列表只出现一次、both 聚合正确。
        from obs import query, emit_explicit, begin_trace, end_trace, emit
        tid = "shared_turn_abc123"

        emit_explicit({
            "type": "asr_call",
            "audio_ms": 1500.0, "audio_len_s": 1.5, "inference_ms": 200.0,
            "elapsed_ms": 230.0, "rtf": 0.1333,
            "text": "北京天气", "text_len": 4, "empty": False,
            "success": True, "fail_reason": None, "err_type": None,
        }, trace_id=tid, session_id="s-m", parent_id=tid, kind="asr")

        # chat 段复用同一 id（浏览器 echo）
        bt = begin_trace("s-m", "北京天气", tool_mode=False, trace_id=tid)
        self.assertEqual(bt, tid)
        emit({"type": "llm_call", "mode": "stream", "model": "qwen", "route": "bailian",
              "has_tools": False, "attempts": 1, "elapsed_ms": 300.0,
              "input_tokens": 5, "output_tokens": 20, "total_tokens": 25,
              "success": True, "fail_reason": None, "err_type": None})
        emit_explicit({
            "type": "tts_call", "provider": "edgetts", "text": "今天天气不错",
            "text_len": 6, "attempts": 1, "elapsed_ms": 120.0, "queue_ms": 8.0,
            "audio_ms": 2600.0, "success": True, "fail_reason": None,
            "err_type": None, "retried": False, "truncated": False,
        }, trace_id=tid, session_id="s-m", parent_id=tid, kind="chat")
        end_trace(success=True, text_len=4)

        # 同一条 trace 内按 seq 见全部事件（asr_call 在最前）
        evs = query.request(tid)
        types = [e["type"] for e in evs]
        self.assertEqual(types, ["asr_call", "trace_start", "llm_call", "tts_call", "trace_end"])
        self.assertEqual(evs[0]["kind"], "asr")
        self.assertEqual(evs[0]["parent_id"], tid)

        # 请求列表只出现一次，正确计为一条 chat trace；
        # pipeline_ms（全链路）应 > 聊天段 elapsed_ms（因为跨到 trace 之前的 ASR）
        rows = query.requests(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["trace_id"], tid)
        self.assertIsNotNone(rows[0]["pipeline_ms"])
        self.assertGreater(rows[0]["pipeline_ms"], rows[0]["elapsed_ms"])

        # 聊天统计与 asr/tts 聚合各自正确；全链路耗时聚合有值
        s = query.summary(window=None)
        self.assertEqual(s["traces"], 1)
        self.assertEqual(s["success"], 1)
        self.assertEqual(s["total_llm_calls"], 1)
        self.assertEqual(s["asr"]["calls"], 1)
        self.assertEqual(s["asr"]["success_rate"], 1.0)
        self.assertEqual(s["tts"]["calls"], 1)
        self.assertGreater(s["pipeline"]["avg"], 0.0)
        self.assertGreater(s["pipeline"]["avg"], s["response_time"]["avg"])

    def test_pipeline_grouping(self):
        from obs import query
        self._write_chat_trace(session_id="p1")   # chat trace（含 TTS 朋友事件不在此）
        self._write_asr_trace(session_id="p1")    # asr trace，同一会话

        groups = query.pipeline("p1", limit=10)
        kinds = sorted(g["kind"] for g in groups)
        self.assertEqual(kinds, ["asr", "chat"])
        for g in groups:
            self.assertEqual(g["session_id"], "p1")
            self.assertTrue(g["events"])           # 每组的明细事件
            self.assertTrue(g["trace_id"])
        # asr 组在 requests() 里不可见，但在 pipeline 里出现
        rows = query.requests(limit=10)
        self.assertEqual(len(rows), 1)
        self.assertTrue(any(g["kind"] == "asr" for g in groups))


if __name__ == "__main__":
    unittest.main(verbosity=2)