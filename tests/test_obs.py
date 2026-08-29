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


if __name__ == "__main__":
    unittest.main(verbosity=2)