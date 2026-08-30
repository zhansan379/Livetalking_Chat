###############################################################################
#  模拟面试状态机测试：fake 评分/eval 注入下走通分环节流转。
#
#  不联网：monkeypatch recall/eval 的 LLM 环节为固定返回，验证 handler 的
#  确定性流转（离散段逐题推进自动换段、对话段整段评分、skip/end/status）与落盘。
###############################################################################

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import tempfile
import unittest

import capabilities.interview.tools as tools_mod


class _Ctx:
    def __init__(self, session_id):
        self.session_id = session_id


class InterviewStateTest(unittest.TestCase):
    Q = [
        {"id": "q1", "text": "解释闭包", "category": "JS", "type": "technical",
         "rubrics": {}, "followups": []},
        {"id": "q2", "text": "讲讲事件循环", "category": "JS", "type": "technical",
         "rubrics": {}, "followups": []},
    ]

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="interview_")
        from agent.config import get_agent_config
        cfg = get_agent_config()
        # 只留一个离散 trivia 段，两题走完即终局，便于断言
        cfg.capabilities["interview"] = {
            "enabled": True, "store_dir": self._tmp,
            "default_role": "前端", "default_level": "初级",
            "sections": [{"type": "trivia", "name": "八股文", "count": 2}],
            "max_questions": 5,
        }
        self.cfg = cfg
        self.sid = "sid-test"

        # 注入 fake，避免联网
        orig_sheet = tools_mod.build_section
        orig_score = tools_mod.score_answer
        orig_sec = tools_mod.score_section
        orig_report = tools_mod.build_report

        async def _fake_section(cfg, section, role, level, resume_text, jd_text):
            return [dict(q) for q in self.Q]

        async def _fake_score(cfg, question, answer):
            return {"question_id": question.get("id"), "score": 8.0,
                    "dimension_notes": {"理解": 8, "表达": 8, "逻辑": 8, "完整": 8},
                    "comment": "不错"}

        async def _fake_sec_score(cfg, section, transcript):
            return {"question_id": section.get("type"), "score": 7.0,
                    "dimension_notes": {"提问质量": 7}, "comment": "环节不错"}

        async def _fake_report(cfg, sections, answers, role, level, jd_text):
            return {"summary": "整体表现良好",
                    "dimension_avg": {"理解": 8, "表达": 8, "逻辑": 8, "完整": 8},
                    "sections": [{"type": "trivia", "name": "八股文", "score": 8.0, "comment": "好"}],
                    "strengths": ["思路清晰"], "improvements": [], "suggested_topics": ["性能优化"]}

        tools_mod.build_section = _fake_section
        tools_mod.score_answer = _fake_score
        tools_mod.score_section = _fake_sec_score
        tools_mod.build_report = _fake_report
        self._orig = (orig_sheet, orig_score, orig_sec, orig_report)

    def tearDown(self):
        (tools_mod.build_section, tools_mod.score_answer,
         tools_mod.score_section, tools_mod.build_report) = self._orig

    def _start(self):
        return asyncio.run(tools_mod._start({}, self.cfg, ctx=_Ctx(self.sid)))

    def _answer(self, answer):
        return asyncio.run(tools_mod._answer({"answer": answer}, self.cfg, ctx=_Ctx(self.sid)))

    def _state(self):
        from capabilities.interview.state import InterviewState
        st = InterviewState(self.cfg, self.sid)
        st.load()
        return st

    # ── 离散段：逐题推进自动换段 / 满题终局 ────────────────────────────
    def test_full_run_finalizes_report(self):
        from capabilities.interview.state import InterviewState
        self.assertIn("模拟面试", self._start())
        # 答第1题 → 进第2题（同段内）
        out2 = self._answer("闭包能捕获外部变量")
        self.assertIn("第2题", out2)
        # 答第2题 → 段内走完 → 终局报告（只有一段，直接 finished）
        out_report = self._answer("事件循环是……")
        self.assertIn("模拟面试结束", out_report)
        st = self._state()
        self.assertEqual(st.status, "finished")
        self.assertIsNotNone(st.get("report"))
        self.assertEqual(len(st.get("answers") or []), 2)

    # ── 对话段：answer 只记 transcript 不评分，next_section 整段判分+推进 ──
    def test_dialogue_section_records_then_next_section_scores(self):
        # 覆盖配置为一段对话式反问，方便只测对话流
        self.cfg.capabilities["interview"]["sections"] = [
            {"type": "reverse_qa", "name": "反问"}]
        out = self._start()
        self.assertIn("反问", out)
        # 对话段 answer 不评分、只记 transcript
        ack = self._answer("你们这项业务怎么商业化？")
        self.assertIn("已记录", ack)
        st = self._state()
        self.assertEqual(st.status, "asking")
        self.assertEqual(len(st.get("answers") or []), 0)          # 未评分
        self.assertEqual(len(st.section_items()), 1)               # transcript 记了一条
        # next_section → 整段判分（score_section=7）+ 推进 → 已是最后一段 → finalize
        out_next = asyncio.run(tools_mod._next_section({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("模拟面试结束", out_next)
        st = self._state()
        self.assertEqual(st.status, "finished")
        evals = [(a.get("eval") or {}).get("score") for a in (st.get("answers") or [])]
        self.assertIn(7.0, evals)

    def test_next_section_rejected_in_discrete(self):
        self._start()
        out = asyncio.run(tools_mod._next_section({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("答题环节", out)
        self.assertEqual(self._state().status, "asking")

    # ── 确定性控制 ────────────────────────────────────────────────────
    def test_end_immediately_finalizes(self):
        self._start()
        out = asyncio.run(tools_mod._end({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("模拟面试结束", out)
        self.assertEqual(self._state().status, "finished")

    def test_skip_advances_without_score(self):
        self._start()
        out = asyncio.run(tools_mod._skip({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("跳过", out)
        self.assertIn("第2题", out)
        self.assertEqual(len(self._state().get("answers") or []), 0)

    def test_skip_rejected_in_dialogue(self):
        self.cfg.capabilities["interview"]["sections"] = [
            {"type": "self_intro", "name": "自我介绍"}]
        self._start()
        out = asyncio.run(tools_mod._skip({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("自由交流", out)

    def test_hint_in_asking_discrete(self):
        self._start()
        out = asyncio.run(tools_mod._hint({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("提示", out)

    def test_status_progress_and_finished(self):
        self._start()
        out = asyncio.run(tools_mod._status({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("进行中", out)
        self.assertIn("八股文", out)
        asyncio.run(tools_mod._end({}, self.cfg, ctx=_Ctx(self.sid)))
        out2 = asyncio.run(tools_mod._status({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("已结束", out2)

    def test_answer_without_active_interview(self):
        out = self._answer("随便说说")
        self.assertIn("没有进行中", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)