###############################################################################
#  模拟面试状态机测试：fake 评分/eval 注入下走通 start→answer→…→终局报告。
#
#  不联网：monkeypatch recall/eval 的 LLM 环节为固定返回，验证 handler 的
#  确定性流转（题数收敛、skip/end 收尾、status 只读）与状态落盘。
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
        cfg.capabilities["interview"] = {
            "enabled": True, "store_dir": self._tmp,
            "default_role": "前端", "default_level": "初级",
        }
        self.cfg = cfg
        self.sid = "sid-test"

        # 注入 fake，避免联网
        orig_sheet = tools_mod.build_question_sheet
        orig_score = tools_mod.score_answer
        orig_report = tools_mod.build_report

        async def _fake_sheet(cfg, role, level, resume_text, jd_text):
            return [dict(q) for q in self.Q]

        async def _fake_score(cfg, question, answer):
            return {"question_id": question.get("id"), "score": 8.0,
                    "dimension_notes": {"理解": 8, "表达": 8, "逻辑": 8, "完整": 8},
                    "comment": "不错"}

        async def _fake_report(cfg, questions, answers, role, level, jd_text):
            return {"summary": "整体表现良好", "dimension_avg": {"理解": 8, "表达": 8, "逻辑": 8, "完整": 8},
                    "strengths": ["思路清晰"], "improvements": [], "suggested_topics": ["性能优化"]}

        tools_mod.build_question_sheet = _fake_sheet
        tools_mod.score_answer = _fake_score
        tools_mod.build_report = _fake_report
        self._orig = (orig_sheet, orig_score, orig_report)

    def tearDown(self):
        tools_mod.build_question_sheet, tools_mod.score_answer, tools_mod.build_report = self._orig

    def _start(self):
        return asyncio.run(tools_mod._start({}, self.cfg, ctx=_Ctx(self.sid)))

    def _answer(self, answer):
        return asyncio.run(tools_mod._answer({"answer": answer}, self.cfg, ctx=_Ctx(self.sid)))

    def _state(self):
        from capabilities.interview.state import InterviewState
        st = InterviewState(self.cfg, self.sid)
        st.load()
        return st

    def test_full_run_finalizes_report(self):
        from capabilities.interview.state import InterviewState
        self.assertIn("模拟面试", self._start())
        # 答第1题 → 进第2题
        out2 = self._answer("闭包能捕获外部变量")
        self.assertIn("第2题", out2)
        # 答第2题 → 满题，终局报告
        out_report = self._answer("事件循环是……")
        self.assertIn("模拟面试结束", out_report)
        st = self._state()
        self.assertEqual(st.status, "finished")
        self.assertIsNotNone(st.get("report"))
        self.assertEqual(len(st.get("answers") or []), 2)

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

    def test_hint_in_asking(self):
        self._start()
        out = asyncio.run(tools_mod._hint({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("提示", out)

    def test_status_finished_after_end(self):
        self._start()
        self.assertIn("进行中", asyncio.run(tools_mod._status({}, self.cfg, ctx=_Ctx(self.sid))))
        asyncio.run(tools_mod._end({}, self.cfg, ctx=_Ctx(self.sid)))
        out = asyncio.run(tools_mod._status({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("已结束", out)

    def test_answer_without_active_interview(self):
        out = self._answer("随便说说")
        self.assertIn("没有进行中", out)

    def test_recall_top_k_read_from_config(self):
        self.assertEqual(self.cfg.interview_max_questions, 5)
        self.assertEqual(self.cfg.interview_recall_top_k, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)