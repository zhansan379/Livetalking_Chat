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
            "default_level": "初级",
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
        orig_route = tools_mod._route_dialogue

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

        async def _fake_route(cfg, stype, transcript):
            return None  # 默认回退整段判分；路由场景在用例内单独打桩

        tools_mod.build_section = _fake_section
        tools_mod.score_answer = _fake_score
        tools_mod.score_section = _fake_sec_score
        tools_mod.build_report = _fake_report
        tools_mod._route_dialogue = _fake_route
        self._orig = (orig_sheet, orig_score, orig_sec, orig_report, orig_route)

    def tearDown(self):
        (tools_mod.build_section, tools_mod.score_answer,
         tools_mod.score_section, tools_mod.build_report,
         tools_mod._route_dialogue) = self._orig

    def _start(self, args=None):
        # 显式带岗位，避免走「先问岗位」澄清；要测澄清时传 {} 再单独断言
        if args is None:
            args = {"role": "前端"}
        return asyncio.run(tools_mod._start(args, self.cfg, ctx=_Ctx(self.sid)))

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

    # ── 未指明岗位先询问，而非静默用默认岗位 ─────────────────────────
    def test_start_asks_role_when_not_given(self):
        out = self._start({})                       # 不带岗位
        self.assertIn("岗位", out)
        self.assertEqual(self._state().status, "idle")   # 尚未开场
        # 补上岗位 → 正常开场
        out2 = self._start({"role": "后端"})
        self.assertIn("模拟面试", out2)
        self.assertEqual(self._state().status, "asking")

    def test_start_repeated_no_role_falls_back(self):
        # 第一次问；用户再次只说『开始』（仍无岗位）→ 通用兜底开场，不卡流程
        self._start({})
        out = self._start({})
        self.assertIn("模拟面试", out)
        self.assertEqual(self._state().status, "asking")


    def test_end_mid_dialogue_preserves_answers(self):
        # 复现实测故障：自我介绍对话段答了几段，用户没说『进入下一环节』直接 end，
        # 之前会 0/0；现在应把当段 transcript 补评分并入 answers。
        self.cfg.capabilities["interview"]["sections"] = [
            {"type": "self_intro", "name": "自我介绍"}]
        self._start()
        self._answer("我叫小李，两年后端经验。")
        self._answer("主要做网关与订单服务。")
        out = asyncio.run(tools_mod._end({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("模拟面试结束", out)
        self.assertNotIn("0 个作答", out)
        self.assertIn("1 个作答", out)
        st = self._state()
        self.assertEqual(st.status, "finished")
        answers = st.get("answers") or []
        self.assertEqual(len(answers), 1)               # 对话段整段一条
        self.assertEqual(answers[0]["section_type"], "self_intro")
        self.assertEqual((answers[0]["eval"] or {}).get("score"), 7.0)

    def test_end_mid_self_intro_routes_technical_to_project_anchors(self):
        # 自我介绍里夹带了具体技术内容，用户在结束面试时没走「进入下一环节」→ 收尾应按
        # 内容切段分到各环节：技术部分落到 project 走 4 维锚点评（score_answer=8），
        # 而不是整段糊成一个「表达/结构/匹配」的自我介绍。
        self.cfg.capabilities["interview"]["sections"] = [
            {"type": "self_intro", "name": "自我介绍"}]

        async def _route(cfg, stype, transcript):
            return [
                {"type": "self_intro", "name": "自我介绍", "topic": "背景",
                 "content": "我叫小李，两年后端经验。"},
                {"type": "project", "name": "项目问答", "topic": "网关限流",
                 "content": "我实现了网关限流，用令牌桶。"},
            ]
        tools_mod._route_dialogue = _route

        self._start()
        self._answer("我叫小李，两年后端经验。")
        self._answer("我做过网关，实现了限流。")
        out = asyncio.run(tools_mod._end({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("2 个作答", out)          # 切段后按 2 条计，不再只算 1 段
        answers = self._state().get("answers") or []
        types = [a["section_type"] for a in answers]
        self.assertIn("self_intro", types)
        self.assertIn("project", types)         # 技术内容被路由到项目问答
        proj = next(a for a in answers if a["section_type"] == "project")
        self.assertEqual((proj["eval"] or {}).get("score"), 8.0)   # 走 score_answer 4 维
        self.assertIn("网关限流", str(proj.get("question", {}).get("text", "")))

    def test_end_after_next_section_no_duplicate(self):
        # 对话段已用 next_section 判分推进后 end，不得因收拢逻辑二次计分。
        self.cfg.capabilities["interview"]["sections"] = [
            {"type": "self_intro", "name": "自我介绍"},
            {"type": "project", "name": "项目问答", "count": 1}]
        self._start()
        self._answer("我叫小李，两年后端经验。")
        asyncio.run(tools_mod._next_section({}, self.cfg, ctx=_Ctx(self.sid)))
        # 现在停在 project 离散段，直接 end
        out = asyncio.run(tools_mod._end({}, self.cfg, ctx=_Ctx(self.sid)))
        self.assertIn("模拟面试结束", out)
        answers = self._state().get("answers") or []
        self.assertEqual(len(answers), 1)               # 仅 self_intro 一条，无重复


if __name__ == "__main__":
    unittest.main(verbosity=2)