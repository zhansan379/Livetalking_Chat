###############################################################################
#  可插拔能力中枢测试：discovery / 注册 / 门控 / 会话状态条件暴露 / 系统注入。
#
#  运行：python -m pytest tests/ 或 python tests/test_capability_hub.py
###############################################################################

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import tempfile
import unittest


class CapabilityHubTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="cap_hub_")

    def tearDown(self):
        # 复原单例能力配置，避免污染其它用例
        from agent.config import get_agent_config
        get_agent_config().capabilities = dict(get_agent_config().capabilities)

    def test_discovery_finds_interview_and_drops_removed_hello(self):
        from capabilities.hub import all_capabilities
        caps = all_capabilities()
        self.assertIn("interview", caps)
        # hello 冒烟示例能力已从代码库移除，不应再被 discovery 发现
        self.assertNotIn("hello", caps)

    def test_disabled_capability_exposes_no_tools_and_no_block(self):
        from capabilities.hub import session_tools, capability_system_block
        from agent.config import get_agent_config
        cfg = get_agent_config()
        cfg.capabilities["interview"] = {"enabled": False}
        names = session_tools("any", cfg)
        self.assertNotIn("interview.start", names)
        self.assertNotIn("start", capability_system_block("any", cfg))

    def test_enabled_interview_appears_and_is_cap_gated(self):
        """启用的能力工具要出现在 session_tools，且带能力命名空间前缀(interview.*)。"""
        from capabilities.hub import session_tools
        from agent.config import get_agent_config
        cfg = get_agent_config()
        # 用 temp store_dir，避免测试在仓库 data/ 目录落状态
        cfg.capabilities["interview"] = {"enabled": True, "store_dir": self._tmp}
        names = session_tools("any", cfg)
        self.assertIn("interview.start", names)   # idle 态暴露入口工具

    def test_list_enabled_tools_excludes_capability_tools(self):
        """全局工具列表（reminder 后台路径用）不得含能力工具。"""
        from capabilities.hub import register_capability_tools
        from agent.tool_loop import list_enabled_tools
        from agent.config import get_agent_config
        register_capability_tools()
        global_tools = list_enabled_tools(get_agent_config())
        self.assertNotIn("interview.start", global_tools)

    def test_interview_state_driven_exposure(self):
        """idle→[start]；asking→[answer...]无start；finished→[start,status]。"""
        asyncio.run(self._state_driven())

    async def _state_driven(self):
        from capabilities.hub import session_tools, capability_system_block
        from agent.config import get_agent_config
        from capabilities.interview.state import InterviewState

        cfg = get_agent_config()
        cfg.capabilities["interview"] = {"enabled": True, "store_dir": self._tmp}
        cap_names = lambda tools: [t for t in tools if "interview" in t]

        def explore(state_data):
            return (cap_names(session_tools("sid", cfg)),
                    capability_system_block("sid", cfg))

        idle_tools, idle_blk = explore({})
        self.assertEqual(idle_tools, ["interview.start"])

        st = InterviewState(cfg, "sid")
        await st.save({
            "status": "asking", "role": "前端", "level": "初级",
            "questions": [{"id": "q1", "text": "解释闭包", "category": "JS"}],
            "idx": 0, "answers": [], "started_at": "x",
        })
        ask_tools, ask_blk = explore({})
        self.assertIn("interview.answer", ask_tools)
        self.assertNotIn("interview.start", ask_tools)
        self.assertIn("模拟面试官", ask_blk)

        await st.save({"status": "finished"})
        fin_tools, fin_blk = explore({})
        self.assertEqual(sorted(fin_tools), ["interview.start", "interview.status"])

    def test_config_defaults_merged(self):
        """config_defaults 并入 capabilities 覆盖节，用户可覆盖。"""
        from capabilities.hub import capability_config_defaults
        d = capability_config_defaults()
        self.assertIn("interview", d)
        # 已移除的 hello 冒烟能力不应再有默认配置
        self.assertNotIn("hello", d)
        self.assertEqual(d["interview"]["max_questions"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)