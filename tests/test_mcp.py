###############################################################################
#  MCP 工具接入单元测试（仅覆盖离线纯路径：不 spawn 子进程、不联网）
#
#  覆盖：_mcp_call 服务器未连接兜底 / 结果归一化(text·error·structured) /
#      输入 schema 兜底 / config 读取(mcp.servers 按名排序、per-server spec)。
#  联网/子进程的端到端验证见 docs/how-to-mcp.md（需要真实 MCP 服务器）。
###############################################################################

import unittest


def _mkcfg(servers=(), enabled=True, timeout=15):
    base = {"tools": {"mcp": {"enabled": enabled, "connect_timeout": timeout,
                              "servers": {n: s for n, s in servers}}}}
    from agent.config import AgentConfig
    return AgentConfig(base)


class McpModuleTest(unittest.TestCase):
    def test_call_when_server_not_connected_returns_graceful_text(self):
        import asyncio
        from types import SimpleNamespace
        from agent import mcp as m

        async def go():
            out = await m._mcp_call({"q": 1}, SimpleNamespace(), server="absent", tool="foo")
            return out

        out = asyncio.run(go())
        self.assertIn("absent", out)      # 明确指出未连接的服务器
        self.assertIn("foo", out)

    def test_format_text_result(self):
        from agent import mcp as m

        class Blk:
            type = "text"
            text = "hello 世界"

        class Res:
            content = [Blk()]
            is_error = False
            structured_content = None

        self.assertEqual(m._format_result(Res()), "hello 世界")

    def test_format_error_with_structured_content(self):
        from agent import mcp as m

        class Res:
            content = []
            is_error = True
            structured_content = [{"k": "v"}]

        out = m._format_result(Res())
        self.assertIn("错误", out)
        self.assertIn("k", out)

    def test_empty_result_graceful(self):
        from agent import mcp as m

        class Res:
            content = []
            is_error = False
            structured_content = None

        self.assertIn("无输出", m._format_result(Res()))

    def test_input_schema_fallback_when_missing(self):
        from types import SimpleNamespace
        from agent import mcp as m

        tool = SimpleNamespace(input_schema=None)   # 缺 schema 时兜底空对象协议
        self.assertEqual(m._input_schema(tool), {"type": "object", "properties": {}})


class McpConfigTest(unittest.TestCase):
    def test_defaults(self):
        cfg = _mkcfg(enabled=False, servers=())
        self.assertFalse(cfg.tool_mcp_enabled)
        self.assertEqual(cfg.mcp_connect_timeout, 15)
        self.assertEqual(cfg.mcp_servers, [])

    def test_servers_parsed_and_sorted_by_name(self):
        cfg = _mkcfg(servers=[
            ("fs", {"transport": "stdio", "command": "npx", "args": ["-y", "x"], "env": {"K": "V"}}),
            ("zz", {"transport": "http", "url": "https://a/mcp", "enabled": False}),
            ("aa", {"transport": "sse", "url": "https://b/sse"}),
        ])
        self.assertTrue(cfg.tool_mcp_enabled)
        self.assertEqual(cfg.mcp_connect_timeout, 15)
        self.assertEqual([n for n, _ in cfg.mcp_servers], ["aa", "fs", "zz"])
        spec = dict(cfg.mcp_servers)["fs"]
        self.assertEqual(spec["transport"], "stdio")
        self.assertFalse(dict(cfg.mcp_servers)["zz"].get("enabled", True))


if __name__ == "__main__":
    unittest.main()