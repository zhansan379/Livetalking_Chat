###############################################################################
#  通用文件工具测试：会话范围安全隔离 + 越权拒绝 + 截断 + 缺解析器降级。
###############################################################################

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import asyncio
import os
import tempfile
import unittest


class _Ctx:
    def __init__(self, session_id):
        self.session_id = session_id


class FileToolsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="file_tools_")
        self.sid = "sess-A1"
        self.sid_dir = os.path.join(self._tmp, self.sid)
        os.makedirs(self.sid_dir, exist_ok=True)
        with open(os.path.join(self.sid_dir, "resume.txt"), "w", encoding="utf-8") as f:
            f.write("我叫小明，有五年前端经验")

        from agent.config import get_agent_config
        cfg = get_agent_config()
        cfg.file_upload_dir = self._tmp  # 指向临时根，避免污染 data/uploads

    def _cfg(self):
        from agent.config import get_agent_config
        return get_agent_config()

    def test_list_files_shows_uploaded(self):
        from agent.files import list_files
        out = asyncio.run(list_files({}, self._cfg(), ctx=_Ctx(self.sid)))
        self.assertIn("resume.txt", out)

    def test_read_bare_filename_returns_content(self):
        from agent.files import read_file
        out = asyncio.run(read_file({"path": "resume.txt"}, self._cfg(), ctx=_Ctx(self.sid)))
        self.assertIn("小明", out)
        self.assertIn("<file-content>", out)

    def test_path_traversal_refused(self):
        from agent.files import read_file
        out = asyncio.run(read_file({"path": "../other/x.txt"}, self._cfg(), ctx=_Ctx(self.sid)))
        self.assertIn("只提供文件名", out)

    def test_absolute_path_refused(self):
        from agent.files import read_file
        evil = os.path.join(self._tmp, "..") or "C:/Windows/win.ini"
        out = asyncio.run(read_file({"path": evil}, self._cfg(), ctx=_Ctx(self.sid)))
        self.assertIn("只提供文件名", out)

    def test_cross_session_refused(self):
        from agent.files import read_file
        out = asyncio.run(read_file({"path": "resume.txt"}, self._cfg(), ctx=_Ctx("sess-B2")))
        self.assertNotIn("小明", out)
        self.assertIn("找不到", out)

    def test_no_session_refused(self):
        from agent.files import read_file
        out = asyncio.run(read_file({"path": "resume.txt"}, self._cfg(), ctx=_Ctx("")))
        self.assertIn("没有有效", out)

    def test_max_chars_truncates(self):
        from agent.files import read_file
        with open(os.path.join(self.sid_dir, "long.txt"), "w", encoding="utf-8") as f:
            f.write("中" * 500)
        # max_chars 有 200 字下限：传 300 → 500 字文件被截到 ~300
        out = asyncio.run(read_file({"path": "long.txt", "max_chars": 300}, self._cfg(), ctx=_Ctx(self.sid)))
        body = out.split("<file-content>")[1].split("</file-content>")[0].strip()
        self.assertLessEqual(len(body), 300)
        self.assertLess(len(body), 500)  # 确证确实截断了

    def test_unknown_ext_degrades(self):
        from agent.files import read_file
        with open(os.path.join(self.sid_dir, "blob.bin"), "wb") as f:
            f.write(b"\x00\x01\x02")
        out = asyncio.run(read_file({"path": "blob.bin"}, self._cfg(), ctx=_Ctx(self.sid)))
        self.assertIn("无法解析", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)