###############################################################################
#  模拟面试 · 状态机 + 原子持久化
#
#  一场面试的状态收敛到一个 dict，读写走「临时文件 + os.replace」原子落盘
#  （复用 longterm 的既有约定）。会话状态跨轮续面：system_block 读到 status==asking
#  即注入激活态片段；answer/end/超时 三种收敛都置 status=finished。
#
#  字段：
#     session_id, role, level, resume_text, jd_text,
#     sections     : 本场环节描述（每项 {type,name,count?,dialogue:bool}），有序
#     section_idx  : 当前环节下标（sections[section_idx]）
#     items        : 当前环节内容——
#                      离散段 → 题目 dict 列表 {id,text,category,type,rubrics,followups};
#                      对话段 → transcript [{role:"candidate",text:...}, ...]
#     idx          : 离散段段内游标（当前 items[idx]）；对话段恒 0 不用
#     answers      : 已作答列表 [{question, answer, eval, section_type, section_name}]
#                    离散题一条；对话段在 next_section 追加一条整段评分记录
#     status       : idle | asking | finished
#     started_at, finished_at
###############################################################################

import asyncio
import json
import os
import tempfile

from utils.logger import logger

# 可配环节的 type 枚举（sections 里每项的 type）
SECTION_TYPES = ("self_intro", "project", "trivia", "reverse_qa")
# 对话式环节：自由多轮交谈、段末整段一次性评分；其余为离散题环节
DIALOGUE_TYPES = {"self_intro", "reverse_qa"}
# 默认环节序列（config_defaults 与 run-time 兜底共享）。对话段不设 count。
DEFAULT_SECTIONS = [
    {"type": "self_intro", "name": "自我介绍"},
    {"type": "project", "name": "项目问答", "count": 3},
    {"type": "trivia", "name": "八股文", "count": 3},
    {"type": "reverse_qa", "name": "反问"},
]


def _base_dir(cfg) -> str:
    return getattr(cfg, "interview_store_dir", None) or "data/capabilities/interview"


def _state_path(cfg, session_id: str) -> str:
    """session_id 仅保留安全字符，杜绝越权路径。"""
    safe = "".join(c for c in (session_id or "anon") if c.isalnum() or c in "._-")
    return os.path.join(_base_dir(cfg), f"{safe or 'anon'}.json")


class InterviewState:
    """一个会话的面试状态：读写 + 持久化 + 每会话锁串行化。"""

    def __init__(self, cfg, session_id: str):
        self.cfg = cfg
        self.session_id = session_id or ""
        self.path = _state_path(cfg, self.session_id)
        # 每会话独立锁：同会话上一条消息的 tool handler 可能与本条交错，写需串行化
        self._lock = asyncio.Lock()
        self._data = {}

    # ── 读取 ────────────────────────────────────────────────────────────────
    def load(self) -> dict:
        """读入状态；文件缺失/损坏回退空态（绝不崩）。"""
        self._data = {}
        try:
            if os.path.isfile(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    self._data = json.load(f) or {}
        except Exception as e:  # noqa: BLE001 - 读失败按空态处理，能力可重新开始
            logger.warning("interview state load failed (%s): %s", self.session_id, e)
            self._data = {}
        return self._data

    def get(self, key, default=None):
        return self._data.get(key, default)

    @property
    def status(self) -> str:
        return self._data.get("status", "idle")

    @property
    def is_active(self) -> bool:
        """是否有未完成的面试在进行中（供 system_block / 超时判定）。"""
        return self.status == "asking"

    # ── 段/题游标（取代旧 questions/idx 单指针）───────────────
    def current_section(self) -> dict:
        """当前环节描述（sections[section_idx] 或空 dict）。"""
        secs = self.get("sections") or []
        i = int(self.get("section_idx") or 0)
        return secs[i] if secs and 0 <= i < len(secs) else {}

    def section_type(self) -> str:
        return self.current_section().get("type") or ""

    def is_dialogue(self) -> bool:
        """当前环节是否对话式（自由多轮、段末整段评分）。"""
        return self.section_type() in DIALOGUE_TYPES

    def section_items(self, default=None):
        """当前环节 items（离散段题单 / 对话段 transcript）。"""
        return self.get("items", [] if default is None else default)

    def inline_idx(self) -> int:
        """离散段内游标（越界则取最后一项安全位）；对话段恒 0。"""
        n = len(self.section_items())
        if not n or self.is_dialogue():
            return 0
        return max(0, min(int(self.get("idx") or 0), n - 1))

    def in_last_section(self) -> bool:
        secs = self.get("sections") or []
        i = int(self.get("section_idx") or 0)
        return bool(secs) and i >= len(secs) - 1

    # ── 写入 ────────────────────────────────────────────────────────────────
    async def save(self, data: dict | None = None) -> dict:
        """在会话锁内更新并原子落盘。返回合并后的最新数据。"""
        if data is not None:
            self._data.update(data)
        async with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                dir=os.path.dirname(self.path), prefix=".iv-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
            except Exception as e:  # noqa: BLE001
                logger.warning("interview state save failed: %s", e)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        return self._data

    async def clear(self) -> None:
        """会话结束/彻底重置时删除状态文件。"""
        async with self._lock:
            try:
                if os.path.isfile(self.path):
                    os.unlink(self.path)
            except OSError:
                pass
        self._data = {}