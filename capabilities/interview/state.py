###############################################################################
#  模拟面试 · 状态机 + 原子持久化
#
#  一场面试的状态收敛到一个 dict，读写走「临时文件 + os.replace」原子落盘
#  （复用 longterm 的既有约定）。会话状态跨轮续面：system_block 读到 status==asking
#  即注入激活态片段；answer/end/超时 三种收敛都置 status=finished。
#
#  字段：
#     session_id, role, level, resume_text, jd_text,
#     questions  : 本场题单（每项 {id,text,category,type,rubrics,followups}）
#     idx        : 当前正被问到的题目下标（questions[idx]）
#     answers    : 已作答列表 [{question, answer, eval}]
#     status     : idle | asking | finished
#     started_at, finished_at
###############################################################################

import asyncio
import json
import os
import tempfile

from utils.logger import logger


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

    @property
    def questions(self) -> list[dict]:
        """本场题单（读缓存态；load()/save() 后与磁盘一致）。"""
        return self.get("questions") or []

    @property
    def idx(self) -> int:
        """当前正被问到的题目下标（越过题数上限则取最后一题安全位）。"""
        n = len(self.questions)
        if not n:
            return 0
        return max(0, min(int(self.get("idx") or 0), n - 1))

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