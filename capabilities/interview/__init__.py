###############################################################################
#  模拟面试能力（interview）——可插拔能力的首个实体业务能力
#
#  · 模型驱动进入：模型识别“开始模拟面试”意图 → 调 interview.start（无 claim/拦截）。
#  · 状态制导：status ∈ idle|asking|finished。进行中把激活态片段注入 system_block，
#    模型据此续当面试官；结束/超时置 finished 后片段消失，自然退回普通助手。
#  · 按状态条件暴露工具子集：idle→[start]；asking→[answer,skip,hint,end,status]；
#    finished→[start,status]。正常闲聊一个面试工具都不进列表。
#  · 记忆/观测自动继承：面试轮仍走同一个 stream_llm_chat，零配置生效。
###############################################################################

from capabilities.base import Capability
from capabilities.interview.state import InterviewState
from capabilities.interview.tools import _tools
from capabilities.interview.prompts import (
    INTERVIEWER_PROMPT,
    WRAPUP_NOTICE,
    activation_block,
)


class InterviewCapability(Capability):
    name = "interview"
    priority = 10

    # ── 会话状态 → 工具子集映射（s07 按需加载）───────────────────────────
    # idle:     只留入口，模型能“开始面试”
    # asking:   进行中，撤下 start（已开始不能再开始）
    # finished: 可复盘或重新开一场
    _STATE_TOOLS = {
        "idle": ["interview.start"],
        "asking": ["interview.answer", "interview.skip", "interview.hint",
                    "interview.end", "interview.status"],
        "finished": ["interview.start", "interview.status"],
    }

    def tools(self) -> list[dict]:
        return _tools()

    def active_tools(self, session_id: str) -> list[str]:
        try:
            st = InterviewState(_cfg(), session_id)
            st.load()
            return list(self._STATE_TOOLS.get(st.status, self._STATE_TOOLS["idle"]))
        except Exception:  # noqa: BLE001 - 读态失败按 idle（只给入口，不崩）
            return list(self._STATE_TOOLS["idle"])

    def system_block(self, session_id: str, cfg) -> str:
        try:
            st = InterviewState(cfg, session_id)
            st.load()
        except Exception:  # noqa: BLE001 - 注入失败退回目录提示
            return _DIRECTORY_HINT

        if st.status == "asking":
            n_total = len(st.questions)
            n_ans = len(st.get("answers") or [])
            idx = min(st.get("idx", 0), n_total - 1) if n_total else 0
            cur = st.questions[idx] if n_total else {}
            return "\n\n".join([
                INTERVIEWER_PROMPT,
                activation_block(
                    st.get("role"), st.get("level"),
                    f"{n_ans}/{n_total} 题",
                    cur.get("text", "") if cur else "",
                ),
            ])
        if st.status == "finished":
            return WRAPUP_NOTICE
        return _DIRECTORY_HINT


_DIRECTORY_HINT = (
    "本助手提供【模拟面试】能力：当用户表达想练习面试（如\"来场前端初级面试\"、"
    "\"模拟面试\"）时，可调用 interview.start 开一场，我担任面试官出题、点评并出终局报告。"
)


def _cfg():
    from agent.config import get_agent_config
    return get_agent_config()


def config_defaults() -> dict:
    return {
        "enabled": False,
        "default_role": None,       # 缺省岗位方向
        "default_level": "初级",     # 缺省难度
        "max_questions": 5,
        "idle_timeout_s": 1800,
        "remember_policy": "identity_only",  # identity_only | full | none
        "bank_override": None,      # 覆盖题库 yaml；null→内置 bank_data.yaml
        "recall_top_k": 8,
        "store_dir": None,          # 状态目录；null→data/capabilities/interview
    }


CAPABILITY = InterviewCapability()