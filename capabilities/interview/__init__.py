###############################################################################
#  模拟面试能力（interview）——可插拔能力的首个实体业务能力
#
#  · 确定性进入：用户消息命中「开始面试」关键词时，pre_entry 由规则强制拉起
#    interview.start（hub → tool_loop 接管，不依赖模型是否自觉调用）。
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

    # ── 确定性进入：命中「开始面试」关键词时由规则拉起 interview.start ─────
    def pre_entry(self, message: str, session_id: str) -> dict | None:
        """用户消息命中开始-面试意图且非进行中 → 强制拉起 interview.start。

        进行中（status==asking）不抢占：面试已通过非 entry 工具在续（answer/skip/
        hint/end/status），若再接管会把已暴露的工具子集弄出与状态不一致的矛盾。
        """
        try:
            st = InterviewState(_cfg(), session_id)
            st.load()
            if st.status == "asking":
                return None
        except Exception:  # noqa: BLE001 - 读态失败按未进行处理，不崩
            pass
        args = _resolve_start(message)
        if args is None:  # 未命中；注意 {}（命中但无 role/level）是合法触发，不能判假
            return None
        return {"tool": "interview.start", "args": args}

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

# ── 确定性开始-面试意图（供 pre_entry 解析）──────────────────────────
# 只判定「要不要接管」，不给完整语义解析；role/level 尽力抽出、缺省交给配置默认。
# 注意：不要放「想面试/要面试/准备面试」这类也能出现在陈述句里的词（如
# “这个岗位要面试”）——会被误判成命令。真正命令式的这些词因 ≤6 字短口令规则
# 本就会触发，无需放进动词表。
_START_VERBS = ("开启", "开一场", "开一个", "开个", "开始", "来一场", "来场", "来一个",
                "进行", "举行", "模拟", "练习", "面试一下", "面一面", "组织")
_LEVEL_TOKENS = ("初级", "中级", "高级", "入门", "资深", "专家")
# 长 token 在前：先命中「数据分析」再退到「数据」，避免被短 token 截胡
_ROLE_TOKENS = ("springboot", "spring", "机器学习", "深度学习", "人工智能", "大模型",
                "数据分析", "大数据", "python", "java", "golang", "前端", "后端",
                "全栈", "客户端", "安卓", "android", "算法", "数仓", "测试", "运维",
                "嵌入式", "安全", "web", "nlp", "go", "react", "vue", "c++",
                "产品", "项目管理", "硬件", "游戏", "数据")
_BYTES_THRESHOLD = 8  # 结构化命令（role/level、无动词）允许的最大长度，区分"点名出题"与"闲聊提到"


def _resolve_start(message: str) -> dict | None:
    """命中「开始一场模拟面试」意图 → 返回 {"role":…,"level":…}（可只给其一）；否则 None。

    规则（尽力避免拦截闲聊）——满足其一即视为「开始」命令：
      1) `≤6 字`的短口令（如「面试」「开始面试」「模拟面试」）；
      2) 含一个开始动词（如「来一场前端初级面试」「准备后端面试」）；
      3) 不含动词但很短的结构化点名（如「高级Java面试」「数据分析中级面试」，
         含 role/level 且长度 ≤ 阈值）——这类通常就是"要开始"。
    其余（如「我明天有个面试，紧张」——提到但没有动词、也非结构化点名）不接管。
    """
    msg = (message or "").strip() or ""
    if "面试" not in msg:
        return None
    if len(msg) <= 6:
        pass
    elif any(v in msg for v in _START_VERBS):
        pass
    elif len(msg) <= _BYTES_THRESHOLD and any(
            t in msg for t in (*_LEVEL_TOKENS, *_ROLE_TOKENS)):
        pass
    else:
        return None

    # role/level 按表顺序取第一个命中（长 token 已排在前面，避免被短 token 截胡）。
    # 拉丁 token 大小写可能不同（Java/Python），用全小写比对。
    msg_low = msg.lower()
    args: dict = {}
    for lv in _LEVEL_TOKENS:
        if lv.lower() in msg_low:
            args["level"] = lv
            break
    for ro in _ROLE_TOKENS:
        if ro.lower() in msg_low:
            args["role"] = ro
            break
    return args


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
        "recall_top_k": 8,
        "bank_csv": None,           # 题库目录 CSV；null→data/question_bank.csv
        "essay_csv": None,          # 题目 CSV；null→data/question_essay.csv
        "index_dir": "data/capabilities/interview/chroma",  # chromadb 落盘目录
        "store_dir": None,          # 状态目录；null→data/capabilities/interview
    }


CAPABILITY = InterviewCapability()