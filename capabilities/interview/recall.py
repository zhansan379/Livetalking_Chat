###############################################################################
#  模拟面试 · 题目召回与个性化（向量检索 + 一次 LLM）
#
#  ① 向量检索（bank.search：chromadb 本地 onnx 模型，question_essay 题库 top-k）
#  ② LLM 个性化（只对 top-k 做一次 LLM：排序、按岗位润色、针对简历补 1-2 道现场追问）
#
#  ①把 4.9 万题库压到 top-k，②只对 top-k 做一次 LLM——不把题库搬进上下文。
#  无简历/JD 时走纯库题；向量检索失败降级返回空 → 由调用方兜底，绝不空手而归。
###############################################################################

import json

from utils.logger import logger

from capabilities.interview.bank import search as bank_search


def _recall_query(role: str | None, level: str | None,
                  resume_text: str | None, jd_text: str | None) -> str:
    """召回 query：方向/难度 + 简历 + 岗位要求（作为排序的语义锚点）。"""
    parts = [f"模拟面试：{role or '通用'} · {level or '难度自适应'}"]
    if jd_text and jd_text.strip():
        parts.append(f"岗位要求：{jd_text.strip()[:800]}")
    if resume_text and resume_text.strip():
        parts.append(f"候选人简历：{resume_text.strip()[:800]}")
    return "\n".join(parts)


async def _personalize(cfg, candidates: list[dict], resume_text: str | None,
                       jd_text: str | None, max_q: int) -> list[dict]:
    """③ LLM 个性化：对已排序 top-k 生成本场题单（排序+润色+简历追问）。"""
    seed = [
        {"id": q.get("id"), "text": q.get("text"), "category": q.get("category"),
         "type": q.get("type"), "answer": (q.get("answer") or "")[:400],
         "rubrics": q.get("rubrics") or {}, "followups": list(q.get("followups") or [])}
        for q in candidates[:max(1, max_q + 2)]
    ]
    prompt_msgs = [
        {"role": "system", "content": (
            "你是模拟面试的出题老师。给定候选题目池（含每题参考答案）和（可选的）岗位要求/"
            "候选人简历，产出一场【不超过 max_q 题】的面试题单。返回严格 JSON："
            "{\"questions\":[{\"id\":\"原id或新id\",\"text\":\"题目(可按岗位润色/结合简历生成)\","
            "\"category\":\"分类\",\"type\":\"technical或behavioral或situational\","
            "\"rubrics\":{\"理解\":\"...\",\"表达\":\"...\",\"逻辑\":\"...\",\"完整\":\"...\"},"
            "\"followups\":[\"...\"]}]}。"
            "优先保留与岗位要求最相关的技术题；rubrics 可结合题目参考答案定评分锚点；"
            "若无简历可只从池里选题；题目数不超过 "
            f"{max_q} 题。只输出 JSON。"
        )},
        {"role": "user", "content": (
            f"候选题目池（JSON）：{seed}\n"
            f"岗位要求：{jd_text or '(无)'}\n候选人简历：{resume_text or '(无)'}"
        )},
    ]
    import re as _re
    from infra_ai import async_call_llm
    try:
        raw = await async_call_llm(
            prompt_msgs, use_json=True,
            extra={"kind": "interview_sheet", "n_questions": max_q},
        )
        data = {}
        text = (raw or "").strip()
        try:
            data = json.loads(text) if text.startswith("{") else {}
        except json.JSONDecodeError:
            m = _re.search(r"\{.*\}", text, _re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group(0))
                except json.JSONDecodeError:
                    data = {}
        qs = (data.get("questions") if isinstance(data, dict) else None) or []
        return [
            {**q, "id": q.get("id") or q.get("text"), "text": q.get("text", "")}
            for q in qs if q.get("text")
        ][:max_q]
    except Exception as e:  # noqa: BLE001 - 个性化失败退 top-k 原题
        logger.warning("interview personalize failed, fallback to top-k: %s", e)
        return [
            {k: q.get(k) for k in ("id", "text", "category", "type", "rubrics", "followups")}
            for q in candidates[:max_q]
        ]


async def build_question_sheet(cfg, role: str | None, level: str | None,
                               resume_text: str | None, jd_text: str | None,
                               max_q: int | None = None) -> list[dict]:
    """组装题单（向量检索 → LLM 个性化）。永不返回空表。max_q 可覆盖配置题数。"""
    max_q = max_q or int(getattr(cfg, "interview_max_questions", 5) or 5)
    top_k = max(max_q, int(getattr(cfg, "interview_recall_top_k", 8) or 8))

    query = _recall_query(role, level, resume_text, jd_text)
    pool = bank_search(cfg, query, top_k)
    if not pool:
        return []

    sheet = await _personalize(cfg, pool, resume_text, jd_text, max_q)
    return sheet or _plain(pool[:max_q])


def _plain(qs: list[dict]) -> list[dict]:
    """把 chromadb 记录摊成题单结构（type/rubrics/followups 缺失时 LLM 已兜底）。"""
    out = []
    for q in qs:
        out.append({
            "id": q.get("id"), "text": q.get("text"), "category": q.get("category"),
            "type": "", "rubrics": {}, "followups": [],
        })
    return out


# ── 按环节生产题目（self_intro / project / trivia / reverse_qa）─────────
async def build_section(cfg, section: dict, role: str | None, level: str | None,
                        resume_text: str | None, jd_text: str | None) -> list:
    """按环节 type 生产当前段 items。对话段返回 []（自由交流，零 LLM）；离散段永不空表。"""
    from capabilities.interview.state import DIALOGUE_TYPES
    stype = (section or {}).get("type") or "trivia"
    count = int((section or {}).get("count") or 0) or int(
        getattr(cfg, "interview_max_questions", 5) or 5)
    if stype in DIALOGUE_TYPES:
        return []
    if stype == "project":
        return await build_project_section(cfg, role, level, resume_text, count)
    return await build_question_sheet(cfg, role, level, resume_text, jd_text, max_q=count)


# ── 项目问答段：有简历按项目深挖；无简历/无项目用通用项目题兜底 ────────────
_PROJECT_RUBRICS = {
    "理解": "是否讲清项目背景与自己在其中的角色",
    "表达": "是否条理清楚地描述项目脉络",
    "逻辑": "难点→方案→结果是否自洽、有因果",
    "完整": "是否覆盖背景/难点/方案/结果/复盘",
}
_PROJECT_FOLLOWUPS = [
    "当时的难点具体是什么？你怎么定位并找到根因的？",
    "技术选型为什么这样选？有没有对比过别的方案？",
    "结果如何度量（性能/收益/工时）？有没有量化数据？",
    "如果再重做一次，哪里会改进？",
]


def _projects_to_questions(projects: list[dict]) -> list[dict]:
    out = []
    for i, p in enumerate(projects):
        name = (p.get("name") or f"项目{i + 1}").strip()
        brief = "\n".join(str(p.get(k) or "") for k in ("summary", "difficulty", "highlights")).strip()
        out.append({
            "id": f"proj-{i}",
            "text": f"请挑你简历里的项目『{name}』，完整讲一遍：背景 → 你的困难 → 你的方案 → 结果与复盘。",
            "category": "项目问答",
            "type": "behavioral",
            "rubrics": dict(_PROJECT_RUBRICS),
            "followups": list(_PROJECT_FOLLOWUPS),
            # 供 system_block/hint 深挖；评分与报告忽略该字段
            "brief": brief[:800],
        })
    return out


async def _extract_projects(cfg, resume_text: str | None, want: int) -> list[dict] | None:
    """从简历抽项目（capability=extract 便宜档）；失败/无 → None，交由兜底。"""
    if not resume_text or not resume_text.strip():
        return None
    msgs = [
        {"role": "system", "content": (
            "你是简历解析器。读候选人简历，抽出其做过的项目，每个给："
            "{\"name\":名称,\"summary\":一两句简介,\"difficulty\":技术或业务难点,\"highlights\":亮点与量化结果}。"
            f"最多 {want} 个。只输出严格 JSON：{{\"projects\":[...]}}。若简历没有项目则返回 {{\"projects\":[]}}。"
        )},
        {"role": "user", "content": resume_text.strip()[:4000]},
    ]
    try:
        from infra_ai import async_call_llm
        raw = await async_call_llm(
            msgs, use_json=True,
            capability="extract",
            extra={"kind": "interview_resume_extract"},
            model_kwargs={"max_tokens": 600},
        )
        import json as _json
        data = _json.loads((raw or "").strip()) if (raw or "").strip().startswith("{") else {}
        projects = (data.get("projects") if isinstance(data, dict) else None) or []
        return [p for p in projects if (p or {}).get("name")][:want]
    except Exception as e:  # noqa: BLE001 - 抽取失败退 None 走兜底，绝不弹错
        logger.warning("interview extract projects failed: %s", e)
        return None


async def build_project_section(cfg, role: str | None, level: str | None,
                                resume_text: str | None, count: int) -> list:
    """项目问答段 items：有简历→按项目逐段深挖；无简历/抽取失败→通用项目题兜底。永不空表。"""
    projects = await _extract_projects(cfg, resume_text, max(1, count))
    if projects:
        return _projects_to_questions(projects)
    # 兜底：通用『项目式』题
    q = "描述一个你做过的有挑战、印象最深的项目，讲清背景、难点、你的方案与结果。"
    pool = bank_search(cfg, q, max(count, 3))
    if pool:
        sheet = await _personalize(cfg, pool, None, None, max(1, count))
        if sheet:
            return [dict(x) for x in sheet]
    return [{
        "id": "proj-fallback",
        "text": q,
        "category": "项目问答",
        "type": "behavioral",
        "rubrics": dict(_PROJECT_RUBRICS),
        "followups": list(_PROJECT_FOLLOWUPS),
    }]