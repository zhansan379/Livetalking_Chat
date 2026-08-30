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
                               resume_text: str | None, jd_text: str | None) -> list[dict]:
    """组装本场题单（向量检索 → LLM 个性化）。永不返回空表。"""
    max_q = int(getattr(cfg, "interview_max_questions", 5) or 5)
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