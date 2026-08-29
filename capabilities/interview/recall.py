###############################################################################
#  模拟面试 · 题目召回与个性化（两级检索 + 一次 LLM）
#
#  ① filter（确定性过滤，bank.filter_bank，缩小候选池）
#  ② rerank（复用 infra_ai.async_rerank，候选池 × 简历+JD+方向 → top-k）
#  ③ LLM 个性化（只对 top-k 做一次 LLM：排序、按岗位润色、针对简历补 1-2 道现场追问）
#
#  ①②把大题库压到 top-k，③只对 top-k 做一次 LLM——不把题库搬进上下文。
#  无简历/JD 时走纯库题；任一环节失败降级到前一步结果，绝不空手而归。
###############################################################################

import json
import asyncio

from utils.logger import logger

from capabilities.interview.bank import filter_bank


def _recall_query(role: str | None, level: str | None,
                  resume_text: str | None, jd_text: str | None) -> str:
    """召回 query：方向/难度 + 简历 + 岗位要求（作为排序的语义锚点）。"""
    parts = [f"模拟面试：{role or '通用'} · {level or '难度自适应'}"]
    if jd_text and jd_text.strip():
        parts.append(f"岗位要求：{jd_text.strip()[:800]}")
    if resume_text and resume_text.strip():
        parts.append(f"候选人简历：{resume_text.strip()[:800]}")
    return "\n".join(parts)


async def _rerank_topk(query: str, docs: list[str], top_n: int) -> list[int]:
    """返回 docs 按 query 相关度排序后的下标列表（前 top_n）。"""
    from infra_ai import async_rerank
    try:
        resp = await async_rerank(query, docs, top_n=top_n)
        scored = sorted(resp.results, key=lambda r: r.index)
        # results 为按相关度排序的命中；还原成在 docs 内的原下标
        order = [r.index for r in sorted(resp.results, key=lambda r: r.relevance_score, reverse=True)]
        return order
    except Exception as e:  # noqa: BLE001 - rerank 失败退原始顺序
        logger.warning("interview rerank failed, fallback to bank order: %s", e)
        return list(range(len(docs)))


async def _personalize(cfg, candidates: list[dict], resume_text: str | None,
                       jd_text: str | None, max_q: int) -> list[dict]:
    """③ LLM 个性化：对已排序 top-k 生成本场题单（排序+润色+简历追问）。"""
    seed = [
        {"id": q.get("id"), "text": q.get("text"), "category": q.get("category"),
         "type": q.get("type"), "rubrics": q.get("rubrics") or {},
         "followups": list(q.get("followups") or [])}
        for q in candidates[:max(1, max_q + 2)]
    ]
    prompt_msgs = [
        {"role": "system", "content": (
            "你是模拟面试的出题老师。给定候选题目池和（可选的）岗位要求/候选人简历，"
            "产出一场【不超过 max_q 题】的面试题单。返回严格 JSON："
            "{\"questions\":[{\"id\":\"原id或新id\",\"text\":\"题目(可按岗位润色/结合简历生成)\","
            "\"category\":\"分类\",\"type\":\"technical或behavioral或situational\","
            "\"rubrics\":{\"理解\":\"...\",\"表达\":\"...\",\"逻辑\":\"...\",\"完整\":\"...\"},"
            "\"followups\":[\"...\"]}]}。"
            "优先保留与岗位要求最相关的技术题；若无简历可只从池里选题；题目数不超过 "
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
    """组装本场题单（filter→rerank→个性化）。永不返回空表。"""
    max_q = int(getattr(cfg, "interview_max_questions", 5) or 5)
    top_k = max(max_q, int(getattr(cfg, "interview_recall_top_k", 8) or 8))

    pool = filter_bank(cfg, role, level)
    if not pool:
        return []

    query = _recall_query(role, level, resume_text, jd_text)
    docs = [
        f"{q.get('category') or ''} {q.get('type') or ''}：{q.get('text') or ''}"
        for q in pool
    ]
    order = await _rerank_topk(query, docs, top_n=min(top_k, len(docs)))
    ranked = [pool[i] for i in order if 0 <= i < len(pool)] or pool

    sheet = await _personalize(cfg, ranked, resume_text, jd_text, max_q)
    return sheet or [
        {k: q.get(k) for k in ("id", "text", "category", "type", "rubrics", "followups")}
        for q in ranked[:max_q]
    ]