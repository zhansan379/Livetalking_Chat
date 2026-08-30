###############################################################################
#  模拟面试 · 评分：每题 LLM judge + 终局汇总报告
#
#  每题：async_call_llm(use_json=True) 判 {score 0-10, dimension_notes{理解/表达/逻辑/完整},
#  comment}，rubrics 作锚点。终局：汇总全部作答再调一次 LLM 生成
#  {summary, strengths[], improvements[], suggested_topics[]}。
#  任一环节失败降级为保守/空态，绝不让能力崩掉主流程。
###############################################################################

import json
import re

from utils.logger import logger

DIMENSIONS = ("理解", "表达", "逻辑", "完整")

# 对话段整段评分用维度（离散段沿用上面 4 维）
_SECTION_DIMS = {
    "self_intro": ("表达", "结构", "匹配"),
    "reverse_qa": ("提问质量",),
}


def _extract_json_dict(text: str) -> dict:
    """从模型输出稳妥取 JSON 对象；失败返回 {}。"""
    text = (text or "").strip()
    try:
        p = json.loads(text)
        return p if isinstance(p, dict) else {}
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            p = json.loads(m.group(0))
            return p if isinstance(p, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


async def score_answer(cfg, question: dict, answer: str) -> dict:
    """判一题：返回 {question_id, score, dimension_notes, comment}。"""
    from infra_ai import async_call_llm
    rubrics = question.get("rubrics") or {}
    rubric_lines = "\n".join(f"  {d}: {rubrics.get(d, '')}" for d in DIMENSIONS) or "  (无)"
    prompt_msgs = [
        {"role": "system", "content": (
            "你是面试评分官。依据题目自带的评分锚点（rubrics），给候选人答案打分。\n"
            "维度固定四项：理解 / 表达 / 逻辑 / 完整，每项 0-10；score 为总分（0-10）。\n"
            f"评分锚点：\n{rubric_lines}\n"
            "返回严格 JSON：{\"score\":0-10,\"dimension_notes\":{\"理解\":0,\"表达\":0,\"逻辑\":0,\"完整\":0},"
            "\"comment\":\"一句话中肯点评\"}。只输出 JSON。"
        )},
        {"role": "user", "content": f"题目：{question.get('text')}\n候选人的回答：{answer}"},
    ]
    try:
        raw = await async_call_llm(
            prompt_msgs, use_json=True,
            extra={"kind": "interview_eval_question", "qid": question.get("id")},
            model_kwargs={"max_tokens": 512},
        )
        d = _extract_json_dict(raw)
        dims = d.get("dimension_notes") or {}
        try:
            score = min(10, max(0, float(d.get("score", 6))))
        except (TypeError, ValueError):
            score = 6.0
        notes = {k: dims.get(k) for k in DIMENSIONS}
        return {
            "question_id": question.get("id"),
            "score": round(score, 1),
            "dimension_notes": notes,
            "comment": (d.get("comment") or "")[:300],
        }
    except Exception as e:  # noqa: BLE001 - 评分失败给保守默认，不崩
        logger.warning("interview score_answer failed: %s", e)
        return {
            "question_id": question.get("id"),
            "score": 6.0,
            "dimension_notes": {},
            "comment": "（该题评分失败，已按保守分值计入）",
        }


async def score_section(cfg, section: dict, transcript: str) -> dict:
    """对话段整段评分：按环节维度判分一次。返回与 score_answer 同 shape。"""
    stype = (section or {}).get("type") or ""
    dims = _SECTION_DIMS.get(stype, DIMENSIONS)
    name = (section or {}).get("name") or stype
    dim_desc = "、".join(dims)
    from infra_ai import async_call_llm
    prompt_msgs = [
        {"role": "system", "content": (
            f"你是面试评分官。候选人在「{name}」这一环节的整段表现记录如下，"
            f"请按维度（{dim_desc}）逐项 0-10 打分，并给总分 score(0-10) 与一句话点评。"
            "返回严格 JSON：{\"score\":0-10,\"dimension_notes\":{"
            + ",".join(f"\"{d}\":0" for d in dims) + "},\"comment\":\"...\"}。只输出 JSON。"
        )},
        {"role": "user", "content": f"环节表现记录：\n{(transcript or '').strip()[:3000]}"},
    ]
    try:
        raw = await async_call_llm(
            prompt_msgs, use_json=True,
            extra={"kind": "interview_eval_section", "type": stype},
            model_kwargs={"max_tokens": 512},
        )
        d = _extract_json_dict(raw)
        dims_o = d.get("dimension_notes") or {}
        try:
            score = min(10, max(0, float(d.get("score", 6))))
        except (TypeError, ValueError):
            score = 6.0
        return {
            "question_id": f"section-{stype}",
            "score": round(score, 1),
            "dimension_notes": {k: dims_o.get(k) for k in dims},
            "comment": (d.get("comment") or "")[:300],
        }
    except Exception as e:  # noqa: BLE001 - 失败给保守默认，不崩
        logger.warning("interview score_section failed: %s", e)
        return {
            "question_id": f"section-{stype}",
            "score": 6.0,
            "dimension_notes": {},
            "comment": "（该环节评分失败，已按保守分值计入）",
        }


def _per_section_avg(answers: list) -> list:
    """逐段汇总：{type,name,score,comment}。对话段整段一条，离散段取段内均值。"""
    sec_map: dict[str, dict] = {}
    for a in answers:
        t = a.get("section_type") or "通用"
        e = sec_map.setdefault(t, {"name": a.get("section_name") or t, "scores": [], "comments": []})
        ev = a.get("eval") or {}
        sc = ev.get("score")
        if sc is not None:
            e["scores"].append(sc)
        cm = (ev.get("comment") or "").strip()
        if cm:
            e["comments"].append(cm)
    rows = []
    for t, e in sec_map.items():
        scs = e["scores"]
        rows.append({
            "type": t,
            "name": e["name"],
            "score": round(sum(scs) / len(scs), 1) if scs else None,
            "comment": "；".join(e["comments"][:3]) if e["comments"] else "",
        })
    return rows


async def build_report(cfg, sections, answers: list, role, level,
                       jd_text: str | None) -> dict:
    """终局汇总：按环节分段拼 transcript 给 LLM；失败给确定性汇总。"""
    # 先算确定性汇总（永不失败）
    evals = [a.get("eval") or {} for a in answers]
    scores = [e.get("score", 0) for e in evals if e.get("score") is not None]
    # 整体 4 维护：只统计含 4 维（离散段）作答，对话段不同维度天然被 key 过滤
    dim_avg = {}
    for d in DIMENSIONS:
        vals = [e.get("dimension_notes", {}).get(d) for e in evals
                if e.get("dimension_notes", {}).get(d) is not None]
        dim_avg[d] = round(sum(vals) / len(vals), 1) if vals else None
    section_rows = _per_section_avg(answers)
    base_report = {
        "summary": f"本场共 {len(answers)} 个作答，总分均值 {round(sum(scores) / len(scores), 1) if scores else 0}/10。",
        "dimension_avg": dim_avg,
        "sections": section_rows,
        "strengths": [], "improvements": [], "suggested_topics": [],
    }

    if not answers:
        return base_report

    from infra_ai import async_call_llm
    # 按环节分组拼 transcript
    groups: dict[str, list] = {}
    for a in answers:
        groups.setdefault(a.get("section_type") or "通用", []).append(a)
    transcript = "\n\n".join(
        f"【环节：{g.get('section_name') or t}】\n" + "\n".join(
            f"  {a.get('question', {}).get('text', '')}\n"
            f"    答：{str(a.get('answer', '')).strip()[:400]}\n"
            f"    得分+点评：{str((a.get('eval') or {}).get('comment', '')).strip()[:200]}"
            for a in g)
        for t, g in groups.items()
    )
    prompt_msgs = [
        {"role": "system", "content": (
            "你是模拟面试的复盘教练。根据整场按环节分段的作答记录，产出一份精炼报告。"
            "返回严格 JSON：{\"summary\":\"两三句总评已掌握/欠缺\",\"strengths\":[\"...\"],"
            "\"improvements\":[\"针对性改进\"],\"suggested_topics\":[\"建议继续学习的主题/面试前准备\"]},"
            "\"sections\":[{\"name\":\"环节名\",\"comment\":\"该环节一句点评\",\"score\":0-10}]}。"
            "只输出 JSON。"
        )},
        {"role": "user", "content": (
            f"岗位方向：{role or '通用'} · 难度：{level or '自适应'}\n"
            f"岗位要求（如有）：{(jd_text or '').strip()[:800]}\n\n逐环节记录：\n{transcript}"
        )},
    ]
    try:
        raw = await async_call_llm(
            prompt_msgs, use_json=True,
            extra={"kind": "interview_eval_report"},
            model_kwargs={"max_tokens": 1024},
        )
        d = _extract_json_dict(raw)
        # LLM 的逐环节点评覆盖确定性均值；score 缺失时保留均值
        llm_secs = d.get("sections") if isinstance(d.get("sections"), list) else []
        by_name = {str(s.get("name")): s for s in llm_secs if isinstance(s, dict)}
        merged_rows = []
        for r in section_rows:
            upd = by_name.get(r["name"]) or {}
            merged_rows.append({
                **r,
                "comment": str(upd.get("comment") or r["comment"]),
                "score": (float(upd["score"]) if upd.get("score") is not None else r["score"]),
            })
        return {
            **base_report,
            "summary": (d.get("summary") or base_report["summary"]).strip(),
            "strengths": [str(x) for x in d.get("strengths", [])][:6],
            "improvements": [str(x) for x in d.get("improvements", [])][:6],
            "suggested_topics": [str(x) for x in d.get("suggested_topics", [])][:8],
            "sections": merged_rows or section_rows,
        }
    except Exception as e:  # noqa: BLE001 - 报告失败退确定性汇总
        logger.warning("interview build_report failed, use deterministic: %s", e)
        return base_report