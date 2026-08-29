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


async def build_report(cfg, questions: list, answers: list, role, level,
                       jd_text: str | None) -> dict:
    """终局汇总：把全部作答与分项汇总给 LLM 生成报告。失败给确定性汇总。"""
    # 先算确定性汇总（永不失败）
    evals = [a.get("eval") or {} for a in answers]
    scores = [e.get("score", 0) for e in evals]
    dim_avg = {}
    for d in DIMENSIONS:
        vals = [e.get("dimension_notes", {}).get(d) for e in evals if e.get("dimension_notes", {}).get(d) is not None]
        dim_avg[d] = round(sum(vals) / len(vals), 1) if vals else None
    base_report = {
        "summary": f"本场共 {len(answers)} 题，总分均值 {round(sum(scores) / len(scores), 1) if scores else 0}/10。",
        "dimension_avg": dim_avg,
        "strengths": [], "improvements": [], "suggested_topics": [],
    }

    if not answers:
        return base_report

    from infra_ai import async_call_llm
    transcript = "\n".join(
        f"Q{i + 1}（{a.get('question', {}).get('text', '')}）："
        f"\n  答：{str(a.get('answer', '')).strip()[:500]}\n  得分+点评：{str((a.get('eval') or {}).get('comment', '')).strip()[:200]}"
        for i, a in enumerate(answers)
    )
    prompt_msgs = [
        {"role": "system", "content": (
            "你是模拟面试的复盘教练。根据整场作答逐题记录，产出一份精炼报告。"
            "返回严格 JSON：{\"summary\":\"两三句总评已掌握/欠缺\",\"strengths\":[\"...\"],"
            "\"improvements\":[\"针对性改进\"],\"suggested_topics\":[\"建议继续学习的主题/面试前准备\"]}。"
            "只输出 JSON。"
        )},
        {"role": "user", "content": (
            f"岗位方向：{role or '通用'} · 难度：{level or '自适应'}\n"
            f"岗位要求（如有）：{(jd_text or '').strip()[:800]}\n\n逐题记录：\n{transcript}"
        )},
    ]
    try:
        raw = await async_call_llm(
            prompt_msgs, use_json=True,
            extra={"kind": "interview_eval_report"},
            model_kwargs={"max_tokens": 1024},
        )
        d = _extract_json_dict(raw)
        return {
            **base_report,
            "summary": (d.get("summary") or base_report["summary"]).strip(),
            "strengths": [str(x) for x in d.get("strengths", [])][:6],
            "improvements": [str(x) for x in d.get("improvements", [])][:6],
            "suggested_topics": [str(x) for x in d.get("suggested_topics", [])][:8],
        }
    except Exception as e:  # noqa: BLE001 - 报告失败退确定性汇总
        logger.warning("interview build_report failed, use deterministic: %s", e)
        return base_report