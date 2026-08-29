###############################################################################
#  模拟面试 · 工具 handlers（模型在 run_tool_loop 内调用）
#
#  确定性逻辑（评分/题数/收尾判定）全在 handler：即使模型偶尔出戏，流程仍由
#  handler 兜住。每个 handler 用会话锁串行化状态读写。
#
#  工具名用 "interview.<tool>" 前缀与全局工具天然隔离。
###############################################################################

import asyncio

from utils.logger import logger

from capabilities.interview.state import InterviewState
from capabilities.interview.eval import score_answer, build_report
from capabilities.interview.recall import build_question_sheet

# 每会话一把锁（进程内缓存），保证同会话 handler 串行写状态
_session_locks: dict[str, asyncio.Lock] = {}

def _lock(sid: str) -> asyncio.Lock:
    return _session_locks.setdefault(sid, asyncio.Lock())


def _read_files(ctx, args) -> tuple[str, str]:
    """可选：从通用工具读取会话内上传的简历/JD（resume_path/jd_path）。"""
    return (args or {}).get("resume_path"), (args or {}).get("jd_path")


async def _grab_text(ctx, paths: tuple) -> str:
    """尽力从上传文件抽取文本；失败返回空串（不阻塞）。"""
    from agent.files import _read_scoped
    from agent.config import get_agent_config
    sid = getattr(ctx, "session_id", None) if ctx else None
    cfg = get_agent_config()
    parts = []
    for p in paths:
        if p:
            status, body = _read_scoped(p, cfg, sid)
            if status == "ok":
                parts.append(body)
    return "\n".join(parts)


# ─── interview.start ────────────────────────────────────────────────────────
async def _start(args, cfg, ctx=None):
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话，无法开始模拟面试）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.is_active:
        q = st.questions[st.idx]
        return ("模拟面试正在进行中（不要再开新场）。请继续回答当前题："
                f"\n第{st.idx + 1}题：{q.get('text')}")

    role = (args or {}).get("role") or getattr(cfg, "interview_default_role", None) or "通用"
    level = (args or {}).get("level") or getattr(cfg, "interview_default_level", None) or "初级"
    resume_path, jd_path = _read_files(ctx, args)
    resume_text = (args or {}).get("resume_text") or await _grab_text(ctx, (resume_path,))
    jd_text = (args or {}).get("jd_text") or await _grab_text(ctx, ("", jd_path))

    questions = await build_question_sheet(cfg, role, level, resume_text, jd_text)
    if not questions:
        return "（当前题库没有可用题目，无法开始面试。可稍后再试。）"

    st = InterviewState(cfg, sid)
    st.load()
    async with _lock(sid):
        await st.save({
            "status": "asking", "role": role, "level": level,
            "resume_text": resume_text or "", "jd_text": jd_text or "",
            "questions": questions, "idx": 0, "answers": [],
            "started_at": __import__("datetime").datetime.now().isoformat(),
            "finished_at": None,
        })
    first = questions[0]
    return (f"好，我担任面试官，来一场{role} · {level}的模拟面试。一共"
            f"{len(questions)}题，你直接回答或说'下一题'。\n"
            f"第1题（{first.get('category')}）：{first.get('text')}")


# ─── interview.answer ───────────────────────────────────────────────────────
async def _answer(args, cfg, ctx=None):
    sid = getattr(ctx, "session_id", None) if ctx else None
    answer = ((args or {}).get("answer") or "").strip()
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status != "asking":
        return "（当前没有进行中的模拟面试。需要的话可以用 interview.start 开一场。）"
    if not answer:
        return "（请先给出你的回答，例如：'我们可以用……来实现'。）"

    q = st.questions[st.idx]
    eval_result = await score_answer(cfg, q, answer)
    async with _lock(sid):
        answers = list(st.get("answers") or [])
        answers.append({"question": q, "answer": answer, "eval": eval_result})
        new_idx = st.get("idx", 0) + 1
        await st.save({"answers": answers, "idx": new_idx})

    if new_idx >= len(st.questions):
        return await _finalize(st, sid, cfg)
    nxt = st.questions[new_idx]
    return (f"第{new_idx + 1}题（{nxt.get('category')}）：{nxt.get('text')}  "
            f"（点评会记入终局报告）")


async def _finalize(st, sid, cfg) -> str:
    """收敛到 finished：生成报告文本 + 落盘 + 可选写一条长期记忆总结。"""
    questions = st.questions
    answers = st.get("answers") or []
    report = await build_report(cfg, questions, answers,
                                st.get("role"), st.get("level"), st.get("jd_text"))
    async with _lock(sid):
        st.load()  # 刷新，避免并发写遗漏
        await st.save({"status": "finished", "finished_at": __import__("datetime").datetime.now().isoformat(),
                       "report": report})
    _maybe_remember(cfg, st, report)
    return _format_report(report, len(questions), len(answers))


def _format_report(report: dict, n_total: int, n_ans: int) -> str:
    dim = report.get("dimension_avg") or {}
    dim_s = "  ".join(f"{k}:{dim[k]}" if dim.get(k) is not None else f"{k}:-" for k in ("理解", "表达", "逻辑", "完整"))
    lines = [
        f"模拟面试结束（答 {n_ans}/{n_total} 题）。总评：{report.get('summary')}",
        f"分项均分：{dim_s}",
    ]
    if report.get("strengths"):
        lines.append("亮点：" + "；".join(report["strengths"]))
    if report.get("improvements"):
        lines.append("建议改进：" + "；".join(report["improvements"]))
    if report.get("suggested_topics"):
        lines.append("建议继续准备：" + "、".join(report["suggested_topics"]))
    return "\n".join(lines)


def _maybe_remember(cfg, st, report: dict) -> None:
    """把面试总结沉淀进长期记忆（精炼一条；失败静默）。"""
    try:
        from agent.longterm import write_memory, MemoryRecord
        body = (report.get("summary") or "") + "。"
        if report.get("suggested_topics"):
            body += "建议准备：" + "、".join(report["suggested_topics"])
        write_memory(MemoryRecord(
            name=f"模拟面试复盘({st.get('role')}·{st.get('level')})",
            description=f"{st.get('role')}·{st.get('level')} 模拟面试结果",
            type="project",
            body=body,
            slug=f"interview-review-{st.get('role', '')}-{__import__('time').strftime('%Y%m%d%H%M')}",
        ))
    except Exception as e:  # noqa: BLE001 - 记忆沉淀失败不影响主流程
        logger.warning("interview remember failed: %s", e)


# ─── interview.skip / hint / end / status ───────────────────────────────────
async def _skip(args, cfg, ctx=None):
    """换一题（不出分）：直接进入下一题；已是最后一题则提示可 end。"""
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status != "asking":
        return "（当前没有进行中的模拟面试）"
    async with _lock(sid):
        new_idx = st.get("idx", 0) + 1
        await st.save({"idx": new_idx})
    if new_idx >= len(st.questions):
        return await _finalize(st, sid, cfg)
    nxt = st.questions[new_idx]
    return f"好，跳过。第{new_idx + 1}题（{nxt.get('category')}）：{nxt.get('text')}"


async def _hint(args, cfg, ctx=None):
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status != "asking":
        return "（当前没有进行中的模拟面试）"
    q = st.questions[st.idx]
    followups = q.get("followups") or []
    hint = followups[0] if followups else "（没有预设提示，你可以从定义、原理、适用场景、反例四个角度展开。）"
    return f"提示：{q.get('text')}。不妨想想：{hint}"


async def _end(args, cfg, ctx=None):
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status != "asking":
        return "（当前没有正在进行的模拟面试）"
    return await _finalize(st, sid, cfg)


async def _status(args, cfg, ctx=None):
    """只读：面到哪了。finished 后返回报告要点。"""
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status == "idle":
        return "目前没有进行中的模拟面试，你可以说'开始一场模拟面试'。"
    n_total = len(st.get("questions") or [])
    n_ans = len(st.get("answers") or [])
    if st.status == "finished":
        rep = st.get("report") or {}
        return (f"上次模拟面试已结束：{rep.get('summary')}。想复盘或再面一场都可以告诉我。")
    q = st.get("questions") or [{}]
    idx = min(st.get("idx", 0), len(q) - 1)
    cur = q[idx] if q else {}
    return (f"模拟面试进行中：{st.get('role')} · {st.get('level')}，"
            f"已答 {n_ans}/{n_total} 题，当前第{idx + 1}题：{cur.get('text')}")


def _tools() -> list[dict]:
    """interview 能力工具集定义。"""
    return [
        {
            "name": "interview.start",
            "description": (
                "开始一场模拟面试。参数 role/level 可选（缺省用配置默认）；"
                "可将简历/岗位要求以 resume_text/jd_text 直接带入，或传 resume_path/jd_path "
                "指向本会话已上传文件（需先用 read_file 确认存在）。开启后返回第1题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "description": "岗位方向，如 前端/后端"},
                    "level": {"type": "string", "description": "难度，如 初级/中级/高级"},
                    "resume_text": {"type": "string", "description": "可选的简历文本"},
                    "jd_text": {"type": "string", "description": "可选的岗位要求文本"},
                    "resume_path": {"type": "string", "description": "可选：本会话上传的简历文件名"},
                    "jd_path": {"type": "string", "description": "可选：本会话上传的岗位要求文件名"},
                },
                "required": [],
            },
            "handler": _start,
        },
        {
            "name": "interview.answer",
            "description": "回答当前面试题。把用户这段话作为 answer 传入。答满题数或要求结束时自动出终局报告。",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "用户对当前题的回答内容"}},
                "required": ["answer"],
            },
            "handler": _answer,
        },
        {
            "name": "interview.skip",
            "description": "用户不想答当前这题，换一题（不出分）。",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "handler": _skip,
        },
        {
            "name": "interview.hint",
            "description": "用户索要当前题提示时调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "handler": _hint,
        },
        {
            "name": "interview.end",
            "description": "用户要求结束当前模拟面试时调用，生成终局报告。",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "handler": _end,
        },
        {
            "name": "interview.status",
            "description": "用户问'面到哪了'/'面试进度'时调用，只读返回进度或已结束状态。",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "handler": _status,
        },
    ]