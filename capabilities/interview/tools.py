###############################################################################
#  模拟面试 · 工具 handlers（模型在 run_tool_loop 内调用）
#
#  确定性逻辑（评分/题数/收尾判定）全在 handler：即使模型偶尔出戏，流程仍由
#  handler 兜住。每个 handler 用会话锁串行化状态读写。
#
#  工具名用 "interview.<tool>" 前缀与全局工具天然隔离。
###############################################################################

import asyncio
import json
import os
import tempfile
import time

from utils.logger import logger

from capabilities.interview.state import (
    InterviewState, DEFAULT_SECTIONS, SECTION_TYPES, DIALOGUE_TYPES,
)
from capabilities.interview.eval import score_answer, score_section, build_report
from capabilities.interview.recall import build_section

# 每会话一把锁（进程内缓存），保证同会话 handler 串行写状态
_session_locks: dict[str, asyncio.Lock] = {}

def _lock(sid: str) -> asyncio.Lock:
    return _session_locks.setdefault(sid, asyncio.Lock())


# ── 环节解析 / 进入 / 推进（取代旧的单题单指针）───────────────────────
_SECTION_NAMES = {"self_intro": "自我介绍", "project": "项目问答",
                  "trivia": "八股文", "reverse_qa": "反问"}


def _resolve_sections(cfg) -> list[dict]:
    """把配置的 sections 规范化为 {type,name,count?,dialogue:bool}；坏 type 丢弃。"""
    raw = getattr(cfg, "interview_sections", None) or []
    if not raw:
        raw = (cfg.capabilities.get("interview") or {}).get("sections") or DEFAULT_SECTIONS
    out = []
    for it in raw:
        if not isinstance(it, dict):
            continue
        stype = str(it.get("type") or "").strip()
        if stype not in SECTION_TYPES:
            continue
        out.append({
            "type": stype,
            "name": str(it.get("name") or _SECTION_NAMES.get(stype, stype)),
            "count": int(it.get("count") or 0) or None,
            "dialogue": stype in DIALOGUE_TYPES,
        })
    return out or [dict(x) for x in DEFAULT_SECTIONS]


async def _enter_section(st: InterviewState, cfg) -> str:
    """对当前环节建 items + idx=0 + save，返回该段开场话术。"""
    sec = st.current_section()
    idx = int(st.get("section_idx") or 0)
    total = len(st.get("sections") or [])
    if st.is_dialogue():
        await st.save({"items": [], "idx": 0})
        body = ("请先做个 1-2 分钟自我介绍（说说经历、技能亮点、为什么适合这个岗位）。"
                if sec.get("type") == "self_intro" else
                "现在轮到你问我了：就这份岗位/业务关心什么都可以问我，我作为面试官作答。")
        return f"——第 {idx + 1}/{total} 环节·{sec.get('name')}——\n{body}\n（聊完说『进入下一环节』自动评分并推进）"
    count = sec.get("count") or int(getattr(cfg, "interview_max_questions", 5) or 5)
    items = await build_section(cfg, sec, st.get("role"), st.get("level"),
                                st.get("resume_text"), st.get("jd_text"))
    if not items:  # 离散段不应空；万一空给占位题避免卡住
        items = [{"id": "empty", "text": "（本环节暂无题，直接进入下一环节吧）",
                  "category": sec.get("name"), "type": "technical", "rubrics": {}, "followups": []}]
    await st.save({"items": items, "idx": 0})
    first = items[0]
    return (f"——第 {idx + 1}/{total} 环节·{sec.get('name')}——\n"
            f"第1题（{first.get('category')}）：{first.get('text')}")


async def _advance(st: InterviewState, cfg) -> str:
    """推进到下一环节；越界则终局。遇到连续对话段也逐段进入（对话段均可入）。"""
    secs = st.get("sections") or []
    ni = int(st.get("section_idx") or 0) + 1
    if ni >= len(secs):
        return await _finalize(st, st.session_id, cfg)
    await st.save({"section_idx": ni})
    return await _enter_section(st, cfg)


def _read_files(ctx, args) -> tuple[str, str]:
    """可选：从通用工具读取会话内上传的简历/JD（resume_path/jd_path）。"""
    return (args or {}).get("resume_path"), (args or {}).get("jd_path")


def _section_names_joined(sections: list) -> str:
    """环节总览文案：『自我介绍 → 项目问答 → 八股文 → 反问』。"""
    return " → ".join((s.get("name") or "") for s in sections)


def _session_file_names(cfg, ctx) -> list[str]:
    """本会话已上传的文件名列表（尽力而为；目录不存在/出错返回空，不阻塞）。

    只用于「有没有文件、叫什么」，不做内容判断——避免把非简历文件误当简历带入。
    """
    try:
        from agent.files import _session_dir
        d = _session_dir(cfg, getattr(ctx, "session_id", None) if ctx else None)
        if d and os.path.isdir(d):
            return sorted(n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n)))
    except Exception as e:  # noqa: BLE001 - 列目录失败按无文件处理
        logger.warning("interview list session files failed: %s", e)
    return []


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
    # 新环节模型：有 sections 才算真正进行中；旧态（无 sections）视为过期，直接开新场
    if st.is_active and st.get("sections"):
        qs = st.section_items()
        if qs:
            q = qs[st.inline_idx()]
            sec = st.current_section()
            return ("模拟面试正在进行中（不要再开新场）。请继续当前环节"
                    f"『{sec.get('name')}』第{st.inline_idx() + 1}题：\n{q.get('text')}")

    role = (args or {}).get("role") or getattr(cfg, "interview_default_role", None) or "通用"
    level = (args or {}).get("level") or getattr(cfg, "interview_default_level", None) or "初级"
    resume_path, jd_path = _read_files(ctx, args)
    resume_text = (args or {}).get("resume_text") or await _grab_text(ctx, (resume_path,))
    jd_text = (args or {}).get("jd_text") or await _grab_text(ctx, ("", jd_path))

    # 有上传文件但用户没指明用哪份：不瞎猜哪份是简历。首次问一次（列出候选，
    # 多份也都在），等用户点名或明说"不用"；问过仍未选则下次直接开通用场。
    files = _session_file_names(cfg, ctx)
    if not resume_text and files:
        st_now = InterviewState(cfg, sid)
        st_now.load()
        if st_now.get("pending_resume"):
            await st_now.save({"pending_resume": False})   # 已问过仍未选 → 通用场
        else:
            await st_now.save({"pending_resume": True})
            names = "、".join(f"『{n}』" for n in files)
            return (
                f"我注意到你本会话上传了：{names}。"
                "要结合哪一份作为简历或岗位要求来出题吗？直接说用哪份就行；"
                "不用的话说『不用简历，直接来』，我用通用题库开一场。"
            )
    # 带简历入场或已表态 → 清掉可能遗留的澄清标记（不影响正常开场）
    if resume_text or jd_text:
        try:
            st_now = InterviewState(cfg, sid)
            st_now.load()
            if st_now.get("pending_resume"):
                await st_now.save({"pending_resume": False})
        except Exception as e:  # noqa: BLE001 - 清标记失败不阻塞开场
            logger.warning("interview clear pending_resume failed: %s", e)

    sections = _resolve_sections(cfg)
    if not sections:
        return "（当前没有配置可用的面试环节，无法开始面试。）"

    st = InterviewState(cfg, sid)
    st.load()
    async with _lock(sid):
        await st.save({
            "status": "asking", "role": role, "level": level,
            "resume_text": resume_text or "", "jd_text": jd_text or "",
            "sections": sections, "section_idx": 0, "items": [], "idx": 0,
            "answers": [],
            "started_at": __import__("datetime").datetime.now().isoformat(),
            "finished_at": None,
        })
    overview = _section_names_joined(sections)
    opening = await _enter_section(st, cfg)
    return (f"好，我担任面试官，来一场{role} · {level}的模拟面试。整场分"
            f"{len(sections)} 个环节：{overview}。\n{opening}")


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

    if st.is_dialogue():
        # 对话段：只记 transcript 不评分，段末由 next_section 整段判分
        items = list(st.section_items()) + [{"role": "candidate", "text": answer}]
        async with _lock(sid):
            st.load()
            await st.save({"items": items, "idx": 0})
        return _dialogue_ack(st.section_type())

    qs = st.section_items()
    if not qs:
        return "（当前环节暂无题，说『进入下一环节』或让我换一环。）"
    q = qs[st.inline_idx()]
    eval_result = await score_answer(cfg, q, answer)
    async with _lock(sid):
        st.load()
        sec = st.current_section()
        answers = list(st.get("answers") or [])
        answers.append({"question": q, "answer": answer, "eval": eval_result,
                        "section_type": sec.get("type"), "section_name": sec.get("name")})
        new_idx = st.inline_idx() + 1
        await st.save({"answers": answers, "idx": new_idx})

    if new_idx >= len(qs):
        return await _advance(st, cfg)
    nxt = qs[new_idx]
    return (f"第{new_idx + 1}题（{nxt.get('category')}）：{nxt.get('text')}  "
            f"（点评会记入终局报告）")


def _dialogue_ack(stype: str) -> str:
    """对话段 answer 后的轻量确认（引导面试官 persona 继续）。"""
    if stype == "self_intro":
        return ("（已记录自我介绍。可结合候选人所述与简历继续追问；"
                "候选人想结束本环节时说『进入下一环节』。）")
    if stype == "reverse_qa":
        return ("（已记录这条提问。请作为招聘方专业作答；"
                "候选人想结束反问环节时说『进入下一环节』。）")
    return "（已记录。请继续本环节互动。）"


async def _finalize(st, sid, cfg) -> str:
    """收敛到 finished：生成按环节分段的报告 + 落盘 + 可选写一条长期记忆。"""
    sections = st.get("sections") or []
    answers = st.get("answers") or []
    report = await build_report(cfg, sections, answers,
                                st.get("role"), st.get("level"), st.get("jd_text"))
    async with _lock(sid):
        st.load()  # 刷新，避免并发写遗漏
        await st.save({"status": "finished", "finished_at": __import__("datetime").datetime.now().isoformat(),
                       "report": report})
    _maybe_remember(cfg, st, report)
    return _format_report(report, len(answers), len(answers))


def _format_report(report: dict, n_total: int, n_ans: int) -> str:
    dim = report.get("dimension_avg") or {}
    dim_s = "  ".join(f"{k}:{dim[k]}" if dim.get(k) is not None else f"{k}:-" for k in ("理解", "表达", "逻辑", "完整"))
    lines = [
        f"模拟面试结束（{n_ans}/{n_total} 个作答）。总评：{report.get('summary')}",
        f"分项均分：{dim_s}",
    ]
    secs = report.get("sections") or []
    if secs:
        lines.append("分环节：")
        for s in secs:
            sc = s.get("score")
            lines.append(f"  「{s.get('name')}」{sc if sc is not None else '-'} 分：{s.get('comment') or ''}")
    if report.get("strengths"):
        lines.append("亮点：" + "；".join(report["strengths"]))
    if report.get("improvements"):
        lines.append("建议改进：" + "；".join(report["improvements"]))
    if report.get("suggested_topics"):
        lines.append("建议继续准备：" + "、".join(report["suggested_topics"]))
    return "\n".join(lines)


def _maybe_remember(cfg, st, report: dict) -> None:
    """面试总结沉淀：长期记忆只留最新一条（同名覆盖），完整报告归档到能力层 history/。

    不再造时间戳 slug——同名 step→同固定 slug，write_memory 覆盖旧文件，所以同岗位
    长期记忆永远只有「最近一次复盘」一条，杜绝 interview-review 时间戳文件堆叠；
    完整历史报告按时间戳落到 interview/history/，由能力层保存、不占长期记忆。
    """
    try:
        role = st.get('role') or ''
        level = st.get('level') or ''
        # ① 长期记忆：只沉淀「用户欠缺」——针对性改进 + 待复习薄弱主题。
        #    问答正文/总评是一次性内容（不是用户画像），不写入长期记忆；
        #    用户稳定的知识缺口才是跨会话可用的画像。固定 name → 固定 slug，
        #    同名覆盖，每岗位只留最新一条。
        from agent.longterm import write_memory, MemoryRecord
        parts = []
        imps = report.get("improvements") or []
        topics = report.get("suggested_topics") or []
        if imps:
            parts.append("需针对性改进：" + "；".join(imps))
        if topics:
            parts.append("薄弱主题待复习：" + "、".join(topics))
        body = "。".join(parts) or "本场无明显欠缺，存档仅作记录。"
        write_memory(MemoryRecord(
            name=f"面试薄弱点({role}·{level})",
            description=f"{role}·{level} 面试暴露的知识欠缺，面试前待复习",
            type="user",
            body=body,
        ), rebuild=True)

        # ② 能力层历史归档：每场一份带时间戳的完整报告，跨场历史不丢
        base = getattr(cfg, "interview_store_dir", None) or "data/capabilities/interview"
        hist_dir = os.path.join(base, "history")
        os.makedirs(hist_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d%H%M%S")
        safe = "".join(c for c in f"{role}-{level}" if c.isalnum() or c in "._-") or "interview"
        path = os.path.join(hist_dir, f"{safe}-{ts}.json")
        doc = {
            "role": role,
            "level": level,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "report": report,
        }
        fd, tmp = tempfile.mkstemp(dir=hist_dir, prefix=".hist-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
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
    if st.is_dialogue():
        return "本环节是自由交流，直接回答即可；想结束时说『进入下一环节』。"
    qs = st.section_items()
    async with _lock(sid):
        st.load()
        new_idx = st.inline_idx() + 1
        await st.save({"idx": new_idx})
    if new_idx >= len(qs):
        return await _advance(st, cfg)
    nxt = qs[new_idx]
    return f"好，跳过。第{new_idx + 1}题（{nxt.get('category')}）：{nxt.get('text')}"


async def _hint(args, cfg, ctx=None):
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status != "asking":
        return "（当前没有进行中的模拟面试）"
    if st.is_dialogue():
        return "（本环节自由交流；若不知道聊什么，围绕岗位与自身经历展开即可。）"
    q = st.section_items()[st.inline_idx()]
    followups = q.get("followups") or []
    hint = followups[0] if followups else (q.get("brief") or "（从背景、难点、方案、结果、复盘五个角度展开。）")
    return f"提示：{q.get('text')}。不妨想想：{hint}"


async def _next_section(args, cfg, ctx=None):
    """对话段收尾：整段判分 → 追加一条整段 answer → 推进下一环节。"""
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status != "asking":
        return "（当前没有进行中的模拟面试）"
    if not st.is_dialogue():
        return "当前是答题环节，答完当前题会自动进入下一环节，无需手动切换。"
    sec = st.current_section()
    items = st.section_items()
    transcript = "\n".join(f"候选人：{t.get('text', '')}" for t in items if t.get("role") == "candidate")
    eval_result = await score_section(cfg, sec, transcript)
    async with _lock(sid):
        st.load()
        answers = list(st.get("answers") or [])
        answers.append({
            "question": {"text": f"【{sec.get('name')}】这段自由交流", "type": sec.get("type")},
            "answer": transcript, "eval": eval_result,
            "section_type": sec.get("type"), "section_name": sec.get("name"),
        })
        await st.save({"answers": answers})
    return await _advance(st, cfg)


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
    """只读：面到哪个环节了。finished 后返回报告要点。"""
    sid = getattr(ctx, "session_id", None) if ctx else None
    if not sid:
        return "（当前上下文没有有效的会话）"
    st = InterviewState(cfg, sid)
    st.load()
    if st.status == "idle":
        return "目前没有进行中的模拟面试，你可以说'开始一场模拟面试'。"
    if st.status == "finished":
        rep = st.get("report") or {}
        return (f"上次模拟面试已结束：{rep.get('summary')}。想复盘或再面一场都可以告诉我。")
    secs = st.get("sections") or []
    idx = int(st.get("section_idx") or 0)
    sec = st.current_section()
    base = f"模拟面试进行中：{st.get('role')} · {st.get('level')}，第 {idx + 1}/{len(secs)} 环节·{sec.get('name')}。"
    if st.is_dialogue():
        n_turns = len(st.section_items())
        return base + f"已完成 {n_turns} 轮自由交流，可继续提问或说『进入下一环节』。"
    qs = st.section_items()
    i = st.inline_idx()
    cur = qs[i].get("text") if qs else ""
    return base + f"已答 {len(st.get('answers') or [])} 个作答，当前第{i + 1}/{len(qs)}题：{cur}"


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
            "description": "提交用户的回复。离散题环节（八股/项目）时回答当前题，答满走完自动进下一环节；对话环节（自我介绍/反问）时记录这段发言，自由交流推进。",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "description": "用户本轮要说/答的内容"}},
                "required": ["answer"],
            },
            "handler": _answer,
        },
        {
            "name": "interview.next_section",
            "description": "对话环节（自我介绍/反问）结束当前环段：对整段自由交流判分并进入下一环节。用户表达『进入下一环节/聊完了/该下个环节了/就这样吧』时调用。离散题环节无需调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "handler": _next_section,
        },
        {
            "name": "interview.skip",
            "description": "用户不想答当前这题，换一题（不出分）。仅离散题环节可用。",
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