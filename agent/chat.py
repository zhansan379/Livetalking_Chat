###############################################################################
#  agent 域对话编排 — 从 server/routes.py 拆出的业务逻辑
#  负责：LLM 流式/工具循环问答、切句推送 TTS、LLM 语气探测、长期记忆、trace 闭环。
#  只接收 avatar_session 参数（不反向依赖 server.session_manager），避免循环 import。
###############################################################################

import json
import asyncio
import re
from utils.logger import logger
from server.action_prep import list_usable_emotions


# ─── 回答清洗：只保留纯文本与标点 ──────────────────────────────────────────
# 组合一个「保留字符集」，其余一律剔除：过滤 emoji/表情符号、Markdown 记号
# ( * _ # ` ) 以及箭头、几何、装饰等非文本字符，保证 TTS 只朗读纯文本与标点。
_KEEP_RE = re.compile(
    r"[一-鿿㐀-䶿぀-ヿ가-힯"  # 中日韩、假名、谚文
    r"々〆〇"                                     # 々 〆 〇
    r"A-Za-z0-9"                                             # 字母与数字
    r"\s"
    r"，。！？；：、（）《》〈〉「」『』【】—…·"
    r",.!?;:()%\-+/<>'\""
    r"]"
)


def _sanitize(text: str) -> str:
    """把 LLM 回答清洗成纯文本+标点；若清洗后为空返回空串。"""
    if not text:
        return text
    return "".join(_KEEP_RE.findall(text)).strip()


# ─── 对话编排 ──────────────────────────────────────────────────────────

# 触发切句推送的标点
_SENTENCE_PUNCT = set(",.!;:，。！？：；")

# 工具循环触顶/异常时的应用层降级话术（区别于模型生成的答案）
_TOOL_FAIL_FALLBACK = "我这次没查到足够的信息，换个问法试试好吗？"


def _feed_talk(avatar_session, text: str, datainfo: dict, gen: int = 0):
    """把一段文本按标点切句、超过 10 字即推送给 TTS，末尾剩余部分补推。"""
    if not text:
        return
    # 代际自检：本回合已被新回合取代则立即停止喂料（避免前后两次回答叠读）
    if gen and hasattr(avatar_session, 'is_stale') and avatar_session.is_stale(gen):
        return
    buf = ""
    lastpos = 0
    for i, ch in enumerate(text):
        if ch in _SENTENCE_PUNCT:
            buf += text[lastpos:i + 1]
            lastpos = i + 1
            if len(buf) > 10:
                avatar_session.put_msg_txt(buf, datainfo)
                buf = ""
    buf += text[lastpos:]
    if buf:
        avatar_session.put_msg_txt(buf, datainfo)


def notify_reply_start(avatar_session):
    """新一轮回答开始：通知前端清空字幕，后续句子逐句追加成完整回答。"""
    avatar_session.send_msg(json.dumps({"status": "reply_start"}))


async def _probe_tone(agent, message: str, emotion_names=("happy", "surprised", "sad", "angry")) -> dict:
    """让 LLM 按最近对话判定数字人说话语气 → tts 语气字段（{} = 无特别情绪）。

    这是一个廉价的前置"语气探测器"：只输出简短 JSON，作为 datainfo['tts'] 的侧信道
    补充（context_texts 语音指令 / pitch / speech_rate / loudness_rate）。
    - emotion_names 由 data/actions 派生（只有底座且启用且绑定当前形象的表情才会被放行），
      LLM 只在这些候选内选或 neutral，避免探测出无底座的表情；
    - 手动传入的 datainfo['tts'] 键优先级更高（上层用 setdefault 合并，不覆盖已存在键）；
    - LLM 判定"无特别情绪"或任何探测失败 → 返回 {}，保持全局 doubao_tone 默认，绝不
      阻塞/劣化正常回复。
    """
    names = [n for n in (emotion_names or ()) if n and n != "neutral"]
    if not names:
        names = ["happy", "surprised", "sad", "angry"]  # 功能配置缺省时的历史候选
    EmoLabel = {"happy": "高兴", "surprised": "惊讶", "sad": "委屈难过", "angry": "生气"}
    emo_opts = "".join(f"/{n}({EmoLabel.get(n, n)})" for n in names)
    label_map = {n: EmoLabel.get(n, n) for n in names}
    ctx = (agent.recent_raw_window(rounds=4) or "").strip()
    prompt = (
        "你是数字人主播的语气与表情导演。根据下面的最近对话，判断此刻说话应有的语气与表情，"
        "只输出一个 JSON 对象（不要输出任何其它文本）。可选键：\n"
        "- context_texts: 数组，自然语言语气指令，如 [\"用温柔平和的语气\"]、[\"用激动兴奋的语气\"]\n"
        "- pitch: 整数 [-12,12]（>0 更明亮高昂，<0 更低沉）\n"
        "- speech_rate: 整数 [-50,100]（100=2倍速）\n"
        "- loudness_rate: 整数 [-50,100]（100=2倍音量）\n"
        "- emotion: 数字人直播画面此刻的表情基调，取且仅取其中之一："
        f"neutral(中性){emo_opts}。"
        "没有明显情绪一律返回 neutral。\n"
        "若对话没有明显情绪，语气键返回 {\"context_texts\": []} 且 emotion 用 neutral。\n\n"
        f"最近对话：\n{ctx}\n\n用户刚刚说：{message}"
    )
    try:
        from infra_ai import async_call_llm
        raw = await async_call_llm(
            [{"role": "user", "content": prompt}], use_json=True,
            # async=True：标记本调用为异步 span（与主问答并发，不阻塞响应主干），
            # 供观测面板用独立配色标识、并按全链路统计口径排除。
            extra={"kind": "tone_probe", "async": True}, capability="chat_tone",
        )
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        out = {}
        ctx_texts = data.get("context_texts")
        if ctx_texts:
            out["context_texts"] = ctx_texts if isinstance(ctx_texts, list) else [str(ctx_texts)]
        elif ctx_texts == []:
            pass  # LLM 明确"无情绪"→ 不覆盖全局 doubao_tone 默认
        for k in ("pitch", "speech_rate", "loudness_rate"):
            v = data.get(k)
            if v is not None:
                try:
                    out[k] = int(v)
                except (TypeError, ValueError):
                    pass
        # 表情基调：只放行候选集（与配置 emotion.names 对齐），否则丢弃回中性
        emot = str(data.get("emotion") or "").strip().lower()
        if emot == "neutral" or emot in set(names):
            out["emotion"] = emot
        return out
    except Exception:  # noqa: BLE001 - 语气探测失败不影响回复
        logger.exception("tone probe skipped")
        return {}


async def _probe_tone_into(agent, message: str, datainfo: dict, emotion_names=()):
    """并发版语气探测：先探测，再把结果填进共享 datainfo，供 TTS 线程随句读取。

    作为 asyncio.create_task 后台运行，**不阻塞**主回答的 run_tool_loop/流式。
    复用 _probe_tone 的容错（任何失败返回 {} 绝不抛），故本协程不会向主链路抛异常。
    datainfo 是贯穿 put_msg_txt→TTS 工作线程的同一 dict 引用；只要在某句实际合成前
    填好 datainfo["tts"]，该句即带上语气（尽力而为：tone 慢/失败时前句用全局默认）。
    """
    tone = await _probe_tone(agent, message, emotion_names=emotion_names)
    if tone:
        # 语气键 → datainfo['tts']（供 TTS 引擎用；非 doubao 亦无副作用）
        _tts = datainfo.setdefault("tts", {})
        for _k, _v in tone.items():
            if _k == "emotion":
                continue  # 表情是顶层视觉诉求，不下塞到 tts 语气里
            _tts.setdefault(_k, _v)
        # 表情基调 → datainfo['emotion']（顶层；随 utterance 到渲染线程并走 SSE）
        if tone.get("emotion"):
            datainfo.setdefault("emotion", tone["emotion"])


async def stream_llm_chat(avatar_session, session_id: str, message: str,
                          datainfo: dict = {}, trace_id: str | None = None,
                          upload_note: str | None = None, gen: int = 0):
    """基于 infra_ai + agent 记忆的问答：优先走工具循环，无启用工具时退回流式问答。

    - 使用 agent.ChatAgent 加载/保存完整转录，并把「历史摘要 + 最近几轮」拼进上下文；
    - 读取 agent 配置，若启用了工具（如 web_search）则用 run_tool_loop 解析成最终文本再按句说话；
      否则保持原有 async_stream_call_llm 流式消费；
    - 工具循环返回 None（触顶/异常）时说话用的是一句固定的应用层降级话术，不伪造模型回复；
    - 回复完成后开启后台异步压缩（不阻塞本次回复）。
    """
    from agent import ChatAgent
    from agent.config import get_agent_config
    from agent.tool_loop import ToolContext, build_tools, run_tool_loop
    # 插口②③：能力目录注入 + 按会话+状态条件暴露工具（chat 感知不到具体能力）
    from capabilities.hub import capability_system_block, session_tools
    from agent.longterm import (
        extract_longterm_memory,
        inject_memory_block,
        recall_longterm_memories,
    )
    from obs import begin_trace as _begin_trace, end_trace as _end_trace

    agent = ChatAgent(session_id)
    messages = agent.build_messages(message)  # 同步组装上下文，不触发压缩
    # 跨会话长期记忆：召回命中的正文注入 system prompt；失败静默返回空，不影响主问答
    try:
        _lt = await recall_longterm_memories(message)
        if _lt:
            messages = inject_memory_block(messages, _lt)
    except Exception as e:  # noqa: BLE001 - 记忆召回任何失败都不应阻断回复
        logger.exception("longterm recall inject exception: %s", e)
    agent.add_user_message(message)
    # 任务 #7：用户消息立即落盘，即使本回合中途被打断/被取代也不丢失（旧回合并行
    # save 会各自处理自己的增量；这里保证本回合用户输入立刻进入历史文件）。
    try:
        agent.save()
    except Exception as e:  # noqa: BLE001 - 落盘失败不阻断回复
        logger.exception("persist user message exception: %s", e)

    cfg = get_agent_config()
    # 插口②：把启用能力的目录 + 当前会话激活态片段注入 system prompt（只读；无内容则跳过）。
    # 固定插在第一条 system（基础提示+日期）之后，不再 append 到对话末尾——静态能力目录
    # （如"本助手提供【模拟面试】能力…"）每次垫在最新一条既占最新注意力、又无持久性；
    # 放首条 = 能力自述固定在第一系统消息区，跨所有轮次稳定可见。动态激活态（如面试当前
    # 进度）同样早插，避免与最新用户消息争抢注意力。
    try:
        _cap_blk = capability_system_block(session_id, cfg)
        if _cap_blk:
            messages.insert(1, {"role": "system", "content": _cap_blk})
    except Exception as e:  # noqa: BLE001 - 能力注入失败不应阻断主问答
        logger.exception("capability system block inject exception: %s", e)
    # 插口⑤：前端刚上传过文件 → 给本次回复注入一条临时提示（只进本次上下文、
    # 不落 agent 历史），引导模型先 list_files 核实再作答，避免凭记忆断言"没有文件"
    try:
        if upload_note and str(upload_note).strip():
            messages.append({
                "role": "system",
                "content": (
                    f"[本会话刚上传了文件：{upload_note}] 用户正就它提问。"
                    "回答任何关于本会话文件是否存在或其内容的问题前，"
                    "请先调用 list_files 确认文件名，需要内容再调用 read_file；"
                    "不要凭记忆断言没有文件。"
                ),
            })
    except Exception as e:  # noqa: BLE001 - 提示注入失败不应阻断主问答
        logger.exception("upload_note inject exception: %s", e)
    # 插口③：按会话+状态条件暴露工具子集（能力工具仅进行中相关子集；全局工具照常）
    # 名字先捕住，供 trace_start 记录"本请求绑定了哪些工具"（能力按态注入的名单）。
    _tool_names = session_tools(session_id, cfg)
    tools = build_tools(_tool_names)
    # trace_id：复用来自 ASR 服务端下发的回合 id（浏览器 echo），从而把
    # ASR→LLM/工具→TTS 拼成一条 trace；缺省则自旋一条（独立 trace，向后兼容）。
    _tid = _begin_trace(session_id, (message or "")[:200], tool_mode=bool(tools),
                        trace_id=trace_id, bound_tools=_tool_names)
    # 把 trace 身份蹭进 datainfo：沿 put_msg_txt 传到 TTS 工作线程，
    # 供基类 process_tts 用 emit_explicit 显式挂回本聊天 trace（线程拿不到 contextvars）。
    if _tid:
        datainfo["_obs"] = {"trace_id": _tid, "session_id": session_id, "parent_id": _tid}
    notify_reply_start(avatar_session)  # 新一轮回答开始，前端清空字幕后逐句追加

    # ── LLM 语气/表情钩子（LLM 自动驱动的语气调整 + 表情编排）─────────────
    # 触发条件（满足任一即探测）：doubao TTS 且 doubao_tone.llm_tone 开启（供 TTS 语气）；
    # 或 当前形象存在可用表情底座（list_usable_emotions 派生 data/actions 的 manifest，
    # ——纯视觉诉求，不该被 TTS 引擎绑架）。
    # 探测结果填共享 datainfo：语气键进 datainfo['tts']，表情基调进 datainfo['emotion']。
    # 并发执行：以 asyncio.create_task 启动，后台随句填结果，**不阻塞**主回答；
    # 任务内部已 try/except 兜底，绝不向主链路抛异常，fire-and-forget 由回调兜底记录。
    try:
        _opt = getattr(avatar_session, "opt", None)
        _tone_cfg = getattr(_opt, "doubao_tone", None) or {}
        _avatar_id = getattr(_opt, "avatar_id", "")
        # 候选表情 = 对当前形象可用的表情底座（自动筛 manifest.enabled + 绑定）。
        _emo_names = tuple(list_usable_emotions(_avatar_id))
        _want_emotion = bool(_emo_names)
        _want_tone = (_opt and getattr(_opt, "tts", "") == "doubao"
                      and isinstance(_tone_cfg, dict) and _tone_cfg.get("llm_tone"))
        if _want_tone or _want_emotion:
            # 表情关闭时传哨兵候选：避免 _probe_tone 空候选回落到默认四情绪而误带 emotion。
            _emo_arg = _emo_names if _want_emotion else ("__none__",)
            _tone_task = asyncio.create_task(
                _probe_tone_into(agent, message, datainfo, emotion_names=_emo_arg))
            _tone_task.add_done_callback(
                lambda t: t.exception() if not t.cancelled() else None)
    except Exception as e:  # noqa: BLE001 - 探测钩子失败不影响回复
        logger.exception("llm tone/emotion hook exception: %s", e)

    reply = None
    tr_success = True
    tr_fail = None
    try:
        if tools:
            final = await run_tool_loop(messages, tools, cfg,
                                        ctx=ToolContext(session_id=session_id))
            final = _sanitize(final) if final else final
            if final:
                reply = final
                _feed_talk(avatar_session, final, datainfo, gen)
            else:
                logger.warning("tool loop returned None/empty after sanitize, use fallback phrase")
                tr_success = False
                tr_fail = "tool_loop_max_rounds"
                _feed_talk(avatar_session, _TOOL_FAIL_FALLBACK, datainfo, gen)
        else:
            # 无启用工具 → 保留原有流式逐字消费。缓冲区跨 token 累积，
            # 按标点切句、超过 10 字即推送，末尾剩余部分补推。
            # 推送前与收尾时统一过 _sanitize，保证朗读内容不含表情/markdown 字符。
            from infra_ai import async_stream_call_llm
            full_reply = []
            buf = ""
            try:
                async for token in async_stream_call_llm(messages, purpose="chat_reply"):
                    # 代际自检：本回合已被新回合取代 → 立即放弃，不再喂料/拼回复
                    if gen and hasattr(avatar_session, 'is_stale') \
                            and avatar_session.is_stale(gen):
                        full_reply = []
                        buf = ""
                        break
                    if not token:
                        continue
                    full_reply.append(token)
                    lastpos = 0
                    for i, ch in enumerate(token):
                        if ch in _SENTENCE_PUNCT:
                            buf += token[lastpos:i + 1]
                            lastpos = i + 1
                            if len(buf) > 10:
                                s = _sanitize(buf)
                                if s and not avatar_session.is_stale(gen):
                                    avatar_session.put_msg_txt(s, datainfo)
                                buf = ""
                    buf += token[lastpos:]
                if buf:
                    s = _sanitize(buf)
                    if s and not avatar_session.is_stale(gen):
                        avatar_session.put_msg_txt(s, datainfo)
                reply = _sanitize("".join(full_reply)).strip()
            except Exception as e:
                logger.exception("infra_ai chat exception: %s", e)
                tr_success = False
                tr_fail = "llm_error"

        # 代际自检：本回合已被取代则整体放弃（不写 assistant、不落盘），
        # 避免旧回合覆盖掉最新回合刚写入的历史。
        if gen and hasattr(avatar_session, 'is_stale') and avatar_session.is_stale(gen):
            reply = None

        if reply:
            agent.add_assistant_message(reply)
            # 跨会话长期记忆提取：后台异步，不阻塞本次回复、不抛异常。
            # 带上最近几轮原文上下文，让提取器能识别"自介/身份"类陈述。
            try:
                _ctx = agent.recent_raw_window()
                asyncio.create_task(extract_longterm_memory(session_id, message, reply, _ctx))
            except Exception as e:  # noqa: BLE001
                logger.exception("longterm extract trigger exception: %s", e)

        # 完整转录先落盘，再后台异步压缩（不阻塞本次回复）。
        # 代际自检：本回合已被取代则整体跳过落盘——否则旧回合会用它那份
        # 不含新消息的内存历史覆盖掉新回合刚写入的磁盘，造成丢失更新。
        if not (gen and hasattr(avatar_session, 'is_stale')
                and avatar_session.is_stale(gen)):
            try:
                agent.save()
                if agent.should_compress():
                    asyncio.create_task(agent.compress_and_save())
            except Exception as e:
                logger.exception("agent save/compress trigger exception: %s", e)
    finally:
        _end_trace(tr_success, fail_reason=tr_fail, text_len=len(reply or ""))