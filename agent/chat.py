###############################################################################
#  agent 域对话编排 — 从 server/routes.py 拆出的业务逻辑
#  负责：LLM 流式/工具循环问答、切句推送 TTS、LLM 语气探测、长期记忆、trace 闭环。
#  只接收 avatar_session 参数（不反向依赖 server.session_manager），避免循环 import。
###############################################################################

import json
import asyncio
import re
from utils.logger import logger


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


async def _probe_tone(agent, message: str) -> dict:
    """让 LLM 按最近对话判定数字人说话语气 → tts 语气字段（{} = 无特别情绪）。

    这是一个廉价的前置"语气探测器"：只输出简短 JSON，作为 datainfo['tts'] 的侧信道
    补充（context_texts 语音指令 / pitch / speech_rate / loudness_rate）。
    - 手动传入的 datainfo['tts'] 键优先级更高（上层用 setdefault 合并，不覆盖已存在键）；
    - LLM 判定"无特别情绪"或任何探测失败 → 返回 {}，保持全局 doubao_tone 默认，绝不
      阻塞/劣化正常回复。
    """
    ctx = (agent.recent_raw_window(rounds=4) or "").strip()
    prompt = (
        "你是数字人主播的语气导演。根据下面的最近对话，判断此刻说话应有的语气，"
        "只输出一个 JSON 对象（不要输出任何其它文本）。可选键：\n"
        "- context_texts: 数组，自然语言语气指令，如 [\"用温柔平和的语气\"]、[\"用激动兴奋的语气\"]\n"
        "- pitch: 整数 [-12,12]（>0 更明亮高昂，<0 更低沉）\n"
        "- speech_rate: 整数 [-50,100]（100=2倍速）\n"
        "- loudness_rate: 整数 [-50,100]（100=2倍音量）\n"
        "若对话没有明显情绪，返回 {\"context_texts\": []}。\n\n"
        f"最近对话：\n{ctx}\n\n用户刚刚说：{message}"
    )
    try:
        from infra_ai import async_call_llm
        raw = await async_call_llm(
            [{"role": "user", "content": prompt}], use_json=True,
            extra={"kind": "tone_probe"}, capability="chat_tone",
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
        return out
    except Exception:  # noqa: BLE001 - 语气探测失败不影响回复
        logger.exception("tone probe skipped")
        return {}


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
    # 插口②：把启用能力的目录 + 当前会话激活态片段注入 system prompt（只读；无内容则跳过）
    try:
        _cap_blk = capability_system_block(session_id, cfg)
        if _cap_blk:
            messages.append({"role": "system", "content": _cap_blk})
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

    # ── LLM 语气钩子（LLM 自动驱动的语气调整）────────────────────────────
    # 仅 doubao TTS 且 doubao_tone.llm_tone 开启时生效：前置一次廉价语气探测，
    # 把 LLM 判定的语气指令填进 datainfo['tts']（手动传入的键优先 setdefault），
    # 让整段回复从第一句起就带上对应语气。任何失败静默，不影响正常回复。
    try:
        _opt = getattr(avatar_session, "opt", None)
        _tone_cfg = getattr(_opt, "doubao_tone", None)
        if (_opt and getattr(_opt, "tts", "") == "doubao"
                and isinstance(_tone_cfg, dict) and _tone_cfg.get("llm_tone")):
            _tone = await _probe_tone(agent, message)
            if _tone:
                datainfo.setdefault("tts", {})
                for _k, _v in _tone.items():
                    datainfo["tts"].setdefault(_k, _v)
    except Exception as e:  # noqa: BLE001 - 语气钩子失败不影响回复
        logger.exception("llm tone hook exception: %s", e)

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