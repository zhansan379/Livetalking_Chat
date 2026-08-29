###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
#  纯路由层：业务/对话编排见 agent/chat.py，统一响应助手见 server/responses.py。
###############################################################################

import asyncio
from aiohttp import web

from utils.logger import logger

from server.responses import json_ok, json_error
from server.session_manager import session_manager
from server.avatar_routes import setup_avatar_routes
from agent.chat import stream_llm_chat, notify_reply_start
from agent.reminder import reminder_manager


def get_session(request, sessionid: str):
    """从 app 中获取 session 实例"""
    return session_manager.get_session(sessionid)


# ─── 路由处理函数 ──────────────────────────────────────────────────────────

async def human(request):
    """文本输入（echo/chat 模式），支持 voice/emotion 参数"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get('interrupt'):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get('tts'):  # tts 参数透传（voice, emotion 等）
            datainfo['tts'] = params.get('tts')

        if params['type'] == 'echo':
            notify_reply_start(avatar_session)
            avatar_session.put_msg_txt(params['text'], datainfo)
        elif params['type'] == 'chat':
            # 后台流式消费 infra_ai，避免阻塞 /human 响应（与旧 executor 语义一致）
            # trace_id：浏览器 echo 的 ASR 回合 id → chat 段复用，拼成一条全链路 trace
            asyncio.create_task(
                stream_llm_chat(avatar_session, sessionid, params['text'], datainfo,
                                trace_id=params.get('trace_id'))
            )

        return json_ok()
    except Exception as e:
        logger.exception('human route exception:')
        return json_error(str(e))


async def interrupt_talk(request):
    """打断当前说话"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception('interrupt_talk exception:')
        return json_error(str(e))


async def humanaudio(request):
    """上传音频文件"""
    try:
        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, datainfo)
        return json_ok()
    except Exception as e:
        logger.exception('humanaudio exception:')
        return json_error(str(e))


async def set_audiotype(request):
    """设置自定义状态（动作编排）"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params['audiotype'])
        return json_ok()
    except Exception as e:
        logger.exception('set_audiotype exception:')
        return json_error(str(e))


async def record(request):
    """录制控制"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params['type'] == 'start_record':
            avatar_session.start_recording()
        elif params['type'] == 'end_record':
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception('record exception:')
        return json_error(str(e))


async def is_speaking(request):
    """查询是否正在说话"""
    params = await request.json()
    sessionid = params.get('sessionid', '')
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())

async def sse_handler(request):
    """SSE 事件流，推送服务器状态更新到客户端"""
    sessionid = request.query.get('sessionid', '')
    avatar_session = session_manager.get_session(sessionid)
    if avatar_session is None:
        return json_error("session not found")

    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
        }
    )
    await response.prepare(request)

    import queue
    msgqueue = queue.Queue()
    avatar_session.add_msgqueue(msgqueue)

    try:
        while True:
            try:
                msg = msgqueue.get_nowait()
                await response.write(f"data: {msg}\n\n".encode('utf-8'))
            except queue.Empty:
                await asyncio.sleep(0.01)
    except (asyncio.CancelledError, ConnectionResetError):
        logger.info('SSE connection closed for session: %s', sessionid)
    finally:
        if msgqueue in avatar_session.msgqueues:
            avatar_session.msgqueues.remove(msgqueue)

    return response


async def admin_config(request):
    """Admin: 获取全局配置参数"""
    try:
        opt = request.app.get("opt")
        if opt:
            return json_ok(data={"config": vars(opt)})
        return json_error("Config not found")
    except Exception as e:
        logger.exception('admin_config exception:')
        return json_error(str(e))


async def admin_sessions(request):
    """Admin: 获取活跃的会话及其配置"""
    try:
        sessions_info = []
        for sid, avatar_session in session_manager.sessions.items():
            if avatar_session:
                s_opt = getattr(avatar_session, 'opt', None)
                s_data = {
                    "sessionid": sid,
                    "speaking": avatar_session.is_speaking() if hasattr(avatar_session, 'is_speaking') else False,
                    "recording": getattr(avatar_session, 'recording', False),
                }
                if s_opt:
                    s_data.update({
                        "model": getattr(s_opt, "model", ""),
                        "avatar_id": getattr(s_opt, "avatar_id", ""),
                        "REF_FILE": getattr(s_opt, "REF_FILE", ""),
                        "transport": getattr(s_opt, "transport", ""),
                        "batch_size": getattr(s_opt, "batch_size", 0),
                        "customopt": getattr(s_opt, "customopt", []),
                    })
                sessions_info.append(s_data)
        return json_ok(data={"sessions": sessions_info})
    except Exception as e:
        logger.exception('admin_sessions exception:')
        return json_error(str(e))


# ─── 全局定时提醒管理接口 ───────────────────────────────────────────────────

async def api_reminders_list(request):
    """列出当前所有定时提醒（结构化 JSON）。"""
    try:
        rows = reminder_manager.records()
        return json_ok(data={"reminders": rows, "count": len(rows)})
    except Exception as e:
        logger.exception('list reminders exception:')
        return json_error(str(e))


async def api_reminders_create(request):
    """创建一条定时提醒：cron（重复）或 delay_seconds（一次性）+ content。"""
    try:
        params = await request.json()
        content = str(params.get('content', '')).strip()
        if not content:
            return json_error("content required")
        task = str(params.get('task', '')).strip() or content
        cron = str(params.get('cron', '')).strip()
        if cron:
            rid = reminder_manager.schedule_cron(cron, content, task)
        else:
            delay = int(params.get('delay_seconds', 0) or 0)
            if delay <= 0:
                return json_error("need cron or positive delay_seconds")
            rid = reminder_manager.schedule_delay(delay, content, task)
        return json_ok(data={"reminder_id": rid})
    except ValueError as e:
        return json_error(str(e))
    except Exception as e:
        logger.exception('create reminder exception:')
        return json_error(str(e))


async def api_reminders_cancel(request):
    """按 reminder_id 取消一条提醒。"""
    try:
        params = await request.json()
        rid = str(params.get('reminder_id', '')).strip()
        if not rid:
            return json_error("reminder_id required")
        ok = reminder_manager.cancel(rid)
        if ok:
            return json_ok(data={"cancelled": True})
        return json_error(f"reminder {rid} not found")
    except Exception as e:
        logger.exception('cancel reminder exception:')
        return json_error(str(e))


# ─── 路由注册 ──────────────────────────────────────────────────────────────

async def index(request):
    """默认首页重定向"""
    opt = request.app.get("opt")
    pagename = 'avatar-chat.html'
    if opt and opt.transport == 'rtmp':
        pagename = 'rtmpapi.html'
    elif opt and opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    raise web.HTTPFound(f'/{pagename}')


def setup_routes(app):
    """注册所有路由到 aiohttp app"""
    app.router.add_get("/", index)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_get('/sse', sse_handler)

    # ── 全局定时提醒管理 ──
    app.router.add_get('/api/reminders', api_reminders_list)
    app.router.add_post('/api/reminders', api_reminders_create)
    app.router.add_post('/api/reminders/cancel', api_reminders_cancel)

    # ── Local ASR endpoint (SenseVoice/FunASR) ── Issue #604 ──
    try:
        from server.asr_server import asr_websocket_handler, is_funasr_available
        if is_funasr_available():
            app.router.add_get("/api/asr", asr_websocket_handler)
            logger.info("[ASR] Local SenseVoice ASR endpoint enabled at /api/asr")
        else:
            logger.info("[ASR] funasr not installed — local ASR endpoint disabled "
                        "(pip install funasr modelscope)")
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # ── 观测平台 /api/obs/* ──
    try:
        from obs import setup_routes as _setup_obs
        _setup_obs(app)
    except Exception as e:
        logger.warning("obs routes registration failed: %s", e)

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)

    app.router.add_static('/', path='web')