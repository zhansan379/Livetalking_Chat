###############################################################################
#  服务器路由 — 统一异常处理的 API 路由
#  纯路由层：业务/对话编排见 agent/chat.py，统一响应助手见 server/responses.py。
###############################################################################

import asyncio
import base64
import json
import os
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
            # 代际：新回合 begin_chat 升位，取消上一个回合任务，旧回合在 chat.py 内
            # 用 is_stale 自检放弃喂料/落盘，避免并发回合互相覆盖历史与叠读。
            gen = avatar_session.begin_chat()
            task = asyncio.create_task(
                stream_llm_chat(avatar_session, sessionid, params['text'], datainfo,
                                trace_id=params.get('trace_id'),
                                upload_note=params.get('upload_note'),
                                gen=gen)
            )
            avatar_session.attach_task(gen, task)

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


async def resume_talk(request):
    """补播最近一次被打断未播完的内容（前端在「打断后转写为空=假打断」时调用）。"""
    try:
        params = await request.json()
        sessionid = params.get('sessionid', '')
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        resumed = bool(avatar_session.resume_talk())
        if resumed:
            notify_reply_start(avatar_session)
        return json_ok(data={"resumed": resumed})
    except Exception as e:
        logger.exception('resume_talk exception:')
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


async def upload_file(request):
    """通用文件上传（插口④）：multipart 接收 {sessionid, file} → data/uploads/<sessionid>/。

    能力无关暂存区，会话共享、一份副本多方可读；由 agent 通用工具 list_files/
    read_file 在会话范围内读取。仅暂存，生命周期/配额属后续里程碑。
    """
    try:
        from agent.files import _session_dir, sanitize_name
        from agent.config import get_agent_config

        form = await request.post()
        sessionid = str(form.get('sessionid', ''))
        # 仅保留安全字符 + 目录界定——无有效会话则拒绝（防路径穿越/越权目录）
        sid_dir = _session_dir(get_agent_config(), sessionid)
        if sid_dir is None:
            return json_error("invalid sessionid")

        fileobj = form['file']
        fname = sanitize_name(str(fileobj.filename or 'file'))

        os.makedirs(sid_dir, exist_ok=True)
        dest = os.path.join(sid_dir, fname)
        data = fileobj.file.read()
        with open(dest, 'wb') as f:
            f.write(data)

        from utils.logger import logger as _log
        _log.info("[files] uploaded %s (%d bytes) -> data/uploads/%s/", fname, len(data), sessionid)
        return json_ok({'path': f'{sessionid}/{fname}', 'size': len(data)})
    except Exception as e:
        logger.exception('upload_file exception:')
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


# ─── 摄像头「看用户」WebSocket ─────────────────────────────────────────────
# 浏览器开摄像头开关时连到 /api/camera/ws?session=<id>；服务端在数字人被
# 触发 look_at_user 工具时推 {type:'capture_request'}，浏览器现场抓一帧回传
# {type:'snapshot', data:'data:image/jpeg;base64,...'} → 交给 CameraService。
def _extract_jpeg(data_url_or_b64: str) -> bytes | None:
    """从 data URL 或纯 base64 中取出 JPEG 字节；非法返回 None。"""
    s = (data_url_or_b64 or "").strip()
    if not s:
        return None
    if "," in s:
        _, _, s = s.partition(",")  # 剥掉 "data:image/jpeg;base64," 前缀
    try:
        raw = base64.b64decode(s, validate=False)
    except Exception:  # noqa: BLE001 - 坏 base64 按非法帧丢弃
        return None
    return raw or None


async def camera_websocket_handler(request):
    """浏览器摄像头通道：按需接收一帧 JPEG，交给同 session 的 look_at_user 工具。"""
    session_id = (request.query.get("session") or "").strip() or "0"
    ws = web.WebSocketResponse(heartbeat=30)
    await ws.prepare(request)
    try:
        from agent.camera import camera_service
    except Exception as e:  # noqa: BLE001 - 依赖缺失时拒绝而非崩掉服务
        logger.warning("[camera] camera_service 不可用: %s", e)
        await ws.close()
        return ws
    camera_service.register(session_id, ws)
    logger.info("[camera] WS 已连接 session=%s", session_id)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "snapshot":
                    jpeg = _extract_jpeg(data.get("data") or "")
                    if jpeg:
                        camera_service.deliver_frame(session_id, jpeg)
            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break
    finally:
        camera_service.unregister(session_id, ws)
    return ws


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
    app.router.add_post("/resume_talk", resume_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_get("/api/admin/config", admin_config)
    app.router.add_get("/api/admin/sessions", admin_sessions)
    app.router.add_get('/sse', sse_handler)
    app.router.add_get('/api/camera/ws', camera_websocket_handler)  # 摄像头「看用户」通道

    # ── 全局定时提醒管理 ──
    app.router.add_get('/api/reminders', api_reminders_list)
    app.router.add_post('/api/reminders', api_reminders_create)
    app.router.add_post('/api/reminders/cancel', api_reminders_cancel)

    # ── 通用文件上传（插口④：会话共享暂存区）──
    app.router.add_post('/api/files/upload', upload_file)

    # ── ASR WebSocket endpoint — asr 包自注册（候选池启用态作 gate）──
    try:
        from asr import setup_routes as _setup_asr
        _setup_asr(app)
    except Exception as e:
        logger.warning(f"[ASR] Failed to register ASR endpoint: {e}")

    # ── 关键词唤醒（KWS）WebSocket — 模型就绪才可用，否则前端降级普通通话 ──
    try:
        from server.kws import kws_websocket_handler
        app.router.add_get('/api/kws/ws', kws_websocket_handler)
    except Exception as e:
        logger.warning(f"[KWS] Failed to register KWS endpoint: {e}")

    # ── 观测平台 /api/obs/* ──
    try:
        from obs import setup_routes as _setup_obs
        _setup_obs(app)
    except Exception as e:
        logger.warning("obs routes registration failed: %s", e)

    # 注册 avatar 生成相关的路由
    setup_avatar_routes(app)

    app.router.add_static('/', path='web')