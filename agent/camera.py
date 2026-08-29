###############################################################################
#  摄像头「看用户」工具：浏览器按需抓帧 → 后端取最新帧 → 整帧给视觉模型描述
#
#  数据流（完全按需，画面即取即弃）：
#     浏览器的 /api/camera/ws 连到本服务的 _sessions[session_id]
#     工具 look_at_user() 被模型调用 →
#        CameraService.request_capture() 向浏览器推一条 {type:'capture_request'}
#        → 浏览器现场抓一帧 JPEG 回传 {type:'snapshot', data:...} →
#        deliver_frame() 唤醒等待中的 future → 拿到 jpeg bytes
#     千字节画面只在内存暂存、用完即弃，不进对话历史、不落盘。
#
#  隐私：画面仅在 handler 内部转成 data URL 喂给视觉模型，随后 GC；
#        无帧（未授权/未连接）优雅降级为「看不到」，绝不阻塞对话。
###############################################################################

import asyncio
import base64

from utils.logger import logger

# 等待浏览器回传一帧的最长秒数（超过即视为拿不到画面，降级）
_CAPTURE_TIMEOUT = 3.0


class _CameraSession:
    """单个 session 的摄像头 WS 会话：持有浏览器 socket + 一次在等的抓帧 future。

    同一时刻至多有一个 in-flight 抓帧请求（工具循环是串行的）。
    """

    __slots__ = ("ws", "pending")

    def __init__(self, ws):
        self.ws = ws         # aiohttp.WebSocketResponse（浏览器端）
        self.pending = None  # asyncio.Future | None


class CameraService:
    """按 session 管理浏览器摄像头 WS，提供「按需抓一帧」的能力。"""

    def __init__(self):
        self._sessions: dict[str, _CameraSession] = {}

    # ── WS 生命周期（由 /api/camera/ws handler 调用）───────────────────────
    def register(self, session_id: str, ws) -> None:
        self._sessions[session_id] = _CameraSession(ws)
        logger.info("[camera] session %s 已连接", session_id)

    def unregister(self, session_id: str, ws) -> None:
        s = self._sessions.get(session_id)
        if s is not None and s.ws is ws:
            self._sessions.pop(session_id, None)
            # 连接断开：唤醒在等帧的等待方，让其走「看不到」降级路径
            if s.pending is not None and not s.pending.done():
                s.pending.set_result(None)
            logger.info("[camera] session %s 已断开", session_id)

    # ── 浏览器回传一帧（由 /api/camera/ws handler 收到 snapshot 时调用）──────
    def deliver_frame(self, session_id: str, jpeg: bytes) -> None:
        s = self._sessions.get(session_id)
        if s is None:
            return
        if s.pending is not None and not s.pending.done():
            s.pending.set_result(jpeg)  # 唤醒 request_capture()
        s.pending = None

    # ── 工具侧：按需向浏览器要一帧 ─────────────────────────────────────────
    async def request_capture(
        self, session_id: str, timeout: float = _CAPTURE_TIMEOUT
    ) -> bytes | None:
        """主动推 capture_request，等到浏览器回传一帧 JPEG bytes；超时/断开返回 None。"""
        s = self._sessions.get(session_id)
        if s is None or s.ws is None or s.ws.closed:
            return None

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        s.pending = fut
        try:
            await s.ws.send_json({"type": "capture_request"})
        except Exception as e:  # noqa: BLE001 - 推送失败按拿不到帧处理
            logger.warning("[camera] send capture_request failed: %s", e)
            s.pending = None
            return None

        done, _ = await asyncio.wait({fut}, timeout=timeout)
        if fut in done:
            return fut.result()  # bytes 或 None（连接断开时由 unregister 置 None）
        s.pending = None  # 超时：作废这次等待
        return None


# 全局单例：WS handler 与工具 handler 共享
camera_service = CameraService()


# ─── 工具 handler（契约：async def handler(args, cfg, ctx=None) -> str）────────
async def look_at_user(args: dict, cfg, ctx=None) -> str:
    """让数字人「看一眼」正在对话的用户：按需抓一帧，整帧交给视觉模型描述。

    返回一段口语化的、面向用户的描述文本（如情绪/表情/衣着/环境）。拿不到
    画面时返回一句降级话术，不伪造。
    """
    from infra_ai import async_call_vlm

    session_id = getattr(ctx, "session_id", None)
    if not session_id:
        return "（当前会话未绑定摄像头）"

    jpeg = await camera_service.request_capture(session_id)
    if not jpeg:
        # 拿不到帧的原因未知，绝不武断归因于「没授权/没开摄像头」——
        # 用户可能已授权并开启，只是不在镜头里。给中性的、可行动的说辞。
        return ("（这一下没收到你的画面——可能是摄像头没开/没连上，也可能是你人不在镜头里。"
                "如果你确认摄像头是开的，就让自己出现在画面里，我再看看。）")

    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    # instructions：主模型可选参数，本意是让 VLM「专门分析某方面」——优先于默认的整体描述
    focus = ((args or {}).get("instructions") or "").strip()
    if focus:
        prompt = (
            "这是此刻正在和你说话的人的实时摄像头画面。请用简短、口语化的中文，"
            "专门分析下面这个关注点，答案要具体、只依据画面里看得见的证据，不要臆测；"
            f"相关细节看不清就如实说明。关注点：{focus}。\n"
            "如果画面里根本看不到这个人的脸或人（比如躲开了、人不在画面里、离得太远看不清），"
            "就明确说『看不到你人』，并按关注点如实说明，不要凭想象推测或编造。"
        )
    else:
        prompt = (
            "这是此刻正在和你说话的人的实时摄像头画面。请用简短、口语化的中文，"
            "描述用户当前的状态：情绪/表情、动作、大致穿着，以及画面里与你对话相关的环境信息。"
            "如果画面里看不到人（比如他躲开了、人不在画面里、或离得太远看不清），"
            "就明确说『看不到你人』，再补一句画面里其实看到的是什么环境；"
            "不要硬编造他的状态，也不要把它当成摄像头没开/没授权的问题。"
            "画面很清楚但有真人出镜时就正常描述；不要提『摄像头/画面』这类机制词。"
        )
    try:
        text = await async_call_vlm([{"role": "user", "content": prompt}],
                                    use_json=False, images=[data_url],
                                    extra={"kind": "camera_view"})
    except Exception as e:  # noqa: BLE001 - 视觉调用失败不中断对话
        logger.warning("[camera] VLM 描述失败: %s", e)
        return "（我看到你了，但这一下说不上来具体状态）"

    text = (text or "").strip()
    return text or "（我看到你了）"