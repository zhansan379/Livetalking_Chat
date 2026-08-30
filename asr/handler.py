"""
ASR WebSocket 服务（迁移自原 server/asr_server.py）。

保留 FunASR 客户端协议与边界处理（mode 映射 / 短音频跳过 / 奇数字节修正 /
utterance trace_id 生成下发），仅把「推理」从直接调 `_run_inference` 改为
经 `get_pool().transcribe` 走候选池 + 熔断回退，并整段卸到 executor 线程。

trace_id 随响应下发浏览器，浏览器在 /human echo 回来，chat 段复用 → 拼一条
ASR→LLM→TTS 全链路 trace（asr_call 的 span_id==parent_id==trace_id，直连 OBS）。
"""

import asyncio
import json
import uuid

import numpy as np
from aiohttp import web

from utils.logger import logger
from .base import _obs_emit_explicit
from .config_loader import get_config
from .executor import get_pool


def _emit_too_short(trace_id, session_id, audio_seconds) -> None:
    """短音频（<20ms）跳过时发一条 asr_call（fail_reason=audio_too_short），obs 缺失/异常时静默。"""
    if not (_obs_emit_explicit and trace_id):
        return
    try:
        _obs_emit_explicit({
        "type": "asr_call", "span_id": trace_id, "parent_id": trace_id,
        "audio_ms": round(audio_seconds * 1000, 1),
        "audio_len_s": round(audio_seconds, 3),
        "inference_ms": 0.0, "elapsed_ms": 0.0, "rtf": 0.0,
        "text": "", "text_len": 0, "empty": True,
        "success": False, "fail_reason": "audio_too_short", "err_type": None,
    }, trace_id=trace_id, session_id=session_id, parent_id=trace_id, kind="asr")
    except Exception:  # noqa: BLE001 - 观测失败不影响 ASR 主流程
        logger.debug("[ASR] obs emit too_short failed (ignored)", exc_info=True)


async def asr_websocket_handler(request):
    """
    FunASR 兼容 WebSocket handler。

    协议:1) 开连接 → 2) 发 JSON 配置({"is_speaking":true, mode, itn, ...})
    → 3) 流式发 PCM16 二进制块(960B=60ms@16k) → 4) 发 {"is_speaking":false}
    → 5) 服务端响应 {"text", "mode", "is_final":true, "timestamp", "trace_id"}。
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    client_ip = request.remote
    cfg = get_config()
    sample_rate = cfg.SAMPLE_RATE
    min_bytes = cfg.MIN_AUDIO_BYTES

    logger.info(f"[ASR] 🔌 WebSocket connected from {client_ip}")

    audio_buffer = bytearray()
    config: dict = {}
    session_start = 0.0
    chunks_received = 0
    utterance_tid: str | None = None  # 本次语音的回合 trace_id（服务端生成，随响应下发）

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.warning("[ASR] Received invalid JSON, ignoring")
                    continue

                if data.get("is_speaking") is True:
                    # ── Session start ──────────────────────────────────
                    config = data
                    audio_buffer = bytearray()
                    chunks_received = 0
                    session_start = __import__("time").perf_counter()
                    utterance_tid = uuid.uuid4().hex
                    logger.info(
                        f"[ASR] 🎙️  Recording started | "
                        f"mode={config.get('mode', 'offline')} | "
                        f"itn={config.get('itn', False)} | "
                        f"hotwords={bool(config.get('hotwords'))}"
                    )

                elif data.get("is_speaking") is False:
                    # ── End of speech → run inference (候选池回退) ─────
                    buf_bytes = len(audio_buffer)
                    audio_seconds = buf_bytes / (sample_rate * 2)  # 2 bytes per int16
                    session_elapsed = __import__("time").perf_counter() - session_start

                    logger.info(
                        f"[ASR] 🛑 Recording stopped | "
                        f"{chunks_received} chunks | "
                        f"{buf_bytes:,} bytes | "
                        f"{audio_seconds:.1f}s audio | "
                        f"session wall time {session_elapsed:.1f}s"
                    )

                    session_id = str(config.get("wav_name") or client_ip)

                    if buf_bytes < min_bytes:  # <20ms — skip
                        logger.warning("[ASR] Audio too short (< %dB), returning empty", min_bytes)
                        _emit_too_short(utterance_tid, session_id, audio_seconds)
                        await ws.send_str(json.dumps({
                            "text": "", "mode": config.get("mode", "offline"),
                            "is_final": True, "timestamp": None,
                            "trace_id": utterance_tid,
                        }))
                        continue

                    # 奇数字节修正（drop 半个采样）
                    if buf_bytes % 2 != 0:
                        logger.warning(
                            f"[ASR] Odd number of bytes received ({buf_bytes}), "
                            "dropping incomplete sample")
                        audio_buffer = audio_buffer[:-1]
                        buf_bytes -= 1

                    # PCM16 → float32 in [-1, 1]
                    audio_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0
                    use_itn = bool(config.get("itn", False))

                    # 整段回退卸线程：一次 run_in_executor 跑完整候选循环（阻塞推理在线程池）。
                    loop = asyncio.get_event_loop()
                    pool = get_pool()
                    _t0 = __import__("time").perf_counter()
                    res = await loop.run_in_executor(
                        None, pool.transcribe,
                        audio_float32, sample_rate, use_itn,
                        trace_id=utterance_tid, session_id=session_id,
                    )
                    elapsed_ms = (__import__("time").perf_counter() - _t0) * 1000

                    if res.error is not None:
                        logger.warning(
                            f"[ASR] ❌ All ASR candidates failed after {elapsed_ms:.0f}ms — "
                            f"fail_reason={res.fail_reason} err={res.err_type} | "
                            f"engine={res.engine_id or 'none'}"
                        )
                    else:
                        logger.info(
                            f"[ASR] 📤 Transcribed via engine='{res.engine_id}' "
                            f"(attempt #{res.attempts}{', retried' if res.retried else ''}) "
                            f"in {elapsed_ms:.0f}ms → \"{res.text[:60]}\""
                        )

                    # Map client mode → 前端期望的 response mode（原样保留）
                    mode = config.get("mode", "offline")
                    response_mode = "2pass-offline" if mode == "2pass" else mode

                    await ws.send_str(json.dumps({
                        "text": res.text, "mode": response_mode,
                        "is_final": True, "timestamp": None,
                        "trace_id": utterance_tid,
                    }))
                    logger.info(f"[ASR] 📤 Result sent to client (mode={response_mode})")

            elif msg.type == web.WSMsgType.BINARY:
                audio_buffer.extend(msg.data)
                chunks_received += 1

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break

    except asyncio.CancelledError:
        logger.info("[ASR] WebSocket handler cancelled")
    except Exception as e:  # noqa: BLE001
        logger.exception(f"[ASR] ❌ WebSocket handler error: {e}")

    logger.info(f"[ASR] 🔌 WebSocket disconnected ({client_ip})")
    return ws