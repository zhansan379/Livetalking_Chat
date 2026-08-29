import os
import time
import json
import uuid
import asyncio
import numpy as np
import resampy
import websockets

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register
from . import doubao_protocol as proto

# 火山引擎豆包语音合成 - 双向流式 (WebSocket Bidirectional) 接口
# 参考官网示例: TTS Websocket Bidirection
URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"


@register("tts", "doubao")
class DoubaoTTS(BaseTTS):
    """豆包语音合成 (火山引擎) — 双向流式 WebSocket 接口。

    区别于单向 HTTP：双向上可边喂文本边收音频，能进一步压低首包延迟。
    当前实现（路线A）仍按整句输入、每句一个 WS 连接，传输层换成双向 WS；
    后续若要吃满延迟红利，可再接入 LLM 流式 token 逐字喂入（路线B）。

    ── 语气调整机制（分层） ──────────────────────────────────────────
    三层合并（见 _build_req，全部在 TTS 层处理）：
      1. 每句覆盖：datainfo["tts"]（优先级最高；也是 LLM 驱动钩子——
         infra_ai/routes 让 LLM 输出 {"tts":{"context_texts":[...],"pitch":3}} 进
         datainfo，TTS 层零改动即生效）
      2. 全局默认：config.yaml 的 doubao_tone 块
      3. 代码默认：字段缺省不设，走引擎默认
    暴露能力映射至豆包 API：
      context_texts 语音指令 / speech_rate 语速 / loudness_rate 音量 /
      post_process.pitch 音调 / explicit_dialect 方言 / explicit_language 语种 /
      section_id 多轮持续（会话内语气连续）。

    需要设置环境变量 DOUBAO_API_KEY (火山引擎控制台 > API Key 管理)。
    REF_FILE 用作音色ID (speaker)，例如 zh_female_gaolengyujie_uranus_bigtts。
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)

        # API Key 从控制台 > API Key 管理 获取
        self.api_key = os.getenv("DOUBAO_API_KEY")
        # X-Api-Resource-Id: seed-tts-2.0 (大模型2.0音色) / seed-icl-2.0 (声音复刻音色)
        self.resource_id = getattr(opt, "doubao_resource_id", "seed-tts-2.0")
        # 音色ID, 复用 REF_FILE 参数
        self.voice = opt.REF_FILE or "zh_female_gaolengyujie_uranus_bigtts"

        # 播放管线只认裸 PCM，故默认 pcm 才能直接 np.frombuffer；
        # mp3/ogg_opus 需额外解压解码（未内置）。
        self.audio_format = getattr(opt, "doubao_audio_format", "pcm")
        self.src_sr = int(getattr(opt, "doubao_sample_rate", self.sample_rate))

        # 语气调整：全局默认(config.yaml doubao_tone，dict 或 None) + 会话级 section_id
        self.tone = getattr(opt, "doubao_tone", None)
        if not isinstance(self.tone, dict):
            self.tone = {}
        # 多轮持续语气：一次数字人会话(本 TTS 实例生命周期)共用一个 section_id，
        # 服务端据此在连续合成间保存上下文；可被 config 固定或 datainfo['tts'] 覆盖。
        self.section_id = self.tone.get("section_id") or str(uuid.uuid4())

        # 每句话一个独立 WS 连接（简单可靠）；跨句复用连接留作后续优化。
        self._pending = np.array([], dtype=np.float32)  # 跨帧残余
        self._first = True                              # 是否首个分帧

        if not self.api_key:
            logger.warning("DoubaoTTS(wss): DOUBAO_API_KEY 未设置，请设置环境变量")

        logger.info(
            f"DoubaoTTS(wss) init: resource_id={self.resource_id}, voice={self.voice}, "
            f"format={self.audio_format}, sample_rate={self.src_sr}"
        )

    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg
        if not text:
            return

        if not self.api_key:
            logger.error("DoubaoTTS(wss): DOUBAO_API_KEY 未设置，跳过合成")
            self._send_end(text, textevent)
            return

        # 每条消息可覆盖音色 / 模型版本
        tts_cfg = textevent.get("tts", {})
        speaker = tts_cfg.get("ref_file", self.voice)
        resource_id = tts_cfg.get("resource_id", self.resource_id)

        self._pending = np.array([], dtype=np.float32)
        self._first = True
        try:
            # 合并 全局默认(config doubao_tone) + 每句覆盖(datainfo['tts']) → req_params
            req_params = self._build_req(speaker, resource_id, textevent)
            # 在 TTS 线程里驱动一个独立事件循环，跑完整 WS 会话
            asyncio.run(self._run_session(text, req_params, resource_id, msg))
        except Exception:  # noqa: BLE001 - 失败不吞栈，回退到结束标记
            logger.exception("DoubaoTTS(wss) session failed")
        finally:
            self._send_end(text, textevent)

    # ── 语气调整：分层合并请求参数 ────────────────────────────────────

    def _build_req(self, speaker, resource_id, textevent: dict) -> dict:
        """合并 请求级覆盖 > 全局默认(config doubao_tone) → 完整 req_params。

        datainfo["tts"] 见类 docstring 的 LLM 钩子契约；空值一律不写入，避免污染请求。
        返回可直接用于 start_session / task_request 的 req_params。
        """
        msg_cfg = (textevent or {}).get("tts", {}) or {}
        tone = self.tone  # 既有 dict 又防 config 给 None
        if not isinstance(tone, dict):
            tone = {}

        def pick(key: str, default=None):
            # 消息级覆盖 > 全局默认 > 预留默认
            return msg_cfg.get(key, tone.get(key, default))

        req = {
            "speaker": speaker,
            "audio_params": {"format": self.audio_format, "sample_rate": self.src_sr},
        }

        # 语速：兼容旧用法 rate，正式字段 speech_rate
        rate = pick("speech_rate", None)
        if rate is None:
            rate = pick("rate", None)
        if rate is not None:
            req["speech_rate"] = int(rate)

        loudness = pick("loudness_rate")
        if loudness is not None:
            req["loudness_rate"] = int(loudness)

        pitch = pick("pitch")
        if pitch is not None:
            req.setdefault("post_process", {})["pitch"] = int(pitch)

        # 语气核心：自然语言语音指令，规整为数组
        ctx = pick("context_texts")
        if ctx:
            req["context_texts"] = ctx if isinstance(ctx, list) else [str(ctx)]

        # 方言 / 语种（仅非空才带，避免空串触发引擎限制）
        for k in ("explicit_dialect", "explicit_language"):
            v = pick(k)
            if v:
                req[k] = v

        # 多轮持续语气：请求级 > 全局默认 > 会话级自动生成
        sid = pick("section_id", self.section_id)
        if sid:
            req["section_id"] = str(sid)

        return req

    # ── 双向 WS 会话 ──────────────────────────────────────────

    async def _run_session(self, text, req_params: dict, resource_id, msg):
        textevent = msg[1]
        speaker = req_params.get("speaker", self.voice)
        headers = {
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Connect-Id": str(uuid.uuid4()),
            "X-Control-Require-Usage-Tokens-Return": "*",
        }

        async with websockets.connect(
            URL, extra_headers=headers, max_size=10 * 1024 * 1024
        ) as websocket:
            # 建连
            await proto.start_connection(websocket)
            await proto.wait_for_event(
                websocket,
                proto.MsgType.FullServerResponse,
                proto.EventType.ConnectionStarted,
            )

            # 开会话（带音色/音频参数）；线协议要求外层包 {"req_params": {...}}
            session_id = str(uuid.uuid4())
            payload = {"req_params": req_params}
            await proto.start_session(
                websocket, json.dumps(payload).encode(), session_id
            )
            await proto.wait_for_event(
                websocket,
                proto.MsgType.FullServerResponse,
                proto.EventType.SessionStarted,
            )

            # 整句提交（路线A；路线B改为逐 token 调 task_request）
            task = dict(payload)
            task["req_params"]["text"] = text
            await proto.task_request(websocket, json.dumps(task).encode(), session_id)
            # 提交完即发 finish_session —— 服务端收到后才回 SessionFinished，
            # 若等到收流结束再发就会死等。（官网示例是发送任务与收流并发做这一步）
            await proto.finish_session(websocket, session_id)

            # 循环收流
            t0 = time.perf_counter()
            while True:
                mp = await proto.receive_message(websocket)
                if mp.type == proto.MsgType.FullServerResponse:
                    if mp.event == proto.EventType.SessionFinished:
                        break
                    # 其它完整事件（Usage/TTSResponse 等）忽略
                    continue
                elif mp.type == proto.MsgType.AudioOnlyServer:
                    if not mp.payload:
                        continue
                    if self._first:
                        logger.info(
                            f"DoubaoTTS(wss) first audio chunk: "
                            f"{time.perf_counter() - t0:.3f}s "
                            f"(speaker={speaker}, text={len(text)} chars)"
                        )
                    if self.state != State.RUNNING:
                        break
                    self._consume_pcm(mp.payload, text, textevent)
                elif mp.type == proto.MsgType.Error:
                    logger.error(
                        "DoubaoTTS(wss) server error: code=%s, %s",
                        mp.error_code,
                        mp.payload.decode("utf-8", "ignore"),
                    )
                    break
                else:
                    logger.warning("DoubaoTTS(wss) unexpected msg: %s", mp)

            # ── PCM 分帧 → parent (沿旧 doubao / omnitts 语义) ────────

    def _consume_pcm(self, chunk: bytes, text: str, textevent: dict):
        # Raw PCM 16-bit → float32
        stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767

        # 采样率不一致时重采样到 16kHz (self.sample_rate)
        if self.src_sr != self.sample_rate:
            stream = resampy.resample(
                x=stream, sr_orig=self.src_sr, sr_new=self.sample_rate
            )

        # 拼接上次残余（双向流可能拦腰截断一帧）
        if self._pending.shape[0] > 0:
            stream = np.concatenate([self._pending, stream])

        total = stream.shape[0]
        idx = 0
        while total - idx >= self.chunk and self.state == State.RUNNING:
            eventpoint = {}
            if self._first:
                eventpoint = {"status": "start", "text": text}
                self._first = False
            eventpoint.update(**textevent)
            self.parent.put_audio_frame(stream[idx : idx + self.chunk], eventpoint)
            idx += self.chunk

        self._pending = stream[idx:]  # 不足一 chunk 的留到下次

    def _send_end(self, text, textevent):
        eventpoint = {"status": "end", "text": text}
        eventpoint.update(**textevent)
        self.parent.put_audio_frame(
            np.zeros(self.chunk, dtype=np.float32), eventpoint
        )

    def stop_tts(self):
        # 每句各自建/关连接，无长驻连接要清理；占位保持接口一致。
        logger.info("DoubaoTTS(wss) stop (no persistent connection)")