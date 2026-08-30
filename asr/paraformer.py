"""
DashScope（阿里云百炼）Paraformer 实时语音识别引擎。

云端 ASR：走 dashscope.audio.asr.Recognition 双向流式接口（format='pcm'），
把 WS 收集的整段 float32 转成 int16 PCM 内存喂入，on_event 回调累加已断句的最终文本。

- 鉴权：从环境变量 DASHSCOPE_API_KEY 读取（文档要求，不硬编码）。
- 专属域名：可选，配置 api_url 时覆盖 dashscope.base_websocket_api_url
  （如 wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference）。
- 依赖可用性：dashscope 可导入 且 环境变量里配了 DASHSCOPE_API_KEY 才视为可用，
  否则该候选不参与（available()/挂载 gate 自动排除），不阻塞本地 SenseVoice。
- 本引擎是网络调用（阻塞），executor 已保证在 run_in_executor 线程执行。
"""

from __future__ import annotations

import os
import time

import numpy as np
from numpy.typing import NDArray

from registry import register
from utils.logger import logger
from .base import BaseASR


@register("stt", "paraformer")
class ParaformerASR(BaseASR):
    name = "paraformer"

    DEFAULT = {
        "model": "paraformer-realtime-v2",
        "format": "pcm",                 # 喂内存 PCM（非流式 wav 需落盘，此处用流式）
        "sample_rate": 16000,            # 参考值；执行时以 transcribe 传入的实际采样率为准
        "api_url": "",                   # 可选专属域名，非空则覆盖 dashscope.base_websocket_api_url
        "language_hints": ["zh", "en"],  # 仅 paraformer-realtime-v2 生效
        "disfluency_removal_enabled": False,
        "semantic_punctuation_enabled": False,   # 开启语义断句则关闭 VAD 断句
        "max_sentence_silence": 800,             # VAD 断句静音阈值(ms)，200~6000
        "multi_threshold_mode_enabled": False,
        "punctuation_prediction_enabled": True,
        "inverse_text_normalization_enabled": True,  # ITN（默认开，中文数字转阿拉伯）
        "vocabulary_id": "",             # 热词ID（v2 系列模型）
        "phrase_id": "",                 # 热词ID（v1 系列模型）
        "heartbeat": False,
        "chunk_bytes": 3200,             # send_audio_frame 每片大小（建议 1KB~16KB）
    }

    def __init__(self, opt: dict | None = None, **kwargs):
        super().__init__(opt)
        self.p = {**self.DEFAULT, **((opt or {}).get("params") or {})}

    @classmethod
    def is_available(cls) -> bool:
        # 云端引擎：SDK 装了 + 配了 Key 才算可用；否则候选不参与。
        if not os.environ.get("DASHSCOPE_API_KEY"):
            return False
        try:
            import dashscope  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False

    # 无本地模型加载（云端），基类 get_model 骨架保留但不使用。
    def _load_model(self):
        return self.p

    def transcribe(self, audio_float32: NDArray[np.float32], sample_rate: int,
                   use_itn: bool) -> tuple[str, float, float]:
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

        # 专属域名（可选覆盖）；据文档建议从 dashscope.aliyuncs.com 迁移到业务空间专属域名。
        api_url = (self.p.get("api_url") or "").strip()
        if api_url:
            dashscope.base_websocket_api_url = api_url

        # float32 → int16 PCM 内存字节
        pcm = (np.clip(audio_float32, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        audio_duration_s = len(audio_float32) / max(sample_rate, 1)

        # 回调累加确认句（句尾帧，以 begin_time 为标识去重）
        confirmed: dict[int, str] = {}
        errors: list = []

        class _Cb(RecognitionCallback):
            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                if isinstance(sentence, dict) and sentence.get("text"):
                    if RecognitionResult.is_sentence_end(sentence):
                        confirmed[int(sentence.get("begin_time", len(confirmed)))] = sentence["text"]

            def on_error(self, result) -> None:
                try:
                    msg = result.message if hasattr(result, "message") else repr(result)
                except Exception:  # noqa: BLE001
                    msg = "<unreadable>"
                errors.append(msg)
                logger.warning("[ASR] paraformer on_error: %s", msg)

        p = self.p
        # 布尔/数值/语种参数总是传（DEFAULT 均有明确值，保留显式 False 语义）；
        # 空字符串热词ID不传（空值无意义）。
        init_kwargs = {k: p[k] for k in (
            "language_hints", "disfluency_removal_enabled",
            "semantic_punctuation_enabled", "max_sentence_silence",
            "multi_threshold_mode_enabled", "punctuation_prediction_enabled",
            "inverse_text_normalization_enabled", "heartbeat",
        )}
        for sk in ("vocabulary_id", "phrase_id"):
            if p.get(sk):
                init_kwargs[sk] = p[sk]

        rec = Recognition(
            model=p["model"], format=p.get("format", "pcm"),
            sample_rate=sample_rate, callback=_Cb(), **init_kwargs,
        )

        t0 = time.perf_counter()
        rec.start()
        chunk = int(p.get("chunk_bytes", 3200)) or 3200
        for i in range(0, len(pcm), chunk):
            rec.send_audio_frame(pcm[i:i + chunk])
            if errors:  # 已失败，不再喂
                break
        rec.stop()  # 阻塞到 on_complete / on_error
        inference_ms = (time.perf_counter() - t0) * 1000

        if errors:
            raise RuntimeError(f"paraformer recognition failed: {'; '.join(map(str, errors))}")

        text = "".join(t for _, t in sorted(confirmed.items())).strip() or ""
        logger.info(
            f"[ASR] ✅ paraformer inference complete\n"
            f"       ├─ Latency  : {inference_ms:>8.0f} ms\n"
            f"       ├─ Audio len: {audio_duration_s:>8.1f} s\n"
            f"       └─ Text     : \"{text[:100]}{'…' if len(text) > 100 else ''}\""
        )
        self.asr_ok(text, inference_ms, audio_duration_s)
        return text, inference_ms, audio_duration_s