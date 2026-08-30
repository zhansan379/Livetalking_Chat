"""
SiliconFlow（硅基流动）云端 ASR 引擎。

HTTP API（见 api-docs.siliconflow.cn/docs/api/audio-transcriptions-post）：
- POST https://api.siliconflow.cn/v1/audio/transcriptions
- multipart/form-data：`file`（音频，最长 1h / 50MB）+ `model`
- 鉴权：Header `Authorization: Bearer ${SF_API_KEY}`（OpenAI Bearer 兼容）
- 响应：`{"text": "..."}`；单文件非流式，无分块/增量。

本引擎把 WS 收集的整段 float32 写为内存 WAV（BytesIO）后一次 multipart 上传，
阻塞式网络调用（executor 已保证在 run_in_executor 线程执行）。
注意：本开发机出网受限（硅流域名被墙），需在有外网/代理环境部署才能真正联通。
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
from numpy.typing import NDArray

from registry import register
from utils.logger import logger
from .base import BaseASR


@register("stt", "siliconflow")
class SiliconFlowASR(BaseASR):
    name = "siliconflow"

    DEFAULT = {
        "api_url": "https://api.siliconflow.cn/v1/audio/transcriptions",
        # 仅两个模型：FunAudioLLM/SenseVoiceSmall | TeleAI/TeleSpeechASR
        "model": "FunAudioLLM/SenseVoiceSmall",
        "wav_rate": 16000,       # 上传 WAV 的采样率（引擎用实际采样率亦可）
        "request_timeout": 120,  # 单次请求超时（秒）
    }

    def __init__(self, opt: dict | None = None, **kwargs):
        super().__init__(opt)
        self.p = {**self.DEFAULT, **((opt or {}).get("params") or {})}

    @classmethod
    def is_available(cls) -> bool:
        # 云端引擎：SDK 依赖装了 + 配了 SF_API_KEY 才算可用；否则候选不参与。
        if not os.environ.get("SF_API_KEY"):
            return False
        try:
            import requests  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False

    # 无本地模型加载（云端），基类 get_model 骨架保留但不使用。
    def _load_model(self):
        return self.p

    def transcribe(self, audio_float32: NDArray[np.float32], sample_rate: int,
                   use_itn: bool) -> tuple[str, float, float]:
        import requests
        import soundfile as sf

        api_key = os.environ.get("SF_API_KEY", "")
        if not api_key:
            raise RuntimeError("siliconflow: SF_API_KEY 未配置")

        # float32 → 内存 WAV（soundfile 在 BytesIO 写 WAV 头，含采样率）
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_float32, sample_rate, format="WAV")
        audio_duration_s = len(audio_float32) / max(sample_rate, 1)

        t0 = time.perf_counter()
        resp = requests.post(
            self.p["api_url"],
            headers={"Authorization": f"Bearer {api_key}"},
            files={
                "file": ("audio.wav", wav_io.getvalue(), "audio/wav"),
                "model": (None, self.p["model"]),
            },
            timeout=self.p.get("request_timeout", 120),
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code != 200:
            raise RuntimeError(
                f"siliconflow HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        text = (data.get("text") or "").strip()

        logger.info(
            f"[ASR] ✅ SiliconFlow inference complete\n"
            f"       ├─ Latency  : {inference_ms:>8.0f} ms\n"
            f"       ├─ Audio len: {audio_duration_s:>8.1f} s\n"
            f"       └─ Text     : \"{text[:100]}{'…' if len(text) > 100 else ''}\""
        )
        self.asr_ok(text, inference_ms, audio_duration_s)
        return text, inference_ms, audio_duration_s