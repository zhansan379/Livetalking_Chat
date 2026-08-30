"""
SenseVoice 语音识别引擎（funasr / ModelScope 本地）。

迁移自原 server/asr_server.py 的 `_load_sensevoice` + `_run_inference`：
- 模型为实例级懒加载单例（基类 get_model + _load_model 骨架）；
- 推理并发锁放引擎侧（`_infer_lock`）——并发语义随后端/设备而异，不强加给所有引擎；
- 依赖走 ModelScope 本地缓存（`trust_remote_code` 本地使能，不触发外网下载）。
"""

import io
import threading
import time

import numpy as np
from numpy.typing import NDArray

from registry import register
from utils.logger import logger
from .base import BaseASR


@register("stt", "sensevoice")
class SenseVoiceASR(BaseASR):
    name = "sensevoice"

    DEFAULT = {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "max_single_segment_time": 30000,
        "batch_size_s": 60,
        "device": "auto",                # auto → cuda:0 否则 cpu
        "trust_remote_code": True,
    }

    def __init__(self, opt: dict | None = None, **kwargs):
        super().__init__(opt)
        # params 覆盖 DEFAULT（来自候选配置 routing.candidates[].params）
        self.p = {**self.DEFAULT, **((opt or {}).get("params") or {})}
        # 单引擎推理互斥（转移原模块级 _sensevoice_inference_lock 到实例）
        self._infer_lock = threading.Lock()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import funasr  # noqa: F401, PLC0415
            return True
        except ImportError:
            return False

    def _load_model(self):
        """原 _load_sensevoice：懒加载单例，首次调用时加载。"""
        import torch
        from funasr import AutoModel

        device = self.p["device"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"[ASR] Loading SenseVoiceSmall on device='{device}' "
            f"(first run may use ModelScope local cache)..."
        )
        t0 = time.perf_counter()
        model = AutoModel(
            model=self.p["model"],
            vad_model=self.p["vad_model"],
            vad_kwargs={"max_single_segment_time": self.p["max_single_segment_time"]},
            device=device,
            trust_remote_code=self.p["trust_remote_code"],
        )
        elapsed = time.perf_counter() - t0
        logger.info(f"[ASR] ✅ SenseVoiceSmall ready — loaded in {elapsed:.1f}s on {device}")
        return model

    def transcribe(self, audio_float32: NDArray[np.float32], sample_rate: int,
                   use_itn: bool) -> tuple[str, float, float]:
        """原 _run_inference 主体（阻塞，供 run_in_executor / 候选池线程调用）。"""
        import soundfile as sf
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model = self.get_model()

        # 写内存 WAV，让 funasr 从头部读采样率
        wav_buf = io.BytesIO()
        sf.write(wav_buf, audio_float32, sample_rate, format="WAV")
        wav_buf.seek(0)

        t0 = time.perf_counter()
        with self._infer_lock:
            res = model.generate(
                input=wav_buf,
                cache={},
                language="auto",
                use_itn=use_itn,
                batch_size_s=self.p["batch_size_s"],
            )
        inference_ms = (time.perf_counter() - t0) * 1000

        text = ""
        if res and len(res) > 0 and res[0].get("text"):
            text = rich_transcription_postprocess(res[0]["text"])

        audio_duration_s = len(audio_float32) / sample_rate
        logger.info(
            f"[ASR] ✅ SenseVoice inference complete\n"
            f"       ├─ Latency     : {inference_ms:>8.0f} ms\n"
            f"       ├─ Audio length: {audio_duration_s:>8.1f} s\n"
            f"       └─ Text        : \"{text[:100]}{'…' if len(text) > 100 else ''}\""
        )

        self.asr_ok(text, inference_ms, audio_duration_s)
        return text, inference_ms, audio_duration_s