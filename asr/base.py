"""
ASR 引擎抽象基类。

设计对齐 tts/base_tts.py：基类收敛「成败登记 + 统一观测埋点 + 懒加载单例骨架」，
子类只需实现一个热点方法 `transcribe()`（原生形态 = 旧 server/asr_server.py 的
`_run_inference`，一次性无状态：输入一段 float32 音频，输出文本）。

- 成败语义：`transcribe` 抛异常 → error（executor 据此触发候选回退）；
  返回空串 = 正常转录但没词（不触发回退，对齐旧版空麦行为）。
- 观测：`_transcribe_observed()` 单点统一埋点（仿 `_run_tts_observed`），
  新引擎零埋点自动获得 asr_call 统计。obs 缺失/关闭时整段降级为空操作。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from utils.logger import logger

# 先于任何云端引擎（paraformer/siliconflow）读取环境变量前，确保项目根 .env 已加载。
# 所有引擎都 import base → base 一 import 即 load_dotenv，API_KEY（DASHSCOPE/SF）就位，
# 不再依赖 executor.get_pool() 先行触发（镜像 infra_ai/config_loader 的范式）。
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # noqa: BLE001 - dotenv 缺失或路径不可用不阻塞，仍可用显式环境变量运行
    pass

# 观测（asr_call）——可选依赖：obs 缺失/关闭时全部降级为空操作。
try:
    from obs import emit_explicit as _obs_emit_explicit
except Exception:  # noqa: BLE001
    _obs_emit_explicit = None


@dataclass
class TranscriptionResult:
    """一次转录的聚合结果（供 executor / WS handler 消费）。"""

    text: str = ""
    engine_id: str = ""
    error: Exception | None = None        # 非空 = 本次候选失败（触发回退的依据）
    empty_text: bool = False              # 正常转录但没词（不算失败）
    inference_ms: float = 0.0
    audio_duration_s: float = 0.0
    attempts: int = 1                     # 第几个候选选中的（1=首选）
    retried: bool = False                 # attempts > 1
    # 附加诊断信息（fail_reason / err_type），供 WS 层日志用。
    fail_reason: str | None = field(default=None)
    err_type: str | None = field(default=None)


class BaseASR:
    name: str = ""                        # 子类填 registry 名（如 "sensevoice"）
    sample_rate: int = 16000              # WS 侧以 handler/config 为准；此处仅供引擎读算时长用

    def __init__(self, opt: dict | None = None, **kwargs):
        self.opt = opt or {}
        # 单次转录的成败登记，由 asr_ok / asr_fail 填充，_transcribe_observed 消费。
        # 子类只在自然分支点各调一行；未登记即按文本定性（空串→empty，否则成功）。
        self.last_result: dict | None = None

        # 懒加载单例（基类骨架）：锁在实例头，每引擎独立。
        self._load_lock = threading.Lock()
        self._model = None
        self._loaded = False

    # ── 成败登记：子类在「成败分界」处任调其一，统一写入 last_result ──
    # 相比散落的手拼 dict，这里收敛结构；引擎失败应抛异常（触发回退），
    # asr_fail 用于「非异常但视为失败」的情况（目前 SENSEVOICE 不用它）。

    def asr_ok(self, text: str = "", inference_ms: float = 0.0,
               audio_duration_s: float = 0.0):
        """登记本次转录成功（在拿到文本文本后调用一次）。"""
        self.last_result = {"success": True, "fail_reason": None,
                            "err_type": None, "inference_ms": inference_ms,
                            "audio_duration_s": audio_duration_s}

    def asr_fail(self, reason: str, err_type: str | None = None):
        """登记本次转录失败；reason 为可读原因。"""
        self.last_result = {"success": False, "fail_reason": reason,
                            "err_type": err_type, "inference_ms": 0.0,
                            "audio_duration_s": 0.0}

    # ── 懒加载单例：双重检查，锁在实例头（每引擎独立）──
    def get_model(self):
        """获取引擎模型（首次调用触发 _load_model）。"""
        if self._loaded:
            return self._model
        with self._load_lock:
            if self._loaded:
                return self._model
            self._model = self._load_model()
            self._loaded = True
            return self._model

    def _load_model(self):
        """子类实现模型加载（含首次加载的时间埋点/告警日志）。"""
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:
        """该引擎依赖是否可导入（取代旧 is_funasr_available）。"""
        raise NotImplementedError

    # ── 热点方法：子类唯一必须实现 ──
    def transcribe(self, audio_float32: NDArray[np.float32], sample_rate: int,
                   use_itn: bool) -> tuple[str, float, float]:
        """
        输入一段 float32 音频，输出 (text, inference_ms, audio_duration_s)。

        - 成功处调 self.asr_ok(...)；
        - 失败直接 raise（由 _transcribe_observed 捕获，触发候选回退）；
        - 返回空串 = 正常转录但没词，不算失败。
        """
        raise NotImplementedError

    # ── 统一埋点包装：新引擎零埋点自动获得 asr_call 统计（仿 _run_tts_observed）──
    def _transcribe_observed(self, audio_float32: NDArray[np.float32], sample_rate: int,
                             use_itn: bool, *, trace_id: str, session_id: str
                             ) -> TranscriptionResult:
        _t0 = time.time()
        text, inference_ms, audio_dur = "", 0.0, len(audio_float32) / max(sample_rate, 1)
        error = None
        try:
            text, inference_ms, audio_dur = self.transcribe(audio_float32, sample_rate, use_itn)
        except Exception as e:  # noqa: BLE001 - 观测不吞引擎异常
            error = e
        finally:
            lt = self.last_result
            self.last_result = None  # 消费掉，避免残留污染下一句

        # 收敛成败观测语义：优先引擎登记；未登记按文本定性（空串→empty，否则成功）。
        if lt is not None:
            ok = bool(lt.get("success", False))
            fail, err = lt.get("fail_reason"), lt.get("err_type")
        else:
            ok = bool((text or "").strip())
            fail = None if ok else "empty_text"
            err = None
        if error is not None:
            ok, fail, err = False, fail or "exception", error.__class__.__name__

        self._emit(trace_id, session_id, audio_dur, inference_ms,
                   (time.time() - _t0) * 1000, text or "", ok, fail, err)

        return TranscriptionResult(
            text=text or "", engine_id=self.name, error=error,
            empty_text=(not ok) and fail == "empty_text",
            inference_ms=inference_ms, audio_duration_s=audio_dur,
            fail_reason=fail, err_type=err,
        )

    # ── 观测埋点：复刻旧 asr_server._emit_asr 的载荷 ──
    # 单条 trace（ASR→LLM→TTS）里，asr_call 的 span_id==parent_id==trace_id
    # （ASR 是整条链路第一环）；不新增独立 trace_start/end。obs 缺失时静默跳过。
    def _emit(self, trace_id, session_id, audio_seconds, inference_ms, elapsed_ms,
              text, success, fail_reason, err_type=None) -> None:
        if not (_obs_emit_explicit and trace_id):
            return
        try:
            _obs_emit_explicit({
                "type": "asr_call", "span_id": trace_id, "parent_id": trace_id,
                "audio_ms": round(audio_seconds * 1000, 1),
                "audio_len_s": round(audio_seconds, 3),
                "inference_ms": round(inference_ms, 1),
                "elapsed_ms": round(elapsed_ms, 1),
                "rtf": round(inference_ms / 1000.0 / max(audio_seconds, 0.001), 4),
                "text": (text or "")[:40], "text_len": len(text or ""),
                "empty": not bool((text or "").strip()),
                "success": bool(success), "fail_reason": fail_reason,
                "err_type": err_type,
            }, trace_id=trace_id, session_id=session_id, parent_id=trace_id, kind="asr")
        except Exception:  # noqa: BLE001 - 观测失败不影响 ASR 主流程
            logger.debug("[ASR] obs emit failed (ignored)", exc_info=True)


# 兼容导入：部分调用方按模块路径引用 BaseASR。
__all__ = ["BaseASR", "TranscriptionResult"]