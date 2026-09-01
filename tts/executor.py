"""
TTS 候选池 + 熔断回退驱动（句内同步回退）。

设计对齐 asr/executor.py::ASRPool 与 infra_ai/router.py 的 failover 语义，但以 BaseTTS
的生命周期与观测为宿主：

- TTSPool 本身是 BaseTTS 子类——`avatars/base_avatar.py`、`put_msg_txt`/`render`/`flush_talk`
  的使用方无需感知单后端还是池；yr worker 线程由基类 render()/process_tts() 驱动 msgqueue，
  `_run_tts_observed` 统一埋点只发一条 tts_call。
- 句内回退：对每个入队的句子，在 process_tts 同一线程内，按 config.yaml 的
  routing.candidates 顺序遍历 enabled 候选 → 熔断过滤 → 引擎懒实例化 → 调其 `txt_to_audio`；
  成功（last_tts.success 且 audio_ms≥min_audio_ms）立即 mark_success 并返回；失败 mark_failure
  继续下一候选重合成同一句，让用户这一句立刻出声，而不是静音。
- 埋点唯一：池内直接调 `engine.txt_to_audio`（原生版，无埋点），外层 `_run_tts_observed`
  包住的是 TTSPool.txt_to_audio，故每个句子只发一条 tts_call；胜出引擎写入 last_ts["provider"]。
- 停顿传导：flush_talk 除清空自身队列/代际自增外，把 state=PAUSE 同步到已实例化引擎，真正停嘴。
"""

from __future__ import annotations

import copy
import importlib
import time

from registry import create as registry_create
from utils.cand_pool import build_candidates
from utils.health_store import HealthStore
from utils.logger import logger

from . import TTS_MODULES
from .base_tts import BaseTTS, State


class TTSPool(BaseTTS):
    """候选池：构建候选列表 + 熔断 + 引擎懒实例化 + 句内回退驱动。"""

    def __init__(self, base_opt, parent):
        super().__init__(base_opt, parent)
        from .config_loader import get_config

        cfg = get_config()
        self._health = HealthStore(**cfg.CIRCUIT_BREAKER)
        self._candidates = build_candidates(cfg.ROUTING.get("candidates") or [])
        self._engines: dict[str, BaseTTS] = {}  # cand.id → 懒实例化引擎
        self._min_audio_ms = cfg.MIN_AUDIO_MS
        self._last_tried: str | None = None     # 全失败时，最后一个尝试的引擎（诊断用）

        if not self._candidates:
            logger.warning("[TTS] select_TTS 进入候选池但 routing.candidates 为空")

    # ── 引擎懒实例化 （语义对齐 asr/executor.py::_get_engine）──
    def _engine(self, cand) -> BaseTTS:
        if cand.id not in self._engines:
            # cand.params 覆盖到 base_opt 上（浅拷贝），供候选引擎读取其专属参数。
            cand_opt = copy.copy(self.opt)
            if hasattr(cand_opt, "__dict__"):
                cand_opt.__dict__.update(cand.params)
            # importlib 触发 @register("tts", engine)；引擎名 = 模块名 = registry 名。
            importlib.import_module(TTS_MODULES[cand.engine])
            self._engines[cand.id] = registry_create(
                "tts", cand.engine, opt=cand_opt, parent=self.parent
            )
        return self._engines[cand.id]

    # ── 句内回退驱动（同步；在 BaseTTS.process_tts 同线程内执行）──
    def txt_to_audio(self, msg: tuple[str, dict]):
        text, textevent = msg  # noqa: F841 - text 仅用于日志记录

        # 观测上下文：池线程里用显式 ID 把「中间候选失败」挂回当前聊天 trace。
        obs_ctx = {}
        try:
            obs_ctx = (textevent or {}).get("_obs") or {}
        except Exception:  # noqa: BLE001 - 观测缺失不影响回退
            pass

        tried = 0              # 尝试过的候选数（跨候选回退的「候选级」计数）
        total_attempts = 0     # 本句所有候选的真实合成尝试次数之和（含引擎内部重试）
        any_retried = False    # 任一候选发生过重试（引擎内部 或 跨候选回退）

        def _emit_candidate(cand_id, lt, elapsed_ms):
            """失败候选发一条 tts_candidate，定位『哪个候选、为什么挂』。"""
            if not obs_ctx:
                return
            try:
                from obs import emit_explicit
                emit_explicit({
                    "type": "tts_candidate",
                    "engine": cand_id, "success": False,
                    "fail_reason": lt.get("fail_reason"),
                    "err_type": lt.get("err_type"),
                    "attempts": int(lt.get("attempts") or 1),
                    "retried": bool(lt.get("retried")),
                    "audio_ms": round(float(lt.get("audio_ms") or 0), 1),
                    "elapsed_ms": round(elapsed_ms, 1),
                }, trace_id=obs_ctx.get("trace_id"),
                   session_id=obs_ctx.get("session_id"),
                   parent_id=obs_ctx.get("parent_id"), kind="chat")
            except Exception:  # noqa: BLE001 - 观测失败静默
                pass

        def _fmt(lt, reason, err, audio=0):
            """把一次候选尝试归一化成统一 last_tts 结构（lt 可为 None）。"""
            return {
                "success": False, "fail_reason": reason, "err_type": err,
                "audio_ms": float(lt["audio_ms"]) if lt else audio,
                "attempts": int(lt.get("attempts") or 1) if lt else 1,
                "retried": bool(lt.get("retried")) if lt else False,
                "truncated": bool(lt.get("truncated")) if lt else False,
            }

        for cand in self._candidates:
            if not cand.enabled:
                continue
            if not self._health.allow_call(cand.id):
                logger.info("[TTS] 候选 %s 熔断中，跳过", cand.id)
                # 熔断跳过也发一条候选诊断事件，供观测聚合「被熔断次数」
                _emit_candidate(cand.id, _fmt(None, "circuit_open", None, audio=0), 0.0)
                continue

            tried += 1
            self._last_tried = cand.id
            t0 = time.time()
            try:
                eng = self._engine(cand)
            except Exception as e:  # noqa: BLE001 - 引擎加载失败按候选失败处理
                logger.warning("[TTS] 候选 %s 引擎加载失败: %s", cand.id, e)
                self._health.mark_failure(cand.id)
                lt = _fmt(None, "engine_load_failed", type(e).__name__)
                total_attempts += int(lt.get("attempts") or 1)
                _emit_candidate(cand.id, lt, (time.time() - t0) * 1000)
                continue

            # 传导停顿（barge-in）；last_tts 归零以便读取本句结果。
            eng.state = self.state
            eng.last_tts = None
            try:
                eng.txt_to_audio(msg)  # 直接调原生版，不经 _run_tts_observed → 不重复埋点
            except Exception as e:  # noqa: BLE001 - 引擎异常按候选失败处理
                eng.last_tts = {
                    "success": False, "fail_reason": "exception",
                    "err_type": type(e).__name__, "audio_ms": 0,
                    "attempts": 1, "retried": False, "truncated": False,
                }
            lt = eng.last_tts or _fmt(None, "unclassified", None)

            # WS-1.2：本句累计真实合成尝试次数；任一候选发生过重试（内部/回退）都算
            total_attempts += int(lt.get("attempts") or 1)
            any_retried = any_retried or bool(lt.get("retried"))

            ok = bool(lt.get("success")) and lt.get("audio_ms", 0) >= self._min_audio_ms
            if ok:
                self._health.mark_success(cand.id)
                self.last_tts = {
                    **lt, "success": True,
                    "attempts": total_attempts, "retried": any_retried or tried > 1,
                    "provider": cand.engine,          # provider=胜出引擎
                }
                if tried > 1:
                    logger.info(
                        "[TTS] 候选 %s 回退成功（放弃前 %d 个失败候选）", cand.id, tried - 1
                    )
                return

            # 用户打断（barge-in）：这不是引擎故障，是外部要求停嘴。既不该熔断污染
            # 健康状态，也不该拿其它候选重合成同一句（此刻用户正说话，重合成无意义），
            # 更不该被报成 all_tts_candidates_failed 吞句。直接以「被打断」终态返回。
            if lt.get("fail_reason") == "barge_in":
                self.last_tts = {
                    **lt, "success": False,
                    "attempts": total_attempts, "retried": any_retried or tried > 1,
                    "provider": cand.engine,
                }
                return

            self._health.mark_failure(cand.id)
            logger.warning(
                "[TTS] 候选 %s 失败(%s, audio_ms=%s)，回退下一候选",
                cand.id, lt.get("fail_reason"), lt.get("audio_ms"),
            )
            _emit_candidate(cand.id, lt, (time.time() - t0) * 1000)

        # 全候选失败：登记为失败结果，不抛（由 _run_tts_observed 统一埋点）。
        self.last_tts = {
            "success": False, "fail_reason": "all_tts_candidates_failed",
            "err_type": None, "audio_ms": 0,
            "attempts": total_attempts or tried,
            "retried": any_retried or tried > 1, "truncated": False,
            "provider": self._last_tried,
        }

    def flush_talk(self):
        """把停顿传导到已实例化的引擎，真正停嘴。"""
        super().flush_talk()  # 自己的 queue 清空 + _epoch+1 + state=PAUSE
        for eng in self._engines.values():
            eng.state = State.PAUSE


def reset_pool():
    """调试/测试用：与 config_loader.reset_config 配合重建候选池配置。"""
    from .config_loader import reset_config as _reset_config

    _reset_config()