from threading import Thread
import queue
import time
from queue import Queue
from io import BytesIO
from enum import Enum

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar

from utils.logger import logger

class State(Enum):
    RUNNING = 0
    PAUSE = 1

class BaseTTS:
    def __init__(self, opt, parent: "BaseAvatar"):
        self.opt = opt
        self.parent = parent

        #self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING
        # 代际：每次 flush_talk 自增；入队项打上当前代际，process_tts 只处理
        # 当前代际的项——打断后残留的旧代际项被丢弃，做到「真正停嘴」。
        self._epoch = 0

    def flush_talk(self):
        self._epoch += 1
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self, msg: str, datainfo: dict = {}):
        if len(msg) > 0:
            # 打上入队时刻，供 process_tts 算「入队→开始合成」的队列延迟。
            # datainfo 里若已有 _obs（stream_llm_chat 蹭的 trace 身份）则保留；
            # 否则不写入——无 obs 上下文时零负担。
            try:
                from obs.recorder import now_ms
                if isinstance(datainfo, dict):
                    datainfo.setdefault("_obs", {}).setdefault("enqueued_ms", round(now_ms(), 1))
            except Exception:  # noqa: BLE001 - 观测缺失不影响入队
                pass
            self.msgqueue.put((msg, datainfo, self._epoch))

    def render(self, quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()

    def process_tts(self, quit_event):
        while not quit_event.is_set():
            try:
                item: tuple = self.msgqueue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            msg: tuple[str, dict] = (item[0], item[1])
            # 代际自检：项属于被打断前的旧代际 → 直接丢弃，真正停嘴
            if len(item) >= 3 and item[2] != self._epoch:
                continue
            self.state = State.RUNNING
            self._run_tts_observed(msg)
        self.stop_tts()
        logger.info('ttsreal thread stop')

    def _run_tts_observed(self, msg: tuple[str, dict]):
        """单点统一埋点：包住 provider 的 txt_to_audio，发一条 tts_call（跨线程显式 ID）。

        覆盖所有 provider（edge/azure/doubao/cosyvoice/…），新 provider 零埋点自动获得
        耗时/成败/异常统计。结果以 provider 写入的 self.last_tts 为准（可覆盖成败），
        缺省取计算值。观测缺失/关闭时整段零开销（局部 import，obs 非硬依赖）。
        """
        text, textevent = msg
        obs = (textevent or {}).get("_obs")
        _t0 = time.time()
        try:
            self.txt_to_audio(msg)
            lt = getattr(self, "last_tts", None) or {}
            ok   = lt.get("success", True)
            fail = lt.get("fail_reason")
            err  = lt.get("err_type")
            audio, att, trun, retried = (
                lt.get("audio_ms", 0), lt.get("attempts", 1),
                lt.get("truncated", False), lt.get("retried", False))
        except Exception as e:  # noqa: BLE001 - 观测不吞 provider 异常，只记录
            ok, fail, err = False, "exception", e.__class__.__name__
            audio, att, trun, retried = 0, 1, False, False
        finally:
            self.last_tts = None  # 消费掉，避免残留污染下一句
        if not obs:
            return
        try:
            from obs import emit_explicit
            from obs.recorder import now_ms
            emit_explicit({
                "type": "tts_call",
                "provider": getattr(self.opt, "tts", None),
                "text": (text or "")[:40], "text_len": len(text or ""),
                "attempts": att,
                "elapsed_ms": round((time.time() - _t0) * 1000, 1),
                "queue_ms": round(now_ms() - obs.get("enqueued_ms", now_ms()), 1),
                "audio_ms": round(audio or 0, 1),
                "success": ok, "fail_reason": fail, "err_type": err,
                "retried": bool(retried), "truncated": bool(trun),
            }, trace_id=obs.get("trace_id"), session_id=obs.get("session_id"),
               parent_id=obs.get("parent_id"), kind="chat")
        except Exception:  # noqa: BLE001 - 观测失败静默，不影响播放
            pass
    
    def txt_to_audio(self, msg: tuple[str, dict]):
        pass

    def stop_tts(self):
        pass
