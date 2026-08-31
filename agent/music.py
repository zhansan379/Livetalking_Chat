###############################################################################
#  音乐播放工具：play_music（工具层通用工具，非能力，与 shutdown_pc 平级）
#
#  在当前这台电脑上播放本地音乐：扫一个音乐目录，支持播放/暂停/继续/停止/
#  上一首/下一首/调音量/列清单/查状态。解码交给系统 PATH 上的 ffmpeg → 原始 PCM，
#  输出走 pyaudio（本机默认声卡，或系统可配的输出设备索引）。
#
#  属「有状态」工具：后台一个播放线程 + 命令队列，模块级单例 music_player。
#  跨调用保持当前曲目/音量/播放中状态，无需落盘。
#
#  音量闪避(ducking)：播放线程轮询存活会话的「是否正在说话」，数字人开口时自动
#  压低音量、闭嘴后回升，降低外放串麦克风对 ASR 的干扰。依赖缺失时闪避静默失效，
#  不影响正常播放。
###############################################################################

import asyncio
import os
import queue
import shutil
import threading
import time

import numpy as np

from utils.logger import logger

# ── 播放参数 ───────────────────────────────────────────────────────────────
_SAMPLE_RATE = 44100
_CHANNELS = 2
# pyaudio 写入块大小（采样点帧数）；越大越省线程切换，越小切歌越跟手
_CHUNK = 4096
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".wma", ".opus"}
_FFMPEG = shutil.which("ffmpeg")


def _now() -> float:
    return time.time()


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class MusicPlayer:
    """有状态本地音乐播放器：后台线程消费命令队列，串行执行播放控制。

    对外只暴露 dispatch()（阻塞取结果）与各命令；全部播放状态驻留在 worker 线程，
    通过命令队列串行化，避免跨线程竞态。命令开闭子进程/音频流都在 worker 内完成。
    """

    def __init__(self):
        self._queue: "queue.Queue" = queue.Queue()
        self._quit = threading.Event()
        self._thread: threading.Thread | None = None

        # ── worker 线程内持有（首条命令时初始化）──
        self._pyaudio = None
        self._stream = None
        self._proc = None
        self._track_path: str | None = None
        self._started_at = 0.0

        # ── 播放状态（worker 读写；dispatch 放进队列取最终结果）──
        self._playlist: list[str] = []
        self._index = 0
        self._paused = False
        self._volume = 0.8          # 基础音量 0..1，仅首启时被配置覆盖，其后只随 volume 命令
        self._configured = False
        self._gain = 0.8            # 当前实际增益（随闪避渐变）

        # ── 音量闪避 ──
        self._duck = True
        self._duck_gain = 0.25      # 数字人说话时压低到的比例
        self._duck_factor = 1.0
        self._last_duck_poll = 0.0

        # ── 配置映射（来自 cfg，首启/变更时同步）──
        self._music_dir: str | None = None
        self._device: int | None = None

    # ── 启动保证 ─────────────────────────────────────────────────────────
    def _ensure_thread(self):
        if self._thread is None or not self._thread.is_alive():
            self._quit.clear()
            self._thread = threading.Thread(target=self._worker, daemon=True,
                                            name="music_player")
            self._thread.start()

    # ── 对外入口（在 asyncio 事件循环线程调用；到 worker 后取回结果字符串）──
    def dispatch(self, action: str, cfg: dict) -> str:
        self._sync_config(cfg)
        self._ensure_thread()
        action = (action or "").strip().lower()
        if not action:
            action = "play"
        evt = threading.Event()
        slot = {}
        self._queue.put((action, cfg, evt, slot))
        if evt.wait(timeout=15):
            return slot.get("result") or ""
        return "（音乐播放器无响应，请稍后再试。）"

    # ── worker 主体 ──────────────────────────────────────────────────────
    def _worker(self):
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
        except Exception as e:  # noqa: BLE001
            logger.warning("music player: pyaudio unavailable: %s", e)
            self._pyaudio = None
        try:
            self._stream = None
            self._proc = None
            if not self._playlist:
                self._playlist = self._scan()
            while not self._quit.is_set():
                self._drain_one()
                if self._paused or self._proc is None:
                    time.sleep(0.03)
                    continue
                self._tick_stream()
        finally:
            self._close_track()
            if self._pyaudio is not None:
                try:
                    self._pyaudio.terminate()
                except Exception:  # noqa: BLE001
                    pass

    def _drain_one(self):
        """从命令队列取一条并执行（非阻塞式，处理完立即返回）。"""
        try:
            action, cfg, evt, slot = self._queue.get_nowait()
        except queue.Empty:
            return
        try:
            slot["result"] = self._handle(action, cfg)
        except Exception as e:  # noqa: BLE001 - 命令失败不拖垮播放线程
            logger.exception("music command %s failed: %s", action, e)
            slot["result"] = f"（音乐工具出错：{e}）"
        finally:
            evt.set()

    # ── 命令派发 ─────────────────────────────────────────────────────────
    def _handle(self, action: str, cfg: dict) -> str:
        if action in ("play", "播放"):
            return self._cmd_play(cfg.get("target"))
        if action in ("pause", "暂停"):
            self._paused = True
            return self._say("已暂停播放。") if self._has_track() else "（当前没有在播放音乐。）"
        if action in ("resume", "resume_play", "继续"):
            if self._proc is not None:
                self._paused = False
                return "（已继续播放。）"
            return self._cmd_play(None)
        if action in ("stop", "停止"):
            self._close_track()
            self._paused = False
            return "（已停止播放，音乐资源已释放。）"
        if action in ("next", "下一首"):
            return self._cmd_next()
        if action in ("prev", "previous", "上一首"):
            return self._cmd_prev()
        if action in ("volume", "vol", "音量"):
            v = int(cfg.get("volume", round(self._volume * 100)))
            self._volume = _clamp(v / 100.0, 0.0, 1.0)
            return f"（音量已设为 {int(round(self._volume * 100))}%。）"
        if action in ("list", "列表", "有啥"):
            return self._cmd_list()
        if action in ("status", "现在", "now", "查"):
            return self._cmd_status()
        return f"（未知的音乐指令『{action}』。）"

    # ── 播放控制（worker 线程内执行）────────────────────────────────────────
    def _has_track(self) -> bool:
        return self._proc is not None and self._track_path is not None

    def _cmd_play(self, target) -> str:
        if not self._playlist:
            self._playlist = self._scan()
        if not self._playlist:
            return f"（音乐目录『{self._scan_dir_label()}』里没有可播放的音频。请先放几首歌再让我播。）"
        if target is not None and str(target).strip():
            idx = self._resolve_index(target)
            if idx is None:
                return f"（没找到叫『{target}』的歌。可以说『列一下有什么音乐』看看目录里有哪些。）"
            self._open_track(idx)
            return self._say(self._now_line())
        # 无目标：暂停中→继续；已有曲目→保持；否则从当前/第一首开始
        if self._paused and self._has_track():
            self._paused = False
            return self._say(f"（已继续播放：{self._track_name()}]）")
        if self._has_track():
            return self._say(f"（正在播放：{self._track_name()}）")
        self._open_track(_clamp(self._index, 0, len(self._playlist) - 1))
        return self._say(self._now_line())

    def _cmd_next(self) -> str:
        if not self._playlist:
            return "（没有可播放的音乐。）"
        if self._proc is None:
            self._index = (self._index + 1) % len(self._playlist)
        else:
            self._index = (self._index + 1) % len(self._playlist)
        self._open_track(self._index)
        return self._say(self._now_line())

    def _cmd_prev(self) -> str:
        if not self._playlist:
            return "（没有可播放的音乐。）"
        self._index = (self._index - 1) % len(self._playlist)
        self._open_track(self._index)
        return self._say(self._now_line())

    def _cmd_list(self) -> str:
        if not self._playlist:
            self._playlist = self._scan()
        if not self._playlist:
            return f"（音乐目录『{self._scan_dir_label()}』里没有可播放的音频。）"
        lines = [f"（音乐目录『{self._scan_dir_label()}』共 {len(self._playlist)} 首："]
        cur = self._track_name()
        for i, p in enumerate(self._playlist, start=1):
            mark = " →" if os.path.basename(p) == cur else ""
            lines.append(f"  {i}. {os.path.basename(p)}{mark}")
        lines.append("说曲名或序号即可播放，如『播放 3』。如需停止说『停音乐会』。）")
        return "\n".join(lines)

    def _cmd_status(self) -> str:
        if not self._has_track():
            return f"（当前没有播放音乐。目录『{self._scan_dir_label()}』共 {len(self._playlist)} 首，说『播放一首』即可。）"
        vol = int(round(self._volume * 100))
        state = "播放中" if not self._paused else "已暂停"
        return f"（当前{state}：{self._track_name()}，音量 {vol}%，目录共 {len(self._playlist)} 首。）"

    # ── 曲目/resolution ────────────────────────────────────────────────────
    def _scan(self) -> list[str]:
        base = self._music_dir
        out = []
        if base and os.path.isdir(base):
            for root, _dirs, fs in os.walk(base):
                for fn in sorted(fs):
                    if os.path.splitext(fn)[1].lower() in _AUDIO_EXTS:
                        out.append(os.path.join(root, fn))
        return sorted(out)

    def _scan_dir_label(self) -> str:
        return self._music_dir if self._music_dir else "（未配置，默认 data/music）"

    def _resolve_index(self, target) -> int | None:
        s = str(target).strip()
        if s.isdigit():
            i = int(s) - 1  # 对外 1-based
            return i if 0 <= i < len(self._playlist) else None
        low = s.lower()
        for i, p in enumerate(self._playlist):
            if low in os.path.basename(p).lower():
                return i
        return None

    def _open_track(self, idx: int):
        idx = _clamp(idx, 0, len(self._playlist) - 1)
        self._close_track()
        path = self._playlist[idx]
        self._index = idx
        self._track_path = path
        self._paused = False
        if not _FFMPEG:
            logger.warning("music: no ffmpeg on PATH, cannot decode %s", path)
            return
        if self._pyaudio is None:
            logger.warning("music: pyaudio unavailable, cannot play %s", path)
            return
        try:
            from subprocess import Popen, PIPE
            self._proc = Popen(
                [_FFMPEG, "-v", "error", "-i", path,
                 "-f", "s16le", "-ar", str(_SAMPLE_RATE), "-ac", str(_CHANNELS),
                 "-acodec", "pcm_s16le", "-", "-nostdin"],
                stdout=PIPE,
            )
            self._stream = self._pyaudio.open(
                rate=_SAMPLE_RATE, channels=_CHANNELS, format=8,  # paInt16==8
                output=True,
                output_device_index=self._device if self._device is not None else None,
            )
            self._started_at = _now()
        except Exception as e:  # noqa: BLE001
            logger.warning("music: failed to open %s: %s", path, e)
            self._close_track()

    def _close_track(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
            self._proc = None
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        self._track_path = None

    def _track_name(self) -> str:
        return os.path.basename(self._track_path) if self._track_path else ""

    def _now_line(self) -> str:
        return f"（正在播放：{self._track_name()}。可以说『暂停』『下一首』『停音乐』『音量 30』。）"

    def _say(self, s: str) -> str:
        return s

    # ── 音量闪避（worker 内每帧调用；配合 _tick_stream 平滑）────────────────
    def _poll_duck(self):
        if not self._duck:
            self._duck_factor = 1.0
            return
        now = _now()
        if now - self._last_duck_poll < 0.1:
            return
        self._last_duck_poll = now
        speaking = False
        try:
            from server.session_manager import session_manager
            for s in getattr(session_manager, "sessions", {}).values():
                if getattr(s, "is_speaking", None) and s.is_speaking():
                    speaking = True
                    break
        except Exception:  # noqa: BLE001 - 依赖缺失则静默关闭闪避，不影响播放
            self._duck = False
            self._duck_factor = 1.0
            return
        self._duck_factor = self._duck_gain if speaking else 1.0

    def _tick_stream(self):
        self._poll_duck()
        target = self._volume * self._duck_factor
        step = 0.02 * max(self._volume, 0.001)
        self._gain = _ramp(self._gain, target, step)
        if self._proc is None or self._stream is None:
            return
        try:
            data = self._proc.stdout.read(_CHUNK)
        except Exception:  # noqa: BLE001
            data = b""
        if not data:
            if self._proc.poll() is not None:
                # 一首放完 → 顺延下一首（自动连播）
                self._close_track()
                if self._playlist:
                    self._index = (self._index + 1) % len(self._playlist)
                    self._open_track(self._index)
                return
            time.sleep(0.01)
            return
        if self._gain < 0.999:
            try:
                buf = np.frombuffer(data, dtype=np.int16).astype(np.float32) * self._gain
                data = np.clip(buf, -32768.0, 32767.0).astype(np.int16).tobytes()
            except Exception:  # noqa: BLE001 - 音量运算失败则原样输出
                pass
        try:
            self._stream.write(data)
        except Exception as e:  # noqa: BLE001
            logger.warning("music: write failed: %s", e)
            self._close_track()

    # ── 配置同步（dispatch 时从 cfg 拉取；仅首启/变更时生效，不覆盖用户调的音量）──
    def _sync_config(self, cfg: dict):
        d = cfg or {}
        if not self._configured:
            self._music_dir = d.get("music_dir") or os.path.join("data", "music")
            self._volume = _clamp(d.get("volume", 0.8), 0.0, 1.0)
            self._gain = self._volume
            self._configured = True
        # 目录/设备/闪避参数每次同步（一般不变；变了下次播放即生效）
        new_dir = d.get("music_dir") or os.path.join("data", "music")
        if new_dir != self._music_dir:
            self._music_dir = new_dir
            self._playlist = self._scan()
        dev = d.get("device")
        self._device = int(dev) if dev is not None else None
        self._duck = bool(d.get("duck", True))
        self._duck_gain = _clamp(d.get("duck_gain", 0.25), 0.05, 1.0)


def _ramp(gain, target, step):
    if abs(target - gain) <= step:
        return target
    return gain + step if target > gain else gain - step


music_player = MusicPlayer()


async def play_music(args: dict, cfg, ctx=None) -> str:
    """播放/暂停/切歌/调音量本地音乐。有状态：跨调用保持当前曲目与音量。"""
    args = args or {}
    action = (args.get("action") or "play").strip().lower() or "play"
    tool_cfg = {
        "music_dir": getattr(cfg, "play_music_dir", None),
        "device": getattr(cfg, "play_music_device", None),
        "duck": bool(getattr(cfg, "play_music_duck", True)),
        "duck_gain": getattr(cfg, "play_music_duck_gain", 0.25),
        "volume": float(getattr(cfg, "play_music_volume", 80) or 80) / 100.0,
        **args,
    }
    # 阻塞式命令派发挪到线程池，避免卡住 asyncio 事件循环
    return await asyncio.to_thread(music_player.dispatch, action, tool_cfg)


def _music_tool_specs() -> list[dict]:
    """音乐播放工具定义（供 TOOL_REGISTRY 注册）。"""
    return [
        {
            "name": "play_music",
            "description": (
                "在【当前这台电脑】上播放本地音乐（背景音乐），支持播放/暂停/继续/停止/"
                "上一首/下一首/调音量/列清单/查状态。只有当用户表达『放首歌 / 放点音乐 / "
                "放点背景音乐 / 我想听歌 / 暂停 / 停止 / 切歌 / 音量调大点 / 有什么歌』等"
                "与听歌有关的意图时才调用；不要因为提到音乐这个词的闲聊就擅自播放。\n"
                "动作（action）：\n"
                "  - play（默认）：开始/继续播放。target 可选，填歌名（模糊匹配文件名）或序号"
                "（1 起，先从 list 看有哪些）；不填则播放当前曲目或目录第一首。\n"
                "  - pause：暂停当前；resume：继续。\n"
                "  - stop：停止并释放；next：下一首；prev：上一首。\n"
                "  - volume：调音量，volume 填 0-100 整数（如『音量 30』→ volume=30）。\n"
                "  - list：列出音乐目录里的歌曲（用这个拿序号/歌名再 play）。\n"
                "  - status：查询当前播放状态与音量。\n"
                "调用后如实把结果（正在播放哪首 / 已暂停 / 已停止 / 已调到的音量 / 目录里有哪些）"
                "告诉用户。音乐来自本地目录，找不到时如实说明，不要编造歌名。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "resume", "stop", "next", "prev", "volume", "list", "status"],
                        "description": "play=播放/继续（默认）；pause=暂停；resume=继续播放；stop=停止；"
                                       "next=下一首；prev=上一首；volume=调音量（配 volume 参数）；"
                                       "list=列出音乐目录；status=查播放状态。",
                    },
                    "target": {
                        "type": "string",
                        "description": "仅 play 使用：要播放的歌名（模糊匹配目录内文件名）或序号（1 起）。"
                                       "留空则播放当前曲目或从第一首开始。",
                    },
                    "volume": {
                        "type": "integer",
                        "description": "仅 volume 使用：目标音量 0-100 整数。",
                    },
                },
                "required": [],
            },
            "handler": play_music,
        },
    ]