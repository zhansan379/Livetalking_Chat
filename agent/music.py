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


def _scan_dir(base) -> list[str]:
    """扫某个目录下所有支持的音频文件（含子目录），排序返回绝对路径。"""
    out = []
    if base and os.path.isdir(base):
        for root, _dirs, fs in os.walk(base):
            for fn in sorted(fs):
                if os.path.splitext(fn)[1].lower() in _AUDIO_EXTS:
                    out.append(os.path.join(root, fn))
    return sorted(out)


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

        # ── 在线播放（配合 meting MCP 工具；在线单曲不进本地 _playlist）──
        self._online = False        # 当前曲目是「在线流」而非本地文件
        self._online_name: str | None = None
        self._online_url: str | None = None   # 在线可播放链接（供循环/续播重播）

        # ── 循环播放：默认只播一遍（列表顺序播到底或单曲放完即停）；
        #    仅当用户明确说「循环播放/单曲循环」时 self._loop=True 才循环 ──
        self._loop = False

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
        if action in ("loop", "循环", "循环播放", "单曲循环", "反复播放", "重播"):
            # 循环开关：配 loop（bool：true=开/false=关）则按值设定；缺省则切换当前状态
            enabled = cfg.get("loop")
            if enabled is None:
                enabled = cfg.get("enabled")
            if enabled is None:
                self._loop = not self._loop
            elif isinstance(enabled, bool):
                self._loop = enabled
            else:
                val = str(enabled).strip().lower()
                self._loop = val in ("1", "true", "yes", "on", "开", "开启", "是", "循环", "单曲循环", "重播")
            state = "已开启" if self._loop else "已关闭"
            suffix = ("放完会自动续播/循环，不会自行停止。" if self._loop
                      else "每首只播一遍，放完即停。（说『循环播放』可开启循环）")
            return self._say(f"（循环播放{state}。{suffix}）")
        if action in ("volume", "vol", "音量"):
            v = int(cfg.get("volume", round(self._volume * 100)))
            self._volume = _clamp(v / 100.0, 0.0, 1.0)
            return f"（音量已设为 {int(round(self._volume * 100))}%。）"
        if action in ("play_online", "在线"):
            # 在线直播：url 来自 meting MCP 的 url 工具（http/https 流）；不落盘
            url = cfg.get("url")
            if not url or not str(url).strip():
                return ("（在线播放需要可播放链接：请用 mcp_meting_ 系列工具先按平台搜到歌曲、"
                        "并用 url 工具拿到可播放地址。）")
            if self._open_url(str(url), str(cfg.get("name") or "")):
                return self._say(self._now_line())
            return "（在线音乐播放失败：链接取不到音频或设备不可用。）"
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
            loop = "，循环已开" if self._loop else ""
            return f"（当前没有播放音乐{loop}。目录『{self._scan_dir_label()}』共 {len(self._playlist)} 首，说『播放一首』即可。）"
        vol = int(round(self._volume * 100))
        state = "播放中" if not self._paused else "已暂停"
        loop = "，循环已开" if self._loop else ""
        return f"（当前{state}：{self._track_name()}，音量 {vol}%，目录共 {len(self._playlist)} 首{loop}。）"

    # ── 曲目/resolution ────────────────────────────────────────────────────
    def _scan(self) -> list[str]:
        return _scan_dir(self._music_dir)

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
        self._start_ffmpeg(path)

    def _open_url(self, url: str, name: str):
        """在线直播：直接流式解码一个可播放链接（http/https），不落盘。

        用于与 meting MCP 打通：模型先用 mcp_meting_search 找到歌曲、再用
        mcp_meting_url 拿可播放链接，这里喂给 ffmpeg 在线解码。在线单曲属于
        「外来曲目」，不进本地 _playlist（next/prev 仍作用在本地列表上）。
        """
        self._close_track()
        if not url or not str(url).strip():
            return False
        self._online = True
        self._online_url = str(url)
        self._online_name = (name or "").strip() or "在线音乐"
        self._paused = False
        return self._start_ffmpeg(str(url))

    def _start_ffmpeg(self, source: str) -> bool:
        """以 source（本地路径或在线 URL）起一个 ffmpeg→pyaudio 播放子进程。"""
        if not _FFMPEG:
            logger.warning("music: no ffmpeg on PATH, cannot decode %s", source)
            return False
        if self._pyaudio is None:
            logger.warning("music: pyaudio unavailable, cannot play %s", source)
            return False
        try:
            from subprocess import Popen, PIPE
            self._proc = Popen(
                [_FFMPEG, "-v", "error", "-i", source,
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
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("music: failed to open %s: %s", source, e)
            self._close_track()
            return False

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
        self._online = False
        self._online_name = None
        self._online_url = None

    def _download_rescan(self, base: str):
        """下载成功后把新文件并入本地列表，让它可以立即『播放』。

        不依赖 _music_dir 是否已同步过配置：直接增量并入 base 目录下已有的音频，
        因此下载是启动后第一条指令时也能正确离线播放。
        """
        try:
            known = {os.path.abspath(p) for p in self._playlist}
            existing = set(os.path.abspath(p) for p in _scan_dir(base))
            for p in sorted(existing - known):
                self._playlist.append(p)
        except Exception:  # noqa: BLE001 - 刷新失败不致命，下次扫描会补上
            logger.warning("music: playlist rescan after download failed", exc_info=True)

    def _track_name(self) -> str:
        if self._online and self._online_name:
            return self._online_name
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
                # 一曲放完。
                #   - 默认（非循环）：只播放一遍。本地列表顺序播到最后一首即停；
                #     在线单曲放完即停（不进本地列表）。
                #   - 循环模式（用户说「循环播放」后 self._loop=True）：
                #     本地列表接回第一首继续（列表循环）；在线单曲循环重播同一链接。
                was_online = self._online
                url = self._online_url if was_online else None
                name = self._online_name
                self._close_track()
                if self._loop:
                    if was_online and url:
                        self._open_url(url, name)
                    elif not was_online and self._playlist:
                        self._index = (self._index + 1) % len(self._playlist)
                        self._open_track(self._index)
                else:
                    if not was_online and self._playlist and self._index < len(self._playlist) - 1:
                        self._index += 1
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
    # 专属「下载」分支：不占用播放器 worker 队列，独立后台下载，避免阻塞播放
    if action in ("download", "下载", "存音乐"):
        ddir = getattr(cfg, "play_music_download_dir", None) or _download_dir()
        return await asyncio.to_thread(download_music, args, ddir)
    # 阻塞式命令派发挪到线程池，避免卡住 asyncio 事件循环
    return await asyncio.to_thread(music_player.dispatch, action, tool_cfg)


# ── 下载到本地（离线收听）────────────────────────────────────────────────────
# Windows 文件名非法字符（路径分隔 / 控制字符 / 保留符号）——其余一律保留，含中文歌名
_UNSAFE_FN_CHARS = frozenset('/\\:*?"<>|') | frozenset(
    chr(c) for c in range(0, 32)
)


def _download_dir() -> str:
    return os.path.join("data", "music")


def _sanitize_filename(name: str) -> str:
    """把歌名清洗成可安全落盘的文件名（保留中文，仅去掉路径分隔符/控制字符/尾点）。"""
    name = (name or "").strip()
    # 去掉可能跟的扩展名
    base, _ext = os.path.splitext(name)
    cleaned = "".join("_" if ch in _UNSAFE_FN_CHARS else ch for ch in base).strip(" ._")
    return cleaned or "在线音乐"


def _probe_ok(path: str) -> bool:
    """用 ffmpeg 把文件解到空掷掉：能正常解出音频（退出码 0 且产出时长）→ 可用。"""
    if not _FFMPEG or not os.path.isfile(path) or os.path.getsize(path) < 1024:
        return False
    try:
        from subprocess import Popen, PIPE, DEVNULL
        p = Popen(
            [_FFMPEG, "-v", "error", "-i", path, "-f", "null", "-"],
            stdout=DEVNULL, stderr=PIPE,
        )
        _all, err = p.communicate(timeout=30)
    except Exception:  # noqa: BLE001
        return False
    if p.returncode != 0:
        logger.warning("music: download verify failed (no decodable audio): %s", err[:200])
        return False
    return True


def download_music(args: dict, ddir: str) -> str:
    """把 meting 拿到的在线播放链接存成本地文件（离线收听），成功后进本地列表。

    优先原流 `-c copy`（快、保原样）；若落盘文件非可解码音频（平台加密/毒化流），
    回退转码成标准 mp3；仍失败则如实返回，不编造已下载。
    """
    url = (args or {}).get("url") or ""
    if not str(url).strip():
        return ("（下载需要可播放链接：请先用 mcp_meting_search 搜到歌曲、"
                "再用 mcp_meting_url 拿到可播放地址，然后说『下载《歌名》』。）")
    name = str(args.get("name") or "").strip() or "在线音乐"
    if not _FFMPEG:
        return "（下载失败：这台电脑没有 ffmpeg，无法把在线音频存成文件。）"

    base = os.path.abspath(ddir or _download_dir())
    os.makedirs(base, exist_ok=True)
    dest = os.path.join(base, _sanitize_filename(name) + ".mp3")

    # 1) 先试原流直存（不重编码，保留原始质量/速度）
    try:
        from subprocess import Popen, PIPE
        p = Popen(
            [_FFMPEG, "-v", "error", "-y", "-i", str(url), "-c", "copy", dest],
            stdout=PIPE, stderr=PIPE,
        )
        _o, err = p.communicate(timeout=90)
    except Exception as e:  # noqa: BLE001
        logger.warning("music: download(copy) exc: %s", e)
        p, err = None, ""

    if p is not None and p.returncode == 0 and _probe_ok(dest):
        return _download_done(dest, ddir, copied=True)

    # 2) 原流不可/被加密 → 回退重编码成标准 mp3
    try:
        from subprocess import Popen, PIPE
        p = Popen(
            [_FFMPEG, "-v", "error", "-y", "-i", str(url),
             "-acodec", "libmp3lame", "-b:a", "192k", dest],
            stdout=PIPE, stderr=PIPE,
        )
        _o, err = p.communicate(timeout=120)
    except Exception as e:  # noqa: BLE001
        logger.warning("music: download(reencode) exc: %s", e)
        p, err = None, ""

    if p is not None and p.returncode == 0 and _probe_ok(dest):
        return _download_done(dest, ddir, copied=False)
    # 清掉失败产生的残留文件
    if os.path.isfile(dest):
        try:
            os.remove(dest)
        except Exception:  # noqa: BLE001
            pass
    detail = (err or b"").decode("utf-8", "replace")[:120] if isinstance(err, (bytes, bytearray)) \
        else str(err or "")[:120]
    return f"（下载《{name}》失败：该链接无法解出可播放音频（{detail.strip()}）。）"


def _download_done(dest: str, ddir: str, copied: bool) -> str:
    """下载成功后把文件并入本地列表（离线可直接播），返回所存位置说明。"""
    # 让播放列表立即感知新文件（url 是 dst，覆盖可能已存在的同名）
    music_player._download_rescan(os.path.dirname(dest))
    mode = "原样备份" if copied else "已转码成 mp3"
    return (f"（已把《{os.path.basename(dest)}》下载到本地（{mode}，"
            f"存入：{dest}）。现在可以直接说『播放 {os.path.splitext(os.path.basename(dest))[0]} 』离线收听。）")


def _music_tool_specs() -> list[dict]:
    """音乐播放工具定义（供 TOOL_REGISTRY 注册）。"""
    return [
        {
            "name": "play_music",
            "description": (
                "在【当前这台电脑】上播放音乐（背景音乐），支持播放/暂停/继续/停止/"
                "上一首/下一首/调音量/列清单/查状态，也支持播放【在线音乐】。只有当用户表达"
                "『放首歌 / 放点音乐 / 放点背景音乐 / 我想听歌 / 暂停 / 停止 / 切歌 / 音量调大点 "
                "/ 有什么歌 / 帮我搜首XX听』等与听歌有关的意图时才调用；不要因为提到音乐这个词的闲聊就擅自播放。\n"
                "本地播放（action=play，默认）：target 可选，填歌名（模糊匹配文件名）或序号"
                "（1 起，先从 list 看有哪些）；不填则播放当前曲目或目录第一首。\n"
                "在线播放（action=play_online）：当用户点名的歌本地目录里没有、或明确要播网上某首歌时，"
                "先按平台用 mcp_meting_search 搜到歌曲拿到 song id，再用 mcp_meting_url 拿可播放链接，"
                "最后调本工具 action=play_online、url=该链接、name=歌名（可选）。"
                "注意该链接是短期有效的，拿到后应立即播放。\n"
                "播放模式：默认【只播放一遍】——本地列表顺序播到最后一首即停、在线单曲放完即停。"
                "只有用户明确说『循环播放 / 单曲循环』时才用 action=loop 开启循环："
                "本地列表循环续播、在线单曲循环重播。要关闭说『取消循环』即可（action=loop 不再传 loop，或 loop=false）。\n"
                "动作（action）：\n"
                "  - play（默认）：开始/继续播放本地或当前曲目。\n"
                "  - loop：切换循环播放（配 loop=true/false 显式开/关；不配则切换当前状态）。\n"
                "  - play_online：在线播放。配 url 参数（必填，mcp_meting_url 拿到的可播放链接）、"
                "name（可选，显示用歌名，如『稻香』）。\n"
                "  - download：把在线链接下载保存到本地（离线收听，自己已购买的歌曲）。"
                "配 url（必填，同 play_online）+ name（歌名，会成为保存的文件名）。"
                "下载成功后会进本地音乐列表，之后离线可直接说『播放《歌名》』。"
                "优先原样存档；链接被平台加密/非标准音频时自动转码成 mp3，仍失败则如实说明。\n"
                "  - pause：暂停当前；resume：继续。\n"
                "  - stop：停止并释放；next：下一首；prev：上一首。\n"
                "  - volume：调音量，volume 填 0-100 整数（如『音量 30』→ volume=30）。\n"
                "  - download：下载保存到本地（离线收听），配 url + name。\n"
                "  - list：列出本地音乐目录里的歌曲（用这个拿序号/歌名再 play）。\n"
                "  - status：查询当前播放状态与音量。\n"
                "调用后如实把结果（正在播放哪首 / 已暂停 / 已停止 / 已调到的音量 / 目录里有哪些）"
                "告诉用户。本地找不到的音乐与在线播放失败都要如实说明，不要编造歌名或声称已播放。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "play_online", "download", "pause", "resume", "stop", "next", "prev", "volume", "list", "status", "loop"],
                        "description": "play=播放/继续（默认）；play_online=在线播放（配 url 参数）；"
                                       "download=下载到本地离线收听（配 url + name）；"
                                       "pause=暂停；resume=继续播放；stop=停止；next=下一首；prev=上一首；"
                                       "volume=调音量（配 volume 参数）；list=列出本地音乐目录；status=查播放状态；"
                                       "loop=循环播放开关（配 loop=true/false 开/关；不配则切换，仅用户点名『循环播放』时用）。",
                    },
                    "target": {
                        "type": "string",
                        "description": "仅 play 使用：要播放的歌名（模糊匹配目录内文件名）或序号（1 起）。"
                                       "留空则播放当前曲目或从第一首开始。",
                    },
                    "url": {
                        "type": "string",
                        "description": "play_online / download 使用：可播放的在线链接（http/https），来自"
                                       "mcp_meting_url 工具。链接短期有效，应立即使用。",
                    },
                    "name": {
                        "type": "string",
                        "description": "play_online 显示用 / download 作保存文件名（如『稻香』）："
                                       "用于查状态/提示展示；缺省显示『在线音乐』。",
                    },
                    "volume": {
                        "type": "integer",
                        "description": "仅 volume 使用：目标音量 0-100 整数。",
                    },
                    "loop": {
                        "type": "boolean",
                        "description": "仅 loop 使用：true=开启循环播放（放完自动续播/重播）；"
                                       "false=关闭（每首只播一遍）。缺省则切换当前循环状态。",
                    },
                },
                "required": [],
            },
            "handler": play_music,
        },
    ]