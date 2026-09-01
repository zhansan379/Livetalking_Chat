###############################################################################
#  关键词唤醒（KWS）服务 — 基于 sherpa-onnx 开放词表关键词语音识别
#
#  为什么放在后端而不是浏览器：
#    1) sherpa-onnx 的 KWS 官方一等公民是 CLI/Python（浏览器需自编 WASM）；
#    2) 浏览器侧若用「整段 ASR 去匹配」做唤醒，非常驻、高功耗，不是真唤醒词。
#  这儿用真 KWS：只匹配给定关键词，命中才给浏览器一个 {type:'wake'} 事件。
#
#  数据流：
#    浏览器(VAD 有声时) 16k PCM16 ──WS /api/kws/ws──▶ 本服务
#        accept_waveform → decode → get_result → 命中回 {type:'wake', keyword}
#
#  可用性：模型目录缺 / sherpa 未装 → available=False，前端自动降级为普通通话，
#  不破坏既有语音对话（功能可用性与本服务解耦）。
###############################################################################

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from aiohttp import web

from utils.logger import logger

# 拼音声母（最长优先），用于把中文关键词转成该模型认识的 pinyin token 串。
# 该 wenetspeech KWS 模型的建模单元是 拼音(声母+韵母)，见模型 README。
_PY_INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
                "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

# 模型目录：优先环境变量 KWS_MODEL_DIR，否则项目下 models/kws/<模型名>。
# 目录内需含 encoder-*.onnx / decoder-*.onnx / joiner-*.onnx / tokens.txt。
_DEFAULT_MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "kws"
_DEFAULT_MODEL_ID = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"

_KWS_SCORE = 2.0        # 关键词 boosting：越大越容易在 beam search 中活下来（提升召回/命中率）
_KWS_THRESHOLD = 0.25   # 触发阈值：越大越难触发（防误报）；仍觉漏报可下调
_MAX_ACTIVE_PATHS = 4
_WIN_COOLDOWN_S = 3.0   # 命中一次后的冷却，防止一句内连触发


def _resolve_model_dir() -> Path | None:
    """定位模型目录；取不到（未配置、目录不存在）返回 None。"""
    env = os.environ.get("KWS_MODEL_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
        logger.warning("[KWS] KWS_MODEL_DIR=%s 不存在，回退默认", env)
    cand = _DEFAULT_MODEL_ROOT / _DEFAULT_MODEL_ID
    return cand if cand.is_dir() else None


class KwsService:
    """封装 sherpa-onnx KeywordSpotter 的单例；模型就绪才 available。"""

    def __init__(self):
        self._lock = threading.Lock()   # 序列化 decode（单 spotter 多连接共享，防并发竞争）
        self._spotter = None
        self._model_dir: Path | None = None
        self.available = False
        self._init()

    # ── 初始化 ─────────────────────────────────────────────────────────────
    def _init(self):
        model_dir = _resolve_model_dir()
        if model_dir is None:
            logger.warning(
                "[KWS] 模型目录缺失（%s），关键词唤醒停用，前端将降级为普通通话。"
                "放入模型文件后重启即可启用。", _DEFAULT_MODEL_ROOT
            )
            return
        try:
            import sherpa_onnx  # noqa: F401  # 延迟导入：缺依赖也只影响本服务
        except Exception as e:  # noqa: BLE001
            logger.warning("[KWS] sherpa_onnx 未安装（%s），关键词唤醒停用", e)
            return

        enc = _first("encoder", model_dir)
        dec = _first("decoder", model_dir)
        joiner = _first("joiner", model_dir)
        tokens = _first("tokens.txt", model_dir)
        if not (enc and dec and joiner and tokens):
            logger.warning("[KWS] 模型文件不全（缺 encoder/decoder/joiner/tokens.txt），停用")
            return

        # sherpa-onnx KeyWordSpotter 构造要求 keywords_file 存在；用一个占位空文件满足。
        placeholder = _DEFAULT_MODEL_ROOT / "_keywords.txt"
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        if not placeholder.exists():
            placeholder.write_text("", encoding="utf-8")

        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
                tokens=str(tokens), encoder=str(enc), decoder=str(dec),
                joiner=str(joiner), keywords_file=str(placeholder),
                num_threads=2, sample_rate=16000, max_active_paths=_MAX_ACTIVE_PATHS,
                keywords_score=_KWS_SCORE, keywords_threshold=_KWS_THRESHOLD,
                provider="cpu",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[KWS] KeywordSpotter 初始化失败，停用: %s", e)
            return

        self._model_dir = model_dir
        self.available = True
        logger.info("[KWS] 就绪 model_dir=%s", model_dir)

    # ── 关键词处理 ─────────────────────────────────────────────────────────
    def tokenize(self, word: str) -> str:
        """把中文自由文本关键词转成该模型认识的 pinyin token 串（空格分隔）。

        该 wenetspeech KWS 模型建模单元为拼音（声母+韵母），每个汉字拆成
        「声母 + 一个带调韵母 token」。用 pypinyin 出带调音节后切出声母，
        韵母尽量在模型 token 词表里匹配（已对照模型自带 keywords.txt 全量校验）。
        无法覆盖的字返回空串（不影响其它在词表内的字触发）。返回的字符串末尾
        带空格交给调用方拼内联格式，这里只含 token 本身。
        """
        word = (word or "").strip()
        if not word:
            return ""
        try:
            from pypinyin import Style, pinyin
        except Exception:  # noqa: BLE001  # pypinyin 缺失则退化为字切（尽力而为）
            vocab = self._vocab()
            return " ".join(c for c in word if c.strip() and c in vocab)

        vocab = self._vocab()
        parts = []
        for cand in pinyin(word, style=Style.TONE, errors="default"):
            syl = cand[0] if cand else ""
            ini, fin = self._split_initial(syl)
            if ini:
                parts.append(ini)
            if fin in vocab:
                parts.append(fin)
            else:
                for seg in self._greedy_final(fin, vocab):
                    parts.append(seg)
        return " ".join(parts)

    @staticmethod
    def _split_initial(syl: str) -> tuple:
        """从带调音节里切出声母，返回 (声母, 韵母)；无独立声母则声母为空串。"""
        for ini in _PY_INITIALS:
            if syl.startswith(ini) and len(syl) > len(ini):
                return ini, syl[len(ini):]
        return "", syl

    @staticmethod
    def _greedy_final(final: str, vocab: set) -> list:
        """韵母串不在词表时，按词表最长贪心切成分片，切不完退回逐字符。"""
        out, i = [], 0
        while i < len(final):
            for j in range(len(final), i, -1):
                if final[i:j] in vocab:
                    out.append(final[i:j])
                    i = j
                    break
            else:
                out.append(final[i])
                i += 1
        return out

    def _vocab(self) -> set:
        tokens_path = self._model_dir / "tokens.txt" if self._model_dir else None
        if tokens_path and hasattr(self, "_vocab_cache"):
            return self._vocab_cache
        vocab: set = set()
        if tokens_path and tokens_path.is_file():
            try:
                for line in tokens_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        vocab.add(line.split()[0])
            except Exception as e:  # noqa: BLE001
                logger.warning("[KWS] 读取 tokens.txt 失败: %s", e)
        self._vocab_cache = vocab
        return vocab

    def build_inline_keywords(self, words) -> str:
        """把关键词列表转成 sherpa create_stream(stream_keywords) 的内联格式。

        每条:  token串 + " @原文"；多条用 "/" 分隔（对齐 sherpa-onnx 内联关键词语法）。
        """
        entries = []
        for w in (words or []):
            w = (w or "").strip()
            if not w:
                continue
            tokens = self.tokenize(w)
            if not tokens:
                logger.warning("[KWS] 关键词「%s」无可用 token，忽略", w)
                continue
            entries.append(f"{tokens} @{w}")
        return " / ".join(entries)

    # ── 推理（供 WS handler 调用；阻塞，需在 executor 中执行）──────────────
    def create_stream(self, stream_keywords: str):
        """为一次连接建立带关键词的持久流（每次连接只有一条，跨 chunk 增量解码）。"""
        if not self.available or self._spotter is None:
            return None
        with self._lock:
            try:
                return self._spotter.create_stream(stream_keywords or None)
            except Exception as e:  # noqa: BLE001
                logger.warning("[KWS] create_stream 异常: %s", e)
        return None

    def feed_stream(self, stream, pcm16: bytes) -> str | None:
        """向已有流喂一段 PCM16，解码并检测关键词；命中返回关键词文本，否则 None。

        持久流的增量喂入保证跨 chunk 的关键词（非整段在一次 chunk 内）也能被识别。
        需要在 asyncio.run_in_executor 中调用（避免阻塞事件循环）。
        """
        if not self.available or self._spotter is None or stream is None:
            return None
        n = len(pcm16)
        if n % 2 != 0:
            pcm16 = pcm16[:-1]
            n -= 1
        if n <= 0:
            return None
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0

        with self._lock:
            try:
                stream.accept_waveform(16000, audio)
                while self._spotter.is_ready(stream):
                    self._spotter.decode_stream(stream)
                    r = self._spotter.get_result(stream)
                    if r:
                        self._spotter.reset_stream(stream)
                        return r
            except Exception as e:  # noqa: BLE001
                logger.warning("[KWS] 推理异常: %s", e)
        return None


# ── 单例 ───────────────────────────────────────────────────────────────────
_service: KwsService | None = None


def get_kws_service() -> KwsService:
    global _service
    if _service is None:
        _service = KwsService()
    return _service


def _first(prefix: str, model_dir: Path):
    """在模型目录里按前缀找模型文件（实际文件名含 epoch/chunk 后缀）。

    优先取非 int8（fp32）精度版本以提高识别准确率——int8 量化版在字母序上排在
    fp32 前（`...int8.onnx` 的 `i` < `...onnx` 的 `o`），需显式后置。epoch 取最前
    值（官方示例用 epoch-12-avg-2）。
    """
    if prefix == "tokens.txt":
        target = model_dir / "tokens.txt"
        return target if target.is_file() else None
    # 按 (是否int8, 文件名) 排序：fp32 优先，同精度下字母序（epoch-12 在 epoch-99 前）
    cands = sorted((s.as_posix() for s in model_dir.glob(f"{prefix}*.onnx")),
                   key=lambda p: (".int8." in p, p))
    return Path(cands[0]) if cands else None


###############################################################################
#  WebSocket handler ─ 浏览器流式上送 16k PCM16，命中回 {type:'wake', keyword}
###############################################################################

async def kws_websocket_handler(request):
    """仿 asr/asr/handler.py 与 camera_websocket_handler 的 aiohttp WS 范式。

    协议：
      1) 开连接 → 浏览器发 {keywords:["小智同学", ...]} → 回 {type:'ready'}
      2) 持续发二进制 PCM16(16000Hz) → 命中回 {type:'wake', keyword}
      3) 模型不可用/异常 → 回 {type:'error', reason} 并断开
    """
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    svc = get_kws_service()
    if not svc.available:
        logger.warning("[KWS] 服务不可用，拒绝连接（前端应降级为普通通话）")
        try:
            await ws.send_str('{"type":"error","reason":"kws_unavailable"}')
        except Exception:  # noqa: BLE001
            pass
        await ws.close()
        return ws

    session_id = (request.query.get("session") or "").strip() or "0"
    stream_keywords = ""          # 当前连接的关键词内联串
    stream = None                 # 本次连接的持久解码流（跨 chunk 增量喂入）
    cool_down_until = 0.0         # 命中冷却，防一句连触发
    loop = asyncio.get_event_loop()

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(data, dict) and "keywords" in data:
                    stream_keywords = svc.build_inline_keywords(data.get("keywords"))
                    if stream:
                        with svc._lock:
                            stream = None  # 释放旧流占用的上下文
                    stream = await loop.run_in_executor(None, svc.create_stream, stream_keywords)
                    await ws.send_str(json.dumps(
                        {"type": "ready", "keywords": data.get("keywords", [])}))
                    logger.info("[KWS] %s 连接就绪，关键词=%s", session_id, data.get("keywords"))

            elif msg.type == web.WSMsgType.BINARY:
                if time.perf_counter() < cool_down_until:
                    continue
                detected = await loop.run_in_executor(
                    None, svc.feed_stream, stream, msg.data)
                if detected:
                    cool_down_until = time.perf_counter() + _WIN_COOLDOWN_S
                    logger.info("[KWS] session=%s 命中唤醒词「%s」", session_id, detected)
                    await ws.send_str(json.dumps(
                        {"type": "wake", "keyword": detected}))

            elif msg.type in (web.WSMsgType.ERROR, web.WSMsgType.CLOSE):
                break

    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.exception("[KWS] handler 异常: %s", e)

    logger.info("[KWS] %s 连接关闭", session_id)
    return ws