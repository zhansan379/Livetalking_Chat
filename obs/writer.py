###############################################################################
#  JsonlWriter：事件日志写入 + 大小轮转。
#
#  设计取舍：
#  - 每次写一条独立 open/append/close（沿用 agent/history.py 原子风格），
#    避免 Windows 下对已打开文件 rename 轮转失败；aiohttp 单事件循环内
#    同步写同一条 <writer 不跨协程交错，天然安全，无需锁。
#  - 超 OBS_MAX_MB 时把 events.jsonl 改名为 events-<ts>.jsonl 并重开新文件；
#    不删除（文件小），query 按 seq 统一扫描所有文件，轮转边界的 trace 完整。
#  - 写失败绝不抛异常（观测失败不影响主流程）。
###############################################################################

import json
import os
import time

from .config import get_dir, get_max_bytes, is_enabled


class JsonlWriter:
    def __init__(self) -> None:
        self._size = 0

    def append(self, event: dict) -> None:
        if not is_enabled():
            return
        path = self._base_path()
        line = json.dumps(event, ensure_ascii=False)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            return  # 观测失败静默，不影响主流程

        self._size += len(line) + 1
        if self._size >= get_max_bytes():
            self._rotate(path)

    def _base_path(self) -> str:
        return os.path.join(get_dir(), "events.jsonl")

    def _rotate(self, path: str) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        dst = os.path.join(os.path.dirname(path), f"events-{ts}.jsonl")
        try:
            if not os.path.exists(dst):
                os.rename(path, dst)
                self._size = 0
        except OSError:
            pass  # 轮转失败不致命


def iter_event_files() -> list[str]:
    """返回全部事件文件路径，按文件名排序（越老的 events-<ts> 越早）。"""
    base = get_dir()
    if not os.path.isdir(base):
        return []
    names = sorted(f for f in os.listdir(base) if f.startswith("events") and f.endswith(".jsonl"))
    return [os.path.join(base, n) for n in names]