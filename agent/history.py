###############################################################################
#  会话历史持久化：完整转录 + 压缩摘要 + 压缩水位
#
#  JSON schema（data/chat_history/<session_id>.json）：
#    {
#      "session_id": "...",
#      "created_at": "...",
#      "updated_at": "...",
#      "summary": "累积的压缩摘要（有界）",
#      "last_compressed_index": 0,
#      "messages": [ {"role": "user", "content": "..."}, ... ]   # 完整原文，append-only
#    }
###############################################################################

import datetime
import json
import os
import tempfile

from utils.logger import logger

_DEFAULT_DIR = os.path.join("data", "chat_history")


def _history_dir() -> str:
    """历史文件目录，可用 AGENT_HISTORY_DIR 环境变量覆盖。"""
    return os.environ.get("AGENT_HISTORY_DIR", _DEFAULT_DIR)


def _history_path(session_id: str) -> str:
    # sessionid 一般是 uuid，但做一层安全清洗，避免路径注入
    safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
    return os.path.join(_history_dir(), f"{safe_id or 'default'}.json")


def _now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def history_path_for(session_id: str) -> str:
    """暴露文件路径（便于调试/测试）。"""
    return _history_path(session_id)


def load_history(session_id: str) -> tuple[str, int, list[dict]]:
    """加载会话历史。

    :return: (summary, last_compressed_index, messages)
             文件缺失或损坏时返回 ("", 0, [])
    """
    path = _history_path(session_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        messages = data.get("messages") or []
        if not isinstance(messages, list):
            messages = []
        summary = data.get("summary") or ""
        last_idx = int(data.get("last_compressed_index") or 0)
        return str(summary), max(0, last_idx), messages
    except FileNotFoundError:
        return "", 0, []
    except Exception as e:  # noqa: BLE001 - 损坏文件不应导致服务崩溃
        logger.warning("history load failed for %s (%s), reset", session_id, e)
        return "", 0, []


def save_history(
    session_id: str,
    summary: str,
    last_compressed_index: int,
    messages: list[dict],
) -> None:
    """原子写入完整会话历史。"""
    path = _history_path(session_id)
    document = {
        "session_id": session_id,
        "created_at": _existing_created(path) or _now_iso(),
        "updated_at": _now_iso(),
        "summary": summary or "",
        "last_compressed_index": int(last_compressed_index or 0),
        "messages": messages or [],
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=".history-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # 原子替换，避免写一半损坏
    except Exception:
        # 替换失败时清理临时文件，不污染历史
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _existing_created(path: str) -> str | None:
    """已存在文件的 created_at（若可读），保持首次创建时间稳定。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("created_at")
    except Exception:  # noqa: BLE001
        return None


def delete_history(session_id: str) -> None:
    """删除某会话历史（测试/维护用）。"""
    try:
        os.remove(_history_path(session_id))
    except FileNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("history delete failed for %s (%s)", session_id, e)