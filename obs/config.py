###############################################################################
#  观测平台配置：全部 env 驱动，无 config.yaml 改动，零风险。
#  沿袭 agent/history.py 的 AGENT_HISTORY_DIR env 覆盖先例。
#
#    OBS_ENABLED       观测开关（默认 1；0/false/no 关闭 → 所有 obs.* 为空操作）
#    OBS_DIR           事件日志目录（默认 data/obs）
#    OBS_MAX_MB        events.jsonl 大小轮转阈值（MB，默认 50）
#    OBS_QUERY_WINDOW  面板 summary 默认时间窗口（秒，默认 3600）
#    OBS_QUERY_LIMIT   /api/obs/requests 默认条数（默认 50）
###############################################################################

import os


def _falsey(v: str) -> bool:
    return (v or "").strip().lower() in ("0", "false", "no", "off", "")


def is_enabled() -> bool:
    return not _falsey(os.environ.get("OBS_ENABLED", "1"))


def get_dir() -> str:
    return os.environ.get("OBS_DIR", "data/obs") or "data/obs"


def get_max_bytes() -> int:
    try:
        mb = float(os.environ.get("OBS_MAX_MB", "50") or "50")
    except ValueError:
        mb = 50.0
    return max(1, int(mb * 1024 * 1024))


def query_window() -> int:
    try:
        return int(os.environ.get("OBS_QUERY_WINDOW", "3600") or "3600")
    except ValueError:
        return 3600


def query_limit() -> int:
    try:
        return int(os.environ.get("OBS_QUERY_LIMIT", "50") or "50")
    except ValueError:
        return 50