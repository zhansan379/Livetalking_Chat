###############################################################################
#  对话/记忆配置加载器：读取 agent_config.yaml，缺失字段回退默认值
###############################################################################

import os

import yaml

from utils.logger import logger

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_config.yaml")

_DEFAULT = {
    "system_prompt": "你是一个知识助手，尽量以简短、口语化的方式输出。只使用纯文本和正常的标点符号，不要使用任何表情符号/表情包、emoji、Markdown 记号（如 *、**、#、`、_）或其他特殊符号。\n\n用户的提问是通过语音识别（语音转写）输入的，技术名词、专有名词、人名、程序名、版本号、单词等容易因同音或近音被听错/写成错字。请结合上下文与常识判断用户实际想说的正确内容，并按其真实意图作答；直接回答，不要复读或引用原文里疑似听错的字，也不要反问'你是不是想说XX'。\n\n凡是涉及用户自身外貌、身份或当下状态的问题（性别、年龄、长相、发型打扮、佩戴什么、情绪如何等），应优先调用 look_at_user 工具获取实时画面后再作答；以实时画面为准，不要根据对话历史里已经叙述过的视觉细节来推断或『替你在看』。若画面拿不到，如实说明看不到，不要臆测或编造。",
    "memory": {
        "enabled": True,
        "compress_threshold": 10,
        "keep_recent": 5,
        "target_summary_chars": 600,
        "summarize_prompt": "请把以下对话历史压缩成一段简短的要点总结，保留关键信息；直接输出总结，不要加任何前缀或说明。",
        "summarize_model": None,
        "longterm": {
            "enabled": True,
            "dir": None,
            "store_types": ["user", "feedback", "project", "reference", "state"],
            "recall_mode": "auto",
            "recall_top_k": 3,
            "recall_char_limit": 800,
            "recall_score_backend": "auto",
            "extract_model": None,
            "extract_trigger": "every_turn",
            "extract_every_n_turns": 0,
            "consolidate_threshold": 10,
        },
    },
    "tools": {
        "max_rounds": 4,
        "web_search": {
            "enabled": True,
            "max_sources": 5,
        },
        "weather": {
            "enabled": True,
            "forecast_base_url": "https://api.open-meteo.com",
            "geocoding_base_url": "https://geocoding-api.open-meteo.com",
            "coords": {},
        },
        "reminder": {
            "enabled": True,
            "store_path": "data/reminders.json",
            "max_delay_seconds": 604800,
        },
        "look_at_user": {
            "enabled": False,
        },
    },
}


def _merge(base: dict, override: dict) -> dict:
    """浅合并 dict，override 优先；嵌套 dict 递归合并。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


class AgentConfig:
    """agent 运行期配置（含 memory 子配置）。"""

    def __init__(self, data: dict):
        self._data = data
        mem = data.get("memory", {}) or {}

        self.system_prompt: str = data.get("system_prompt", _DEFAULT["system_prompt"])

        self.memory_enabled: bool = mem.get("enabled", True)
        self.compress_threshold: int = int(mem.get("compress_threshold", 10) or 10)
        self.keep_recent: int = int(mem.get("keep_recent", 5) or 5)
        self.target_summary_chars: int = int(mem.get("target_summary_chars", 600) or 600)
        self.summarize_prompt: str = mem.get(
            "summarize_prompt", _DEFAULT["memory"]["summarize_prompt"]
        )
        self.summarize_model: str | None = mem.get("summarize_model") or None

        # —— 跨会话长期记忆（memory.longterm）——
        longterm = mem.get("longterm", {}) or {}
        self.longterm_enabled: bool = bool(longterm.get("enabled", True))
        # 记忆目录：null → 走 AGENT_MEMORY_DIR env 或默认 data/memory
        self.longterm_dir: str | None = longterm.get("dir") or None
        self.longterm_store_types: list[str] = list(
            longterm.get("store_types") or _DEFAULT["memory"]["longterm"]["store_types"]
        )
        # 召回路径：auto→轻量即时注入；model→s09 模型选择
        self.longterm_recall_mode: str = longterm.get("recall_mode", "auto") or "auto"
        self.longterm_recall_top_k: int = int(longterm.get("recall_top_k", 3) or 3)
        self.longterm_recall_char_limit: int = int(
            longterm.get("recall_char_limit", 800) or 800
        )
        # 打分后端：auto→rerank 首选、失败退关键词；keyword→纯关键词
        self.longterm_recall_backend: str = longterm.get(
            "recall_score_backend", "auto"
        ) or "auto"
        self.longterm_extract_model: str | None = longterm.get("extract_model") or None
        self.longterm_extract_trigger: str = longterm.get(
            "extract_trigger", "every_turn"
        ) or "every_turn"
        self.longterm_extract_every_n: int = int(
            longterm.get("extract_every_n_turns", 0) or 0
        )
        self.longterm_consolidate_threshold: int = int(
            longterm.get("consolidate_threshold", 10) or 10
        )

        tools = data.get("tools", {}) or {}
        ws = tools.get("web_search", {}) or {}
        self.tool_max_rounds: int = int(tools.get("max_rounds", 4) or 4)
        self.tool_web_search_enabled: bool = bool(ws.get("enabled", True))
        self.tool_web_search_max_sources: int = int(ws.get("max_sources", 5) or 5)

        w_weather = tools.get("weather", {}) or {}
        self.tool_weather_enabled: bool = bool(w_weather.get("enabled", False))
        self.weather_forecast_base_url: str = w_weather.get(
            "forecast_base_url", "https://api.open-meteo.com"
        ) or "https://api.open-meteo.com"
        self.weather_geocoding_base_url: str = w_weather.get(
            "geocoding_base_url", "https://geocoding-api.open-meteo.com"
        ) or "https://geocoding-api.open-meteo.com"
        # 城市名(中文) → [纬度, 经度] 的补充/覆盖表；值需为 (lat, lon) 二元组
        self.weather_coords: dict[str, tuple[float, float]] = {
            name: (float(pt[0]), float(pt[1]))
            for name, pt in (w_weather.get("coords") or {}).items()
            if len(pt) >= 2
        }

        w_reminder = tools.get("reminder", {}) or {}
        self.tool_reminder_enabled: bool = bool(w_reminder.get("enabled", True))
        self.reminder_store_path: str | None = w_reminder.get("store_path") or None
        self.reminder_max_delay_seconds: int = int(
            w_reminder.get("max_delay_seconds", 604800) or 604800
        )

        w_camera = tools.get("look_at_user", {}) or {}
        self.tool_look_at_user_enabled: bool = bool(w_camera.get("enabled", False))

    @property
    def max_summary_tokens(self) -> int:
        """用于 max_tokens 的硬封顶：预留 ~1.5 倍余量（中文约 1 字/token）。"""
        return max(64, int(self.target_summary_chars * 1.5))


def load_agent_config() -> AgentConfig:
    """加载 agent_config.yaml；文件缺失/解析失败时回退默认值并告警。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001 - 任何加载失败都不应让服务崩掉
        logger.warning("agent config load failed (%s), use defaults", e)
        raw = {}
    return AgentConfig(_merge(_DEFAULT, raw))


_config: AgentConfig | None = None


def get_agent_config() -> AgentConfig:
    """进程内缓存单例配置。"""
    global _config
    if _config is None:
        _config = load_agent_config()
        logger.info(
            "agent config loaded: threshold=%d keep_recent=%d target=%d enable=%s longterm=%s",
            _config.compress_threshold,
            _config.keep_recent,
            _config.target_summary_chars,
            _config.memory_enabled,
            _config.longterm_enabled,
        )
    return _config