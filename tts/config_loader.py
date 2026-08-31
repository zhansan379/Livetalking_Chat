"""
TTS 路由配置加载器：读取 tts/config.yaml，解析 `${ENV}` 占位符，暴露对象式单例。

用法:
    from tts.config_loader import get_config
    cfg = get_config()
    cfg.ENABLED / cfg.CIRCUIT_BREAKER / cfg.MIN_AUDIO_MS / cfg.ROUTING ...

对齐 asr/config_loader.py 的范式，占位符解析复用 utils.config_load.resolve_env。
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils.config_load import resolve_env

logger = __import__("logging").getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# 在解析 ${ENV} 占位符前载入项目根 .env（load_dotenv 默认不覆盖已 export 的变量）。
# 镜像 asr/tts 既有的做法：config_loader 先于入口脚本的 load_dotenv() 被 import 时，
# 占位才不会被缓存成空串。
try:
    from dotenv import load_dotenv

    load_dotenv(_CONFIG_PATH.parent.parent / ".env")
except Exception:
    pass


@dataclass
class TTSRoutingConfig:
    """TTS 路由运行时配置。候选池（ROUTING）供 executor/TTSPool 消费。"""

    ENABLED: bool = False
    CIRCUIT_BREAKER: dict = field(
        default_factory=lambda: {"failure_threshold": 2, "open_duration_sec": 30}
    )
    MIN_AUDIO_MS: int = 300
    ROUTING: dict = field(default_factory=dict)  # {"default_engine":..., "candidates":[...]}


def _load() -> TTSRoutingConfig:
    """读取并解析 config.yaml，装配为 TTSRoutingConfig 对象。"""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"配置文件缺失: {_CONFIG_PATH}. 请确认 tts/config.yaml 已随包分发。"
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    routing = resolve_env(raw.get("tts_routing", {}))

    def _bool(v, default=False):
        if isinstance(v, bool):
            return v
        return "false" not in str(v).lower()

    return TTSRoutingConfig(
        ENABLED=_bool(routing.get("enabled", False)),
        CIRCUIT_BREAKER=routing.get(
            "circuit_breaker", {"failure_threshold": 2, "open_duration_sec": 30}
        ),
        MIN_AUDIO_MS=int(routing.get("min_audio_ms", 300)),
        ROUTING=routing.get("routing", {}),
    )


# ------------------------------------------------------------------
# 单例
# ------------------------------------------------------------------

_config: TTSRoutingConfig | None = None


def get_config() -> TTSRoutingConfig:
    """获取全局配置单例（懒加载，首次调用时读取 config.yaml）。"""
    global _config
    if _config is None:
        try:
            _config = _load()
        except Exception as e:  # noqa: BLE001
            logger.error("加载 tts 路由配置失败: %s", e)
            raise
    return _config


def reset_config():
    """重置单例（调试/测试用）。"""
    global _config
    _config = None