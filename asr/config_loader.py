"""
ASR 配置加载器：读取 asr/config.yaml，解析 `${ENV}` 占位符，暴露对象式单例。

用法:
    from asr.config_loader import get_config
    cfg = get_config()
    cfg.ENABLED / cfg.SAMPLE_RATE / cfg.ROUTING / cfg.CIRCUIT_BREAKER ...

占位符语法（对字符串递归解析，支持嵌套）:
    ${ENV_VAR}                 → os.environ.get("ENV_VAR", "")，未设置则为空串
    ${ENV_VAR:-default}        → 环境变量缺失时回退 default（default 可再含 ${...} 占位）

对齐 infra_ai/core/config_loader.py 的范式，但自包含（不依赖 infra_ai）。
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from utils.config_load import resolve_env

logger = __import__("logging").getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

# 在解析 ${ENV} 占位符前载入项目根 .env（load_dotenv 默认不覆盖已 export 的变量）。
# 与 infra_ai 同理：config_loader 先于入口脚本的 load_dotenv() 被 import，
# 否则 SENSEVOICE_DEVICE 等占位会被缓存成空串。
try:
    from dotenv import load_dotenv

    load_dotenv(_CONFIG_PATH.parent.parent / ".env")
except Exception:
    pass


# ------------------------------------------------------------------
# 占位符解析（复用 utils.config_load.resolve_env）
# ------------------------------------------------------------------

@dataclass
class Config:
    """ASR 运行时配置。候选池（ROUTING）供 executor/ASRPool 消费。"""

    ENABLED: bool = True
    SAMPLE_RATE: int = 16000
    MIN_AUDIO_BYTES: int = 640
    CIRCUIT_BREAKER: dict = field(
        default_factory=lambda: {"failure_threshold": 2, "open_duration_sec": 30}
    )
    ROUTING: dict = field(default_factory=dict)  # {"default_engine":..., "candidates":[...]}


def _load() -> Config:
    """读取并解析 config.yaml，装配为 Config 对象。"""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"配置文件缺失: {_CONFIG_PATH}. 请确认 asr/config.yaml 已随包分发。"
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    asr = resolve_env(raw.get("asr", {}))

    return Config(
        ENABLED=("false" not in str(asr.get("enabled", True)).lower()
                 if not isinstance(asr.get("enabled"), bool)
                 else asr.get("enabled", True)),
        SAMPLE_RATE=int(asr.get("sample_rate", 16000)),
        MIN_AUDIO_BYTES=int(asr.get("min_audio_bytes", 640)),
        CIRCUIT_BREAKER=asr.get(
            "circuit_breaker", {"failure_threshold": 2, "open_duration_sec": 30}
        ),
        ROUTING=asr.get("routing", {}),
    )


# ------------------------------------------------------------------
# 单例
# ------------------------------------------------------------------

_config: Config | None = None


def get_config() -> Config:
    """获取全局配置单例（懒加载，首次调用时读取 config.yaml）。"""
    global _config
    if _config is None:
        try:
            _config = _load()
        except Exception as e:  # noqa: BLE001
            logger.error("加载 asr 配置失败: %s", e)
            raise
    return _config