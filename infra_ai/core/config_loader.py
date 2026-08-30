"""
配置加载器：从 config.yaml 读取配置，解析 `${ENV}` 占位符，暴露对象式单例。

用法:
    from infra_ai.core.config_loader import get_config
    cfg = get_config()
    cfg.LLM_ROUTING / cfg.EMBEDDING_ROUTING / cfg.LLM_REQUEST_TIMEOUT ...

占位符语法（对字符串递归解析，支持嵌套）:
    ${ENV_VAR}                 → os.environ.get("ENV_VAR", "")，未设置则为空串
    ${ENV_VAR:-default}        → 环境变量缺失时回退 default（default 可再含 ${...} 占位）
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = __import__("logging").getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

# 在解析配置里的 ${ENV} 占位符之前，确保项目根目录 .env 已加载。
# 关键：config_loader 是先于任何 get_config() 被 import，而它一 import 就到这行。
# 否则在 `python -m infra_ai...` 下，`infra_ai` 包会先于入口脚本里的
# load_dotenv() 被 import，配置单例里的 API Key 就被缓存成空串——
# 路由时只能兜底到 SF_API_KEY，把硅流 key 发给 DashScope 返回 401。
# shell 里已显式 export 的变量优先，.env 只补缺（load_dotenv 默认不覆盖）。
try:
    from dotenv import load_dotenv

    load_dotenv(_CONFIG_PATH.parent.parent / ".env")
except Exception:
    # dotenv 缺失或路径不可用时不阻塞，仍可用显式环境变量运行
    pass


# ------------------------------------------------------------------
# 占位符解析
# ------------------------------------------------------------------

def _resolve_env(obj: Any) -> Any:
    """递归解析 dict/list/str 中的 `${ENV}` 占位符（支持 ${VAR:-default} 嵌套回退）。"""
    if isinstance(obj, dict):
        return {k: _resolve_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env(i) for i in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        inner = obj[2:-1]
        if ":-" in inner:
            var, _, default = inner.partition(":-")
            default = _resolve_env(default) if default else ""
            return os.environ.get(var.strip(), default)
        return os.environ.get(inner.strip(), "")
    return obj


# ------------------------------------------------------------------
# 配置对象
# ------------------------------------------------------------------

@dataclass
class Config:
    """infra_ai 运行时配置。字段名与原 config.py 常量对齐，便于消费方迁移。"""

    LLM_REQUEST_TIMEOUT: float = 180.0
    LLM_TIMEOUT_SCALE: float = 0.4
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_BACKOFF: float = 2.0
    AUTO_TUNE_CONCURRENCY: bool = False
    LLM_ERROR_LOG: str = "./logs/llm_errors.jsonl"
    DEFAULT_RATE_LIMIT: dict = field(default_factory=dict)
    LLM_RATE_LIMITS: dict = field(default_factory=dict)
    LLM_CIRCUIT_BREAKER: dict = field(default_factory=dict)
    LLM_ROUTING: dict = field(default_factory=dict)
    EMBEDDING_TIMEOUT_SCALE: float = 0.4
    EMBEDDING_RATE_LIMITS: dict = field(default_factory=dict)
    EMBEDDING_ROUTING: dict = field(default_factory=dict)
    RERANK_RATE_LIMITS: dict = field(default_factory=dict)
    RERANK_ROUTING: dict = field(default_factory=dict)


def _load() -> Config:
    """读取并解析 config.yaml，装配为 Config 对象。"""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"配置文件缺失: {_CONFIG_PATH}. 请确认 config.yaml 已随包分发。"
        )
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    llm = _resolve_env(raw.get("llm", {}))
    embedding = _resolve_env(raw.get("embedding", {}))
    rerank = _resolve_env(raw.get("rerank", {}))

    default_rate_limit = llm.get("rate_limits", {}).get(
        "default", {"rpm": 60, "tpm": 100000, "max_concurrent": 10}
    )

    return Config(
        LLM_REQUEST_TIMEOUT=float(llm.get("request_timeout", 180.0)),
        LLM_TIMEOUT_SCALE=float(llm.get("timeout_scale", 0.4)),
        LLM_MAX_RETRIES=int(llm.get("max_retries", 3)),
        LLM_RETRY_BACKOFF=float(llm.get("retry_backoff", 2.0)),
        AUTO_TUNE_CONCURRENCY=bool(llm.get("auto_tune_concurrency", False)),
        LLM_ERROR_LOG=llm.get("error_log", "./logs/llm_errors.jsonl"),
        DEFAULT_RATE_LIMIT=default_rate_limit,
        LLM_RATE_LIMITS={"default": default_rate_limit},
        LLM_CIRCUIT_BREAKER=llm.get(
            "circuit_breaker", {"failure_threshold": 2, "open_duration_sec": 30}
        ),
        LLM_ROUTING=llm.get("routing", {}),
        EMBEDDING_TIMEOUT_SCALE=float(embedding.get("timeout_scale", 0.4)),
        EMBEDDING_RATE_LIMITS=embedding.get(
            "rate_limits", {"default": default_rate_limit}
        ),
        EMBEDDING_ROUTING=embedding.get("routing", {}),
        RERANK_RATE_LIMITS=rerank.get(
            "rate_limits", {"default": default_rate_limit}
        ),
        RERANK_ROUTING=rerank.get("routing", {}),
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
        except Exception as e:
            logger.error("加载配置文件失败: %s", e)
            raise
    return _config