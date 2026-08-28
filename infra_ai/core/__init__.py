"""
infra_ai.core：LLM 调用的规范直达面。

聚合子模块的公开符号，提供 `from infra_ai.core import async_call_llm, ...` 的导入面。
"""

from infra_ai.inference import (
    aclose_all_clients,
    async_call_llm,
    async_call_llm_batch,
    async_call_llm_with_tools,
    async_call_vlm,
    async_call_vlm_batch,
    local_image_to_data_url,
)
from infra_ai.core.config_loader import Config, get_config
from infra_ai.core.rate_limiter import RateLimiter
from infra_ai.core.stats import get_all_stats, get_llm_stats
from infra_ai.core.streaming import async_stream_call_llm
from infra_ai.core.sync import call_llm, call_vlm

__all__ = [
    "Config",
    "RateLimiter",
    "aclose_all_clients",
    "call_llm",
    "call_vlm",
    "async_call_llm",
    "async_call_llm_with_tools",
    "async_call_vlm",
    "async_call_llm_batch",
    "async_call_vlm_batch",
    "async_stream_call_llm",
    "get_config",
    "get_llm_stats",
    "get_all_stats",
    "local_image_to_data_url",
]