"""ASR 语音识别包：可扩展多引擎 + 候选池熔断回退 + 独立配置。

- 引擎：`registry.py` 的 "stt" 槽注册（@register("stt", name)，如 asr/sensevoice.py）。
- 挂载：`setup_routes(app)` 自注册 /api/asr（对齐 obs 包范式），gate 用候选池启用态。
- 配置：asr/config.yaml（独立，仿 infra_ai/config.yaml）。

用法（在 server/routes.py）::

    try:
        from asr import setup_routes as _setup_asr
        _setup_asr(app)
    except Exception as e:
        logger.warning("asr routes registration failed: %s", e)
"""

from utils.logger import logger
from .base import BaseASR, TranscriptionResult

# 触发 SenseVoice 引擎装饰器注册（其余引擎在 executor 懒 import 时自动注册）。
from . import sensevoice  # noqa: F401


def setup_routes(app):
    """注册 ASR WebSocket 路由（gate：存在 ≥1 个 enabled 且依赖可用的候选）。"""
    from .executor import get_pool
    from .handler import asr_websocket_handler

    if not get_pool().available():
        logger.info("[ASR] No enabled/local ASR engine — /api/asr disabled")
        return
    app.router.add_get("/api/asr", asr_websocket_handler)
    logger.info("[ASR] ASR endpoint enabled at /api/asr")


__all__ = ["BaseASR", "TranscriptionResult", "setup_routes"]