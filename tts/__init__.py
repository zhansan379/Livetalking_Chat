"""TTS 引擎注册表：引擎名 → 模块映射 + 路由选择门面。

`TTS_MODULES` 供 `avatars/base_avatar.py` 与 `tts/executor.py::TTSPool` 共用，
消除『引擎名→模块』映射在单后端与候选池两处的重复（曾是 base_avatar.__init__ 的局部 dict）。
"""

from .base_tts import BaseTTS, State
from utils.logger import logger

# 引擎名（registry "tts" 槽注册名 in config.yaml `tts:`）→ 所在模块。
# 模块必须被 import 才会执行 @register("tts", ...) 注册进 registry。
TTS_MODULES = {
    "edgetts": "tts.edge",
    "gpt-sovits": "tts.sovits",
    "xtts": "tts.xtts",
    "cosyvoice": "tts.cosyvoice",
    "fishtts": "tts.fish",
    "tencent": "tts.tencent",
    "doubao": "tts.doubao",
    "indextts2": "tts.indextts2",
    "azuretts": "tts.azure",
    "qwentts": "tts.qwentts",
    "omnitts": "tts.omnitts",
}


def select_tts(opt, parent):
    """选择 TTS 后端：路由开启且有多候选 → TTSPool（句内回退+熔断）；否则单后端（旧行为）。"""
    try:
        from .config_loader import get_config
        from .executor import TTSPool
        from utils.cand_pool import build_candidates

        cfg = get_config()
        enabled = build_candidates((cfg.ROUTING.get("candidates") or []))
        if cfg.ENABLED and sum(1 for c in enabled if c.enabled) >= 2:
            return TTSPool(opt, parent)
    except Exception as e:  # noqa: BLE001 - 路由异常不影响主流程，回退单后端
        logger.warning("select_tts: 路由配置加载失败(%s)，回退单后端 %s", e, opt.tts)

    import importlib

    if opt.tts in TTS_MODULES:
        importlib.import_module(TTS_MODULES[opt.tts])
        from registry import create

        return create("tts", opt.tts, opt=opt, parent=parent)
    logger.error(f"TTS module {opt.tts} not found.")
    return None


__all__ = ["BaseTTS", "State", "TTS_MODULES", "select_tts"]