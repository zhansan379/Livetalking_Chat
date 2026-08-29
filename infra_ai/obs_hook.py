###############################################################################
#  obs_hook：infra_ai 与观测平台之间的零依赖回调槽。
#
#  目的：让 infra_ai 提供观测事件却永不 import obs 包（避免循环/强耦）。
#  obs.install() 在启动时调用 set_obs(回调)，之后 _invoke_with_retry /
#  _stream_single_model 只需 emit_obs({...})；未挂钩或禁用时为空操作。
###############################################################################


_OBS = None


def set_obs(cb):
    """设置观测回调（进程级单例）。cb 签名：cb(event: dict) -> None"""
    global _OBS
    _OBS = cb


def emit_obs(event: dict) -> None:
    """向观测层投递一个事件；无观测回调时任一分发失败都静默。"""
    global _OBS
    cb = _OBS
    if cb is None:
        return
    try:
        cb(event)
    except Exception:  # noqa: BLE001 - 观测失败不影响 LLM 调用
        pass