###############################################################################
#  能力中枢 hub：负载 + 注册 + 会话级工具暴露 + 系统提示注入 + 配置默认值。
#
#  主循环（chat.py / config.py）只认三个通用入口，不感知任何具体能力：
#     - register_capability_tools()     插口③：把能力声明工具合并进 TOOL_REGISTRY
#     - session_tools(session_id, cfg)  插口③：本轮按会话+状态暴露的工具名
#     - capability_system_block(sid,cfg) 插口②：拼接能力目录/激活态到 system prompt
#     - capability_config_defaults()     插口①：供 config.py 并入能力自带默认配置
#
#  hub 不 import 具体能力；用 pkgutil+importlib 反向发现（依赖反转）。任何单个能力
#  加载失败仅告警、不影响其它能力与主流程（沿用全局 try/except 防御风格）。
###############################################################################

import importlib
import pkgutil

from utils.logger import logger

from capabilities import base


# ── 进程内单例，懒加载一次 ─────────────────────────────────────────────────
_loaded = False
_tools_registered = False
_capabilities: dict[str, "base.Capability"] = {}
_config_defaults: dict[str, dict] = {}


def _discover() -> None:
    """扫描 capabilities/ 下每个子包，导入并登记其 CAPABILITY 与 config_defaults。"""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        import capabilities as _pkg
        for mod in pkgutil.iter_modules(_pkg.__path__):
            if mod.name == "base":
                continue
            try:
                m = importlib.import_module(f"capabilities.{mod.name}")
                cap = getattr(m, "CAPABILITY", None)
                if cap is None or not getattr(cap, "name", ""):
                    continue
                cd = getattr(m, "config_defaults", None)
                _capabilities[cap.name] = cap
                _config_defaults[cap.name] = cd() if callable(cd) else {}
                logger.info("capability loaded: %s", cap.name)
            except Exception as e:  # noqa: BLE001 - 单个能力失败不影响其它
                logger.warning("capability %s load failed: %s", mod.name, e)
    except Exception as e:  # noqa: BLE001
        logger.warning("capability discovery failed: %s", e)


def all_capabilities() -> dict[str, "base.Capability"]:
    _discover()
    return dict(_capabilities)


def _is_cap_flag(flag) -> bool:
    """是否为能力工具的 config_flag（形如 ("cap", name) 的二元组）。"""
    return isinstance(flag, (list, tuple)) and len(flag) == 2 and flag[0] == "cap"


def capability_config_defaults() -> dict:
    """返回所有启用配置默认值：{cap_name: {key: default}}（插口①来源）。"""
    _discover()
    return dict(_config_defaults)


# ── 插口③：注册 + 会话级工具暴露 ───────────────────────────────────────────
def register_capability_tools() -> None:
    """把各能力 tools() 合并进 tool_loop.TOOL_REGISTRY，盖上 config_flag=("cap",name)。

    注册表始终完整（常驻）；每轮暴露哪个子集由 session_tools 决定。幂等。
    """
    global _tools_registered
    if _tools_registered:
        return
    _discover()
    from agent import tool_loop  # 延迟 import，避免配置构造期拉起注册表依赖
    for cap in _capabilities.values():
        for spec in cap.tools() or []:
            name = spec.get("name")
            if not name:
                logger.warning("capability %s declared a tool without name", cap.name)
                continue
            entry = dict(spec)
            entry["config_flag"] = ("cap", cap.name)
            tool_loop.TOOL_REGISTRY[name] = entry
            logger.debug("capability tool registered: %s", name)
    _tools_registered = True


def session_tools(session_id: str | None, cfg) -> list[str]:
    """本轮要暴露的工具名。

    = 全局通用工具（注册表中非能力所属，按配置门控）+ 每个启用能力当前 active_tools 子集。
    保证正常闲聊时能力工具不进列表（s07），进行中则只暴露相关子集。
    """
    _discover()
    register_capability_tools()
    from agent import tool_loop  # session_tools 总是需要注册表

    names: list[str] = []
    for name, entry in tool_loop.TOOL_REGISTRY.items():
        flag = entry.get("config_flag")
        if _is_cap_flag(flag):
            cap = _capabilities.get(flag[1])
            if cap is None or not cap.enabled(cfg):
                continue
            if name in cap.active_tools(session_id or ""):
                names.append(name)
        else:
            # 全局工具：沿用原 list_enabled_tools 的门控逻辑
            if getattr(cfg, flag or f"tool_{name}_enabled", False):
                names.append(name)
    return names


# ── 插口②：系统提示注入 ─────────────────────────────────────────────────────
def capability_system_block(session_id: str | None, cfg) -> str:
    """拼接所有启用能力的 system_block() 片段；无内容返回 ""。"""
    _discover()
    parts: list[str] = []
    for cap in _capabilities.values():
        if not cap.enabled(cfg):
            continue
        try:
            s = cap.system_block(session_id or "", cfg)
        except Exception as e:  # noqa: BLE001 - 单个能力注入失败不阻塞
            logger.warning("capability %s system_block failed: %s", cap.name, e)
            continue
        if s and s.strip():
            parts.append(s.strip())
    return "\n\n".join(parts) if parts else ""