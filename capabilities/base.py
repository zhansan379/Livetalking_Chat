###############################################################################
#  能力协议基类 Capability —— 主循环对「能力」的全部感知都收敛到这个接口。
#
#  与 agent/tool_loop.TOOL_REGISTRY 的「工具」不同：能力是一个有状态、多轮、带
#  persona/业务逻辑的「域」。它不对主循环暴露实现细节，只提供三个通用插口：
#    - tools()           声明一批工具（schema+handler），由 hub 合并进 TOOL_REGISTRY；
#    - active_tools(sid) 本轮该暴露的工具子集（按会话状态条件注入，s07 按需加载）；
#    - system_block(sid) 注入 system prompt 的一段文本（能力目录 + 当前会话激活态）。
#
#  能力不接触 avatar_session / TTS / obs —— 只返回文本、读写自己的持久化；
#  TTS 由现有工具调用链、观测由 infra_ai 自动埋点负责（见计划 4·B）。
###############################################################################

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 仅类型标注，避免 import 时拉起 agent.config（防配置构造期循环）
    from agent.config import AgentConfig


class Capability:
    """一个可插拔能力的协议。

    具体能力继承本类并覆盖相应方法；每个能力子包另导出 `CAPABILITY` 单例实例
    与 `config_defaults()`（返回本能力自带配置默认值 dict，供 config.py 并入）。
    """

    name: str = ""       # 唯一 id，如 "interview"
    priority: int = 0    # 同时命中多个能力时 hub 的排序权重（越大越优先）

    def __init__(self, name: str = "", priority: int = 0) -> None:
        if name:
            self.name = name
        self.priority = priority

    # ── 开关（读 agent_config 的 capabilities.<name>.enabled）──────────────
    def enabled(self, cfg: "AgentConfig") -> bool:
        return bool(getattr(cfg, "cap_enabled", None) and cfg.cap_enabled(self.name, False))

    # ── 工具声明（合并进 TOOL_REGISTRY；每条形如 registry entry，带 name/handler）──
    def tools(self) -> list[dict]:
        """返回本能力要向 TOOL_REGISTRY 注册的工具描列表。

        每条元素为 {name, description, parameters, handler}；hub 注册时会给每条
        盖上 config_flag=("cap", self.name)，用于开关门控与「非能力全局工具」区分。
        """
        return []

    # ── 每轮工具暴露子集（按会话状态条件注入，默认全暴露）────────────────
    def active_tools(self, session_id: str) -> list[str]:
        """本轮该暴露的工具名子集。

        默认返回全部注册工具名；面试按状态覆盖：idle→[start]; asking→[answer,skip,
        hint,end,status]; finished→[start,status]。实现 s07「用到时再加载」。
        """
        return [t.get("name", "") for t in self.tools() if t.get("name")]

    # ── system prompt 注入片段（能力目录 + 当前会话激活态）────────────────
    def system_block(self, session_id: str, cfg: "AgentConfig") -> str:
        """注入 system prompt 的一段文本；未启用/无内容返回 ""。"""
        return ""

    # ── 会话结束清理（可选）───────────────────────────────────────────────
    def on_session_end(self, session_id: str) -> None:
        return