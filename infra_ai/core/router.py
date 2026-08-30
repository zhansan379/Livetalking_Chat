"""
多模型路由与故障转移。

ModelSelector: 从配置中选取候选模型列表，按优先级排序，过滤熔断中的模型。
ModelRoutingExecutor: 遍历候选模型执行调用，支持故障转移。

用法:
    from infra_ai.core.router import get_router
    router = get_router()
    result = await router.execute_chat(messages, use_json=True)
"""

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# 数据模型
# ----------------------------------------------------------------

@dataclass
class ModelCandidate:
    """单个模型候选配置（LLM / embedding / rerank 统一建模）。"""
    id: str                     # 唯一标识，如 "qwen2.5-72b"
    provider: str = ""          # provider 名称（embedding/rerank 用于错误文案）
    model: str = ""             # 实际模型名，如 "Qwen/Qwen2.5-72B-Instruct-128K"
    priority: int = 100         # 优先级（越小越优先）
    enabled: bool = True        # 是否启用
    supports_thinking: bool = False  # 是否支持深度思考
    api_key: str = ""           # 可选，覆盖全局 API_KEY
    base_url: str = ""          # 可选，覆盖全局 BASE_URL
    dimensions: int = 0         # embedding 维度（当前透传占位，客户端未消费）


@dataclass
class ModelTarget:
    """运行时模型调用目标。"""
    model_id: str
    candidate: ModelCandidate
    model_name: str

    @property
    def display(self) -> str:
        return f"{self.model_id}({self.model_name})"

    @property
    def id(self) -> str:
        """统一别名，与 ModelCandidate.id 对齐，供 iterate_candidates 驱动取候选标识。"""
        return self.model_id


# ----------------------------------------------------------------
# ModelSelector
# ----------------------------------------------------------------

class ModelSelector:
    """
    模型选择器。

    从配置中读取候选模型列表，按健康状态和优先级排序。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._candidates: dict[str, list[ModelCandidate]] = {}
        self._load()

    def _load(self):
        """从 config 加载候选模型列表（遍历 routing 下全部能力，如 chat/vision/chat_tone 等）。"""
        for capability in self._config:  # 不硬编码能力名，新增能力进配置即自动装载
            cap_cfg = self._config.get(capability, {})
            candidates_list = cap_cfg.get("candidates", [])
            candidates = []
            for c in candidates_list:
                candidates.append(ModelCandidate(
                    id=c.get("id", ""),
                    provider=c.get("provider", ""),
                    model=c.get("model", ""),
                    priority=c.get("priority", 100),
                    enabled=c.get("enabled", True),
                    supports_thinking=c.get("supports_thinking", False),
                    api_key=c.get("api_key", ""),
                    base_url=c.get("base_url", ""),
                    dimensions=c.get("dimensions", 0),
                ))
            self._candidates[capability] = candidates

    def select(self, capability: str, first_choice_id: str | None = None,
               need_thinking: bool = False) -> list[ModelTarget]:
        """
        为指定能力选取候选模型列表。

        过滤规则: 禁用项、深度思考不匹配项、熔断中的模型。
        排序: 首选模型排最前，其余按 priority 升序。

        返回: 已排序的 ModelTarget 列表
        """
        candidates = self._candidates.get(capability, [])
        if not candidates:
            return []

        from .health_store import get_health_store
        health_store = get_health_store()

        # 确定首选模型
        if first_choice_id is None:
            first_choice_id = self._config.get(capability, {}).get("default_model", "")

        targets: list[ModelTarget] = []
        first_target: ModelTarget | None = None

        for c in candidates:
            # 过滤禁用
            if not c.enabled:
                continue
            # 过滤深度思考
            if need_thinking and not c.supports_thinking:
                continue

            target = ModelTarget(
                model_id=c.id,
                candidate=c,
                model_name=c.model,
            )

            # 熔断器过滤（首选模型不在此处过滤，留给 executor 判断）
            if health_store.get_state(c.id).value == "CLOSED" or c.id == first_choice_id:
                if c.id == first_choice_id:
                    first_target = target
                else:
                    # 按优先级插入排序
                    targets.append(target)

        # 排序：优先级小的在前
        targets.sort(key=lambda t: t.candidate.priority)

        # 首选模型放到最前
        if first_target:
            # 如果首选已在列表中，移除后插入最前
            targets = [t for t in targets if t.model_id != first_target.model_id]
            targets.insert(0, first_target)

        if len(targets) > 1:
            logger.debug("候选模型: %s", " -> ".join(t.display for t in targets))

        return targets

    def get_candidate(self, capability: str, model_id: str) -> ModelCandidate | None:
        """按 ID 获取候选模型配置。"""
        for c in self._candidates.get(capability, []):
            if c.id == model_id:
                return c
        return None


# ----------------------------------------------------------------
# 共享路由-回退驱动（LLM / embedding / rerank 三个路子共用）
# ----------------------------------------------------------------

async def iterate_candidates(candidates, attempt, *, label: str = "候选", attempt_logs: bool = False):
    """
    共享路由-回退驱动：顺序遍历候选，熔断过滤 + 逐候选调用 + 全败抛错。

    :param candidates: 候选列表（调用方已排好序，驱动不负责排序）
    :param attempt:   async fn(candidate) -> result；成功返回，抛错则切下一候选
    :param label:     日志前缀（如 "chat" / "embedding" / "rerank"）
    :param attempt_logs: True 时逐候选打 info 日志（LLM 路径保留观察面）
    :param 健康标记: mark_success / mark_failure 由 attempt（叶子/调用方）负责，驱动不标记
    :raise RuntimeError: 全部候选失败
    """
    from .health_store import get_health_store
    health = get_health_store()
    last_error: Exception | None = None

    for c in candidates:
        if not health.allow_call(c.id):
            logger.warning("%s %s 熔断中，跳过", label, c.id)
            continue
        if attempt_logs:
            logger.info("%s 尝试候选 %s", label, c.id)
        try:
            return await attempt(c)
        except Exception as e:
            last_error = e
            logger.warning("%s %s 调用失败: %s", label, c.id, e)

    raise RuntimeError(f"所有 {label} 候选均失败") from last_error


# ----------------------------------------------------------------
# ModelRoutingExecutor
# ----------------------------------------------------------------

class ModelRoutingExecutor:
    """
    故障转移执行器。

    遍历候选模型列表，对每个模型：检查熔断器 → 调用 → 成功返回 / 失败继续下一个。
    """

    def __init__(self, selector: ModelSelector):
        self.selector = selector

    async def execute(
        self,
        capability: str,
        caller: Callable[[ModelTarget], Awaitable[str]],
        first_choice_id: str | None = None,
        need_thinking: bool = False,
    ) -> str:
        """
        带故障转移的模型调用。

        遍历候选模型列表，逐个尝试，成功则返回。
        所有候选都失败则抛出最后一个异常。

        :param capability: "chat" | "vision"
        :param caller: 异步调用函数 (target: ModelTarget) -> str
        :param first_choice_id: 首选模型 ID
        :param need_thinking: 是否需要深度思考
        :return: 模型响应文本
        """
        targets = self.selector.select(
            capability,
            first_choice_id=first_choice_id,
            need_thinking=need_thinking,
        )

        if not targets:
            raise RuntimeError(f"没有可用的 {capability} 模型候选")

        # 委托给共享路由-回退驱动（targets 已由 select() 排好序）
        return await iterate_candidates(
            targets, caller, label=capability, attempt_logs=True,
        )


# ----------------------------------------------------------------
# 全局单例
# ----------------------------------------------------------------

_router: ModelRoutingExecutor | None = None


def get_router() -> ModelRoutingExecutor:
    """获取全局路由器单例。"""
    global _router
    if _router is None:
        from infra_ai.core.config_loader import get_config
        selector = ModelSelector(get_config().LLM_ROUTING)
        _router = ModelRoutingExecutor(selector)
    return _router
