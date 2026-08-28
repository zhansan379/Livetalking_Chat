"""
流式调用探活与故障转移。

ProbeStreamBridge: 拦截流式 token，缓冲首包前的事件，
等待第一个 token 到达后 commit（冲刷缓冲），或超时/失败后 discard。

StreamCallback: 流式回调接口，供上游业务层实现。

用法:
    from infra_ai.core.stream_bridge import ProbeStreamBridge

    bridge = ProbeStreamBridge(timeout=60)
    asyncio.create_task(stream_model(bridge))
    result = await bridge.await_first_packet(60)
    if result == "SUCCESS":
        async for token in bridge.tokens():
            yield token
    else:
        # 尝试下一个候选
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class StreamResult(Enum):
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


@dataclass
class _BufferedEvent:
    """缓冲的流式事件。"""
    type: str  # "token" | "complete" | "error"
    content: str = ""
    error: Exception | None = None


class ProbeStreamBridge:
    """
    探活流式桥接器。

    在第一个 token 到达前缓冲所有事件。收到首包后 commit（冲刷缓冲），
    后续事件直通。超时或错误时 discard（丢弃缓冲，通知上游失败）。

    用法:
        bridge = ProbeStreamBridge(timeout=60)

        # 生产者侧（在 asyncio task 中）
        async def produce():
            async for token in model.astream(messages):
                bridge.on_token(token)
            bridge.on_complete()

        asyncio.create_task(produce())

        # 消费者侧（路由线程）
        result = await bridge.await_first_packet(60)
        if result == StreamResult.SUCCESS:
            async for token in bridge.tokens():
                yield token
        else:
            # 失败，尝试下一个候选
    """

    def __init__(self, timeout: float = 60.0):
        self._timeout = timeout
        self._buffer: list[_BufferedEvent] = []
        self._committed = False
        self._first_packet = asyncio.Event()
        self._result: StreamResult | None = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = asyncio.Event()
        self._error: Exception | None = None

    # ---- 生产者侧 ----

    def on_token(self, content: str):
        """收到一个 token。"""
        event = _BufferedEvent(type="token", content=content)
        if not self._committed:
            self._buffer.append(event)
            if not self._first_packet.is_set():
                self._first_packet.set()
                self._result = StreamResult.SUCCESS
        else:
            self._queue.put_nowait(event)

    def on_complete(self):
        """流正常结束。"""
        event = _BufferedEvent(type="complete")
        if not self._committed:
            self._buffer.append(event)
            if not self._first_packet.is_set():
                self._first_packet.set()
                self._result = StreamResult.SUCCESS
        else:
            self._queue.put_nowait(event)

    def on_error(self, error: Exception):
        """流出错。"""
        event = _BufferedEvent(type="error", error=error)
        if not self._committed:
            self._buffer.append(event)
            self._result = StreamResult.ERROR
            self._first_packet.set()
        else:
            self._queue.put_nowait(event)

    # ---- 路由器侧 ----

    async def await_first_packet(self, timeout: float | None = None) -> StreamResult:
        """
        等待第一个 packet（token/error/complete）。

        返回 StreamResult:
            SUCCESS  → commit 已完成，可通过 tokens() 消费
            ERROR    → 流失败，应切换到下一个候选
            TIMEOUT  → 超时，应切换到下一个候选
        """
        t = timeout or self._timeout
        try:
            await asyncio.wait_for(self._first_packet.wait(), timeout=t)
        except TimeoutError:
            self._result = StreamResult.TIMEOUT
            self._first_packet.set()
            return StreamResult.TIMEOUT

        if self._result == StreamResult.SUCCESS:
            self._commit()
        return self._result or StreamResult.ERROR

    def _commit(self):
        """冲刷缓冲到队列，后续事件直通。"""
        if self._committed:
            return
        self._committed = True
        for event in self._buffer:
            self._queue.put_nowait(event)
        self._buffer.clear()

    def discard(self):
        """丢弃所有缓冲，标记为失败。"""
        self._buffer.clear()
        self._committed = True
        self._result = StreamResult.ERROR
        self._first_packet.set()

    # ---- 消费者侧（commit 后） ----

    async def tokens(self) -> AsyncIterator[str]:
        """消费 commit 后的 token 流（async generator）。"""
        while not self._done.is_set():
            try:
                event = await self._queue.get()
                if event.type == "complete":
                    return
                if event.type == "error":
                    if event.error:
                        raise event.error
                    return
                if event.type == "token":
                    yield event.content
            except asyncio.CancelledError:
                return


# ----------------------------------------------------------------
# 流式路由辅助
# ----------------------------------------------------------------

async def stream_with_fallback(
    selector,
    capability: str,
    stream_caller,
    first_choice_id: str | None = None,
    timeout: float = 60.0,
) -> AsyncIterator[str]:
    """
    带故障转移的流式调用。

    对每个候选模型创建 ProbeStreamBridge，等待首包。
    成功则 yield token，失败则切换到下一个候选。

    :param selector: ModelSelector 实例
    :param capability: "chat" | "vision"
    :param stream_caller: async fn(target, bridge) -> None（在 bridge 上调用 on_token/on_complete/on_error）
    :param first_choice_id: 首选模型 ID
    :param timeout: 首包超时时间（秒）
    :yields: token 字符串
    """
    from infra_ai.core.health_store import get_health_store
    health_store = get_health_store()

    targets = selector.select(capability, first_choice_id=first_choice_id)
    if not targets:
        raise RuntimeError(f"没有可用的 {capability} 流式模型候选")

    last_error: Exception | None = None

    for i, target in enumerate(targets):
        if not health_store.allow_call(target.model_id):
            logger.warning("模型 %s 熔断中，尝试下一个候选", target.display)
            continue

        logger.info("流式调用模型: %s (%d/%d)", target.display, i + 1, len(targets))
        bridge = ProbeStreamBridge(timeout=timeout)

        # 启动生产者 task
        async def _produce():
            try:
                await stream_caller(target, bridge)
            except Exception as e:
                bridge.on_error(e)

        task = asyncio.ensure_future(_produce())

        result = await bridge.await_first_packet(timeout)
        if result == StreamResult.SUCCESS:
            try:
                async for token in bridge.tokens():
                    yield token
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            return  # 成功
        else:
            bridge.discard()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            logger.warning("模型 %s 流式首包失败 (result=%s)，尝试下一个", target.display, result.value)
            last_error = RuntimeError(f"流式首包 {result.value}")
            continue

    raise RuntimeError(
        f"所有 {capability} 流式候选模型均失败 (共 {len(targets)} 个)"
    ) from last_error
