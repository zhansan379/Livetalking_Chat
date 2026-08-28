"""
通用「异步 ↔ 同步」适配工具。

核心思想：检测当前是否在 asyncio 事件循环中。
  - 不在循环中 → 直接用 asyncio.run() 创建临时循环
  - 在循环中   → 在独立线程里创建新循环执行（如 FastAPI/uvicorn 场景）

供 core/sync、rerank、embedding 的同步壳共用，避免各自重复实现。
"""

import asyncio
import concurrent.futures


def run_async_in_sync(coro):
    """在任何上下文中安全地运行一个 async 协程并返回结果。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 不在事件循环中 —— 直接创建临时循环运行
        return asyncio.run(coro)

    # 已在事件循环中 —— 在独立线程中创建新循环
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()