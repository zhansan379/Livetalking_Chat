"""
infra_ai 调用演示：覆盖各种常见调用场景。

运行前请确认：
- 已 `pip install -e .`
- 已配置 config.yaml 所需环境变量（如 SF_API_KEY / SF_BASE_URL），
  否则候选模型默认 enabled=false，调用会走到单模型回退或直接失败。

演示场景：
  1. 同步调用          call_llm / call_vlm（脚本 / FastAPI / Jupyter 均安全）
  2. 异步调用          async_call_llm
  3. JSON 结构化输出    use_json=True
  4. 指定模型覆盖路由   model_name=...
  5. 原生工具调用      async_call_llm_with_tools
  6. 视觉调用 VLM      图片 URL / 本地图片 base64
  7. 流式调用          async_stream_call_llm
  8. 批量并发调用      async_call_llm_batch / async_call_vlm_batch
  9. 重排序            rerank / async_rerank
  10. 向量嵌入         get_embedding_client
  11. 调用统计         get_llm_stats / get_all_stats
  12. 错误处理          熔断 / 重试耗尽 / 指定模型异常的捕获

所有示例都有独立开关（RUN_xxx），方便单独运行某一段。
"""

import asyncio
import json
import time

from infra_ai import (
    aclose_all_clients,
    async_call_llm,
    async_call_llm_batch,
    async_call_llm_with_tools,
    async_call_vlm,
    async_call_vlm_batch,
    async_stream_call_llm,
    async_rerank,
    call_llm,
    call_vlm,
    get_all_stats,
    get_llm_stats,
    get_embedding_client,
    local_image_to_data_url,
    rerank,
)

# ---- 运行开关：把要跑的段落置 True ----
RUN_SYNC = True
RUN_ASYNC = False
RUN_JSON = False
RUN_MODEL_OVERRIDE = False
RUN_TOOLS = False
RUN_VLM = False          # 需要可访问的图片，默认关
RUN_STREAM = False
RUN_BATCH = False
RUN_RERANK = False
RUN_EMBEDDING = False    # 需要 text embedding 候选 enabled
RUN_STATS = False
RUN_ERROR_HANDLING = False  # 故意触发失败，默认关


# ============================================================
# 1. 同步调用（任意上下文安全）
# ============================================================
def demo_sync():
    print("\n=== 1. 同步调用 call_llm / call_vlm ===")
    text = call_llm(
        [
            {"role": "system", "content": "你是 Python 专家。"},
            {"role": "user", "content": "一句话解释什么是 GIL？"},
        ],
        use_json=False,
    )
    print("文本:", text)

    # 同步上下文里字符串返回，可直接当普通函数用


# ============================================================
# 2. 异步调用
# ============================================================
async def demo_async():
    print("\n=== 2. 异步调用 async_call_llm（串行 vs 并发）===")

    questions = [
        "什么是异步编程？",
        "什么是 await？",
        "什么是事件循环？",
        "GIL 对异步有影响吗？",
    ]

    # 方式一：串行 —— 逐个 await，总耗时 ≈ N × 单请求
    print(f"[串行] 依次发起 {len(questions)} 个调用…")
    t0 = time.perf_counter()
    serial = [
        await async_call_llm([{"role": "user", "content": q}], use_json=False)
        for q in questions
    ]
    t_serial = time.perf_counter() - t0

    # 方式二：并发 —— asyncio.gather 同时发起，总耗时 ≈ 单请求
    print(f"[并发] 同时发起 {len(questions)} 个调用…")
    t0 = time.perf_counter()
    concurrent = await asyncio.gather(
        *(async_call_llm([{"role": "user", "content": q}], use_json=False)
          for q in questions)
    )
    t_concurrent = time.perf_counter() - t0

    print(f"[串行] 耗时 {t_serial:.2f}s")
    print(f"[并发] 耗时 {t_concurrent:.2f}s"
          f"  → 加速 {t_serial / t_concurrent:.1f}x")

    for q, r in zip(questions, concurrent):
        print(f"\nQ: {q}\nA: {str(r)[:120]}...")


# ============================================================
# 3. JSON 结构化输出
# ============================================================
async def demo_json_output():
    print("\n=== 3. JSON 结构化输出 use_json=True ===")
    # 绑定了 response_format={"type":"json_object"}，模型会被要求返回合法 JSON
    raw = await async_call_llm(
        [
            {
                "role": "user",
                "content": "列出 3 个 Python Web 框架，输出成 JSON："
                           '{"frameworks": ["名称", ...]}',
            }
        ],
        use_json=True,
    )
    # 有些实现直接返回 dict，这里统一按字符串尝试解析
    data = raw if isinstance(raw, dict) else json.loads(raw)
    print("解析结果:", data)


# ============================================================
# 4. 指定模型覆盖路由
# ============================================================
async def demo_model_override():
    print("\n=== 4. 指定模型覆盖路由 model_name=... ===")
    text = await async_call_llm(
        [{"role": "user", "content": "你是什么模型？"}],
        use_json=False,
        model_name="Qwen/Qwen2.5-72B-Instruct-128K",
    )
    print("指定模型输出:", text)


# ============================================================
# 5. 原生工具调用
# ============================================================
async def demo_tools():
    print("\n=== 5. 原生工具调用 async_call_llm_with_tools ===")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名"}
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    msg = await async_call_llm_with_tools(
        [{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=tools,
    )
    # 返回原生 OpenAI message，含 .content 和 .tool_calls
    print("content:", msg.content)
    tool_calls = getattr(msg, "tool_calls", None)
    print("tool_calls:", tool_calls)

    # 拿到工具参数后，可把结果回填成 assistant + tool 消息继续对话
    if tool_calls:
        # 原生 SDK: tool_call.function.name / tool_call.function.arguments
        tc = tool_calls[0]
        params = tc.function.arguments
        print("模型请求调用", tc.function.name, ", 参数:", params)

        follow_up = await async_call_llm(
            [
                {"role": "user", "content": "北京今天天气怎么样？"},
                # 真实工具循环里这里应插入 assistant 的 tool_calls 与 tool 结果
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "content": '{"temperature": 28, "condition": "晴"}'},
                {"role": "user", "content": "请用一句话总结天气。"},
            ],
            use_json=False,
        )
        print("工具后的总结:", follow_up)


# ============================================================
# 6. 视觉调用 VLM
# ============================================================
async def demo_vlm():
    print("\n=== 6. 视觉调用 async_call_vlm ===")

    # 方式 A：直接给图片 URL
    url_text = await async_call_vlm(
        [{"role": "user", "content": "描述这张图片的内容。"}],
        use_json=False,
        images=["https://avatars.githubusercontent.com/u/142391390?v=4"],
    )
    print("URL 图片:", url_text)

    # 方式 B：本地图片 → base64 data URL
    data_url = local_image_to_data_url("examples/leimu.jpg")  # 换成你自己的图片
    local_text = await async_call_vlm(
        [{"role": "user", "content": "这张图里有几只动物？"}],
        use_json=False,
        images=[data_url],
    )
    print("本地图片:", local_text)


# ============================================================
# 7. 流式调用（SSE 逐 token）
# ============================================================
async def demo_stream():
    print("\n=== 7. 流式调用 async_stream_call_llm ===")
    collected = []
    async for token in async_stream_call_llm(
        [{"role": "user", "content": "写一首关于大海的五言绝句。"}],
        use_json=False,
    ):
        collected.append(token)
        print(token, end="", flush=True)
    print("\n流式完成，共", len("".join(collected)), "字符")


# ============================================================
# 8. 批量并发调用（受限速器约束）
# ============================================================
async def demo_batch():
    print("\n=== 8. 批量并发调用 ===")

    # 文本批量：[(messages, use_json), ...]，返回与输入顺序一致
    qs = ["什么是 GIL？", "什么是装饰器？", "什么是生成器？"]
    reqs = [([{"role": "user", "content": q}], False) for q in qs]
    results = await async_call_llm_batch(reqs, return_exceptions=True)
    for q, r in zip(qs, results):
        if isinstance(r, Exception):
            print(f"[{q}] 失败: {r}")
        else:
            print(f"[{q}] -> {r[:60]}...")

    # 视觉批量：[(messages, use_json, images), ...]
    v_reqs = [
        ([{"role": "user", "content": "描述图片"}], False, ["https://avatars.githubusercontent.com/u/142391390?v=4"]),
        ([{"role": "user", "content": "描述图片"}], False, ["https://avatars.githubusercontent.com/u/142391390?v=4"]),
    ]
    v_results = await async_call_vlm_batch(v_reqs, return_exceptions=True)
    for r in v_results:
        print("VLM 批量结果:", r if isinstance(r, Exception) else r[:60])


# ============================================================
# 9. 重排序
# ============================================================
async def demo_rerank():
    print("\n=== 9. 重排序 rerank / async_rerank ===")
    docs = [
        "GIL 是 CPython 的全局解释器锁，限制同一时刻只有一个线程执行字节码。",
        "Python 的多线程受 GIL 影响，适合 I/O 密集场景。",
        "内存管理在 Python 中由引用计数和垃圾回收负责。",
    ]
    # 异步
    res = await async_rerank(
        query="什么是 Python 的 GIL？",
        documents=docs,
        top_n=2,
        return_documents=True,
    )
    print("结果数:", len(res.results))
    for r in res.results:
        print(f"  score={r.relevance_score:.4f} | {r.document[:30]}...")

    # 同步（同一套，自动适配上下文）
    res_sync = rerank(query="什么是 GIL？", documents=docs, top_n=2)
    print("同步 rerank 首条 score:", res_sync.results[0].relevance_score)


# ============================================================
# 10. 向量嵌入
# ============================================================
def demo_embedding():
    print("\n=== 10. 向量嵌入 get_embedding_client ===")
    client = get_embedding_client()

    # 文本嵌入（默认 1536d）
    vecs = client.embed_batch(["Python 的 GIL", "内存管理"], dimensions=1536)
    print(f"文本嵌入: {len(vecs)} 条, 每条 {len(vecs[0])} 维")

    # 单条
    vec = client.embed("GIL 是什么", dimensions=1536)
    print(f"单条嵌入维度: {len(vec)}")

    # VL 嵌入（768d，跨模态），需要 vl 候选
    # img = local_image_to_data_url("examples/cat.png")
    # vec_vl = client.embed_image_with_context(img, "这是一只猫")
    # print(f"VL 嵌入维度: {len(vec_vl)}")

    # 纯文本 VL 编码
    # text_vec = client.embed_text_vl("这是一段纯文本")


# ============================================================
# 11. 调用统计
# ============================================================
def demo_stats():
    print("\n=== 11. 调用统计 get_llm_stats / get_all_stats ===")
    print("LLM 统计:", get_llm_stats())
    all_stats = get_all_stats()
    print("全量统计键:", list(all_stats.keys()))
    # 通常结构：
    # {"llm": {...}, "vision": {...}, "embedding": {...}, ...}


# ============================================================
# 12. 错误处理
# ============================================================
async def demo_error_handling():
    print("\n=== 12. 错误处理（熔断 / 重试耗尽）===")

    # 场景 A：直接捕获异常
    try:
        text = await async_call_llm(
            [{"role": "user", "content": "你好"}],
            use_json=True,   # 假设该模型不支持 JSON 模式会抛错
        )
        print(text)
    except RuntimeError as e:
        # _invoke_with_retry 重试耗尽后抛出 RuntimeError
        print("调用失败:", e)

    # 场景 B：主动熔断 —— 用一个不存在的模型触发持续失败
    #   连续失败达到 failure_threshold（默认 2 次）后进入 OPEN，
    #   后续调用会立即抛错而不是继续打 API。
    try:
        await async_call_llm(
            [{"role": "user", "content": "hi"}],
            model_name="nonexistent/not-exist",
        )
    except RuntimeError as e:
        print("指定模型失败:", str(e)[:120])

    # 场景 C：用自定义 RateLimiter 控制并发（展示单元能力）
    from infra_ai import RateLimiter

    rl = RateLimiter(rpm=5, tpm=1000, max_concurrent=1, auto_tune=False)
    print("自定义限速器 rpm 上限:", rl._rpm)


# ============================================================
# 主入口
# ============================================================
async def main():
    if RUN_SYNC:
        # 同步函数在异步入口里也能安全调用（内部用独立线程/临时循环适配）
        demo_sync()

    if RUN_ASYNC:
        await demo_async()

    if RUN_JSON:
        await demo_json_output()

    if RUN_MODEL_OVERRIDE:
        await demo_model_override()

    if RUN_TOOLS:
        await demo_tools()

    if RUN_VLM:
        await demo_vlm()

    if RUN_STREAM:
        await demo_stream()

    if RUN_BATCH:
        await demo_batch()

    if RUN_RERANK:
        await demo_rerank()

    if RUN_EMBEDDING:
        demo_embedding()

    if RUN_STATS:
        demo_stats()

    if RUN_ERROR_HANDLING:
        await demo_error_handling()

    print("\n演示结束。")


if __name__ == "__main__":
    asyncio.run(main())
    # 主流程结束后，显式关闭所有 AsyncOpenAI 客户端及其 httpx2 连接池。
    # 若省略，openai 连接池会在进程退出、事件循环已关闭后被 GC 收尾，
    # 打印 “RuntimeError: generator didn't stop after athrow()” 关停噪音。
    asyncio.run(aclose_all_clients())
    print("test")