# infra_ai — AI 基础设施层

LLM / Embedding / Rerank 调用基础设施，提供熔断、多模型路由、故障转移、速率限制、流式探活与调用统计。

**包名**：`infra-ai`（至少需要 Python ≥ 3.10），安装后 `import infra_ai`。

安装 `infra_ai`（二选一，推荐 uv）：

```bash
# 推荐：uv（生成 .venv + uv.lock）
uv sync --all-extras

# 备选：pip
pip install -e .
```

## 架构

实现按职责拆分至 `infra_ai/` 下的职责单一模块，公开 API 统一由 `infra_ai/__init__.py` 聚合并导出。

```
infra_ai/
├── config.yaml               # 全部配置：路由 / 限速 / 熔断 / 超时重试（含 ${ENV} 占位符）
├── inference.py              # 推理核心：客户端构建、熔断/限速/重试、路由、批量、工具调用、异步 API
├── embedding.py              # Embedding 客户端（text 1536d / vl 768d 路由 + 降级）
├── rerank.py                 # Rerank 客户端（路由 + 降级）
├── core/
│   ├── __init__.py           # core 层规范直达面（from infra_ai.core import ...）
│   ├── config_loader.py      # 配置加载器：解析 config.yaml 与 ${ENV} 占位符，暴露对象式单例 get_config()
│   ├── _async_utils.py       # 通用「异步 ↔ 同步」适配：run_async_in_sync
│   ├── rate_limiter.py       # RateLimiter（RPM/TPM 滑动窗口 + 动态并发 + EMA 自适应）
│   ├── streaming.py          # 流式调用路径（SSE token 流 + 多模型故障转移）
│   ├── sync.py               # call_llm / call_vlm 同步封装（任意上下文安全）
│   ├── stats.py              # token 用量提取 + 调用统计
│   ├── router.py             # ModelSelector + ModelRoutingExecutor + iterate_candidates（三栈共用路由驱动）
│   ├── health_store.py       # 三态熔断器（ModelHealthStore）
│   ├── stream_bridge.py      # ProbeStreamBridge（流式探活 + Probe-and-Commit）
│   └── errors.py             # 结构化错误分类
```

## 公开 API

| 类别 | 函数 | 说明 |
|------|------|------|
| 文本 | `async_call_llm(messages, use_json=True)` | 带路由/熔断/限速/重试的异步调用 |
| 文本+工具 | `async_call_llm_with_tools(messages, tools)` | 原生工具调用，返回 OpenAI message（含 `.tool_calls`） |
| 视觉 | `async_call_vlm(messages, use_json=True, images=[...])` | VLM 调用 |
| 批量 | `async_call_llm_batch(requests)` / `async_call_vlm_batch(requests)` | 并发调用，受限速器约束 |
| 流式 | `async_stream_call_llm(messages, use_json=False)` | 逐个 token yield，多模型故障转移 |
| 同步 | `call_llm(...)` / `call_vlm(...)` | 任意上下文（脚本/FastAPI/Jupyter）安全 |
| 工具 | `local_image_to_data_url(image_path)` | 本地图片 → base64 data URL |
| 统计 | `get_llm_stats()` / `get_all_stats()` | 文本 / 视觉 / 全栈累计统计 |
| Embedding | `get_embedding_client()` / `get_embedding_stats()` | text(1536d) / vl(768d) 嵌入 |
| Rerank | `rerank(...)` / `async_rerank(...)` / `get_rerank_client()` / `get_rerank_stats()` | 同步 / 异步重排序 |
| 限速 | `RateLimiter` | 可直接构造的自定义限速器 |

所有符号已在包顶层导出：`from infra_ai import async_call_llm, rerank, get_all_stats, ...`。

## 运行 Demo

`examples/demo.py` 覆盖 12 类调用场景（同步 / 异步 / JSON 结构化 / 工具 / 视觉 / 流式 / 批量 / 重排 / 嵌入 / 统计 / 错误处理），通过文件顶部的 `RUN_xxx` 开关控制一次性跑哪些段。

### 1. 初始化环境

环境已用 [uv](https://docs.astral.sh/uv/) 接管，依赖由 `uv.lock` 锁定：

```bash
uv sync --all-extras   # 创建 .venv 并安装全部依赖（含 dev / pytest）
```

### 2. 配置 API Key

`config.yaml` 中 `${ENV_VAR}` 占位符在加载时从环境变量解析。默认启用的候选依赖：

| 环境变量 | 用途 |
|----------|------|
| `DASHSCOPE_API_KEY` | 文本 / 视觉 LLM 主候选（百炼 qwen） |
| `SF_API_KEY` | embedding / rerank 等（硅基流动） |

推荐复制项目内已备好的 `.env.example` 为 `.env`（已被 `.gitignore` 忽略，不会提交），填入真实 key：

```bash
cp .env.example .env    # Windows PowerShell: Copy-Item .env.example .env
```

### 3. 运行

```bash
uv run --env-file .env python examples/demo.py
```

若 key 已写入系统环境变量，可省略 `--env-file`：`uv run python examples/demo.py`。

### 4. 段落开关

demo 顶部 `RUN_xxx = True/False` 决定跑哪些段，默认开启：

| 开关 | 场景 | 备注 |
|------|------|------|
| `RUN_SYNC` | 同步调用 | 默认开 |
| `RUN_ASYNC` | 异步：串行 vs 并发耗时对比 | 默认开，会真实调用 2×N 次 API |
| `RUN_TOOLS` | 原生工具调用 | 默认开 |
| `RUN_STREAM` | 流式输出（SSE 逐 token） | 默认开 |
| `RUN_RERANK` | 重排序 | 默认开，需 `SF_API_KEY` |
| `RUN_STATS` | 调用统计 | 默认开，本地统计、无需任何 key |
| `RUN_JSON` | JSON 结构化输出 | 默认关 |
| `RUN_MODEL_OVERRIDE` | 指定模型覆盖路由 | 默认关 |
| `RUN_VLM` | 视觉调用 VLM | 默认关，需图片 |
| `RUN_BATCH` | 批量并发调用 | 默认关 |
| `RUN_EMBEDDING` | 向量嵌入 | 默认关 |
| `RUN_ERROR_HANDLING` | 熔断 / 重试耗尽演示 | 默认关，故意触发失败 |

只跑某一段：把其余 `RUN_xxx` 全部置 `False` 即可。若对应 provider 的 key 未配齐，相关调用会走候选回退甚至直接失败——这是预期的。

## 快速开始

```python
from infra_ai import call_llm, async_call_llm, async_stream_call_llm

# 同步调用（自动适配同步/异步上下文）
text = call_llm([{"role": "user", "content": "什么是 GIL？"}])

# 异步调用
text = await async_call_llm([{"role": "user", "content": "什么是 GIL？"}])

# 流式调用
async for token in async_stream_call_llm([{"role": "user", "content": "写一首诗"}], use_json=False):
    print(token, end="")
```

## 调用链路

```
业务层调用
    ↓
async_call_llm / async_call_vlm / async_call_llm_with_tools
    ↓
[ModelRoutingExecutor]                     # router.execute("chat"|"vision", caller)
    ├── ModelSelector.select()             #   优先级排序 + 熔断过滤 + 深度思考过滤
    └── iterate_candidates()               #   共享路由-回退驱动（LLM / embedding / rerank 共用）
        └── 遍历候选模型（首包失败自动切换到下一个）
            ├── health_store.allow_call()  #   熔断器检查（OPEN 拒绝 / HALF_OPEN 单飞探活）
            ├── RateLimiter.acquire()      #   RPM + TPM + 并发三重限速
            ├── _invoke_with_retry()       #   线性递增超时 + jitter 退避重试
            │   └── classify_error()       #   错误分类（致命错误立即终止不重试）
            └── mark_success / mark_failure#   熔断器状态更新（由 attempt/叶子负责）
    ↓
无可用候选 → RuntimeError
```

## 模块说明

### 配置加载 (`core/config_loader.py` + `config.yaml`)

所有配置集中在 `config.yaml`，加载时递归解析 `${ENV_VAR}` 占位符，敏感值（api_key 等）不落盘。

- `${ENV_VAR}` → 环境变量缺失则为空串
- `${ENV_VAR:-default}` → 环境变量缺失时回退 `default`（default 可再含 `${...}` 占位）

通过 `get_config()` 返回对象式单例：

```python
from infra_ai.core.config_loader import get_config
cfg = get_config()
cfg.LLM_ROUTING / cfg.EMBEDDING_ROUTING / cfg.RERANK_ROUTING / cfg.LLM_REQUEST_TIMEOUT ...
```

### 速率限制 (`rate_limiter.py`)

`RateLimiter` 控制 **RPM + TPM + 最大并发数** 三重约束：

- **RPM**：60s 滑动窗口内请求次数上限；**TPM**：60s 窗口内 token 消耗上限（0 = 不限制）
- **并发**：动态可调上限，基于 EMA 观测值自适应推荐（`recommend_max_concurrent()`）
- `auto_tune=True` 时自动应用推荐值，否则仅日志建议（默认）

```python
rl = RateLimiter(rpm=60, tpm=100000, max_concurrent=10, auto_tune=False)
await rl.acquire(max_wait=300.0)   # 获取槽位，超时抛出 TimeoutError
try:
    ...
finally:
    rl.release()
```

### 熔断器 (`health_store.py`)

三态状态机：**CLOSED → OPEN → HALF_OPEN → CLOSED**，每个模型独立跟踪健康。

| 状态 | 含义 | 行为 |
|------|------|------|
| CLOSED | 正常 | 所有请求通过 |
| OPEN | 熔断 | 拒绝所有请求，冷却期后转 HALF_OPEN |
| HALF_OPEN | 探活 | 仅放行 1 个探活请求，成功恢复 CLOSED，失败重新熔断 |

配置（`config.yaml` → `llm.circuit_breaker`）：`failure_threshold: 2, open_duration_sec: 30`。

### 多模型路由 (`router.py`)

- `ModelSelector` — 从 `LLM_ROUTING` 读取候选，过滤禁用项 / 深度思考不匹配项 / 熔断项，按优先级排序，首选模型置顶
- `ModelRoutingExecutor` — 遍历候选调用，成功立即返回，全部失败抛出 `RuntimeError`
- `iterate_candidates` — **LLM / embedding / rerank 三条路径共用的路由-回退驱动**：顺序遍历已排序候选，熔断过滤 + 逐候选调用 + 全败抛错；健康标记 `mark_success/mark_failure` 由叶子（attempt）负责
- 数据模型：`ModelCandidate`（**LLM / embedding / rerank 统一建模**）、`ModelTarget`（运行时目标）

三栈候选均以类型化 `ModelCandidate` 表达，路由决策（排序 → 熔断过滤 → 回退）收敛到一处；各栈的 HTTP retry 叶子（`requests`/`httpx`/LangChain）按自身 I/O 栈保留，不做强行统一。

### 流式故障转移 (`streaming.py` + `stream_bridge.py`)

Probe-and-Commit 模式：

```
候选模型 A → _stream_single_model → ProbeStreamBridge.await_first_packet(60s)
  → 收到首包 → commit() → 冲刷缓冲 → 后续 token 直通 yield
  → 超时/错误 → discard() → 候选模型 B → ...
```

token 用量优先取流末 chunk 的真实 `usage`，字符估算仅作兜底。

### 调用核心 (`inference.py`)

`_invoke_with_retry` 支持：

- **超时线性递增**：`timeout_n = timeout * (1 + scale * n)`（如 base=180s, scale=0.4 → 180/252/324s）
- **jitter 退避重试**：退避时间随重试次数增大并加入随机抖动
- **致命错误短路**：401/403/4xx 配置类错误立即终止，不进入退避重试
- **工具调用**：传入 `tools` 后透传 `create(tools=...)`，返回原生 OpenAI message（含 `.content` / `.tool_calls`）
- 失败写入 JSONL 错误日志（保留完整消息、剥离 base64 图片，记录限速/并发/统计快照）

`async_call_llm` / `call_llm` 支持 `model_name` 参数覆盖路由，直达指定模型。

### 错误分类 (`errors.py`)

`classify_error(exception)` → 8 种错误类型：

- **可重试**（`should_retry()` 返回 True）：`RATE_LIMITED`、`SERVER_ERROR`、`NETWORK_ERROR`
- **致命**（`is_fatal()` 返回 True）：`UNAUTHORIZED`、`CLIENT_ERROR`、`PROVIDER_ERROR`
- **其他**：`INVALID_RESPONSE`、`UNKNOWN`

熔断器只对可重试错误计数；致命错误（问题在配置）不触发熔断、不重试。

### 统计 (`stats.py`)

- `_extract_token_usage` 从 openai SDK 响应读取 `usage`（兼容 CompletionUsage 对象与 dict 包装）
- `get_llm_stats()` 返回文本与视觉的累计统计；`get_all_stats()` 额外包含 embedding / rerank 等所有已注册累加器

### 同步封装 (`core/_async_utils.py` + `sync.py`)

`run_async_in_sync(coro)` 自动检测当前是否在 asyncio 事件循环中：不在循环内用 `asyncio.run()` 创建临时循环；已在循环内（如 FastAPI/uvicorn）则在独立线程创建新循环执行。`call_llm` / `call_vlm` 与 rerank/embedding 的同步壳共用这一工具。

## 配置 (`config.yaml`)

```yaml
llm:
  request_timeout: 180.0        # 单次请求超时（秒）
  timeout_scale: 0.4            # 超时线性递增系数
  max_retries: 3                # 最大重试次数
  retry_backoff: 2.0            # 重试退避（秒）
  auto_tune_concurrency: false  # 是否自动调整并发
  error_log: "./logs/llm_errors.jsonl"

  circuit_breaker:
    failure_threshold: 2        # 连续失败 N 次 → 熔断
    open_duration_sec: 30       # 熔断冷却时间（秒），过期进入 HALF_OPEN 探活

  rate_limits:
    default: { rpm: 60, tpm: 100000, max_concurrent: 10 }

  routing:
    chat:
      default_model: sf-chat
      candidates:
        - id: sf-chat
          provider: siliconflow
          model: ${SF_CHAT_MODEL:-Qwen/Qwen2.5-72B-Instruct-128K}
          base_url: ${SF_BASE_URL:-https://api.siliconflow.cn/v1}
          api_key: ${SF_API_KEY}
          priority: 1
          enabled: false
        # ... 其余候选（如 bailian-chat）同理
    vision:
      default_model: sf-vision
      candidates: [ ... ]

embedding:
  timeout_scale: 0.4
  rate_limits: { default: { rpm: 60, tpm: 100000, max_concurrent: 10 } }
  routing:
    text: { candidates: [ { id: sf-text-embedding, model: ${SF_EMBEDDING_MODEL:-...}, ... } ] }
    vl:   { candidates: [] }

rerank:
  rate_limits: { default: { rpm: 60, tpm: 100000, max_concurrent: 10 } }
  routing:
    text:
      default_model: sf-rerank
      candidates: [ { id: sf-rerank, model: ${SF_RERANK_MODEL:-BAAI/bge-reranker-v2-m3}, ... } ]
```

各候选的 `model` / `base_url` / `api_key` 均可通过 `${ENV_VAR}` / `${ENV_VAR:-default}` 从环境变量读取（如 `SF_API_KEY`、`DASHSCOPE_API_KEY`）。

## Embedding 使用示例

```python
from infra_ai.embedding import get_embedding_client
from infra_ai import local_image_to_data_url

client = get_embedding_client()

# 文本嵌入（1536d）
vecs = client.embed_batch(["Python 的 GIL", "内存管理"], dimensions=1536)

# 多模态 VL 嵌入（768d）：文本 + 图片跨模态检索
img = local_image_to_data_url("cat.png")
vec = client.embed_image_with_context(img, "这是一只猫")

# 用 VL 模型对纯文本编码（跨模态检索用）
text_vec = client.embed_text_vl("这是一段纯文本")
```

带多模型路由与降级复用同一套熔断器（`iterate_candidates` 驱动 + `ModelCandidate` 建模），候选失败自动切换。内部路由层为异步实现，公共方法（`embed_batch` 等）以同步壳暴露，同步/异步上下文均可安全调用。

## Rerank 使用示例

```python
from infra_ai import rerank, async_rerank

# 异步调用
results = await async_rerank(
    query="什么是 Python 的 GIL？",
    documents=["GIL 是全局解释器锁...", "Python 内存管理..."],
    top_n=3,
    return_documents=True,
)

# 同步调用（自动适配同步/异步上下文）
results = rerank(
    query="什么是 Python 的 GIL？",
    documents=["GIL 是全局解释器锁...", "Python 内存管理..."],
    top_n=3,
)

# 使用结果
for r in results.results:
    print(f"[{r.index}] score={r.relevance_score:.4f}")
```

`RerankClient`（httpx 实现）支持 OpenAI 兼容 rerank 接口，独立限速（`RERANK_RATE_LIMITS`），部分模型可传 `instruction` / `max_chunks_per_doc` / `overlap_tokens` 参数。

## 公开 API 导入面

- 顶层聚合：`infra_ai/__init__.py`（`from infra_ai import async_call_llm, rerank, ...`）
- core 直达面：`infra_ai/core/__init__.py`（`from infra_ai.core import async_call_llm, RateLimiter, ...`）
- 直连底层模块：`from infra_ai.inference import ...` / `from infra_ai.embedding import ...` / `from infra_ai.rerank import ...` / `from infra_ai.core.config_loader import get_config` / `from infra_ai.core.router import get_router` / `from infra_ai.core.errors import classify_error, ModelClientErrorType`