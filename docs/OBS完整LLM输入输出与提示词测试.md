# OBS 完整 LLM 输入/输出记录 + 命令行提示词测试

> 对应本仓库新增能力：**每次 LLM 调用完整记录其输入上下文（messages）与返回数据（output）并在观测面板可见**；
> 并提供命令行脚本 `scripts/prompt_lab.py` 做**改输入再测 / 并发对比测试**。
> 相关代码在 feat 分支 `feat/obs-llm-full-recording`。

---

## 1. 两条存储，别混淆

本项目并存两套"记录"，维度不同：

| 存储 | 目录 | 实现 | 内容 |
|---|---|---|---|
| **完整对话历史**（会话级原文） | `data/chat_history/<session_id>.json` | `agent/history.py` | 每会话一个 JSON：`messages` append-only 存最终成型的原文，另有 `summary`（压缩摘要）、`last_compressed_index`（压缩水位）、`created_at/updated_at`。目录可用 `AGENT_HISTORY_DIR` 覆盖 |
| **观测事件**（每次底层 LLM 调用） | `data/obs/events*.jsonl` | `infra_ai/obs_hook.py` → `obs/recorder.py` → `obs/writer.py` | 每次 LLM 调用的完整快照：本件新增的 `messages`(输入) + `output`(返回) + 元数据（model/耗时/token/重试/错误类别）；含工具轮、失败重试、当时实际用的 prompt |

**关系**：`data/chat_history/` = 对话最终长什么样；`data/obs/` = 底层"每次、每个模型、每次重试"到底发了什么、回了什么。做提示词迭代/排障看 `data/obs/` 更直观。

> `OBS_MAX_MB`（默认 50MB）会在 `events.jsonl` 超限时轮转为 `events-<ts>.jsonl`（不删除，query 全量扫描）。

---

## 2. 完整输入/返回记录 —— 数据流

```
data/obs/events*.jsonl                     ← 每次 LLM 调用推入，事件带 messages + output
   ▲ 写入（emit_obs）
infra_ai/inference.py::_invoke_with_retry     非流式成功(+) / 失败
infra_ai/core/streaming.py::_stream_single_model  流式 finally
   │  经 obs_hook.set_obs(tracer.ingest)（app.py::obs.install() 挂钩）
   ▼
obs/recorder.py::Tracer.emit          打信封 seq/ts/ms/trace_id/session_id/parent_id
   ▼
obs/query.py::request(trace_id)       按 trace 读出原始事件（含 messages/output）
   ▼
GET /api/obs/request/{trace_id}       obs/routes.py::obs_request
   ▼
web/obs.html::loadTrace → renderTimeline → llmDetailHTML(ev)   ★ 前端拼接展示
```

### 实现代码（改动点）

**共享序列化器 `infra_ai/core/messages_log.py`（新文件）**
- `serialize_for_obs(messages)`：逐条保 `role`；content 为 str 原样保留；为 list 时按 `text`/`image_url`
  分支，`image_url` 遇 `base64,` 只存 `f"{header}base64,<N chars>"`（**剥图片体，保完整文本/URL**）。
- `output_snippet(message)`：取返回文本 `content`；工具场景 content 为空时回退 tool_calls 函数名列表。
- 放在 `core/` 中性子模块而非 `inference.py`，使 `inference` / `streaming` 都能 import 而不循环依赖。
- `inference._messages_for_file_log` 现委托 `serialize_for_obs`（错误日志行为不变）。

**`infra_ai/inference.py`（非流式）**
- 成功块 emit：`"messages": serialize_for_obs(messages)` + `"output": output_snippet(message)`
  （`message = response.choices[0].message` 已在 emit 前取出）。
- 失败块 emit：`"messages": serialize_for_obs(messages)`（失败也留输入，便于复盘不起效的提示词）。

**`infra_ai/core/streaming.py`（流式）**
- finally emit：`"messages": serialize_for_obs(messages)` + `"output": full_response`。

> 这三个 emit 处都在函数参数作用域内，有现成的 `messages`；改/去字段只动这三处。

### 前端展示 `web/obs.html`

- `renderTimeline()` 的 `llm_call` 分支：`if (ev.messages || ev.output !== undefined) detailHtml = llmDetailHTML(ev)`，
  并给该行加 `clickable` 类 + `data-tl-seq`，行后附隐藏 `<div class="tl-detail">`。
- `llmDetailHTML(ev)`：`messages` → `JSON.stringify(..., null, 2)` 入 `<pre>`；`output` → 字符串原样 / 对象 JSON 入 `<pre>`；均经 `esc()` 转义。
- 文档级委托 handler：点击 `.span-line.clickable` → toggle 同名 `.tl-detail`。
- 旧事件无字段 → 无展开、无箭头（向后兼容）。

> 面板展示的完整输入/输出**只拼在浏览器端渲染**，`summary()`/`requests()` 聚合不读取新字段，服务端接口与统计零改动。

---

## 3. 命令行提示词测试 `scripts/prompt_lab.py`

独立进程、`python scripts/prompt_lab.py ...`；基于观测数据取某次调用、改输入或并发跑，打印对比。

```
# 1) 浏览最近 LLM 调用 + 完整输入/返回（不调 LLM）
python scripts/prompt_lab.py --list
python scripts/prompt_lab.py --show 0          # 0 = 最近一次

# 2) 取某次记录改用户提问，单独跑一次
python scripts/prompt_lab.py --edit 0 --user '换成新问题'
python scripts/prompt_lab.py --edit 0 --user '新问题' --model 'Qwen/Qwen2.5-72B-Instruct-128K' --use-json
python scripts/prompt_lab.py --messages '[{"role":"user","content":"完全自定义"}]'

# 3) 并发测试：同一 base 提示词、逐条替换 user 消息为各变体同时跑
python scripts/prompt_lab.py --edit 0 --variants '["什么是 GIL？","什么是装饰器？","什么是生成器？"]'
```

### 参数

| 参数 | 说明 |
|---|---|
| `--list` | 扫描 `data/obs/events*.jsonl` 列最近 `llm_call`（序号=0 最近；显示时间/模型/作用/成败/输入首行） |
| `--show <idx>` | 打印该次调用完整 `messages` 与 `output` |
| `--edit <idx>` | 以某次已记录调用为 base（其 `messages` 作输入）再跑 |
| `--user 'text'` | （配 `--edit`）替换最后一条 user 消息 content；无则追加 |
| `--messages '<json>'` | 完全自定义 messages（JSON 数组字符串），优先级最高 |
| `--variants '["q1",...]'` | 并发变体：以 base messages 逐条替换 user 消息后 `asyncio.gather` 一起跑，逐条打印耗时不连坐 |
| `--model <name>` | 覆盖模型路由（`model_name=` 传 `async_call_llm`） |
| `--use-json` | 请求 JSON 结构化输出 |
| `--record` | 把本次实验以 `kind="lab"` 落入同一 `data/obs`，可在面板按请求展开查（不计入用户统计） |

### 实现要点

- 骨架沿 `infra_ai/examples/demo.py`：`load_dotenv(根/.env)`、异步 `main`、结束 `aclose_all_clients()`。
- 读事件复用 `obs.writer.iter_event_files()`；**按 `ts`（ISO 时间戳，倒序）排序，不用 `seq`**——脚本自身 Tracer 的
  seq 从 0 起跳，与常驻 app 进程的 seq 互相错位，只有 ts 跨进程可比较。
- Windows 控制台 GBK：入口 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`，避免提示词/返回里的
  任意 Unicode 触发 `UnicodeEncodeError` 崩溃（打不出的字降级为 `?`）。输出用 ASCII 而非 emoji。
- `--record` 时 `obs.install()` 并把运行包在 `obs.new_trace("prompt-lab", kind="lab")`；`obs/query.py::_BACKGROUND_KINDS`
  已含 `"lab"` → 实验不进用户请求统计，仍可经 `/api/obs/request` 查看。
- 并发受 `infra_ai` 内置 `RateLimiter`（`infra_ai/config.yaml` 各模型 rpm/max_concurrent）约束。

---

## 4. 配置

除既有 `obs/config.py` 的 env（`OBS_ENABLED` / `OBS_DIR` / `OBS_MAX_MB` / `OBS_QUERY_WINDOW` / `OBS_QUERY_LIMIT`）外，
本功能**无新增配置**。若要关掉完整输入/返回记录，直接注释/去掉三处 emit 的 `messages`/`output` 字段即可（`OBS_MAX_MB` 负责控制 JSONL 体量）。

---

## 5. 验证

**单元**：`python -m pytest tests/test_obs.py`（本功能不改聚合逻辑，应全绿）。
**记录**：`app.py` 起来发一句对话 → `data/obs/events.jsonl` 新增 `llm_call` 事件含 `messages`+`output`（`kind` 视图在 `/obs.html`）。
**面板**：`/obs.html` → 展开最近请求时间线 → 点某条 `llm_call` 行 → 看到完整输入 messages 与返回 output；暗色主题下显示正常。
**CLI**：
1. `python scripts/prompt_lab.py --list`（列表成功）
2. `python scripts/prompt_lab.py --show 0`（完整打印）
3. 单跑改输入：`--edit 0 --user '新问题'`（返回随输入变化）
4. 并发：`--variants '["q1","q2"]'`（并发 N 条各带耗时）
5. `--record` 后 `/obs.html` 里该 lab trace 可展开（绪 `kind="lab"`，不进用户统计）。
**回归**：`python -m infra_ai.examples.demo`（`RUN_TOOLS=True`）不受影响；`data/chat_history/` 的原文存储不受影响。

---

## 6. 后续开发接线点

- **换/增输入输出字段**：`infra_ai/core/messages_log.py` + `inference.py`/`streaming.py` 三处 emit。
- **改面板展示**：`web/obs.html` 的 `renderTimeline`（llm_call 分支）、`llmDetailHTML`、行点击 handler。
- **扩展 prompt 实验**：`scripts/prompt_lab.py`（变体逻辑在 `_run_variants` / `_set_last_user`）。
- **新增后台实验类 trace**：把新的 kind 加进 `obs/query.py::_BACKGROUND_KINDS`，避免污染用户统计。