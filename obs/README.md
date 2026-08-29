# Agent 对话观测平台（obs）

为 LiveTalking 的智能体对话链路（ASR → LLM → 工具 → TTS → 数字人）提供事件驱动的结构化观测：
**响应耗时**、**成功率**、**每次 LLM 调用明细**（model/耗时/token/重试/错误类别）、**工具调用轮次与逐次明细**。
数据落到 JSONL 事件日志，独立 `web/obs.html` 面板可离线查看。

> 范围仅限 **agent / LLM / 工具层**，不含 TTS / 推流耗时。

---

## 架构一览

```
浏览器 web/obs.html ──fetch──▶ /api/obs/*  (obs/routes.py, aiohttp)
                                 │  读 JSONL
                                 ▼
                          data/obs/events*.jsonl   (obs/writer.py)
                                 ▲  写入
     ┌───────────────────────────┴─────────────────────────────┐
     │  Tracer (obs/recorder.py)：contextvars 追踪 trace 嵌套    │
     └──────────────────────────────────────────────────────────┘
              │ emit()                          │ obs_hook 回调
      server/routes.stream_llm_chat    infra_ai/inference.py + core/streaming.py
      agent/tool_loop.run_tool_loop      （infra_ai 不 import obs，经中性回调槽）
```

- **整条请求 = 一个 async task**（`stream_llm_chat` 里的工具循环/流式/`_feed_talk`/`agent.save`
  同属一个 task）。在 task 开头设置的 `contextvars` 自动向下传播到每个子 `await`，
  **无需改各函数签名传 ID**——子函数只需调 `obs.emit()` 即可自动带上 trace 归属。
- **解耦**：`infra_ai/obs_hook.py` 提供一个中性回调槽 `set_obs(cb)/emit_obs(event)`。
  `infra_ai` 永不 import `obs`；`obs.install()` 启动时 `set_obs(tracer.ingest)` 挂钩。
  好处：观测插件可整体摘除，`infra_ai` 对观测零依赖、零反向耦合。
- **彻底降级**：`OBS_ENABLED=0/false` 时所有 `obs.*` 为空操作（零开销）；写入失败静默跳过，不影响主流程。

## 目录 / 模块

| 模块 | 职责 |
|---|---|
| `obs/config.py`   | env 驱动配置（见下文） |
| `obs/writer.py`   | `JsonlWriter` 追加写盘 + 懒打开 + 超 `OBS_MAX_MB` 轮转 `events-<ts>.jsonl`（不删除，按 seq 全量扫描）；写入失败静默 |
| `obs/recorder.py` | 核心 `Tracer`：contextvars + `emit` 打戳 / `begin_trace` / `end_trace` / `round_span` / `new_trace`；`now_ms()` 用 `time.monotonic()` 纪元 |
| `obs/query.py`    | 纯函数读文件聚合：`summary() / requests() / request(trace_id)`；过滤 `kind=="summary"`；内存算 avg/p50/p90、成功率、per_model、工具次数 |
| `obs/routes.py`   | aiohttp 处理器 + `register(app)` 注册 `/api/obs/*` |
| `obs/__init__.py` | 公开 API（下方） |

## 公开 API（`from obs import ...`）

```python
install()                 # 启动时(app.py main)调用，连接 obs_hook 开始收集
begin_trace(session_id, msg_preview, tool_mode=None, kind="chat")  # 请求开始，返回 trace_id
end_trace(success, fail_reason=None, text_len=0, circuit_open=False,
          tool_rounds=None, llm_calls=None)                        # trace 结束（响应耗时=start→end）
emit(event)               # 发一条 llm_call / tool_call 事件（自动带信封与嵌套归属）
round_span(round_idx)     # async 上下文管理器：包裹一轮工具循环，子调用挂在该轮下
new_trace(session_id, kind="summary")  # 同步上下文管理器：独立 trace（如压缩摘要），隔离统计
setup_routes(app)         # 注册 /api/obs/* 路由（由 server/routes.setup_routes 调用）
is_enabled()              # env 开关
```

## 事件信封（每行 JSONL 都带）

```json
{ "type":"trace_start|llm_call|tool_round|tool_call|trace_end",
  "seq":1042, "ts":"2026-08-29T01:23:45.123", "ms":4827.3,
  "trace_id":"c9f2...", "session_id":"a1b2...",
  "parent_id":null, "span_id":null, "kind":"chat|text|text+tools|summary|stream",
  ... type-specific ... }
```

类型专属字段：

- `trace_start`: `{msg_preview, tool_mode}`
- `llm_call`: `{route, model, mode:"nonstream|stream", use_json, has_tools, attempts, elapsed_ms, input/output/total_tokens, success, fail_reason, err_type}`
- `tool_round`: `{round, n_tool_calls, answered}`
- `tool_call`: `{round, tool, args(截断~200), result_snippet(截断~200), elapsed_ms, success, error}`
- `trace_end`: `{elapsed_ms(=响应耗时), success, fail_reason, tool_rounds, llm_calls, text_len, circuit_open}`

**响应耗时** = `trace_end.elapsed_ms`（trace_start→trace_end 的 monotonic 差）。
**嵌套**：`llm_call` / `tool_call` 挂在 `round_span`（round 号）下，`tool_round` 挂在 trace 下，
末轮直接给答案的 `llm_call` 父回到 trace——由 tracer 读当前 contextvars 自动打 `parent_id`。

## HTTP API

| 端点 | 返回 |
|---|---|
| `GET /api/obs/summary?window=3600` | `{traces, success, success_rate, response_time:{avg,p50,p90}, total_llm_calls, total_tokens, per_model:[{model,calls,fail,avg_elapsed_ms,avg_tokens}], tool_call_counts, total_tool_calls, tool_rounds}` |
| `GET /api/obs/requests?limit=50` | 最近 N 条 `[{trace_id, ts, session_id, msg_preview, elapsed_ms, success, tool_rounds, llm_calls}]` |
| `GET /api/obs/request/<trace_id>` | 该 trace 全部事件，按 seq 升序（供时间线展开） |

`window`（秒）过滤基于 `now_ms()` 的 monotonic 差值；`kind=="summary"` 的压缩 trace 一律不参与统计。
响应都走通用 `{code:0, msg:"ok", data:...}` /错误 `{code:-1, msg}` 约定。

## 配置（env 驱动，无 yaml 改动）

| env | 默认 | 说明 |
|---|---|---|
| `OBS_ENABLED` | `1` | `0/false/no/off` 关闭 → 所有 `obs.*` 立即返回 |
| `OBS_DIR` | `data/obs` | 事件日志目录（首写自动建目录） |
| `OBS_MAX_MB` | `50` | `events.jsonl` 大小轮转阈值（轮转不删除） |
| `OBS_QUERY_WINDOW` | `3600` | 面板 summary 默认时间窗口（秒） |
| `OBS_QUERY_LIMIT` | `50` | `/api/obs/requests` 默认条数 |

## 接线点（改动即在此，供后续开发参考）

1. **trace 根** `server/routes.py::stream_llm_chat`：函数开头 `begin_trace(...)`，`try/finally`
   包住原整体，`finally end_trace(...)`——即使工具循环异常/降级话术/流式错误也保证有 `trace_end`。
   工具循环返回 None ⇒ `fail_reason="tool_loop_max_rounds"`；流式 except ⇒ `"llm_error"`。
2. **LLM 上报（自动）** `infra_ai/inference.py::_invoke_with_retry` 成功/失败块与
   `infra_ai/core/streaming.py::_stream_single_model` 的 `finally` 各 `emit_obs(llm_call)`，
   自动带 attempts / fail_reason / err_type(`classify_error`)。
3. **工具循环** `agent/tool_loop.py::run_tool_loop`：每轮 `async with round_span(idx)`，
   逐工具 `emit(tool_call)`（含计时）。obs 包不可用时优雅降级为 `_DummyRound`，不报错。
4. **压缩摘要独立 trace** `agent/agent.py::_call_summarize`：`with new_trace(sid, kind="summary")`，
   压缩耗时可观测且不污染用户请求的成功率/响应耗时。
5. **启动** `app.py::main`：`obs.install()`（try/except，失败仅告警）。
   路由由 `server/routes.py::setup_routes` 的 try/except 注册 `obs.setup_routes(app)`。

## 前端面板 `web/obs.html`

- 玻璃卡片风格，`nav 渐变 + Bootstrap + jQuery + ECharts`（`web/lib/*` 全离线，无 CDN）。
- 指标卡片（成功率/平均响应/P50/P90/LLM 调用/总 token）＋ 3 张 ECharts 图
  （响应耗时趋势折线、per-model 平均耗时柱状、成功/失败环图）。
- 最近请求表格，行点击展开调用 `/api/obs/request/<trace_id>` 渲染按序时间线
  （llm / tool_round 徽章行，工具参数与结果截断展示并做 HTML 转义）。
- **自动刷新为手动开启**：进入页面只拉一次；导航栏「自动刷新」按钮开启 5s 轮询，
  开启期间展开的 trace 面板保持展开并同步刷新内容（`renderRequests` 重建后对仍展开的 tid 重新 `loadTrace`）。

## 验证

**单元测试** `tests/test_obs.py`（stdlib only，可直接 `python -m unittest tests.test_obs`）：
事件写盘行数 / trace 嵌套（parent_id）/ summary 聚合与 chat 分离 / requests 列表 /
写入失败静默 / `is_enabled()` env 切换。

**端到端**：`run.bat` 启动完整服务后
`POST /human {"sessionid":"demo1","type":"chat","text":"北京今天天气怎么样？"}` 触发 weather 工具循环 →
`GET /api/obs/summary?window=3600` 应见 `success_rate=1.0`、`per_model`、`tool_call_counts.weather=1`、
`response_time`；`/api/obs/request/<trace_id>` 应见 ≥6 事件；浏览器开 `http://<host>:<port>/obs.html`
见图表 + 表格 + 展开时间线。再发一条不需工具的直答应出现 `mode:"stream"` 的 `llm_call`。