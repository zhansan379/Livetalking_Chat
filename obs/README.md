# Agent 对话观测平台（obs）

为 LiveTalking 的智能体对话链路（ASR → LLM → 工具 → TTS → 数字人）提供事件驱动的结构化观测：
**响应耗时**、**成功率**、**每次 LLM 调用明细**（model/耗时/token/重试/错误类别）、**工具调用轮次与逐次明细**，
以及 **ASR 转录**（推理耗时/RTF/音频时长）与 **TTS 合成**（合成耗时/队列延迟/音频时长/重试/截断）。
数据落到 JSONL 事件日志，独立 `web/obs.html` 面板可离线查看。

> 范围：**ASR → agent/LLM/工具 → TTS** 全链路。TTS 合成耗时纳入；推流队列→播放的延迟为可选扩展。

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
emit_explicit(event, *, trace_id, session_id, parent_id=None, kind="chat")
                          # 跨线程显式 ID 的 emit（base_tts 工作线程用它把 TTS 事件挂回聊天 trace）
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
  "parent_id":null, "span_id":null, "kind":"chat|text|text+tools|summary|stream|asr",
  ... type-specific ... }
```

类型专属字段：

- `trace_start`: `{msg_preview, tool_mode}`
- `llm_call`: `{route, model, mode:"nonstream|stream", use_json, has_tools, attempts, elapsed_ms, input/output/total_tokens, success, fail_reason, err_type, messages, output}`
  —— 新增 **`messages`**：本次调用完整输入上下文（经 `infra_ai/core/messages_log.serialize_for_obs` 序列化，保文本/URL、剥 base64 图片体）；
    **`output`**：返回数据（字符串；工具场景无 content 时为 tool_calls 名列表；失败事件无此字段）。面板据此展开看完整输入/返回。
    > 体量提示：完整输入/返回会让 JSONL 变大，靠 `OBS_MAX_MB` 轮转兜底；不想要可在 `_invoke_with_retry`/`_stream_single_model` 的 emit 处去掉该对字段。
- `tool_round`: `{round, n_tool_calls, answered}`
- `tool_call`: `{round, tool, args(截断~200), result_snippet(截断~200), elapsed_ms, success, error}`
- `tts_call`: `{provider, voice, rate, text(截断40), text_len, attempts, elapsed_ms(合成耗时), queue_ms(入队→开始), audio_ms(音频时长), success, fail_reason(exception/empty_audio/truncated/max_retries/barge_in), err_type, retried, truncated}` —— `kind="chat"`、`parent_id == 聊天 trace_id` 的**子事件**（挂在该请求下）
- `asr_call`: `{audio_ms, audio_len_s, inference_ms, elapsed_ms, rtf, text(截断40), text_len, empty, success, fail_reason(inference_exception/audio_too_short), err_type}` —— `kind="asr"` 事件。**全链路模式下**它是 chat trace 的首事件（`span_id/parent_id == 聊天 trace_id`），与 LLM/TTS 同处一条 trace；无 echo 的独立 ASR 则单独成一条 `trace_id`（按 `session_id`(wav_name) 关联）
- `trace_end`: `{elapsed_ms(=响应耗时), success, fail_reason, tool_rounds, llm_calls, text_len, circuit_open}`

**两套耗时口径**：
- `response_time`：聊天段（`trace_start→trace_end`），不含 ASR 与 TTS。
- `pipeline` / `pipeline_ms`：**全链路**（栏级聚合同一 trace 的起止）：起点为 `asr_call` 的 emit 时刻减去推理耗时（近似语料起始），终点为同 trace 内最后一个事件（异步 TTS 合成完成，可晚于 `trace_end`）。纯聊天 trace（无 ASR/TTS）退化为 `elapsed_ms`。分布式 ASR 独立 trace 的语料不参与。

**响应耗时** = `trace_end.elapsed_ms`（trace_start→trace_end 的 monotonic 差）。
**嵌套**：`llm_call` / `tool_call` 挂在 `round_span`（round 号）下，`tool_round` 挂在 trace 下，
末轮直接给答案的 `llm_call` 父回到 trace——由 tracer 读当前 contextvars 自动打 `parent_id`。

## HTTP API

| 端点 | 返回 |
|---|---|
| `GET /api/obs/summary?window=3600` | `{traces, success, success_rate, response_time:{avg,p50,p90}, pipeline:{avg,p50,p90}, total_llm_calls, total_tokens, per_model:[{model,calls,fail,avg_elapsed_ms,avg_tokens}], tool_call_counts, total_tool_calls, tool_rounds, asr:{calls,success_rate,avg_ms,total_audio_ms,avg_rtf}, tts:{calls,success_rate,avg_ms,retry_count,truncation_count}}` |
| `GET /api/obs/requests?limit=50` | 最近 N 条聊天请求 `[{trace_id, ts, session_id, msg_preview, elapsed_ms, pipeline_ms, success, tool_rounds, llm_calls}]`（asr/summary trace 不在此列） |
| `GET /api/obs/request/<trace_id>` | 该 trace 全部事件，按 seq 升序（供时间线展开）；chat trace 会含其下的 `tts_call` 子事件 |
| `GET /api/obs/pipeline?session_id=&limit=` | 该会话的全链路 trace 组 `{session_id, traces:[{trace_id, kind:"asr&#124;chat", session_id, ts, success, events}]}`（ASR 与 chat 拼成整条链） |

`window`（秒）过滤基于 `now_ms()` 的 monotonic 差值；后台维护类 trace（`kind` ∈ `summary`/`longterm_extract`/`longterm_consolidate`/**`lab`**）一律不参与用户请求统计（`obs/query.py::_BACKGROUND_KINDS`）。`lab` 是 `scripts/prompt_lab.py --record` 的并发实验 trace。
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
   并在 `begin_trace` 后把 `_obs = {trace_id, session_id, parent_id}` 蹭进 `datainfo`——
   让它沿 `put_msg_txt` 传到 TTS 工作线程（线程拿不到 contextvars，靠它显式挂回本 trace）。
2. **LLM 上报（自动）** `infra_ai/inference.py::_invoke_with_retry` 成功/失败块与
   `infra_ai/core/streaming.py::_stream_single_model` 的 `finally` 各 `emit_obs(llm_call)`，
   自动带 attempts / fail_reason / err_type(`classify_error`)；并带 **`messages` + `output`**
   （经 `infra_ai/core/messages_log.serialize_for_obs` / `output_snippet`，见下节「完整输入/返回记录」）。
3. **工具循环** `agent/tool_loop.py::run_tool_loop`：每轮 `async with round_span(idx)`，
   逐工具 `emit(tool_call)`（含计时）。obs 包不可用时优雅降级为 `_DummyRound`，不报错。
4. **压缩摘要独立 trace** `agent/agent.py::_call_summarize`：`with new_trace(sid, kind="summary")`，
   压缩耗时可观测且不污染用户请求的成功率/响应耗时。
5. **TTS 合成（统一单点）** `tts/base_tts.py::process_tts`：`_run_tts_observed` 用 try/except 包住
   各 provider 的 `txt_to_audio`，跨线程 `emit_explicit(tts_call, ...)`（盖 provider/耗时/队列/
   成败/重试/截断），覆盖所有 TTS backend，新 provider 零埋点。`put_msg_txt` 打 `enqueued_ms`
   供 queue_ms；provider（如 edge）把咽下的失败/重试写 `self.last_tts` 富化。SSE `notify` 剔除
   `_obs` 前缀字段，避免内部元数据泄出。
6. **ASR 下发回合 id（单条全链路 trace）** `server/asr_server.py::asr_websocket_handler`：
   `is_speaking is True` 时服务端生成回合 `utterance_tid`；`is_speaking is False` 分支（await 推理后、
   handler 协程内）用 `emit_explicit(asr_call, trace_id=utterance_tid, kind="asr")` 打点（不新建
   独立 trace），并把 `trace_id` 塞进转录响应 JSON 随 `is_final` 一起下发给浏览器。
7. **浏览器 echo + chat 复用** `web/avatar-chat.html`：`runTranscription` 把 `m.trace_id` 解析出来，
   `drainQueue → sendChat(text, trace_id)` 在 `/human` POST body 里带 `trace_id`。
   `server/routes.py::human` 读取并传 `stream_llm_chat(..., trace_id=...)`，后者
   `begin_trace(session_id, msg, tool_mode, trace_id=trace_id)` **复用同一 id** —— ASR→LLM/工具→TTS
   由此合并进**一条 trace**（`asr_call` 排最前，`kind="asr"` 只喂 ASR 聚合，不干扰聊天统计）。
   `trace_id` 为空（echo 态/老客户端/键盘直答）时退化为自旋独立 trace，向后兼容。
7. **启动** `app.py::main`：`obs.install()`（try/except，失败仅告警）。
   路由由 `server/routes.py::setup_routes` 的 try/except 注册 `obs.setup_routes(app)`。

## 前端面板 `web/obs.html`

- 玻璃卡片风格，`nav 渐变 + Bootstrap + jQuery + ECharts`（`web/lib/*` 全离线，无 CDN）。
- 指标卡片（成功率/平均响应/P50/P90/**全链路平均·P90**/LLM 调用/总 token）＋ **全链路阶段卡**
  （ASR 调用/成功率/平均推理/RTF、TTS 句子/成功率/平均合成/重试/截断）＋ 5 张 ECharts 图
  （响应耗时趋势折线〔含**全链路(ASR→TTS)虚线**对比〕、per-model 平均耗时柱状、成功/失败环图、
  各阶段平均耗时柱状、TTS 成功率环图）。
- 最近请求表格（含「全链路」耗时列），行点击展开调用 `/api/obs/request/<trace_id>` 渲染按序时间线
  （asr_call → 请求开始 → llm → tool_round / tts_call → 请求结束 徽章行，工具参数与结果截断展示并做 HTML 转义）。
- **LLM 调用行可展开完整输入/返回**：时间线里带 `ev.messages`/`ev.output` 的 `llm_call` 行标为可点击（`.span-line.clickable`），
  点击 toggle 一个 `.tl-detail` 快，由 `llmDetailHTML(ev)` 拼装：`messages`（`JSON.stringify(...,null,2)`）与 `output`（字符串原样/对象 JSON）各入 `<pre>`，
  已做 HTML 转义防注入。旧事件无字段则不显示、无展开箭头（向后兼容）。
  前端拼接点：`web/obs.html` 的 `renderTimeline()`（llm_call 分支 `:788`）+ `llmDetailHTML()`（`:827`）+ 行点击 handler（`:845`）。
- **自动刷新为手动开启**：进入页面只拉一次；导航栏「自动刷新」按钮开启 5s 轮询，
  开启期间展开的 trace 面板保持展开并同步刷新内容（`renderRequests` 重建后对仍展开的 tid 重新 `loadTrace`）。

## 完整输入/返回记录 与 命令行提示词测试

- **记录**：`llm_call` 事件现在完整携带输入上下文（`messages`）与返回数据（`output`），面板可展开查看。
  实现集中在 `infra_ai/core/messages_log.py`（共享序列化器）+ `inference.py` / `core/streaming.py` 的 emit 处。
- **提示词测试/并发**：`scripts/prompt_lab.py` 基于观测数据取某次调用、改输入或并发跑多个变体。

详细开发说明见 **`docs/OBS完整LLM输入输出与提示词测试.md`**（存储位置、数据流、接线点、CLI 用法、验证）。

## 验证

**单元测试** `tests/test_obs.py`（stdlib only，可直接 `python -m unittest tests.test_obs`）：
事件写盘行数 / trace 嵌套（parent_id）/ summary 聚合与 chat 分离 / requests 列表 /
写入失败静默 / `is_enabled()` env 切换。

**端到端**：`run.bat` 启动完整服务后
`POST /human {"sessionid":"demo1","type":"chat","text":"北京今天天气怎么样？"}` 触发 weather 工具循环 →
`GET /api/obs/summary?window=3600` 应见 `success_rate=1.0`、`per_model`、`tool_call_counts.weather=1`、
`response_time`、`tts.calls≥1`（合成句子数）；`/api/obs/request/<trace_id>` 应见 ≥6 事件，且含
`tts_call` 子事件（`parent_id == 该 trace`）；浏览器开 `http://<host>:<port>/obs.html` 见图表 + 表格
+ 展开时间线（含 TTS 合成徽章）。再发一条不需工具的直答应出现 `mode:"stream"` 的 `llm_call`。

**全链路（ASR 单 trace）**：浏览器连 ASR WebSocket（`/api/asr`，`config.wav_name` 设为会话 id）说一句话 →
浏览器把响应里的 `trace_id` echo 进 `/human`。`GET /api/obs/request/<trace_id>` 返回**同一条** trace 的
`asr_call → trace_start → llm_call → tts_call → trace_end`（按 seq 升序，ASR 在前）；`summary()["asr"]`
见 `calls/success_rate/avg_ms/avg_rtf`，且聊天 `traces`/`success` 计数不因合并而改变。面板「全链路阶段」卡、
「各阶段平均耗时」图与最近请求展开时间线都把这 ASR/LLM/TTS 三段呈现为一条时间线。
`GET /api/obs/pipeline?session_id=<sid>` 仍可用：按 `session_id` 分组（kind 取根 `trace_start`），
对旧版独立 `kind="asr"` trace 与 echo 缺失的回退场景照常起关联作用。