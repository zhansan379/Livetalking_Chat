# TTS 服务大白话：实现、线程、父类管理、观测埋点

> 读者对象：刚接触这项目的同学。全文用大白话，配代码位置，好对着源码读。
> 一句话总览：TTS 不是"一条同步从头等到尾"，也不是"全异步协程"，而是 **"生产者-消费者 + 每条引擎一条独立 worker 线程"**，由 `BaseTTS` 父类统一管线程、管队列、管埋点。

---

## 1. 它到底是什么并发模型

| 你可能的猜测 | 实际 |
|---|---|
| 主线程同步等合成完才继续？ | 不是。HTTP 层只管"把句子扔进队列"就立刻返回 |
| 全是 asyncio 异步协程？ | 不是。核心是靠**线程 + 队列**，合成本身是同步的 |
| 每个引擎各拉线程同时抢活？ | 不是。只有一个 TTS 实例被 render，只有一条 worker 线程在消费 |

实际是：**`BaseTTS` 每个实例自己拉起一条 `threading.Thread`**，用 `queue.Queue` 收句子、死循环消费。整条链路"同步（堵塞）"执行，靠"每层一个队列 + 一条线程"把消息一级一级往下传。

好处：生产者丢句不等待 → HTTP 协程不被卡；TTS 卡住只卡它自己的线程，不拖累 ASR / 渲染 / 网络线程。

---

## 2. 总体过程：文字怎样变成声音

```
server /human 路由（aiohttp 异步层）
   │  put_msg_txt(text, datainfo)        ← 生产者，只入队，立刻返回
   ▼
┌─ BaseTTS.msgqueue (Queue) ──── 队列缓冲
   │
   ▼  worker 线程：process_tts 死循环 (base_tts.py:80)
   │   取句 → 代际自检(旧代际丢弃=真停嘴) → state=RUNNING
   ▼
   _run_tts_observed(msg)                ← 统一埋点(发 tts_call)
   │   调 self.txt_to_audio(msg)         ← 真正合成，同步执行
   │     · 单引擎: edge/豆包…直接合成
   │     · 池子:   TTSPool 逐候选回退 (executor.py)
   ▼
   parent.put_audio_frame(chunk, eventpoint)  ← 每20ms一帧塞给 avatar
   ▼
   avatar 内部 asr / res_frame_queue / WebRTC…（继续下一级线程）
```

要点：**异步协程（aiohttp）只在最上层**，负责收请求、把句子塞进队列就完事；**真正的合成耗时发生在 TTS 自己的 worker 线程里**。

---

## 3. 父类统一管什么（BaseTTS）

任何 TTS 引擎（edge / azure / 豆包 / 候选池…）都继承 `BaseTTS`，**线程管理被父类统一封装**，子类只写合成，完全不碰线程：

```
BaseTTS (父类)
 ├─ __init__      排队列 Queue、算 chunk(20ms)、代际 _epoch、登记 last_tts
 ├─ render()      = 启动一条 worker 线程跑 process_tts        ← 线程入口
 ├─ process_tts() = 死循环：取句→代际自检→_run_tts_observed   ← 线程体
 ├─ put_msg_txt() = 引流入口：入队 + 打 enqueued_ms 时间戳
 ├─ flush_talk()  = 代际+1 + 清队列 + state=PAUSE（打断停嘴）
 └─ txt_to_audio()= 抽象方法，子类实现真正的合成
```

父类统一管理还体现在三处：

- **统一埋点**：`_run_tts_observed` 在父类，**所有新引擎零埋点自动获得统计** (base_tts.py:98-100)。
- **统一结果登记**：`tts_ok()` / `tts_fail()` 在父类，统一写 `last_tts`，子类只在"成败分界"处调一行。
- **统一打断**：`flush_talk` 父类实现，候选池再覆写去传导给已实例化的子引擎 (executor.py:180)。

### 引擎内部那一层异步

worker 线程里合成是同步阻塞的，但引擎内部自己想办法拿数据不干等：

- `edge.py:43`：在同步线程里套 `asyncio.new_event_loop()` + `run_until_complete()`，**同步跑协程**去拿数据。
- 其他流式引擎（豆包/azure…）：靠**异步回调**往流里灌 PCM，但对上层仍是同步返回。

---

## 4. 到底选择谁 render？只选一个

**只 `render` 一个**：要么是那个单引擎，要么是一个装着 N 个替补候选的 `TTSPool`。判定逻辑在 `tts/__init__.py::select_tts`：

```python
enabled_count = sum(1 for c in build_candidates(cfg.ROUTING.candidates) if c.enabled)  # 只数"启用"的
if cfg.ENABLED and enabled_count >= 2:
    return TTSPool(opt, parent)     # 候选池（句内回退+熔断）
else:
    return create("tts", opt.tts, ...)  # 单后端（旧行为）
```

关键：**只数 `enabled` 的候选**。配置文件里列了 10 个，若只启用 1 个也走单引擎分支。

### 池子也是 BaseTTS → 对外无感

**`TTSPool` 自己也是 `BaseTTS` 子类** (executor.py:34)。上层 avatar 调 `select_tts()` 返回的可能是单引擎也可能是池子，**对外只认 `put_msg_txt` / `render` / `flush_talk`**，完全不用管里面是谁干活。

- **线程层面**：永远只有一个 BaseTTS 实例被 render、只有一条 worker 线程消费 TTS 队列。
- **池子内部**：复用在池子这条线程上，**同一条线程里逐个试候选** —— 候选A挂→B→C，谁成谁出声，成功即返回 (executor.py:150-162)。候选引擎**不 render、不起线程**，只是在池线程里被同步调用。

打比方：
> 单引擎 = 只雇一个员工干活。
> 候选池 = 雇了一个"调度经理"(TTSPool，就一条线程)，经理手下挂着几个随时能顶上的**备胎员工**。来了活经理在工位上挨个叫，这个不行叫下一个，但**同一时刻就一个人在动手**。

### 为什么这样（而不是并行抢）
1. **句级/引擎级串行**：同一句话绝不被两个引擎同时合成，避免出声串扰。
2. **换引擎零成本**：备胎引擎**懒实例化**，第一次被用到才造出来 (executor.py:52)，平时不占资源。
3. **入口统一**：avatar 永远只调那三个方法。

---

## 5. 观测埋点：3 个事件

同一个 `obs` 是**观测面板**（不是 OBS 直播）。全链路一共 3 种 TTS 事件：

| 事件 | 埋点函数 | 衡量什么 |
|---|---|---|
| `tts_call` | `base_tts.py::_run_tts_observed()`（每句一条） | 合成成败 / 耗时 / audio_ms / attempts / retried / truncated |
| `tts_candidate` | `executor.py::_emit_candidate()`（每个失败/熔断跳过候选） | 哪个候选挂、失败原因、被熔断跳过 |
| `tts_playback` | `base_avatar.py::notify()`（端送达） | 音频**真正送达**播放端 |

### 一条句子完整打点链路
```
/human 路由 → agent/chat.py::stream_llm_chat()
             用 begin_trace() 开 trace，把 _obs{trace_id} 塞进 datainfo (chat.py:203)
  → put_msg_txt() 打 enqueued_ms 戳 (base_tts.py:69)
  → process_tts() → _run_tts_observed()  → 发 tts_call
  → TTSPool 逐候选 _emit_candidate()     → 发 tts_candidate （池模式才有）
  → 播放端 notify()                      → 发 tts_playback
  → obs/query.py::summary() 聚合 → web/obs.html 面板展示
```

### `emit_explicit` 跨线程挂 trace
TTS 合成跑在**独立 worker 线程**里，而 tracer 是线程绑定的（trace_id 在线程里才有效）。所以用 `obs/__init__.py::emit_explicit`，**显式带上** `trace_id/session_id/parent_id`（从 `textevent._obs` 一路带下来），把 TTS 事件挂回当前聊天对话的时间线。

---

## 6. 熔断器（circuit_open）大白话

`circuit_open` = **"熔断器判了这引擎死刑、正把它关小黑屋，所以压根没调用，只是发条诊断告诉你它现在被闸着"**。

### 三态生命周期 (utils/health_store.py)
```
CLOSED ──连续失败≥阈值──▶ OPEN ──冷却时间到──▶ HALF_OPEN ──探活成功──▶ CLOSED
  ▲                          │                      │
  └──────── 探活失败，重新 OPEN ◀─────────────────────┘
```
1. **CLOSED（正常）**：健康，随便调。连续失败达到阈值（默认 2 次）→ `mark_failure` → 打翻成 OPEN。
2. **OPEN（熔断中）**：接下来 `open_duration_sec`（默认 30s）内 `allow_call` 返回 False，**这个候选被跳过**，不试它。正是这时发 `circuit_open` 事件。
3. **HALF_OPEN（探活）**：冷却到后自动进入，**只放一个请求进去试**（`half_open_inflight` 保证同时只有一次探活）。成功 → 回 CLOSED 痊愈；失败 → 重新 OPEN，再冷却一轮。

### 代码位置
- 池子调用前先问熔断器：`executor.py:114` `if not self._health.allow_call(cand.id):`
- 被熔断跳过也发候选诊断事件：`executor.py:117` `fail_reason="circuit_open"`
- 面板把它当"熔断跳过次数"而不当失败：`obs/query.py:203-204` `circuit_skip`

### 设计意图
别把 30 秒反复砸在一个明显坏掉的引擎上。熔断器扛过前几次失败后就拉闸，让池子雨露均沾、直接跳下一个候选，用户体验好。

---

## 7. 相关文件速查表

### 埋点产出（引擎层）
- `tts/base_tts.py` — 统一线程/队列 + `_run_tts_observed` 主埋点 + `tts_ok/tts_fail`
- `tts/executor.py` — `TTSPool` 候选池回退 + 池层 `tts_candidate`
- 各引擎埋点：`tts/edge.py`(重试/截断最全)、`tts/doubao.py`(累加 audio_ms)、`tts/azure.py`、`cosyvoice/fish/indextts2/omnitts/qwentts/sovits/tencent/xtts.py`

### 观测平台（obs 子系统）
- `obs/__init__.py` — 公开 API `emit_explicit` / `begin_trace` / `end_trace`
- `obs/recorder.py` — `Tracer` 类 + `emit_explicit` + `now_ms`
- `obs/writer.py` — JSONL 落盘
- `obs/query.py` — 聚合面板指标
- `obs/routes.py` — REST 端点 `/api/obs/*`
- `obs/config.py` — 开关/目录配置

### 链路追溯 / 展示
- `agent/chat.py`、`server/routes.py` — 开 trace 把 `_obs` 传给 TTS 线程
- `avatars/base_avatar.py::notify()` — 打 `tts_playback`
- `web/obs.html` — 面板前端（`renderStageStats()` 渲染 TTS 指标卡）

### 熔断 / 候选
- `utils/health_store.py` — 三态熔断器（ASR 与 TTS 共用）
- `utils/cand_pool.py` — 候选构建/排序
- `tts/config.yaml` — `routing.candidates`、`CIRCUIT_BREAKER` 配置

### 关联文档/提交
- `docs/OBS-TTS播放观测盲区与修复.md` — 分层观测方案详解
- 提交 `a89aab2`（audio_ms 上报 + 强度/熔断指标）、`ebee290`（观测链修复）、`82a50c5`（重试/强度/熔断/记忆面板）
- 测试：`tests/test_tts_obs_chain.py`（逐条验证观测链）、`tests/test_tts_routing.py`