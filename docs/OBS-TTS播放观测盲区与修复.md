# TTS「只播一半」盲区：合成 × 播放分层观测

> 现象：回答被切 4 句，第 3 句观众听不到，但 OBS 里 `success_rate = 100%`。
> 根因：`tts_call.success` 只度量「合成」，度量不到「播放」。

---

## 一、问题与诊断

### 检测记录在哪

- **成败分界**：`tts/base_tts.py` 的 `tts_ok()` / `tts_fail()`，由
  `_run_tts_observed()`（base_tts.py:89-137）包住每句 `txt_to_audio`，消费
  `self.last_tts` 后发一条 `tts_call` 事件。
- **doubao 成功标准**：`tts/doubao.py` —— `self._got_audio == True`（在
  `_consume_pcm` 收到 ≥1 个非空音频 payload 时置位）→ `tts_ok()`；否则按
  `barge_in` / `no_audio` / `session_failed` 记失败。
- **本质**：`success=true` 只等于「真实 PCM 字节已灌进 `parent.put_audio_frame`
  → `asr.queue`」，**不等于**「浏览器播出来了」。

### 播放链路（`tts_call` 的盲区）

```
doubao 合成 → put_audio_frame → asr.queue → lip-sync(mel/hubert/whisper)
           → server/webrtc.py push_audio → PlayerStreamTrack._queue
           → recv() 出队实发 → 浏览器
```

`tts_call` 只盖住最左边「字节产出」；其余全部是盲区。

### 两个具体遗漏

1. **`audio_ms` 恒为 0**：`tts/doubao.py` 调 `tts_ok()` 不传时长 →
   `base_tts.py` 默认 `audio_ms=0`。「每句合成了几毫秒音频」完全记不到，
   「合成时长 vs 文本应有时长」校验缺失。
2. **端标记送达无观测**：doubao `_send_end` 把 `{status:"end", text}` 的零帧
   入队；当 `server/webrtc.py` recv() 把该帧出队 → `base_avatar.notify()` →
   发浏览器 SSE。这个「端标记是否真的送出服务器」点此前完全没进 OBS。

> 已知丢弃机：`flush_talk` 的 `queue.clear()`（avatars/audio_features/base_asr.py）
> 在新回合/打断时丢未消费尾部。但本例下一个 ASR 在 5s 后、非打断，故更偏
> 下游（浏览器不消费后续回包 / WebRTC 抖动缓冲）。

### 一句话归因

「成功率 100% 但只听到一半」= 指标只度量了**合成字节产出**，没度量
**端→浏览器送达**。

---

## 二、改进方案：三层分层观测

| 层 | 度量 | 位置 | 能解释本次吗 |
|---|---|---|---|
| 层1 | 合成音频毫秒 `audio_ms` | 后端 doubao | 否（本次已全合成） |
| 层2 | 端标记送达率 `end_serve_rate` | 后端 | **是（核心）** |
| 层3 | 浏览器实播确认 `played_rate` | 前端 | 是（权威，最重） |

本次落地 **层1 + 层2**（均后端，一处一发）。层3 视需要再上：若
`end_serve_rate≈100%` 仍掉句，才需浏览器实播确认（涉网络/抖动缓冲）。

---

## 三、实现

### 层 1 — 合成时长归位（`tts/doubao.py`）

- `_consume_pcm` 累加实际送入播放的样本 → `self._audio_ms`
  （`consumed_samples / sample_rate * 1000`，不足一 chunk 的残余不计）。
- 每句 `txt_to_audio` 开头清零。
- 成功分支改调 `tts_ok(audio_ms=self._audio_ms)`。

### 层 2 — 端标记送达观测（`avatars/base_avatar.py` notify）

对 `status=="end"` 的零帧被 WebRTC 出队的时刻，补一条 obs 事件：

```python
emit_explicit({
    "type": "tts_playback", "text": ..., "text_len": ..., "status": "end",
}, trace_id=_obs.trace_id, session_id=_obs.session_id,
   parent_id=_obs.parent_id, kind="chat")
```

- 与 `_run_tts_observed` 同款跨线程显式 ID，打回对应 trace。
- `_obs` 本就在 eventpoint 里（`_send_end` 蹭来的 `datainfo`），
  `base_avatar.notify` 写给浏览器的 SSE 才剥离 `_` 前缀，obs 用原始 eventpoint。

### 聚合 + 前端

`obs/query.py` 新增聚合（`elif t == "tts_playback"`）：

```jsonc
"tts": {
  "calls": 4,
  "success_rate": 1.0,        // 合成成功 = 100%（老指标）
  "audio_ms": 8000.0,         // 合成音频总毫秒（层1）
  "avg_audio_ms": 2000.0,     // 平均每句合成毫秒（层1）
  "ends_served": 3,           // 端标记送达次数（层2）
  "end_serve_rate": 0.75,     // 端送达率 = ends_served / calls（层2，核心）
  "retry_count": 0,
  "truncation_count": 0
}
```

`web/obs.html` `renderStageStats` 新增两卡片：

- **TTS 合成时长**：`avg_audio_ms`，副标题 `audio_ms` 总时长。
- **TTS 端送达率**：`end_serve_rate`，副标题 `ends_served/calls 句送抵`；
  **<100% 时橙色报警**（`color = end_serve_rate>=1 ? C_OK : C_ORG`）。

---

## 四、怎么读 OBS

对本次「播一半」：

- `success_rate=100%` + `end_serve_rate<100%` → 有句合成了但没送抵浏览器前
  → 丢失在 WebRTC 之前（后端队列 / 打断丢队列）。
- `end_serve_rate=100%` 仍掉句 → 丢失在 WebRTC 之后（浏览器/网络抖动缓冲）
  → 需层3前端实播确认。

---

## 五、验证

- **单测**（`tests/test_obs.py`，通过 12/12）：
  - `test_tts_call_emit_explicit_nests_and_counts`：断言 `audio_ms` / `avg_audio_ms`。
  - `test_tts_end_serve_rate_partial`：2 句合成、1 句送抵 →
    `success_rate=1.0` 但 `end_serve_rate=0.5`，正是本次盲区的判别样本。
  - `test_tts_playback_is_chat_subevent`：`tts_playback` 不新增聊天 trace。
  - `test_doubao_audio_ms_accumulated`：`_consume_pcm` 源码级时长累加正确。
- **端到端**：起服务发一句跨 ≥2 标点分句的长回答，查 summary：
  每句 `audio_ms` 非 0、无打断时 `ends_served == calls`；手动 `flush_talk`
  （打断）应见 `end_serve_rate` 反映缺句。
- **回归**：`python -m pytest tests/ -q`（note：
  `test_capability_hub.py` 的 3 个失败是 `hello` 能力缺失，与本次无关）。

---

## 六、相关改动文件

- `tts/doubao.py` — 层1 audio_ms
- `avatars/base_avatar.py` — 层2 tts_playback 事件
- `obs/query.py` — 聚合字段
- `web/obs.html` — 前端卡片
- `tests/test_obs.py` — 用例