# 关键词唤醒（KWS）与双词语音开关 —— 实现与开发记录

> 本文档记录「数字人像 Hey Siri 那样，用说关键词来开启/关闭语音对话（各带语音播报）」的完整实现。基于 2026-08/09 实际开发与联调，供后续再上手时直接续接。

## 1. 目标与现状

让 LiveTalking 支持**无手开启**语音：

- **开启词**（默认 `小智同学`）：说它即进入语音对话，并播报「好的，已开启语音识别，请说话」。
- **关闭词**（默认 `小智再见`）：对话中说它即回到「待命」态，并播报「好的，已关闭语音识别，随时喊我」。
- 两个词各自持久化，唤醒**开关状态**也持久化：刷新/重开页面后自动恢复到上次开/闭。

**关键选型**：真 KWS（开放式关键词唤醒引擎，sherpa-onnx），跑在**本地 Python 后端**；不是浏览器自编 WASM，也不需要联网云密钥。浏览器只负责在有需要时把 16k PCM 帧送往后端。

**当前状态**：功能完整落地；准确率侧已切 fp32 + boost=2.0，实机命中/无误报验证通过。

## 2. 端到端数据流

```
麦克风 → MicVAD(浏览器, Silero) → onFrameProcessed 每帧
        └─(唤醒功能开启 & KWS WS 已连) 50ms 一批 → WS /api/kws/ws?session=…
              └─→ 后端 sherpa KWS(持久流) ──命中──→ {type:'wake', keyword}
                                                     ├─ keyword==开启词 → onKwsWake  → 进入对话 + 播报「已开启」
                                                     └─ keyword==关闭词 → onKwsSleep → 回到待命 + 播报「已关闭」
```

对话中 KWS **保持常开**（不因进入对话而关闭），目的是还能听到「关闭词」。两条路并存：
- **KWS 路**：只匹配两个关键词（开启词/关闭词），命中即切状态。
- **ASR 路**：进入对话后，VAD 的概率 `onSpeechEnd` 将普通语音切片 → `/api/asr` → `sendChat` → LLM/TTS。开启/关闭词本句用 `_wokeAt` / `_sleptAt` 丢窗丢弃，不发给 LLM。

**语音播报** = 前端 `sendText(短语)` → `POST /human {type:'echo', interrupt:true}` → `put_msg_txt` → TTS 池合成 → WebRTC 出声，全程**不经 LLM**。文案常量 `WAKE_ANNOUNCE` / `SLEEP_ANNOUNCE`（前端 globals 区）。

## 3. 关键实现（按文件）

### 3.1 后端 `server/kws.py`（新增）

- `KwsService` 单例：`_init()` 定位模型目录（`models/kws/<model_id>/`，可 `KWS_MODEL_DIR` 覆盖），延迟导入 `sherpa_onnx`，缺模型/缺依赖则 `available=False` —— 前端自动降级为普通通话，不破坏原功能。
- `_first(prefix)` **优先取非 int8（fp32）** 精度版本算准确率：`...int8.onnx` 的 `i` 字母序在 `...onnx` 的 `o` 前，若直接取 `sorted()` 第一个会误用 int8（量化后分值贴阈值抖动 → 「忽高忽低」）。
- 调参常量：`_KWS_SCORE=2.0`（关键词 boosting，越大越易命中，提升召回）、`_KWS_THRESHOLD=0.25`（触发阈值，越大越难触发）、`_WIN_COOLDOWN_S=3.0`（命中冷却防连触发）。
- `create_stream(stream_keywords)`：**连接级持久流**，跨 50ms 小帧增量解码（不是每 chunk 新建流，否则 4 字词跨帧识别不了）。
- `feed_stream(stream, pcm16)`：`accept_waveform→decode_stream`，`is_ready` 循环取 `get_result` 返回命中标签。
- `kws_websocket_handler`：协议 = 首帧 `{keywords:[开启词,关闭词]}` → `{type:'ready'}`；BINARY PCM16 → feed；命中 → `{type:'wake', keyword}`（keyword 是 `@` 后的标签，前端据此分流）。`available=False` → `{type:'error', reason:'kws_unavailable'}` 并断开。

### 3.2 G2P 分词（关键！）—— `server/kws.py::tokenize`

该 wenetspeech KWS 模型建模单元是**拼音（声母+韵母）**，不是单字。`tokens.txt` 是带调韵母 + 声母的 BPE 词表，中文关键词必须转成拼音 token 串：

```
小智同学 → x iǎo zh ì t óng x ué
小智再见 → x iǎo zh ì z ài j iàn
```

算法（已对照模型自带 `keywords.txt`/`test_wavs/test_keywords.txt` 共 16 条 ground-truth 全量校验通过）：
1. `pypinyin(Style.TONE)` 出带调音节（如 `zhì`）。
2. `_split_initial` 切出声母（`zh/ch/sh` 等最长优先）→ 剩余为带调韵母。
3. 韵母在 `tokens.txt` 词表内直接匹配；不在则 `_greedy_final` 按词表最长贪心切分。
4. 无法覆盖的字返回空串，不影响其它在词表内的字触发。`pypinyin` 缺失时退化为按字切（尽力而为）。

> 依赖 `pypinyin`（已加入 `requirements.txt`）。缺它只是退化，不影响降级路径。

### 3.3 前端 `web/avatar-chat.html`

- **设置**：`#wakeword`（开启词）、`#sleepword`（关闭词），二者经 localStorage（`kws.wakeword`/`kws.sleepword`）持久化；`#sleepword` 新建。
- **唤醒开关记忆**：`_persistWakeOn()`/`_wakeOnSaved()`（`kws.wakeon`）。所有改变 `wakeOn` 处都保存：`startWake` 开→存 `1`；`stopCall`/`startCall`(切换手动)/`onKwsError`/`startWake`失败→存 `0`。页面 ready 时若上次开 → 自动 `startWake()`（麦克风未授权会被浏览器拒 → 走失败分支存 0，不反复弹权限）。
- **双词状态机**：
  - `openKwsWS` 一次发 `{keywords:[开启词,关闭词]}`。
  - `onKwsWake(kw)`：`kw===sleepKeyword` → `onKwsSleep()`；否则开启词：`wakeArmed=false; callOn=true`，`startPolling`，播报「已开启」；**不再 `closeKwsWS()`**（KWS 常开）。
  - `onKwsSleep()`：`callOn=false; wakeArmed=true; _sleptAt=now`，清队列、停轮询，播报「已关闭」，回到待命可无手再唤醒。
  - `onFrameProcessed` 喂帧条件由 `wakeOn && wakeArmed` 放宽为 `wakeOn`（对话中也送帧给 KWS）。
  - `onSpeechEnd` 加 `_sleptAt` 丢窗，避免关闭词发成普通语音给 LLM。
- **手动「开启通话」**不变：`startCall` 会先关掉唤醒（`wakeOn=false`），走原 ASR 管线。

## 4. 模型获取与配置

官方**不把 KWS 模型放 HF 仓库、也不在 ModelScope**，而是打成单文件 tar.bz2 发布在 **GitHub releases（tag=`kws-models`）**：

```
https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/
    sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2   (~32.6MB)
```

`python scripts/fetch_kws_model.py` 的主路径即从该 tar 下载并解压到 `models/kws/<model_id>/`（含 encoder/decoder/joiner 的 fp32+int8、tokens.txt、关键词样例、test_wavs）。**本机 GitHub 可达，不需要任何 token。**

- Git 忽略：`models/`、`scripts/` 整体被 gitignore。`fetch_kws_model.py` 用 `git add -f` 强制入库（其它脚本 `build_interview_index.py` 等同理），模型文件本身不入库。
- 依赖：`sherpa-onnx`、`pypinyin`（清华 PyPI 可装）。
- 后端启动加载模型成功 → 日志 `[KWS] 就绪`；缺模型 → `available=False`，前端降级普通通话。

## 5. 验证与测试

- 模型 token 化对照：`tokenize("小智再见")` → `x iǎo zh ì z ài j iàn`，与官方一致。
- 真实推理（服务级 `feed_stream`，test_wavs）：周望军音频 → 命中「周望军」标签；小智同学/小智再见在其它 wav 无误报。
- 双词场景：同一流同时挂两个词，周望军音频只回 `周望军`（可据此分流），不回关闭词。
- 端到端：浏览器点「唤醒」→ 喊开启词 → 进对话 + 播报；喊关闭词 → 回待命 + 播报；刷新页面恢复上次开关状态。

## 6. 边界与调参

- **准确率**：优先 fp32 + boost=2.0；仍漏报/不稳 → `_KWS_SCORE` 升或 `_KWS_THRESHOLD` 降（0.25→0.15~0.2）；误唤醒 → 反向。（都在 `server/kws.py` 顶部常量。）
- **隐私权衡**：唤醒功能开着=麦克风常驻监听（Hey Siri 同款）。数据只在待命/对话时发往本机后端 KWS。想「只在待命听/对话中不听」需另做（当前默认常驻监听）。
- **浏览器手势**：首次未授权麦克风的浏览器，自动恢复监听可能被拒，需用户点一次「唤醒」授权。
- **模型不在全部源可达时**：不硬编码依赖，前端自动降级，装好模型重启后启用。已在网控机遇墙时用 `fetch_kws_model.py` 取一次后**完全离线可用**。

## 7. 涉及文件

| 文件 | 作用 |
|---|---|
| `server/kws.py`（新增） | KWS 服务 + G2P 分词 + WS handler |
| `server/routes.py` | 注册 `/api/kws/ws`（try/except 容错） |
| `web/avatar-chat.html` | 双词状态机 + 播报 + 开关记忆 |
| `scripts/fetch_kws_model.py`（新增，`git add -f`） | GitHub tar 下载工具 |
| `requirements.txt` | `sherpa-onnx`、`pypinyin` |