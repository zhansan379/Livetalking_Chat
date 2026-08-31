# Livetalking_Chat

基于 [lipku/LiveTalking](https://github.com/lipku/LiveTalking) 的实时数字人语音对话系统。浏览器端通过 WebRTC 接入，支持语音识别（ASR）、大模型对话（LLM）、语音合成（TTS）与数字人口型生成，实现与数字人的自然语音交互。

- 前端：`web/avatar-chat.html`（全屏语音对话，内置浏览器端 VAD / ASR / WebRTC）
- 后端：基于 `aiohttp` + `aiortc`，Python 3.12
- 授权：Apache-2.0（继承上游 LiveTalking）

---

<img width="2549" height="1191" alt="image" src="https://github.com/user-attachments/assets/0d07cea1-8769-4bed-a92a-a65255e89260" />


## ✨ 功能特性

- **实时数字人对话**：浏览器 WebRTC 推拉流，数字人口型/动作实时合成
- **多模态输入**：语音（浏览器端 VAD + ASR）/ 文本（底栏输入框回车走 LLM 对话）
- **语音打断（插话）**：数字人说话时可开口打断（barge-in）
- **静默解锁 + 首次点击遮罩**：加载即静音渲染画面，点击遮罩后开声（规避浏览器自动播放策略）
- **逐句字幕**：订阅 `/sse` 流，随数字人朗读逐句显示，底部可「字幕 开/关」
- **多轮对话记忆**：`agent/` 包持久化完整转录，超阈值后后台压缩为有界摘要，回复不阻塞
- **可插拔模型**：支持 `wav2lip` / `musetalk` / `ultralight` 数字人模型
- **多 TTS 引擎**：edge-tts / gpt-sovits / cosyvoice / fishtts / tencent / doubao / indextts2 / azuretts / qwentts
- **本地 ASR 端点**：可选集成 FunASR/SenseVoice（`/api/asr`），亦支持外部 FunASR 服务并带热词
- **摄像头「看用户」**：数字人可经 agent 工具按需抓一帧摄像头画面交给视觉模型，描述用户当下状态（见 [`docs/摄像头看用户工具.md`](docs/摄像头看用户工具.md)）
- **MCP 外部工具接入**：通过 `agent/mcp.py` 连接多台 MCP 服务器（stdio / sse / http），把其工具以 `mcp_<server>_<tool>` 注入对话工具表，复用生态工具（filesystem / sqlite / git 等）（见 [`docs/how-to-mcp.md`](docs/how-to-mcp.md)）
- **管理后台**：`web/admin.html` 查看全局配置与活跃会话
- **多种推流方式**：WebRTC / RTMP / RTCPush / 虚拟摄像头（virtualcam）

---

## 🚀 快速开始（Windows）

### 1. 创建虚拟环境

```bat
uv venv --python 3.12 .venv
.venv\Scripts\activate.bat
```

或使用 `conda`：

```bat
conda create -n livetalking python=3.12
conda activate livetalking
```

### 2. 安装依赖

```bat
pip install -r requirements.txt
```

> 可选本地 ASR：`requirements.txt` 已含 `funasr`、`modelscope`，若不需要可自行删除该行。

### 3. 准备模型文件

默认使用 `wav2lip` + 中文音色：

| 文件 | 位置 | 说明 |
| ---- | ---- | ---- |
| `wav2lip256.pth` | `models/wav2lip.pth` | 嘴型同步模型，下载后重命名 |
| `wav2lip256_avatar1.tar.gz` | `data/avatars/` | 数字人形象，解压到 `data/avatars/`（CLI 默认 `avatar_id`） |
| `rem.tar.gz` | `data/avatars/rem/` | 中文音色形象，`run.bat` 默认使用（需 `rem/full_imgs`） |

### 4. 配置

- 编辑 `config.yaml`（默认值，CLI 参数可覆盖）或改 `.env`（TTS/LLM 密钥）
- `.env.example` 提供了所需的密钥占位：`TENCENT_*`、`DASHSCOPE_API_KEY`、`DOUBAO_API_KEY`

### 5. 启动

```bat
run.bat
```

`run.bat` 等价于（默认模型 `wav2lip`、形象 `rem`，端口 8010）：

```bat
python app.py --transport webrtc --model wav2lip --avatar_id rem
```

浏览器打开：

```
http://<serverip>:8010/
```

> 使用 WebRTC 时，浏览器需能访问 STUN 服务器（默认 `stun:stun.freeswitch.org:3478`，可在 `config.yaml` 修改或页面内勾选 Use STUN server）。

---

## 🎭 支持的 avatar 模型

| 模型 | `--model` | 说明 |
| ---- | --- | ---- |
| wav2lip | `wav2lip` | 嘴型同步，通用性强（默认） |
| musetalk | `musetalk` | 更真实，依赖官方模型 |
| ultralight | `ultralight` | 轻量化模型 |

对应的数字人插件位于 `avatars/`，通过 `@register` 机制加载。

---

## 🌊 推流 / 传输方式

`config.yaml` → `transport` 字段，或 CLI `--transport`：

| 值 | 说明 |
| --- | --- |
| `webrtc` | 浏览器实时对话（默认，推荐） |
| `rtmp` | 推流到 RTMP 服务器，需 `push_url` |
| `rtcpush` | RTCPush 推流，需 `push_url` |
| `virtualcam` | 渲染到虚拟摄像头，会话 0 |

---

## 🔌 HTTP API

| 方法与路径 | 说明 |
| ---- | ---- |
| `POST /offer` | WebRTC SDP Offer / Answer |
| `POST /human` | 文本输入（`type: echo`/`chat`），支持 `voice`/`emotion` |
| `POST /humanaudio` | 上传音频驱动数字人 |
| `POST /interrupt_talk` | 打断当前说话 |
| `POST /set_audiotype` | 设置自定义动作编排 |
| `POST /is_speaking` | 查询是否正在说话 |
| `POST /record` | 开始/停止录制 |
| `GET /sse` | SSE 事件流（服务器状态推送） |
| `GET /record/{sessionid}` | 下载录制 MP4 |
| `GET /api/asr` | 本地 ASR WebSocket 端点（需安装 funasr） |
| `GET /api/admin/config` | 管理后台：全局配置 |
| `GET /api/admin/sessions` | 管理后台：活跃会话 |

---

## 🗂️ 目录结构

```
app.py                       # 服务入口，路由注册、CORS、会话管理
config.py                    # CLI 参数解析
config.yaml                  # 配置文件（默认值）
agent/                       # 多轮对话记忆（历史 JSON 持久化 + 有界摘要压缩）
  ├─ agent.py                # ChatAgent：组装上下文、后台异步压缩
  ├─ history.py              # 会话历史持久化
  └─ agent_config.yaml       # 记忆/压缩阈值配置（system_prompt 等）
infra_ai/                    # LLM 基础设施（熔断/路由/限流/流式/嵌入/重排），取代原 llm.py
  ├─ config.yaml             # LLM 路由与模型配置（支持 ${ENV} 占位符）
  └─ inference.py            # 推理核心 + 工具调用
registry.py                  # avatar 插件注册
avatars/                     # 数字人模型插件（musetalk/wav2lip/ultralight）
server/                      # 后端核心
  ├─ routes.py               # HTTP/SSE 通用 API + chat 流式问答（接线 agent + infra_ai）
  ├─ webrtc.py               # WebRTC HumanPlayer
  ├─ rtc_manager.py          # RTC 连接管理
  ├─ session_manager.py      # 会话管理
  ├─ avatar_routes.py        # avatar 生成路由
  ├─ task_manager.py         # 后台任务管理
web/                         # 前端
  ├─ avatar-chat.html        # 全屏语音对话页（默认首页）
  ├─ admin.html              # 管理后台
  └─ lib/                    # 本地化依赖（jquery/onnxruntime/vad/bootstrap）
data/                        # 数字人形象、录制、自定义动作
models/                      # 模型权重
asr/                         # 本地 ASR（可扩展多引擎 + 候选池熔断回退）
tts/ streamout/ utils/       # TTS、输出、工具模块
```

---

## 💬 LLM 对话与多轮记忆

文本/语音对话统一经 `infra_ai` 调用大模型：

- **默认对话模型**：`infra_ai/config.yaml` 中 `routing.chat` 默认启用 bailian（`qwen-plus`，走 `DASHSCOPE_API_KEY`）；可改 `SF_CHAT_MODEL` / `SF_API_KEY` 切换 SiliconFlow 通道
- **多轮记忆**：`agent/agent_config.yaml` 控制 `system_prompt` 与压缩阈值（`compress_threshold` / `keep_recent` / `target_summary_chars`），历史持久化到 JSON，回复期间后台异步压缩不阻塞
- 键位占位（`.env`）：

| 变量 | 用途 |
| ---- | ---- |
| `DASHSCOPE_API_KEY` | 百炼通话（默认对话模型）+ CosyVoice 等 |
| `SF_API_KEY` | SiliconFlow 文本/视觉/嵌入通道 |
| `DOUBAO_API_KEY` / `TENCENT_*` | 豆包 / 腾讯 TTS |
| 未配置对应密钥 | TTS 退化为 edge-tts（免费在线合成）；对话需至少一种 LLM 通道 |

---

## 📄 License

[Apache-2.0](LICENSE) · Copyright (C) 2024 LiveTalking@lipku (https://github.com/lipku/LiveTalking)
