# Livetalking_Chat

一个**会和你说活的数字人**：电脑屏幕上有一个虚拟形象，你开口对它说话，它就能听懂、想好该说啥、再用嘴巴一张一合地对你说回来。

这套系统来自开源的 [lipku/LiveTalking](https://github.com/lipku/LiveTalking)，我们在它的基础上加了很多实用功能（说话能随时被打断、看得见你在干嘛、能连各种 AI 对话和工具箱等，见下方「它能做什么」）。

- 浏览器打开就能用，靠网页标准通话技术（WebRTC）推流，配有全屏对话页面
- 后端起作用的是 Python 3.12
- 授权：Apache-2.0（沿袭上游 LiveTalking，同款开源协议）

---

## 🖼️ 界面长这样



https://github.com/user-attachments/assets/36e5465a-6636-48ef-8d5a-8a11c7ace7df


---

## 它能做什么（大白话版）

- **会听会答**：你按着说话，它认出你在讲什么（语音识别），交给大模型想好怎么回（对话），再用能发出声音的引擎说出来（语音合成），同时嘴巴跟着动。
  > ① 识别  ② 想怎么答  ③ 发出声音  ④ 动嘴 —— 四步连接成一次对话。
- **说一半能打岔**：它正说着话，你直接开口就能打断它，它停下来听你说。
- **想打文字也行**：不想开口时，页面底部有输入框，敲回车就走的是一样的对话。
- **一句话一个字地显示字幕**：它说到哪句，字幕跟到哪句，可随时开关。
- **记得住聊过啥**：它不会"失忆"，历史对话会存下来，太长时会悄悄压缩成摘要，聊很久也能接上话。
- **能看到你在干嘛**（可选）：它需要时可以抓一帧你摄像头画面，交给看得懂图的大模型，描述你当下的状态（比如"你在笑"）。
- **能接外面的工具箱**（可选）：通过 MCP 机制，可以把文件、数据库、git 等外部能力接进来，让数字人能实际调用它们。
- **换形象、换音色都随你**：同一套系统可换好几款数字人（wav2lip / musetalk / ultralight），语音也有多种引擎可选（见下方「更多设置」）。
- **管理台**：`web/admin.html` 一个页面能看全局配置和在线会话。

---

## 🧰 现在有哪些工具 / 能力（tool 一览）

下面这张表是把当前系统里真真切切挂载着的能力都列了一遍。分三类：**随时能用的通用工具**、**只在面试场景出现的面试能力**、**后台自动跑的**（不用你管）。

### 一、通用工具（说得着就用）

| 代码里叫这个名字 | 大白话：能干嘛 | 谁在用 / 开关 |
| ---- | ---- | ---- |
| `web_search` | 上网帮你查资料、找答案 | 全局，`tool_web_search_enabled` |
| `weather` | 查某个城市的实时天气（热不热、下不下雨） | 全局，`tool_weather_enabled` |
| `schedule_reminder` | 定个闹钟提醒你：几秒后／今天几点／每天固定点（喝水、播天气…）都行 | 全局，`tool_reminder_enabled` |
| `cancel_reminder` | 把之前设的提醒取消掉 | 同上 |
| `list_reminders` | 看看现在设了哪些提醒 | 同上 |
| `question_search` | 从本地面试题库里"按意思"搜相近的题，用来复习 / 找考点 | 全局，`tool_question_search_enabled` |
| `look_at_user` | 「看你一眼」：抓一帧你摄像头画面，让看得懂图的说你现在长啥样／啥表情 | 全局，`tool_look_at_user_enabled` |
| `list_files` / `read_file` | 浏览、打开会话里传的文件 | 全局，`tool_files_enabled` |
| `shutdown_pc` | 你明确说关机时，它帮你关这台电脑 | 全局，`tool_shutdown_pc_enabled` |
| `play_music` | 放歌、放背景音乐 | 全局，`tool_play_music_enabled` |
| `mcp_*` | 再接进来的"外部工具箱"，比如文件、数据库、git，能真正被它调用 | 见 `docs/how-to-mcp.md` |

### 二、面试能力（你一说要面试才上场）

| 代码里叫这个名字 | 大白话：能干嘛 |
| ---- | ---- |
| `interview.start` / `.answer` / `.next_section` / `.skip` / `.hint` / `.end` / `.status` | 当你说"我想面试"，它就当**面试官**带你走完整套流程：出题、听你答、判分、给提示、跳题、换环节、结束总结 |

### 接下来的开发计划

- [ ] 提供角色扮演功能，嘿嘿嘿，谁不想真正使用动漫人物作为自己的女朋友呢

- [ ] 寻找低成本的语音克隆服务，穷逼qwq专用

- [ ] ai主动和你唠嗑，改变一问一答的形式

- [ ] 更完整的测试，稳定的服务

- [ ] 与claude code或codex的集成，别人都有的功能，我就没必要写了吧qwq

- [ ] 自进化，参照hermes自己构建skill或者提高强化学习路径

- [ ] 添加抖音直播/哔哩哔哩直播等接入场景


### 三、后台自动跑的

| 名字 | 大白话：能干嘛 |
| ---- | ---- |
| 长期记忆（`longterm`） | 你说完话，它闲下来时偷偷把重点存下来，下次聊还能想起来（不用你管，自己生效） |

> 想要哪个工具，去 `agent/agent_config.yaml` 里看对应开关；想自己加新工具，打开说明见 `capabilities/how-to-add-capability.md`（其中「工具」和「能力」的区别也写在里面）。

---

## ⚙️ 我们相比原版做了什么优化

上面的功能是"多了什么"，这里讲的是"底子更稳了"。整套系统在原版 `lipku/LiveTalking` 基础上，重点在**说话 / 听话 / 打断 / 记忆**这几条链路上做了大量防抖加固，挑重点说：

### 🔊 说话（TTS）下功夫最深
- **多引擎自动降级**：挂了豆包 / 腾讯 / edge 等好几个语音引擎，主引擎出错会**自动换下一个**；哪个引擎**连续出错会自动熔断**（暂时停用它），不再死磕同一家，一句话里念失败也能句子内回退重来。
- **断流自愈**：语音通道断了会自动重连，检出"只播了一半"就自动补上，不吞异常、不假装成功。
- **不把"被你打断"当故障**：你说"别说了"只是打断、不计入失败；要是误听成打断，会自动把**那句没说完的补播出来**。

### 👂 听话（ASR）也能自己换引擎
- 听写做成**多引擎候选池 + 熔断回退**：首选本地 Paraformer，出问题自动退回 SenseVoice，再不行走云端，谁先稳定用谁。

### 💬 打断更"懂人话"
- 只有**确认真发出声**才打断播报，不会因一点噪音误触、误清掉还没播完的话；对话代际、TTS 代次、立即落盘，多端一致不跳戏。

### 🧠 记得住，也记得对（长期记忆加固）
- 加了**归纳/冷却护栏**，不会什么都记走、也不会记着记着把整库重写成一句；修掉了"把系统想法张冠李戴到你头上"的归因污染和反复提取的毛病。
- 面试复盘**只沉淀你还欠缺的**，不记整套问答；出题会**回读历史薄弱点**，专挑上次露怯的方向考。

### 📊 自己会"看病"（全链路观测）
- 从你说话→它听懂→想答案→发出声音，**整条链路耗时**在观测面板一眼可见，哪一步慢当场揪出；每个引擎试了几次、熔断到哪家都有数据说话，后台维护动作也不会污染统计。

### 🎙️ 唤醒词不依赖外网
- 关键词唤醒（"小爱同学"那种）改用**本地模型**，离线也能叫醒；换词**热更**、断线 **1.5 秒自愈重连**，改词不用重启。

### 🧩 其他的顺手加固
- 接入 **MCP 外部工具**（stdio / sse / http 都能接）；能力做成**插件式框架**，新能力即插即用
- 上传的**中文文件名**不再被压成乱码 `_`
- 定时关机支持**查询/取消/按绝对时间点**，破坏性动作默认关
- 音乐可在线播放、可下载到本地离线听，默认不用循环吵你

---

## 🚀 快速跑起来（Windows）

下面几步，照着抄就行。

### 1. 先准备个干净的运行环境

用 `uv`（推荐）：

```bat
uv venv --python 3.12 .venv
.venv\Scripts\activate.bat
```

或者用 `conda`：

```bat
conda create -n livetalking python=3.12
conda activate livetalking
```

### 2. 安装要用到的软件包

```bat
pip install -r requirements.txt
```

> 可选项：如果不需要本地"听写"（语音识别）功能，`requirements.txt` 里那行 `funasr`、`modelscope` 可以自己删掉。
> —— 要不要本地听写，看你想不想一开机就能离线听懂英文；不装也能跑。

### 3. 准备数字人需要的「模型」和「形象」

第一次跑要准备两样东西：**会动嘴的模型**（`models/`）和**形象素材**（`data/`）。两套办法：

**办法 A（省事，推荐）：从 Release 一键拉全**
`models/` 和 `data/` 加起来约 1GB，太大，不适合塞进 GitHub 仓库里，所以打包成一个大压缩包放在 GitHub Releases 上。运行：

```bat
python scripts/download_models.py
```

它会自动把最新的数据包下载下来，并解压到 `models/` 和 `data/` 该有的位置，全程不用你手动挑文件。（`scripts/package_release.py` 是给维护者打包用的，一般用不到。）

**办法 B（自己手动逐项下载）：**
默认是 `wav2lip` 模型 + 中文音色形象：

| 文件 | 放到哪里 | 是干嘛的 |
| ---- | ---- | ---- |
| `wav2lip256.pth` | `models/wav2lip.pth` | 让嘴巴跟着声音动的模型，下好后重命名成这个名 |
| `wav2lip256_avatar1.tar.gz` | 解压到 `data/avatars/` | 数字人的长相 |
| `rem.tar.gz` | 解压到 `data/avatars/rem/` | 一套中文音色的形象，`run.bat` 默认用这套 |

### 4. 填两处配置

- **配置多少、用啥默认**：改 `config.yaml`
- **语音和 AI 对话的密钥**：改 `.env`（`.env.example` 里有每种密钥该填在哪的占位，照着填就行）
- 要点填的：`TENCENT_*`、`DASHSCOPE_API_KEY`、`DOUBAO_API_KEY`（分别对应腾讯语音、百炼对话、豆包语音的账号钥匙）

### 5. 启动

```bat
run.bat
```

它等价于下面这条命令（默认 `wav2lip`、形象 `rem`、端口 8010）：

```bat
python app.py --transport webrtc --model wav2lip --avatar_id rem
```

### 6. 浏览器打开

```
http://<你的服务器IP>:8010/
```

> 用的是网页实时通话（WebRTC）时，浏览器要能连得上中转服务器（默认 `stun:stun.freeswitch.org:3478`）。网络受限环境可在 `config.yaml` 改，或页面上勾上「Use STUN server」。

---

## 🖼️ 更多功能截图

<img width="2549" height="1191" alt="image" src="https://github.com/user-attachments/assets/c355c3fd-5018-47eb-be5b-a7cba08ca885" />

<img width="2549" height="1191" alt="image" src="https://github.com/user-attachments/assets/cc8cdcfd-a893-4917-bf43-a850e310ad2e" />

<img width="2549" height="1191" alt="image" src="https://github.com/user-attachments/assets/266dec2e-bd3c-4e0e-8493-78e91ed9ca3a" />

---

## 🎭 我能用几种形象 / 音色

### 数字人类别（决定"像不像真人"）

| 类别 | 命令行里叫 | 大白话 |
| ---- | --- | ---- |
| wav2lip | `wav2lip` | 最常用、最稳，给声音配上嘴形（默认） |
| musetalk | `musetalk` | 画面更接近真人，但要额外下官方模型 |
| ultralight | `ultralight` | 更轻量，老一点/配置差一点的机器也能跑 |

### 语音（声音）引擎

edge-tts(免费用) / gpt-sovits / cosyvoice / fishtts / tencent 腾讯 / doubao 豆包 / indextts2 / azuretts 微软 / qwentts 通义 —— 想要哪种声音，在 `config.yaml` 或启动参数 `--tts` 里指定。

---

## 🌊 想让画面去哪个"屏幕"（推流方式）

| 值 | 大白话 |
| --- | --- |
| `webrtc` | 直接在浏览器页面上看（默认，最省事） |
| `rtmp` | 推到直播服务器，别人可看 |
| `rtcpush` | 推到指定地址直播，需要 `push_url` |
| `virtualcam` | 把数字人当成一个"虚拟摄像头"，让别的软件（如钉钉/腾讯会议）从它这儿取画面 |

---

## 💬 想接自己的 AI 对话？在这里改

承接"听懂→想话→说话"这一环的 AI 对话，走 `infra_ai` 这套：

- **默认对话模型**：`infra_ai/config.yaml` 里的 `routing.chat` 默认用百炼（`qwen-plus`，靠 `DASHSCOPE_API_KEY`）；想换通道，改 `SF_CHAT_MODEL` / `SF_API_KEY`
- **聊过啥记多久**：`agent/agent_config.yaml` 里有 `compress_threshold`、`keep_recent`、`target_summary_chars` 这些阈值，管多少条历史触发压缩、留最近几条；回复期间后台悄悄压缩，不卡你
- **key 速查**（`.env`）：

| 变量 | 干啥用 |
| ---- | ---- |
| `DASHSCOPE_API_KEY` | 百炼对话（默认对话模型）+ 通义声音等 |
| `SF_API_KEY` | 硅基流动：文字 / 看图 / 向量这几类通道 |
| `DOUBAO_API_KEY` / `TENCENT_*` | 豆包 / 腾讯语音 |
| 没填对应密钥 | 声音退回到免费的 edge-tts；对话通道至少要有一个才说得起来 |

---

## 🛠️ 给开发者看的技术速查（看不懂可跳过）

### HTTP 接口一览

| 方法与路径 | 干嘛的 |
| ---- | ---- |
| `POST /offer` | 建立网页实时通话的手续 |
| `POST /human` | 让数字人应答一句话（`type: echo`/`chat`） |
| `POST /humanaudio` | 上传一段音频驱动数字人 |
| `POST /interrupt_talk` | 打断它当前说的话 |
| `POST /set_audiotype` | 配自定义动作编排 |
| `POST /is_speaking` | 问它"你现在在路上话吗" |
| `POST /record` | 开始 / 停止录像 |
| `GET /sse` | 服务端实时状态推送（字幕用它） |
| `GET /record/{sessionid}` | 下载录好的视频 |
| `GET /api/asr` | 本地听写端点（需装 funasr） |
| `GET /api/admin/config` | 管理台：全局配置 |
| `GET /api/admin/sessions` | 管理台：在线会话 |

### 目录速览

```
app.py                 # 入口：注册路由、登录跨域、管会话
config.py              # 读命令行参数的解析器
config.yaml            # 配置文件（默认值）
agent/                 # 多轮对话记忆：历史存 JSON + 太长时压缩成摘要
infra_ai/              # AI 对话的"基础设施"：切换/限流/流式/看图/向量等
registry.py            # 数字人插件的登记表
avatars/               # 数字人类别插件（musetalk / wav2lip / ultralight）
server/                # 后端核心
web/                   # 前端：对话页(avatar-chat.html)、管理台(admin.html)、本地依赖
data/                  # 形象素材、录制、自定义动作
models/                # 模型权重
asr/                   # 本地听写
tts/ streamout/ utils/ # 语音、输出、工具
```

---

## 📄 协议

[Apache-2.0](LICENSE) · Copyright (C) 2024 LiveTalking@lipku (https://github.com/lipku/LiveTalking)
