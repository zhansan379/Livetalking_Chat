# persona 能力设计 · 人物角色扮演域(脚本蒸馏 + 选人扮演)

> **状态**:设计稿(尚未实现)。参考 `.claude/skills/create-ex`(把「前任」蒸馏成 AI Skill),为 LiveTalking 落地**通用人物**角色扮演能力。
>
> 定位收敛:蒸馏(创建)不是对话式多轮交互,而是**后端离线脚本**一次执行;运行时只在**配置或对话里选择用谁**来扮演。

---

## 一、目标与取舍

| 项 | 决定 | 说明 |
|---|---|---|
| 能力定位 | **通用人物角色扮演** | 前任只是其中一种素材。有状态、多轮、自带 persona 的「域」,符合 `capabilities/how-to-add-capability.md` 对能力的定义,贴合并语音数字人场景 |
| 蒸馏方式 | **后端离线脚本** `scripts/build_persona.py` | 一次跑完 → 内部调 LLM 分析素材 → 生成人物文件。没有 creating 状态机,不占用对话 |
| 选人方式 | **配置(`active_persona`)或对话(`persona.activate`)** | 会话只在 idle ↔ active 两态切换 |
| 扮演方式 | 激活后注入 persona+memories+扮演规则,system_block 驱动 | 模型全程按该人口吻;进化(`persona.remember/correct`)是轻量单步工具 |
| 解析脚本 | **移植 create-ex 的 wechat/photo/sms/social** 解析器 | 纯标准库,供蒸馏脚本进程内调用 |
| 不移植 | create-ex 的 iMessage(读 Mac chat.db,本机 Windows 不可用)、`skill_writer.py`/`version_manager.py`(职责由 store.py 承担) | — |

**关键适配**:create-ex 是文本对话(emoji/颜文字/表情包),LiveTalking 是语音(系统 prompt 已禁 emoji/markdown)。扮演输出须**保持口吻但转纯文本口语**(VOICE_RULES)。

---

## 二、交付物总览

```
scripts/build_persona.py        # 蒸馏 CLI:素材→解析→call_llm→写人物库(对齐 build_interview_index.py)
capabilities/persona/
  __init__.py                   # PersonaCapability(name="persona") + config_defaults()
                                #   active_tools / system_block / pre_entry;主循环零改动
  store.py                      # 全局人物库 CRUD + 版本化 + slug(脚本与运行时共用)
  state.py                      # 每会话状态(极简):{active_slug, status: idle|active}
  prompts.py                    # 扮演注入块 + VOICE_RULES + 蒸馏 author prompt(脚本用)
  tools.py                      # persona.activate/switch/end/status/list/remember/correct
  parsers/                      # 移植解析器(纯标准库,进程内 import)
    wechat.py  photo.py  sms.py  social.py
tests/
  test_capability_persona.py
```

> 严格执行既定设计决策:只新建 `capabilities/persona/` 子包 + 导出 `CAPABILITY`/`config_defaults()`,**主循环零改动**;插口①②③④由 `capabilities/base.py`/`hub.py` 自动装配。

---

## 三、蒸馏脚本 `scripts/build_persona.py`(后端直接执行)

```
python scripts/build_persona.py --name "小美" \
    [--info "在一起两年半 大学同学"] [--personality "ENFP 焦虑型 爱撒娇"] \
    [--chat chat.txt]        # 微信/短信/社交导出 → wechat/sms/social 解析器
    [--photos ~/Photos/her]  # 照片目录 → photo 解析器,提取时间线
    [--text material.txt]    # 任意粘贴/文本素材
    [--output <dir>]         # 缺省由 name 生成 slug 落在 data/capabilities/persona/personas/<slug>
```

流程:
1. **解析**:按参数把素材喂给 `parsers`,复用 create-ex 的抽取逻辑(长消息/情感消息/日常风格分类、照片 EXIF 时间线),产出结构化的「人物待分析素材」。
2. **蒸馏**:调用 `infra_ai.core.sync.call_llm(messages, use_json=True)`(已确认同步入口,独立脚本可用;承载 create-ex `persona_analyzer`+`persona_builder` 的 Layer0-5 维度记忆线),让模型一次产出结构化 `{persona_md, memories_md, meta:{profile, tags, impression}}`。
3. **落盘**:`store.save_persona(...)` 写 `personas/<slug>/{persona.md, memories.md, corrections.md, meta.json}`,打印路径与触发方式(「`persona.activate <slug>` 或说『让小美陪我』」)。
4. `store.slugify(name)` 生成 slug:优先可选 `pypinyin`;未装则回退 ASCII/时间戳(无硬依赖)。

脚本在项目根运行,必要时 `sys.path.insert` 根目录(参照 `scripts/build_interview_index.py`)。

---

## 四、运行时能力 `persona`(选人扮演域)

### 1. 全局人物库 `store.py`(跨会话,脚本与运行时共用)
数据目录 `data/capabilities/persona/personas/<slug>/`:

| 文件 | 内容 |
|---|---|
| `persona.md` | 人物的性格画像(Layer0 核心性格 / 表达风格 / 情感逻辑 / 关系行为 / 边界雷区) |
| `memories.md` | 与该人物的共同记忆、日常、时间线 |
| `corrections.md` | 纠正/进化记录 |
| `meta.json` | name/slug/created_at/updated_at/version/profile/tags/impression/corrections_count |

API:`list_personas / get_persona(slug) / load_text(slug, "persona"|"memories") / save_persona(meta) / append_memory(slug, text) / apply_correction(slug, text) / rollback(slug, ver) / delete_persona(slug)`(仅删该 slug)。写入前把旧文件备份到 `versions/<ts>/`(保留 `version_limit`,默认 20)。

### 2. 每会话状态 `state.py`(极简)
`data/capabilities/persona/sessions/<sid>.json`,字段 `{active_slug, active_since, status: idle|active}`;原子写(`tempfile+os.replace`)+ 每会话 `asyncio.Lock`(照 `interview/state.py`)。配置 `active_persona` 时,会话开始时由 `state.ensure()` 自动装上。

### 3. 工具集(`persona.*`,按状态暴露)
| 状态 | 暴露工具 |
|---|---|
| idle | `persona.activate`, `persona.list`, `persona.status` |
| active | `persona.switch`, `persona.remember`, `persona.correct`, `persona.end`, `persona.status`, `persona.list` |

| 工具 | 作用 |
|---|---|
| `persona.activate {slug|name}` | 从人物库选中并激活(按 slug 或名称模糊匹配) |
| `persona.switch {slug|name}` | 切换陪伴对象 |
| `persona.end` | 解除激活,退回被动助手 |
| `persona.status` | 当前激活哪位 / (常驻)有哪些人物 |
| `persona.list` | 列出全部人物(名称 + 一句印象) |
| `persona.remember` | 追加一条新记忆到当前人物 memories.md(+版本) |
| `persona.correct` | 追加纠正到 corrections.md,并可传新 persona_text 改写 persona.md(+版本;模型单步完成,无多轮 intake) |

handler 统一 `async def handler(args, cfg, ctx=None) -> str`(per how-to §5)。蒸馏已在脚本侧完成,运行时不再碰上传区。

### 4. system_block / pre_entry
- **`active_tools(sid)`**:如上;闲聊(未激活)时陪伴工具一个都不进列表。
- **`system_block(sid, cfg)`**:
  - active → 注入 `[扮演协议 + persona.md + memories.md + VOICE_RULES]`。**VOICE_RULES 关键**:保持该人口吻/称呼/口头禅/短句连发节奏,**但输出仍为纯文本口语、禁 emoji/markdown**(语音适配);有 `roleplay_char_limit`(默认 6000)兜底防撑爆。
  - 否则 → 一行目录提示(「可切换角色:…」)。
- **`pre_entry(msg, sid)`**:非 active 时,命中「扮演/让X陪我/切换成X/和X聊」选人意图 → 返回 `persona.activate`(用已有人物名匹配 slug);active 时返回 None(避免与状态矛盾,照 interview 约定)。**不拦截**创建意图(蒸馏已移出对话)。

### 5. 配置(`config_defaults()` → agent_config.yaml `capabilities.persona`)
```python
def config_defaults():
    return {
        "enabled": False,          # 默认关,显式开启
        "store_dir": None,         # null → data/capabilities/persona
        "active_persona": None,    # 可选默认唤醒人物 slug(会话开始时自动激活)
        "voice_output": True,      # 扮演输出语音适配(禁 emoji/markdown)
        "roleplay_char_limit": 6000,
        "version_limit": 20,
    }
```
config 访问器走通用 `cap_enabled("persona")` / `cap_param("persona", ...)`,不加硬编码 property。`agent_config.yaml` 的 `capabilities:` 下新增 `persona:` 节(带注释,默认 enabled 注释置 true)。

---

## 五、依赖
- 解析器:纯标准库(wechat 用 re/csv/html.parser;photo 用 struct/re/datetime)。
- 蒸馏:复用现成 `infra_ai.core.sync.call_llm`,无新增模型接入。
- `pypinyin`:可选(未装回退),不进 requirements 硬依赖。

---

## 六、测试(`tests/test_capability_persona.py`)
遵循 `test_capability_hub.py` / `test_interview_state.py`,用临时 `store_dir` 避免污染仓库 data/:
- discovery 发现 `persona`;未启用 → `session_tools` 无 `persona.*`、`system_block` 无片段;
- enabled → idle/active 工具子集暴露正确;
- `store` CRUD + 版本 rollback 往返;slug 清洗(非法字符);
- `pre_entry`:选人意图命中 → `persona.activate`;active 不再抢占;
- 解析器冒烟:wechat 解析样例 txt(含目标与分类)、photo 对临时 JPEG 目录出时间线;
- `config_defaults` 并入。

跑法:`python -m pytest tests/test_capability_persona.py -q`(无需新增依赖)。

---

## 七、端到端验证(实现后)
1. 跑一次 `python scripts/build_persona.py --name 小美 --chat x.txt [--photos dir]` → `data/capabilities/persona/personas/xiaomei/` 生成 persona/memories/meta,并打印触发方式。
2. `agent_config.yaml` 开 `persona.enabled: true`(可设 `active_persona: xiaomei`),重启。
3. 配置自动激活,或对话说「让小美陪我」→ pre_entry 拉起 activate → 此后按小美口吻(语音适配无 emoji)陪伴。说新记忆 → `persona.remember`;说「她不会这样」→ `persona.correct`;「退出角色扮演」→ `persona.end` 回到助手。
4. 多人物:`persona.list` / `persona.switch 另一个`。

---

## 参考复用
- 原子持久化 / 会话锁:`capabilities/interview/state.py`
- 工具 schema + handler、状态暴露:`capabilities/interview/tools.py`、`__init__.py`
- 提示词注入结构:`capabilities/interview/prompts.py`
- 插口①-④:`capabilities/base.py` / `hub.py`(主循环零改动)
- 离线构建脚本范式:`scripts/build_interview_index.py`
- 蒸馏语义来源:`.claude/skills/create-ex/prompts/{persona_analyzer, persona_builder, memories_builder, correction_handler}.md`