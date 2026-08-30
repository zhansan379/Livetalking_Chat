# 模拟面试·环节可配（interview sections）

模拟面试能力把整场面试从「一份平铺题单」升级为**有序多环节**，支持
自我介绍 / 项目问答 / 八股文 / 反问 等结构，且**环节序列可通过配置整体覆盖**。
改动全部收敛在 `capabilities/interview/` 内部，主流程（chat / tool_loop / hub）零改动。

- 关联：`capabilities/interview/`、`agent/config.py`、`agent/agent_config.yaml`
- 上一版（平铺题单 + chromadb 题库）：见 `docs/interview_question_bank_chromadb.md`

---

## 1. 环节类型

| type          | 环节名   | 形态   | 题目来源                                                  | 评分维度               |
|---------------|---------|--------|----------------------------------------------------------|------------------------|
| `self_intro`  | 自我介绍 | 对话段 | 无题库，靠 persona 引导 + 简历追问                         | 表达 / 结构 / 匹配      |
| `project`     | 项目问答 | 离散段 | 有简历→抽项目逐段深挖；无→题库通用项目题兜底              | 理解 / 表达 / 逻辑 / 完整 |
| `trivia`      | 八股文   | 离散段 | chromadb 向量检索 top-k → LLM 个性化                      | 理解 / 表达 / 逻辑 / 完整 |
| `reverse_qa`  | 反问     | 对话段 | 无题库，角色反转（候选人提问、面试官作答）                | 提问质量               |

**两种形态**：
- **离散段**（project / trivia）：一组固定题目，`interview.answer` 逐条评分快进，`interview.skip` 换题、`interview.hint` 求提示，段内题走完自动进入下一环节。
- **对话段**（self_intro / reverse_qa）：进入后自由多轮交谈；`interview.answer` 只记录 transcript **不评分**；段末由 `interview.next_section` 对整段**一次性评分**并推进。

---

## 2. 配置

### 默认值（`capabilities/interview/__init__.py` → `config_defaults()`）

```python
"sections": [
    {"type": "self_intro", "name": "自我介绍"},
    {"type": "project",    "name": "项目问答", "count": 3},
    {"type": "trivia",     "name": "八股文",   "count": 3},
    {"type": "reverse_qa", "name": "反问"},
]
```

### 覆盖（`agent/agent_config.yaml` → `capabilities.interview.sections`）

```yaml
capabilities:
  interview:
    enabled: true
    # …其它现有项…
    sections:                      # 整体替换默认列表
      - { type: self_intro, name: 开场自我介绍 }
      - { type: trivia,     name: 基础八股,   count: 5 }
      - { type: project,    name: 项目深挖,   count: 2 }
      - { type: reverse_qa, name: 反向提问 }
```

约定：
- `type` ∈ `self_intro / project / trivia / reverse_qa`；未知 type 会被丢弃。
- **对话段不设 `count`**（自由交流，无固定题量）。
- **离散段 `count` 缺省**取 `max_questions`（默认 5）。
- `sections` 是整体替换，不是按 type 合并——配了就整份生效。

---

## 3. 状态机（`state.py`）

取代旧的 `questions[] + idx` 单指针：

```python
{
  "sections":    [ {type, name, count?, dialogue}, ... ],   # 环节描述（有序）
  "section_idx": int,                                        # 当前环节
  "items":       list,                                       # 离散→题目 dict 列表；对话→transcript
  "idx":         int,                                        # 离散段内游标；对话段恒 0
  "answers":     [ {question, answer, eval, section_type, section_name}, ... ],
  "status":      "idle | asking | finished",
  ...
}
```

新增 helper：`current_section() / section_type() / is_dialogue() / section_items() /
inline_idx() / in_last_section()`。`status` 仍 `idle|asking|finished`，「换段」仍是 `asking`，
只是 `section_idx` 前进；对话段结束时若已是最后一段则收敛为 `finished`。
读写/原子落盘/锁不变。`answers` 每条带 `section_type/section_name`，报告据此分组。

**旧态兼容**：旧 `start` 残留且无 `sections` 的状态视为过期，开新场直接重建，不做迁移。

---

## 4. 工具（`tools.py` + `__init__.py`）

| 工具                 | 离散段                 | 对话段                          |
|----------------------|------------------------|--------------------------------|
| `interview.answer`   | 评分 + 快进            | 记录 transcript，不评分          |
| `interview.skip`     | 换题（不出分）          | ✗ 返回「自由交流，直接答」        |
| `interview.hint`     | 求提示                 | ✗ 返回通用引导                   |
| `interview.next_section` | ✗ 返回「答题环节无需手动切换」 | ✓ 整段判分 + 推进             |
| `interview.end`      | 终局报告               | 终局报告                        |
| `interview.status`   | 段+题进度               | 段+交流轮数                      |

**双维工具子集**（`active_tools` 按「状态 + 当前段类型」暴露，模型不会误用）：

```python
_STATE_TOOLS_IDLE = ["interview.start"]
_STATE_TOOLS_DISC = ["interview.answer", "interview.skip", "interview.hint",
                     "interview.end", "interview.status"]
_STATE_TOOLS_DIAL = ["interview.answer", "interview.next_section",
                     "interview.end", "interview.status"]
_STATE_TOOLS_FIN  = ["interview.start", "interview.status"]
```

---

## 5. 各段题目生产（`recall.py`）

统一分发 `build_section(cfg, section, role, level, resume_text, jd_text)`：

- **self_intro / reverse_qa**：返回 `[]`，零 LLM，靠 persona。
- **trivia**：复用 `build_question_sheet(max_q=count)`（银行检索 → `_personalize` → `_plain`）。
- **project**：
  - `_extract_projects(cfg, resume_text, want)`：一次 `async_call_llm(capability="extract", extra={"kind":"interview_resume_extract"}, max_tokens=600)` 抽 `{projects:[{name,summary,difficulty,highlights}]}`；失败/空 → `None`。
  - 有项目 → 每项目一条深挖题（`text`=`背景→难点→方案→结果→复盘`，`brief` 字段存项目摘要供面试官深挖）。
  - 兜底 `_build_project_fallback(cfg, ...)`：题库检索「通用项目式」题，仍空 → 硬编码模板题。**永不返回空表**。

> `capability="extract"` 走 `infra_ai/config.yaml` 里已有的 `extract` 分档（便宜档）；`kind` 仅供观测面板标识，不参与路由。选模型/分档机制详见 `docs/llm_capability_routing.md`。

---

## 6. 评分与报告（`eval.py`）

- **离散题**：`score_answer` 沿用 4 维（理解/表达/逻辑/完整），不变。
- **对话段整段**：`score_section(cfg, section, transcript)` 按环节维度判分一次
  （self_intro→表达/结构/匹配；reverse_qa→提问质量），返回与 score_answer 同 shape。
- **终局**：`build_report(cfg, sections, answers, role, level, jd_text)` 按 `section_type`
  分组拼逐段 transcript 给一次 LLM → `{summary, strengths[], improvements[], suggested_topics[], sections:[{name,score,comment}]}`；失败退确定性汇总。逐段 `score`（离散=段内均值，对话=整段分）。整体 `dimension_avg` 只统计含 4 维作答，避免对话段维度混入 NaN。
- **长期记忆**：`_maybe_remember` 沉淀 `summary + suggested_topics`，兼容不变。

---

## 7. Persona / 提示词（`prompts.py`）

`INTERVIEWER_PROMPT` 为通用面试官人设，各环节叠加 `SECTION_PROMPTS[type]`：

- self_intro：引导 1-2 分钟自介 → 结合简历逐点追问，先 `answer` 记录再回应。
- project：围绕项目逐段深挖 背景→难点→方案→结果→复盘，当前题问深透再进下一题。
- trivia：出题 + 简短点评，平缓过渡下一题。
- reverse_qa：角色反转，候选人提问你作答，默默评估提问质量。

`activation_block` 携带 当前环节名 / 进度 / 当前题（对话段无题时显示交流轮数指令）。

---

## 8. 主要文件

- `capabilities/interview/state.py` —— sections 状态 + 段 helper
- `capabilities/interview/prompts.py` —— SECTION_PROMPTS + activation_block
- `capabilities/interview/recall.py` —— build_section / 项目抽简历 + 兜底
- `capabilities/interview/eval.py` —— score_section / build_report 按段
- `capabilities/interview/tools.py` —— 分段流转 + next_section
- `capabilities/interview/__init__.py` —— config_defaults sections + 双维工具子集 + persona
- `agent/config.py`、`agent/agent_config.yaml` —— `interview_sections` 配置
- `tests/test_interview_state.py` —— 按段模型单测（离散逐题推进 / 对话段整段评分）

## 9. 端到端验证（本机）

开一场「前端·初级模拟面试」预期链路：
自我介绍（自由交谈 →「进入下一环节」整段评分）→ 项目问答（无简历出通用项目题兜底）
→ 八股文（按题快进自动换段）→ 反问（角色反转并评分）→ 终局报告按环节分段输出。
LLM 调用走本机可用路由（bailian max/plus/flash 均 enabled）；断网时走确定性降级，不崩。