# 模拟面试题库：chromadb 向量化加载与召回

> 本文记录 interview 能力题库从「内置 YAML」迁移到「chromadb 向量库」的完整方案，含数据源、存储、召回、分类口径、脚本用法与已知取舍。

## 1. 背景与目标

原题库是能力包内一个内置 YAML（`capabilities/interview/bank_data.yaml`，几十道题），经「确定性过滤 → 远程 rerank → LLM 个性化」三段式召回，只适合小体量。

现在换用两份真实数据 CSV（`data/` 下），GB 级真实题库，迁移到轻量级向量库 **chromadb** 承载：

- **删 YAML 机制**：`bank_data.yaml` 已删除，`load_bank`/`filter_bank`/`bank_override` 全部移除。
- **本地向量化**：chromadb 内置本地 onnx 模型（all-MiniLM-L6-v2），离线、免费，不需要付费 embedding API。
- **撤销远程 rerank**：chromadb 相似度排序即为最终候选池，直接喂 LLM 个性化。
- **独立建索引脚本**：离线全量灌库，运行期只查询。

## 2. 数据源（两份 CSV）

| 文件 | 行数 | 用途 | 关键字段 |
|---|---|---|---|
| `data/question_bank.csv` | 107 | 题库目录（分类来源） | `question_bank_id`、`question_bank_name`(题库名)、`relative_position`(旧宽分类) |
| `data/question_essay.csv` | 5158 | 题目 | `question_id`、`bank_id`(关联目录)、`title`(题干)、`answer`(参考答案)、`keyword`、`is_deleted` |

> 注意 `wc -l` 显示 49309 行是因为多行 `answer` 被折行，实际题目记录 **5158** 条。CSV 首列带 BOM，读取用 `utf-8-sig` + 列名规整（见 `bank.py:_reader`）。

## 3. 分类口径（关键决策）

- **`category`** = **题库名**（`question_bank_name`），经 `_clean_bank_name()` 清洗成技能主体（98 类，0 空）。入库 metadata 与向量文档都用它。
- **`channel`** = 旧的 `relative_position`，因**不全且无意义**已弃用检索，仅作为归档字段存档保留。
- 二者都来自同一题库目录 CSV，按 `bank_id` 关联；同一题库的题共享同一分类。

`_clean_bank_name()` 清洗规则：
```
Redis面试题 _ 小林coding _ Java面试学习   -> Redis        # 剥 _来源注释_ 段，取首段主体
MySQL 面试题及答案整理，最新面试题          -> MySQL        # 按长到短循环剥固定尾缀
JDK 17 新特性实战，答案整理，最新面试题     -> JDK 17 新特性
美团Java社招面试题真题，最新面试题           -> 美团Java社招面试题真题   # 保留公司+真题语义
interview_questions                      -> interview questions  # 纯 ASCII 仅下划线换空格
```

## 4. 数据流与存储

```
question_bank.csv ──┐
                    ├─ join on bank_id → 归一化记录{id,text,category,channel,keywords,answer}
question_essay.csv ─┘
        │ build_index（scripts/build_interview_index.py）
        ▼
chromadb PersistentClient（data/capabilities/interview/chroma/，本地 onnx 向量化，cosine）
        ▼ search（recall.py 开场时）
        ◄ _recall_query(role/level/resume/JD)
top_k 记录 → _personalize LLM → 面试题单{id,text,category,type,rubrics,followups}
```

- 每个 question 的 metadata：`{id, text, category, channel, keywords, answer(截断2000字)}`
- 向量文档（embedding 的输入）= `title | category | keywords | answer[:500]`
- `search` 空库时**自动兜底建一次**，避免空手上阵。

## 5. 召回路（recall.py）

开场 `tools.py:_start` → `recall.build_question_sheet(cfg, role, level, resume, jd)`：

1. **向量检索**（`bank.search`）：用 `_recall_query`（含方向/难度/简历/岗位要求）对 chromadb top-k 检索。
2. **LLM 个性化**（`_personalize`）：只对 top-k 做一次 LLM——排序、按岗位润色、结合参考答案定 rubrics、针对简历补追问；产出终局题单。

任何环节失败逐级降级，永不返回空表。题库整本永不进 prompt。

**题型契约不变**：终局题单结构 `{id,text,category,type,rubrics,followups}` 与旧实现一致；`tools.py` / `state.py` / `eval.py` / `prompts.py` 均未改动。

## 6. 配置接线

新增/替换的 `capabilities: interview:` 配置键（`agent/agent_config.yaml` + `config_defaults()` + `agent/config.py` getter）：

| 键 | 默认 | 说明 |
|---|---|---|
| `bank_csv` | `null`(→data/question_bank.csv) | 题库目录 CSV |
| `essay_csv` | `null`(→data/question_essay.csv) | 题目 CSV |
| `index_dir` | `data/capabilities/interview/chroma` | chromadb 落盘目录 |
| `recall_top_k` | 8 | 检索返回条数 |
| `max_questions` | 5 | 题单题数上限 |

（`bank_override` 已移除。）

## 7. 脚本

### 建索引（离线、幂等）
```bash
python scripts/build_interview_index.py                     # 默认读 data/ 下两张 CSV
python scripts/build_interview_index.py --bank-csv ... --essay-csv ... --out ...
```
每次全量重建（先删旧集合）。首次需联网下载一次向量模型（见 §9）。

### 巡检 / 演示（纯本地、零网依赖）
```bash
python scripts/inspect_interview_bank.py stats             # 题量 + 按分类分布
python scripts/inspect_interview_bank.py sample --n 8      # 抽题预览（含参考答案片段）
python scripts/inspect_interview_bank.py query "Java 并发" --k 8   # 向量检索 + 相似度距离
python scripts/inspect_interview_bank.py probe 15870       # 按 id 看原文与入库文档
python scripts/inspect_interview_bank.py --o 文件.txt       # 输出落盘 UTF-8
```

## 8. 依赖

`requirements.txt` 追加：

```
chromadb                      # 自带 onnxruntime / tokenizers，供本地 onnx 描述函数
websockets>=14                # 原==12.0；chromadb 需新 websockets，豆包 TTS 用 deprecated 别名仍可用
```

## 9. 已知取舍 / 备忘

1. **all-MiniLM 偏英文语义**，中文召回会弱于远程 rerank 精排（抽查「前端」曾混入 Go/嵌入式题）。若需更强中文召回，可换一套中文本地 onnx 模型（BGE 系）作 `embedding_function` 重灌即可，架构不变。
   - **本地模型来源坑**：chromadb 默认从 S3 下载 all-MiniLM，在本机被墙。解法：从 **ModelScope**（`Xenova/all-MiniLM-L6-v2`，本是阿里源可达）下载 6 个文件到 `~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx/`，chromadb 见文件存在即离线用。
2. **分类偏细碎**：`category` 是题库名（98 类，容量不均）。若嫌太碎，可一行切回 `channel` 那 28 个宽分类参与检索。
3. **`.venv` 的 websockets 已升到 17.1**：功能上豆包 TTS 正常（`WebSocketClientProtocol` 仅 deprecated 未移除）；requirements 已放宽 `>=14` 防拉回 12.0 与 chromadb 冲突。
4. 本机出网受限：仅清华 pypi 镜像 + ModelScope 可达；S3 / hf-mirror / siliconflow 被墙。详见记忆 `dev-machine-network-reachability`。

## 10. 验证

- 建索引：`python scripts/build_interview_index.py` → 打印「共入库 5158 题」；确认 `data/capabilities/interview/chroma/` 生成。
- 检索冒烟：`inspect query "线程池 并发 原理"` 返回相关题。
- 单测：`python -m pytest tests/test_interview_state.py` → 7 passed（题单已被 mock，不受本题库改动影响）。