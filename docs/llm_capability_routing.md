# LLM 按能力路由（capability routing）约定

> 本文件说明 `infra_ai` 里「按环节/能力选模型」的机制，特别是**新增功能不指定能力时默认走哪条降级链**。改动配置后重启服务生效。

## 一句话结论

**新增功能调用 `async_call_llm(...)` 时若不传 `capability`，默认值是 `"chat"`，会走与主对话相同的回退链：**

```
bailian-max (qwen3.8-max) → bailian-flash (qwen3.8-flash) → bailian-plus (qwen3.7-plus)
```

也就是说：**不指定 = 安心复用主对话档位（max 级）**，行为最保守、不会更贵。想让某个环节用**更省/更合适的档位**时，才显式传 `capability` 去走它自己的链。

## 机制：两条选模型路径

调用文本 LLM 的统一入口是 `async_call_llm`（非流式）/ `async_call_llm_with_tools`（带工具）/ `async_stream_call_llm`（流式）。它们都接受 `capability: str = "chat"` 与 `first_choice_id` 两参数：

| 路径 | 触发条件 | 行为 |
|---|---|---|
| **能力路由**（推荐） | 只传 `capability`（或都用默认） | 按 `LLM_ROUTING.<capability>` 的 `default_model` 首选 + 候选池按 priority 排序 + 熔断健康滚动，**带故障转移 + 每候选独立 api_key/base_url** |
| **原始单模型覆盖** | 传了 `model_name="..."` | 绕开路由，直接用该模型名建客户端。**优先级最高**——一旦传入即覆盖能力路由 |

执行叠加的机制：
- **熔断转移**：首选连续失败达到 `circuit_breaker.failure_threshold`（默认 2）后熔断冷却 `open_duration_sec`（默认 30s），期间跳过该候选，换下一个。
- **单模型回退**：仅当 `router` 模块异常缺失（`ImportError`）才落到 `_create_llm_from_routing("chat", ...)`，取 `chat` 候选第一个 = `bailian-max`。

## 现有能力与默认档位

候选池都用三家 bailian 模型（flash / max / plus）作故障转移，`default_model` 决定首选，其余按 priority 依次回退。

| 能力 | 环节 | 首选档位 | 降级链 |
|---|---|---|---|
| `chat` | 主对话（流式 + 工具循环） | bailian-max | max → flash → plus |
| `chat_tone` | 语调调整（`agent/chat.py:_probe_tone`） | bailian-flash | flash → max → plus |
| `compress` | 会话摘要压缩（`agent/agent.py:compress_and_save`） | bailian-plus | plus → max → flash |
| `extract` | 长期记忆提取（`agent/longterm.py:_call_extract`） | bailian-plus | plus → max → flash |
| `consolidate` | 长期记忆整理（`agent/longterm.py:_call_consolidate`） | bailian-plus | plus → max → flash |
| `vision` | 视觉理解（摄像头看用户等） | bailian-vision | （独立池） |

接线位置（本次改动已挂好）：

- `agent/chat.py:_probe_tone` → `capability="chat_tone"`
- `agent/agent.py:compress_and_save` → `capability="compress"`
- `agent/longterm.py:_call_extract` → `capability="extract"`
- `agent/longterm.py:_call_consolidate` → `capability="consolidate"`

## 如何新增一个「自己的档位」的环节

1. **在 `infra_ai/config.yaml` 的 `llm.routing` 下加一个能力块**（`router.py` 已改为遍历全部能力键自动装载，加配置即生效）：

```yaml
    <your_capability>:
      default_model: bailian-flash          # 首选档位，值 = 下方某个候选 id
      candidates:
        - id: bailian-flash
          provider: bailian
          model: ${BAILIAN_CHAT_MODEL_FLASH:-qwen3.8-flash}
          base_url: ${BAILIAN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
          api_key: ${DASHSCOPE_API_KEY}
          priority: 1
          enabled: true
        - id: bailian-max
          provider: bailian
          model: ${BAILIAN_CHAT_MODEL_MAX:-qwen3.8-max}
          base_url: ${BAILIAN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
          api_key: ${DASHSCOPE_API_KEY}
          priority: 2
          enabled: true
        - id: bailian-plus
          provider: bailian
          model: ${BAILIAN_CHAT_MODEL_PLUS:-qwen3.7-plus}
          base_url: ${BAILIAN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
          api_key: ${DASHSCOPE_API_KEY}
          priority: 3
          enabled: true
```

2. **在调用点传能力名**：

```python
await async_call_llm(messages, use_json=True,
                     extra={"kind": "my_stage"},
                     capability="<your_capability>")
```

3. （可选）若想保留「配置里可显式覆盖成特定模型」的能力，增设一个 `model_name` 字段（如 `summarize_model` / `extract_model`），在调用时代入 `model_name=`——它会优先于能力路由。

## 决策建议

- **新环节默认（不传 capability）**：走主对话 max 链，安全、不用想。适合需要高质量答案的环节。
- **廉价/高频辅助环节**（语调探测、提取、压缩、整理这类 JSON 小调用）：别让它们烧 max，显式给更便宜的 `capability`。
- **不传 `model_name` 且想有故障转移**：用能力路由；只有确认锁死某单个模型时才传 `model_name`。

## 验证

```bash
# 各能力首选顺序是否正确
python -c "
from infra_ai.core.router import get_router
sel = get_router().selector
for cap in ('chat','chat_tone','compress','extract','consolidate'):
    print(cap, '->', [t.model_id for t in sel.select(cap)])
"
```

运行期观测：`obs` 面板 / `logs/llm_errors.jsonl` 里每个 `llm_call` 事件带 `model`（实际模型名）和 `purpose`（如 `tone_probe` / `compress` / `longterm_extract` / `longterm_merge`），可据此确认各环节落在预期档位。