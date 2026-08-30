# 如何新增一个能力（Capability）

本文件是「可插拔能力框架」的注入指南。目标是：**只新建一个子包 + 导出两项约定，主循环零改动**，能力即可被识别、按配置启停、按会话状态暴露工具、向 system prompt 注入内容。

> 先读 `base.py`（协议）与 `hub.py`（中枢），再读 `interview/`（首个实体能力，是唯一完整参考实现）。

---

## 1. 能力 vs 普通工具

先明确边界，避免把两类东西做混：

| | 普通工具（TOOL_REGISTRY） | 能力（Capability） |
|---|---|---|
| 例子 | `web_search` / `weather` / `list_files` | `interview`（模拟面试） |
| 性质 | 通用、无状态、按配置常驻 | 整域、有状态、多轮、带 persona/业务逻辑 |
| 进入 | 模型自觉调用 | 模型自觉调用 **或** 规则强制拉起（`pre_entry`），随会话状态推进 |
| 工具暴露 | 启用即全暴露 | 按会话**状态**条件注入子集（s07 用到才加载） |
| 是否碰 avatar/TTS/obs | 否 | 否（只返回文本、读写自己的持久化） |

**关键约束：**
- 能力**不接触** `avatar_session` / TTS / `obs`，不玩摄像头。它只「返回文本」+「读写自己的持久化目录」。
- 文件读写**不是能力**，是通用工具（`agent/files.py` 的 `list_files/read_file`）。你的能力要读上传文件，用通用工具的暂存区（`data/uploads/<sessionid>/`），别在能力包里自造一套文件服务。

---

## 2. 四个插口（它们就是主循环对能力的**全部**感知）

主循环（`agent/chat.py`、`agent/tool_loop.py`、`agent/config.py`）只认识这四处（编号沿用代码注释），不认识任何具体能力：

| 插口 | 中枢入口（`hub.py`） | 被谁调用 | 作用 |
|---|---|---|---|
| ① | `capability_config_defaults()` | `config.py` | 把能力的配置默认值并入 `capabilities.<name>` |
| ② | `capability_system_block(sid, cfg)` | `chat.py` | 把能力目录/激活态片段注入 system prompt |
| ③ | `register_capability_tools()` / `session_tools(sid, cfg)` | `tool_loop.py` / `chat.py` | 合并工具声明 + 每轮按状态暴露子集 |
| ④ | `capability_pre_entry(sid, msg, cfg)` | `tool_loop.py` | 关键词命中时由**规则**强制拉起入口工具 |

新增能力=实现 `base.Capability` 的对应方法，其余靠 hub 自动装配。

---

## 3. 能力协议：`base.Capability`

具体能力继承 `Capability`，按需覆盖以下方法（未覆盖的用默认值）：

```python
class Capability:
    name: str = ""        # 唯一 id，如 "interview"（必填）
    priority: int = 0     # pre_entry 同时命中多个能力时 hub 的排序权重（越大越优先）

    def enabled(self, cfg) -> bool            # 读 capabilities.<name>.enabled（一般不用动）
    def tools(self) -> list[dict]             # 声明工具，合入 TOOL_REGISTRY
    def active_tools(self, session_id) -> list[str]  # 本轮暴露的工具名子集（状态制导）
    def system_block(self, session_id, cfg) -> str   # 注入 system prompt 的一整块文本
    def pre_entry(self, message, session_id) -> dict | None  # 确定性入口（可省）
    def on_session_end(self, session_id) -> None  # 会话结束清理（可省）
```

`tools()` 里每条形如 `{name, description, parameters, handler}`，`parameters` 是标准 JSON Schema。hub 注册时会给每条盖上 `config_flag=("cap", <name>)`，用于：启停门控、以及和「全局工具」（`list_enabled_tools`）区分。

**工具命名用 `capname.toolname` 前缀**，与全局工具天然隔离，如 `interview.start`、`interview.answer`。

---

## 4. 子包必须导出的两项约定

每个能力子包（`capabilities/<your_cap>/`）必须导出：

```python
# capabilities/<your_cap>/__init__.py
from capabilities.base import Capability

class MyCapability(Capability):
    name = "mycap"
    priority = 5
    # ... 覆盖 tools / active_tools / system_block / pre_entry ...

CAPABILITY = MyCapability()        # ① 单例能力实例

def config_defaults() -> dict:     # ② 本能力的配置默认值（并入 capabilities.mycap）
    return {
        "enabled": False,          # 默认关，开启需在 agent_config.yaml 显式置 true
        "my_param": "默认值",
    }
```

发现逻辑在 `hub._discover()`（`base` 模块被跳过，<your_cap> 会被扫到并 import）。**任何一个能力加载失败只告警，不影响其它能力与主流程。**

---

## 5. 分步：写一个最小能力 hello（参考 / 冒烟用）

`capabilities/hello/__init__.py`（hello 曾存在作为冒烟用例，下述是复原其骨架）：

```python
from capabilities.base import Capability

class HelloCapability(Capability):
    name = "hello"
    priority = 0

    def tools(self):
        return [{
            "name": "hello.say_hi",
            "description": "用户打招呼时调用，返回一句问候。",
            "parameters": {"type": "object", "properties": {}, "required": []},
            "handler": self._say_hi,           # handler 签名见下
        }]

    async def _say_hi(self, args, cfg, ctx=None) -> str:
        return "你好呀，我是数字人助手 👋"

    # 不覆盖 active_tools → 默认全暴露（hello 无状态）
    # 不覆盖 system_block / pre_entry → 无注入、不接管

CAPABILITY = HelloCapability()
```

**handler 签名（关键）**——工具循环按 `await handler(args, cfg, ctx=ctx)` 调用：

```python
async def handler(args: dict, cfg, ctx=None) -> str:
    # args: 模型按 schema 填的参数；可空
    # cfg:  AgentConfig（读能力参数用 cap_param / 直接取属性）
    # ctx:  可选，含 session_id 等会话上下文
    ses = getattr(ctx, "session_id", None) if ctx else None
    ...
    return "给模型的文本回复"
```

有状态能力要按会话读写状态时，用**每会话一把 `asyncio.Lock`** 串行化（参考 `interview/tools.py` 顶部的 `_lock(sid)`），避免并发把状态写坏。

---

## 6. 加入配置体系

1. **默认值**：在 `config_defaults()` 返回 `{enabled: False, ...}`。`config.py` 会自动并入 `capabilities.<your_cap>`（用户 yaml 优先覆盖）。**子键默认关**——新能力除非确实要默认上线，否则别在默认配置里置 `enabled: true`（hello 冒烟默认关就是这么约定的）。
2. **启停**：`agent/agent_config.yaml` 的 `capabilities:` 节加入：

```yaml
capabilities:
  mycap:
    enabled: true
    my_param: 覆盖值
```

3. **代码里读**：config 提供通用访问器
   - `cfg.cap_enabled("mycap")` → 是否启用
   - `cfg.cap_param("mycap", "my_param", 缺省)` → 读任意子参数
   - 也可以（如 interview）为常用参数定义 property 便捷属性：`def mycap_my_param(self): return self.cap_param("mycap", "my_param")`

---

## 7. 按会话状态条件暴露工具（s07 用到才加载）

若你的能力**有状态**，别一次把全部工具喂给模型——按状态暴露子集：

```python
_STATE_TOOLS_IDLE = ["mycap.start"]                        # idle: 只给入口
_STATE_TOOLS_ACTIVE = ["mycap.answer", "mycap.end", ...]   # 进行中: 只给相关子集

def active_tools(self, session_id: str) -> list[str]:
    st = MyState(cfg, session_id); st.load()
    if st.status == "active":
        return list(self._STATE_TOOLS_ACTIVE)
    return list(self._STATE_TOOLS_IDLE)
```

原则：**正常闲聊时，能力工具一个都不进列表**；进入进行中后只暴露当前状态相关子集。这由 `hub.session_tools` 实现——全局工具照常门控，能力工具只在 `active_tools(sid)` 命中时暴露。

> 注意：`config_flag=("cap",name)` 的能力工具**不会**进入 `tool_loop.list_enabled_tools`（reminder 等后台路径用），它们只随 `session_tools` 走。

---

## 8. 向 system prompt 注入内容 + 确定性入口（可选）

- **注入**：覆盖 `system_block(self, session_id, cfg)`，返回一整块文本（能力目录 + 当前激活态片段）。进行中激活态给模型当「指挥棒」（面试例中让模型续当面试官），结束/超时时片段消失、自然退回普通助手。默认（或未启用/无内容）返回 `""` 即可。

- **确定性进入**：若希望「用户说什么词就真的进你的能力、不赌模型会不会自觉调用」，覆盖 `pre_entry(self, message, session_id)`：命中关键词返回 `{"tool": <name>, "args": {...}}`，hub→tool_loop 会由**规则直接执行**该工具并把输出当本轮回答（跳过模型自觉调用）；未命中返回 `None`。

  注意与 `active_tools` 的一致性：进行时若你的能力靠非入口工具在续（answer/end…），`pre_entry` 要**避免再抢占**（参考 `interview.__init__` 里 `status=="asking"` 时返回 `None`），否则会暴露与状态矛盾的子集。

---

## 9. 自动继承 / 你不需要做的事

- **对话/记忆/TTS/观测都自动继承**：能力轮仍走主循环同一个 `stream_llm_chat`。会话记忆、长期记忆、TTS 播报、OBS 观测**零配置生效**，不要在能力里重复实现。
- **评分/报告等需要「读会话」的**：读自己能力的持久化状态（如 `data/capabilities/<name>/<sid>.json`），不要改读会话转录。
- 新增文件读写要加在**通用工具**（`agent/files.py`）、别塞进能力包。

---

## 10. 测试

在 `tests/` 建 `test_capability_<name>.py`，至少覆盖：
- discovery 能看到你的能力（`hub.all_capabilities()` 含 `<name>`）；
- 未启用 → `session_tools` 不含其工具、`capability_system_block` 不含其片段；
- 启用 → 工具按状态暴露正确子集（可参考 `tests/test_capability_hub.py` 的 `test_interview_state_driven_exposure`，用 `store_dir` 指向临时目录避免污染）；
- `config_defaults` 被并入（`capability_config_defaults()` 含默认值）。

跑法：`python -m pytest tests/`。

---

## 11. 自检清单

- [ ] `capabilities/<your_cap>/` 独立子包，导出 `CAPABILITY` + `config_defaults()`？
- [ ] `name` 唯一、工具名用 `<cap>.<tool>` 前缀？
- [ ] `tools()` 里每条都带 `handler`，且 handler 是 `async def handler(args, cfg, ctx=None) -> str`？
- [ ] 有状态就覆盖 `active_tools` 按状态暴露子集，闲聊时能力工具不进列表？
- [ ] 默认 `enabled: false`（除非确要默认上线）？
- [ ] 需要时覆盖 `system_block`（注入激活态）与 `pre_entry`（规则入口），且两者状态一致？
- [ ] 没碰 `avatar_session` / TTS / `obs`；只返回文本 + 读写自己的持久化？
- [ ] `tests/test_capability_<name>.py` 覆盖识别/启停/状态暴露/默认值合并？