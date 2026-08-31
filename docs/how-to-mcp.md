# 让对话 Agent 接入 MCP 外部工具

LiveTalking 的数字人 Agent 通过 **MCP（Model Context Protocol）** 连接外部 MCP 服务器，
把服务器暴露的工具以 `mcp_<server>_<tool>` 注入对话工具表，复用生态工具（filesystem / sqlite / git 等），
能力边界不再局限于内置的 web_search / weather / files 等。

实现位于 `agent/mcp.py`（通用工具，非能力域），走既有 TOOL_REGISTRY 链路接入，主循环零改动。

## 原理

- 每台配置的服务器在**服务启动时**建连（stdio / sse / http 三种传输是 mcp SDK 统一的
  「异步上下文管理器 → 解包成 (read, write) 流」协议）。
- `session.list_tools()` 拿到服务器工具列表，逐条向 `tool_loop.TOOL_REGISTRY` 注册一把工具，
  命名 `mcp_<server>_<tool>`，`config_flag` 统一为 `tool_mcp_enabled`。
- 之后 `session_tools → build_tools → run_tool_loop` 的既有链路全自动接管：
  模型像调内置工具一样调 MCP 工具，handler 里往对应 `ClientSession` 发 `tools/call`，
  观测/耗时统计走 run_tool_loop 的既有埋点，免费生效。
- 关闭：服务退出时 `close_mcp_servers` cancel 常驻连接、清掉 `mcp_` 前缀工具。

> 依赖：`pip install mcp`（官方 SDK，`requirements.txt` 已含）。
> `mcp` 包未装 / 服务器配置/建连失败 → 只告警跳过，绝不影响服务启动与正常对话。

## 配置

编辑 `agent/agent_config.yaml` → `tools.mcp`：

```yaml
tools:
  mcp:
    enabled: false          # 主开关；false 时零开销不连任何服务器
    connect_timeout: 15     # 每台服务器建连+注册的超时上限（秒）
    servers:
      my_fs:                # 名字仅作工具名前缀（mcp_my_fs_<tool>），同一个列表内唯一
        transport: stdio
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "C:/data"]
        env: {}
      remote_http:
        transport: http     # sse | http | stdio
        url: "https://example.com/mcp"
        headers: {}         # http/sse 可带请求头
```

- 每台服务器一个条目；键名 `my_fs` 会成为工具名里的一段。
- `transport ∈ stdio | sse | http`。stdio 走本地子进程（离线可用），sse/http 走远程（需网络可达）。
- 默认所有服务器启用；给某台写 `enabled: false` 可单独停用。
- 改配置后重启服务生效。改 `enabled: true` 后，启动日志会出现：
  `MCP server 'my_fs' connected: N tools registered (transport=stdio)`。

## 使用示例

数字人当前没有 MCP 工具，自然对话时模型不会主动调它；只有当模型判断「问题适合用那把工具」
才会调用，结果回填后复述给你。stdio 起一台 filesystem 服务器后，你可以问：
「我电脑上 C:/data 里有哪些文件」→ 模型会调 `mcp_my_fs_list_directory` 等工具作答。

## 校验与调试

1. 装依赖：`pip install mcp` 后 `python -c "import mcp"` 无报错。
2. 语法：`python -m py_compile agent/mcp.py`。
3. 启动 `app.py`，观察 `livetalking.log`：
   - 每台上线服务器应有 `MCP server '...' connected: N tools registered`。
   - 调用了 MCP 工具时会有一行 `tool_call ... tool=mcp_<server>_<tool>`。
4. 降级验证：`tools.mcp.enabled` 保持 false（或卸载 mcp 包）时启动无异常、正常闲聊。

## 常见问题

- **连接超时**：`connect_timeout` 太小 / stdio 子进程启动慢（如 npx 首次拉包）。
  调大 `connect_timeout`，或确认 npx/node 在 PATH 里。
- **工具没出现**：`tools.mcp.enabled` 没开；或服务器没连上（看启动日志告警）。
- **远程连不上**：开发机出网受限时，sse/http 服务器的域名可能不可达，优先用本地 stdio 服务器。