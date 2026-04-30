# GenAI2OpenAI

OpenAI / Claude Code 兼容的代理服务，将上海科技大学 GenAI 平台的 API 转换为标准 OpenAI Chat Completion 和 Anthropic Messages 接口。

## 特性

- **OpenAI 兼容接口** — 直接对接任何支持 OpenAI API 的客户端（ChatGPT UI、Cursor、Continue 等）
- **Claude Code 兼容接口** — 提供 Anthropic Messages API (`/v1/messages`)，可通过 `ANTHROPIC_BASE_URL` 直连 Claude Code
- **Tool Calling** — 通过 prompt 注入实现 OpenAI 格式的 function calling，兼容不原生支持的模型
- **Claude Code Tool Use** — 支持 Claude `tool_use/tool_result` 历史转换，并把 GenAI 生成的 `<tool_call>` 转回 Anthropic `tool_use`
- **流式 / 非流式** — 同时支持 SSE 流式和一次性返回两种模式
- **Token 智能识别** — `--token` 同时支持 JWT 字符串和 `学号@密码` 格式
- **自动登录与刷新** — 使用学号密码模式时，启动时自动通过 CAS 登录获取 JWT，过期时静默刷新，对客户端完全透明
- **动态模型列表** — 自动从 GenAI 平台拉取可用模型，无需硬编码
- **API Key 认证** — 可选的客户端认证，保护代理不被未授权访问

## 安装与运行

### 环境要求

- Python 3.11+
- 推荐使用 [uv](https://github.com/astral-sh/uv) 管理环境

### 安装依赖

```bash
uv sync
```

### 启动服务

```bash
# 使用学号密码（推荐，支持自动刷新）
uv run main.py --token "2024000001@mypassword"

# 使用 JWT（需要手动更换过期 token）
uv run main.py --token "eyJ..."

# 完整参数
uv run main.py --token <token> [--port 5000] [--api-key <key>] [--api-format both] [--debug]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--token` | JWT 令牌或 `学号@密码`（必需） | — |
| `--port` | 服务监听端口 | `5000` |
| `--api-key` | 客户端认证密钥（也可通过 `API_KEY` 环境变量设置） | 无（不校验） |
| `--api-format` | `openai`、`anthropic` 或 `both`（也可通过 `API_FORMAT` 设置） | `both` |
| `--debug` | 启用详细日志输出 | 关闭 |

## Token 模式

### 学号密码模式（推荐）

```bash
uv run main.py --token "2024134022@mypassword"
```

- 启动时通过上海科技大学统一身份认证 (CAS) 自动登录
- JWT 过期时自动重新登录获取新 token，对客户端完全透明
- 上游返回 401 时自动尝试刷新 token 并重试请求
- 密码错误时启动即报错退出，便于排查

### JWT 模式

```bash
uv run main.py --token "eyJ..."
```

- 直接使用已有的 JWT 令牌
- 过期后返回 401 错误，需要手动更换

**手动获取 JWT：**

1. 前往 [GenAI 对话平台](https://genai.shanghaitech.edu.cn/)
2. 打开浏览器开发者工具（F12），切到 Network 标签
3. 发送一条消息，捕获 `chat` 请求
4. 在请求头中复制 `x-access-token` 字段的值

## API 接口

### 聊天补全

```
POST /v1/chat/completions
```

支持流式和非流式模式，兼容 OpenAI Chat Completion API 格式。

### Claude Code / Anthropic Messages

```
POST /v1/messages
POST /v1/messages/count_tokens
```

兼容 Claude Code 使用的 Anthropic Messages API。`/messages` 和 `/messages/count_tokens` 也可用，便于兼容不同客户端拼接方式。

### 模型列表

```
GET /v1/models
```

动态返回 GenAI 平台当前可用的所有模型。

### 健康检查

```
GET /health
```

## 使用示例

### 基本对话

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="your-api-key"  # 如果设置了 --api-key
)

response = client.chat.completions.create(
    model="GPT-4.1",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Tool Calling

```python
response = client.chat.completions.create(
    model="GPT-4.1",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }]
)
```

支持 `tool_choice` 参数：`"auto"`（默认）、`"required"`、指定函数名。

### Claude Code

1. 启动代理：

```bash
uv run main.py --token "学号@密码" --port 5000 --api-format both
```

2. 设置环境变量（写入 `~/.bashrc` 或 `~/.zshrc`）：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:5000"
export ANTHROPIC_AUTH_TOKEN="local-proxy"   # 如果设了 --api-key，改成同一个值
export ANTHROPIC_MODEL="chatglm"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="MiniMax-M1"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-pro"
export ANTHROPIC_DEFAULT_OPUS_MODEL="chatglm"
export ANTHROPIC_REASONING_MODEL="MiniMax-M1"
```

3. 配置 Claude Code 项目设置（`.claude/settings.json`）：

在项目根目录创建 `.claude/settings.json`，减少权限弹窗：

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run *)",
      "Bash(python *)",
      "Bash(curl *localhost*)"
    ]
  }
}
```

4. 启动 Claude Code：

```bash
claude
```

代理同时兼容 Claude Code 常用的 `x-api-key` 和 OpenAI 常用的 `Authorization: Bearer`，无需额外配置。

## API Key 认证

设置 `--api-key` 或环境变量 `API_KEY` 后，所有 `/v1/` 请求需要携带 `Authorization: Bearer <key>` 或 `x-api-key: <key>` 请求头。未设置时跳过认证（开发模式）。

## 项目结构

```
GenAI2OpenAI/
├── main.py                 # 入口，参数解析与启动
├── app.py                  # Flask 应用工厂
├── config.py               # 配置、模型注册表
├── errors.py               # OpenAI 格式错误响应
├── auth/
│   ├── apikey.py            # API Key 中间件
│   ├── cas_login.py         # CAS 统一身份认证登录
│   └── token_manager.py     # Token 智能管理（识别、校验、刷新）
├── api/
│   ├── chat.py              # /v1/chat/completions
│   ├── messages.py          # /v1/messages
│   └── models.py            # /v1/models
├── provider/
│   ├── anthropic.py         # Claude Code / Anthropic Messages 转换
│   └── genai.py             # GenAI 上游请求与流式转换
└── tools/
    ├── parsing.py           # Tool call XML 解析
    └── prompts.py           # Tool prompt 注入
```

## 致谢

本项目基于 [HeZeBang/GenAI2OpenAI](https://github.com/HeZeBang/GenAI2OpenAI) 开发，感谢原作者的工作。

## 许可

MIT License — 详见 LICENSE 文件。
