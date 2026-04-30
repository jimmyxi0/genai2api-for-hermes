# GenAI2OpenAI

将上海科技大学 GenAI 平台接入 Claude Code 的代理服务。

A proxy that connects ShanghaiTech's GenAI platform to Claude Code.

## 快速开始

### 1. 安装

```bash
uv sync
```

### 2. 启动代理

```bash
# 学号密码模式（推荐，支持自动刷新）
uv run main.py --token "学号@密码" --port 5000 --api-format both

# JWT 模式（过期需手动更换）
uv run main.py --token "eyJ..." --port 5000 --api-format both
```

手动获取 JWT：前往 [GenAI 对话平台](https://genai.shanghaitech.edu.cn/)，F12 打开 Network，发送消息后从 `chat` 请求头中复制 `x-access-token`。

### 3. 配置 Claude Code

设置环境变量（写入 `~/.bashrc` 或 `~/.zshrc`）：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:5000"
export ANTHROPIC_AUTH_TOKEN="local-proxy"   # 如果设了 --api-key，改成同一个值
export ANTHROPIC_MODEL="chatglm"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="MiniMax-M1"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-pro"
export ANTHROPIC_DEFAULT_OPUS_MODEL="chatglm"
export ANTHROPIC_REASONING_MODEL="MiniMax-M1"
```

### 4. 配置项目设置（减少权限弹窗）

在项目根目录创建 `.claude/settings.json`：

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

### 5. 启动

```bash
claude
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--token` | JWT 令牌或 `学号@密码`（必需） | — |
| `--port` | 服务监听端口 | `5000` |
| `--api-key` | 客户端认证密钥（也可通过 `API_KEY` 环境变量设置） | 无 |
| `--api-format` | `openai`、`anthropic` 或 `both` | `both` |
| `--debug` | 启用详细日志输出 | 关闭 |

## 特性

- **Claude Code 兼容** — 提供 Anthropic Messages API，支持 `tool_use/tool_result` 转换，可直接接入 Claude Code
- **OpenAI 兼容** — 同时提供 OpenAI Chat Completion API，支持 Cursor、Continue 等客户端
- **Tool Calling** — 通过 prompt 注入实现 function calling，兼容不原生支持的模型
- **自动登录与刷新** — 学号密码模式通过 CAS 自动登录，JWT 过期静默刷新
- **动态模型列表** — 自动从 GenAI 平台拉取可用模型

## 致谢

本项目基于 [HeZeBang/GenAI2OpenAI](https://github.com/HeZeBang/GenAI2OpenAI) 开发，感谢原作者的工作。

## 许可

MIT License — 详见 LICENSE 文件。
