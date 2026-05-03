# GenAI2API-for-Agents

A proxy that connects ShanghaiTech GenAI platform to AI coding agents (Claude Code, Cursor, Continue, etc.) via OpenAI and Anthropic compatible APIs.

将上海科技大学 GenAI 平台接入 AI 编程代理（Claude Code、Cursor、Continue 等）的代理服务。

## Quick Start

### 1. Install

```bash
uv sync
```

### 2. Start the Proxy

**Option A: One-command start (recommended)**

1. Edit `.env` with your credentials:
   - `GENAI_TOKEN` — student ID + password (e.g. `your_student_id@your_password`) or a JWT token
   - `API_KEY` — optional API key for client auth
   - `API_FORMAT` — `openai`, `anthropic`, or `both` (default: `both`)
   - `PORT` — server port (default: `5000`)

2. Run:

```bash
./start.sh
```

**Option B: Manual start**

```bash
# Student ID + password mode (recommended, supports auto-refresh)
uv run main.py --token "student_id@password" --port 5000 --api-format both

# JWT mode (requires manual token renewal)
uv run main.py --token "eyJ..." --port 5000 --api-format both
```

To obtain a JWT manually: go to [GenAI](https://genai.shanghaitech.edu.cn/), open DevTools (F12), send a message, and copy the `x-access-token` from the `chat` request header.

### 3. Configure Claude Code

Set environment variables in `~/.bashrc` or `~/.zshrc`:

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:5000"
export ANTHROPIC_AUTH_TOKEN="local-proxy"   # match --api-key if set
export ANTHROPIC_MODEL="chatglm"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="MiniMax-M1"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-pro"
export ANTHROPIC_DEFAULT_OPUS_MODEL="chatglm"
export ANTHROPIC_REASONING_MODEL="MiniMax-M1"
```

### 4. Configure Project Settings (reduce permission prompts)

Create `.claude/settings.json` in your project root:

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

### 5. Launch

```bash
claude
```

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--token` | JWT token or `student_id@password` (required) | — |
| `--port` | Server listening port | `5000` |
| `--api-key` | Client authentication key (or set `API_KEY` env var) | none |
| `--api-format` | `openai`, `anthropic`, or `both` | `both` |
| `--debug` | Enable verbose logging | off |

## Features

- **Claude Code compatible** — Anthropic Messages API with `tool_use/tool_result` conversion
- **OpenAI compatible** — Chat Completion API for Cursor, Continue, etc.
- **Tool Calling** — Function calling via prompt injection for models without native support
- **Auto-login & refresh** — CAS auto-login with student ID/password, silent JWT renewal
- **Dynamic model list** — Automatically fetches available models from GenAI platform

## Acknowledgments

Based on [HeZeBang/GenAI2OpenAI](https://github.com/HeZeBang/GenAI2OpenAI). Thanks to the original author.

## License

MIT License — see LICENSE file.