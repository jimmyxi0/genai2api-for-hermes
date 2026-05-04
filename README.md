# GenAI2API-for-Agents

A proxy that connects ShanghaiTech GenAI platform to AI coding agents (Hermes Agent, Claude Code, Cursor, Continue, etc.) via OpenAI and Anthropic compatible APIs.

将上海科技大学 GenAI 平台接入 AI 编程代理（Hermes Agent、Claude Code、Cursor、Continue 等）的代理服务。

## Prerequisites

- **ShanghaiTech University account** — You must be a current student or faculty with access to [GenAI](https://genai.shanghaitech.edu.cn/)
- **Python 3.10+** with [uv](https://docs.astral.sh/uv/) package manager
- **Linux or macOS** (Windows WSL also works)

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/jimmyxi0/genai2api-for-hermes.git
cd genai2api-for-hermes
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
uv run python main.py --token "student_id@password" --port 5000 --api-format both

# JWT mode (requires manual token renewal)
uv run python main.py --token "eyJ..." --port 5000 --api-format both

# With debug logging
uv run python main.py --token "student_id@password" --port 5000 --api-format both --debug
```

To obtain a JWT manually: go to [GenAI](https://genai.shanghaitech.edu.cn/), open DevTools (F12), send a message, and copy the `x-access-token` from the `chat` request header.

### 3. Configure Your Agent

#### Hermes Agent

1. Install [Hermes Agent](https://hermes-agent.nousresearch.com/):

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

2. Edit `~/.hermes/config.yaml`:

```yaml
model:
  default: chatglm          # or qwen-instruct, deepseek-pro, etc.
  provider: custom
  base_url: http://127.0.0.1:5000/v1
  default_max_tokens: 128000

providers: {}

agent:
  max_turns: 150
```

3. Launch:

```bash
hermes
```

That's it! Hermes will use your campus GenAI for free.

#### Claude Code

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

Create `.claude/settings.json` in your project root:

```json
{
  "permissions": {
    "allow": [
      "Bash(npm install)",
      "Bash(npm run build)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git diff:*)",
      "Read",
      "Write"
    ]
  }
}
```

Launch:

```bash
claude
```

#### Cursor / Continue / Other OpenAI-compatible tools

Set the API base URL to `http://127.0.0.1:5000/v1` and use any model name available on the GenAI platform. No API key needed unless you set `--api-key`.

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--token` | JWT token or `student_id@password` (required) | — |
| `--port` | Server listening port | `5000` |
| `--api-key` | Client authentication key (or set `API_KEY` env var) | none |
| `--api-format` | `openai`, `anthropic`, or `both` | `both` |
| `--debug` | Enable verbose logging | off |

## Available Models

Models are fetched dynamically from the GenAI platform. Common ones include:

| Model ID | Description |
|----------|-------------|
| `chatglm` | GLM general-purpose model |
| `qwen-instruct` | Qwen instruction-following model |
| `deepseek-pro` | DeepSeek Pro (strong coding) |
| `MiniMax-M1` | MiniMax reasoning model |

Run with `--debug` and check logs for the full list of available models on your account.

## Features

- **Hermes Agent compatible** — OpenAI Chat Completion API with tool calling
- **Claude Code compatible** — Anthropic Messages API with `tool_use/tool_result` conversion
- **OpenAI compatible** — Chat Completion API for Cursor, Continue, etc.
- **Tool Calling** — Function calling via prompt injection for models without native support
- **Auto-login & refresh** — CAS auto-login with student ID/password, silent JWT renewal
- **Dynamic model list** — Automatically fetches available models from GenAI platform
- **Smart token estimation** — Accurate heuristic tokenizer with safety buffer to prevent truncation
- **Context window protection** — Auto-caps `max_tokens` when prompt uses >50% of context window
- **Streaming support** — Full SSE streaming for both OpenAI and Anthropic formats

## Troubleshooting

### Empty responses or 500 errors

- Check that your token is valid: visit [GenAI](https://genai.shanghaitech.edu.cn/) directly
- Restart the proxy to force a fresh login
- Use `--debug` flag to see detailed request/response logs

### Truncation (responses cut off mid-sentence)

- The proxy automatically estimates prompt tokens and reserves space for the response
- If you still see truncation, check logs for `finish_reason: length` — this means the upstream GenAI API cut the response
- Try reducing your prompt length or switching to a model with a larger context window

### "estimate_messages_tokens() got unexpected keyword argument"

- Make sure you're on the latest version: `git pull && uv sync`
- This was a bug in earlier versions, fixed in commit 4755b92

### Connection refused

- Ensure the proxy is running: `curl http://localhost:5000/health`
- Check the port matches between proxy and your agent config
- If using a different port, update `--port` and your agent's `base_url`

### Token expires / auth fails

- Use student ID + password mode (not JWT) for automatic token refresh
- If CAS login fails, check that your campus credentials are correct at [ids.shanghaitech.edu.cn](https://ids.shanghaitech.edu.cn/)

## Architecture

```
Your Agent (Hermes/Claude Code/Cursor)
        |
        | OpenAI / Anthropic API
        v
  +-------------------+
  |   GenAI2API Proxy |  (localhost:5000)
  |                   |
  |  - Token estimation & context protection
  |  - Tool call prompt injection
  |  - API format conversion
  |  - Auto JWT refresh
  +-------------------+
        |
        | ShanghaiTech GenAI API
        v
  genai.shanghaitech.edu.cn
```

## Acknowledgments

Based on [HeZeBang/GenAI2OpenAI](https://github.com/HeZeBang/GenAI2OpenAI). Thanks to the original author.

## License

MIT License — see LICENSE file.
