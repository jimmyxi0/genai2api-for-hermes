from provider.genai import convert_messages_to_genai_format, estimate_text_tokens
from tools.prompts import (
    format_tool_examples,
    flatten_message_content,
    inject_tool_prompt,
    normalize_message_content,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo text",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"],
            },
        },
    }
]


def test_inject_tool_prompt_accepts_block_content():
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "System instructions"},
                {"type": "text", "text": "Second line"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
            ],
        },
    ]

    result = inject_tool_prompt(messages, TOOLS)

    assert result[0]["role"] == "system"
    assert isinstance(result[0]["content"], str)
    assert "System instructions" in result[0]["content"]
    assert "Second line" in result[0]["content"]
    assert "<tools>" in result[0]["content"]
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Hello"


def test_inject_tool_prompt_includes_claude_code_tool_examples():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Grep",
                "description": "Search files",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    result = inject_tool_prompt([{"role": "user", "content": "search"}], tools)

    assert '"name": "Grep"' in result[0]["content"]
    assert '"pattern": "search text"' in result[0]["content"]
    assert "NEVER use <arg_key>" in result[0]["content"]


def test_format_tool_examples_omits_unknown_tools():
    assert format_tool_examples(TOOLS) == ""


def test_flatten_message_content_handles_nested_blocks():
    content = [
        {"type": "text", "text": "Line 1"},
        {
            "type": "tool_result",
            "content": [
                {"type": "text", "text": "Nested"},
                {"type": "text", "text": "Blocks"},
            ],
        },
        {"type": "tool_use", "input": {"value": 1}},
    ]

    result = flatten_message_content(content)

    assert result == 'Line 1\nNested\nBlocks\n{"value": 1}'


def test_normalize_message_content_converts_blocks_to_string():
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Alpha"},
            {"type": "text", "text": "Beta"},
        ],
    }

    result = normalize_message_content(message)

    assert result == {"role": "user", "content": "Alpha\nBeta"}


def test_convert_messages_to_genai_format_uses_latest_user_block_content():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Older"}]},
        {"role": "assistant", "content": "ignored"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Latest"},
                {"type": "text", "text": "Prompt"},
            ],
        },
    ]

    result = convert_messages_to_genai_format(messages)

    assert result == "Latest\nPrompt"


def test_estimate_text_tokens_handles_mixed_text():
    result = estimate_text_tokens("Hello, 世界!")

    assert result == 5
