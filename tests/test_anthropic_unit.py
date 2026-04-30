import json

from provider import anthropic
from provider.anthropic import (
    anthropic_allowed_tool_names,
    anthropic_messages_to_genai_format,
    normalize_tool_input,
    parse_tool_arguments,
    stream_genai_as_anthropic,
)


def _events(chunks):
    events = []
    for chunk in chunks:
        event_name = None
        data = None
        for line in chunk.strip().splitlines():
            if line.startswith("event: "):
                event_name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_name and data:
            events.append((event_name, data))
    return events


def test_anthropic_messages_convert_tool_history_to_genai_promptable_messages():
    body = {
        "model": "GPT-5.5",
        "system": "You are Claude Code.",
        "tool_choice": {"type": "auto"},
        "tools": [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ],
        "messages": [
            {"role": "user", "content": "Read README.md"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [{"type": "text", "text": "file contents"}],
                    },
                    {"type": "text", "text": "Now summarize it."},
                ],
            },
        ],
    }

    _, messages, model = anthropic_messages_to_genai_format(body, "token")

    assert model == "GPT-5.5"
    assert anthropic_allowed_tool_names(body) == {"read_file"}
    assert messages[0]["role"] == "system"
    assert "<tools>" in messages[0]["content"]
    assert "read_file" in messages[0]["content"]
    assert any(
        message["role"] == "assistant"
        and "<tool_call>" in message["content"]
        and '"name": "read_file"' in message["content"]
        for message in messages
    )
    assert any(
        message["role"] == "user"
        and "<tool_result" in message["content"]
        and "file contents" in message["content"]
        and "Now summarize it." in message["content"]
        for message in messages
    )


def test_stream_genai_as_anthropic_emits_tool_use_blocks(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = ""

        def iter_lines(self):
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    '<tool_call>{"name": "Bash", '
                                    '"arguments": {"command": "pwd"}}</tool_call>'
                                )
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode()

    class DummyTokenManager:
        def force_refresh(self):
            return None

    class DummyConfig:
        token_manager = DummyTokenManager()

    monkeypatch.setattr(anthropic.model_registry, "get_root_ai_type", lambda model, token: "xinference")
    monkeypatch.setattr(anthropic.requests, "post", lambda *args, **kwargs: FakeResponse())

    events = _events(
        stream_genai_as_anthropic(
            [{"role": "user", "content": "run pwd"}],
            "GPT-5.5",
            1000,
            "token",
            DummyConfig(),
            allowed_tool_names={"Bash"},
        )
    )

    assert not any("<tool_call>" in json.dumps(data) for _, data in events)
    tool_starts = [
        data
        for event, data in events
        if event == "content_block_start"
        and data["content_block"]["type"] == "tool_use"
    ]
    assert tool_starts[0]["content_block"]["name"] == "Bash"
    input_deltas = [
        data["delta"]["partial_json"]
        for event, data in events
        if event == "content_block_delta"
        and data["delta"]["type"] == "input_json_delta"
    ]
    assert json.loads(input_deltas[0]) == {"command": "pwd"}
    message_delta = [data for event, data in events if event == "message_delta"][0]
    assert message_delta["delta"]["stop_reason"] == "tool_use"


def test_stream_genai_as_anthropic_emits_bare_malformed_json_tool_use(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = ""

        def iter_lines(self):
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "content": '{"name": "Bash", "arguments": {"command": "printf "$(pwd)""}}'
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode()

    class DummyTokenManager:
        def force_refresh(self):
            return None

    class DummyConfig:
        token_manager = DummyTokenManager()

    monkeypatch.setattr(anthropic.model_registry, "get_root_ai_type", lambda model, token: "xinference")
    monkeypatch.setattr(anthropic.requests, "post", lambda *args, **kwargs: FakeResponse())

    events = _events(
        stream_genai_as_anthropic(
            [{"role": "user", "content": "run pwd"}],
            "chatglm",
            1000,
            "token",
            DummyConfig(),
            allowed_tool_names={"Bash"},
        )
    )

    tool_starts = [
        data
        for event, data in events
        if event == "content_block_start"
        and data["content_block"]["type"] == "tool_use"
    ]
    assert tool_starts[0]["content_block"]["name"] == "Bash"
    input_delta = next(
        data["delta"]["partial_json"]
        for event, data in events
        if event == "content_block_delta"
        and data["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(input_delta) == {"command": 'printf "$(pwd)"'}


def test_stream_genai_as_anthropic_filters_split_thinking_blocks(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = ""

        def iter_lines(self):
            chunks = [
                {"choices": [{"delta": {"content": "<think>hidden"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " reasoning</think>\n\nvisible"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            for chunk in chunks:
                yield ("data: " + json.dumps(chunk)).encode()

    class DummyTokenManager:
        def force_refresh(self):
            return None

    class DummyConfig:
        token_manager = DummyTokenManager()

    monkeypatch.setattr(anthropic.model_registry, "get_root_ai_type", lambda model, token: "xinference")
    monkeypatch.setattr(anthropic.requests, "post", lambda *args, **kwargs: FakeResponse())

    events = _events(
        stream_genai_as_anthropic(
            [{"role": "user", "content": "reply"}],
            "MiniMax-M1",
            1000,
            "token",
            DummyConfig(),
        )
    )
    text = "".join(
        data["delta"]["text"]
        for event, data in events
        if event == "content_block_delta"
        and data.get("delta", {}).get("type") == "text_delta"
    )

    assert text == "\n\nvisible"


def test_parse_tool_arguments_unwraps_nested_json_strings():
    assert parse_tool_arguments('{"arguments":"{\\"path\\": \\"README.md\\"}"}') == {
        "path": "README.md"
    }


def test_normalize_tool_input_maps_bash_arguments_string_to_command():
    assert normalize_tool_input("Bash", '"pwd"') == {"command": "pwd"}
    assert normalize_tool_input("Bash", '{"arguments": "pwd"}') == {"command": "pwd"}
    assert normalize_tool_input("Bash", '{"arguments": {"command": "pwd"}}') == {"command": "pwd"}


def test_normalize_tool_input_maps_claude_code_tool_aliases():
    assert normalize_tool_input("Grep", {"cpattern": "Claude Code", "file_path": "README.md"}) == {
        "pattern": "Claude Code",
        "path": "README.md",
    }
    assert normalize_tool_input("Read", {"path": "README.md"}) == {"file_path": "README.md"}
    assert normalize_tool_input("Glob", {"glob": "tests/*.py", "filePath": "/tmp/project"}) == {
        "pattern": "tests/*.py",
        "path": "/tmp/project",
    }
