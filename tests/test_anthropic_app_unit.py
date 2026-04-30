import json

from app import create_app
from config import Config
from provider import anthropic


class DummyTokenManager:
    mode = "static"

    def get_token(self):
        return "token"

    def force_refresh(self):
        return None


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


def make_app(api_key=None):
    return create_app(
        Config(
            token_manager=DummyTokenManager(),
            port=0,
            api_key=api_key,
            debug=False,
            api_format="both",
        )
    )


def test_non_streaming_messages_route_returns_anthropic_tool_use(monkeypatch):
    monkeypatch.setattr(anthropic.model_registry, "get_root_ai_type", lambda model, token: "xinference")
    monkeypatch.setattr(anthropic.requests, "post", lambda *args, **kwargs: FakeResponse())
    client = make_app().test_client()

    response = client.post(
        "/v1/messages",
        json={
            "model": "GPT-5.5",
            "stream": False,
            "max_tokens": 100,
            "tools": [
                {
                    "name": "Bash",
                    "description": "Run a command",
                    "input_schema": {"type": "object"},
                }
            ],
            "messages": [{"role": "user", "content": "run pwd"}],
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["stop_reason"] == "tool_use"
    assert data["content"] == [
        {
            "type": "tool_use",
            "id": data["content"][0]["id"],
            "name": "Bash",
            "input": {"command": "pwd"},
        }
    ]


def test_auth_accepts_x_api_key_for_anthropic_clients():
    client = make_app(api_key="local-proxy").test_client()

    response = client.post(
        "/v1/messages/count_tokens",
        headers={"x-api-key": "local-proxy"},
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    assert response.get_json()["input_tokens"] > 0
