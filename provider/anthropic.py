import json
import logging
import ast
import re
import time
import uuid
# from datetime import datetime  # unused
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from config import GENAI_URL, build_genai_headers, model_registry
from tools.parsing import extract_tool_calls
from tools.prompts import inject_tool_prompt

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"
BARE_TOOL_CALL_RE = re.compile(r'\{\s*"name"\s*:\s*"')

DSML_OPEN_RE = r"<\s*[|｜]DSML[|｜](?:tool_calls|invoke|parameter)\b"
DSML_CLOSE_RE = r"</\s*[|｜]DSML[|｜](?:tool_calls|invoke|parameter)\s*>"
THINK_OPEN_PREFIX = "<think"
THINK_CLOSE = "</think>"


# Incremental tool detection markers (lowercase)
TOOL_MARKER_PATTERNS = [
    "<tool_call",
    "<tool_calls",
    "<invoke",
    "<tool_call>",          # thinking/parameter tag
    "<|dsml",
    "ｰ",           # fullwidth horizontal bar used in DSML
]

# How many chars to buffer before deciding "this is probably text, not tools"
DETECTION_WINDOW = 512

# Max chars to buffer when we've detected a tool call marker (safety cap)
MAX_TOOL_BUFFER = 65536

NATURAL_ENDING_RE = re.compile(
    r'[.。!！?？\n:：;；\)）\]】』》""…\u2026]$'
    r'|[\U0001F300-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]\s*$'
)
MIN_TRUNCATION_CHECK_LEN = 200


def _looks_like_natural_ending(text: str) -> bool:
    if not text:
        return True
    if len(text) < MIN_TRUNCATION_CHECK_LEN:
        return True
    stripped = text.rstrip()
    if not stripped:
        return True
    if NATURAL_ENDING_RE.search(stripped):
        return True
    return False


def anthropic_content_to_text(content: Any) -> str:
    """Convert Anthropic content format to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    text_parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(part.get("text", ""))
        elif part_type == "tool_use":
            text_parts.append(
                f"[tool_use name={part.get('name', '')} id={part.get('id', '')}] "
                f"{json.dumps(part.get('input', {}), ensure_ascii=False)}"
            )
        elif part_type == "tool_result":
            text_parts.append(f"[tool_result id={part.get('tool_use_id', '')}] {tool_result_content_to_text(part.get('content', ''))}")
        elif part_type == "image":
            text_parts.append("[image]")
        else:
            text_parts.append(json.dumps(part, ensure_ascii=False))
    return "\n".join(part for part in text_parts if part)


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


def anthropic_allowed_tool_names(body: Dict[str, Any]) -> set[str]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return set()
    return {
        tool["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str) and tool["name"]
    }


def anthropic_tools_to_openai_tools(tools: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted

    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        converted.append({
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return converted


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return None
    choice_type = tool_choice.get("type")
    if choice_type == "any":
        return "required"
    if choice_type == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    if choice_type == "none":
        return "none"
    return None


def tool_result_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return anthropic_content_to_text(content)
    if content is None:
        return ""
    return str(content)


def escape_tool_result_attr(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


def anthropic_tool_results_visible_text(tool_results: List[Dict[str, Any]]) -> str:
    blocks: List[str] = []
    for result in tool_results:
        attrs = [f'tool_use_id="{escape_tool_result_attr(result.get("tool_use_id", ""))}"']
        if "is_error" in result:
            attrs.append(f'is_error="{str(bool(result.get("is_error"))).lower()}"')
        content = tool_result_content_to_text(result.get("content", ""))
        blocks.append(f"<tool_result {' '.join(attrs)}>\n{content}\n</tool_result>")
    return "<tool_results>\n" + "\n".join(blocks) + "\n</tool_results>" if blocks else ""


def split_anthropic_content(content: Any) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_parts: List[str] = []
    tool_uses: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    if isinstance(content, str):
        return content, tool_uses, tool_results
    if not isinstance(content, list):
        return ("" if content is None else str(content)), tool_uses, tool_results

    for part in content:
        if isinstance(part, str):
            text_parts.append(part)
            continue
        if not isinstance(part, dict):
            text_parts.append(str(part))
            continue
        part_type = part.get("type")
        if part_type == "text":
            text_parts.append(part.get("text", ""))
        elif part_type == "tool_use":
            tool_uses.append(part)
        elif part_type == "tool_result":
            tool_results.append(part)
        elif part_type == "image":
            text_parts.append("[image]")
        else:
            text_parts.append(json.dumps(part, ensure_ascii=False))

    return "\n".join(part for part in text_parts if part), tool_uses, tool_results


def anthropic_message_to_genai_messages(message: Dict[str, Any]) -> List[Dict[str, Any]]:
    role = message.get("role", "user")
    if role not in ("user", "assistant", "system"):
        role = "user"

    text, tool_uses, tool_results = split_anthropic_content(message.get("content", ""))
    if tool_results:
        visible_text = anthropic_tool_results_visible_text(tool_results)
        if text:
            visible_text = f"{visible_text}\n\n{text}" if visible_text else text
        return [{"role": "user", "content": visible_text}]

    if role == "assistant" and tool_uses:
        content = text or ""
        for tool_use in tool_uses:
            call_obj = {
                "name": tool_use.get("name", ""),
                "arguments": tool_use.get("input", {}),
            }
            content += f"\n<tool_call>\n{json.dumps(call_obj, ensure_ascii=False)}\n</arg_value>"
        return [{"role": "assistant", "content": content.strip()}]

    return [{"role": role, "content": text}]


def anthropic_messages_to_genai_format(body: Dict[str, Any], token: str) -> Tuple[str, List[Dict[str, Any]], str]:
    """Convert Anthropic Messages API format to GenAI format.

    Returns:
        Tuple of (system_prompt, messages, model)
    """
    model = body.get("model", "GPT-5.5")
    # Extract system prompt
    system_texts: List[str] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        system_texts.append(system)
    elif isinstance(system, list):
        for item in system:
            if isinstance(item, dict) and item.get("type") == "text":
                system_texts.append(item.get("text", ""))
            elif isinstance(item, str):
                system_texts.append(item)

    system_prompt = "\n\n".join(system_texts)

    # Convert messages
    messages: List[Dict[str, Any]] = []
    for message in body.get("messages", []):
        if not isinstance(message, dict):
            continue
        messages.extend(anthropic_message_to_genai_messages(message))

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    openai_tools = anthropic_tools_to_openai_tools(body.get("tools"))
    tool_choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if openai_tools and tool_choice != "none":
        messages = inject_tool_prompt(messages, openai_tools, tool_choice)

    return system_prompt, messages, model


def extract_content_from_genai(response_data: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract content and reasoning from GenAI response."""
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            content = delta.get("content") or None
            reasoning = delta.get("reasoning_content") or None
            return content, reasoning
    except (KeyError, IndexError, TypeError):
        pass
    return None, None


def filter_thinking_and_dsml(text: str) -> str:
    """Filter out  aż...  and DSML tool tags."""
    # Remove thinking tags
    text = re.sub(r"<think\b[^>]*>.*? ", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove DSML tags
    text = re.sub(DSML_OPEN_RE, "", text, flags=re.IGNORECASE)
    text = re.sub(DSML_CLOSE_RE, "", text, flags=re.IGNORECASE)

    return text


def partial_marker_start(text: str, position: int, marker: str) -> int:
    marker = marker.lower()
    for index in range(len(text) - 1, position - 1, -1):
        suffix = text[index:].lower()
        if marker.startswith(suffix) and suffix != marker:
            return index
    return -1


def filter_thinking_text_delta(text: str, state: Dict[str, Any]) -> str:
    """Filter thinking markup even when   spans multiple stream chunks."""
    pending = state.pop("pending_think_open", "")
    if pending:
        text = pending + text

    output: List[str] = []
    position = 0
    while position < len(text):
        lower = text.lower()
        if state.get("in_thinking"):
            close_index = lower.find(THINK_CLOSE, position)
            if close_index < 0:
                return "".join(output)
            position = close_index + len(THINK_CLOSE)
            state["in_thinking"] = False
            continue

        open_index = lower.find(THINK_OPEN_PREFIX, position)
        stray_close_index = lower.find(THINK_CLOSE, position)
        if stray_close_index >= 0 and (open_index < 0 or stray_close_index < open_index):
            output.append(text[position:stray_close_index])
            position = stray_close_index + len(THINK_CLOSE)
            continue

        if open_index < 0:
            partial_open = partial_marker_start(text, position, THINK_OPEN_PREFIX)
            if partial_open >= 0:
                output.append(text[position:partial_open])
                state["pending_think_open"] = text[partial_open:]
            else:
                output.append(text[position:])
            break

        output.append(text[position:open_index])
        tag_end = text.find(">", open_index)
        if tag_end < 0:
            state["in_thinking"] = True
            break
        position = tag_end + 1
        state["in_thinking"] = True

    return filter_thinking_and_dsml("".join(output))


def strip_thinking_markup(text: str) -> str:
    cleaned = re.sub(r"<think\b[^>]*>.*? ", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*</think>\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*<think\b[^>]*>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def strip_markdown_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def extract_balanced_json(text: str) -> Optional[str]:
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    stack: List[str] = []
    in_string = False
    escape = False
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in ("'", '"'):
            in_string = True
            quote = char
            continue
        if char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return text[start:index + 1]
    return None


def quote_unquoted_json_keys(text: str) -> str:
    return re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)', r'\1"\2"\3', text)


def parse_json_like_object(arguments: str) -> Optional[Any]:
    candidates = []
    cleaned = strip_markdown_json_fence(strip_thinking_markup(arguments))
    if cleaned:
        candidates.append(cleaned)
    balanced = extract_balanced_json(cleaned or arguments)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    for candidate in candidates:
        for value in (candidate, quote_unquoted_json_keys(candidate)):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                pass
    return None


def parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        if set(arguments) == {"arguments"}:
            nested_value = arguments.get("arguments")
            if isinstance(nested_value, dict):
                return parse_tool_arguments(nested_value)
            if isinstance(nested_value, str):
                return parse_tool_arguments(nested_value)
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    parsed = parse_json_like_object(arguments)
    if parsed is None:
        return {"arguments": arguments}
    if isinstance(parsed, str):
        nested = parse_tool_arguments(parsed)
        return nested if nested else {"arguments": parsed}
    if isinstance(parsed, dict):
        return parse_tool_arguments(parsed)
    return {"arguments": parsed}


def tool_arguments_json(arguments: Any) -> str:
    return json_dumps_compact(parse_tool_arguments(arguments))


def normalize_tool_input(tool_name: str, arguments: Any) -> Dict[str, Any]:
    parsed = parse_tool_arguments(arguments)
    aliases = {
        "Grep": {
            "cpattern": "pattern",
            "regex": "pattern",
            "file_path": "path",
            "filePath": "path",
        },
        "Glob": {
            "glob": "pattern",
            "file_path": "path",
            "filePath": "path",
        },
        "Read": {
            "path": "file_path",
            "filePath": "file_path",
        },
    }
    for alias, canonical in aliases.get(tool_name, {}).items():
        if canonical not in parsed and alias in parsed:
            parsed[canonical] = parsed.pop(alias)

    if tool_name == "Bash" and "command" not in parsed:
        value = parsed.get("arguments")
        if isinstance(value, str) and value:
            return {"command": value}
        if isinstance(value, dict):
            nested = normalize_tool_input(tool_name, value)
            if nested.get("command"):
                return nested
    return parsed


def tool_input_json(tool_name: str, arguments: Any) -> str:
    return json_dumps_compact(normalize_tool_input(tool_name, arguments))


def _has_tool_marker(text: str) -> bool:
    """Check if text contains any tool call marker."""
    lower = text.lower()
    for marker in TOOL_MARKER_PATTERNS:
        if marker in lower:
            return True
    # Also check for JSON-style tool patterns
    if '{"name"' in text:
        return True
    # Check for plain text tool patterns like "edit>", "write>", "Bash "
    if re.search(r'\b(bash|read|glob|grep|edit|write|ls|towrite|webfetch|websearch)\s*[>(\s]', text, re.IGNORECASE):
        return True
    return False


def _has_partial_tool_marker(text: str) -> bool:
    """Check if text *ends* with a partial tool marker that could complete on next chunk."""
    tail = text[-32:] if len(text) > 32 else text
    lower_tail = tail.lower()
    for marker in TOOL_MARKER_PATTERNS:
        for i in range(len(marker)):
            if lower_tail.endswith(marker[:i + 1]) and i < len(marker) - 1:
                return True
    return False


def _try_parse_tool_calls_from_buffer(buffer: str, allowed_tool_names: Optional[set[str]]) -> Tuple[Optional[list], Optional[str]]:
    """Try to extract tool calls from buffer. Returns (tool_calls, remaining) or (None, None)."""
    if not _has_tool_marker(buffer):
        return None, None

    tool_calls, remaining = extract_tool_calls(buffer, allowed_tool_names=allowed_tool_names)
    if not tool_calls:
        tool_calls, remaining = extract_tool_calls(buffer, allowed_tool_names=None)
    return tool_calls, remaining


def stream_genai_as_anthropic(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    token: str,
    config: Any,
    allowed_tool_names: Optional[set[str]] = None,
) -> Generator[str, None, None]:
    """Stream GenAI response in Anthropic Messages API format with incremental tool detection.

    Strategy (same as OpenAI path):
    - Phase 1 (detecting): Buffer up to DETECTION_WINDOW chars while checking for tool markers.
      If markers found, switch to tool_buffering.
      If no markers after the window, flush buffer as text and switch to streaming.
    - Phase 2a (streaming): Stream text deltas directly. If a partial tool marker appears,
      hold it back. If a full marker appears, switch to tool_buffering.
    - Phase 2b (tool_buffering): Continue buffering until tool call is complete, then emit
      tool_use blocks. Safety cap at MAX_TOOL_BUFFER.
    - On stream end: flush any remaining buffer appropriately.
    """
    started_at = time.monotonic()
    first_token_at = None
    content_parts: List[str] = []
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    headers = build_genai_headers(token)
    root_ai_type = model_registry.get_root_ai_type(model, token)

    genai_data = {
        "chatInfo": "",
        "messages": messages,
        "type": "3",
        "stream": True,
        "aiType": model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens
    }

    logger.debug("=== GenAI Request (Anthropic mode) ===")
    logger.debug("Model: %s, rootAiType: %s", model, root_ai_type)

    def write_sse(event: str, data: Dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"

    # Send message_start event
    yield write_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })

    text_block_started = False
    text_block_stopped = False
    thinking_state: Dict[str, Any] = {"in_thinking": False}

    # States: "detecting" -> "streaming" or "tool_buffering"
    state = "detecting"
    buffer = ""
    tool_detected = False

    def emit_text(text: str) -> Generator[str, None, None]:
        nonlocal first_token_at, text_block_started
        if not text:
            return
        if first_token_at is None:
            first_token_at = time.monotonic()
        if not text_block_started:
            yield write_sse("content_block_start", {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })
            text_block_started = True
        content_parts.append(text)
        yield write_sse("content_block_delta", {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        })

    def close_text_block() -> Generator[str, None, None]:
        nonlocal text_block_stopped
        if text_block_started and not text_block_stopped:
            yield write_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            text_block_stopped = True

    def emit_tool_blocks(tool_calls: list, remaining: Optional[str] = None) -> Generator[str, None, None]:
        """Emit Anthropic tool_use content blocks."""
        nonlocal text_block_started
        next_index = 1 if text_block_started else 0
        if remaining:
            yield from emit_text(remaining)
        yield from close_text_block()

        for offset, tool_call in enumerate(tool_calls):
            function = tool_call.get("function", {})
            block_index = next_index + offset
            tool_id = tool_call.get("id") or f"toolu_genai_{uuid.uuid4().hex[:24]}"
            tool_name = function.get("name") or "tool"
            arguments = function.get("arguments", "{}")
            yield write_sse("content_block_start", {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {},
                },
            })
            yield write_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": tool_input_json(tool_name, arguments),
                },
            })
            yield write_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": block_index,
            })

    try:
        response = requests.post(
            GENAI_URL,
            headers=headers,
            json=genai_data,
            stream=True,
            timeout=120
        )

        if response.status_code == 401:
            new_token = config.token_manager.force_refresh()
            if new_token:
                logger.info("Token refreshed after 401, retrying request")
                headers = build_genai_headers(new_token)
                response = requests.post(
                    GENAI_URL, headers=headers, json=genai_data,
                    stream=True, timeout=120
                )

        if response.status_code != 200:
            logger.warning("GenAI API error %d: %s", response.status_code, response.text[:500])
            yield write_sse("error", {
                "type": "error",
                "error": {"type": "api_error", "message": f"Upstream API error: {response.status_code}"},
            })
            return

        finished = False
        upstream_finish_reason = "stop"
        for line in response.iter_lines():
            if finished:
                break

            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()

                    if line_str:
                        genai_json = json.loads(line_str)

                        if isinstance(genai_json, dict) and genai_json.get("success") is False:
                            err_msg = genai_json.get("message", "Unknown upstream error")
                            logger.warning("GenAI business error: %s", err_msg)
                            yield write_sse("error", {
                                "type": "error",
                                "error": {"type": "api_error", "message": f"Upstream error: {err_msg}"},
                            })
                            return

                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            fr = choice.get("finish_reason")
                            if fr is not None:
                                upstream_finish_reason = fr
                                finished = True

                        content, _reasoning = extract_content_from_genai(genai_json)

                        if content:
                            content = filter_thinking_text_delta(content, thinking_state)

                        if not content:
                            continue

                        # === Incremental streaming state machine ===

                        if state == "detecting":
                            buffer += content

                            if _has_tool_marker(buffer):
                                tool_detected = True
                                state = "tool_buffering"
                                logger.debug("Anthropic: tool marker detected in detection window, switching to tool_buffering (buf=%d chars)", len(buffer))
                                continue

                            if len(buffer) >= DETECTION_WINDOW:
                                state = "streaming"
                                logger.debug("Anthropic: no tool markers in %d chars, flushing as text, switching to streaming", len(buffer))
                                yield from emit_text(buffer)
                                buffer = ""
                                continue

                            # Still within detection window, keep buffering
                            continue

                        elif state == "streaming":
                            if _has_tool_marker(content):
                                tool_detected = True
                                state = "tool_buffering"
                                buffer = content
                                logger.debug("Anthropic: tool marker appeared mid-stream, switching to tool_buffering")
                                continue

                            if _has_partial_tool_marker(content):
                                safe_len = len(content) - 32
                                if safe_len > 0:
                                    yield from emit_text(content[:safe_len])
                                    buffer = content[safe_len:]
                                else:
                                    buffer = content
                                continue

                            if buffer:
                                content = buffer + content
                                buffer = ""

                            yield from emit_text(content)

                        elif state == "tool_buffering":
                            buffer += content

                            if len(buffer) > MAX_TOOL_BUFFER:
                                logger.warning("Anthropic: tool buffer exceeded %d chars, flushing as text", MAX_TOOL_BUFFER)
                                yield from emit_text(buffer)
                                buffer = ""
                                state = "streaming"
                                tool_detected = False
                                continue

                            tool_calls, remaining = _try_parse_tool_calls_from_buffer(buffer, allowed_tool_names)
                            if tool_calls:
                                logger.debug("Anthropic: parsed %d tool_call(s) from buffer (%d chars)", len(tool_calls), len(buffer))
                                yield from emit_tool_blocks(tool_calls, remaining)
                                # Emit final events and return
                                output_text = "".join(content_parts)
                                output_tokens = max(1, len(output_text) // 4) if output_text else 0
                                yield write_sse("message_delta", {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                                    "usage": {"output_tokens": output_tokens},
                                })
                                yield write_sse("message_stop", {"type": "message_stop"})
                                return

                            # Not yet complete, keep buffering
                            continue

                except json.JSONDecodeError:
                    pass

        # Stream ended — handle remaining buffer
        stop_reason = "end_turn"
        output_text = "".join(content_parts)

        if upstream_finish_reason == "stop" and not _looks_like_natural_ending(output_text):
            logger.warning(
                "Anthropic: treating suspicious stop response as truncated (len=%d)",
                len(output_text),
            )
            stop_reason = "max_tokens"
        elif upstream_finish_reason == "length":
            logger.warning(
                "Anthropic: response truncated (upstream finish_reason='length')"
            )
            stop_reason = "max_tokens"

        if buffer:
            if state == "tool_buffering" or tool_detected:
                tool_calls, remaining = _try_parse_tool_calls_from_buffer(buffer, allowed_tool_names)
                if tool_calls:
                    logger.debug("Anthropic: final parse: %d tool_call(s) from buffer", len(tool_calls))
                    yield from emit_tool_blocks(tool_calls, remaining)
                    stop_reason = "tool_use"
                else:
                    logger.debug("Anthropic: tool buffering ended without valid parse, emitting %d chars as text", len(buffer))
                    yield from emit_text(buffer)
            else:
                yield from emit_text(buffer)

        yield from close_text_block()

        # Calculate output tokens
        output_tokens = max(1, len(output_text) // 4) if output_text else 0

        # Send message_delta with stop_reason
        yield write_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

        # Send message_stop
        yield write_sse("message_stop", {"type": "message_stop"})

        total_elapsed = time.monotonic() - started_at
        ttft_ms = (first_token_at - started_at) * 1000 if first_token_at else 0
        logger.info(
            "stream metrics model=%s output_chars=%d output_tokens=%d ttft_ms=%.0f total=%.2fs",
            model,
            len(output_text),
            output_tokens,
            ttft_ms,
            total_elapsed,
        )

    except Exception as e:
        logger.exception("Error in stream_genai_as_anthropic")
        yield write_sse("error", {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })
