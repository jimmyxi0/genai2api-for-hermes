import json
import logging
import re
import time
import uuid
from datetime import datetime

import requests

from config import GENAI_URL, build_genai_headers, model_registry
from errors import make_error_chunk
from tools.parsing import extract_tool_calls, _tag_prefix_len
from tools.prompts import flatten_message_content, normalize_message_content

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")

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


def convert_messages_to_genai_format(messages):
    chat_info = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            chat_info = flatten_message_content(msg.get("content", ""))
            break
    return chat_info


def extract_content_from_genai(response_data):
    try:
        if "choices" in response_data and len(response_data["choices"]) > 0:
            delta = response_data["choices"][0].get("delta", {})
            content = delta.get("content") or None
            reasoning = delta.get("reasoning_content") or None
            return content, reasoning
    except (KeyError, IndexError, TypeError):
        pass
    return None, None


def estimate_text_tokens(text):
    if not text:
        return 0
    return len(TOKEN_PATTERN.findall(text))


def log_stream_metrics(model, started_at, first_token_at, content_text, reasoning_text):
    total_elapsed = max(time.monotonic() - started_at, 1e-6)
    content_tokens = estimate_text_tokens(content_text)
    reasoning_tokens = estimate_text_tokens(reasoning_text)
    total_tokens = content_tokens + reasoning_tokens

    extra = ""
    if first_token_at is not None:
        ttft_ms = (first_token_at - started_at) * 1000
        extra = f" ttft_ms={ttft_ms:.0f}"

    logger.info(
        "stream metrics model=%s est_tokens=%d content_est=%d reasoning_est=%d toks_per_s=%.2f%s",
        model,
        total_tokens,
        content_tokens,
        reasoning_tokens,
        total_tokens / total_elapsed,
        extra,
    )


def _has_tool_marker(text):
    """Check if text contains any tool call marker."""
    lower = text.lower()
    for marker in TOOL_MARKER_PATTERNS:
        if marker in lower:
            return True
    # Also check for JSON-style tool patterns
    if '{"name"' in text and '"function"' in text:
        return True
    return False


def _has_partial_tool_marker(text):
    """Check if text *ends* with a partial tool marker that could complete on next chunk."""
    tail = text[-32:] if len(text) > 32 else text
    lower_tail = tail.lower()
    for marker in TOOL_MARKER_PATTERNS:
        # Check if the tail is a prefix of the marker, or the marker starts in the tail
        for i in range(len(marker)):
            if lower_tail.endswith(marker[:i + 1]) and i < len(marker) - 1:
                return True
    return False


def stream_genai_response(chat_info, messages, model, max_tokens, config):
    started_at = time.monotonic()
    first_token_at = None
    content_parts = []
    reasoning_parts = []
    token = config.token_manager.get_token()
    root_ai_type = model_registry.get_root_ai_type(model, token)
    headers = build_genai_headers(token)
    normalized_messages = [normalize_message_content(msg) for msg in messages]

    genai_data = {
        "chatInfo": "",
        "messages": normalized_messages,
        "type": "3",
        "stream": True,
        "aiType": model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000
    }

    logger.debug("=== GenAI Request ===")
    logger.debug("Model: %s, rootAiType: %s", model, root_ai_type)
    logger.debug("Messages count: %d", len(normalized_messages))
    for i, msg in enumerate(normalized_messages):
        role = msg.get('role', '?')
        content = flatten_message_content(msg.get('content', ''))
        preview = (content[:200] + '...') if content and len(content) > 200 else content
        logger.debug("  [%d] role=%s, content=%s", i, role, preview)

    try:
        response = requests.post(
            GENAI_URL,
            headers=headers,
            json=genai_data,
            stream=True,
            timeout=120
        )

        logger.debug("GenAI Response Status: %d", response.status_code)

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
            if response.status_code == 401:
                yield make_error_chunk("Upstream authentication failed", model)
            elif response.status_code == 429:
                yield make_error_chunk("Upstream rate limit exceeded", model)
            else:
                yield make_error_chunk(f"Upstream API error: {response.status_code}", model)
            return

        finished = False
        emitted_done = False
        line_count = 0
        for line in response.iter_lines():
            if finished:
                break

            if line:
                try:
                    line_str = line.decode('utf-8') if isinstance(line, bytes) else line

                    if line_count < 5:
                        logger.debug("Raw line [%d]: %s", line_count, line_str[:300])
                    line_count += 1

                    if line_str.startswith('data:'):
                        line_str = line_str[5:].strip()

                    if line_str:
                        genai_json = json.loads(line_str)

                        if isinstance(genai_json, dict) and genai_json.get("success") is False:
                            err_msg = genai_json.get("message", "Unknown upstream error")
                            err_code = genai_json.get("code", 500)
                            logger.warning("GenAI business error (code=%s): %s", err_code, err_msg)
                            yield make_error_chunk(f"Upstream error: {err_msg}", model)
                            return

                        if "choices" in genai_json and len(genai_json["choices"]) > 0:
                            choice = genai_json["choices"][0]
                            if choice.get("finish_reason") is not None:
                                finished = True

                        if finished:
                            log_stream_metrics(
                                model,
                                started_at,
                                first_token_at,
                                "".join(content_parts),
                                "".join(reasoning_parts),
                            )
                            final_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            emitted_done = True
                            yield f"data: {json.dumps(final_response)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                        content, reasoning = extract_content_from_genai(genai_json)

                        delta = {}
                        if content:
                            delta["content"] = content
                        if reasoning:
                            delta["reasoning_content"] = reasoning

                        if delta:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            if content:
                                content_parts.append(content)
                            if reasoning:
                                reasoning_parts.append(reasoning)
                            openai_response = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(datetime.now().timestamp()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(openai_response)}\n\n"

                except json.JSONDecodeError as e:
                    logger.debug("JSON decode error: %s, line: %s", e, line_str[:200])

        logger.debug("Total lines received: %d, finished: %s", line_count, finished)

        if emitted_done:
            return

        log_stream_metrics(
            model,
            started_at,
            first_token_at,
            "".join(content_parts),
            "".join(reasoning_parts),
        )
        final_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_response)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in stream_genai_response")
        yield make_error_chunk(str(e), model)


def _emit_text_chunk(completion_id, created, model, text):
    """Emit an OpenAI streaming text chunk."""
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": text},
            "finish_reason": None
        }]
    }
    return f"data: {json.dumps(chunk)}\n\n"


def _emit_tool_call_chunks(completion_id, created, model, tool_calls, remaining=None):
    """Emit OpenAI streaming chunks for tool calls."""
    chunks = []
    # Emit role first
    role_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None
        }]
    }
    chunks.append(f"data: {json.dumps(role_chunk)}\n\n")

    for i, tc in enumerate(tool_calls):
        tc_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }]
                },
                "finish_reason": None
            }]
        }
        chunks.append(f"data: {json.dumps(tc_chunk)}\n\n")

    if remaining and remaining.strip():
        chunks.append(_emit_text_chunk(completion_id, created, model, remaining.strip()))

    # Finish
    finish_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }
    chunks.append(f"data: {json.dumps(finish_chunk)}\n\n")
    chunks.append("data: [DONE]\n\n")
    return chunks


def _try_parse_tool_calls_from_buffer(buffer, allowed_tool_names):
    """Try to extract tool calls from buffer. Returns (tool_calls, remaining) or (None, None)."""
    if not _has_tool_marker(buffer):
        return None, None

    tool_calls, remaining = extract_tool_calls(buffer, allowed_tool_names=allowed_tool_names)

    if not tool_calls:
        # Retry with no name filter
        tool_calls, remaining = extract_tool_calls(buffer, allowed_tool_names=None)

    return tool_calls, remaining


def stream_genai_response_with_tools(chat_info, messages, model, max_tokens, config, allowed_tool_names=None):
    """Stream GenAI response with incremental tool detection.

    Strategy:
    - Phase 1 (detection): Buffer up to DETECTION_WINDOW chars while checking for tool markers.
      If markers found, switch to tool-buffering mode.
      If no markers after the window, flush buffer as text and switch to streaming mode.
    - Phase 2a (streaming): Stream text chunks directly. If a partial tool marker appears at
      the tail of a chunk, hold it back for the next chunk.
    - Phase 2b (tool-buffering): Continue buffering until tool call is complete, then emit
      tool_call chunks. Safety cap at MAX_TOOL_BUFFER.
    - On stream end: flush any remaining buffer appropriately.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())

    # States: "detecting" -> "streaming" or "tool_buffering"
    state = "detecting"
    buffer = ""
    sent_role = False
    tool_detected = False

    def _send_role():
        nonlocal sent_role
        if not sent_role:
            sent_role = True
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }]
            }
            return f"data: {json.dumps(role_chunk)}\n\n"
        return None

    for line in stream_genai_response(chat_info, messages, model, max_tokens, config):
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if "choices" not in data or not data["choices"]:
            continue

        chunk_delta = data["choices"][0].get("delta", {})
        content = chunk_delta.get("content", "")
        finish_reason = data["choices"][0].get("finish_reason")

        if not content:
            # Handle finish_reason from upstream (e.g., "stop")
            if finish_reason:
                break
            continue

        if state == "detecting":
            buffer += content

            # Check if we have a tool marker
            if _has_tool_marker(buffer):
                tool_detected = True
                state = "tool_buffering"
                logger.debug("Tool marker detected in detection window, switching to tool_buffering (buf=%d chars)", len(buffer))
                continue

            # Check if buffer exceeds detection window and no marker yet
            if len(buffer) >= DETECTION_WINDOW:
                # No tool markers — this is probably plain text.
                # Flush the buffer as text and switch to streaming mode.
                state = "streaming"
                logger.debug("No tool markers in %d chars, flushing as text, switching to streaming", len(buffer))
                role_chunk = _send_role()
                if role_chunk:
                    yield role_chunk
                yield _emit_text_chunk(completion_id, created, model, buffer)
                buffer = ""
                continue

            # Still within detection window, keep buffering
            continue

        elif state == "streaming":
            # In streaming mode, check for tool markers in new content
            if _has_tool_marker(content):
                # Tool marker appeared mid-stream! Switch to tool buffering.
                # We need to re-combine: previously emitted text + new content
                # Since we already emitted text, we can't un-emit it.
                # Best approach: buffer from this point onward and try to extract
                # the tool call from the buffered portion.
                tool_detected = True
                state = "tool_buffering"
                buffer = content  # start fresh buffer from the marker point
                logger.debug("Tool marker appeared mid-stream at %d chars, switching to tool_buffering", len(content))
                continue

            # Check if the tail of content has a partial marker
            if _has_partial_tool_marker(content):
                # Hold back the suspicious tail
                safe_len = len(content) - 32
                if safe_len > 0:
                    yield _emit_text_chunk(completion_id, created, model, content[:safe_len])
                    buffer = content[safe_len:]
                    # Don't change state yet — wait for next chunk
                    # Actually, we need a "pending" mini-state. Let's keep it simple:
                    # just keep buffer and check next chunk.
                    continue
                else:
                    buffer = content
                    continue

            # If we had a held-back tail, prepend it
            if buffer:
                content = buffer + content
                buffer = ""

            yield _emit_text_chunk(completion_id, created, model, content)

        elif state == "tool_buffering":
            buffer += content

            # Safety cap: if buffer grows too large without completing, emit as text
            if len(buffer) > MAX_TOOL_BUFFER:
                logger.warning("Tool buffer exceeded %d chars, flushing as text", MAX_TOOL_BUFFER)
                role_chunk = _send_role()
                if role_chunk:
                    yield role_chunk
                yield _emit_text_chunk(completion_id, created, model, buffer)
                buffer = ""
                state = "streaming"
                tool_detected = False
                continue

            # Try to parse tool calls from buffer
            tool_calls, remaining = _try_parse_tool_calls_from_buffer(buffer, allowed_tool_names)
            if tool_calls:
                logger.debug("Parsed %d tool_call(s) from buffer (%d chars)", len(tool_calls), len(buffer))
                role_chunk = _send_role()
                if role_chunk:
                    yield role_chunk
                for chunk in _emit_tool_call_chunks(completion_id, created, model, tool_calls, remaining):
                    yield chunk
                return  # Done — tool calls emitted

            # Not yet complete, keep buffering
            continue

    # Stream ended — handle remaining buffer
    if not buffer:
        if not sent_role:
            role_chunk = _send_role()
            if role_chunk:
                yield role_chunk
            yield _emit_text_chunk(completion_id, created, model, "")
        # Emit stop
        stop_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(stop_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if state == "tool_buffering" or tool_detected:
        # Try one final parse
        tool_calls, remaining = _try_parse_tool_calls_from_buffer(buffer, allowed_tool_names)
        if tool_calls:
            logger.debug("Final parse: %d tool_call(s) from buffer (%d chars)", len(tool_calls), len(buffer))
            role_chunk = _send_role()
            if role_chunk:
                yield role_chunk
            for chunk in _emit_tool_call_chunks(completion_id, created, model, tool_calls, remaining):
                yield chunk
            return

        # Failed to parse as tools — emit as text
        logger.debug("Tool buffering ended without valid parse, emitting %d chars as text", len(buffer))

    # Emit buffer as text
    role_chunk = _send_role()
    if role_chunk:
        yield role_chunk
    yield _emit_text_chunk(completion_id, created, model, buffer)

    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"