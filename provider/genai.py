import json
import logging
import re
import time
import uuid
from datetime import datetime

import requests

from config import GENAI_URL, build_genai_headers, model_registry
from errors import make_error_chunk
from tools.parsing import extract_tool_calls, parse_truncated_tool_call
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
    "ｰ",                # fullwidth horizontal bar used in DSML
    "〉",                # U+3009 RIGHT CORNER BRACKET (from model output)
]

# How many chars to buffer before deciding "this is probably text, not tools"
DETECTION_WINDOW = 128

# Max chars to buffer when we've detected a tool call marker (safety cap)
MAX_TOOL_BUFFER = 65536

NATURAL_ENDING_RE = re.compile(
    r'[.。!！?？\n:：;；\)）\]】』》""…\u2026]$'
    r'|[\U0001F300-\U0001F9FF\U00002600-\U000026FF\U00002700-\U000027BF]\s*$'
)
MIN_TRUNCATION_CHECK_LEN = 200
MAX_EMPTY_RETRIES = 3
CONTINUATION_PROMPT = (
    "Your previous response was incomplete. Please continue exactly from where you left off."
)
TOOL_EMPTY_NUDGE = (
    "You just made tool calls but did not provide a response. "
    "Please provide your response now."
)


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


def _detect_suspicious_stop(content: str, upstream_finish_reason: str) -> str:
    if upstream_finish_reason != "stop":
        return upstream_finish_reason
    if not content:
        return "stop"
    if not _looks_like_natural_ending(content):
        logger.warning(
            "Treating suspicious stop response as truncated (len=%d, ending=%r)",
            len(content),
            content[-20:],
        )
        return "length"
    return "stop"


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
    """Estimate token count using a simple heuristic.
    
    Uses character-type analysis:
    - Chinese/CJK chars ≈ 2 tokens each
    - English/code: ~4 chars per token
    - This is a rough approximation, not as accurate as tiktoken
    """
    if not text:
        return 0
    
    # Count Chinese/CJK characters (typically 2+ tokens each in real tokenizers)
    chinese_cjk = len(re.findall(r'[\u4e00-\u9fff\u3040-\u30ff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]', text))
    
    # Other characters (English, code, etc.) - ~4 chars per token
    other_chars = len(text) - chinese_cjk
    
    # Rough estimation
    estimated = int(chinese_cjk * 2 + other_chars / 4)
    return max(estimated, 1)  # At least 1 token


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
        for i in range(len(marker)):
            if lower_tail.endswith(marker[:i + 1]) and i < len(marker) - 1:
                return True
    return False


def _split_safe_detection_text(text):
    if not text:
        return "", ""

    marker_positions = [
        pos for marker in TOOL_MARKER_PATTERNS
        for pos in [text.lower().find(marker)]
        if pos != -1
    ]
    if marker_positions:
        split_at = min(marker_positions)
        return text[:split_at], text[split_at:]

    max_prefix = 0
    lower = text.lower()
    for marker in TOOL_MARKER_PATTERNS:
        for length in range(1, min(len(marker), len(lower))):
            if lower.endswith(marker[:length]):
                max_prefix = max(max_prefix, length)

    if max_prefix:
        return text[:-max_prefix], text[-max_prefix:]

    return text, ""


def _calculate_input_tokens(messages):
    """Calculate total input tokens from all messages."""
    total = 0
    for msg in messages:
        content = ""
        if isinstance(msg.get("content"), str):
            content = msg["content"]
        elif isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "text":
                    content += item.get("text", "")
                elif isinstance(item, str):
                    content += item
        total += estimate_text_tokens(content)
    return total


def stream_genai_response(chat_info, messages, model, max_tokens, config):
    started_at = time.monotonic()
    first_token_at = None
    content_parts = []
    reasoning_parts = []
    token = config.token_manager.get_token()
    root_ai_type = model_registry.get_root_ai_type(model, token)
    headers = build_genai_headers(token)
    normalized_messages = [normalize_message_content(msg) for msg in messages]

    # Calculate input tokens
    input_tokens = _calculate_input_tokens(normalized_messages)

    # Get model's max tokens from registry (the model's total context window)
    model_info = model_registry.get_models(token).get(model)
    model_max_total = getattr(model_info, 'max_tokens', None) if model_info else None

    # Fallback for common models if registry fetch fails
    MODEL_FALLBACK_LIMITS = {
        'chatglm': 128000,
        'gpt-4': 128000,
        'gpt-3.5': 16385,
        'claude': 200000,
        'deepseek': 64000,
        'MiniMax': 245000,
    }

    # Use larger default context window: 200000 (instead of 128000)
    LARGE_CONTEXT_DEFAULT = 200000

    if model_max_total and model_max_total > 0:
        # Model's total context window
        total_context = model_max_total
    else:
        # Try fallback based on model name
        fallback = None
        for key, val in MODEL_FALLBACK_LIMITS.items():
            if key.lower() in model.lower():
                fallback = val
                break
        total_context = fallback or LARGE_CONTEXT_DEFAULT
        if not fallback:
            logger.warning("Model %s has unknown context limit, using %d", model, total_context)

    # Calculate max output tokens:
    # maxToken = min(requested_max, total_context - input_tokens)
    # Reserve at least 1000 tokens for output
    requested_max = max_tokens or total_context
    max_output = min(requested_max, total_context - input_tokens)
    max_output = max(max_output, 1000)  # Minimum 1000 output tokens

    # Check if tools are being used (be more generous with output)
    has_tool_calls = any(
        msg.get('role') == 'assistant' and 'tool_calls' in msg
        for msg in messages
    )
    if has_tool_calls:
        # Tools might need more output for parameters
        max_output = max(max_output, 5000)
        logger.debug("Tool calls detected, increased min output to 5000")

    # Also cap at 95% of total context as safety (increased from 80%)
    max_output = min(max_output, int(total_context * 0.95))

    logger.info(
        "Token calc: input=%d, model_max=%s, total_context=%d, max_output=%d",
        input_tokens, model_max_total, total_context, max_output
    )

    genai_data = {
        "chatInfo": "",
        "messages": normalized_messages,
        "type": "3",
        "stream": True,
        "aiType": model,
        "aiSecType": "1",
        "promptTokens": input_tokens,
        "rootAiType": root_ai_type,
        "maxToken": max_output
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
            timeout=300
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
        upstream_finish_reason = "stop"
        for line in response.iter_lines():
            if finished:
                break

            if line:
                line_str = ""
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
                            fr = choice.get("finish_reason")
                            if fr is not None:
                                upstream_finish_reason = fr
                                finished = True

                        if finished:
                            full_content = "".join(content_parts)
                            effective_finish_reason = _detect_suspicious_stop(
                                full_content, upstream_finish_reason
                            )
                            if effective_finish_reason == "length":
                                logger.warning(
                                    "Response truncated (finish_reason='length') - model hit max output tokens"
                                )
                            log_stream_metrics(
                                model,
                                started_at,
                                first_token_at,
                                full_content,
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
                                    "finish_reason": effective_finish_reason
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

        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts)

        if not content_parts and not reasoning_parts:
            logger.warning("Empty response from model")

        effective_finish_reason = _detect_suspicious_stop(
            full_content, upstream_finish_reason
        )
        if effective_finish_reason == "length":
            logger.warning(
                "Response truncated (finish_reason='length') - model hit max output tokens"
            )

        log_stream_metrics(
            model,
            started_at,
            first_token_at,
            full_content,
            full_reasoning,
        )
        final_response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": effective_finish_reason
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
    upstream_finish_reason = "stop"

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

        if finish_reason:
            upstream_finish_reason = finish_reason

        if not content:
            # Handle finish_reason from upstream (e.g., "stop")
            if finish_reason:
                break
            continue

        if state == "detecting":
            buffer += content

            if _has_tool_marker(buffer):
                tool_detected = True
                state = "tool_buffering"
                logger.debug("Tool marker detected in detection window, switching to tool_buffering (buf=%d chars)", len(buffer))
                continue

            flushable_text, pending_tail = _split_safe_detection_text(buffer)

            if len(buffer) >= DETECTION_WINDOW and flushable_text:
                state = "streaming"
                logger.debug("No tool markers in %d chars, flushing as text, switching to streaming", len(buffer))
                role_chunk = _send_role()
                if role_chunk:
                    yield role_chunk
                yield _emit_text_chunk(completion_id, created, model, flushable_text)
                buffer = pending_tail
                continue

            if len(flushable_text) >= DETECTION_WINDOW:
                role_chunk = _send_role()
                if role_chunk:
                    yield role_chunk
                yield _emit_text_chunk(completion_id, created, model, flushable_text)
                buffer = pending_tail
                state = "streaming"
                continue

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

            # Try truncated parse (for incomplete tool calls)
            tool_calls, remaining = parse_truncated_tool_call(buffer, allowed_tool_names)
            if tool_calls:
                logger.warning("Parsed %d truncated tool_call(s) from buffer (%d chars) — content may be incomplete", len(tool_calls), len(buffer))
                role_chunk = _send_role()
                if role_chunk:
                    yield role_chunk
                for chunk in _emit_tool_call_chunks(completion_id, created, model, tool_calls, remaining):
                    yield chunk
                return  # Done — partial tool calls emitted

            # Not yet complete, keep buffering
            continue

    # Stream ended — handle remaining buffer
    if not buffer:
        effective_finish_reason = _detect_suspicious_stop("", upstream_finish_reason)
        if effective_finish_reason == "length":
            logger.warning("Response truncated (finish_reason='length') - model hit max output tokens")
        if not sent_role:
            logger.warning("Model returned empty after tool calls — emitting empty response")
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
                "finish_reason": effective_finish_reason
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

        # Try truncated parse (for incomplete tool calls)
        tool_calls, remaining = parse_truncated_tool_call(buffer, allowed_tool_names)
        if tool_calls:
            logger.warning("Final parse (truncated): %d tool_call(s) from buffer (%d chars) — content may be incomplete", len(tool_calls), len(buffer))
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

    effective_finish_reason = _detect_suspicious_stop(buffer, upstream_finish_reason)
    if effective_finish_reason == "length":
        logger.warning(
            "Response truncated (finish_reason='length') - model hit max output tokens"
        )

    stop_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": effective_finish_reason
        }]
    }
    yield f"data: {json.dumps(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"