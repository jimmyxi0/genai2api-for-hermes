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
from provider.chat_group import default_manager as chat_group_manager

logger = logging.getLogger(__name__)
# Token estimation regex - balanced for accuracy
# Chinese chars: each = ~1 token
# ASCII: split into word-like chunks (not entire runs)
# This matches: individual Chinese chars, then ASCII words/symbols
TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z][A-Za-z0-9_]*|[0-9]+|[^\s]")

# Safety multiplier for token estimation
# Our regex approximates real tokenizers:
#   - Chinese: 1 char = 1 token (accurate)
#   - English: ~1 word = 1 token (close to cl100k_base)
# Using 1.3x as safety buffer for edge cases
TOKEN_ESTIMATE_SAFETY_MULTIPLIER = 1.3

MAX_EMPTY_RETRIES = 10
CONTINUATION_PROMPT = "Please continue from where you left off. Do not repeat previous content."
TOOL_EMPTY_NUDGE = "You did not call any tool. Please call at least one tool from the provided list."


def sanitize_json_string(s: str) -> str:
    """
    Ensure a string is safe for JSON serialization.
    Removes control characters and ensures quotes/backslashes are escaped.
    """
    if not isinstance(s, str):
        s = str(s)
    # Remove control characters except newline/tab
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)
    # Escape backslashes and double quotes (json.dumps will do this, but we do it early to catch problems)
    # Use json.dumps to get a properly escaped JSON string literal, then strip the outer quotes.
    # This ensures any embedded special characters are escaped.
    try:
        # Wrap in quotes and parse to test validity
        json.loads(json.dumps(s))
        return s
    except Exception:
        # Fallback: force ASCII escape
        return s.encode('unicode_escape').decode('ascii')


def safe_normalize_message(msg: dict) -> dict:
    """Normalize and aggressively sanitize a single message."""
    normalized = dict(msg)
    content = flatten_message_content(msg.get("content", ""))
    if isinstance(content, str):
        normalized["content"] = sanitize_json_string(content)
    else:
        normalized["content"] = sanitize_json_string(str(content))
    return normalized


def validate_payload(payload: dict) -> tuple[dict, bool]:
    """
    Attempt to serialize the payload to JSON.
    If it fails, recursively find and replace offending message content.
    Returns (fixed_payload, whether it was fixed).
    """
    def check_and_fix(obj, path=""):
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                new_path = f"{path}.{k}" if path else k
                if isinstance(v, str):
                    # Test if this string can be JSON serialized
                    try:
                        json.dumps(v)
                    except (TypeError, ValueError):
                        logger.warning(f"Invalid JSON string at {new_path}: {v[:100]}... replacing with placeholder")
                        obj[k] = "[Invalid content replaced by proxy]"
                        return True
                else:
                    if check_and_fix(v, new_path):
                        return True
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if check_and_fix(item, f"{path}[{i}]"):
                    return True
        return False

    fixed = False
    try:
        json.dumps(payload)
        return payload, False
    except Exception as e:
        logger.warning(f"Payload JSON serialization failed: {e}. Attempting to fix.")
        fixed = check_and_fix(payload)
        if fixed:
            # Verify again after fix
            try:
                json.dumps(payload)
                return payload, True
            except Exception as e2:
                logger.error(f"Payload still invalid after fix: {e2}. Giving up.")
                return None, False
        else:
            logger.error("Could not locate invalid part in payload.")
            return None, False


def convert_messages_to_genai_format(messages):
    """Legacy compatibility: returns the last user message as a string."""
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
            content = delta.get("content") or delta.get("text") or None
            reasoning = delta.get("reasoning_content") or None
            if content is not None or reasoning is not None:
                return content, reasoning
        if "content" in response_data:
            return response_data["content"], None
        if response_data.get("success") and "result" in response_data:
            result = response_data["result"]
            if isinstance(result, dict) and "content" in result:
                return result["content"], None
            if isinstance(result, str):
                return result, None
    except (KeyError, IndexError, TypeError):
        pass
    return None, None


def estimate_text_tokens(text):
    if not text:
        return 0
    raw_count = len(TOKEN_PATTERN.findall(text))
    # Apply safety multiplier to account for heuristic undercounting
    return int(raw_count * TOKEN_ESTIMATE_SAFETY_MULTIPLIER)


def estimate_messages_tokens(messages):
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_text_tokens(part.get("text", ""))
                else:
                    total += estimate_text_tokens(str(part))
        elif isinstance(content, str):
            total += estimate_text_tokens(content)
        total += 4
    return total


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
        model, total_tokens, content_tokens, reasoning_tokens,
        total_tokens / total_elapsed, extra
    )


def stream_genai_response(chat_info, messages, model, max_tokens, config):
    started_at = time.monotonic()
    first_token_at = None
    content_parts = []
    reasoning_parts = []
    token = config.token_manager.get_token()
    root_ai_type = model_registry.get_root_ai_type(model, token)
    headers = build_genai_headers(token)

    # Sanitize all messages
    sanitized_messages = [safe_normalize_message(msg) for msg in messages]

    chat_group_id = chat_group_manager.get()
    genai_data = {
        "chatInfo": "",
        "messages": sanitized_messages,
        "type": "3",
        "stream": True,
        "aiType": model,
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": root_ai_type,
        "maxToken": max_tokens or 30000,
        "chatGroupId": chat_group_id,
    }

    # Validate and fix the entire payload before sending
    fixed_payload, was_fixed = validate_payload(genai_data)
    if fixed_payload is None:
        logger.error("Payload validation failed even after fixing. Aborting request.")
        yield make_error_chunk("Internal request serialization error", model)
        return
    if was_fixed:
        logger.warning("Payload was modified to be JSON‑safe. Some content may have been altered.")
    genai_data = fixed_payload

    logger.debug("=== GenAI Request ===")
    logger.debug("Model: %s, rootAiType: %s, chatGroupId: %s", model, root_ai_type, chat_group_id)
    logger.debug("Messages count: %d", len(sanitized_messages))

    try:
        response = requests.post(
            GENAI_URL,
            headers=headers,
            json=genai_data,
            stream=True,
            timeout=60
        )
        logger.debug("GenAI Response Status: %d", response.status_code)

        if response.status_code == 401:
            new_token = config.token_manager.force_refresh()
            if new_token:
                logger.info("Token refreshed after 401, retrying request")
                headers = build_genai_headers(new_token)
                response = requests.post(
                    GENAI_URL, headers=headers, json=genai_data,
                    stream=True, timeout=60
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
                    if not line_str:
                        continue

                    genai_json = json.loads(line_str)

                    if isinstance(genai_json, dict) and genai_json.get("success") is False:
                        err_msg = genai_json.get("message", "Unknown upstream error")
                        err_code = genai_json.get("code", 500)
                        logger.warning("GenAI business error (code=%s): %s", err_code, err_msg)
                        yield make_error_chunk(f"Upstream error: {err_msg}", model)
                        return

                    if "choices" in genai_json and len(genai_json["choices"]) > 0:
                        choice = genai_json["choices"][0]
                        finish_reason = choice.get("finish_reason")
                        if finish_reason is not None:
                            finished = True
                            logger.info("Upstream finish_reason=%s, total_chars=%d", finish_reason, len("".join(content_parts)))

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

                    if finished:
                        log_stream_metrics(
                            model, started_at, first_token_at,
                            "".join(content_parts), "".join(reasoning_parts)
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
                        prompt_tokens = estimate_messages_tokens(sanitized_messages)
                        completion_tokens = estimate_text_tokens("".join(content_parts)) + estimate_text_tokens("".join(reasoning_parts))
                        usage_only_chunk = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": model,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens
                            }
                        }
                        yield f"data: {json.dumps(usage_only_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                except json.JSONDecodeError as e:
                    logger.warning("JSON decode error in upstream stream chunk: %s, line: %s", e, line_str[:200])
                except Exception as e:
                    logger.exception("Unexpected error during stream parsing")

        if not emitted_done:
            log_stream_metrics(
                model, started_at, first_token_at,
                "".join(content_parts), "".join(reasoning_parts)
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
            prompt_tokens = estimate_messages_tokens(sanitized_messages)
            completion_tokens = estimate_text_tokens("".join(content_parts)) + estimate_text_tokens("".join(reasoning_parts))
            usage_only_chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            yield f"data: {json.dumps(usage_only_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as e:
        logger.exception("Error in stream_genai_response")
        yield make_error_chunk(str(e), model)


def stream_genai_response_with_tools(chat_info, messages, model, max_tokens, config, allowed_tool_names=None):
    """Same as before – unchanged."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(datetime.now().timestamp())
    OPEN_TAG = "<tool_call"
    buffer = ""
    tool_buffer = ""
    sent_role = False
    tool_detected = False
    accumulated_content = ""

    def make_chunk(delta, finish_reason=None):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason
            }]
        }
        return f"data: {json.dumps(chunk)}\n\n"

    def emit_text(text):
        nonlocal sent_role, accumulated_content
        accumulated_content += text
        delta = {"content": text}
        if not sent_role:
            delta["role"] = "assistant"
            sent_role = True
        return make_chunk(delta)

    def make_usage_only_chunk():
        prompt_tokens = estimate_messages_tokens(messages)
        completion_tokens = estimate_text_tokens(accumulated_content)
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    for line in stream_genai_response(chat_info, messages, model, max_tokens, config):
        if not line.startswith("data: "):
            continue
        try:
            data_str = line[6:].strip()
            if data_str == '[DONE]':
                break
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if "choices" not in data or not data["choices"]:
            continue
        chunk_delta = data["choices"][0].get("delta", {})
        content = chunk_delta.get("content", "")
        if not content:
            continue

        if tool_detected:
            tool_buffer += content
            continue

        buffer += content
        tag_pos = buffer.find(OPEN_TAG)
        if tag_pos >= 0:
            pre = buffer[:tag_pos]
            if pre.strip():
                yield emit_text(pre)
            tool_detected = True
            tool_buffer = buffer[tag_pos:]
            buffer = ""
            continue

        plen = _tag_prefix_len(buffer, OPEN_TAG)
        if plen > 0:
            safe = buffer[:-plen]
            if safe:
                yield emit_text(safe)
            buffer = buffer[-plen:]
        else:
            if buffer:
                yield emit_text(buffer)
            buffer = ""

    if tool_detected:
        tool_calls, remaining = extract_tool_calls(
            tool_buffer,
            allowed_tool_names=allowed_tool_names,
        )
        if tool_calls:
            logger.debug("Streaming tool calling: detected %d tool_call(s)", len(tool_calls))
            if remaining and remaining.strip():
                yield emit_text(remaining.strip())
            if not sent_role:
                yield make_chunk({"role": "assistant"})
                sent_role = True
            for i, tc in enumerate(tool_calls):
                yield make_chunk({
                    "tool_calls": [{
                        "index": i,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    }]
                })
            yield make_chunk({}, finish_reason="tool_calls")
            yield f"data: {json.dumps(make_usage_only_chunk())}\n\n"
            yield "data: [DONE]\n\n"
        else:
            logger.warning("Tool tag detected but parsing failed — emitting as text")
            yield emit_text(tool_buffer)
            yield make_chunk({}, finish_reason="stop")
            yield f"data: {json.dumps(make_usage_only_chunk())}\n\n"
            yield "data: [DONE]\n\n"
    else:
        if buffer:
            yield emit_text(buffer)
        if not sent_role:
            yield make_chunk({"role": "assistant", "content": ""})
        yield make_chunk({}, finish_reason="stop")
        yield f"data: {json.dumps(make_usage_only_chunk())}\n\n"
        yield "data: [DONE]\n\n"
