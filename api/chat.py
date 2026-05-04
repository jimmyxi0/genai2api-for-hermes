import json
import logging
import re
import time
import uuid
from datetime import datetime

from flask import Blueprint, current_app, request, jsonify, stream_with_context, Response

from errors import openai_error, make_error_chunk
from tools.prompts import inject_tool_prompt
from tools.parsing import extract_tool_calls
from provider.genai import (
    convert_messages_to_genai_format,
    estimate_text_tokens,
    estimate_messages_tokens,
    stream_genai_response,
    stream_genai_response_with_tools,
    MAX_EMPTY_RETRIES,
    CONTINUATION_PROMPT,
    TOOL_EMPTY_NUDGE,
)
from config import model_registry

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

# Monotonicity tracker: ensures prompt_tokens never decreases within a conversation.
# Keyed by conversation identifier derived from first user message + model + tools.
_last_prompt_tokens = {}


def _conversation_key(messages, model, tools=None):
    """Derive a stable conversation key from the first user message + model + tools.
    
    Tools are included in the key because they affect prompt_tokens estimation.
    Without this, changing tools between turns would break monotonicity tracking.
    """
    # Build a tools signature from function names only (stable, order-independent)
    tools_sig = ""
    if tools:
        tool_names = sorted([
            t.get("function", {}).get("name", "")
            for t in tools
            if t.get("type") == "function" and t.get("function", {}).get("name")
        ])
        tools_sig = ":" + ",".join(tool_names)
    
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return f"{model}:{content[:80]}{tools_sig}"
    return f"{model}:empty{tools_sig}"

MID_SENTENCE_ENDINGS = re.compile(
    r'[.。!！?？\n:：;；\)）\]】』》""…\u2026\w]$'
)


def get_model_max_tokens(model: str, config) -> int | None:
    try:
        token = config.token_manager.get_token()
        models_info = model_registry.get_models(token)
        model_info = models_info.get(model)
        if model_info and model_info.max_tokens:
            return model_info.max_tokens
    except Exception as e:
        logger.warning("Failed to fetch model max_tokens: %s", e)

    model_lower = model.lower()
    if "deepseek" in model_lower:
        return 64000
    if "glm" in model_lower:
        return 8192
    if "gpt-4" in model_lower:
        return 8192
    if "gpt-3.5" in model_lower:
        return 4096
    if "claude" in model_lower:
        return 8192
    return 8192


def _parse_chunks(chunks):
    has_content = False
    has_reasoning = False
    has_tool_calls = False
    finish_reason = "stop"
    content_parts = []

    for chunk in chunks:
        if not chunk.startswith('data: '):
            continue
        try:
            data_str = chunk[6:].strip()
            if data_str == '[DONE]':
                continue
            data = json.loads(data_str)
            if 'choices' in data and data['choices']:
                delta = data['choices'][0].get('delta', {})
                content = delta.get('content', '')
                reasoning = delta.get('reasoning_content', '')
                tool_calls = delta.get('tool_calls')
                if content:
                    has_content = True
                    content_parts.append(content)
                if reasoning:
                    has_reasoning = True
                if tool_calls:
                    has_tool_calls = True
                fr = data['choices'][0].get('finish_reason')
                if fr:
                    finish_reason = fr
        except (json.JSONDecodeError, IndexError, KeyError):
            pass

    if not (has_content or has_reasoning or has_tool_calls) and chunks:
        logger.debug("_parse_chunks: no content, first chunk sample: %s", chunks[0][:200])

    return has_content or has_reasoning or has_tool_calls, finish_reason, "".join(content_parts)


def _looks_like_mid_sentence(text):
    if not text or len(text) < 50:
        return False
    stripped = text.rstrip()
    if not stripped:
        return False
    return MID_SENTENCE_ENDINGS.search(stripped) is None


def _count_nudge_messages(messages):
    count = 0
    for msg in messages:
        content = msg.get('content', '')
        if content in (TOOL_EMPTY_NUDGE, CONTINUATION_PROMPT):
            count += 1
    return count


def _stream_with_retry(chat_info, messages, model, max_tokens, config, has_tools, allowed_tool_names, max_retries):
    total_retries = 0
    chunks = []
    current_messages = list(messages)

    while total_retries <= max_retries:
        if has_tools:
            gen = stream_genai_response_with_tools(
                chat_info, current_messages, model, max_tokens, config, allowed_tool_names
            )
        else:
            gen = stream_genai_response(
                chat_info, current_messages, model, max_tokens, config
            )

        chunks = list(gen)
        has_content, finish_reason, all_content = _parse_chunks(chunks)

        if has_content and finish_reason != "length":
            for chunk in chunks:
                yield chunk
            return

        if has_content and finish_reason == "length":
            total_retries += 1
            logger.warning(
                "Streaming response truncated (finish_reason='length') — auto-continuing (%d/%d)",
                total_retries, max_retries
            )
            if total_retries > max_retries:
                logger.warning("Max continuation retries reached, returning partial")
                for chunk in chunks:
                    yield chunk
                return
            continuation_msg = f"{CONTINUATION_PROMPT}\n\n[Partial response]:\n{all_content}"
            current_messages.append({"role": "user", "content": continuation_msg})
            continue

        if not has_content:
            total_retries += 1
            if total_retries > max_retries:
                logger.error(
                    "Model returned no content after %d retries — attempting clean reset",
                    max_retries,
                )
                logger.warning("Attempting clean reset (without trimming)")
                current_messages.append({"role": "user", "content": "Please respond to the tool results above with your analysis or next action."})

                if has_tools:
                    gen = stream_genai_response_with_tools(
                        chat_info, current_messages, model, max_tokens, config, allowed_tool_names
                    )
                else:
                    gen = stream_genai_response(
                        chat_info, current_messages, model, max_tokens, config
                    )
                reset_chunks = list(gen)
                reset_has, reset_fr, _ = _parse_chunks(reset_chunks)
                if reset_has:
                    for chunk in reset_chunks:
                        yield chunk
                    return

                logger.error("Clean reset failed. Returning empty response.")
                yield make_error_chunk("Model returned empty response", model)
                return

            logger.warning("Empty streaming response — retrying (%d/%d)", total_retries, max_retries)
            nudge = TOOL_EMPTY_NUDGE if has_tools else CONTINUATION_PROMPT
            current_messages.append({"role": "user", "content": nudge})
            continue

    logger.error("Exited retry loop unexpectedly. Returning what we have.")
    for chunk in chunks:
        yield chunk


@chat_bp.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    config = current_app.config["APP_CONFIG"]
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    start_time = time.monotonic()

    try:
        req_data = request.get_json()
        if not req_data or 'messages' not in req_data:
            return openai_error("Missing 'messages' field in request body")

        messages = req_data.get('messages', [])
        model = req_data.get('model', 'gpt-3.5-turbo')
        stream = req_data.get('stream', False)
        max_tokens = req_data.get('max_tokens', None)
        tools = req_data.get('tools', None)
        tool_choice = req_data.get('tool_choice', None)

        # Optional warning for very long conversations (no truncation)
        if len(messages) > 50:
            logger.warning(
                "[%s] Conversation has %d messages. Very long contexts may hit token limits.",
                request_id, len(messages)
            )

        # ---- dynamic max_tokens limit ----
        model_max_limit = get_model_max_tokens(model, config)
        if max_tokens is None:
            max_tokens = model_max_limit
        else:
            if model_max_limit is not None and max_tokens > model_max_limit:
                logger.warning(
                    "Requested max_tokens=%d exceeds model limit %d for %s, reducing.",
                    max_tokens, model_max_limit, model
                )
                max_tokens = model_max_limit
        max_tokens = max(1, max_tokens)

        has_tools = tools and len(tools) > 0
        allowed_tool_names = {
            tool["function"]["name"]
            for tool in (tools or [])
            if tool.get("type") == "function" and tool.get("function", {}).get("name")
        }

        tool_choice_mode = None
        if tool_choice == "none":
            tool_choice_mode = "none"
        elif tool_choice == "required":
            tool_choice_mode = "required"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            tool_choice_mode = "required"
        else:
            tool_choice_mode = "auto"

        logger.info(
            "[%s] model=%s stream=%s tools=%s messages=%d max_tokens=%d tool_choice_mode=%s",
            request_id, model, stream, bool(has_tools), len(messages), max_tokens, tool_choice_mode
        )

        if has_tools:
            messages = inject_tool_prompt(messages, tools, tool_choice)

        chat_info = convert_messages_to_genai_format(messages)
        if not chat_info:
            return openai_error("No user message found in 'messages'")

        if stream:
            return Response(
                stream_with_context(_stream_with_retry(
                    chat_info, messages, model, max_tokens, config,
                    has_tools, allowed_tool_names, MAX_EMPTY_RETRIES
                )),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                }
            )
        else:
            complete_content = ""
            finish_reason = "stop"
            total_retries = 0
            current_messages = list(messages)

            while total_retries <= MAX_EMPTY_RETRIES:
                partial_content = ""
                partial_finish = "stop"
                for line in stream_genai_response(chat_info, current_messages, model, max_tokens, config):
                    if line.startswith('data: '):
                        data_str = line[6:].strip()
                        if data_str == '[DONE]':
                            continue
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    partial_content += content
                                fr = data['choices'][0].get('finish_reason')
                                if fr:
                                    partial_finish = fr
                        except json.JSONDecodeError:
                            pass

                if partial_content.strip():
                    complete_content += partial_content
                    if partial_finish == "length":
                        total_retries += 1
                        if total_retries <= MAX_EMPTY_RETRIES:
                            logger.warning(
                                "Response truncated (finish_reason='length') - requesting continuation (%d/%d)",
                                total_retries, MAX_EMPTY_RETRIES
                            )
                            current_messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                            continue
                        else:
                            logger.warning("Response truncated - max retries reached, returning partial")
                    finish_reason = partial_finish
                    break

                total_retries += 1
                if total_retries > MAX_EMPTY_RETRIES:
                    logger.warning("Non-streaming: attempting clean reset (without trimming)")
                    current_messages.append({"role": "user", "content": "Please respond to the tool results above."})
                    reset_content = ""
                    reset_finish = "stop"
                    for line in stream_genai_response(chat_info, current_messages, model, max_tokens, config):
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            if data_str == '[DONE]':
                                continue
                            try:
                                data = json.loads(data_str)
                                if 'choices' in data and data['choices']:
                                    delta = data['choices'][0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content:
                                        reset_content += content
                                    fr = data['choices'][0].get('finish_reason')
                                    if fr:
                                        reset_finish = fr
                            except json.JSONDecodeError:
                                pass
                    if reset_content.strip():
                        complete_content += reset_content
                        finish_reason = reset_finish
                        break
                    logger.error("Model returned no content after all retries.")
                    return openai_error(
                        "Model returned empty response after multiple retries",
                        code="empty_response",
                        status=502,
                    )

                logger.warning("Empty response from model — retrying (%d/%d)", total_retries, MAX_EMPTY_RETRIES)
                nudge = TOOL_EMPTY_NUDGE if has_tools else CONTINUATION_PROMPT
                current_messages.append({"role": "user", "content": nudge})

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if has_tools:
                tool_calls, remaining_text = extract_tool_calls(
                    complete_content,
                    allowed_tool_names=allowed_tool_names,
                    tool_choice_mode=tool_choice_mode
                )
            else:
                tool_calls, remaining_text = None, complete_content

            if tool_calls:
                message_obj = {
                    "role": "assistant",
                    "content": remaining_text,
                    "tool_calls": tool_calls
                }
                finish_reason = "tool_calls"
            else:
                message_obj = {
                    "role": "assistant",
                    "content": complete_content
                }

            # Compute prompt tokens from ORIGINAL messages with proper estimation
            # (not from json.dumps which double-counts structural characters)
            prompt_tokens = estimate_messages_tokens(messages, tools=tools)

            # Monotonicity: prompt_tokens should never decrease within a conversation
            conv_key = _conversation_key(messages, model, tools)
            if conv_key in _last_prompt_tokens:
                prompt_tokens = max(prompt_tokens, _last_prompt_tokens[conv_key])
            _last_prompt_tokens[conv_key] = prompt_tokens

            # Prune old conversation keys to avoid memory leak
            if len(_last_prompt_tokens) > 1000:
                keys = list(_last_prompt_tokens.keys())
                for k in keys[:500]:
                    del _last_prompt_tokens[k]

            completion_tokens = estimate_text_tokens(complete_content)

            response = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": message_obj,
                    "finish_reason": finish_reason
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
            return jsonify(response)

    except Exception as e:
        logger.exception("[%s] Unhandled error", request_id)
        return openai_error(
            str(e),
            error_type="server_error",
            code="internal_error",
            status=500
        )
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("[%s] completed in %.2fs", request_id, elapsed)
