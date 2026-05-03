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
    stream_genai_response,
    stream_genai_response_with_tools,
    MAX_EMPTY_RETRIES,
    CONTINUATION_PROMPT,
    TOOL_EMPTY_NUDGE,
)

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)

MID_SENTENCE_ENDINGS = re.compile(
    r'[.。!！?？\n:：;；\)）\]】』》""…\u2026\w]$'
)


def _parse_chunks(chunks):
    """Parse SSE chunks and return (has_any_content, finish_reason, all_content)."""
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

    return has_content or has_reasoning or has_tool_calls, finish_reason, "".join(content_parts)


def _looks_like_mid_sentence(text):
    if not text or len(text) < 50:
        return False
    stripped = text.rstrip()
    if not stripped:
        return False
    return MID_SENTENCE_ENDINGS.search(stripped) is None


def _proactively_inject_nudge(messages, has_tools):
    """Detect truncated conversation state and inject a continuation nudge.

    A truncated state occurs when:
    - Assistant made tool calls
    - User provided tool results
    - But assistant has no text response after the tool calls
    """
    if not has_tools or len(messages) < 2:
        return messages

    last_assistant_has_tools = False
    has_user_after_tool_assistant = False
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if '<tool_call' in content or '<invoke' in content:
                last_assistant_has_tools = True
                has_user_after_tool_assistant = False
        elif msg.get('role') == 'user' and last_assistant_has_tools:
            has_user_after_tool_assistant = True

    if last_assistant_has_tools and has_user_after_tool_assistant:
        last_msg = messages[-1]
        if last_msg.get('role') == 'user':
            last_content = last_msg.get('content', '')
            is_tool_result = (
                '<tool_result' in last_content
                or '<result>' in last_content
                or '[tool_result' in last_content
            )
            if is_tool_result:
                logger.warning("Detected truncated conversation state — injecting proactive nudge")
                messages = list(messages)
                messages.append({"role": "user", "content": TOOL_EMPTY_NUDGE})
                return messages

    return messages


def _count_nudge_messages(messages):
    """Count how many nudge/continuation messages are in the conversation."""
    count = 0
    for msg in messages:
        content = msg.get('content', '')
        if content in (TOOL_EMPTY_NUDGE, CONTINUATION_PROMPT):
            count += 1
    return count


def _trim_conversation(messages, keep_last_n=6):
    """Trim conversation to essentials: system message + last N messages."""
    system_msgs = [msg for msg in messages if msg.get('role') == 'system']
    other_msgs = [msg for msg in messages if msg.get('role') != 'system']

    # Remove nudge/continuation messages
    filtered = [
        msg for msg in other_msgs
        if msg.get('content') not in (TOOL_EMPTY_NUDGE, CONTINUATION_PROMPT)
    ]

    # Keep last N messages
    if len(filtered) > keep_last_n:
        filtered = filtered[-keep_last_n:]

    return system_msgs + filtered


def _stream_with_retry(chat_info, messages, model, max_tokens, config, has_tools, allowed_tool_names, max_retries):
    """Buffer streaming response, retry with nudge if empty or truncated.

    Handles two failure modes:
    1. Empty response → retry with continuation nudge
    2. Truncated response (finish_reason='length') → auto-continue

    Uses a combined retry counter to prevent infinite loops.
    Trims conversation history on retries to prevent context overflow.
    """
    total_retries = 0
    chunks = []  # Initialize to avoid unbound variable warning
    current_messages = list(messages)
    # Only inject nudge on first attempt if needed
    current_messages = _proactively_inject_nudge(current_messages, has_tools)

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
            current_messages = _trim_conversation(current_messages, keep_last_n=8)
            current_messages.append({"role": "user", "content": continuation_msg})
            continue

        if not has_content:
            total_retries += 1
            if total_retries > max_retries:
                logger.error(
                    "Model returned no content after %d retries — attempting clean reset",
                    max_retries,
                )
                # Try clean reset — remove all nudge messages and trim
                clean_messages = _trim_conversation(current_messages, keep_last_n=4)
                if clean_messages == current_messages:
                    # Already clean, give up
                    logger.error("All retries failed. Returning empty response.")
                    yield make_error_chunk("Model returned empty response after retries", model)
                    return

                logger.warning("Attempting clean reset with trimmed history")
                current_messages = clean_messages
                current_messages.append({"role": "user", "content": "Please respond to the tool results above with your analysis or next action."})

                # One final attempt with clean history
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
            # Trim conversation on retry to prevent infinite growth
            if total_retries == 1:
                # First retry: just add nudge
                nudge = TOOL_EMPTY_NUDGE if has_tools else CONTINUATION_PROMPT
                current_messages.append({"role": "user", "content": nudge})
            else:
                # Subsequent retries: trim and add nudge
                current_messages = _trim_conversation(current_messages, keep_last_n=6)
                nudge = TOOL_EMPTY_NUDGE if has_tools else CONTINUATION_PROMPT
                current_messages.append({"role": "user", "content": nudge})
            continue

    # Should not reach here, but just in case
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
        max_tokens = req_data.get('max_tokens', 128000)
        tools = req_data.get('tools', None)
        tool_choice = req_data.get('tool_choice', None)

        has_tools = tools and len(tools) > 0
        allowed_tool_names = {
            tool["function"]["name"]
            for tool in (tools or [])
            if tool.get("type") == "function" and tool.get("function", {}).get("name")
        }

        logger.info("[%s] model=%s stream=%s tools=%s messages=%d",
                     request_id, model, stream, bool(has_tools), len(messages))

        # Proactive context management: trim if conversation is too long
        if len(messages) > 20:
            logger.warning(
                "[%s] Long conversation detected (%d messages) - trimming proactively",
                request_id, len(messages)
            )
            messages = _trim_conversation(messages, keep_last_n=18)

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
            current_messages = _proactively_inject_nudge(current_messages, has_tools)

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
                            # Trim conversation and add continuation prompt
                            current_messages = _trim_conversation(current_messages, keep_last_n=8)
                            current_messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                            continue
                        else:
                            logger.warning(
                                "Response truncated (finish_reason='length') - max retries reached, returning partial"
                            )
                    finish_reason = partial_finish
                    break

                total_retries += 1
                if total_retries > MAX_EMPTY_RETRIES:
                    # Try clean reset with trimmed history
                    clean_messages = _trim_conversation(current_messages, keep_last_n=4)
                    if clean_messages == current_messages:
                        # Already clean, give up
                        logger.error(
                            "Model returned no content after all retries. No fallback providers configured."
                        )
                        return openai_error(
                            "Model returned empty response after multiple retries",
                            code="empty_response",
                            status=502,
                        )

                    logger.warning("Non-streaming: attempting clean reset with trimmed history")
                    current_messages = clean_messages
                    current_messages.append({"role": "user", "content": "Please respond to the tool results above."})
                    # One final attempt
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

                    logger.error(
                        "Model returned no content after all retries. No fallback providers configured."
                    )
                    return openai_error(
                        "Model returned empty response after multiple retries",
                        code="empty_response",
                        status=502,
                    )

                logger.warning("Empty response from model — retrying (%d/%d)", total_retries, MAX_EMPTY_RETRIES)
                # Trim conversation on retry to prevent infinite growth
                if total_retries > 1:
                    current_messages = _trim_conversation(current_messages, keep_last_n=6)
                nudge = TOOL_EMPTY_NUDGE if has_tools else CONTINUATION_PROMPT
                current_messages.append({"role": "user", "content": nudge})

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

            if has_tools:
                tool_calls, remaining_text = extract_tool_calls(
                    complete_content,
                    allowed_tool_names=allowed_tool_names,
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
                    "prompt_tokens": 0,
                    "completion_tokens": estimate_text_tokens(complete_content),
                    "total_tokens": estimate_text_tokens(complete_content)
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
