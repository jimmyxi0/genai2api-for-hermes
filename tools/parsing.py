import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)

COMMON_TOOL_ARG_KEYS = (
    "notebook_path",
    "new_string",
    "old_string",
    "file_path",
    "head_limit",
    "output_mode",
    "edit_mode",
    "description",
    "command",
    "pattern",
    "offset",
    "limit",
    "path",
)

COMMON_TOOL_NAMES = (
    "NotebookEdit",
    "TodoWrite",
    "MultiEdit",
    "WebFetch",
    "WebSearch",
    "Bash",
    "Read",
    "Glob",
    "Grep",
    "Edit",
    "Write",
    "LS",
)

TOOL_ARG_KEYS_BY_TOOL = {
    "Bash": ("command", "description"),
    "Read": ("file_path", "path", "offset", "limit"),
    "Glob": ("pattern", "path"),
    "Grep": ("pattern", "path", "output_mode", "head_limit"),
    "LS": ("path",),
}


def strip_think_blocks(content):
    return re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)


def normalize_dsml_tags(text):
    """Convert DSML tags to standard XML format."""
    if not text:
        return text
    text = re.sub(r'<\s*[｜|]DSML[｜|]\s*tool_calls\s*>', '<tool_calls>', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*[｜|]DSML[｜|]\s*/tool_calls\s*>', '</tool_calls>', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*[｜|]DSML[｜|]\s*invoke\s+name="([^"]+)"\s*>', r'<invoke name="\1">', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*[｜|]DSML[｜|]\s*/invoke\s*>', '</invoke>', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*[｜|]DSML[｜|]\s*parameter\s+name="([^"]+)"\s*>', r'<parameter name="\1">', text, flags=re.IGNORECASE)
    text = re.sub(r'<\s*[｜|]DSML[｜|]\s*/parameter\s*>', '</parameter>', text, flags=re.IGNORECASE)
    return text


def strip_fenced_code_blocks(text):
    if not text or '```' not in text:
        return text
    lines = text.split('\n')
    result = []
    in_fence = False
    fence_char = None
    for line in lines:
        if not in_fence:
            if line.startswith('```') or line.startswith('~~~'):
                in_fence = True
                fence_char = line[:3]
                continue
            result.append(line)
        else:
            if line.startswith(fence_char):
                in_fence = False
                fence_char = None
    return '\n'.join(result)


def parse_invoke_style_calls(text, allowed_tool_names=None):
    if not text or '<invoke' not in text:
        return [], None
    
    tool_calls = []
    spans = []
    wrapper_pattern = re.compile(r'<tool_calls[^>]*>(.*?)</tool_calls>', re.DOTALL | re.IGNORECASE)
    for wrapper_match in wrapper_pattern.finditer(text):
        spans.append((wrapper_match.start(), wrapper_match.end()))
    
    if not spans:
        invoke_pattern = re.compile(r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE)
        for match in invoke_pattern.finditer(text):
            tool_name = match.group(1).strip()
            body = match.group(2).strip()
            if allowed_tool_names and tool_name not in allowed_tool_names:
                continue
            args = {}
            param_pattern = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', re.DOTALL | re.IGNORECASE)
            for param_match in param_pattern.finditer(body):
                key = param_match.group(1).strip()
                value = param_match.group(2).strip()
                args[key] = value
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })
        return tool_calls, None
    
    for start, end in spans:
        block = text[start:end]
        invoke_pattern = re.compile(r'<invoke\s+name="([^"]+)"[^>]*>(.*?)</invoke>', re.DOTALL | re.IGNORECASE)
        for match in invoke_pattern.finditer(block):
            tool_name = match.group(1).strip()
            body = match.group(2).strip()
            if allowed_tool_names and tool_name not in allowed_tool_names:
                continue
            args = {}
            param_pattern = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)</parameter>', re.DOTALL | re.IGNORECASE)
            for param_match in param_pattern.finditer(body):
                key = param_match.group(1).strip()
                value = param_match.group(2).strip()
                args[key] = value
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })
    
    if spans:
        remaining_parts = []
        last_end = 0
        for start, end in spans:
            if start > last_end:
                remaining_parts.append(text[last_end:start])
            last_end = end
        if last_end < len(text):
            remaining_parts.append(text[last_end:])
        remaining = ' '.join(remaining_parts).strip()
    else:
        remaining = None
    
    return tool_calls, remaining


def _extract_json_object(raw):
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start:idx + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break
        start = raw.find("{", start + 1)
    return None


def _extract_balanced_brace_segment(text, start):
    if start < 0 or start >= len(text) or text[start] != "{":
        return None, None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1], idx + 1
    return None, None


def _sanitize_json_escapes(raw):
    return re.sub(r'\\(?=[^"\\/bfnrtu])', r'\\\\', raw)


def _load_relaxed_json(raw):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    sanitized = _sanitize_json_escapes(raw)
    if sanitized != raw:
        try:
            return json.loads(sanitized)
        except:
            pass
    stripped = sanitized.strip()
    if not stripped.startswith("{"):
        return None
    depth = 0
    in_string = False
    escape = False
    for ch in stripped:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
    if depth <= 0:
        return None
    try:
        return json.loads(stripped + ("}" * depth))
    except:
        return None


def _clean_arg_value(value):
    cleaned = value.strip()
    cleaned = re.sub(r'</?tool_call>\s*$', '', cleaned, flags=re.DOTALL).strip()
    cleaned = re.sub(r'</?arg_value>\s*$', '', cleaned, flags=re.DOTALL).strip()
    for suffix in ('"}})', '"}}', '"})', '"}', '">', '"'):
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)].rstrip()
            break
    return cleaned


def _clean_tool_name(name):
    cleaned = name.strip()
    match = re.search(r'[A-Za-z_][A-Za-z0-9_.-]*', cleaned)
    return match.group(0) if match else cleaned


def _extract_tool_name_from_attrs(attrs):
    if not attrs:
        return None
    match = re.search(r'\bname\s*=\s*["\']([^"\']+)["\']', attrs)
    return _clean_tool_name(match.group(1)) if match else None


def _normalize_tool_payload(payload, tool_name=None):
    if not isinstance(payload, dict):
        return None
    if "name" in payload:
        name = _clean_tool_name(payload["name"])
        arguments = payload.get("arguments", {})
        if name == "Bash" and isinstance(arguments, str):
            arguments = {"command": arguments}
        return {"name": name, "arguments": arguments}
    if len(payload) == 1:
        candidate_name, candidate_args = next(iter(payload.items()))
        cleaned_name = _clean_tool_name(candidate_name)
        if cleaned_name and isinstance(candidate_args, dict):
            return {"name": cleaned_name, "arguments": candidate_args}
    if tool_name:
        return {"name": tool_name, "arguments": payload}
    return None


def _parse_malformed_named_tool_payload(raw, tool_name=None):
    normalized = raw.strip()
    name = tool_name
    if not name:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', normalized)
        if name_match:
            name = _clean_tool_name(name_match.group(1))
    if not name or '"arguments"' not in normalized:
        return None
    command_match = re.search(r'"command"\s*:\s*"', normalized, re.DOTALL)
    if not command_match:
        return None
    tail = normalized[command_match.end():]
    value_match = re.match(r'(?s)(.*)"\s*}\s*}?\s*$', tail)
    if not value_match:
        return None
    command = value_match.group(1).replace('\\"', '"')
    return {"name": name, "arguments": {"command": command}}


def _parse_tagged_arguments(raw, tool_name=None):
    normalized = raw.replace('\\"', '"').strip()
    if '<arg_key>' not in normalized:
        return None
    name, remainder = normalized.split('<arg_key>', 1)
    name = _clean_tool_name(name)
    if not name:
        name = tool_name
    if not name:
        return None
    remainder = re.sub(
        r'</arg_value>\s*([A-Za-z0-9_]+)\s*(?=(?:":\s*"|":\s*|="\s*|=\s*|:\s*"|:\s*))',
        r'</arg_value><arg_key>\1',
        remainder,
    )
    remainder = re.sub(
        r'"\s*,\s*"([A-Za-z0-9_]+)"\s*:',
        r'"</arg_value><arg_key>\1":',
        remainder,
    )
    arg_pattern = re.compile(
        r'<arg_key>\s*"?([^"<:=]+)"?\s*'
        r'(?:</arg_key>\s*<arg_value>\s*|</arg_key>\s*"|">\s*|":\s*"|":\s*|="\s*|=\s*"|=\s*|:\s*"|:\s*)'
        r'(.*?)'
        r'(?=(?:</arg_value>\s*<arg_key>)|(?:</arg_value>)|(?:</parameter>)|(?:</invoke>)|(?:<arg_key>)|(?:</tool_call>)|$)',
        re.DOTALL,
    )
    arguments = {}
    for key, value in arg_pattern.findall('<arg_key>' + remainder):
        key = key.strip()
        if not key:
            continue
        arguments[key] = _clean_arg_value(value)
    return {"name": name, "arguments": arguments} if arguments else None


def _parse_broken_arg_key_lines(raw, tool_name=None):
    normalized = raw.replace('\\"', '"').strip()
    if '</arg_key>' not in normalized:
        return None
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if not lines:
        return None
    name = tool_name or _clean_tool_name(lines[0])
    if not name:
        return None
    arguments = {}
    for line in lines[1:]:
        line = re.sub(r'</?tool_call\b[^>]*>', '', line).strip()
        line = re.sub(r'</?arg_value>', '', line).strip()
        if not line:
            continue
        if '</arg_key>' in line:
            key, value = line.split('</arg_key>', 1)
            key = re.sub(r'</?arg_key>', '', key).strip().strip('"')
            if key and value:
                arguments[key] = _clean_arg_value(value)
            continue
        for key in COMMON_TOOL_ARG_KEYS:
            if line.startswith(key) and len(line) > len(key):
                value = line[len(key):]
                if value:
                    arguments[key] = _clean_arg_value(value)
                    break
    return {"name": name, "arguments": arguments} if arguments else None


def _extract_compact_tool_name(text, tool_name=None):
    if tool_name:
        return tool_name, text
    for candidate in COMMON_TOOL_NAMES:
        if text.startswith(candidate):
            return candidate, text[len(candidate):]
    return None, text


def _parse_compact_tool_arguments(raw, tool_name=None):
    normalized = raw.replace('\\"', '"').strip()
    normalized = re.sub(r'</?tool_call\b[^>]*>', '', normalized).strip()
    normalized = re.sub(r'</?arg_value>', '', normalized).strip()
    if not normalized:
        return None
    name, remainder = _extract_compact_tool_name(normalized, tool_name=tool_name)
    if not name or not remainder:
        return None
    remainder = remainder.strip()
    keys = TOOL_ARG_KEYS_BY_TOOL.get(name, COMMON_TOOL_ARG_KEYS)
    sorted_keys = sorted(keys, key=len, reverse=True)
    matches = []
    position = 0
    while position < len(remainder):
        matched_key = None
        for key in sorted_keys:
            if remainder.startswith(key, position):
                matched_key = key
                break
        if matched_key:
            matches.append((position, matched_key))
            position += len(matched_key)
            continue
        position += 1
    if not matches or matches[0][0] != 0:
        return None
    arguments = {}
    for index, (start, key) in enumerate(matches):
        value_start = start + len(key)
        value_end = matches[index + 1][0] if index + 1 < len(matches) else len(remainder)
        value = remainder[value_start:value_end].strip()
        if value:
            arguments[key] = _clean_arg_value(value)
    return {"name": name, "arguments": arguments} if arguments else None


def _parse_argument_tag(raw, tool_name=None):
    if not tool_name:
        return None
    match = re.search(r'<argument>\s*(.*?)\s*</argument>', raw, re.DOTALL)
    if not match:
        return None
    argument = match.group(1).strip()
    if not argument:
        return None
    key = "command" if tool_name == "Bash" else "argument"
    return {"name": tool_name, "arguments": {key: argument}}


def _parse_fullwidth_bracket_format(raw, tool_name=None):
    RIGHT_CORNER = '〉'
    if RIGHT_CORNER not in raw:
        return None, None
    parts_by_bracket = raw.split(RIGHT_CORNER)
    first_part = parts_by_bracket[0].strip() if parts_by_bracket else ""
    if not first_part:
        return None, None
    words = first_part.split()
    raw_name = words[-1] if words else first_part
    if not raw_name:
        return None, None
    tool_name_map = {
        'terminal': 'Bash', 'shell': 'Bash', 'sh': 'Bash',
        'read': 'Read', 'cat': 'Read', 'glob': 'Glob', 'ls': 'Glob',
        'grep': 'Grep', 'search': 'Grep', 'edit': 'Edit', 'write': 'Write',
    }
    name = tool_name or tool_name_map.get(raw_name.lower(), raw_name.capitalize())
    closing_tag = f"{RIGHT_CORNER}/{raw_name}{RIGHT_CORNER}"
    closing_pos = raw.find(closing_tag)
    if closing_pos < 0:
        closing_tag = f"{RIGHT_CORNER}/{name}{RIGHT_CORNER}"
        closing_pos = raw.find(closing_tag)
    first_sep = raw.find(RIGHT_CORNER)
    if closing_pos > first_sep + 1:
        body = raw[first_sep + 1:closing_pos]
    else:
        body = raw[first_sep + 1:]
    body_parts = body.split(RIGHT_CORNER)
    arguments = {}
    i = 0
    while i < len(body_parts) - 1:
        key = body_parts[i].strip()
        if not key:
            i += 1
            continue
        key = _clean_tool_name(key)
        if not key:
            i += 1
            continue
        value = body_parts[i + 1].strip() if i + 1 < len(body_parts) else ""
        arguments[key] = value
        i += 2
    if not arguments:
        return None, None
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }
    }
    return [tool_call], None


def _parse_tool_call_body(raw, tool_name=None):
    raw = raw.strip()
    normalized_raw = raw.replace('\\"', '"')
    call = _load_relaxed_json(raw)
    normalized_call = _normalize_tool_payload(call, tool_name=tool_name)
    if normalized_call:
        return normalized_call
    embedded_json = _extract_json_object(raw)
    normalized_call = _normalize_tool_payload(embedded_json, tool_name=tool_name)
    if normalized_call:
        return normalized_call
    embedded_json = _extract_json_object(normalized_raw)
    normalized_call = _normalize_tool_payload(embedded_json, tool_name=tool_name)
    if normalized_call:
        return normalized_call
    malformed = _parse_malformed_named_tool_payload(raw, tool_name=tool_name)
    if malformed:
        return malformed
    tagged = _parse_tagged_arguments(raw, tool_name=tool_name)
    if tagged:
        return tagged
    broken = _parse_broken_arg_key_lines(raw, tool_name=tool_name)
    if broken:
        return broken
    compact = _parse_compact_tool_arguments(raw, tool_name=tool_name)
    if compact:
        return compact
    arg_tag = _parse_argument_tag(raw, tool_name=tool_name)
    if arg_tag:
        return arg_tag
    inline_pairs = re.findall(r'([A-Za-z0-9_]+)"\s*:\s*"([^"]*)"', normalized_raw, re.DOTALL)
    if inline_pairs:
        name = _clean_tool_name(normalized_raw.split('<arg_key>', 1)[0].split('{', 1)[0].strip())
        if not name:
            name = tool_name
        if name and not name.startswith('"'):
            arguments = {key.strip(): value.strip() for key, value in inline_pairs if key.strip()}
            if arguments:
                return {"name": name, "arguments": arguments}
    name_m = re.search(r'<name>\s*(.*?)\s*</name>', raw, re.DOTALL)
    args_m = re.search(r'<arguments>\s*(.*?)\s*</arguments>', raw, re.DOTALL)
    if name_m:
        name = _clean_tool_name(name_m.group(1))
        arguments = {}
        if args_m:
            args_str = args_m.group(1).strip()
            try:
                arguments = json.loads(args_str)
            except:
                arguments = {"raw": args_str}
        return {"name": name, "arguments": arguments}
    fullwidth = _parse_fullwidth_bracket_format(raw, tool_name=tool_name)
    if fullwidth:
        calls_list, _ = fullwidth
        if calls_list:
            tc = calls_list[0]
            return {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
    return None


def _clean_remaining_text(text):
    if not text:
        return None
    cleaned = re.sub(r'<tool_result>.*?</tool_result>', '', text, flags=re.DOTALL)
    cleaned = re.sub(r'<tool_condition>.*?</tool_condition>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<agent-other-thinking>.*?</agent-other-thinking>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'</?tool_condition\b[^>]*>', '', cleaned)
    cleaned = re.sub(r'</?tool_call\b[^>]*>', '', cleaned)
    cleaned = re.sub(r'</?tool_calls\b[^>]*>', '', cleaned)
    cleaned = re.sub(r'</?tool_\d+>', '', cleaned)
    cleaned = cleaned.replace('</think>', '')
    cleaned = re.sub(r'^\s*\[\]\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip() or None


def _is_likely_truncated(text: str) -> bool:
    if not text:
        return False
    incomplete_patterns = [
        r'<invoke\s+name="[^"]*$',
        r'<parameter\s+name="[^"]*$',
        r'<tool_calls>\s*$',
        r'<invoke\s+name="[^"]+"\s*>$',
        r'<parameter\s+name="[^"]+"\s*>$',
    ]
    for pattern in incomplete_patterns:
        if re.search(pattern, text, re.MULTILINE):
            return True
    open_invoke = text.count('<invoke')
    close_invoke = text.count('</invoke>')
    if open_invoke > close_invoke:
        return True
    open_param = text.count('<parameter')
    close_param = text.count('</parameter>')
    if open_param > close_param:
        return True
    return False


def parse_truncated_tool_call(raw, allowed_tool_names=None):
    if not raw or ('<invoke' not in raw and '<tool_call' not in raw):
        return None, None
    name_match = re.search(r'<invoke\s+name="([^"]+)"', raw, re.IGNORECASE)
    if not name_match:
        name_match = re.search(r'<tool_call\s+name="([^"]+)"', raw, re.IGNORECASE)
    if not name_match:
        return None, None
    tool_name = _clean_tool_name(name_match.group(1))
    if not tool_name or (allowed_tool_names and tool_name not in allowed_tool_names):
        return None, None
    arguments = {}
    param_pattern = re.compile(r'<parameter\s+name="([^"]+)"[^>]*>(.*?)(?:</parameter>|$)', re.DOTALL | re.IGNORECASE)
    for pmatch in param_pattern.finditer(raw):
        key = _clean_tool_name(pmatch.group(1))
        value = pmatch.group(2).strip()
        if key:
            arguments[key] = value
    if not arguments and tool_name == "Bash" and '<parameter' not in raw:
        after_invoke = re.sub(r'^.*<invoke\s+name="[^"]+"\s*>', '', raw, flags=re.DOTALL)
        after_invoke = re.sub(r'</invoke>.*$', '', after_invoke, flags=re.DOTALL)
        if after_invoke.strip():
            arguments["command"] = after_invoke.strip()
    if not arguments:
        return None, None
    tool_call = {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }
    }
    return [tool_call], "[WARNING: Tool call was truncated - content may be incomplete]"


def _should_skip_fallback(allowed_tool_names, tool_choice_mode):
    """Only use fallback extractors when tool_choice_mode == 'required'."""
    if tool_choice_mode != "required":
        return True
    if not allowed_tool_names:
        return True
    return False


def _filter_by_allowed(tool_calls, allowed_tool_names):
    if not allowed_tool_names:
        return tool_calls
    return [tc for tc in tool_calls if tc["function"]["name"] in allowed_tool_names]


def _extract_function_style_calls(cleaned, allowed_tool_names):
    call_pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_.-]*)\(\s*{')
    matches = list(call_pattern.finditer(cleaned))
    if not matches:
        return [], None
    spans = []
    tool_calls = []
    for i, match in enumerate(matches):
        tool_name = _clean_tool_name(match.group(1))
        json_text, end_idx = _extract_balanced_brace_segment(cleaned, match.end() - 1)
        if not json_text:
            continue
        payload = _load_relaxed_json(json_text)
        if not isinstance(payload, dict):
            continue
        if allowed_tool_names and tool_name not in allowed_tool_names:
            continue
        span_end = end_idx
        if span_end is None:
            continue
        while span_end < len(cleaned) and cleaned[span_end].isspace():
            span_end += 1
        if span_end < len(cleaned) and cleaned[span_end] == ")":
            span_end += 1
        spans.append((match.start(), span_end))
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(payload, ensure_ascii=False),
            }
        })
    if not tool_calls:
        return [], None
    remaining_parts = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            remaining_parts.append(cleaned[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(cleaned):
        remaining_parts.append(cleaned[cursor:])
    return tool_calls, _clean_remaining_text("".join(remaining_parts))


def _run_fallback_extractors(cleaned, allowed_tool_names):
    for extractor in (
        _parse_fullwidth_bracket_format,
        _extract_edit_tool_calls,
        _extract_plain_text_tool_calls,
        _extract_loose_xml_tool_calls,
        _extract_mixed_format_calls,
        _extract_numbered_tool_calls,
        _extract_function_style_calls,
        _extract_bare_json_tool_calls,
        _extract_malformed_named_tool_calls,
        _extract_fenced_argument_calls,
    ):
        tool_calls, remaining = extractor(cleaned, allowed_tool_names)
        if tool_calls:
            return tool_calls, remaining
    return None, None


def _extract_bare_json_tool_calls(cleaned, allowed_tool_names):
    tool_calls = []
    spans = []
    idx = 0
    if not cleaned:
        return [], None
    while idx < len(cleaned):
        start = cleaned.find("{", idx)
        if start == -1:
            break
        json_text, end_idx = _extract_balanced_brace_segment(cleaned, start)
        if not json_text:
            idx = start + 1
            continue
        payload = _load_relaxed_json(json_text)
        call = _normalize_tool_payload(payload)
        if not call:
            idx = start + 1
            continue
        if allowed_tool_names and call["name"] not in allowed_tool_names:
            idx = end_idx
            continue
        spans.append((start, end_idx))
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            }
        })
        idx = end_idx
    if not tool_calls:
        return [], None
    remaining_parts = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            remaining_parts.append(cleaned[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(cleaned):
        remaining_parts.append(cleaned[cursor:])
    return tool_calls, _clean_remaining_text("".join(remaining_parts))


def _extract_loose_xml_tool_calls(cleaned, allowed_tool_names):
    if not cleaned:
        return [], None
    tool_calls = []
    spans = []
    loose_pattern = re.compile(r'<\s*(?:tool_call|invoke|tool_calls)\s+([^>]*?)>(.*?)</\s*(?:tool_call|invoke|tool_calls)\s*>', re.DOTALL | re.IGNORECASE)
    for match in loose_pattern.finditer(cleaned):
        attrs = match.group(1).strip()
        body = match.group(2).strip()
        tool_name = _extract_tool_name_from_attrs(attrs)
        if not tool_name:
            name_match = re.search(r'(?:name|function)\s*[=:]\s*["\']?([A-Za-z_][A-Za-z0-9_.-]*)', attrs, re.IGNORECASE)
            if name_match:
                tool_name = _clean_tool_name(name_match.group(1))
        if not tool_name or (allowed_tool_names and tool_name not in allowed_tool_names):
            continue
        args = {}
        param_pattern = re.compile(r'<\s*parameter\s+([^>]*?)\s*>(.*?)<\s*/\s*parameter\s*>', re.DOTALL | re.IGNORECASE)
        for pmatch in param_pattern.finditer(body):
            pattrs = pmatch.group(1).strip()
            pvalue = pmatch.group(2).strip()
            key_match = re.search(r'name\s*[=:]\s*["\']?([^"\'>\s]+)', pattrs, re.IGNORECASE)
            if key_match:
                args[key_match.group(1)] = pvalue
        if not args:
            json_match = re.search(r'\{.*\}', body, re.DOTALL)
            if json_match:
                try:
                    args = json.loads(json_match.group(0))
                except:
                    pass
        if args:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })
            spans.append((match.start(), match.end()))
    if not tool_calls:
        return [], None
    remaining_parts = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            remaining_parts.append(cleaned[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(cleaned):
        remaining_parts.append(cleaned[cursor:])
    return tool_calls, _clean_remaining_text("".join(remaining_parts))


def _extract_mixed_format_calls(cleaned, allowed_tool_names):
    if not cleaned:
        return [], None
    tool_calls = []
    mixed_pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_.-]*)\s+((?:[A-Za-z_][A-Za-z0-9_]*\s*[=:]\s*[^\n]+(?:\n|$))+)', re.MULTILINE)
    for match in mixed_pattern.finditer(cleaned):
        tool_name = _clean_tool_name(match.group(1))
        if not tool_name or (allowed_tool_names and tool_name not in allowed_tool_names):
            continue
        args_str = match.group(2)
        args = {}
        for kv_match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*([^\n]+)', args_str):
            key = kv_match.group(1).strip()
            value = kv_match.group(2).strip().strip('"\'')
            if key and value:
                args[key] = value
        if args:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })
    if not tool_calls:
        return [], None
    remaining = mixed_pattern.sub('', cleaned).strip()
    return tool_calls, _clean_remaining_text(remaining)


def _extract_fenced_argument_calls(cleaned, allowed_tool_names):
    if not allowed_tool_names or len(allowed_tool_names) != 1:
        return [], None
    tool_name = next(iter(allowed_tool_names))
    fence_pattern = re.compile(r'```(?:json)?\s*({.*?})\s*```', re.DOTALL)
    matches = list(fence_pattern.finditer(cleaned))
    if not matches:
        return [], None
    tool_calls = []
    for match in matches:
        payload = _load_relaxed_json(match.group(1))
        if isinstance(payload, dict) and "arguments" in payload:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(payload["arguments"], ensure_ascii=False),
                }
            })
    if not tool_calls:
        return [], None
    remaining = fence_pattern.sub('', cleaned).strip()
    return tool_calls, _clean_remaining_text(remaining)


def _extract_plain_text_tool_calls(cleaned, allowed_tool_names):
    if not cleaned:
        return [], None
    # If XML tags are present, skip plain text extraction entirely
    if '<tool' in cleaned or '<invoke' in cleaned:
        return [], None
    tool_name_map = {
        'bash': 'Bash', 'shell': 'Bash', 'sh': 'Bash',
        'read': 'Read', 'file': 'Read', 'cat': 'Read',
        'glob': 'Glob', 'ls': 'Glob', 'grep': 'Grep', 'search': 'Grep',
        'edit': 'Edit', 'modify': 'Edit', 'write': 'Write', 'save': 'Write',
        'webfetch': 'WebFetch', 'fetch': 'WebFetch',
    }
    tool_calls = []
    spans = []
    patterns = [
        re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s+(.+?)(?=\n|$|<tool|$)', re.MULTILINE | re.DOTALL),
        re.compile(r'\b([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*\s*=\s*"?[^"\s]+"?)', re.DOTALL),
    ]
    seen_positions = set()
    for pattern in patterns:
        for match in pattern.finditer(cleaned):
            if match.start() in seen_positions:
                continue
            seen_positions.add(match.start())
            raw_name = match.group(1).lower()
            args_str = match.group(2).strip()
            tool_name = tool_name_map.get(raw_name)
            if not tool_name:
                continue
            # Require '=' for non-Bash tools
            if '=' not in args_str and tool_name != 'Bash':
                continue
            args = {}
            if '=' in args_str:
                for kv_match in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"?([^"\s]+)"?', args_str):
                    key = kv_match.group(1).strip()
                    value = kv_match.group(2).strip()
                    if key and value:
                        key_map = {
                            'filePath': 'file_path', 'filepath': 'file_path', 'path': 'file_path',
                            'command': 'command', 'outputMode': 'output_mode', 'headLimit': 'head_limit',
                        }
                        key = key_map.get(key, key)
                        args[key] = value
            else:
                if tool_name == 'Bash':
                    args = {"command": args_str.strip().strip('"')}
                else:
                    continue
            if args and len(args_str) > 2:
                if allowed_tool_names and tool_name not in allowed_tool_names:
                    continue
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False)
                    }
                })
                spans.append((match.start(), match.end()))
    if not tool_calls:
        return [], None
    remaining_parts = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            remaining_parts.append(cleaned[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(cleaned):
        remaining_parts.append(cleaned[cursor:])
    return tool_calls, _clean_remaining_text("".join(remaining_parts))


def _extract_edit_tool_calls(cleaned, allowed_tool_names):
    if not cleaned:
        return [], None
    tool_calls = []
    spans = []
    normalized = re.sub(r'\\\s*\n', '\n', cleaned)
    tool_pattern = re.compile(r'\b(edit|write)\s*>', re.IGNORECASE)
    matches = list(tool_pattern.finditer(normalized))
    if not matches:
        tool_pattern2 = re.compile(r'\b(edit|write)\s*>\\\s*\n', re.IGNORECASE)
        matches = list(tool_pattern2.finditer(cleaned))
        if not matches:
            return [], None
    for idx, match in enumerate(matches):
        raw_tool = match.group(1).lower()
        tool_name = 'Edit' if raw_tool == 'edit' else 'Write'
        if allowed_tool_names and tool_name not in allowed_tool_names:
            continue
        start = match.end()
        if idx + 1 < len(matches):
            end = matches[idx + 1].start()
            full_args = normalized[start:end]
        else:
            full_args = normalized[start:]
        if not full_args.strip():
            continue
        args = {}
        param_pattern = re.compile(r'<parameter\s+name\s*=\s*"([^"]+)"\s*>(.*?)</parameter>', re.DOTALL | re.IGNORECASE)
        found_params = False
        for param_match in param_pattern.finditer(full_args):
            key = param_match.group(1).lower()
            value = param_match.group(2).strip()
            key_map = {'filepath': 'file_path', 'file': 'file_path', 'path': 'file_path',
                       'oldstring': 'oldString', 'newstring': 'newString'}
            key = key_map.get(key, key)
            args[key] = value
            found_params = True
        if not found_params:
            fp_match = re.search(r'filePath\s*=\s*"([^"]+)"', full_args, re.IGNORECASE)
            if not fp_match:
                fp_match = re.search(r'filePath\s*=\s*(\S+)', full_args, re.IGNORECASE)
            if fp_match:
                args['file_path'] = fp_match.group(1)
            content_start = re.search(r'content\s*=\s*"', full_args, re.IGNORECASE)
            if content_start:
                start_idx = content_start.end() - 1
                search_start = start_idx + 1
                while True:
                    quote_idx = full_args.find('"', search_start)
                    if quote_idx == -1:
                        args['content'] = full_args[start_idx+1:]
                        break
                    if quote_idx > 0 and full_args[quote_idx-1] == '\\':
                        search_start = quote_idx + 1
                        continue
                    args['content'] = full_args[start_idx+1:quote_idx]
                    break
            if tool_name == 'Edit':
                for key in ['oldString', 'newString']:
                    val_match = re.search(rf'{key}\s*=\s*"([^"]*)"', full_args, re.IGNORECASE)
                    if not val_match:
                        val_match = re.search(rf'{key}\s*=\s*(\S+)', full_args, re.IGNORECASE)
                    if val_match:
                        args[key] = val_match.group(1)
        valid = (tool_name == 'Edit' and 'oldString' in args and 'newString' in args and 'file_path' in args) or \
                (tool_name == 'Write' and 'file_path' in args and 'content' in args)
        if valid:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })
            spans.append((match.start(), match.end()))
    if not tool_calls:
        return [], None
    remaining_parts = []
    cursor = 0
    for start, end in sorted(spans):
        if start > cursor:
            remaining_parts.append(cleaned[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(cleaned):
        remaining_parts.append(cleaned[cursor:])
    return tool_calls, _clean_remaining_text("".join(remaining_parts))


def _extract_malformed_named_tool_calls(cleaned, allowed_tool_names):
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', cleaned)
    if not name_match:
        return [], None
    start = cleaned.rfind("{", 0, name_match.start())
    if start < 0:
        start = name_match.start()
    call = _parse_malformed_named_tool_payload(cleaned[start:])
    if not call or (allowed_tool_names and call["name"] not in allowed_tool_names):
        return [], None
    return [{
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": call["name"],
            "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
        }
    }], _clean_remaining_text(cleaned[:start])


def _extract_numbered_tool_calls(cleaned, allowed_tool_names):
    block_pattern = re.compile(r'<tool_\d+>\s*(.*?)\s*</tool_\d+>', re.DOTALL)
    matches = list(block_pattern.finditer(cleaned))
    if not matches:
        return [], None
    tool_calls = []
    for i, match in enumerate(matches):
        call = _parse_tool_call_body(match.group(1))
        if call and (not allowed_tool_names or call["name"] in allowed_tool_names):
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
                }
            })
    if not tool_calls:
        return [], None
    remaining = block_pattern.sub('', cleaned)
    remaining = re.sub(r'</?tool_calls>', '', remaining).strip()
    return tool_calls, _clean_remaining_text(remaining)


def extract_tool_calls(content, allowed_tool_names=None, tool_choice_mode=None):
    """
    Extract tool calls from model output.
    
    Args:
        content: Raw output text.
        allowed_tool_names: List of allowed tool names, or None.
        tool_choice_mode: 'none', 'auto', 'required' or None.
    """
    if tool_choice_mode == 'none':
        return None, content
    
    cleaned = strip_think_blocks(content)
    cleaned = normalize_dsml_tags(cleaned)
    
    # Truncation recovery
    if _is_likely_truncated(cleaned):
        truncated_calls, _ = parse_truncated_tool_call(cleaned, allowed_tool_names)
        if truncated_calls:
            logger.warning("Recovered truncated tool call(s) from incomplete response")
            return _filter_by_allowed(truncated_calls, allowed_tool_names), ""
    
    # Try invoke style first
    invoke_calls, invoke_remaining = parse_invoke_style_calls(cleaned, allowed_tool_names)
    if invoke_calls:
        logger.debug("Found %d invoke-style tool calls", len(invoke_calls))
        return _filter_by_allowed(invoke_calls, allowed_tool_names), invoke_remaining
    
    # Process <tool_call> tags
    has_multiple_tools = allowed_tool_names and len(allowed_tool_names) > 1
    if has_multiple_tools:
        cleaned = strip_fenced_code_blocks(cleaned)
    
    cleaned = re.sub(
        r'```(?:xml|json|plaintext|text)?\s*\n?\s*(<tool_call\b[^>]*>.*?</tool_call>)\s*\n?\s*```',
        r'\1',
        cleaned,
        flags=re.DOTALL
    )
    
    match_pattern = re.compile(r'<tool_call\b([^>]*)>\s*(.*?)\s*</tool_call>', re.DOTALL)
    open_pattern = re.compile(r'<tool_call\b([^>]*)>', re.DOTALL)
    matches = list(match_pattern.finditer(cleaned))
    remaining = None
    
    if not matches and '<tool_call' in cleaned:
        openings = list(open_pattern.finditer(cleaned))
        if openings:
            remaining = cleaned[:openings[0].start()].strip()
            matches = []
            for index, opening in enumerate(openings):
                end = openings[index + 1].start() if index + 1 < len(openings) else len(cleaned)
                body = cleaned[opening.end():end].strip()
                if body:
                    matches.append((opening.group(1), body))
    
    if matches:
        tool_calls = []
        for i, match in enumerate(matches):
            if isinstance(match, tuple):
                attrs, body = match
            else:
                attrs, body = match.group(1), match.group(2)
            tool_name = _extract_tool_name_from_attrs(attrs)
            call = _parse_tool_call_body(body, tool_name=tool_name)
            if call:
                if allowed_tool_names and call["name"] not in allowed_tool_names:
                    logger.warning("Skipping disallowed tool_call[%d] name=%s", i, call["name"])
                    continue
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)
                    }
                })
            else:
                logger.warning("Failed to parse tool_call[%d] — raw: %s", i, body[:300])
        if tool_calls:
            remaining_text = remaining if remaining is not None else match_pattern.sub('', cleaned).strip()
            return _filter_by_allowed(tool_calls, allowed_tool_names), _clean_remaining_text(remaining_text)
    
    # Fallback extractors only when required
    if _should_skip_fallback(allowed_tool_names, tool_choice_mode):
        logger.debug("Skipping fallback extractors (mode=%s)", tool_choice_mode)
        return None, content
    
    fallback_tool_calls, fallback_remaining = _run_fallback_extractors(cleaned, allowed_tool_names)
    if fallback_tool_calls:
        logger.debug("Used fallback extractor, found %d tool calls", len(fallback_tool_calls))
        return _filter_by_allowed(fallback_tool_calls, allowed_tool_names), fallback_remaining
    
    return None, content


def _tag_prefix_len(text, tag):
    max_len = min(len(tag) - 1, len(text))
    for length in range(max_len, 0, -1):
        if text[-length:] == tag[:length]:
            return length
    return 0
