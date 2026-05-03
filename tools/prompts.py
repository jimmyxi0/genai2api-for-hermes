import json


TOOL_SYSTEM_PROMPT = """\
You have access to the following tools:

<tools>
{tool_definitions}
</tools>

CRITICAL: When you need to call a tool, you MUST use ONLY the following XML format. Do NOT use any other format.

<tool_calls>
<invoke name="FUNC_NAME">
<parameter name="KEY">VALUE</parameter>
</invoke>
</tool_calls>

{tool_examples}

CRITICAL FORMAT RULES:
1. You MUST use <tool_calls><invoke name="TOOL_NAME"><parameter name="KEY">VALUE</parameter></invoke></tool_calls>
2. NEVER use bare tool names like: Bash command=ls
3. NEVER use formats like: terminal〉command〉ls (fullwidth brackets)
4. NEVER use JSON-only tool calls without XML wrapper
5. NEVER wrap tool calls in markdown code blocks like ```xml or ```json
6. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
7. If you made tool calls and received results, you MUST respond with either more tool calls or a final answer."""


TOOL_CHOICE_REQUIRED_PROMPT = "\nYou MUST call at least one tool in your response. Do NOT respond with plain text only."
TOOL_CHOICE_SPECIFIC_PROMPT = (
    '\nYou MUST call the tool named "{name}" in your response.'
)

COMMON_TOOL_EXAMPLES = {
    "Bash": {"command": "pwd"},
    "Read": {"file_path": "/absolute/path/to/file"},
    "Glob": {"pattern": "**/*.py", "path": "/absolute/project/path"},
    "Grep": {
        "pattern": "search text",
        "path": "/absolute/project/path",
        "output_mode": "content",
    },
    "LS": {"path": "/absolute/project/path"},
}


def flatten_message_content(content):
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = [flatten_message_content(item) for item in content]
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        if "content" in content:
            return flatten_message_content(content["content"])
        if "input" in content:
            return json.dumps(content["input"], ensure_ascii=False)
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def normalize_message_content(message):
    normalized = dict(message)
    normalized["content"] = flatten_message_content(message.get("content", ""))
    return normalized


def format_tool_definitions(tools):
    definitions = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        params = func.get("parameters", {})
        params_json = json.dumps(params, ensure_ascii=False, indent=2)
        definitions.append(
            f"<tool_definition\n"
            f"  <name>{func['name']}</name>\n"
            f"  <description>{func.get('description', '')}</description>\n"
            f"  <parameters>\n{params_json}\n  </parameters>\n"
            f"</tool_definition>"
        )
    return "\n".join(definitions)


def format_tool_examples(tools):
    examples = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        name = tool.get("function", {}).get("name")
        if name not in COMMON_TOOL_EXAMPLES:
            continue
        args = COMMON_TOOL_EXAMPLES[name]
        params = []
        for k, v in args.items():
            params.append(f'<parameter name="{k}">{v}</parameter>')
        examples.append(
            "<tool_calls>\n"
            f'<invoke name="{name}">\n'
            + "\n".join(params) + "\n"
            + "</invoke>\n"
            + "</tool_calls>"
        )

    if not examples:
        return ""
    return "Examples of valid tool calls:\n" + "\n".join(examples)


def inject_tool_prompt(messages, tools, tool_choice=None):
    tool_defs = format_tool_definitions(tools)
    tool_examples = format_tool_examples(tools)
    tool_prompt = TOOL_SYSTEM_PROMPT.format(
        tool_definitions=tool_defs,
        tool_examples=tool_examples,
    )

    if tool_choice == "required":
        tool_prompt += TOOL_CHOICE_REQUIRED_PROMPT
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = tool_choice["function"]["name"]
        tool_prompt += TOOL_CHOICE_SPECIFIC_PROMPT.format(name=name)

    new_messages = []
    has_system = False

    for msg in messages:
        role = msg.get("role")

        if role == "system":
            system_content = flatten_message_content(msg.get("content", ""))
            new_messages.append(
                {
                    "role": "system",
                    "content": system_content + "\n\n" + tool_prompt,
                }
            )
            has_system = True

        elif role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            tool_content = flatten_message_content(msg.get("content", ""))

            # Truncate tool results aggressively to prevent context overflow (max 1000 chars)
            MAX_TOOL_RESULT_LEN = 1000
            if len(tool_content) > MAX_TOOL_RESULT_LEN:
                tool_content = tool_content[:MAX_TOOL_RESULT_LEN] + f"\n... [truncated, {len(tool_content)} chars total]"

            new_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"<tool_result>\n"
                        f"  <tool_call_id>{tool_call_id}</tool_call_id>\n"
                        f"  <result>\n{tool_content}\n  </result>\n"
                        f"</tool_result>"
                    ),
                }
            )

        elif role == "assistant" and msg.get("tool_calls"):
            tc_text = flatten_message_content(msg.get("content")) or ""
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                params = []
                for k, v in args.items():
                    params.append(f'<parameter name="{k}">{json.dumps(v) if isinstance(v, (dict, list)) else v}</parameter>')
                tc_text += (
                    "\n<tool_calls>\n"
                    f'<invoke name="{name}">\n'
                    + "\n".join(params) + "\n"
                    + "</invoke>\n"
                    + "</tool_calls>"
                )
            new_messages.append({"role": "assistant", "content": tc_text.strip()})

        else:
            new_messages.append(normalize_message_content(msg))

    if not has_system:
        new_messages.insert(0, {"role": "system", "content": tool_prompt})

    return new_messages
