#!/usr/bin/env python3
import re

with open('tools/prompts.py', 'r') as f:
    content = f.read()

# 1. Replace TOOL_SYSTEM_PROMPT
old_prompt = '''<tool_call>
{{"name": "<function-name>", "arguments": {{<arguments-as-json>}}}}
</tool_call>

{tool_examples}

Rules:
1. You can call multiple tools by using multiple <tool_call> blocks.
2. If you don't need any tool, just respond normally in plain text without any <tool_call> tags.
3. After receiving tool results, analyze them and either call more tools or give a final answer in plain text.
4. The "arguments" field MUST be a valid JSON object matching the tool's parameter schema.
5. NEVER use <arg_key>, <arg_value>, dotted names like Grep.datasource, or a bare tool name.
6. NEVER wrap <tool_call> in markdown code blocks like ```xml or ```json."""'''

new_prompt = '''<tool_calls>
<invoke name="TOOL_NAME">
<parameter name="arg_name">arg_value