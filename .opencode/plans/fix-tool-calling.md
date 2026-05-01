# Fix Tool Calling: Replace Unicode Markers with ASCII XML + Anti-Leak + Buffer-and-Parse

## Problem

Tool calls frequently fail and get emitted as plain text when using openclaw (Anthropic Messages API client), but work reliably with opencode. Root causes:

1. **Unicode markers `<tool_call>`/`</arg_value>`**: Xinference-deployed models (on `http://127.0.0.1:5000/v1`) don't reliably handle these Unicode characters. They may garble or strip them, breaking tool call detection.
2. **No code fence anti-leak**: When the model wraps tool calls in markdown code fences (\`\`\`xml ... \`\`\`), they get detected as tool calls but the fences corrupt parsing. ds2api skips tool detection inside code fences.
3. **No `<tool_calls>`/`<invoke>`/`<parameter>` format**: The ds2api-recommended format uses standard ASCII XML tags that all models handle well. Our parser only understands `<tool_call>` (singular) with Unicode closing tags.
4. **No DSML normalization**: Some models (especially DeepSeek) emit `<|DSML|tool_calls>` / `<|DSML|invoke>` / `<|DSML|parameter>` tags. These aren't normalized to the canonical format.
5. **Streaming text leak**: The current streaming approach emits text chunks immediately. When a tool call appears mid-stream, text before it is already sent. If parsing fails, the tool call XML is emitted as raw text (the "emitting as text" warning). ds2api uses a buffer-and-parse strategy (`bufferToolContent`) that holds all text when tools are expected, then parses at finalize.

## Plan (8 items)

### 1. Replace Unicode markers with ASCII XML in `tools/prompts.py`

**File**: `tools/prompts.py`

Change the system prompt to use `<tool_calls>`/`<invoke>`/`<parameter>` format:

```
<tool_calls>
<invoke name="<function-name>">
<parameter name="<param-name>"><param-value></parameter>
</invoke>
</tool_calls>
```

Also update:
- `format_tool_examples()` to generate `<invoke>` examples instead of `<tool_call>` JSON
- `inject_tool_prompt()` to convert tool_results and tool_uses to use `<tool_calls>`/`<invoke>` format for assistant messages
- Rules text to reference XML tags instead of Unicode markers

### 2. Add `<invoke>`/`<parameter>` parsing in `tools/parsing.py`

**File**: `tools/parsing.py`

Add new parsing functions:

- `parse_invoke_style_calls(text, allowed_tool_names)` — Parses `<tool_calls><invoke name="..."><parameter name="...">value</parameter></invoke></tool_calls>` format
- `parse_invoke_body(body, tool_name)` — Parses the inner content of an `<invoke>` block: extracts `<parameter>` tags, handles JSON values and CDATA
- Normalize `<![CDATA[...]]>` sections inside parameter values
- Add `<invoke>` / `<function_call>` / `<tool_use>` as recognized tool call opening/closing tag pairs (like ds2api's `xmlToolCallTagPairs`)

Update `extract_tool_calls()` to try the new format as primary strategy (before fallback extractors).

Also support legacy `<tool_call>` format (current) as a fallback for backward compatibility.

### 3. Add code fence anti-leak detection in `tools/parsing.py`

**File**: `tools/parsing.py` (new helper functions)

Port ds2api's `stripFencedCodeBlocks()` and `insideCodeFence()` logic:

- `strip_fenced_code_blocks(text)` — Remove content inside markdown code fences (\`\`\` or ~~~), preserving content outside fences. Tool calls inside code fences are ignored (they're examples/documentation, not actual calls).
- `is_inside_code_fence(text_up_to_position)` — Check if a position is inside a code fence, used during streaming to avoid false tool call detection.
- Handle nested fences, mixed backtick/tilde, unclosed fences.

Call `strip_fenced_code_blocks()` early in `extract_tool_calls()` before XML parsing.

### 4. Add DSML tag normalization in `tools/parsing.py`

**File**: `tools/parsing.py` (new helper)

Add `normalize_dsml_tags(text)` function:

- Convert `<|DSML|tool_calls>` → `<tool_calls>`
- Convert `<|DSML|invoke name="...">` → `<invoke name="...">`
- Convert `<|DSML|parameter name="...">` → `<parameter name="...">`
- Convert `<|DSML|/tool_calls>` → `</tool_calls>`
- Convert `<|DSML|/invoke>` → `</invoke>`
- Convert `<|DSML|/parameter>` → `</parameter>`
- Also handle `｜` (fullwidth vertical bar) variant

Call this early in `extract_tool_calls()` after stripping think blocks.

### 5. Refactor streaming to buffer-and-parse strategy

**Files**: `provider/genai.py`, `provider/anthropic.py`

This is the most impactful change. Current approach:
- Stream text immediately
- When `<tool_call` tag detected, switch to buffering
- If parsing fails → emit buffered text (the "emitting as text" warning)

New approach (modeled on ds2api's `bufferToolContent` + `ToolSieve`):
- When tools are present in the request, **buffer ALL content** (don't emit text immediately)
- Track code fence state to avoid false detection inside fences
- At stream end (finalize), parse the complete buffered text with `extract_tool_calls()`
- If tool calls found → emit them, emit any remaining text as a text block
- If no tool calls found → emit the entire buffered text as a text block

This eliminates the "parsing failed — emitting as text" scenario because we have the complete output to parse.

For the OpenAI streaming path (`stream_genai_response_with_tools` in `genai.py`):
- Remove the per-chunk tag detection logic
- Instead, accumulate all content into a buffer
- At finalize, parse and emit tool_calls or text

For the Anthropic streaming path (`stream_genai_as_anthropic` in `anthropic.py`):
- Remove the per-chunk tag detection + `BARE_TOOL_CALL_RE` logic
- Buffer all content (respecting thinking filter)
- At finalize, parse and emit `tool_use` content blocks or text

### 6. Ensure Xinference compatibility

**Files**: `tools/prompts.py`, `provider/anthropic.py`

- System prompt must NOT contain any Unicode special characters (→ already fixed by item 1)
- Tool result injection in `inject_tool_prompt()` must use plain ASCII XML
- `anthropic_message_to_genai_messages()` must convert assistant tool_uses to `<tool_calls>/<invoke>` format instead of `<tool_call>` JSON
- Remove `BARE_TOOL_CALL_RE` detection (it matches `{"name":` which is unreliable with Xinference models)

### 7. Update `provider/anthropic.py` streaming

**File**: `provider/anthropic.py`

- Remove `BARE_TOOL_CALL_RE` and associated detection logic
- Implement buffer-and-parse: accumulate `content` in `text` builder
- Track code fence state during streaming (for anti-leak)
- In finalize: parse buffered text with `extract_tool_calls()`, emit `tool_use` blocks or text
- Keep thinking/reasoning filtering as-is (it works well)
- Update `anthropic_message_to_genai_messages()` to use `<invoke>` format for assistant tool_uses

### 8. Update `provider/genai.py` streaming

**File**: `provider/genai.py`

- Rewrite `stream_genai_response_with_tools()` to use buffer-and-parse
- Accumulate all content into a single buffer
- At finalize, call `extract_tool_calls()` on complete buffer
- Emit tool_calls chunks or text chunks based on parse result
- Remove per-chunk `_tag_prefix_len()` detection

## Implementation Order

1. `tools/prompts.py` — New system prompt format (items 1, 6)
2. `tools/parsing.py` — New parsers + anti-leak + DSML normalization (items 2, 3, 4)
3. `provider/anthropic.py` — Buffer-and-parse streaming (items 5, 7)
4. `provider/genai.py` — Buffer-and-parse streaming (items 5, 8)
5. Tests — Update existing tests, add new test cases

## Key Design Decisions

- **Primary format**: `<tool_calls>/<invoke>/<parameter>` (ASCII XML, Xinference-safe)
- **Legacy support**: Keep `<tool_call>` (singular) as fallback parser for older models
- **Buffering strategy**: Buffer ALL content when tools present, parse at finalize (ds2api approach)
- **Code fence handling**: Skip tool detection inside code fences (ds2api approach)
- **DSML support**: Normalize to canonical XML before parsing
- **No Unicode markers**: System prompt and all injected text use ASCII-only characters
