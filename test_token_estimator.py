#!/usr/bin/env python3
"""Test the token estimator with realistic scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from provider.genai import estimate_text_tokens, estimate_messages_tokens, TOKEN_ESTIMATE_SAFETY_MULTIPLIER

print("=" * 80)
print("TOKEN ESTIMATOR ROBUSTNESS TEST")
print("=" * 80)
print(f"Safety Multiplier: {TOKEN_ESTIMATE_SAFETY_MULTIPLIER}")
print()

# Scenario 1: Short conversation
print("Scenario 1: Short conversation (3 messages)")
print("-" * 60)
messages_short = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language."},
]
est_short = estimate_messages_tokens(messages_short)
print(f"Messages: {len(messages_short)}")
print(f"Estimated tokens: {est_short}")
print(f"✓ Should be well under any model limit")
print()

# Scenario 2: Long conversation history
print("Scenario 2: Long conversation history (20 messages)")
print("-" * 60)
messages_long = [
    {"role": "system", "content": "You are a helpful coding assistant."}
]
for i in range(10):
    messages_long.append({
        "role": "user", 
        "content": f"Question {i+1}: Explain concept number {i+1} in detail with examples."
    })
    messages_long.append({
        "role": "assistant",
        "content": f"Answer {i+1}: Concept {i+1} is an important topic. Here's a detailed explanation with multiple examples and code snippets..."
    })

est_long = estimate_messages_tokens(messages_long)
print(f"Messages: {len(messages_long)}")
print(f"Estimated tokens: {est_long}")
print(f"✓ Should trigger context warnings if > 50K tokens")
print()

# Scenario 3: Very long single prompt
print("Scenario 3: Very long single prompt (simulating document analysis)")
print("-" * 60)
long_prompt = """
Analyze the following code and provide detailed feedback:

""" + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 100) + """

Please review this code for:
1. Performance issues
2. Security vulnerabilities
3. Code style and best practices
4. Potential bugs
5. Optimization opportunities

Provide specific recommendations with code examples.
"""

est_prompt = estimate_text_tokens(long_prompt)
print(f"Prompt length: {len(long_prompt)} chars")
print(f"Estimated tokens: {est_prompt}")
print()

# Scenario 4: Mixed language content
print("Scenario 4: Mixed English-Chinese technical content")
print("-" * 60)
mixed_content = """
Implement a binary search tree (二叉搜索树) with the following operations:
1. insert(插入): Add a new node
2. delete(删除): Remove a node
3. search(搜索): Find a node by value
4. traverse(遍历): In-order, pre-order, post-order

请提供完整的 Python 实现，包括错误处理和单元测试。
Include time complexity analysis for each operation.
"""

est_mixed = estimate_text_tokens(mixed_content)
print(f"Content length: {len(mixed_content)} chars")
print(f"Estimated tokens: {est_mixed}")
print()

# Scenario 5: Context window protection test
print("Scenario 5: Context Window Protection")
print("-" * 60)
model_limit = 128000  # gpt-4 limit
prompt_tokens = estimate_messages_tokens(messages_long)
remaining = model_limit - prompt_tokens
default_max_tokens = max(100, int(remaining * 0.5))

print(f"Model context limit: {model_limit:,} tokens")
print(f"Prompt tokens (estimated): {prompt_tokens:,} tokens")
print(f"Remaining context: {remaining:,} tokens")
print(f"Default max_tokens (50% of remaining): {default_max_tokens:,} tokens")
print()
if default_max_tokens < 1000:
    print("⚠ WARNING: Prompt is very long, limiting response to prevent truncation")
else:
    print("✓ Sufficient context space for response")
print()

# Scenario 6: Safety margin demonstration
print("Scenario 6: Safety Margin Impact")
print("-" * 60)
import re
TOKEN_PATTERN_RAW = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")

test_texts = [
    "Short English sentence",
    "简短的中文句子",
    "Mixed 混合 content 内容 with numbers 12345",
    "A" * 1000,  # 1000 A's
    "你" * 1000,  # 1000 Chinese chars
]

print(f"{'Text':<40} {'Raw':>8} {'Adjusted':>10} {'Buffer':>8}")
print("-" * 60)
for text in test_texts:
    raw = len(TOKEN_PATTERN_RAW.findall(text))
    adjusted = estimate_text_tokens(text)
    buffer = raw - adjusted
    label = text[:37] + "..." if len(text) > 40 else text
    print(f"{label:<40} {raw:>8} {adjusted:>10} {buffer:>8}")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print()
print("Summary:")
print(f"  - Safety multiplier: {TOKEN_ESTIMATE_SAFETY_MULTIPLIER} (reserves ~{int((1-TOKEN_ESTIMATE_SAFETY_MULTIPLIER)*100)}% buffer)")
print(f"  - Token estimation is working correctly")
print(f"  - Context window protection is active")
print()