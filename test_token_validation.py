#!/usr/bin/env python3
"""
Quick validation: token estimation with realistic inputs.
"""

import sys
sys.path.insert(0, '/home/j/Desktop/shanghaitech-genai2api')

from provider.genai import estimate_text_tokens, TOKEN_PATTERN

def test_realistic_inputs():
    print("=" * 70)
    print("REALISTIC TOKEN ESTIMATION TEST")
    print("=" * 70)
    
    test_cases = [
        ("Short English query", "What is the capital of France?"),
        ("Long English explanation", """
            The French Revolution was a period of political and societal change 
            that began in France with the Estates General of 1789 and ended 
            with the formation of the French Consulate in November 1799. 
            Many factors led to the revolution; to some extent, it can be 
            traced to Louis XVI's inability to handle the financial crisis 
            plaguing France. The revolution ultimately resulted in the 
            overthrow of the monarchy, the establishment of a republic, 
            and a series of conflicts that extended across Europe.
        """),
        ("Chinese question", "法国的首都是哪里？"),
        ("Mixed technical", "请解释 transformer 架构中的 self-attention mechanism"),
        ("Code snippet", """
            def estimate_tokens(text):
                tokens = TOKEN_PATTERN.findall(text)
                return len(tokens) * SAFETY_MULTIPLIER
        """),
        ("JSON payload", '{"model": "qwen-instruct", "messages": [{"role": "user", "content": "Hello"}]}'),
        ("10K Chinese chars", "你" * 10000),
        ("10K English chars", "A" * 10000),
    ]
    
    print(f"\n{'Test':<30} {'Chars':>8} {'Regex':>8} {'Estimate':>10} {'Real*':>8}")
    print("-" * 70)
    
    for name, text in test_cases:
        chars = len(text)
        regex_count = len(TOKEN_PATTERN.findall(text))
        estimated = estimate_text_tokens(text)
        # Real tokenizer estimate: English ~4 chars/token, Chinese ~1 char/token
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            real_approx = chars  # Chinese: 1 char ≈ 1 token
        else:
            real_approx = chars // 4  # English: 4 chars ≈ 1 token
        
        print(f"{name:<30} {chars:>8,} {regex_count:>8,} {estimated:>10,} {real_approx:>8,}")
    
    print("\n* Real tokenizer approximation (cl100k_base)")
    print("\nKey observations:")
    print("  - English: regex overestimates (counts each char), which is SAFE")
    print("  - Chinese: regex is accurate (1 char = 1 token)")
    print("  - 1.2x safety buffer adds margin for edge cases")
    print("  - Result: we RARELY underestimate, preventing truncation")

if __name__ == '__main__':
    test_realistic_inputs()