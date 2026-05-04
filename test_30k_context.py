#!/usr/bin/env python3
"""
Generate ~30K token context and test through the proxy.
This validates the token estimation fixes under real load.
"""

import sys
import json
import requests

sys.path.insert(0, '/home/j/Desktop/shanghaitech-genai2api')

from provider.genai import estimate_text_tokens, TOKEN_PATTERN

# Generate test content
def generate_content(target_tokens, lang='en'):
    """Generate text approximating target token count."""
    if lang == 'en':
        # English: ~4 chars per token average
        chars_needed = target_tokens * 4
        word = "function"
        return (word + " ") * (chars_needed // 9)
    elif lang == 'zh':
        # Chinese: ~1 char per token
        return "你" * target_tokens
    else:
        # Mixed
        return ("Hello 世界 " * (target_tokens // 5))

def test_30k_context():
    print("=" * 70)
    print("30K TOKEN CONTEXT TEST")
    print("=" * 70)
    
    # Generate ~30K tokens of context
    target_tokens = 30000
    
    print(f"\nGenerating {target_tokens:,} tokens of context...")
    
    # Create a realistic multi-message conversation
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
    ]
    
    # Add user message with ~30K tokens of context
    long_context = generate_content(target_tokens, lang='en')
    messages.append({
        "role": "user",
        "content": f"Please analyze the following context and provide a summary:\n\n{long_context}"
    })
    
    # Estimate tokens
    total_chars = sum(len(m.get('content', '')) for m in messages)
    regex_count = sum(len(TOKEN_PATTERN.findall(m.get('content', ''))) for m in messages)
    estimated = estimate_text_tokens(long_context) + sum(4 for _ in messages)  # +4 per message overhead
    
    print(f"\nMessage structure:")
    print(f"  Total messages: {len(messages)}")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Regex token count: {regex_count:,}")
    print(f"  Estimated tokens (with safety): {estimated:,}")
    print(f"  Target tokens: {target_tokens:,}")
    
    # Calculate remaining context
    context_limit = 128000
    remaining = context_limit - estimated
    max_tokens = max(100, int(remaining * 0.5))
    
    print(f"\nContext window analysis:")
    print(f"  Context limit: {context_limit:,}")
    print(f"  Used by prompt: {estimated:,} ({estimated/context_limit*100:.1f}%)")
    print(f"  Remaining: {remaining:,}")
    print(f"  Max tokens (50% of remaining): {max_tokens:,}")
    
    if remaining <= 0:
        print(f"\n⚠️  WARNING: Prompt exceeds context window!")
    elif max_tokens < 1000:
        print(f"\n⚠️  WARNING: Low response budget ({max_tokens:,} tokens)")
    else:
        print(f"\n✓ Sufficient context space for response")
    
    # Test through proxy (if running)
    print(f"\n{'='*70}")
    print("TESTING THROUGH PROXY")
    print(f"{'='*70}")
    
    proxy_url = "http://localhost:5000/v1/chat/completions"
    
    payload = {
        "model": "qwen-instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    print(f"\nSending request to {proxy_url}...")
    print(f"  Payload size: {len(json.dumps(payload)):,} bytes")
    print(f"  Max tokens: {max_tokens:,}")
    
    try:
        response = requests.post(proxy_url, json=payload, timeout=120)
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get('usage', {})
            choices = result.get('choices', [])
            
            print(f"\nUsage:")
            print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A'):,}")
            print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A'):,}")
            print(f"  Total tokens: {usage.get('total_tokens', 'N/A'):,}")
            
            if choices:
                finish_reason = choices[0].get('finish_reason', 'N/A')
                content_length = len(choices[0].get('message', {}).get('content', ''))
                print(f"\nResponse:")
                print(f"  Finish reason: {finish_reason}")
                print(f"  Content length: {content_length:,} chars")
                
                if finish_reason == 'length':
                    print(f"\n⚠️  Response was truncated (hit max_tokens limit)")
                elif finish_reason == 'stop':
                    print(f"\n✓ Response completed normally")
        else:
            print(f"\nError: {response.status_code}")
            print(f"  {response.text[:500]}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n⚠️  Proxy not running at {proxy_url}")
        print(f"  Start with: cd ~/Desktop/shanghaitech-genai2api && .venv/bin/python -m api.server")
    except requests.exceptions.Timeout:
        print(f"\n⚠️  Request timed out (120s limit)")
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
    
    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    test_30k_context()