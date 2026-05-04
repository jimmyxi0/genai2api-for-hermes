#!/usr/bin/env python3
"""Test script to capture raw GenAI API responses and check for usage/token fields."""

import json
import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from auth.token_manager import TokenManager
from config import GENAI_URL, model_registry

def test_genai_response():
    """Make a direct call to GenAI API and log the full response structure."""
    
    # Get token from .env
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    token_input = os.getenv("GENAI_TOKEN")
    if not token_input:
        print("ERROR: GENAI_TOKEN not found in .env")
        return
    
    token_mgr = TokenManager(token_input)
    token = token_mgr.get_token()
    
    # Force refresh if token is expired
    if not token or token_mgr._is_expired(token, margin=300):
        print("Token expired or missing, forcing refresh...")
        try:
            token = token_mgr.force_refresh()
        except Exception as e:
            print(f"Token refresh failed: {e}")
            print("You may need to log in manually first.")
            return
    
    if not token:
        print("ERROR: Could not get token. Please log in first.")
        return
    
    # Simple test message
    messages = [
        {"role": "user", "content": "Count from 1 to 50. Just numbers, nothing else."}
    ]
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "chatInfo": "test",
        "messages": messages,
        "type": "3",
        "stream": False,  # Non-streaming to get complete response
        "aiType": "gpt-4",
        "aiSecType": "1",
        "promptTokens": 0,
        "rootAiType": "xinference",
        "maxToken": 1000,
        "chatGroupId": None
    }
    
    print("=" * 80)
    print("Making request to GenAI API...")
    print(f"URL: {GENAI_URL}")
    print(f"Model: gpt-4")
    print("=" * 80)
    
    try:
        response = requests.post(
            GENAI_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Save full response to file for inspection
        with open("/tmp/genai_raw_response.json", "w") as f:
            f.write(response.text)
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"\nFull response saved to: /tmp/genai_raw_response.json")
        
        try:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Check for usage field
            print("\n" + "=" * 80)
            print("FIELD ANALYSIS:")
            print("=" * 80)
            
            if "usage" in data:
                print(f"✓ usage field present: {data['usage']}")
            else:
                print("✗ NO usage field in response")
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                print(f"\nchoices[0] keys: {list(choice.keys())}")
                
                if "message" in choice:
                    print(f"message keys: {list(choice['message'].keys())}")
                    if "content" in choice["message"]:
                        content = choice["message"]["content"]
                        print(f"\nContent length: {len(content)} chars")
                        print(f"Content preview: {content[:200]}")
                
                if "finish_reason" in choice:
                    print(f"finish_reason: {choice['finish_reason']}")
            
            # Check all top-level keys
            print(f"\nTop-level keys: {list(data.keys())}")
            
        except json.JSONDecodeError as e:
            print(f"Response is not valid JSON: {e}")
            print(f"Raw response: {response.text[:2000]}")
            
    except requests.RequestException as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    test_genai_response()