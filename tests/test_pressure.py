#!/usr/bin/env python3
"""
Tool Calling Pressure Test for Xinference Models
Tests all available models for tool calling stability and success rate.

Usage:
    python tests/test_pressure.py [--base-url http://localhost:5000] [--retries 3] [--timeout 60]

Test Scenarios:
    1. Single tool call (Bash)
    2. Multiple tool selection (3 tools, pick correct)
    3. Multi-turn (tool -> result -> answer)
    4. No tool needed (simple question)
    5. Streaming tool call
    6. Edit tool call (multiline)
    7. Read tool call
    8. Glob tool call
    9. Parallel tool calls (multiple at once)
"""

import argparse
import json
import time
import requests
from datetime import datetime

# === Tool Definitions (Real Claude Code Tools) ===

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Execute a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "description": {"type": "string", "description": "What this command does"}
            },
            "required": ["command"]
        }
    }
}

READ_TOOL = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "offset": {"type": "integer", "description": "Line offset to start reading from"},
                "limit": {"type": "integer", "description": "Maximum number of lines to read"}
            },
            "required": ["file_path"]
        }
    }
}

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "Edit",
        "description": "Replace content in an existing file.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "oldString": {"type": "string", "description": "The exact text to replace"},
                "newString": {"type": "string", "description": "The replacement text"}
            },
            "required": ["file_path", "oldString", "newString"]
        }
    }
}

WRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "Write",
        "description": "Write content to a file (creates or overwrites).",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path to the file"},
                "content": {"type": "string", "description": "The content to write"}
            },
            "required": ["file_path", "content"]
        }
    }
}

GLOB_TOOL = {
    "type": "function",
    "function": {
        "name": "Glob",
        "description": "List files matching a glob pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g., '**/*.py'"},
                "path": {"type": "string", "description": "Root directory to search in"}
            },
            "required": ["pattern"]
        }
    }
}

GREP_TOOL = {
    "type": "function",
    "function": {
        "name": "Grep",
        "description": "Search for text in files.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "The search pattern"},
                "path": {"type": "string", "description": "Directory to search in"},
                "output_mode": {"type": "string", "enum": ["content", "files", "count"], "description": "Output format"},
                "head_limit": {"type": "integer", "description": "Max results"}
            },
            "required": ["pattern"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g., '2+3*4'"}
            },
            "required": ["expression"]
        }
    }
}


class PressureTest:
    def __init__(self, base_url: str, retries: int, timeout: int):
        self.base_url = base_url
        self.retries = retries
        self.timeout = timeout
        self.results = {}  # {model: {test_name: [attempts]}}
        self.models = []
    
    def log(self, msg: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def discover_models(self) -> list:
        """Get all available models from the API."""
        self.log("Discovering models...")
        try:
            resp = requests.get(f"{self.base_url}/v1/models", timeout=10)
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            self.log(f"Found {len(models)} models: {models}")
            return models
        except Exception as e:
            self.log(f"Failed to discover models: {e}")
            # Fallback: try to get from config
            return ["GPT-4.1", "GPT-3.5", "chatglm", "claude-3"]
    
    def call_api(self, model: str, messages: list, tools: list, stream: bool = False) -> dict:
        """Make a chat completion call."""
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "stream": stream
                },
                timeout=self.timeout,
                stream=stream
            )
            if stream:
                return self._parse_stream(resp)
            result = resp.json()
            # Check for API errors
            if "error" in result:
                return result
            return result
        except requests.exceptions.Timeout:
            return {"error": "timeout", "message": f"Request timed out after {self.timeout}s"}
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_stream(self, resp) -> dict:
        """Parse streaming response."""
        tool_calls = []
        content = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            if not line_str.startswith('data: '):
                continue
            data_str = line_str[6:].strip()
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    content += delta["content"]
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
            except Exception:
                pass
        return {"content": content, "tool_calls": tool_calls}
    
    def test_single_tool(self, model: str) -> bool:
        """Test 1: Single tool call (Bash)."""
        messages = [{"role": "user", "content": "List the files in the current directory."}]
        result = self.call_api(model, messages, [BASH_TOOL])
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        return bool(msg.get("tool_calls"))
    
    def test_tool_selection(self, model: str) -> bool:
        """Test 2: Multiple tools - should pick correct one."""
        messages = [{"role": "user", "content": "What is 123 * 456 + 789?"}]
        tools = [BASH_TOOL, READ_TOOL, CALCULATOR_TOOL]
        result = self.call_api(model, messages, tools)
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        tool_names = [tc["function"]["name"] for tc in msg.get("tool_calls", [])]
        
        # Should pick 'calculate' or 'Bash' (for shell calculation)
        return any(n in ["calculate", "Bash"] for n in tool_names)
    
    def test_multi_turn(self, model: str) -> bool:
        """Test 3: Multi-turn (tool -> result -> answer)."""
        # Round 1: Get tool call
        messages = [{"role": "user", "content": "List files in /tmp"}]
        result1 = self.call_api(model, messages, [BASH_TOOL])
        
        if "error" in result1:
            return False
        
        msg1 = result1.get("choices", [{}])[0].get("message", {})
        tool_calls = msg1.get("tool_calls", [])
        
        if not tool_calls:
            return False
        
        # Round 2: Send tool result
        tc = tool_calls[0]
        messages2 = messages + [
            {"role": "assistant", "content": None, "tool_calls": tool_calls},
            {"role": "tool", "tool_call_id": tc["id"], "content": "file1.txt\nfile2.txt"}
        ]
        result2 = self.call_api(model, messages2, [BASH_TOOL])
        
        if "error" in result2:
            return False
        
        msg2 = result2.get("choices", [{}])[0].get("message", {})
        # Should have final answer (not another tool call)
        return bool(msg2.get("content")) and not msg2.get("tool_calls")
    
    def test_no_tool_needed(self, model: str) -> bool:
        """Test 4: No tool needed - shouldn't hallucinate tool calls."""
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        tools = [BASH_TOOL, READ_TOOL, CALCULATOR_TOOL]
        result = self.call_api(model, messages, tools)
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        # No tool calls = good (unless it explicitly explains why it doesn't need tools)
        return not msg.get("tool_calls") or "don't need" in msg.get("content", "").lower()
    
    def test_stream_tool(self, model: str) -> bool:
        """Test 5: Streaming tool call."""
        messages = [{"role": "user", "content": "Run 'echo hello'"}]
        result = self.call_api(model, messages, [BASH_TOOL], stream=True)
        
        if "error" in result:
            return False
        
        return bool(result.get("tool_calls"))
    
    def test_edit_tool(self, model: str) -> bool:
        """Test 6: Edit tool call with multiline content."""
        messages = [{"role": "user", "content": 'Edit file /tmp/test.py: replace "def foo():" with "def bar():"'}]
        result = self.call_api(model, messages, [EDIT_TOOL])
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        return bool(msg.get("tool_calls"))
    
    def test_read_tool(self, model: str) -> bool:
        """Test 7: Read tool call."""
        messages = [{"role": "user", "content": "Read file /etc/hostname"}]
        result = self.call_api(model, messages, [READ_TOOL])
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        return bool(msg.get("tool_calls"))
    
    def test_glob_tool(self, model: str) -> bool:
        """Test 8: Glob tool call."""
        messages = [{"role": "user", "content": "Find all Python files in /tmp"}]
        result = self.call_api(model, messages, [GLOB_TOOL])
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        return bool(msg.get("tool_calls"))
    
    def test_parallel_calls(self, model: str) -> bool:
        """Test 9: Parallel tool calls."""
        messages = [{"role": "user", "content": "List files in /tmp and read /etc/hostname"}]
        tools = [BASH_TOOL, READ_TOOL]
        result = self.call_api(model, messages, tools)
        
        if "error" in result:
            return False
        
        msg = result.get("choices", [{}])[0].get("message", {})
        tc = msg.get("tool_calls", [])
        # Has multiple tool calls = success
        return len(tc) >= 2
    
    def run_test(self, model: str, test_name: str, test_fn) -> bool:
        """Run a single test with retries."""
        results = []
        for attempt in range(self.retries):
            self.log(f"  {test_name} (attempt {attempt + 1}/{self.retries})...")
            try:
                success = test_fn(model)
                results.append(success)
                if success:
                    self.log("    ✓ PASS")
                    break
                else:
                    self.log("    ✗ FAIL")
            except Exception as e:
                self.log(f"    ✗ ERROR: {type(e).__name__}: {e}")
                results.append(False)
            time.sleep(0.3)
        
        if model not in self.results:
            self.results[model] = {}
        self.results[model][test_name] = results
        
        return any(results)
    
    def run_all(self):
        """Run pressure test on all models."""
        self.models = self.discover_models()
        
        if not self.models:
            self.log("No models found, exiting.")
            return
        
        # Test definitions
        tests = [
            ("single_tool", self.test_single_tool),
            ("tool_selection", self.test_tool_selection),
            ("multi_turn", self.test_multi_turn),
            ("no_tool_needed", self.test_no_tool_needed),
            ("stream_tool", self.test_stream_tool),
            ("edit_tool", self.test_edit_tool),
            ("read_tool", self.test_read_tool),
            ("glob_tool", self.test_glob_tool),
            ("parallel_calls", self.test_parallel_calls),
        ]
        
        self.log(f"Starting pressure test on {len(self.models)} models...")
        self.log(f"Each test will be retried {self.retries} times.\n")
        
        start_time = time.time()
        
        for model in self.models:
            self.log(f"\n{'='*60}")
            self.log(f"Testing model: {model}")
            self.log(f"{'='*60}")
            
            for test_name, test_fn in tests:
                self.run_test(model, test_name, test_fn)
        
        elapsed = time.time() - start_time
        self.log(f"\n\nCompleted in {elapsed:.1f}s")
        
        self.print_summary()
    
    def print_summary(self):
        """Print summary table."""
        self.log("\n" + "="*80)
        self.log("RESULTS SUMMARY")
        self.log("="*80)
        
        # Header
        test_names = ["single_tool", "tool_selection", "multi_turn", "no_tool_needed",
                      "stream_tool", "edit_tool", "read_tool", "glob_tool", "parallel_calls"]
        
        header = f"{'Model':<20}" + "".join([f"{t:<14}" for t in ["single", "select", "multi", "none", "stream", "edit", "read", "glob", "parallel"]])
        print(header)
        print("-" * len(header))
        
        # Rows
        for model, model_results in self.results.items():
            row = f"{model:<20}"
            for test in test_names:
                attempts = model_results.get(test, [False, False, False])
                passed = any(attempts)
                rate = f"{sum(attempts)}/{len(attempts)}"
                symbol = "✓" if passed else "✗"
                row += f" {symbol}{rate:<11}"
            print(row)
        
        # Summary
        print("-" * len(header))
        print("\nSuccess Rate by Test:")
        for test in test_names:
            total = 0
            passed = 0
            for model_results in self.results.values():
                attempts = model_results.get(test, [False, False, False])
                total += len(attempts)
                passed += sum(attempts)
            rate = f"{passed}/{total} ({100*passed//total if total else 0}%)"
            print(f"  {test:<20} {rate}")
        
        # JSON output
        output = {
            "timestamp": datetime.now().isoformat(),
            "models": self.models,
            "retries": self.retries,
            "results": {
                model: {
                    test: {
                        "passed": any(results),
                        "attempts": results
                    }
                    for test, results in model_results.items()
                }
                for model, model_results in self.results.items()
            }
        }
        
        print("\nJSON output saved to: pressure_test_results.json")
        with open("pressure_test_results.json", "w") as f:
            json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Tool Calling Pressure Test")
    parser.add_argument('--base-url', default='http://localhost:5000', help='API base URL')
    parser.add_argument('--retries', type=int, default=3, help='Retries per test')
    parser.add_argument('--timeout', type=int, default=60, help='Request timeout (seconds)')
    args = parser.parse_args()
    
    test = PressureTest(args.base_url, args.retries, args.timeout)
    test.run_all()


if __name__ == '__main__':
    main()