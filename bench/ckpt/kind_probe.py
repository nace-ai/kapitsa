#!/usr/bin/env python3
"""Sends one-token requests whose last message has each shape the checkpoint policy distinguishes, so the
server log (NI_LOG=2: "checkpoints: query|tool boundary at ...") shows how each turn boundary was classified.

usage: kind_probe.py [--base http://127.0.0.1:8097/v1]
Prints one line per request: the expected class and the shape sent, in order, to set against the log. Every case
renders a different prompt: one identical to the previous request would restore its boundary checkpoint and save none.
"""
import argparse, json, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8097/v1")
a = ap.parse_args()

SYS = {"role": "system", "content": "You are a test assistant."}
CALL = {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
    "name": "read_file", "arguments": "{\"path\": \"a.txt\"}"}}]}
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
Q = {"role": "user", "content": "Read a.txt please."}
CASES = [
    ("query", "plain user query", [SYS, Q]),
    ("tool", "role tool after a tool call", [SYS, Q, CALL, {"role": "tool", "tool_call_id": "call_1", "content": "hello"}]),
    ("tool", "user message wrapped in <tool_response> (with surrounding whitespace)",
     [SYS, Q, CALL, {"role": "user", "content": "\n <tool_response>\nhello 2\n</tool_response> \n"}]),
    ("query", "user message that only starts with <tool_response>",
     [SYS, Q, CALL, {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
      {"role": "user", "content": "<tool_response> looks odd, what is it?"}]),
    ("query", "user query after a tool response (harness note)",
     [SYS, Q, CALL, {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
      {"role": "user", "content": "Also check b.txt."}]),
    ("tool", "multi-part text content wrapped in <tool_response>",
     [SYS, Q, CALL, {"role": "user", "content": [{"type": "text", "text": "<tool_response>\nhel"},
                                                  {"type": "text", "text": "lo 3\n</tool_response>"}]}]),
    ("query", "multi-part text content (a query)",
     [SYS, {"role": "user", "content": [{"type": "text", "text": "Read "}, {"type": "text", "text": "a.txt"}]}]),
    ("tool", "last message from the assistant (continuation)", [SYS, Q, {"role": "assistant", "content": "Sure"}]),
]
for want, what, msgs in CASES:
    b = {"messages": msgs, "tools": TOOLS, "max_tokens": 1, "temperature": 0.0, "top_k": 1,
         "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=600).read())
    print(f"expect {want:<5} {what}  (prompt {r['usage']['prompt_tokens']})", flush=True)
