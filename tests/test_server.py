#!/usr/bin/env python3
"""End-to-end checks against a running nace-infer-server (stdlib only).
usage: test_server.py [base_url]   (default http://127.0.0.1:8090)"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file from disk.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute path"},
            "max_lines": {"type": "integer", "description": "Optional line limit"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

def post(body, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.loads(r.read())
    out = {"reasoning": "", "content": "", "tool_calls": [], "finish": None, "usage": None, "chunks": 0,
           "t_first": None}
    t0 = time.time()
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        c = json.loads(line[6:])
        out["chunks"] += 1
        if "error" in c:
            raise RuntimeError(c["error"])
        ch = c["choices"][0]
        d = ch["delta"]
        if (d.get("reasoning_content") or d.get("content")) and out["t_first"] is None:
            out["t_first"] = time.time() - t0
        out["reasoning"] += d.get("reasoning_content") or ""
        out["content"] += d.get("content") or ""
        out["tool_calls"] += d.get("tool_calls") or []
        if ch.get("finish_reason"):
            out["finish"] = ch["finish_reason"]
            out["usage"] = c.get("usage")
    return out

fails = 0
def check(name, ok, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "  " + detail))
    if not ok:
        fails += 1

# 1. thinking on, streaming: reasoning separate from content
r = post({"messages": [{"role": "user", "content": "What is 17*23? Answer with just the number."}],
          "max_tokens": 2048, "stream": True}, stream=True)
check("thinking: reasoning streamed separately", len(r["reasoning"]) > 0 and "<think>" not in r["content"], repr(r)[:300])
check("thinking: correct answer", "391" in r["content"], repr(r["content"])[:200])
check("thinking: finish=stop + usage", r["finish"] == "stop" and r["usage"] and r["usage"]["completion_tokens"] > 0, str(r["finish"]))

# 2. tool call (streaming), then tool result round trip with prefix reuse
msgs = [{"role": "system", "content": "You are a coding agent. Use tools to answer."},
        {"role": "user", "content": "Read the file /etc/hosts and tell me how many lines it has."}]
r = post({"messages": msgs, "tools": TOOLS, "max_tokens": 4096, "stream": True,
          "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
tc = r["tool_calls"]
check("tool call: finish=tool_calls", r["finish"] == "tool_calls", str(r["finish"]) + " " + repr(r["content"])[:200])
ok = len(tc) == 1 and tc[0]["function"]["name"] == "read_file"
args = json.loads(tc[0]["function"]["arguments"]) if ok else {}
check("tool call: read_file with path", ok and args.get("path") == "/etc/hosts", json.dumps(tc)[:300])
check("tool call: no markup in content", "<tool_call>" not in r["content"] and "<function" not in r["content"], repr(r["content"]))
first_prompt = r["usage"]["prompt_tokens"] if r["usage"] else 0
msgs += [{"role": "assistant", "content": r["content"], "tool_calls": [
            {"id": tc[0]["id"], "type": "function", "function": tc[0]["function"]}] if ok else []},
         {"role": "tool", "tool_call_id": tc[0]["id"] if ok else "x",
          "content": "127.0.0.1 localhost\n255.255.255.255 broadcasthost\n::1 localhost\n"}]
r2 = post({"messages": msgs, "tools": TOOLS, "max_tokens": 2048, "stream": True,
           "chat_template_kwargs": {"enable_thinking": False}}, stream=True)
check("tool result: final answer mentions 3", "3" in r2["content"], repr(r2["content"])[:200])
cached = r2["usage"]["prompt_tokens_details"]["cached_tokens"] if r2["usage"] else 0
check("tool result: prefix reused (cached >= first prompt)", cached >= first_prompt,
      f"cached={cached} first_prompt={first_prompt} usage={r2['usage']}")

# 3. next user turn: reuse from the checkpoint
msgs += [{"role": "assistant", "content": r2["content"]},
         {"role": "user", "content": "Thanks. Now in one word: what OS uses /etc/hosts?"}]
r3 = post({"messages": msgs, "tools": TOOLS, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}})
u = r3["usage"]
check("next turn: cached prefix", u["prompt_tokens_details"]["cached_tokens"] > first_prompt // 2, json.dumps(u))
check("next turn: answer", len(r3["choices"][0]["message"]["content"]) > 0, json.dumps(r3)[:300])

# 4. non-streaming tool call with integer argument coercion
r4 = post({"messages": [{"role": "user", "content": "Use read_file to read the first 5 lines of /etc/passwd."}],
           "tools": TOOLS, "max_tokens": 2048, "chat_template_kwargs": {"enable_thinking": False}})
m = r4["choices"][0]["message"]
tcs = m.get("tool_calls") or []
a = json.loads(tcs[0]["function"]["arguments"]) if tcs else {}
check("non-stream tool call + int arg", tcs and a.get("path") == "/etc/passwd" and a.get("max_lines") in (5, None),
      json.dumps(m)[:300])
if tcs and "max_lines" in a:
    check("int argument is a JSON number", isinstance(a["max_lines"], int), repr(a))

# 5. side slot: a short unrelated request must not evict the main conversation
post({"messages": [{"role": "user", "content": "Title for: hosts file question"}], "max_tokens": 20,
      "chat_template_kwargs": {"enable_thinking": False}})
msgs += [{"role": "assistant", "content": r3["choices"][0]["message"]["content"]},
         {"role": "user", "content": "And on Windows, where is it?"}]
r5 = post({"messages": msgs, "tools": TOOLS, "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}})
check("main slot survives side request", r5["usage"]["prompt_tokens_details"]["cached_tokens"] > first_prompt // 2,
      json.dumps(r5["usage"]))

# 6. thinking on + tools: the call comes back as a tool call (Qwen3.6 sometimes writes it inside its think
#    block and stops without closing it; the server returns such a call as a tool call, NI_THINK_TOOLCALLS)
r6 = post({"messages": [{"role": "system", "content": "You are a coding agent. Use tools to answer."},
                        {"role": "user", "content": "Read the file /etc/hosts and tell me how many lines it has."}],
           "tools": TOOLS, "max_tokens": 4096, "temperature": 0, "top_k": 1, "stream": True}, stream=True)
tc6 = r6["tool_calls"]
check("thinking + tools: read_file returned as a tool call",
      r6["finish"] == "tool_calls" and len(tc6) >= 1 and tc6[0]["function"]["name"] == "read_file",
      str(r6["finish"]) + " " + json.dumps(tc6)[:200] + " reasoning tail " + repr(r6["reasoning"][-200:]))

print("FAILED: %d" % fails if fails else "ALL PASSED")
sys.exit(1 if fails else 0)
