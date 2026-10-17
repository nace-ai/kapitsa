#!/usr/bin/env python3
"""End-to-end checks against a running `kapitsa serve` (stdlib only).
usage: test_server.py [base_url] [--api-key KEY] [--model-id ID]   (default http://127.0.0.1:8090)
       test_server.py --cli <path to kapitsa>    offline checks of the command line: model names and aliases,
                                                 `pull --dry-run`, `list`, `rm`, early port errors (no server)"""
import json, os, re, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request

ARGS = sys.argv[1:]
def opt(name):
    if name in ARGS:
        i = ARGS.index(name)
        v = ARGS[i + 1]
        del ARGS[i:i + 2]
        return v
    return None
CLI = opt("--cli")
API_KEY = opt("--api-key")
MODEL_ID = opt("--model-id")
BASE = ARGS[0] if ARGS else "http://127.0.0.1:8090"
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

def auth():
    return {"Authorization": "Bearer " + API_KEY} if API_KEY else {}

def get(path, headers=None):
    """(status, headers, body) of a GET; HTTP errors are returned, not raised."""
    req = urllib.request.Request(BASE + path, headers=headers or {})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()

def post(body, stream=False):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth()})
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

# The app and the CLI download the same pinned files (the default variants of the catalog, src/hub/catalog.cpp).
PINNED = {
    "qwen3.6-35b-a3b": ("Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed", "95460516fbfe34c48f577f67c9ed191f5d61fbaf",
                        [("dflash2", "incoai/Qwen3.6-35B-A3B-DFlash2", "51ef7b6923ad6c14cb1bb41c37a9041446496ab6", "dflash2")]),
    "qwen3.8-27b": ("lmstudio-community/Qwen3.8-27B-MLX-4bit", "6067b15cf581666a4aecf6af3afaba4bb5efc20c",
                    [("mtp", "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed", "1d5087d2062c02b279180a53e4016cf9cd7a3d7e", "."),
                     ("dflash2", "incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5", "dflash2")]),
    "qwen3.5-4b": ("lmstudio-community/Qwen3.5-4B-MLX-4bit", "c43ee1d65576a5d98de1e8405cac93c371a655c1",
                   [("mtp-sidecar", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", ".")]),
}

def cli_checks(kapitsa):
    home = tempfile.mkdtemp(prefix="kapitsa-test-")
    # a closed port as the Hub: any network access in these checks fails loudly
    env = dict(os.environ, KAPITSA_HOME=home, HF_ENDPOINT="http://127.0.0.1:9", NI_LOG="0")
    def run(*a):
        p = subprocess.run([kapitsa, *a], env=env, capture_output=True, text=True, timeout=120)
        return p.returncode, p.stdout, p.stderr
    def plan(name):
        rc, out, err = run("pull", "--dry-run", "--json", name)
        return json.loads(out) if rc == 0 else {"rc": rc, "err": err}
    models = os.path.join(home, "models")
    rc, out, _ = run("list", "--json")
    check("cli: list --json on an empty store", rc == 0 and json.loads(out) == [], out)
    for name, aliases in [("qwen3.6-35b-a3b", ["35b", "qwen3.6-35b", "QWEN3.6-35B-A3B", "35b:4bit",
                                               "Youssofal/Qwen3.6-35B-A3B-MTPLX-Optimized-Speed"]),
                          ("qwen3.8-27b", ["27b", "27B:4BIT", "lmstudio-community/Qwen3.8-27B-MLX-4bit",
                                           "hf.co/lmstudio-community/Qwen3.8-27B-MLX-4bit"]),
                          ("qwen3.5-4b", ["4b", "qwen3.5-4b", "qwen3.5-4b:4bit",
                                          "lmstudio-community/Qwen3.5-4B-MLX-4bit@c43ee1d65576a5d98de1e8405cac93c371a655c1"])]:
        repo, rev, comps = PINNED[name]
        for a in aliases:
            j = plan(a)
            got = [(c["kind"], c["repo"], c["revision"], c["into"]) for c in j.get("companions", [])]
            check(f"cli: {a} -> {name}:4bit at the app's pinned revisions",
                  j.get("model") == name + ":4bit" and j.get("id") == name and j.get("kind") == "catalog" and
                  j.get("variant") == "4bit" and j.get("dir") == os.path.join(models, name + "-4bit") and
                  j.get("repo") == repo and j.get("revision") == rev and got == comps and j.get("downloaded") is False,
                  json.dumps(j))
    rc, _, err = run("pull", "--dry-run", "4b:9bit")
    check("cli: an unknown variant is refused with the model's variants",
          rc == 2 and "has no variant" in err and "4bit (the default)" in err, err)
    j = plan("mlx-community/Qwen3.5-9B-4bit@main")
    check("cli: org/name@rev is a Hugging Face repo",
          j.get("kind") == "repo" and j.get("id") == "mlx-community/Qwen3.5-9B-4bit" and j.get("revision") == "main" and
          j.get("dir") == os.path.join(models, "mlx-community--Qwen3.5-9B-4bit@main") and j.get("companions") == [], json.dumps(j))
    j = plan("https://huggingface.co/mlx-community/Qwen3.5-9B-4bit")
    check("cli: a huggingface.co URL is a repo", j.get("kind") == "repo" and j.get("revision") == "main", json.dumps(j))
    folder = os.path.join(home, "my-model")
    os.makedirs(folder)
    with open(os.path.join(folder, "config.json"), "w") as f:
        f.write("{}")
    j = plan(folder)
    check("cli: a folder is used as is", j.get("kind") == "local" and j.get("dir") == folder and j.get("id") == "my-model", json.dumps(j))
    for bad in ["llama-70b", "./no-such-folder", "org/"]:
        rc, _, err = run("pull", "--dry-run", bad)
        check(f"cli: {bad!r} is refused with the choices", rc == 2 and ("Unknown model" in err or "No folder" in err) and "35b" in err, err)
    rc, _, err = run("pull", "some-org/some-model")
    check("cli: a failed first pull leaves no folder behind",
          rc == 1 and not os.path.exists(os.path.join(models, "some-org--some-model")), err)
    # a finished download (manifest written last) is reused without any network access, listed, and removed
    m4 = os.path.join(models, "qwen3.5-4b-4bit")
    os.makedirs(os.path.join(m4, ".kapitsa"))
    with open(os.path.join(m4, "config.json"), "w") as f:
        f.write("{}")
    with open(os.path.join(m4, "mtp.safetensors"), "wb") as f:
        f.write(b"x" * 10)
    with open(os.path.join(m4, ".kapitsa", "manifest.json"), "w") as f:
        repo, rev, comps = PINNED["qwen3.5-4b"]
        json.dump({"name": "qwen3.5-4b:4bit", "variant": "4bit", "repo": repo, "revision": rev, "commit": rev,
                   "files": [{"path": "config.json", "size": 2}, {"path": "mtp.safetensors", "size": 10}],
                   "companions": [{"kind": k, "repo": r, "revision": v, "files": ["mtp.safetensors"]} for k, r, v, _ in comps]}, f)
    rc, out, err = run("pull", "4b")
    check("cli: a finished download is reused offline", rc == 0 and "already downloaded" in out, out + err)
    rc, out, _ = run("list", "--json")
    lst = json.loads(out) if rc == 0 else []
    check("cli: list shows it complete, with its variant and MTP head",
          len(lst) == 1 and lst[0]["name"] == "qwen3.5-4b:4bit" and lst[0]["variant"] == "4bit" and lst[0]["complete"] and
          "MTP head" in lst[0]["includes"], out)
    os.remove(os.path.join(m4, "mtp.safetensors"))
    rc, out, _ = run("list", "--json")
    check("cli: a missing file makes it incomplete", rc == 0 and not json.loads(out)[0]["complete"], out)
    rc, out, err = run("pull", "4b")
    check("cli: completing it needs the Hub (offline here: a clear error)", rc == 1 and "Can't reach" in err, out + err)
    rc, out, _ = run("list", "--json")
    check("cli: a failed repair keeps what is there", rc == 0 and os.path.exists(os.path.join(m4, ".kapitsa", "manifest.json")), out)
    rc, out, _ = run("rm", "4b")
    check("cli: rm deletes it", rc == 0 and "Deleted qwen3.5-4b:4bit" in out and not os.path.exists(m4), out)
    rc, _, err = run("rm", folder)
    check("cli: rm leaves a folder of yours alone", rc == 2 and os.path.exists(folder), err)
    # the port is checked before the model loads
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    busy = s.getsockname()[1]
    t0 = time.time()
    rc, _, err = run("serve", "--model", folder, "--port", str(busy))
    check("cli: a busy port fails at once, before loading", rc == 1 and "already in use" in err and time.time() - t0 < 10, err)
    s.close()
    rc, _, err = run("serve", "--model", "nonsense")
    check("cli: serve with an unknown model", rc == 2 and "Unknown model" in err, err)
    rc, _, err = run("serve", "--port", str(busy))
    check("cli: serve with no model, offline and nothing downloaded: a clear error", rc == 1 and "Can't reach" in err, err)
    rc, out, _ = run("--help")
    check("cli: --help names the commands and models", rc == 0 and all(w in out for w in ("serve", "pull", "list", "35b", "27b", "4b")), out)

if CLI:
    cli_checks(CLI)
    print("FAILED: %d" % fails if fails else "ALL PASSED")
    sys.exit(1 if fails else 0)

# 0. the chat page and the model list
st, hd, page = get("/")
html = page.decode("utf-8", "replace")
check("GET / serves the chat page", st == 200 and hd.get("Content-Type", "").startswith("text/html") and
      "<title>Kapitsa</title>" in html and "/v1/chat/completions" in html, f"{st} {hd.get('Content-Type')}")
check("chat page is self-contained (no external requests)",
      not re.search(r"""(src|href)\s*=\s*["']?(https?:)?//|@import|url\(\s*["']?(https?:)?//""", html, re.I) and
      "default-src 'none'" in hd.get("Content-Security-Policy", "") and len(page) < 64 * 1024,
      hd.get("Content-Security-Policy", "") + f" ({len(page)} bytes)")
st, _, body = get("/v1/models", auth())
models_ok = st == 200 and json.loads(body)["data"][0]["id"]
check("/v1/models lists the model" + (f" as {MODEL_ID}" if MODEL_ID else ""),
      bool(models_ok) and (not MODEL_ID or json.loads(body)["data"][0]["id"] == MODEL_ID), body[:200].decode())
if API_KEY:
    check("--api-key: the page loads without a key", get("/")[0] == 200)
    check("--api-key: the API needs it", get("/v1/models")[0] == 401 and get("/v1/models", auth())[0] == 200)

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
