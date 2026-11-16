#!/usr/bin/env python3
"""Requests the SSD prefix cache has nothing for: main (no cache for Nemotron-H) against the branch (cache on, as by
default), interleaved A B A B A B. Run under the GPU lock (lockf ... env NI_LOCK_HELD=1 ...).
usage: ab.py KAPITSA_MAIN KAPITSA_BRANCH MODEL_DIR CORPUS [--pairs 3] [--tokens 4000] [--port 8197] [--log FILE]

Each arm is a fresh server with its own empty KAPITSA_HOME (so an empty cache), started the same way with no
--disk-cache option: `serve --model M --port P --ctx 32768 --slots 1`. In it: a short warm-up request; request 1 (a
long prompt, 256 greedy tokens); 15 s without requests (the branch's write-behind saves the conversation after 10 s);
a warm-up again; request 2 (another long prompt, which replaces request 1's conversation in the one slot). Printed per
request: the client's time to the first token and the server's own log line (prefill ms, decode tok/s); per arm, the
cache's log lines and whether both arms wrote the same text."""
import json, os, re, shutil, signal, subprocess, sys, tempfile, time, urllib.error, urllib.request

A = sys.argv[1:]


def opt(name, default):
    if name in A:
        i = A.index(name)
        v = A[i + 1]
        del A[i:i + 2]
        return v
    return default


PAIRS, TOKENS, PORT = int(opt("--pairs", "3")), int(opt("--tokens", "4000")), int(opt("--port", "8197"))
log_path = os.path.abspath(opt("--log", "ab-serve.log"))
MAIN, BRANCH, MODEL, CORPUS = [os.path.abspath(os.path.expanduser(x)) for x in A[:4]]
BASE = f"http://127.0.0.1:{PORT}"
with open(CORPUS, errors="replace") as f:
    TEXT = f.read()


def prompt(k):
    n = int(TOKENS * 4.25)
    off = (k * 7919) % max(1, len(TEXT) - n)
    return [{"role": "system", "content": f"You are a careful code reviewer (session {k})."},
            {"role": "user", "content": "Here is some source code:\n\n" + TEXT[off:off + n] +
                                        "\n\nDescribe what this code does, part by part, in about 200 words."}]


def chat(messages, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, first, text = time.time(), None, ""
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[6:]).get("choices", []):
                d = ch.get("delta", {})
                if ((d.get("content") or "") + (d.get("reasoning_content") or "")) and first is None:
                    first = time.time() - t0
                text += d.get("content") or ""
    return text, first


def arm(name, binary, log_path):
    home = tempfile.mkdtemp(prefix="kapitsa-ab-")
    e = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
    e.update({"KAPITSA_HOME": home, "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1"})
    log = open(log_path, "a")
    log.write(f"\n==== {name}\n")
    log.flush()
    at = os.path.getsize(log_path)
    p = subprocess.Popen([binary, "serve", "--model", MODEL, "--port", str(PORT), "--ctx", "32768", "--slots", "1"],
                         env=e, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        if p.poll() is not None or time.time() - t0 > 600:
            raise SystemExit(f"{name}: the server did not start; its log: {log_path}")
        try:
            if urllib.request.urlopen(BASE + "/health", timeout=2).status == 200:
                break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    time.sleep(3)  # (the cache's files open in the background, right after the model loads)
    out = []
    for k in (1, 2):
        chat([{"role": "user", "content": "Say hello in five words."}], 8)
        text, first = chat(prompt(100 + k), 256)
        out.append((text, first))
        if k == 1:
            time.sleep(15)
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    log.close()
    with open(log_path, errors="replace") as f:
        f.seek(at)
        mine = f.read()
    shutil.rmtree(home, ignore_errors=True)
    lines = [m for m in re.finditer(r"chat: prompt (\d+) \(cached (\d+)\) in (\d+) ms .*?, (\d+) tokens in (\d+) ms "
                                    r"\(([\d.]+) tok/s\)", mine) if int(m.group(1)) > 1000]
    for k, ((text, first), m) in enumerate(zip(out, lines), 1):
        print(f"{name} request {k}: {m.group(1)} prompt tokens ({m.group(2)} cached), first token {first:.3f} s, "
              f"prefill {m.group(3)} ms, {m.group(4)} tokens at {m.group(6)} tok/s", flush=True)
    for line in re.findall(r"\[kapitsa\] (SSD (?:prefix )?cache: .*)", mine):
        print(f"{name}   log: {re.sub(r'/[^ ]*/prefix-cache/', '<cache>/', line)[:200]}", flush=True)
    return [t for t, _ in out]


print(f"model {MODEL}, prompts of ~{TOKENS} tokens; servers' log: {log_path}", flush=True)
same = True
for rnd in range(1, PAIRS + 1):
    a = arm(f"round {rnd} main  ", MAIN, log_path)
    b = arm(f"round {rnd} branch", BRANCH, log_path)
    ok = a == b
    same = same and ok
    print(f"round {rnd}: both arms write the same text: {'yes' if ok else 'no'}", flush=True)
print("ALL THE SAME TEXT" if same else "TEXT DIFFERS", flush=True)
