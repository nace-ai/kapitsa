#!/usr/bin/env python3
"""A new conversation takes the one slot from a long one the SSD does not hold yet: main (no cache for Nemotron-H)
against the branch (cache on, as by default), interleaved A B A B A B. Run under the GPU lock
(lockf ... env NI_LOCK_HELD=1 ...).
usage: ab_nohit.py KAPITSA_MAIN KAPITSA_BRANCH MODEL_DIR CORPUS [--pairs 3] [--tokens 16000] [--tokens2 2000]
                   [--gap 3] [--port 8197] [--log FILE]

Each arm is a fresh server with its own empty KAPITSA_HOME (so an empty cache), started the same way with no
--disk-cache option: `serve --model M --port P --ctx 32768 --slots 1`. In it: a short warm-up request; request 1 (a
--tokens prompt, 64 greedy tokens); --gap s without requests (less than the 10 s after which the branch's write-behind
saves an idle slot, so request 1's conversation is not on the SSD yet); request 2 (another conversation of --tokens2,
64 greedy tokens), with no warm-up before it: it takes the slot, and the branch copies request 1's conversation for the
SSD before replacing it (src/engine/engine_disk.cpp, disk_leaving). Printed per request: the client's time to the first
token and to the last, and the server's own log line; per arm, the cache's log lines and whether both arms wrote the
same text."""
import json, os, re, shutil, signal, subprocess, sys, tempfile, time, urllib.error, urllib.request

A = sys.argv[1:]


def opt(name, default):
    if name in A:
        i = A.index(name)
        v = A[i + 1]
        del A[i:i + 2]
        return v
    return default


PAIRS, PORT = int(opt("--pairs", "3")), int(opt("--port", "8197"))
TOKENS, TOKENS2, GAP = int(opt("--tokens", "16000")), int(opt("--tokens2", "2000")), float(opt("--gap", "3"))
log_path = os.path.abspath(opt("--log", "ab_nohit-serve.log"))
MAIN, BRANCH, MODEL, CORPUS = [os.path.abspath(os.path.expanduser(x)) for x in A[:4]]
BASE = f"http://127.0.0.1:{PORT}"
with open(CORPUS, errors="replace") as f:
    TEXT = f.read()


def prompt(k, tokens):
    n = int(tokens * 4.25)
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
    return text, first, time.time() - t0


def arm(name, binary):
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
    chat([{"role": "user", "content": "Say hello in five words."}], 8)
    out = [chat(prompt(201, TOKENS), 64)]
    time.sleep(GAP)
    out.append(chat(prompt(202, TOKENS2), 64))
    time.sleep(1)
    p.send_signal(signal.SIGKILL)  # (nothing saved at stop: the next arm's server starts at once)
    p.wait()
    log.close()
    with open(log_path, errors="replace") as f:
        f.seek(at)
        mine = f.read()
    shutil.rmtree(home, ignore_errors=True)
    lines = [m for m in re.finditer(r"chat: prompt (\d+) \(cached (\d+)\) in (\d+) ms .*?, (\d+) tokens in (\d+) ms "
                                    r"\(([\d.]+) tok/s\)", mine) if int(m.group(1)) > 1000]
    for k, ((text, first, total), m) in enumerate(zip(out, lines), 1):
        print(f"{name} request {k}: {m.group(1)} prompt tokens ({m.group(2)} cached), first token {first:.3f} s, "
              f"last {total:.3f} s; server: prefill {m.group(3)} ms, {m.group(4)} tokens at {m.group(6)} tok/s",
              flush=True)
    for line in re.findall(r"\[kapitsa\] (SSD (?:prefix )?cache: .*)", mine):
        print(f"{name}   log: {re.sub(r'/[^ ]*/prefix-cache/', '<cache>/', line)[:220]}", flush=True)
    return [t for t, _, _ in out]


print(f"model {MODEL}, request 1 ~{TOKENS} tokens, request 2 ~{TOKENS2} tokens {GAP:g} s later; servers' log: "
      f"{log_path}", flush=True)
same = True
for rnd in range(1, PAIRS + 1):
    a = arm(f"round {rnd} main  ", MAIN)
    b = arm(f"round {rnd} branch", BRANCH)
    ok = a == b
    same = same and ok
    print(f"round {rnd}: both arms write the same text: {'yes' if ok else 'no'}", flush=True)
print("ALL THE SAME TEXT" if same else "TEXT DIFFERS", flush=True)
