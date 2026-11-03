#!/usr/bin/env python3
# Temperature-0 continuations of Dory with its decode / verify attention one key at a time (NI_DORY_ATTN_K8=0) and 8 at
# a time (the default): the same cold documents from the long corpus (about 4K and 16K tokens) and the same question,
# 512 tokens each; prints where the two texts first differ. usage: greedy_check.py KAPITSA_BIN MODEL_DIR CORPUS OUT_JSON
import json, os, subprocess, sys, time, urllib.request

kbin, model, corpus, out = sys.argv[1:5]
text = open(corpus, encoding="utf-8").read()
DOCS = {"4K": text[:17000], "16K": text[:68000]}
Q = "\n\nWhat does the code above do? Describe its main classes in a few sentences."


def run(env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
    env.update(env_extra)
    env["NI_DISK_CACHE_DIR"] = os.path.join(os.path.dirname(out), "greedy-cache")
    env["NI_LOCK_HELD"] = "1"  # (run under lockf on the GPU lock)
    p = subprocess.Popen([kbin, "serve", "--model", model, "--model-id", "m", "--loop-kv", "per", "--host", "127.0.0.1",
                          "--port", "8299", "--ctx", "65536", "--slots", "1", "--disk-cache", "0"], env=env,
                         stdout=subprocess.DEVNULL, stderr=open(out + f".{env_extra.get('NI_DORY_ATTN_K8', '1')}.log", "w"))
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:8299/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(1)
        res = {}
        for name, doc in DOCS.items():
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": doc + Q}], "temperature": 0,
                               "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}).encode()
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8299/v1/chat/completions", body,
                                                              {"Content-Type": "application/json"}), timeout=3600)
            res[name] = json.load(r)["choices"][0]["message"]["content"]
        return res
    finally:
        p.terminate()
        p.wait(timeout=60)


old = run({"NI_DORY_ATTN_K8": "0"})
new = run({})
report = {}
for name in DOCS:
    a, b = old[name], new[name]
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
    if i is None and len(a) != len(b):
        i = min(len(a), len(b))
    report[name] = {"identical": a == b, "first_diff_char": i, "chars": [len(a), len(b)], "old": a, "new": b}
    print(f"{name}: {'identical' if a == b else f'differ from character {i} of {len(a)} / {len(b)}'}")
json.dump(report, open(out, "w"), indent=1)
