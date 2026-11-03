#!/usr/bin/env python3
# Flint's decode at temperature 0 with prompt lookup (--spec 7), its verify attention on flint_attn_partial_rows
# (NI_FLINT_ATTN_T=0) and on flint_attn_t_rows (the default), interleaved, beside plain decoding (no --spec): the same cold
# documents of the long corpus (~133K and ~252K tokens) and the same question, 512 tokens. The arms share an SSD prefix
# cache: the first server prefills each document once, the rest restore it. usage: ab_t0.py KAPITSA_BIN MODEL_DIR CORPUS OUT_DIR
import json, os, re, subprocess, sys, time, urllib.request

kbin, model, corpus, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
text = open(corpus, encoding="utf-8").read()
DOCS = {"133K": text[:548000], "252K": text[:1040000]}
Q = "\n\nWhat does the code above do? Describe its main classes in a few sentences."
cache = os.path.join(out, "prefix-cache")


def arm(i, tag, env_extra, args):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
    env.update(env_extra)
    env["NI_DISK_CACHE_DIR"] = cache
    env["NI_LOCK_HELD"] = "1"  # (run under lockf on the GPU lock)
    log = os.path.join(out, f"arm{i}-{tag}.log")
    p = subprocess.Popen([kbin, "serve", "--model", model, "--model-id", "m", "--host", "127.0.0.1",
                          "--port", "8298", "--ctx", "262144", "--slots", "1", *args], env=env, stdout=subprocess.DEVNULL,
                         stderr=open(log, "w"))
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:8298/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(1)
        res = {}
        for name, doc in DOCS.items():
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": doc + Q}], "temperature": 0,
                               "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False},
                               "reasoning_effort": "none"}).encode()
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8298/v1/chat/completions", body,
                                                              {"Content-Type": "application/json"}), timeout=7200)
            res[name] = json.load(r)["choices"][0]["message"]["content"]
        return res, log
    finally:
        p.terminate()
        p.wait(timeout=120)


rows = []
texts = {}
ARMS = [("plain", {}, []), ("old", {"NI_FLINT_ATTN_T": "0"}, ["--spec", "7"]), ("new", {}, ["--spec", "7"]),
        ("old", {"NI_FLINT_ATTN_T": "0"}, ["--spec", "7"]), ("new", {}, ["--spec", "7"])]
for i, (tag, e, args) in enumerate(ARMS):
    res, log = arm(i, tag, e, args)
    texts[i] = res
    for line in open(log):
        m = re.search(r"chat: prompt (\d+) \(cached (\d+)\).*?(\d+) tokens in (\d+) ms \(([\d.]+) tok/s\), (\w+)(?:, ([\d.]+) tok/cycle)?", line)
        if m:
            p, c, n, ms, tps, fin, tpc = m.groups()
            rows.append(dict(arm=i, tag=tag, prompt=int(p), cached=int(c), tokens=int(n), ms=int(ms), tps=float(tps),
                             tok_per_cycle=float(tpc) if tpc else None, finish=fin))
            print(f"arm {i} {tag}: prompt {p} (cached {c}): {n} tok in {ms} ms, {tps} tok/s, {tpc or '-'} t/c, {fin}", flush=True)
same = {k: [texts[i][k] == texts[0][k] for i in texts] for k in DOCS}
print("texts equal to the plain arm's:", same)
json.dump({"rows": rows, "identical": same, "texts": texts[0]}, open(os.path.join(out, "ab_t0.json"), "w"), indent=1)
