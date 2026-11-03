#!/usr/bin/env python3
# Decode A/B of the window of Dory self-drafts' output layers (NI_DORY_DRAFT_WIN; unset: every position), the arms
# interleaved, at temperature 0: the same cold documents of the long corpus (~133K and ~252K tokens) and the same
# question, 512 tokens. The verify does not read the draft's keys, so every arm writes the same text; only the drafts,
# and so the tokens a cycle, differ. The arms share an SSD prefix cache (each document restored or prefilled once).
# usage: ab_dw.py KAPITSA_BIN MODEL_DIR CORPUS OUT_DIR ARMS_JSON (a list of environments, one per arm)
import json, os, re, subprocess, sys, time, urllib.request

kbin, model, corpus, out = sys.argv[1:5]  # ARMS_JSON: sys.argv[5]
os.makedirs(out, exist_ok=True)
text = open(corpus, encoding="utf-8").read()
DOCS = {"133K": text[:548000], "252K": text[:1040000]}
Q = "\n\nWhat does the code above do? Describe its main classes in a few sentences."
cache = os.path.join(out, "prefix-cache")


def arm(i, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
    env.update(env_extra)
    env["NI_DISK_CACHE_DIR"] = cache
    env["NI_LOCK_HELD"] = "1"  # (run under lockf on the GPU lock)
    log = os.path.join(out, f"arm{i}-" + (env_extra.get('NI_DORY_DRAFT_WIN', 'full')) + ".log")
    p = subprocess.Popen([kbin, "serve", "--model", model, "--model-id", "m", "--loop-kv", "per", "--host", "127.0.0.1",
                          "--port", "8298", "--ctx", "262144", "--slots", "1"], env=env, stdout=subprocess.DEVNULL,
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
ARMS = json.loads(sys.argv[5])
for i, e in enumerate(ARMS):
    res, log = arm(i, e)
    texts[i] = res
    for line in open(log):
        m = re.search(r"chat: prompt (\d+) \(cached (\d+)\).*?(\d+) tokens in (\d+) ms \(([\d.]+) tok/s\), (\w+), ([\d.]+) tok/cycle.*?: (\d+) cycles", line)
        if m:
            p, c, n, ms, tps, fin, tpc, cyc = m.groups()
            rows.append(dict(arm=i, win=e.get('NI_DORY_DRAFT_WIN', 'full'), prompt=int(p), cached=int(c), tokens=int(n), ms=int(ms), tps=float(tps),
                             tok_per_cycle=float(tpc), cycles=int(cyc), ms_per_cycle=round(int(ms) / int(cyc), 2)))
            print(f"arm {i} win {e.get('NI_DORY_DRAFT_WIN', 'full')}: prompt {p} (cached {c}): {n} tok, {tps} tok/s, {tpc} t/c, "
                  f"{int(ms) / int(cyc):.1f} ms/cycle", flush=True)
same = {k: all(texts[i][k] == texts[0][k] for i in texts) for k in DOCS}
print("texts identical across the four arms:", same)
json.dump({"rows": rows, "identical": same, "texts": texts[0]}, open(os.path.join(out, "ab_t0.json"), "w"), indent=1)
