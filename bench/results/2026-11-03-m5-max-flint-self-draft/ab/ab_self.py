#!/usr/bin/env python3
# Flint's self-drafts (NI_FLINT_SELF=K drafts a cycle, NI_FLINT_SELF_WIN=W: the drafts' full-attention window) against
# plain decoding: the same cold documents of the long corpus (~133K and ~252K tokens) and the same question, 512 tokens,
# at temperature 0 and at the model's sampling (T 1.0, top-p 0.95, top-k 20, seed 7). One server per arm; the arms share
# an SSD prefix cache. usage: ab_self.py KAPITSA_BIN MODEL_DIR CORPUS OUT_DIR ARMS_JSON
import json, os, re, subprocess, sys, time, urllib.request

kbin, model, corpus, out, arms_json = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
text = open(corpus, encoding="utf-8").read()
DOCS = {k: text[:int(v)] for k, v in json.loads(os.environ.get("AB_DOCS", '{"133K": 548000, "252K": 1040000}')).items()}
Q = "\n\nWhat does the code above do? Describe its main classes in a few sentences."
SAMPLING = {"t0": {"temperature": 0}, "t1": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": 7}}
SAMPLING = {k: v for k, v in SAMPLING.items() if k in os.environ.get("AB_SAMPLING", "t0,t1").split(",")}
cache = os.path.join(out, "prefix-cache")
ARMS = json.loads(arms_json)  # [[tag, {env}], ...]


def arm(i, tag, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
    env.update(env_extra)
    env["NI_DISK_CACHE_DIR"] = cache
    env["NI_LOCK_HELD"] = "1"  # (run under lockf on the GPU lock)
    log = os.path.join(out, f"arm{i}-{tag}.log")
    p = subprocess.Popen([kbin, "serve", "--model", model, "--model-id", "m", "--host", "127.0.0.1", "--port", "8298",
                          "--ctx", "262144", "--slots", "1"], env=env, stdout=subprocess.DEVNULL, stderr=open(log, "w"))
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:8298/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(1)
        res = {}
        for name, doc in DOCS.items():
            for sname, s in SAMPLING.items():
                body = json.dumps({"model": "m", "messages": [{"role": "user", "content": doc + Q}], "max_tokens": 512,
                                   **s}).encode()
                r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8298/v1/chat/completions", body,
                                                                  {"Content-Type": "application/json"}), timeout=7200)
                msg = json.load(r)["choices"][0]["message"]
                res[f"{name}-{sname}"] = (msg.get("reasoning_content") or "") + "\n</think>\n" + (msg.get("content") or "")
        return res, log
    finally:
        p.terminate()
        p.wait(timeout=120)


rows, texts = [], {}
for i, (tag, e) in enumerate(ARMS):
    res, log = arm(i, tag, e)
    texts[i] = res
    keys = list(res)
    k = 0
    for line in open(log):
        m = re.search(r"chat: prompt (\d+) \(cached (\d+)\).*?(\d+) tokens in (\d+) ms \(([\d.]+) tok/s\), (\w+)(?:, ([\d.]+) tok/cycle)?", line)
        if m:
            p, c, n, ms, tps, fin, tpc = m.groups()
            key = keys[k] if k < len(keys) else "?"
            k += 1
            rows.append(dict(arm=i, tag=tag, req=key, prompt=int(p), cached=int(c), tokens=int(n), ms=int(ms),
                             tps=float(tps), tok_per_cycle=float(tpc) if tpc else None, finish=fin))
            print(f"arm {i} {tag} {key}: prompt {p} (cached {c}): {n} tok in {ms} ms, {tps} tok/s, {tpc or '-'} t/c, {fin}",
                  flush=True)
same = {k: [texts[i][k] == texts[0][k] for i in texts] for k in texts[0]}
print("texts equal to arm 0's:", same, flush=True)
json.dump({"arms": ARMS, "rows": rows, "identical": same, "texts": texts}, open(os.path.join(out, "ab_self.json"), "w"),
          indent=1)
