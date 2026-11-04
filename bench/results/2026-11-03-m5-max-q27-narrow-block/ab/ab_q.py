#!/usr/bin/env python3
# Decode A/B of a Qwen verify change, the arms set by one switch (AB_SW, its value 0 = old; default NI_V8T_STAGE),
# interleaved, at temperature 0: the same cold documents of the long corpus (~133K and ~252K tokens) and the same
# question, 512 tokens; AB_SLOTS slots (default 1); arms before AB_WARM only fill the SSD cache, which every arm
# shares. Every arm's text is kept and compared with the first measured arm's. Each request waits for a quiet GPU (no
# foreign process above 5% for 15 s, at most 60 s; bench/harness/gpu_hogs.py, AB_HARNESS its folder) and samples the
# foreign GPU load every 2 s while it runs (the server's process tree is never foreign); both go into the output.
# usage: ab_q.py KAPITSA_BIN MODEL_DIR CORPUS OUT_DIR ARMS_JSON (a list of environments, one per arm)
import json, os, re, subprocess, sys, time, urllib.request

sys.path.insert(0, os.environ["AB_HARNESS"])  # <repo>/bench/harness: gpu_hogs, the harness's GPU load monitor
import gpu_hogs  # noqa: E402

LOAD = {}  # "arm-doc": the request's quiet wait and the foreign GPU load sampled while it ran

kbin, model, corpus, out = sys.argv[1:5]  # ARMS_JSON: sys.argv[5]
os.makedirs(out, exist_ok=True)
text = open(corpus, encoding="utf-8").read()
DOCS = {k: text[:int(v)] for k, v in json.loads(os.environ.get("AB_DOCS", '{"133K": 548000, "252K": 1040000}')).items()}
Q = "\n\nWhat does the code above do? Describe its main classes in a few sentences."
cache = os.path.join(out, "prefix-cache")


def arm(i, env_extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
    env.update(env_extra)
    env["NI_DISK_CACHE_DIR"] = cache
    env["NI_LOCK_HELD"] = "1"  # (run under lockf on the GPU lock)
    log = os.path.join(out, f"arm{i}-" + ('old' if env_extra.get(os.environ.get('AB_SW', 'NI_V8T_STAGE')) == '0' else 'new') + ".log")
    p = subprocess.Popen([kbin, "serve", "--model", model, "--model-id", "m", "--host", "127.0.0.1",
                          "--port", "8298", "--ctx", "262144", "--slots", os.environ.get("AB_SLOTS", "1")], env=env, stdout=subprocess.DEVNULL,
                         stderr=open(log, "w"))
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen("http://127.0.0.1:8298/v1/models", timeout=2)
                break
            except Exception:
                time.sleep(1)
        res = {}
        mon = gpu_hogs.ContentionMonitor(lambda: [p.pid])  # (the server's process tree is never foreign)
        mon.start()
        for name, doc in DOCS.items():
            # each request after a quiet GPU: no foreign process above 5% for 15 s (at most 60 s of waiting)
            ok, waited, busy = gpu_hogs.wait_quiet(5.0, 15, 60, exclude_roots=lambda: [p.pid])
            t0 = time.time()
            body = json.dumps({"model": "m", "messages": [{"role": "user", "content": doc + Q}], "temperature": 0,
                               "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False},
                               "reasoning_effort": "none"}).encode()
            r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8298/v1/chat/completions", body,
                                                              {"Content-Type": "application/json"}), timeout=7200)
            res[name] = json.load(r)["choices"][0]["message"]["content"]
            LOAD[f"{i}-{name}"] = dict(quiet=ok, quiet_wait_s=round(waited, 1),
                                       busy_at_timeout=[f"{n} {pct:.0f}%" for pct, (_, n) in busy],
                                       **gpu_hogs.summarize_samples(mon.samples, t0, time.time()))
        mon.stop()
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
    k = 0
    for line in open(log):
        m = re.search(r"chat: prompt (\d+) \(cached (\d+)\).*?(\d+) tokens in (\d+) ms \(([\d.]+) tok/s\), (\w+), ([\d.]+) tok/cycle.*?: (\d+) cycles", line)
        if m:
            p, c, n, ms, tps, fin, tpc, cyc = m.groups()
            rows.append(dict(arm=i, switch=e.get(os.environ.get('AB_SW', 'NI_V8T_STAGE'), '1'), prompt=int(p), cached=int(c), tokens=int(n), ms=int(ms), tps=float(tps),
                             tok_per_cycle=float(tpc), cycles=int(cyc), ms_per_cycle=round(int(ms) / int(cyc), 2)))
            ld = LOAD.get(f"{i}-{list(DOCS)[k]}", {}) if k < len(DOCS) else {}
            rows[-1]["load"] = ld
            k += 1
            print(f"arm {i} {'old' if e.get(os.environ.get('AB_SW', 'NI_V8T_STAGE')) == '0' else 'new'}: prompt {p} (cached {c}): {n} tok, {tps} tok/s, {tpc} t/c, "
                  f"{int(ms) / int(cyc):.1f} ms/cycle | quiet {ld.get('quiet')} after {ld.get('quiet_wait_s')} s, foreign GPU "
                  f"mean {ld.get('mean_foreign_gpu_pct')}% max {ld.get('max_foreign_gpu_pct')}% top {ld.get('top')}", flush=True)
W = int(os.environ.get("AB_WARM", "0"))  # arms before W only warm the SSD cache
same = {k: all(texts[i][k] == texts[W][k] for i in texts if i >= W) for k in DOCS}
print(f"texts identical across arms {W}..:", same)
json.dump({"arms": ARMS, "warm_arms": W, "rows": rows, "identical": same, "load": LOAD, "texts": texts}, open(os.path.join(out, "ab_t0.json"), "w"), indent=1)
