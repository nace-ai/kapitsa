#!/usr/bin/env python3
"""sampled_summary.py DIR: sampled.sh's runs in DIR (K-BUILD-PLAN-MODE.json): per plan, each run's tok/s in all
(prefills included) and per-request decode tok/s; texts of every run against the first pr run's alone replies."""
import glob, json, os, re, sys
d = sys.argv[1]
runs = {}
for f in sorted(glob.glob(os.path.join(d, "*-*-*-*.json"))):
    m = re.match(r"(\d)-(main|pr)-(seeded|ctx6k)-(alone|together)\.json", os.path.basename(f))
    if m:
        runs[m.groups()] = json.load(open(f))
for plan in ("seeded", "ctx6k"):
    print(f"== {plan}")
    ref = runs.get(("2", "pr", plan, "alone"))
    for (k, b, p, mode), r in sorted(runs.items()):
        if p != plan:
            continue
        reqs = {n: v for n, v in r.items() if n != "_run"}
        assert not any(v["error"] for v in reqs.values()), (k, b, p, mode)
        same = [n for n in reqs if ref and reqs[n]["text"] == ref[n]["text"]]
        desc = []
        for n, v in reqs.items():
            t = "same" if n in same else "differs"
            if n not in same and ref:
                x, y = v["text"], ref[n]["text"]
                i = next((i for i, (a, c) in enumerate(zip(x, y)) if a != c), min(len(x), len(y)))
                t += f" at char {i}"
            desc.append(f"{n}: {v['tokens']} tok, prompt {v['prompt_tokens']}, ttft {v['ttft']}, {v['decode_tps']} tok/s, {t}")
        print(f"{k}-{b} {mode:8s} total {r['_run']['total_tps']} tok/s, span {r['_run']['span_s']} s")
        for x in desc:
            print("    " + x)
