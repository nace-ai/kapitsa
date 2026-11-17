#!/usr/bin/env python3
"""summarize.py DIR: the interleaved A/B in DIR (pairK-main, pairK-pr): per arm the total tok/s (prefills included),
the decode-window tok/s and the per-stream decode tok/s at 1, 2 and 4 requests at once, and the single-request
decode tok/s; then the medians per build and pr / main."""
import glob, json, os, statistics, sys
d = sys.argv[1]
rows = {"main": [], "pr": []}
for arm in sorted(glob.glob(os.path.join(d, "pair*-*"))):
    name = os.path.basename(arm)
    build = "main" if name.endswith("-main") else "pr"
    r = {"arm": name}
    for n in (1, 2, 4):
        try:
            s = json.load(open(os.path.join(arm, f"streams{n}.json")))
        except Exception:
            continue
        assert not s["errors"], (arm, s["errors"])
        r[f"tot{n}"] = s["total_tps"]
        r[f"win{n}"] = s["decode_window_tps"]
        r[f"per{n}"] = s["per_stream_tps"]
        r[f"ttft{n}"] = s["ttft"]
    try:
        r["single"] = json.load(open(os.path.join(arm, "single.json")))["decode_tps"]
    except Exception:
        pass
    env = open(os.path.join(arm, "env.txt")).read().split("\n")
    r["env"] = [x for x in env if x]
    rows[build].append(r)
for b in ("main", "pr"):
    for r in rows[b]:
        print(f"{r['arm']:14s} " + "  ".join(f"n{n}: total {r.get(f'tot{n}')} window {r.get(f'win{n}')} per-stream {r.get(f'per{n}')}" for n in (1, 2, 4)) + f"  single {r.get('single')}")
print()
med = {}
for b in ("main", "pr"):
    for k in ("tot1", "win1", "tot2", "win2", "tot4", "win4"):
        v = [r[k] for r in rows[b] if k in r]
        med[(b, k)] = statistics.median(v) if v else None
    v = [x for r in rows[b] for x in (r.get("single") or []) if x]
    med[(b, "single")] = statistics.median(v) if v else None
for k in ("tot1", "win1", "tot2", "win2", "tot4", "win4", "single"):
    m, p = med[("main", k)], med[("pr", k)]
    print(f"{k:7s} main {m}  pr {p}  pr/main {p / m:.3f}" if m and p else f"{k} {m} {p}")
