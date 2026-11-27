#!/usr/bin/env python3
# The rounds of run2/ and run3/ (ab_lc.py's ab.json) in which both runs found the GPU quiet, ended below load 4 and drew
# from AC power before and after, and those whose runs also saw 0.0% foreign GPU load (ab_lc.py's "clean"); medians.
# usage: pick.py RUN_DIR...
import json, statistics, sys

rounds = []
for d in sys.argv[1:]:
    by = {}
    for r in json.load(open(f"{d}/ab.json")):
        by.setdefault(r["round"], {})[r["arm"]] = r
    rounds += [(d, k, v) for k, v in sorted(by.items()) if len(v) == 2]


def ok(r):
    return r["quiet"] and r["load1_after"] < 4 and r["ac_before"] and r["ac_after"]


for name, test in (("quiet, load and AC", ok), ("also 0.0% foreign GPU", lambda r: r["clean"])):
    sel = [(d, k, v) for d, k, v in rounds if test(v["old"]) and test(v["new"])]
    print(f"{name}: {len(sel)} rounds of {len(rounds)}")
    for d, k, v in sel:
        print(f"  {d} round {k}: old {v['old']['tok_s']} tok/s {v['old']['gpu_ms_cycle']} ms, "
              f"new {v['new']['tok_s']} tok/s {v['new']['gpu_ms_cycle']} ms, foreign GPU mean "
              f"{v['old']['gpu_load']['mean_foreign_gpu_pct']}% / {v['new']['gpu_load']['mean_foreign_gpu_pct']}%")
    if sel:
        for m in ("tok_s", "gpu_ms_cycle"):
            o = statistics.median(v["old"][m] for _, _, v in sel)
            n = statistics.median(v["new"][m] for _, _, v in sel)
            print(f"  median {m}: old {o:.3f}, new {n:.3f} ({(n / o - 1) * 100:+.2f}%)")
