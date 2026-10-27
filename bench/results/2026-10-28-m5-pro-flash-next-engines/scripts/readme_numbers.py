#!/usr/bin/env python3
"""Figures quoted in README prose, recomputed from legs/summary.json and the leg JSONs. Usage: readme_numbers.py [dir]"""
import glob, json, sys
d = sys.argv[1] if len(sys.argv) > 1 else "."
A = json.load(open(d + "/legs/summary.json"))["arms"]
med = lambda a, k: A[a][k]["median"]
def spread(arms, keys):
    v = [x for a in arms for k in keys for x in A[a][k]["values"]]
    return min(v), max(v)
def pct(a, b, k): return 100 * (med(a, k) - med(b, k)) / med(b, k)
SPL = ["K", "K2", "S", "S2"]
out = {}
out["decode_all_min"], out["decode_all_max"] = spread(SPL, ["decode_T0_tps", "decode_T06_tps"])
cells = [A[a][k]["max"] - A[a][k]["min"] for a in SPL for k in ("decode_T0_tps", "decode_T06_tps")]
out["decode_cell_spread_min"], out["decode_cell_spread_max"] = min(cells), max(cells)
for k in ("decode_T0_tps", "decode_T06_tps"):
    m = [med(a, k) for a in SPL]; out[k + "_median_gap"] = max(m) - min(m)
for s in ("1K", "4K", "16K"):
    k = "prefill_%s_tps" % s; m = [med(a, k) for a in SPL]
    out["prefill_%s_max_gap_pct" % s] = 100 * (max(m) - min(m)) / max(m)
out["16K_K2_vs_K_pct"], out["16K_K2_vs_S_pct"], out["16K_S2_vs_S_pct"] = (pct("K2", "K", "prefill_16K_tps"), pct("K2", "S", "prefill_16K_tps"), pct("S2", "S", "prefill_16K_tps"))
out["turn2_prompt_tokens_splash"] = sorted({int(med(a, "turn2_prompt_tokens")) for a in SPL})
out["turn2_cached_splash"] = sorted({int(med(a, "turn2_cached_tokens")) for a in SPL})
L = {}
for f in sorted(glob.glob(d + "/legs/leg*-L.json")):
    j = json.load(open(f))
    if j.get("idx", 0) in (0,): continue
    L[f.split("/")[-1]] = (round(j["memory"]["swap_gb_start"], 2), round(j["memory"]["free_reclaimable_gib_min"], 1))
out["L_legs_swap_start_gb_and_min_free_gib"] = L
json.dump(out, sys.stdout, indent=1, default=lambda x: round(x, 2)); print()
