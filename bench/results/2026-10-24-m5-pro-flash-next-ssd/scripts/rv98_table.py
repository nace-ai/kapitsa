#!/usr/bin/env python3
"""rv98_table.py DIR: #98 SSD-offload table from DIR/run_<engine>_<prompt>_<leg>.txt + <tag>.log (rv98b.sh)."""
import os, glob, json, os, re, sys, statistics as st
from collections import defaultdict

d = sys.argv[1]
rows = []
for p in sorted(glob.glob(os.path.join(d, "run_*.txt"))):
    m = re.match(r"run_(kap|llama-gpu|llama-cpumoe)_(dec|c1k|pf4|pf16|c32k)_(c?\d+)\.txt$", os.path.basename(p))
    if not m:
        continue
    eng, pr, leg = m.groups()
    t = open(p, errors="replace").read()
    lp = os.path.join(d, f"{eng}_{pr}_{leg}.log")
    lg = open(lp, errors="replace").read() if os.path.exists(lp) else ""
    res = re.search(r"RESULT \S+ rc=(\d+) swap_delta_mb=([-\d.]+) max_pressure=(\d+) stop=(\S+)", t)
    r = dict(eng=eng, pr=pr, leg=leg, cold=leg.startswith("c"), ok=False, ptok=None, pfs=None, pms=None, dec=None,
             load=None, swap=None, toks=None, store=None, real=None, dms=None)
    rm_ = re.search(r"^\s*([\d.]+) real", lg, re.M)
    r["real"] = rm_ and float(rm_.group(1))
    if res:
        r["ok"] = res.group(1) == "0" and res.group(4) == "none"
        r["swap"] = float(res.group(2))
    if eng == "kap":
        pm = re.search(r"prompt (\d+) tok in ([\d.]+) ms", lg)
        if pm:
            r["ptok"], r["pms"] = int(pm.group(1)), float(pm.group(2))
            r["pfs"] = r["ptok"] / r["pms"] * 1e3
        dm = re.search(r"decode \d+ tok in ([\d.]+) ms \(([\d.]+) tok/s\)", lg)
        r["dec"] = dm and float(dm.group(2)); r["dms"] = dm and float(dm.group(1))
        lm = re.search(r"\[kapitsa\] loaded \S+: .*? in (\d+) ms", lg)
        r["load"] = lm and int(lm.group(1)) / 1e3
        sm = re.search(r"expert store: (\d+) slots of [\d.]+ MB \(([\d.]+) GB, ([\d.]+)% ", lg)
        r["store"] = sm and f"{sm.group(2)} GB ({sm.group(3)}%)"
        im = re.search(r"^ids:(.*)$", lg, re.M)
        r["toks"] = im and re.findall(r"\d+", im.group(1))
    else:
        tm = re.search(r"^TIMINGS (\{.*\})", lg, re.M)
        if tm:
            j = json.loads(tm.group(1))
            r["ptok"], r["pms"], r["pfs"], r["dec"] = j.get("prompt_n"), j.get("prompt_ms"), j.get("prompt_per_second"), j.get("predicted_per_second")
            r["dms"] = j.get("predicted_ms")
        lm = re.search(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) I srv\s+llama_server: model loaded", lg, re.M)
        if lm:
            r["load"] = int(lm.group(1)) * 60 + int(lm.group(2)) + int(lm.group(3)) / 1e3
        km = re.search(r"^TOKENS (.*)$", lg, re.M)
        r["toks"] = km and km.group(1).split()
    rows.append(r)

kst = sorted({r["store"] for r in rows if r["eng"] == "kap" and r["store"]})
kl = "auto store" if not os.environ.get("RV_KAP_STORE") else f"store pinned {os.environ['RV_KAP_STORE']} GB, not the auto default"
name = {"kap": f"Kapitsa main 4b087c8 (Q2_0 pack, expert-store streaming, {kl}, heater on)",
        "llama-gpu": "llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 -fa on)",
        "llama-cpumoe": "llama.cpp 0a9ef00 (ISTA Q2_0 GGUF, mmap, -ngl 99 --cpu-moe -fa on)"}
f = lambda v, fmt="%.1f": "-" if v is None else fmt % v
print("## Legs\n")
print("| engine | prompt | leg | ok | prompt tok | prefill tok/s | prompt ms | decode tok/s | load s | store | swap Δ MB |")
print("|---|---|---|---|---:|---:|---:|---:|---:|---|---:|")
for r in sorted(rows, key=lambda r: (r["pr"], r["leg"], r["eng"])):
    print(f"| {r['eng']} | {r['pr']} | {r['leg']} | {'ok' if r['ok'] else 'FAIL'} | {f(r['ptok'], '%d')} | {f(r['pfs'])} | {f(r['pms'], '%.0f')} | "
          f"{f(r['dec'], '%.2f')} | {f(r['load'], '%.2f')} | {r['store'] or '-'} | {f(r['swap'], '%.0f')} |")

def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return "-", None
    m = st.mean(v)
    return (f"{m:.1f} (n {len(v)}, ±{(max(v) - min(v)) / 2 / m * 100:.1f}%)" if len(v) > 1 else f"{m:.1f} (n 1)"), m

ok = [r for r in rows if r["ok"]]
order = ["dec", "c1k", "pf4", "pf16", "c32k"]
print("\n## Summary (all model files on the internal disk; ± = half the range over the mean)\n")
print("| engine | length (prompt tok) | prefill tok/s | decode tok/s | TTFT ms (warm, prompt) |")
print("|---|---:|---:|---:|---:|")
best = defaultdict(dict)
for p in order:
    for e in ("kap", "llama-gpu", "llama-cpumoe"):
        L = [r for r in ok if r["eng"] == e and r["pr"] == p and not r["cold"]]
        if not L:
            continue
        a, pm_ = agg([r["pfs"] for r in L]); b, dm_ = agg([r["dec"] for r in L]); c, tm_ = agg([r["pms"] for r in L])
        best[p][e] = (pm_, dm_, tm_)
        print(f"| {name[e]} | {p} ({L[0]['ptok']}) | {a} | {b} | {c} |")
print("\n## Cold start (the other engine's files read through the page cache first; prompt dec, 85 tok)\n")
print("| engine | load s (engine's own) | TTFT ms (prompt) | launch to first token s (wall: process time - decode time) |")
print("|---|---:|---:|---:|")
for e in ("kap", "llama-gpu"):
    L = [r for r in ok if r["eng"] == e and r["cold"]]
    if L:
        a, lm = agg([r["load"] for r in L]); b, _ = agg([r["pms"] for r in L])
        c, cm = agg([r["real"] - r["dms"] / 1e3 for r in L if r["real"] and r["dms"]])
        best["cold"][e] = (cm,)
        print(f"| {name[e]} | {a} | {b} | {c} |")
print("\n## Fastest? (Kapitsa vs the best llama.cpp configuration at each length)\n")
for p in order:
    if "kap" not in best[p]:
        continue
    k = best[p]["kap"]; ll = [v for e, v in best[p].items() if e != "kap"]
    if not ll:
        continue
    for i, nm in ((0, "prefill"), (1, "decode")):
        o = max(v[i] for v in ll)
        print(f"- {p} {nm}: Kapitsa {k[i]:.1f} vs llama.cpp {o:.1f} tok/s -> fastest: {'yes' if k[i] > o else 'no'}, by {abs(k[i] / o - 1) * 100:.1f}%")
if "kap" in best["cold"] and "llama-gpu" in best["cold"]:
    a, b = best["cold"]["kap"][0], best["cold"]["llama-gpu"][0]
    print(f"- cold start to first token: Kapitsa {a:.2f} s vs llama.cpp {b:.2f} s -> fastest: {'yes' if a < b else 'no'}, by {abs(b / a - 1) * 100:.1f}%")
print("\n## Generated tokens (greedy, first 128): Kapitsa vs llama.cpp, common prefix\n")
for p in order:
    k = next((r["toks"] for r in ok if r["eng"] == "kap" and r["pr"] == p and r["toks"]), None)
    l = next((r["toks"] for r in ok if r["eng"] == "llama-gpu" and r["pr"] == p and r["toks"]), None)
    if k and l:
        n = next((i for i, (x, y) in enumerate(zip(k, l)) if x != y), min(len(k), len(l)))
        print(f"- {p}: {n} of {min(len(k), len(l))} tokens equal before the first difference")

print("\n## Caveats\n")
print(f"- Kapitsa store sizes seen: {', '.join(kst) or '-'}.")
for c in os.environ.get("RV_CAVEATS", "").split("|"):
    if c.strip(): print(f"- {c.strip()}")
