#!/usr/bin/env python3
"""#343: VERDICT.json per PR from out/*/ab.json (ab_lc.py) and out/exl3-realbench-*/result.json; no typed numbers.
usage: verdict.py   (writes out/VERDICT-922.json, -923.json, -926.json and prints a one-line summary each)"""
import hashlib, json, os, statistics, glob, re, subprocess
K = os.path.expanduser("~/Documents/nace/k343"); O = K + "/out"
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def say(pct, tol=1.0):
    return "about the same" if abs(pct) < tol else ("faster" if pct > 0 else "slower")
def ab(name, lower_better=True):
    d = f"{O}/{name}/ab.json"
    if not os.path.exists(d): return None
    res = json.load(open(d)); cfg = json.load(open(f"{K}/cfg-{name}.json"))
    key = next(iter(cfg["metrics"]))
    pairs = {}
    for r in res: pairs.setdefault(r["round"], {})[r["arm"]] = r
    full = [v for v in pairs.values() if len(v) == 2 and v["old"].get(key) and v["new"].get(key)]
    clean = [v for v in full if v["old"]["clean"] and v["new"]["clean"]]
    acp = [v for v in full if all(v[a].get(k) for a in ("old", "new") for k in ("ac_before", "ac_after"))]
    full = acp  # a pair with any leg on battery is dropped (GPU-JOBS.md)
    clean = [v for v in clean if v in acp]
    use = clean or full
    o = statistics.median(v["old"][key] for v in use); n = statistics.median(v["new"][key] for v in use)
    per = [(v["old"][key] / v["new"][key] - 1) * 100 for v in use]  # % faster of new (time metrics: old/new)
    def label(r):
        g = r.get("gpu_load", {})
        cap = "; GPU quiet-wait capped at 60 s (DSH visible / not quiet)" if r.get("quiet") is False else ""
        if not g.get("samples"): return "foreign GPU not sampled (leg shorter than the 2 s sampler)" + cap
        return "foreign GPU mean %s%% max %s%%" % (g.get("mean_foreign_gpu_pct"), g.get("max_foreign_gpu_pct")) + cap
    bins = {a: cfg["arms"][a]["cmd"][0] for a in ("old", "new")}
    return dict(name=name, metric=key, old_median=o, new_median=n, faster_pct=(o / n - 1) * 100, pair_range_pct=[min(per), max(per)],
                rounds=len(full), clean_rounds=len(clean), used="clean rounds" if clean else "all AC pairs; rounds are not labelled clean because the 2 s foreign-GPU sampler gave no sample on short legs or load/foreign GPU was above the harness bar (see legs[].gpu)",
                argv={a: cfg["arms"][a]["cmd"] for a in cfg["arms"]}, env={a: cfg["arms"][a].get("env", {}) for a in cfg["arms"]},
                bin_sha256={a: sha(bins[a]) for a in bins},
                legs=[dict({k: r.get(k) for k in ("round", "arm", key, "ac_before", "ac_after", "load1_before", "load1_after")}, gpu=label(r)) for r in res])
json.dump({n: ab(n) for n in map(os.path.basename, glob.glob(O + "/*")) if os.path.exists(f"{O}/{n}/ab.json")}, open(O + "/all-ab.json", "w"), indent=1)
HEAD = "PR headline config (Flash-Next 4-bit) cannot be measured on this Mac: the 4-bit pack is about 100 GB, the internal disk has 55 GiB free and the Mac has 48 GB of RAM"
PR = {"922": dict(commits=dict(parent="b210e9ab", merge="f9e85aa7"), timed="Flint 4-bit 8-row verify (flint_bench --rows 8 --steps 128 --reps 3) and the moe4_unique kernel at Flash-Next's shape (D 2560, k 10)", ab=["922-flint", "922-unique-R3", "922-unique-R4"]),
      "923": dict(commits=dict(parent="467ff747", merge="153dff95"), timed="kernel on vs off at M5 Pro default=off: moe4_gate_up_d/moe4_down_d against moe4_gate_up/moe4_down at Flash-Next's shape, R 3 and 4 (moe4_time)", ab=["923-gateup-R3", "923-gateup-R4", "923-down-R3", "923-down-R4"]),
      "926": dict(commits=dict(parent="153dff95", merge="af7d7958"), timed="layer-0 EXL3 fixture (beamster/Qwen3.8-Flash-Next-Sushi-2bpw@cb6c85e4, the PR's own real-weight A/B, synthetic activations)", ab=[])}
def verdict(pr):
    c = PR[pr]; v = dict(pr=pr, task="#343", machine=subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        macos=subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(), commits=c["commits"], what_was_timed=c["timed"], headline_config_note=HEAD, ab=[])
    for n in c["ab"]:
        if os.path.exists(f"{O}/{n}/ab.json"):
            a = ab(n); a["verdict_words"] = say(a["faster_pct"]); v["ab"].append(a)
    mlx = sorted(glob.glob(O + "/mlx/run*.out"))
    if pr == "922":
        res = []
        for f in mlx:
            for l in open(f):
                if l.startswith("RESULT"): res.append(l.strip())
        import re as _re
        gt = [float(_re.search(r"gen_tps=([\d.]+)", r).group(1)) for r in res]
        pt = [float(_re.search(r"prompt_tps=([\d.]+)", r).group(1)) for r in res]
        kf = sorted(glob.glob(O + "/kapitsa-flint/run*.out"))
        kdec = []
        for f in kf:
            for l in open(f):
                m = _re.search(r"^rep \d: .*decode .*\(([\d.]+) tok/s", l)
                if m: kdec.append(float(m.group(1)))
        v["mlx_summary"] = dict(mlx_vlm_decode_tok_s=gt, mlx_vlm_decode_median=statistics.median(gt) if gt else None, mlx_vlm_prefill_tok_s=pt,
            kapitsa_flint_bench_decode_tok_s=kdec, kapitsa_flint_bench_decode_median=statistics.median(kdec) if kdec else None,
            note="flint_bench is the Kapitsa engine's own timer: 893-token corpus prompt, 128 greedy steps; mlx-vlm: same length, different text, 128 tokens, temperature 0, one process per run (cold load). Indicative only, not a harness leg.")
        v["mlx"] = dict(model="Flint 4-bit via mlx-vlm 0.7.11 (greedy, 4K-char prompt, 128 tokens; config patched in an APFS clone, U8 weights viewed as uint32)", runs=res) if res else "not measured: GPU lock held by other jobs (#350 coding10 legs, #348 park) until after the 03:30 cutoff; the mlx-vlm Flint run itself works (lazy CPU load validated)"
    else: v["mlx"] = "cannot measure on MLX: the changed kernels are Flash-Next/EXL3 specific; no MLX model of it fits here"
    for f in sorted(glob.glob(O + "/exl3-realbench-*/result.json")) if pr == "926" else []:
        r = json.load(open(f)); v.setdefault("realbench", []).append(dict(file=f, timing_eligible=r.get("timing_eligible"), returncode=r.get("returncode"), command=r.get("command"), env=r.get("env"), hashes=r.get("hashes"), power=r.get("power")))
    if pr == "926":
        rows = {}
        for f in sorted(glob.glob(O + "/exl3-realbench-*/run.log")):
            for l in open(f):
                if not l.startswith("{"): continue
                j = json.loads(l)
                if j.get("kind") != "bench" or j.get("phase") != "chain": continue
                ref = [k for k in j if "reference" in k][0]
                key = "T=%d %s" % (j["T"], j["routing"])
                rows.setdefault(key, []).append((statistics.median(j[ref]), statistics.median(j["candidate_us"])))
        v["chain_original_vs_registers"] = {k: dict(original_ms=[round(a / 1000, 3) for a, b in x], registers_ms=[round(b / 1000, 3) for a, b in x],
            less_time_pct=[round(100 * (1 - b / a), 2) for a, b in x], words=say(100 * (1 - statistics.median(b / a for a, b in x)))) for k, x in rows.items()}
        v["m4_pro_pr_claim"] = "4.3-7.7% less expert-chain GPU time on M4 Pro (PR body); measured here on M5 Pro, original vs register kernel in one executable"
        v["binary_sha256"] = {"exl3_register_check": sha(O + "/../wt-af7d7958/build/exl3_register_check")}
    json.dump(v, open(f"{O}/VERDICT-{pr}.json", "w"), indent=1); return v
for pr in PR: verdict(pr)
print("wrote VERDICT-922/923/926.json")
for n in sorted(os.listdir(O)):
    if os.path.exists(f"{O}/{n}/ab.json"):
        a = ab(n); print(f"{n}: old {a['old_median']:.3f} new {a['new_median']:.3f} {a['metric']} -> new {a['faster_pct']:+.2f}% faster ({say(a['faster_pct'])}), pairs {a['pair_range_pct'][0]:+.1f}..{a['pair_range_pct'][1]:+.1f}%, rounds {a['rounds']} clean {a['clean_rounds']}")
