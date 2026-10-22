#!/usr/bin/env python3
"""Decode and prefill speed of a model's precision variants with ni_run, interleaved: the measurement behind the
default precisions (internal PERF.md, Weight formats, Defaults).

  speed_variants.py --ni-run build/ni_run --drafter <dflash2 dir> --variant NAME=DIR[:ENV=V[;ENV=V]] ...
                    --out results.json [--rounds 2] [--seeds 4] [--timeout 900]

Per round and per measurement, every variant in turn, the order reversed every other round (so a slow spell of the
GPU hits them all): DFlash2 decode at T = 0.6 (top-p 0.95, top-k 20) over seeds 1..--seeds, 512 tokens each;
DFlash2 greedy twice, 256 tokens; plain greedy decode once, 256 tokens; a 4k-token prefill twice (the default
schedule: dense models through the Neural Engine split). The decode prompt is bench/prompts/code_27b.txt (the
harness's coding prompt), the prefill one bench/prompts/p4000.txt. --ignore-eos everywhere. A variant's ENV is set
for its runs only. NI_REQUANT=<bits> re-encodes the 4-bit weights (the cost per byte of a format, with 4-bit's
DFlash2 acceptance); a mix is judged on its real pack, whose acceptance can differ. Sampled decode varies with the
text each seed draws (4 seeds: a few points either way), so the summary gives the spread and the GPU time per
DFlash2 cycle (the cost side, which does not depend on the text) beside the means. Results are written after every
run; a run that exceeds --timeout seconds is killed (it would hold the GPU lock) and ends the measurement. Takes
/tmp/kapitsa-gpu.lock unless NI_LOCK_HELD=1 (the caller holds it).
"""
import argparse, fcntl, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "..", "prompts")


def run(ni_run, d, ids, n, temp, extra, env, timeout):
    e = dict(os.environ)
    e.update(env)
    cmd = [ni_run, d, "@" + ids, str(n), str(temp), "--ignore-eos"] + extra
    t = time.time()
    try:
        p = subprocess.run(cmd, env=e, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout} s", "cmd": " ".join(cmd)}
    if p.returncode:
        return {"error": f"exit {p.returncode}: {p.stderr[-1500:]}", "cmd": " ".join(cmd)}
    err = p.stderr
    r = {"wall_s": round(time.time() - t, 1)}
    m = re.search(r"prompt (\d+) tok in ([0-9.]+) ms \(([0-9.]+) tok/s\)", err)
    if m:
        r["prefill_tok_s"] = float(m.group(3))
    m = re.search(r"dflash decode \(verify (\d+) of 7, (\w+)\): (\d+) tok in ([0-9.]+) ms \(([0-9.]+) tok/s\), (\d+) cycles, "
                  r"([0-9.]+) tok/cycle, ([0-9.]+) ms GPU/cycle", err)
    if m:
        r.update(decode_tok_s=float(m.group(5)), tok_cycle=float(m.group(7)), gpu_ms_cycle=float(m.group(8)))
    else:
        m = re.search(r"; decode (\d+) tok in ([0-9.]+) ms \(([0-9.]+) tok/s\)", err)
        if m:
            r["decode_tok_s"] = float(m.group(3))
    m = re.search(r"ids:([0-9 ]+)", p.stdout)
    if m:
        r["ids_head"] = m.group(1).split()[:24]
    return r


def summarize(res, variants):
    summary = {}
    for name, rs in res.items():
        s = {}
        for kind, key in (("t06", "decode_tok_s"), ("greedy", "decode_tok_s"), ("plain", "decode_tok_s"),
                          ("prefill", "prefill_tok_s")):
            v = [r[key] for r in rs if r.get("kind") == kind and key in r]
            if not v:
                continue
            s[kind] = {"mean": round(statistics.mean(v), 2), "min": min(v), "max": max(v), "n": len(v),
                       "sd": round(statistics.stdev(v), 2) if len(v) > 1 else 0.0}
            for k2 in ("tok_cycle", "gpu_ms_cycle"):
                w = [r[k2] for r in rs if r.get("kind") == kind and k2 in r]
                if w:
                    s[kind][k2] = round(statistics.mean(w), 3)
        summary[name] = s
    base = variants[0][0]
    for name, s in summary.items():
        for kind, v in s.items():
            b = summary[base].get(kind)
            if name != base and b:
                v["vs_" + base] = f"{100 * (v['mean'] / b['mean'] - 1):+.1f}%"
                if "gpu_ms_cycle" in v and "gpu_ms_cycle" in b:
                    v["cycle_vs_" + base] = f"{100 * (v['gpu_ms_cycle'] / b['gpu_ms_cycle'] - 1):+.1f}%"
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ni-run", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--variant", action="append", required=True, help="NAME=DIR[:ENV=V[;ENV=V]]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per ni_run")
    a = ap.parse_args()
    if a.rounds % 2:
        print("note: an odd --rounds runs the variants in the forward order once more than reversed", flush=True)
    variants = []
    for v in a.variant:
        name, _, rest = v.partition("=")
        d, _, envs = rest.partition(":")
        env = dict(kv.split("=", 1) for kv in envs.split(";") if kv)
        variants.append((name, os.path.expanduser(d), env))
    lock = None
    if not os.environ.get("NI_LOCK_HELD"):
        lock = os.fdopen(os.open(os.path.expanduser("/tmp/kapitsa-gpu.lock"), os.O_RDONLY | os.O_CREAT, 0o644))
        fcntl.flock(lock, fcntl.LOCK_EX)
    code, p4k = os.path.join(PROMPTS, "code_27b.txt"), os.path.join(PROMPTS, "p4000.txt")
    dfl = ["--dflash", a.drafter]
    plan = [("t06", s, code, 512, 0.6, dfl, {"NI_SEED": str(s)}) for s in range(1, a.seeds + 1)] + \
           [("greedy", k, code, 256, 0, dfl, {}) for k in (1, 2)] + \
           [("plain", 1, code, 256, 0, ["--dflash", "none"], {})] + \
           [("prefill", k, p4k, 1, 0, ["--dflash", "none"], {}) for k in (1, 2)]
    res = {name: [] for name, _, _ in variants}
    t0 = time.time()

    def save(done):
        json.dump({"variants": [{"name": n, "dir": d, "env": e} for n, d, e in variants], "runs": res,
                   "summary": summarize(res, variants), "minutes": round((time.time() - t0) / 60, 1),
                   "complete": done, "rounds": a.rounds, "seeds": a.seeds}, open(a.out, "w"), indent=1)

    for rnd in range(a.rounds):
        for kind, k, ids, n, temp, extra, env0 in plan:
            for name, d, env in (variants if rnd % 2 == 0 else variants[::-1]):
                r = run(a.ni_run, d, ids, n, temp, extra, {**env0, **env}, a.timeout)
                r.update(kind=kind, k=k, round=rnd)
                res[name].append(r)
                print(time.strftime("%H:%M:%S"), name, kind, k, json.dumps({x: r[x] for x in r if x != "ids_head"}),
                      flush=True)
                save(False)
                if "error" in r:
                    sys.exit(f"{name} {kind} {k}: {r['error']}")
    save(True)
    for name, s in summarize(res, variants).items():
        print(name, json.dumps(s))


if __name__ == "__main__":
    main()
