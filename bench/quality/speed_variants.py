#!/usr/bin/env python3
"""Decode and prefill speed of a model's precision variants with ni_run, interleaved: the measurement behind the
default precisions (internal PERF.md, Weight formats, Defaults).

  speed_variants.py --ni-run build/ni_run --drafter <dflash2 dir> --variant NAME=DIR[:ENV=V[;ENV=V]] ...
                    [--rounds 1] [--out results.json]

Per round and per measurement, every variant in turn (so a slow spell of the GPU hits them all): DFlash2 decode at
T = 0.6 (top-p 0.95, top-k 20) over seeds 1-4, 512 tokens each; DFlash2 greedy twice, 256 tokens; plain greedy
decode once, 256 tokens; a 4k-token prefill twice (the default schedule: dense models through the Neural Engine
split). The decode prompt is bench/prompts/code_27b.txt (the harness's coding prompt), the prefill one
bench/prompts/p4000.txt. --ignore-eos everywhere. A variant's ENV is set for its runs only (e.g. NI_REQUANT=6 for the
4-bit weights re-encoded as 6-bit). Takes ~/.cache/nace-gpu.lock unless NI_LOCK_HELD=1 (the caller holds it).
"""
import argparse, fcntl, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "..", "prompts")


def run(ni_run, d, ids, n, temp, extra, env):
    e = dict(os.environ)
    e.update(env)
    cmd = [ni_run, d, "@" + ids, str(n), str(temp), "--ignore-eos"] + extra
    t = time.time()
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"{' '.join(cmd)} failed:\n{p.stderr[-2000:]}")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ni-run", required=True)
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--variant", action="append", required=True, help="NAME=DIR[:ENV=V[;ENV=V]]")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()
    variants = []
    for v in a.variant:
        name, _, rest = v.partition("=")
        d, _, envs = rest.partition(":")
        env = dict(kv.split("=", 1) for kv in envs.split(";") if kv)
        variants.append((name, os.path.expanduser(d), env))
    lock = None
    if not os.environ.get("NI_LOCK_HELD"):
        lock = os.fdopen(os.open(os.path.expanduser("~/.cache/nace-gpu.lock"), os.O_RDONLY | os.O_CREAT, 0o644))
        fcntl.flock(lock, fcntl.LOCK_EX)
    code, p4k = os.path.join(PROMPTS, "code_27b.txt"), os.path.join(PROMPTS, "p4000.txt")
    dfl = ["--dflash", a.drafter]
    plan = [("t06", s, code, 512, 0.6, dfl, {"NI_SEED": str(s)}) for s in (1, 2, 3, 4)] + \
           [("greedy", k, code, 256, 0, dfl, {}) for k in (1, 2)] + \
           [("plain", 1, code, 256, 0, ["--dflash", "none"], {})] + \
           [("prefill", k, p4k, 1, 0, ["--dflash", "none"], {}) for k in (1, 2)]
    res = {name: [] for name, _, _ in variants}
    t0 = time.time()
    for rnd in range(a.rounds):
        for kind, k, ids, n, temp, extra, env0 in plan:
            for name, d, env in (variants if rnd % 2 == 0 else variants[::-1]):
                r = run(a.ni_run, d, ids, n, temp, extra, {**env0, **env})
                r.update(kind=kind, k=k, round=rnd)
                res[name].append(r)
                print(time.strftime("%H:%M:%S"), name, kind, k, json.dumps({x: r[x] for x in r if x not in ("ids_head",)}),
                      flush=True)
    summary = {}
    for name, rs in res.items():
        s = {}
        for kind, key in (("t06", "decode_tok_s"), ("greedy", "decode_tok_s"), ("plain", "decode_tok_s"),
                          ("prefill", "prefill_tok_s")):
            v = [r[key] for r in rs if r["kind"] == kind and key in r]
            s[kind] = round(sum(v) / len(v), 2) if v else None
            tc = [r["tok_cycle"] for r in rs if r["kind"] == kind and "tok_cycle" in r]
            if tc:
                s[kind + "_tok_cycle"] = round(sum(tc) / len(tc), 3)
        summary[name] = s
    base = variants[0][0]
    for name, s in summary.items():
        rel = {k: (f"{100 * (v / summary[base][k] - 1):+.1f}%" if v and summary[base].get(k) and name != base and
                   not k.endswith("cycle") else "") for k, v in s.items()}
        print(name, json.dumps(s), json.dumps({k: v for k, v in rel.items() if v}))
    if a.out:
        json.dump({"variants": [{"name": n, "dir": d, "env": e} for n, d, e in variants], "runs": res,
                   "summary": summary, "minutes": round((time.time() - t0) / 60, 1)}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
