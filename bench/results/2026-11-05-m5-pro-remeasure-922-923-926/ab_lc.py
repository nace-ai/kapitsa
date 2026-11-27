#!/usr/bin/env python3
# Interleaved A/B (rounds alternate the arm order: old new, new old, ...) with every leg's conditions logged. Before a
# leg: a quiet GPU (no foreign process above 5% for 15 s, waiting up to 900 s: bench/harness/gpu_hogs.py), then the
# 1-minute load average below LOAD_START (waiting up to 900 s); the load average and the power source (`pmset -g batt`)
# are read just before and just after the leg, and the foreign GPU load is sampled every 2 s while it runs. A leg is
# clean when the GPU was quiet, the foreign GPU load was 0 throughout (mean and max), the load average at its end was
# below 4, and the Mac drew from AC power before and after; a round is clean when both its legs are. A leg that ends at
# load 4 or more records the five busiest processes then (`ps`), to name what raised it.
# usage: ab_lc.py CONFIG.json OUT_DIR ROUNDS   (AB_HARNESS: <repo>/bench/harness)
# CONFIG: {"arms": {"old": {"cmd": [...], "env": {...}, "cwd": "..." (optional)}, "new": {...}},
#          "metrics": {"name": "regex with one group", ...}, "ids": "regex with one group" (optional)}
import json, os, re, statistics, subprocess, sys, time
sys.path.insert(0, os.environ["AB_HARNESS"])
import gpu_hogs  # noqa: E402

LOAD_START, LOAD_END = 3.9, 4.0  # #343: GPU-JOBS.md bar is load < 4 (the PR's 2.0 stalled every leg 900 s at idle CPU)


def load1():
    return os.getloadavg()[0]


def on_ac():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    return "'AC Power'" in out.splitlines()[0] if out else False


cfg = json.load(open(sys.argv[1]))
out, rounds = sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
res = []
for rd in range(rounds):
    for arm in (("old", "new") if rd % 2 == 0 else ("new", "old")):
        a = cfg["arms"][arm]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
        env["NI_LOCK_HELD"] = "1"
        env.update(a.get("env", {}))
        quiet, waited, _ = gpu_hogs.wait_quiet(5.0, 15, 60)
        t = time.time()
        while load1() >= LOAD_START and time.time() - t < 120:
            time.sleep(5)
        load_wait = time.time() - t
        l0, ac0 = load1(), on_ac()
        log = os.path.join(out, f"r{rd}-{arm}.out")
        t0 = time.time()
        with open(log, "w") as f:
            p = subprocess.Popen(a["cmd"], env=env, cwd=a.get("cwd"), stdout=f, stderr=subprocess.STDOUT)
            mon = gpu_hogs.ContentionMonitor(lambda: [p.pid])  # (the measured process's own GPU time is never foreign)
            mon.start()
            rc = p.wait()
        mon.stop()
        l1, ac1 = load1(), on_ac()
        top = []
        if l1 >= LOAD_END:
            ps = subprocess.run(["ps", "-Ao", "pcpu=,comm=", "-r"], capture_output=True, text=True, timeout=10).stdout
            top = [" ".join(ln.split(None, 1)) for ln in ps.splitlines()[:5]]
        if rc != 0:
            sys.exit(f"{a['cmd'][0]} failed: {log}")
        gl = gpu_hogs.summarize_samples(mon.samples, t0, time.time())
        txt = open(log).read()
        r = dict(round=rd, arm=arm, quiet=quiet, quiet_wait_s=round(waited, 1), load_wait_s=round(load_wait, 1),
                 load1_before=round(l0, 2), load1_after=round(l1, 2), ac_before=ac0, ac_after=ac1, gpu_load=gl)
        if top:
            r["busiest_at_end"] = top
        for k, rx in cfg["metrics"].items():
            m = re.search(rx, txt, re.M)
            r[k] = float(m.group(1)) if m else None
        if cfg.get("ids"):
            m = re.search(cfg["ids"], txt, re.M)
            r["ids"] = m.group(1) if m else None
        fz = gl.get("mean_foreign_gpu_pct") == 0 and gl.get("max_foreign_gpu_pct") == 0
        r["clean"] = bool(quiet and fz and l1 < LOAD_END and ac0 and ac1)
        res.append(r)
        ms = ", ".join(f"{k} {r[k]}" for k in cfg["metrics"])
        print(f"round {rd} {arm}: {ms} | quiet {quiet} after {waited:.1f} s, foreign GPU mean "
              f"{gl.get('mean_foreign_gpu_pct')}% max {gl.get('max_foreign_gpu_pct')}%, load {l0:.2f} -> {l1:.2f} "
              f"(waited {load_wait:.0f} s), AC {ac0} -> {ac1}, clean {r['clean']}"
              + (f" | busiest at the end: {'; '.join(top)}" if top else ""), flush=True)
json.dump(res, open(os.path.join(out, "ab.json"), "w"), indent=1)
by = {}
for r in res:
    by.setdefault(r["round"], {})[r["arm"]] = r
clean = [v for v in by.values() if len(v) == 2 and v["old"]["clean"] and v["new"]["clean"]]
print(f"clean rounds: {len(clean)} of {len(by)}")
for k in cfg["metrics"]:
    if clean and all(v[a][k] is not None for v in clean for a in ("old", "new")):
        o = statistics.median(v["old"][k] for v in clean)
        n = statistics.median(v["new"][k] for v in clean)
        print(f"median {k} over clean rounds: old {o:.3f}, new {n:.3f} ({(n / o - 1) * 100:+.2f}%)")
if cfg.get("ids"):
    print("ids identical across every run:", len({r["ids"] for r in res}) == 1)
