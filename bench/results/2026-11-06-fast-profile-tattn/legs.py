#!/usr/bin/env python3
"""One CSV row per timed leg from a run directory of `<n>-<cell>-<arm>.out` files (ni_run output) and its run log.

  python3 legs.py RUNDIR [RUNLOG] > legs.csv

status is the run log's verdict for the leg (ok, void-load, void-battery, void-guard, plus "+mempressure" when ni_run
logged memory pressure); a leg with no log line is "unlogged". Stdlib only.
"""
import csv, glob, os, re, sys

run = sys.argv[1]
log = open(sys.argv[2]).read() if len(sys.argv) > 2 else ""
w = csv.writer(sys.stdout)
w.writerow(["leg", "cell", "arm", "status", "prefill_tokens", "prefill_ms", "decode_tokens", "decode_ms", "cycles",
            "tok_per_cycle", "ms_wall_per_cycle", "memory_pressure_lines"])
for f in sorted(glob.glob(os.path.join(run, "*.out")), key=lambda p: int(os.path.basename(p).split("-")[0])):
    leg, cell, arm = os.path.basename(f)[:-4].split("-", 2)
    t = open(f, errors="replace").read()
    m = re.search(r"^\S+ %s \S+ arm=%s rc=\S+ load \S+ (\S+)" % (leg, re.escape(arm)), log, re.M)
    status = m.group(1) if m else "unlogged"
    p = re.search(r"prompt (\d+) tok in (\d+) ms", t)
    ptok, pms = (p.group(1), p.group(2)) if p else (None, None)
    reps = re.findall(r"prefill rep \d+: (\d+) tok in ([\d.]+) ms", t)
    if not p and reps:  # bench_prefill legs: the median of the repeats
        ms = sorted(float(x[1]) for x in reps)
        ptok, pms = reps[0][0], "%.1f" % ms[len(ms) // 2]
    d = re.search(r"decode(?: D=\d+)?: (\d+) tok in (\d+) ms", t)
    c = re.search(r"(\d+) cycles, ([\d.]+) tok/cycle.*?([\d.]+) ms wall/cycle", t)
    w.writerow([leg, cell, arm, status, ptok, pms, d and d.group(1), d and d.group(2),
                c and c.group(1), c and c.group(2), c and c.group(3), t.count("memory pressure warning")])
