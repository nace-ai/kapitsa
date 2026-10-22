#!/usr/bin/env python3
"""Rebuild sweep.json's "chosen" cells from the committed files alone (stdlib, no GPU).

For each engine, size and metric (prefill, decode), every run that measured the point (the first sweep and, where
there is one, the rerun) gets the foreign GPU load sampled inside that metric's window, and the cell takes the run
with the lower mean (the first sweep on a tie). Both runs' load is written next to the chosen value.

Windows:
- rec: its log stamps each prompt's start ("Prefill: prompt=N", local time, UTC+5 here). Prefill is [start,
  start + ttft], decode the next completion_tokens / decode_tps seconds.
- Kapitsa: its log has no timestamps, only each request's prompt and decode durations, so the windows are laid end
  to end backwards from the sweep's end, taken as the monitor's last sample (it stops with the server; it matched the
  end-of-sweep vm_stat snapshot within 1 s in all four runs). The gaps between requests are ignored, about 1 s of error.
- llama.cpp: the same, from its log's per-request "prompt eval time" and "eval time"; its 128K point ran on its own
  load (run "llamacpp-128k", in sweep.json's "separate_loads"), the other points in run "llamacpp". It had no
  foreign load in either.
- A sample counts if its time is inside the window. The monitor samples every 2 s, so a 256-token decode (about 2 s)
  holds 1-2 samples and its choice rests on one or two readings; prefill windows from 16K up hold many.

Usage: python3 choose_cells.py [--write]   (prints the table; --write stores it in sweep.json's "chosen")
"""
import datetime
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TZ = datetime.timezone(datetime.timedelta(hours=5))  # the Mac's local time (Asia/Almaty) in rec's log
RUNS = {"kapitsa": ["kapitsa", "kapitsa-rerun"], "rec": ["rec", "rec-rerun"], "llamacpp": ["llamacpp", "llamacpp-128k"]}


def samples(run):
    out = []
    with open(os.path.join(HERE, "foreign-gpu", run + ".txt")) as f:
        for line in f:
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                out.append((float(parts[0]), float(parts[1])))
    return out


def kapitsa_windows(run, points):
    log = open(os.path.join(HERE, "logs", run + ".log")).read()
    if run.startswith("llamacpp"):
        reqs = [(float(p), float(d)) for p, d in
                re.findall(r"prompt eval time =\s*([\d.]+) ms.*\n.*?\beval time =\s*([\d.]+) ms", log)]
    else:
        reqs = [(float(p), float(d)) for p, d in
                re.findall(r"chat: prompt \d+ \(cached 0\) in (\d+) ms .*?, \d+ tokens in (\d+) ms", log)]
    reqs = reqs[len(reqs) - len(points):]  # the first is the warm-up request
    end = samples(run)[-1][0]
    wins = []
    for p_ms, d_ms in reversed(reqs):
        d0 = end - d_ms / 1000
        p0 = d0 - p_ms / 1000
        wins.append(((p0, d0), (d0, end)))
        end = p0
    return list(reversed(wins))


def rec_windows(run, points):
    log = open(os.path.join(HERE, "logs", run + ".log")).read()
    starts = {}
    for ts, n in re.findall(r"^(\S+ \S+),\d+ Prefill: prompt=(\d+) ", log, re.M):
        t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ).timestamp()
        starts.setdefault(int(n), []).append(t)
    wins = []
    for pt in points:
        t = starts[pt["prompt_tokens"]][-1]  # warm-up prompts differ in length from the points
        pe = t + pt["ttft_s"]
        wins.append(((t, pe), (pe, pe + pt["completion_tokens"] / pt["decode_tps"])))
    return wins


def load(smp, win):
    v = [x for t, x in smp if win[0] <= t <= win[1]]
    return [round(sum(v) / len(v), 1) if v else 0.0, max(v) if v else 0.0, len(v)]


def main():
    path = os.path.join(HERE, "sweep.json")
    sw = json.load(open(path))
    chosen = {}
    for eng, runs in RUNS.items():
        per_run = {}
        for run in runs:
            pts = sw["points"][eng] if run == eng else (sw["reruns"].get(eng) if run.endswith("-rerun")
                                                         else sw["separate_loads"][run])
            wins = (rec_windows if eng == "rec" else kapitsa_windows)(run, pts)
            smp = samples(run)
            per_run[run] = {pt["size"]: (pt, load(smp, w[0]), load(smp, w[1])) for pt, w in zip(pts, wins)}
        chosen[eng] = {}
        for size in sorted(set().union(*per_run.values())):  # llama.cpp's 128K is only in its separate load
            cell = {}
            for k, metric in ((1, "prefill"), (2, "decode")):
                cands = [(r, per_run[r][size]) for r in runs if size in per_run[r]]
                best = min(cands, key=lambda c: c[1][k][0])  # min() keeps the first sweep on a tie
                val = best[1][0][metric + "_tps"]
                cell[metric] = round(val) if metric == "prefill" else round(val, 1)
                cell[metric + "_run"] = best[0]
                cell[metric + "_foreign"] = {r: c[k] for r, c in cands}  # [mean %, max %, samples]
                cell[metric + "_values"] = {r: (round(c[0][metric + "_tps"]) if metric == "prefill"
                                                else round(c[0][metric + "_tps"], 1)) for r, c in cands}
            chosen[eng][str(size)] = cell
    for eng in chosen:
        print(eng)
        for size, c in chosen[eng].items():
            print(f"  {size:>6}  prefill {c['prefill']:>5} ({c['prefill_run']}; {c['prefill_foreign']})"
                  f"  decode {c['decode']:>5} ({c['decode_run']}; {c['decode_foreign']})")
    if "--write" in sys.argv:
        sw["chosen"] = chosen
        with open(path, "w") as f:
            json.dump(sw, f, indent=1)
            f.write("\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
