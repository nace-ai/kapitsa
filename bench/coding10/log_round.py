#!/usr/bin/env python3
"""Append one TSV row per (model, arm) of coding10 result files to a hillclimb log.

  python3 log_round.py --log hillclimb.tsv --round 1 --change "thinking off" results/4b.json results/9b.json
"""

import argparse
import json
import os

COLS = ["round", "model", "arm", "pass", "n", "wall_s_total", "wall_s_median", "completion_tokens", "change", "json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--round", required=True)
    ap.add_argument("--change", default="")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    new = not os.path.exists(a.log)
    with open(a.log, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        for p in a.files:
            d = json.load(open(p))
            for arm, s in d["summary"].items():
                row = [a.round, d.get("label", ""), arm, s["pass"], s["n"], s["wall_s_total"], s["wall_s_median"],
                       s["completion_tokens"], a.change, os.path.abspath(p)]
                f.write("\t".join(str(x) for x in row) + "\n")
                print("\t".join(str(x) for x in row))


if __name__ == "__main__":
    main()
