#!/usr/bin/env python3
"""Prints this record's leg table and docs/benchmarks.md's 35B-A3B decode figures from the leg files in legs/, and checks them.

    python3 rows.py           # the table, the pooled cells and the per-mode figures docs/benchmarks.md carries
    python3 rows.py --check   # every leg ran on AC, bench started its own server (no invalid note), no flagged loop
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGS = {"MTPLX": ["1-A", "4-A"], "KQuant": ["2-B", "3-B"]}  # order A B B A
MODES = ("decode", "decode_greedy")  # T 0.6, T 0


def leg(name):
    return json.loads((HERE / "legs" / f"{name}.json").read_text())


def mean(xs):
    return sum(xs) / len(xs)


def tps(name, mode):
    return leg(name)["results"][mode]["median_tps"]


def check():
    bad = []
    for names in LEGS.values():
        for name in names:
            d = leg(name)
            if d["conditions"]["power_source"] != "ac":
                bad.append(f"{name}: power {d['conditions']['power_source']}")
            if d.get("invalid"):
                bad.append(f"{name}: invalid {d['invalid']}")
            loops = [f for f in d["flags"] if "repetition loop" in f]
            if loops:
                bad.append(f"{name}: {loops}")
    print("\n".join(bad) or "ok: 4 legs on AC, bench's own server, no repetition-loop flags")
    return 1 if bad else 0


def main():
    if "--check" in sys.argv:
        return check()
    print("| Leg | Pack | T 0.6, tok/s | T 0, tok/s | Accepted share, T 0.6 / T 0 |\n| --- | --- | --- | --- | --- |")
    for pack, names in LEGS.items():
        for name in names:
            r = leg(name)["results"]
            acc = " / ".join(f"{r[m]['median_accepted_draft_share']:.3f}" for m in MODES)
            print(f"| {name} | {pack} | {r['decode']['median_tps']:.1f} | {r['decode_greedy']['median_tps']:.1f} | {acc} |")
    per = {p: [mean([tps(n, m) for n in names]) for m in MODES] for p, names in LEGS.items()}
    pooled = {p: mean([mean([tps(n, m) for m in MODES]) for n in names]) for p, names in LEGS.items()}
    a, b = pooled["MTPLX"], pooled["KQuant"]
    print(f"\n| 35B-A3B MTPLX | 0.096 | 0.948 | {a:.0f} |\n| 35B-A3B KQuant | **0.059** | 0.952 | **{b:.0f}** |")
    print(f"\npooled {b:.1f} against {a:.1f} tok/s, {100 * (b / a - 1):+.1f}%")
    for i, label in enumerate(("T 0.6", "T 0")):
        x, y = per["MTPLX"][i], per["KQuant"][i]
        print(f"{label}: {y:.1f} against {x:.1f} tok/s, {100 * (y / x - 1):+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
