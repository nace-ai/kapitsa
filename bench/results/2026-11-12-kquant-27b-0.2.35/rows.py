#!/usr/bin/env python3
"""Prints this record's leg tables and the 27B decode figures docs/benchmarks.md carries, from the leg files, and checks them.

    python3 rows.py           # per-setup leg table, per-mode means and the pooled figure
    python3 rows.py --check   # also fails unless the figures equal the ones docs/benchmarks.md states
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ARMS = {"A": "plain 4-bit", "B": "KQuant"}
# A leg counts when legs.log marks it ok; the M5 Pro's 4-A is void (load rose to 7.46 during it).
VALID = {"m5-pro": ["1-A", "2-B", "3-B", "5-A"], "m4-pro": ["1-A", "2-B", "3-B", "4-A"]}
STATED = {  # setup: (A T0.6, B T0.6, A T0, B T0, A pooled, B pooled)
    "m5-pro": (80.6, 82.9, 88.4, 93.0, 84.5, 87.9),
    "m4-pro": (63.5, 65.4, 70.0, 68.9, 66.8, 67.2),
}


def mean(xs):
    return sum(xs) / len(xs)


def setup(name):
    legs = {}
    for f in sorted((HERE / name / "legs").glob("[0-9]-[AB].json")):
        legs[f.stem] = json.loads(f.read_text())
    mac = next(iter(legs.values()))["mac"]
    print(f"\n{name}: {mac['chip']} {mac['memory_gb']} GB, macOS {mac['macos']} ({mac['macos_build']})")
    print("| Leg | Pack | T 0.6 tok/s | T 0 tok/s | accepted share (T 0.6 / T 0) | counted |")
    print("| --- | --- | --- | --- | --- | --- |")
    by = {"A": [], "B": []}
    for leg, j in legs.items():
        r = j["results"]
        t6, t0 = r["decode"]["median_tps"], r["decode_greedy"]["median_tps"]
        s6, s0 = r["decode"]["median_accepted_draft_share"], r["decode_greedy"]["median_accepted_draft_share"]
        ok = leg in VALID[name] and not j.get("invalid")
        print(f"| {leg} | {ARMS[leg[-1]]} | {t6:.1f} | {t0:.1f} | {s6:.3f} / {s0:.3f} | {'yes' if ok else 'no'} |")
        if ok:
            by[leg[-1]].append((t6, t0))
    a6, a0 = mean([x[0] for x in by["A"]]), mean([x[1] for x in by["A"]])
    b6, b0 = mean([x[0] for x in by["B"]]), mean([x[1] for x in by["B"]])
    pa, pb = (a6 + a0) / 2, (b6 + b0) / 2
    print(f"T 0.6: {b6:.1f} against {a6:.1f} tok/s, {100 * (b6 / a6 - 1):+.1f}%")
    print(f"T 0: {b0:.1f} against {a0:.1f} tok/s, {100 * (b0 / a0 - 1):+.1f}%")
    print(f"pooled {pb:.1f} against {pa:.1f} tok/s, {100 * (pb / pa - 1):+.1f}%")
    return tuple(round(x, 1) for x in (a6, b6, a0, b0, pa, pb))


def main():
    bad = []
    for name in VALID:
        got = setup(name)
        if got != STATED[name]:
            bad.append(f"{name}: computed {got}, stated {STATED[name]}")
    if "--check" in sys.argv:
        if bad:
            sys.exit("MISMATCH " + "; ".join(bad))
        print("\nok: the stated figures match the leg files")


if __name__ == "__main__":
    main()
