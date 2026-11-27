#!/usr/bin/env python3
"""Prints the README's 4B decode cells and this record's leg table from sp35.json, and checks sp35.json against legs/.

    python3 rows.py           # the cells and sentences the README carries
    python3 rows.py --check   # every leg in sp35.json matches its leg file in legs/ (to 0.1 tok/s), and no leg is flagged
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "sp35.json").read_text())
PACK = next(iter(DATA["packs"].values()))
LEGS = {"base": ["1-A", "6-A"], "ours": ["4-B", "5-B"]}  # order A B B A; legs.log's X legs ran nothing


def mean(xs):
    return sum(xs) / len(xs)


def pooled(arm):
    """Mean over legs of the T 0.6 and greedy medians, the figure the README's decode column shows."""
    return mean([mean(leg) for leg in PACK["legs"][arm]])


def check():
    bad = []
    for arm, names in LEGS.items():
        for name, (s, g) in zip(names, PACK["legs"][arm]):
            leg = json.loads((HERE / "legs" / f"{name}.json").read_text())
            r = leg["results"]
            got = (r["decode"]["median_tps"], r["decode_greedy"]["median_tps"])
            if abs(got[0] - s) > 0.05 or abs(got[1] - g) > 0.05:
                bad.append(f"{name}: sp35.json {s}/{g}, leg file {got[0]}/{got[1]}")
            if leg["conditions"]["power_source"] != "ac":
                bad.append(f"{name}: power {leg['conditions']['power_source']}")
            loops = [f for f in leg["flags"] if "repetition loop" in f]
            if loops:
                bad.append(f"{name}: {loops}")
    print("\n".join(bad) or "ok: 4 legs match sp35.json, all on AC, no repetition-loop flags")
    return 1 if bad else 0


def main():
    if "--check" in sys.argv:
        return check()
    a, b = pooled("base"), pooled("ours")
    (s_a, s_b), (g_a, g_b) = [(r["base"], r["ours"]) for r in PACK["rows"]]
    print(f"| 4B lmstudio-community | 0.070 | 0.914 | {a:.0f} |")
    print(f"| 4B KQuant | **0.048** | 0.923 | {b:.0f} |")
    print(f"\npooled {b:.1f} against {a:.1f} tok/s, {100 * (b / a - 1):+.1f}% (tie band +-{DATA['packs'][next(iter(DATA['packs']))].get('noise_pct', 3):.0f}%)")
    print(f"T 0.6: {s_b:.0f} against {s_a:.0f} tok/s; T 0: {g_b:.0f} against {g_a:.0f}")
    print("\n| Leg | Pack | T 0.6, tok/s | T 0, tok/s |\n| --- | --- | --- | --- |")
    for arm, names in LEGS.items():
        for name, (s, g) in zip(names, PACK["legs"][arm]):
            label = "lmstudio-community" if arm == "base" else "KQuant"
            print(f"| {name} | {label} | {s:.1f} | {g:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
