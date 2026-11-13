#!/usr/bin/env python3
"""Prints this record's leg tables and the KQuant-against-lmstudio figures from the leg files, and checks them.

    python3 rows.py           # per model: the leg table, the per-mode means, the pooled figure, 4K prefill
    python3 rows.py --check   # every leg ran on AC, bench started its own server (no invalid note), no contention
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = ("9b", "35b")
LEGS = {"lmstudio": ["1-A", "4-A"], "KQuant": ["2-B", "3-B"]}  # order A B B A
MODES = (("decode", "T 0.6"), ("decode_greedy", "T 0"))


def leg(model, name):
    return json.loads((HERE / model / "legs" / f"{name}.json").read_text())


def mean(xs):
    return sum(xs) / len(xs)


def loops(d, mode):
    return [f for f in d["flags"] if f.startswith(mode + ":") and "repetition loop" in f]


def check():
    bad = []
    for model in MODELS:
        for names in LEGS.values():
            for name in names:
                d = leg(model, name)
                c = d["conditions"]
                if c["power_source"] != "ac":
                    bad.append(f"{model} {name}: power {c['power_source']}")
                if d.get("invalid"):
                    bad.append(f"{model} {name}: invalid {d['invalid']}")
                if c.get("contention"):
                    bad.append(f"{model} {name}: contention {c['contention']}")
    print("\n".join(bad) or "ok: 8 legs on AC, bench's own server, no contention")
    return 1 if bad else 0


def model_rows(model):
    print(f"\n{model}\n\n| Leg | Pack | T 0.6, tok/s | T 0, tok/s | Accepted share, T 0.6 / T 0 | 4K prefill, tok/s | Loop flags |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for pack, names in LEGS.items():
        for name in names:
            d = leg(model, name)
            r = d["results"]
            acc = " / ".join(f"{r[m]['median_accepted_draft_share']:.3f}" for m, _ in MODES)
            pf = next(p["tps"] for p in r["prefill"] if p["size"] == 4000)
            fl = ", ".join(label for m, label in MODES if loops(d, m)) or "none"
            print(f"| {name} | {pack} | {r['decode']['median_tps']:.1f} | {r['decode_greedy']['median_tps']:.1f} | {acc} | "
                  f"{pf:.0f} | {fl} |")
    t = {p: {m: mean([leg(model, n)["results"][m]["median_tps"] for n in names]) for m, _ in MODES}
         for p, names in LEGS.items()}
    print()
    for m, label in MODES:
        a, b = t["lmstudio"][m], t["KQuant"][m]
        print(f"{label}: KQuant {b:.1f} against lmstudio {a:.1f} tok/s, {100 * (b / a - 1):+.1f}%")
    a, b = (mean(list(t[p].values())) for p in ("lmstudio", "KQuant"))
    print(f"pooled: KQuant {b:.1f} against lmstudio {a:.1f} tok/s, {100 * (b / a - 1):+.1f}%")
    pf = {p: mean([next(x["tps"] for x in leg(model, n)["results"]["prefill"] if x["size"] == 4000) for n in names])
          for p, names in LEGS.items()}
    print(f"4K prefill: KQuant {pf['KQuant']:.0f} against lmstudio {pf['lmstudio']:.0f} tok/s, "
          f"{100 * (pf['KQuant'] / pf['lmstudio'] - 1):+.1f}%")


def main():
    if "--check" in sys.argv:
        return check()
    for model in MODELS:
        model_rows(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
