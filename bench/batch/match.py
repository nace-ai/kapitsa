#!/usr/bin/env python3
"""Batched against single-stream outputs, from two NI_TRACE_TOKENS files of the same requests (the server writes one
JSON line per request: its prompt's hash, the tokens of its speculative cycles and the margin between the two largest
logits of the row that chose each).

  match.py SOLO.jsonl BATCHED.jsonl [--near 0.5]

Pairs the greedy requests by prompt and reports the greedy-match rate (requests whose tokens are all the same, and
tokens before the first difference over all tokens) and, at each first difference, the single-stream margin of the
token it chose there: a difference at a near-tie has a small margin (two tokens within numerical noise of each
other)."""
import argparse, json, statistics, sys


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                out[j["prompt_hash"]] = j
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("solo")
    p.add_argument("batched")
    p.add_argument("--near", type=float, default=0.5, help="a margin below this is a near-tie (logits)")
    a = p.parse_args()
    solo, bat = load(a.solo), load(a.batched)
    pairs = [(h, solo[h], bat[h]) for h in solo if h in bat and solo[h].get("temperature", 0) == 0]
    if not pairs:
        print("no greedy requests in both files")
        sys.exit(1)
    same = tok_same = tok_all = 0
    divs = []
    for h, s, b in pairs:
        st = [s["first"]] + s["tokens"]
        bt = [b["first"]] + b["tokens"]
        n = min(len(st), len(bt))
        k = next((i for i in range(n) if st[i] != bt[i]), n)
        full = k == n and len(st) == len(bt)
        same += full
        tok_same += k
        tok_all += len(st)
        if not full and k < n:
            # margins are indexed by the traced tokens (after the first)
            sm = s["margins"][k - 1] if k >= 1 and k - 1 < len(s["margins"]) else None
            bm = b["margins"][k - 1] if k >= 1 and k - 1 < len(b["margins"]) else None
            divs.append({"prompt_tokens": s["prompt_tokens"], "at": k, "of": len(st), "solo_margin": sm,
                         "batched_margin": bm, "batched_cycles": b.get("batched", 0)})
    print(json.dumps({"requests": len(pairs), "identical": same, "tokens_before_first_difference": tok_same,
                      "tokens": tok_all, "differences": divs}, indent=1))
    ms = [d["solo_margin"] for d in divs if d["solo_margin"] is not None]
    near = sum(1 for m in ms if m < a.near)
    print(f"greedy match: {same} of {len(pairs)} requests identical; {tok_same} of {tok_all} tokens "
          f"({100.0 * tok_same / max(tok_all, 1):.1f}%) before the first difference; first differences at single-stream "
          f"margins {('median %.3g, max %.3g' % (statistics.median(ms), max(ms))) if ms else '-'} "
          f"({near} of {len(ms)} below {a.near})")


if __name__ == "__main__":
    main()
