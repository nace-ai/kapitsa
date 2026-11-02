#!/usr/bin/env python3
"""How much of a user's text can the DFlash2 reduced draft head even propose? CPU only, stdlib only.

The drafter drafts only tokens whose rows are in its reduced head: by default the first 98,304 token ids in id order
plus every id from the first end-of-sequence id on (src/model/dflash2.cpp:540-545 and build_head :764, kHeadRowsIdOrder
in dflash2.h:93). The code comment says that covers 99.8-100% of generated tokens on English text and code. For text
in a language whose tokens sit past id 98,303 it does not.

Input: files of comma-separated token ids (the format of bench/prompts ids and `kapitsa-ksx` ids/*.ids), one text per
file; tokenise elsewhere (the model's tokenizer.json). For each file the first --train-frac of the tokens is "what the
user wrote so far" and the rest is the future. Printed per file, on the future part:
  in-head share      fraction of tokens inside the id-order head, and inside a head that keeps the same number of
                     rows but fills it with the user's own tokens first (then id order)
  ceiling tok/cycle  tokens per cycle of a PERFECT drafter (every proposal the target would accept) limited to the
                     head, with at most --k drafts per cycle: the most the head lets any drafter reach
A ceiling is not a speed-up: a real drafter accepts a fraction of what it can propose. It says where a bigger or
re-ranked head can matter at all.

usage: draft_head_coverage.py IDS_FILE [IDS_FILE ...] [--head 98304] [--k 7] [--train-frac 0.5]
"""
import argparse
from collections import Counter

VOCAB = 248077  # the Qwen 3.x / 3.8 tokenizer


def read_ids(path):
    with open(path) as f:
        return [int(x) for x in f.read().replace("\n", ",").split(",") if x.strip()]


def ceiling(ids, in_head, k):
    """Tokens per cycle of a perfect drafter that can only propose in-head tokens, at most k drafts per cycle."""
    p = cycles = 0
    while p < len(ids):
        d = 0
        while d < k and p + 1 + d < len(ids) and in_head(ids[p + 1 + d]):
            d += 1
        p += d + 1
        cycles += 1
    return len(ids) / max(cycles, 1)


def user_first_head(train, head, vocab=VOCAB):
    """The head's token set when the user's own tokens (by frequency) come first and id order fills the rest."""
    rows = [t for t, _ in Counter(train).most_common()]
    seen = set(rows)
    t = 0
    while len(rows) < head and t < vocab:
        if t not in seen:
            rows.append(t)
            seen.add(t)
        t += 1
    return set(rows[:head])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--head", type=int, default=98304)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()
    print(f"{'file':28} {'tokens':>7} {'in-head id-order':>17} {'user-first':>11} {'ceiling id-order':>17} {'user-first':>11} {'full':>6}")
    for path in args.files:
        ids = read_ids(path)
        cut = int(len(ids) * args.train_frac)
        train, test = ids[:cut], ids[cut:]
        if not test:
            continue
        idord = lambda t: t < args.head
        uf = user_first_head(train, args.head)
        share = lambda f: 100.0 * sum(1 for t in test if f(t)) / len(test)
        print(f"{path.rsplit('/', 1)[-1]:28} {len(test):7d} {share(idord):16.1f}% {share(lambda t: t in uf):10.1f}% "
              f"{ceiling(test, idord, args.k):17.2f} {ceiling(test, lambda t: t in uf, args.k):11.2f} "
              f"{ceiling(test, lambda t: True, args.k):6.2f}")


if __name__ == "__main__":
    main()
