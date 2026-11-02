#!/usr/bin/env python3
"""How much of a user's text can the DFlash2 reduced draft head even propose? CPU only, stdlib only.

The drafter drafts only tokens whose rows are in its reduced head. The engine's rules (src/model/dflash2.cpp
build_head :764, :784; dflash2.h:93): with no ranked list the head is the first 98,304 token ids in id order (kHeadRowsIdOrder);
with NI_DFLASH_HEAD_IDS it is 65,536 ranked rows (kHeadRows) unless NI_DFLASH_HEAD=n says otherwise; in both cases every id from the
first end-of-sequence id on (--special0, 248044 for the Qwen 3.x tokenizer) is added. The code comment says the id-order head covers
99.8-100% of generated tokens on English text and code; it was measured on English only. For text in a language whose tokens sit
past id 98,303 it does not cover them.

Input: files of comma-separated token ids (the format of bench/prompts ids and `kapitsa-ksx` ids/*.ids), one text per
file; tokenise elsewhere (the model's tokenizer.json). For each file the first --train-frac of the tokens is "what the
user wrote so far" and the rest is the future. Printed per file, on the future part:
  in-head share      fraction of tokens inside the id-order head, inside a head of the same number of rows filled with the
                     user's own tokens first (then id order), and inside the engine's default ranked head (65,536 rows)
  ceiling tok/cycle  tokens per cycle of a PERFECT drafter (every proposal the target would accept) limited to the
                     head, with at most --k drafts per cycle: the most the head lets any drafter reach
A ceiling is not a speed-up: a real drafter accepts a fraction of what it can propose. It says where a bigger or
re-ranked head can matter at all.

usage: draft_head_coverage.py IDS_FILE [IDS_FILE ...] [--head 98304] [--ranked-head 65536] [--special0 248044] [--k 7] [--train-frac 0.5]
"""
import argparse
from collections import Counter

VOCAB = 248077  # the Qwen 3.x / 3.8 tokenizer
SPECIAL0 = 248044  # first end-of-sequence id of that tokenizer (config.json eos_token_id [248046, 248044])


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


def user_first_head(train, head, vocab=VOCAB, special0=VOCAB):
    """The head's token set when the user's own tokens (by frequency) come first and id order fills the rest; every id >= special0
    is added on top, as the engine does."""
    rows = [t for t, _ in Counter(train).most_common()]
    seen = set(rows)
    t = 0
    while len(rows) < head and t < vocab:
        if t not in seen:
            rows.append(t)
            seen.add(t)
        t += 1
    return set(rows[:head]) | set(range(special0, vocab))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--head", type=int, default=98304)
    ap.add_argument("--ranked-head", type=int, default=65536, help="rows of the engine's default ranked head")
    ap.add_argument("--special0", type=int, default=SPECIAL0)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--train-frac", type=float, default=0.5)
    args = ap.parse_args()
    print(f"{'file':28} {'tokens':>7} {'in-head: id-order':>18} {'user-first':>11} {'ranked-65k':>11} {'ceiling: id-order':>18} {'user-first':>11} {'ranked-65k':>11} {'full':>6}")
    for path in args.files:
        ids = read_ids(path)
        cut = int(len(ids) * args.train_frac)
        train, test = ids[:cut], ids[cut:]
        if not test:
            continue
        idord = lambda t: t < args.head or t >= args.special0
        uf = user_first_head(train, args.head, special0=args.special0)
        rk = user_first_head(train, args.ranked_head, special0=args.special0)
        share = lambda f: 100.0 * sum(1 for t in test if f(t)) / len(test)
        print(f"{path.rsplit('/', 1)[-1]:28} {len(test):7d} {share(idord):17.1f}% {share(lambda t: t in uf):10.1f}% {share(lambda t: t in rk):10.1f}% "
              f"{ceiling(test, idord, args.k):18.2f} {ceiling(test, lambda t: t in uf, args.k):11.2f} {ceiling(test, lambda t: t in rk, args.k):11.2f} "
              f"{ceiling(test, lambda t: True, args.k):6.2f}")


if __name__ == "__main__":
    main()
