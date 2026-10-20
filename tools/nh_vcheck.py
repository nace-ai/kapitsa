#!/usr/bin/env python3
"""Compares two logits files of nh_eval --vcheck (fp32 [rows][vocab]): a reference A and a candidate B.

  nh_vcheck.py A B [--vocab 131072]

Prints, over the rows: bitwise-identical rows, argmax agreement (disagreements with the reference's top-2 gap, a
near-tie when it is under 0.05), the max |logit diff|, and KL(softmax A || softmax B) mean / p99 / max.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--vocab", type=int, default=131072)
    a = ap.parse_args()
    A = np.fromfile(a.a, dtype=np.float32).reshape(-1, a.vocab)
    B = np.fromfile(a.b, dtype=np.float32).reshape(-1, a.vocab)
    if A.shape != B.shape:
        sys.exit(f"shapes differ: {A.shape} vs {B.shape}")
    same = int(np.sum(np.all(A == B, axis=1)))
    am, bm = A.argmax(1), B.argmax(1)
    top2 = np.sort(A, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    dis = np.nonzero(am != bm)[0]
    la = A - A.max(1, keepdims=True)
    lb = B - B.max(1, keepdims=True)
    la -= np.log(np.exp(la).sum(1, keepdims=True))
    lb -= np.log(np.exp(lb).sum(1, keepdims=True))
    kl = (np.exp(la) * (la - lb)).sum(1)
    print(f"rows {len(A)}: bitwise identical {same}, argmax agree {len(A) - len(dis)}, "
          f"max |diff| {np.abs(A - B).max():.4g}, KL mean {kl.mean():.3g} p99 {np.quantile(kl, 0.99):.3g} max {kl.max():.3g}")
    for r in dis:
        print(f"  row {r}: argmax {am[r]} vs {bm[r]}, reference top-2 gap {gap[r]:.4f}{' (near-tie)' if gap[r] < 0.05 else ''}")


if __name__ == "__main__":
    main()
