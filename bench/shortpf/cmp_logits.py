#!/usr/bin/env python3
"""cmp_logits.py ref.bin new.bin -> argmax agreement, top-2 margin of ref, max |diff|, KL(ref||new)"""
import sys, struct, math, array
def load(p):
    a = array.array("f"); a.frombytes(open(p, "rb").read()); return a
r, n = load(sys.argv[1]), load(sys.argv[2])
assert len(r) == len(n)
ar = max(range(len(r)), key=r.__getitem__); an = max(range(len(n)), key=n.__getitem__)
top = sorted(range(len(r)), key=r.__getitem__, reverse=True)[:2]
margin = r[top[0]] - r[top[1]]
md = max(abs(x - y) for x, y in zip(r, n))
def lsm(a):
    m = max(a); s = sum(math.exp(x - m) for x in a); return [x - m - math.log(s) for x in a]
lr, ln = lsm(r), lsm(n)
kl = sum(math.exp(x) * (x - y) for x, y in zip(lr, ln))
print(f"argmax ref={ar} new={an} {'SAME' if ar == an else 'DIFF'}  ref top2 margin={margin:.4f}  max|d|={md:.4f}  KL={kl:.2e}")
