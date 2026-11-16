#!/usr/bin/env python3
"""cmp_texts.py A.json B.json: per request, identical text or the character where the two first differ"""
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = 0
for k in a:
    x, y = a[k]["text"], b[k]["text"]
    if x == y:
        same += 1
        print(f"{k}: identical ({a[k]['tokens']} tokens)")
    else:
        i = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"{k}: differs at character {i} of {len(x)} / {len(y)} ({a[k]['tokens']} / {b[k]['tokens']} tokens)")
print(f"{same} of {len(a)} identical")
