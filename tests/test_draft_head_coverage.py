#!/usr/bin/env python3
"""tools/draft_head_coverage.py on synthetic token ids: the ceiling of a head that cannot propose a token, the
user-first head recovering tokens the id-order head misses at the same row count. CPU only, stdlib only.

usage: test_draft_head_coverage.py
"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOL)
import draft_head_coverage as d  # noqa: E402

FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


check(abs(d.ceiling(list(range(16)), lambda t: True, 7) - 8.0) < 1e-9, "ceiling: full head, k=7 -> 8 tokens per cycle")
check(abs(d.ceiling(list(range(16)), lambda t: False, 7) - 1.0) < 1e-9, "ceiling: nothing in the head -> 1")
check(d.ceiling([5, 200000, 6, 7, 8], lambda t: t < 100, 7) < d.ceiling([5, 6, 7, 8, 9], lambda t: t < 100, 7),
      "ceiling: a token outside the head ends the chain")

# a text of 4 ids all past the head, repeated: id-order head misses them, a user-first head of the same size has them
text = [200000, 200001, 200002, 200003] * 40
uf = d.user_first_head(text[:80], 1000)
check(len(uf) == 1000 and 200000 in uf and 5 in uf, "user-first head: same rows, user's tokens in, id order fills")
check(d.user_first_head([], 10) == set(range(10)), "user-first head with no history is id order")

with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "x.ids")
    open(p, "w").write(",".join(str(t) for t in text))
    r = subprocess.run([sys.executable, os.path.join(TOOL, "draft_head_coverage.py"), p, "--head", "1000"],
                       capture_output=True, text=True)
    row = r.stdout.strip().splitlines()[-1].split() if r.returncode == 0 else []
    check(r.returncode == 0 and len(row) >= 7, "CLI prints a row", r.stderr[-200:])
    if row:
        check(row[2] == "0.0%" and row[3] == "100.0%", "id-order head covers 0 %, user-first 100 %", row)
        check(float(row[4]) == 1.0 and float(row[5]) > 7.0, "ceiling 1.00 -> 8.00", row)

if FAILED:
    print(f"{len(FAILED)} FAILED")
    sys.exit(1)
print("all passed")
