#!/usr/bin/env python3
"""Add a new Qwen's model_type to kQwenTypes (src/model/qwen4_config.cpp) once its config.json was read and found to be
one of Kapitsa's two Qwen layouts under a new name, then check that the row landed and list the tests naming the type.

usage: tools/add_qwen_type.py MODEL_DIR --layout qwen35|qwen4exp [--repo .]

qwen35: Gated DeltaNet + full attention (Qwen3.5, 3.6, 3.8). qwen4exp: Qwen3.8-Flash-Next's (hyper-connections, the
sparse indexer, the n-gram table). Exit 0 with the row added or already there, 1 when the edit did not land.
"""
import argparse
import json
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("model_dir")
ap.add_argument("--layout", required=True, choices=("qwen35", "qwen4exp"))
ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
a = ap.parse_args()
c = json.load(open(os.path.join(a.model_dir, "config.json")))
tc = c.get("text_config") if isinstance(c.get("text_config"), dict) else {}
mt = c.get("model_type") or ""
tt = tc.get("model_type") or (mt + "_text")
if not mt:
    sys.exit("config.json has no model_type")
row = f'    {{"{mt}", "{tt}", QwenLayout::{a.layout}}},\n'
p = os.path.join(a.repo, "src/model/qwen4_config.cpp")
s = open(p).read()
anchor = "constexpr QwenType kQwenTypes[] = {\n"
if anchor not in s:
    sys.exit(f"kQwenTypes not found in {p}")
if row not in s:
    j = s.index("};\n", s.index(anchor))  # appended last: the existing rows keep their precedence
    open(p, "w").write(s[:j] + row + s[j:])
ok = open(p).read().count(row) == 1
hits = [ln.strip() for ln in open(os.path.join(a.repo, "tools/qwen4_config_test.cpp")) if re.search(rf'"{re.escape(mt)}"', ln)]
print(f"row: {row.strip()}  in {p}: {'ok' if ok else 'MISSING'}")
print(f"tests naming {mt!r} (a refusal test for this name flips): {len(hits)}")
for ln in hits[:12]:
    print("   ", ln)
sys.exit(0 if ok else 1)
