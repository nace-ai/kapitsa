#!/usr/bin/env python3
"""tools/readme_images: every Kapitsa number renders as plain KFlash, whatever its record's method; a rival's plain
decoding renders with no label. CPU only, stdlib only."""
import importlib.util, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("readme_images", os.path.join(HERE, "..", "tools", "readme_images", "readme_images.py"))
ri = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ri)

for m in ("KFlash", "DFlash2", "DSpark", "LiLiCorr", "MTP", "prompt lookup", "no spec", "loops", None):
    row = {"model": "m", "cells": {"Kapitsa": {"value": 100, "method": m}, "llama.cpp": {"value": 50, "method": m}}}
    assert ri.cell(row, "Kapitsa")[1] == "KFlash", m
    assert ri.cell(row, "llama.cpp")[1] == (None if m == "no spec" else m), m   # a rival keeps its own method
assert ri.cell({"model": "m", "cells": {"Kapitsa": 100}}, "Kapitsa")[1] == "KFlash"   # a bare number too
assert ri.cell({"model": "m", "cells": {"Kapitsa": "x"}}, "Kapitsa")[1] is None       # × and — carry no label

# every Kapitsa number in readme_images.json (poster, tables and header) renders KFlash
data = json.load(open(os.path.join(HERE, "..", "tools", "readme_images", "readme_images.json")))
ri.check(data)
rows = [r for m in data["poster"]["macs"] for r in m["rows"]] + [r for m in data["tables"]["macs"] for r in m["rows"]]
n = 0
for r in rows:
    v, label, _ = ri.cell(r, "Kapitsa")
    if ri.is_num(v):
        n += 1
        assert label == "KFlash", (r["model"], label)
assert n == 37, n   # every Kapitsa number in readme_images.json; update when cells are added or removed
assert all(h["method"] == "KFlash" for h in ri.header_rows(data) if h["kap"] and h["value"] is not None)
hr = ri.header_rows(data)   # the header SVGs' <desc> (and the README alt text) name Kapitsa's method as KFlash
assert "Kapitsa with KFlash " in ri.header_desc(data, hr), ri.header_desc(data, hr)
assert "no spec" not in ri.header_desc(data, hr), ri.header_desc(data, hr)
print(f"test_readme_images: ok ({n} Kapitsa cells, all KFlash)")
