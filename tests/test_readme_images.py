#!/usr/bin/env python3
"""tools/readme_images: every Kapitsa number renders as plain KFlash, whatever its record's method, unless the cell names
its own label; a rival's plain decoding renders with no label. CPU
only, stdlib only."""
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
assert ri.cell({"model": "m", "cells": {"Kapitsa": {"value": 100, "label": "custom"}}}, "Kapitsa")[1] == "custom"   # a cell may name its own
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
        c = r["cells"]["Kapitsa"]
        want = (c.get("label") if isinstance(c, dict) else None) or "KFlash"
        assert label == want, (r["model"], label)
assert n == 40, n   # every Kapitsa number in readme_images.json; update when cells are added or removed
labelled = [r for r in rows if isinstance(r["cells"]["Kapitsa"], dict) and r["cells"]["Kapitsa"].get("label")]
assert labelled == [], labelled   # every Kapitsa cell in the data reads KFlash
assert all(h["method"] == "KFlash" for h in ri.header_rows(data) if h["kap"] and h["value"] is not None)
hr = ri.header_rows(data)   # the header SVGs' <desc> names Kapitsa's method as KFlash
assert "Kapitsa with KFlash " in ri.header_desc(data, hr), ri.header_desc(data, hr)
assert "no spec" not in ri.header_desc(data, hr), ri.header_desc(data, hr)
assert ri.header_alt(data, hr) == ("Kapitsa writes 351 tok/s on Qwen3.6 35B-A3B on an M5 Max, against 138-281 for "
                                   "MLX, Ollama, llama.cpp, oMLX, mlx-serve and Splash."), ri.header_alt(data, hr)
C = data["context"]   # the context charts: every model has a value per size, and each chart builds without Chrome
for kind in ("prefill", "decode"):
    assert all(len(m[kind]) == len(C["sizes"]) for m in C["models"]), kind
    html, w, h = ri.context_chart(data, "light", kind)
    assert html.count("<polyline") == 2 * len(C["models"]) and w == data["poster"]["size"][0], kind  # line, leader
    assert ri.context_alt(data, kind).startswith(C[kind]["alt"] + ": " + C["models"][0]["model"]), kind
# column headers name an engine, never a version, so every rival number carries its version in its own cell detail
# (tables and README), or else the line that names it in docs/benchmarks.md, which the poster footer points at
import re
docs = open(os.path.join(HERE, "..", "docs", "benchmarks.md"), encoding="utf-8").read().split("\n")
assert not any(e.get("version") and e["name"] != "MLX" for e in data["engines"]), "a rival column header names a version"
assert not any(re.search(r"\d", v) for m in data["tables"]["macs"] for v in m.get("versions", {}).values()), "a Mac's header names a version"
rival_cells = [(m["name"], r["model"], e, c) for m in data["poster"]["macs"] + data["tables"]["macs"] for r in m["rows"]
               for e, c in r["cells"].items() if e != "Kapitsa" and isinstance(c, (dict, int, float)) and not isinstance(c, bool)]
assert rival_cells
for mac, model, e, c in rival_cells:
    detail = c.get("detail", "") if isinstance(c, dict) else ""
    in_docs = any(e.split()[0] in line and re.search(r"\d+\.\d+|\bb\d{4,5}\b", line) for line in docs)
    assert re.search(r"\d", detail) or in_docs, f"{mac} / {model} / {e}: no version in its detail or in docs/benchmarks.md"
for e in {e for _, _, e, _ in rival_cells}:  # the footer's claim: the doc names a version for every rival engine on the poster
    assert any(e.split()[0] in line and re.search(r"\d+\.\d+|\bb\d{4,5}\b", line) for line in docs), f"docs/benchmarks.md names no version for {e}"
assert any("docs/benchmarks.md" in x for x in data["poster"]["footer"])
print(f"test_readme_images: ok ({n} Kapitsa cells)")
