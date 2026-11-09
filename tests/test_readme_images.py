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
# column headers name an engine, never a version; every rival number shows the version of the record behind it:
# a cell's `src` files (bench/results) give the number and the version, which readme_images.resolve_versions fills into
# its detail; a cell whose files name no version carries `version` and, unless it says "version not recorded",
# `version_from`, a record that states it
import copy
import re
assert not any(e.get("version") and e["name"] != "MLX" for e in data["engines"]), "a rival column header names a version"
assert not any(re.search(r"\d", v) for m in data["tables"]["macs"] for v in m.get("versions", {}).values()), "a Mac's header names a version"
rd = copy.deepcopy(data)
ri.resolve_versions(rd)
rival_cells = [(m["name"], r["model"], e, c) for m in rd["poster"]["macs"] + rd["tables"]["macs"] for r in m["rows"]
               for e, c in r["cells"].items() if e != "Kapitsa" and isinstance(c, dict) and ri.is_num(c.get("value"))]
assert rival_cells
NOT_RECORDED = "version not recorded"
VERSION_WORDS = re.compile(r"mlx-lm [\d.]+|mlx-vlm [\d.]+|MLX runtime [\d.]+|llama\.cpp runtime [\d.]+|\bb\d{4,5}\b|\d+\.\d+[\w.]*|\b[0-9a-f]{7,8}\b")
for mac, model, e, c in rival_cells:
    who = f"{mac} / {model} / {e}"
    assert "src" in c or "version" in c, f"{who}: no result file and no stated version behind its number"
    if "src" in c:  # the files behind the number: right engine, and one of their decodes (or their mean) is the cell's number
        recs = [ri.result_record(p) for p in c["src"]]
        assert all(r["engine"] == ri.RESULT_ENGINE[e] for r in recs), who
        meds = [r["decode_summary"]["median_tps"] for r in recs]
        vals = [x for r in recs for x in (r["decode_summary"].get("median_tps"), r["decode_summary"].get("median_tps_e2e"),
                                           r["decode_summary"].get("mean_tps")) if x is not None] + [sum(meds) / len(meds)]
        assert any(abs(v - c["value"]) <= 0.55 for v in vals), f"{who}: {c['value']} is in none of its files {c['src']}"
        lab = ri.source_version(e, c["src"])
        if lab:
            assert c["detail"].count(lab) == 1, f"{who}: shows {c['detail']!r}, its files record {lab!r}"
            assert "version" not in c or c["version"] == lab, who
            assert set(VERSION_WORDS.findall(c["detail"].replace(lab, ""))) <= set(), f"{who}: a second version in {c['detail']!r}"
    if "version" in c and c["version"] != NOT_RECORDED and not ("src" in c and ri.source_version(e, c["src"])):
        # the record names this version on the engine's own line (a table row or sentence about that engine), not
        # just somewhere in the file: "mlx-vlm 0.7.8" elsewhere in a README says nothing about this engine's build
        ENGINE_WORDS = {"Splash": ("splash",), "llama.cpp": ("llama.cpp",), "MLX": ("mlx-lm", "mlx-vlm"), "oMLX": ("omlx",),
                        "mlx-serve": ("mlx-serve",), "Ollama": ("ollama",), "MTPLX": ("mtplx",), "LM Studio": ("lm studio", "lmstudio")}
        src_text = open(os.path.join(ri.RESULTS, c["version_from"]), encoding="utf-8").read()
        lines = [ln for ln in src_text.split("\n") if c["version"] in ln and any(w in ln.lower() for w in ENGINE_WORDS[e])]
        assert lines, f"{who}: no line of {c['version_from']} names {e} with {c['version']!r}"
        assert c["version"] in c["detail"], who
    if c.get("version") == NOT_RECORDED:
        assert NOT_RECORDED in c["detail"], who
# a cell whose version differs from its result file fails: a wrong src version, a hand-typed second version, a wrong number
bad = copy.deepcopy(data)
cell0 = next(c for m in bad["poster"]["macs"] for r in m["rows"] for e, c in r["cells"].items()
             if e == "MLX" and isinstance(c, dict) and c.get("src") and "mlx-vlm" in (ri.source_version("MLX", c["src"]) or ""))
other = next(p for p in ri.RESULTS.glob("**/*mlxlm*.json") if ri.version_label("MLX", json.load(open(p)).get("version", "")) == "mlx-lm 0.32.2")
cell0["src"] = [str(other.relative_to(ri.RESULTS))]
assert ri.source_version("MLX", cell0["src"]) == "mlx-lm 0.32.2"
cell0["detail"] = "mlx-vlm 0.7.12"
try:
    ri.resolve_versions(bad)
    raise SystemExit("a hand-typed version was accepted")
except AssertionError as err:
    assert "types a version" in str(err), err
cell0["detail"] = "{v}"
bad = copy.deepcopy(data)   # the first resolve above stopped part-way
cell0 = next(c for m in bad["poster"]["macs"] for r in m["rows"] for e, c in r["cells"].items()
             if e == "MLX" and isinstance(c, dict) and c.get("src") and "mlx-vlm" in (ri.source_version("MLX", c["src"]) or ""))
cell0["src"] = [str(other.relative_to(ri.RESULTS))]
cell0["detail"] = "{v}"
ri.resolve_versions(bad)
assert cell0["detail"] == "mlx-lm 0.32.2", "the version shown is the one in the file"
assert any("docs/benchmarks.md" in x for x in data["poster"]["footer"])

# the M5 Max Super-VL MLX cell ran mlx-vlm git main 73d366e, unreleased: the 0.7.8 release refuses the model, and the
# result file's own version field says so (the session README names the build on the mlx-vlm line)
cell = next(r["cells"]["MLX"] for m in rd["tables"]["macs"] if "M5 Max" in m["name"] for r in m["rows"] if r["model"] == "Super-VL 120B")
assert cell["detail"] == "mlx-vlm git main 73d366e", cell["detail"]
assert "0.7.8" not in cell["detail"]
assert "73d366e" in open(os.path.join(ri.RESULTS, cell["src"][0]), encoding="utf-8").read()
print(f"test_readme_images: ok ({n} Kapitsa cells)")
