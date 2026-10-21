#!/usr/bin/env python3
"""16:9 Reddit benchmark grid in the README poster's style: Splash 1.4.0 vs Kapitsa 0.2 (KFlash), MLX & more.
python3 reddit_grid.py OUT_DIR   -> kflash-engines-benchmark-16x9.png (1920x1080) and -2x.png (3840x2160)"""
import copy, json, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent  # this folder: readme_images.py, its data and assets
sys.path.insert(0, str(D))
import readme_images as R  # noqa: E402
R.RH_MIN = 88  # the grid has no prefill line, so its rows are shorter than the README poster's

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
# The numbers below are a second copy of readme_images.json's: when the poster data changes, re-check this file.
base = json.load(open(D / "readme_images.json"))
by = {e["name"]: copy.deepcopy(e) for e in base["engines"]}
ORDER = ["Kapitsa", "Splash", "mlx-serve", "oMLX", "Ollama", "MTPLX", "MLX", "llama.cpp", "LM Studio"]
VER = {"Splash": "1.4.0", "mlx-serve": "26.10.2", "oMLX": "0.7.1rc1", "Ollama": "0.40.5 / 0.40.6", "MTPLX": "2.12.2",
       "MLX": "mlx-lm 0.32.2\nmlx-vlm 0.7.8", "llama.cpp": "b11669", "LM Studio": "MLX runtime\n1.13.1"}
engines = []
for n in ORDER:
    e = by[n]
    e.pop("method", None)
    if n in VER:
        e["version"] = VER[n]
    engines.append(e)


def row(model, vals):
    cells = {}
    for n, v in zip(ORDER, vals):
        if v == "x":
            cells[n] = "x"
        else:
            val, meth = v
            cells[n] = {"value": val, "method": meth}
    return {"model": model, "cells": cells}


K, D2, M, NS = "KFlash", "DFlash2", "MTP", "no spec"
data = {
    "engines": engines,
    "poster": {
        "size": [1920, 1080],
        "title": "Splash 1.4.0 vs Kapitsa 0.2 (KFlash), MLX & more",
        "subtitle": ["Decode tok/s of a 512-token coding answer, T 0.6, median of 5 · 4-bit weights · each engine in its fastest setup",
                     "Qwen3.8 27B · Qwen3.6 35B-A3B · Qwen3.5 4B · the newest releases, one harness, two MacBook Pros"],
        "footer": ["× can't run it (Splash: only the 27B and 35B, on an M3 or newer) · no spec: no speculative decoding · grey: share of the row's fastest",
                   "Kapitsa 0.2 and Splash 1.4.0 measured 2026-10-21, the others' latest releases at that time · github.com/nace-ai/kapitsa · kapitsa bench"],
        "macs": [
            {"name": "MacBook Pro M4 Pro (2024) 48 GB", "rows": [
                row("Qwen3.8 27B", [(63, K), (53, D2), (54, D2), (45, D2), (32, M), (36, M), (28, M), (18, M), (15, NS)]),
                row("Qwen3.6 35B-A3B", [(161, K), (135, D2), (124, M), (127, D2), (89, M), (80, M), (86, D2), (69, M), (86, NS)]),
                row("Qwen3.5 4B", [(201, K), "x", (137, M), (102, M), (109, M), (104, M), (88, M), (70, M), (81, NS)]),
            ]},
            {"name": "MacBook Pro M1 Pro (2021) 16 GB", "rows": [
                row("Qwen3.5 4B", [(95, K), "x", (62, M), (55, M), (55, M), (42, M), (46, NS), (38, M), (47, NS)]),
            ]},
        ],
    },
}

for scale, suffix in ((1, ""), (2, "-2x")):
    html, w, h, notes = R.poster(data, "dark")
    for n in notes:
        print("note:", n)
    page = out / f"grid{suffix}.html"
    page.write_text(html, encoding="utf-8")
    png = out / f"kflash-engines-benchmark-16x9{suffix}.png"
    subprocess.run([R.CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check",
                    f"--force-device-scale-factor={scale}", f"--user-data-dir={out / 'chrome'}", f"--window-size={w},{h}",
                    f"--screenshot={png}", page.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", png, w, h)
