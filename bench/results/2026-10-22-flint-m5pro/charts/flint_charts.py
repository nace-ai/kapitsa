#!/usr/bin/env python3
"""Flint 40B-A2.5B on the M5 Pro 48 GB: the three 16:9 charts for a post (headline, decode by context, prefill by
context), in the README poster's dark style (tools/readme_images/readme_images.py: its colours, fonts and mark, and
Chrome for the rendering).
python3 flint_charts.py [OUT_DIR]  -> flint-kapitsa-m5pro-{headline,decode-by-context,prefill-by-context}{,-2x}.png
(1920x1080 and 3840x2160). The sweep's points come from ../context-sweep/sweep.json 'chosen'; the headline's from
the poster-protocol means in ../README.md (two runs each for Kapitsa and rec, llama.cpp's one run)."""
import json, math, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "tools" / "readme_images"))
import readme_images as R  # noqa: E402

T = R.POSTER["dark"]
W, H = 1920, 1080
SWEEP = json.load(open(HERE.parent / "context-sweep" / "sweep.json"))["chosen"]
SIZES = sorted(int(s) for s in SWEEP["kapitsa"])
ENG = [("Kapitsa", "kapitsa", T["kap"]), ("rec 0.1.0", "rec", "#C9D1CC"), ("llama.cpp b11804", "llamacpp", T["blue"])]
SETUP = ("MacBook Pro M5 Pro (20-core GPU), 48 GB · macOS 26.5.2 · plugged in · High Power mode · "
         "T 1.0, top_p 0.95, top_k 20 · one engine at a time · no speculative decoding")
LLAMA = ("llama.cpp: Recursive's Q4_1 GGUF, -b/-ub 2048 (not the card's 256/128), run about 1.5 h after the others "
         "on a GPU with no background load")
FOOT = [LLAMA + "; one run, its 128K rows from its own server load", "Microsoft Edge used 3-10% of the GPU on average in each Kapitsa and rec protocol run (all "
        "flagged), 0% in llama.cpp's · bench/results/2026-10-22-flint-m5pro · github.com/nace-ai/kapitsa"]
FOOT_SWEEP = [LLAMA + "; its 128K point on its own server load", "Background (Edge) GPU load in the plotted windows: "
              "Kapitsa and rec 0-24% mean per cell, llama.cpp 0% · context-sweep/sweep.json · github.com/nace-ai/kapitsa"]


def k(n):
    return f"{n // 1024}K"


def frame(title, sub, body, foot=FOOT):
    s = [R.rect(0, 0, W, H, 0, T["bg"]),
         R.mark(64, 52, 64, T["mark"], "mk0"),
         R.text(146, 92, title, "serif", 46, T["ink"]),
         R.text(64, 148, sub, "sans-sb", 26, T["ink"]),
         R.text(64, 184, SETUP, "sans", 19, T["ink2"]),
         body]
    for i, line in enumerate(foot):
        s.append(R.text(64, H - 70 + i * 30, line, "sans", 18, T["muted"]))
    css = R.CSS % {"f": R.FONTS, "bg": T["bg"]}
    return (f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            + "".join(s) + "</svg></body></html>")


def headline():
    sw = lambda m: [SWEEP[e]["131072"][m] for _, e, _ in ENG]  # noqa: E731
    rows = [  # (label, [Kapitsa, rec, llama.cpp], higher is better, format, noise tolerance)
        # tolerance: two ±3% means, ±6%; the README gives the ratios above 8K (one loaded request each) as ±10%
        ("Decode, short context (tok/s, T 1.0)", [145.7, 144.9, 92.4], True, "{:.1f}", 0.06),
        ("Decode, short context, greedy (tok/s)", [148.8, 151.9, 92.8], True, "{:.1f}", 0.06),
        ("Prefill, 4K-token prompt (tok/s)", [1374, 2138, 2072], True, "{:,.0f}", 0.06),
        ("Prefill, 16K-token prompt (tok/s)", [1108, 1941, 1773], True, "{:,.0f}", 0.06),
        ("Decode at 128K context (tok/s, one request)", sw("decode"), True, "{:.1f}", 0.10),
        ("Prefill at 128K (tok/s, one request)", sw("prefill"), True, "{:,.0f}", 0.10),
        ("Follow-up turn at 4K, first token (s, lower is better)", [0.12, 0.28, 0.22], False, "{:.2f} s", 0.06),
        ("Peak memory (GB, lower is better)", [23.3, 29.1, 26.9], False, "{:.1f} GB", 0.06),
    ]
    x0, bx, bw = 64, 690, 940
    y, rh = 250, 90
    s = []
    for j, (name, _, c) in enumerate(ENG):
        s += [R.rect(bx + j * 250, 214, 22, 22, 5, c), R.text(bx + j * 250 + 34, 232, name, "sans-sb", 22, T["ink"])]
    s.append(R.text(W - 64, 232, "bold: best, or within noise of it", "sans", 19, T["muted"], "end"))
    for i, (label, vs, hi, f, tol) in enumerate(rows):
        yy = y + i * rh
        s.append(R.rect(x0 - 16, yy, W - 2 * (x0 - 16), rh - 10, 14, T["panel"]))
        s.append(R.text(x0, yy + rh / 2 + 3, label, "sans-m", 25, T["ink"]))
        m, best = max(vs), (max(vs) if hi else min(vs))
        for j, (v, (_, _, c)) in enumerate(zip(vs, ENG)):
            by = yy + 10 + j * 22
            w = bw * v / m
            top = abs(v - best) / max(v, best) <= tol  # the best, or inside the noise of it
            s.append(R.rect(bx, by, w, 16, 5, c))
            s.append(R.text(bx + w + 14, by + 15, f.format(v), "mono-b" if top else "mono", 19,
                            T["ink"] if top else T["ink2"]))
        if tol > 0.06:
            s.append(R.text(W - 64, yy + rh / 2 + 3, "ratio ±10%", "sans-sb", 20, T["muted"], "end"))
    return frame("Flint 40B-A2.5B on the M5 Pro: Kapitsa, rec 0.1.0 and llama.cpp",
                 "Kapitsa ties rec on decode, 1.6× llama.cpp; rec and llama.cpp read prompts 1.5-1.75× as fast as Kapitsa",
                 "".join(s))


def by_context(metric, unit, title, sub):
    cx0, cx1, cy0, cy1 = 340, W - 130, 262, 800
    vals = {e: [SWEEP[e][str(n)][metric] for n in SIZES] for _, e, _ in ENG}
    top = max(max(v) for v in vals.values())
    step = 25 if top < 200 else 250
    ymax = math.ceil(top * 1.12 / step) * step
    X = lambda i: cx0 + (cx1 - cx0) * i / (len(SIZES) - 1)  # noqa: E731
    Y = lambda v: cy1 - (cy1 - cy0) * v / ymax  # noqa: E731
    s = [R.rect(48, cy0 - 42, W - 96, cy1 - cy0 + 192, 16, T["panel"])]
    for v in range(0, ymax + 1, step * (2 if ymax / step > 10 else 1)):
        s.append(f'<line x1="{cx0}" x2="{cx1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{T["dash"]}" stroke-width="1"/>')
        s.append(R.text(cx0 - 60, Y(v) + 7, f"{v:,}", "mono", 20, T["muted"], "end"))
    s.append(R.text(cx0 - 60, cy0 - 26, unit, "sans", 19, T["muted"], "end"))
    for i, n in enumerate(SIZES):
        s.append(R.text(X(i), cy1 + 34, k(n), "mono", 22, T["ink2"], "middle"))
        tol = 0.06 if n <= 8192 else 0.10  # two ±3% cells up to 8K; above 8K the README's ±10% for the ratios
        for row, e in enumerate(("rec", "llamacpp")):
            r = vals["kapitsa"][i] / vals[e][i]
            s.append(R.text(X(i), cy1 + 66 + row * 30, f"{r:.2f}×", "mono", 20,
                            T["kap"] if r > 1 + tol else (ORANGE if r < 1 - tol else T["ink2"]), "middle"))
    s.append(R.text(cx0 - 60, cy1 + 66, "Kapitsa ÷ rec", "sans", 18, T["muted"], "end"))
    s.append(R.text(cx0 - 60, cy1 + 96, "Kapitsa ÷ llama.cpp", "sans", 18, T["muted"], "end"))
    s.append(R.text((cx0 + cx1) / 2, cy1 + 134, "prompt length, tokens · one cold request per point, then 256 tokens · a point ±3% on a quiet GPU; "
                    "the ratios above 8K, with background load, ±10% (grey ratio: within that noise)", "sans", 18, T["muted"], "middle"))
    for j, (name, e, c) in enumerate(ENG):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals[e]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{5 if j == 0 else 4}" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(vals[e]):
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="7" fill="{c}" stroke="{T["panel"]}" stroke-width="3"/>')
            col = sorted((vals[e2][i] for _, e2, _ in ENG), reverse=True)
            rank = col.index(v)  # 0 top: label above; bottom: below; middle: on the side with more room
            above = rank == 0 or (rank == 1 and col[0] - v > v - col[2])
            fmt = f"{v:.1f}" if metric == "decode" else f"{v:,.0f}"
            room = max(Y(col[2]) - Y(v), Y(v) - Y(col[0])) if rank == 1 else 99  # the side the label goes to
            lx, ly, anc = (X(i) + 16, Y(v) + 7, "start") if room < 44 else (X(i), Y(v) + (-18 if above else 34), "middle")
            s.append(R.text(lx, ly, fmt, "mono-b" if j == 0 else "mono", 20, c if j == 0 else T["ink2"], anc))
        s += [R.rect(cx1 - 600 + j * 200, cy0 - 14, 22, 22, 5, c),
              R.text(cx1 - 566 + j * 200, cy0 + 4, name, "sans-sb", 22, T["ink"])]
    return frame(title, sub, "".join(s), FOOT_SWEEP)


ORANGE = R.ORANGE


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    pages = {
        "headline": headline(),
        "decode-by-context": by_context(
            "decode", "tok/s", "Flint 40B-A2.5B on the M5 Pro: decode by context",
            "Kapitsa vs rec: level to 8K, 1.10× at 16-32K (noise), 1.12-1.15× at 64-128K; both 1.4-1.6× llama.cpp"),
        "prefill-by-context": by_context(
            "prefill", "tok/s", "Flint 40B-A2.5B on the M5 Pro: prefill by context",
            "llama.cpp reads prompts 1.6-1.7× as fast as Kapitsa at every length, rec 1.4-2.3×; llama.cpp and rec level from 2K to 8K"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, html in pages.items():
            page = tmp / f"{name}.html"
            page.write_text(html, encoding="utf-8")
            for scale, suf in ((1, ""), (2, "-2x")):
                png = out / f"flint-kapitsa-m5pro-{name}{suf}.png"
                subprocess.run([R.CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                                "--no-default-browser-check", f"--force-device-scale-factor={scale}",
                                f"--user-data-dir={tmp / 'chrome'}", f"--window-size={W},{H}", f"--screenshot={png}",
                                page.as_uri()], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("wrote", png)


if __name__ == "__main__":
    main()
