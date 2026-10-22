#!/usr/bin/env python3
"""Flint 40B-A2.5B on the M5 Pro 48 GB: the three 16:9 charts for a post (headline, decode by context, prefill by
context), in the README poster's dark style (tools/readme_images/readme_images.py: its colours, fonts and mark, and
Chrome for the rendering).
python3 flint_charts.py [OUT_DIR]  -> flint-kapitsa-m5pro-{headline,decode-by-context,prefill-by-context}{,-2x}.png
(1920x1080 and 3840x2160). The sweep's points come from ../context-sweep/sweep.json 'chosen'; the headline's from
the poster-protocol means in ../README.md (two runs each)."""
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
ENG = [("Kapitsa", "kapitsa", T["kap"]), ("rec 0.1.0", "rec", "#C9D1CC")]
SETUP = ("MacBook Pro M5 Pro (20-core GPU), 48 GB · macOS 26.5.2 · plugged in · High Power mode · "
         "T 1.0, top_p 0.95, top_k 20 · one engine at a time · no speculative decoding")
LLAMA = ("llama.cpp b11804 not run on this Mac: its 24.2 GiB Q4_1 GGUF did not fit next to the MLX weights within the "
         "session's 30 GiB free-disk floor")
FOOT = [LLAMA, "Microsoft Edge used 3-10% of the GPU on average per protocol run (all flagged in the results) · "
               "bench/results/2026-10-22-flint-m5pro · github.com/nace-ai/kapitsa"]
FOOT_SWEEP = [LLAMA, "Background (Edge) GPU load in the plotted windows: 0-24% mean per cell, listed in "
                     "context-sweep/sweep.json · bench/results/2026-10-22-flint-m5pro · github.com/nace-ai/kapitsa"]


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
    rows = [  # (label, kapitsa, rec, higher is better, format, tie tolerance)
        # tolerance: two ±3% means, ±6%; the README gives the ratios above 8K (one loaded request each) as ±10%
        ("Decode, short context (tok/s, T 1.0)", 145.7, 144.9, True, "{:.1f}", 0.06),
        ("Decode, short context, greedy (tok/s)", 148.8, 151.9, True, "{:.1f}", 0.06),
        ("Prefill, 4K-token prompt (tok/s)", 1374, 2138, True, "{:,.0f}", 0.06),
        ("Prefill, 16K-token prompt (tok/s)", 1108, 1941, True, "{:,.0f}", 0.06),
        ("Decode at 128K context (tok/s, one request)", SWEEP["kapitsa"]["131072"]["decode"],
         SWEEP["rec"]["131072"]["decode"], True, "{:.1f}", 0.10),
        ("Prefill at 128K (tok/s, one request)", SWEEP["kapitsa"]["131072"]["prefill"],
         SWEEP["rec"]["131072"]["prefill"], True, "{:,.0f}", 0.10),
        ("Follow-up turn at 4K, first token (s, lower is better)", 0.12, 0.28, False, "{:.2f} s", 0.06),
        ("Peak memory (GB, lower is better)", 23.3, 29.1, False, "{:.1f} GB", 0.06),
    ]
    x0, lw, bx, bw = 64, 600, 690, 940
    y, rh = 250, 86
    s = []
    for j, (name, _, c) in enumerate(ENG):
        s += [R.rect(bx + j * 220, 214, 22, 22, 5, c), R.text(bx + j * 220 + 34, 232, name, "sans-sb", 22, T["ink"])]
    for i, (label, kv, rv, hi, f, tol) in enumerate(rows):
        yy = y + i * rh
        s.append(R.rect(x0 - 16, yy, W - 2 * (x0 - 16), rh - 10, 14, T["panel"]))
        s.append(R.text(x0, yy + rh / 2 + 3, label, "sans-m", 25, T["ink"]))
        m = max(kv, rv)
        win = (kv >= rv) if hi else (kv <= rv)
        tie = abs(kv - rv) / m <= tol
        for j, (v, (_, _, c)) in enumerate(zip((kv, rv), ENG)):
            by = yy + 12 + j * 32
            w = bw * v / m
            best = (j == 0) == win and not tie
            s.append(R.rect(bx, by, w, 24, 6, c if j == 0 else T["greys"][4]))
            s.append(R.text(bx + w + 14, by + 20, f.format(v), "mono-b" if best else "mono", 22,
                            T["ink"] if best else T["ink2"]))
        if tie:
            s.append(R.text(W - 64, yy + rh / 2 + 3, "tie", "sans-sb", 20, T["muted"], "end"))
        elif tol > 0.06:
            s.append(R.text(W - 64, yy + rh / 2 + 3, "ratio ±10%", "sans-sb", 20, T["muted"], "end"))
    return frame("Flint 40B-A2.5B on the M5 Pro: Kapitsa vs rec 0.1.0",
                 "Decode is a tie; rec reads prompts 1.4-2.3× as fast; Kapitsa starts follow-up turns sooner and uses less memory",
                 "".join(s))


def by_context(metric, unit, title, sub):
    cx0, cx1, cy0, cy1 = 340, W - 130, 262, 830
    vals = {e: [SWEEP[e][str(n)][metric] for n in SIZES] for _, e, _ in ENG}
    top = max(max(v) for v in vals.values())
    step = 25 if top < 200 else 250
    ymax = math.ceil(top * 1.12 / step) * step
    X = lambda i: cx0 + (cx1 - cx0) * i / (len(SIZES) - 1)  # noqa: E731
    Y = lambda v: cy1 - (cy1 - cy0) * v / ymax  # noqa: E731
    s = [R.rect(48, cy0 - 42, W - 96, cy1 - cy0 + 162, 16, T["panel"])]
    for v in range(0, ymax + 1, step * (2 if ymax / step > 10 else 1)):
        s.append(f'<line x1="{cx0}" x2="{cx1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{T["dash"]}" stroke-width="1"/>')
        s.append(R.text(cx0 - 60, Y(v) + 7, f"{v:,}", "mono", 20, T["muted"], "end"))
    s.append(R.text(cx0 - 60, cy0 - 26, unit, "sans", 19, T["muted"], "end"))
    for i, n in enumerate(SIZES):
        s.append(R.text(X(i), cy1 + 34, k(n), "mono", 22, T["ink2"], "middle"))
        r = vals["kapitsa"][i] / vals["rec"][i]
        tol = 0.06 if n <= 8192 else 0.10  # two ±3% cells up to 8K; above 8K the README's ±10% for the ratios
        s.append(R.text(X(i), cy1 + 70, f"{r:.2f}×", "mono", 20,
                        T["kap"] if r > 1 + tol else (ORANGE if r < 1 - tol else T["ink2"]), "middle"))
    s.append(R.text(cx0 - 60, cy1 + 70, "Kapitsa ÷ rec, grey = noise", "sans", 18, T["muted"], "end"))
    s.append(R.text((cx0 + cx1) / 2, cy1 + 104, "prompt length, tokens · one cold request per point, then 256 tokens · a point ±3% on a quiet GPU; "
                    "the ratios above 8K, with background load, ±10%", "sans", 18, T["muted"], "middle"))
    for j, (name, e, c) in enumerate(ENG):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals[e]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="{5 if j == 0 else 4}" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for i, v in enumerate(vals[e]):
            s.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="7" fill="{c}" stroke="{T["panel"]}" stroke-width="3"/>')
            other = vals[ENG[1 - j][1]][i]
            above = v >= other
            fmt = f"{v:.1f}" if metric == "decode" else f"{v:,.0f}"
            s.append(R.text(X(i), Y(v) + (-18 if above else 34), fmt, "mono-b" if j == 0 else "mono", 20,
                            c if j == 0 else T["ink2"], "middle"))
        s += [R.rect(cx1 - 360 + j * 190, cy0 - 14, 22, 22, 5, c),
              R.text(cx1 - 326 + j * 190, cy0 + 4, name, "sans-sb", 22, T["ink"])]
    return frame(title, sub, "".join(s), FOOT_SWEEP)


ORANGE = R.ORANGE


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    pages = {
        "headline": headline(),
        "decode-by-context": by_context(
            "decode", "tok/s", "Flint 40B-A2.5B on the M5 Pro: decode by context",
            "Level with rec up to 8K; Kapitsa 1.10× at 16-32K (within noise), 1.12-1.15× at 64-128K (just above it)"),
        "prefill-by-context": by_context(
            "prefill", "tok/s", "Flint 40B-A2.5B on the M5 Pro: prefill by context",
            "rec reads prompts faster at every length: 1.4× at 1K, 1.6× at 4K, 2.3× at 128K"),
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
