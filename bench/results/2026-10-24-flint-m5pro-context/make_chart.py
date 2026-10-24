#!/usr/bin/env python3
"""Flint 40B-A2.5B context sweep on the M5 Pro 48 GB: summary, table and chart from the raw runs. Stdlib only.

  python3 make_chart.py [--png]

Reads raw/<engine>-<leg>/runs.json (one object per request, written by sweep/ctx_sweep.py) and writes
  summary.json             per engine and prompt size: the median, min and max of prefill and decode tok/s over the
                           runs kept, every run's values, and the runs left out with the reason
  table.md                 the two tables (median, min-max) the README quotes
  flint-m5pro-context.svg  two panels, prefill and decode against the prompt length (log2 x axis): the median as a
                           point, the min-max of the runs as a bar
  --png                    also flint-m5pro-context.png (2x), rendered by headless Chrome on the CPU

A run is left out of a metric when the foreign GPU load (gpu_hogs.py monitor: every process but WindowServer, the
macOS media daemons and the engine's own tree) averaged over 5% in that metric's window: the prefill window (request
to first token) for prefill, the first token to the end for decode. A run is left out of decode when the model ended
its answer before 256 tokens (the request asks for 512). Their numbers stay in summary.json."""

import argparse, glob, json, math, os, statistics, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINES = [
    ("kapitsa", "Kapitsa 71d6891", "#0E9F6E"),
    ("llamacpp", "llama.cpp b11840", "#2F6FDE"),
]
UNSUPPORTED = [
    ("MLX (mlx-lm)", "not supported (mlx-lm 0.32.2)")
]  # no Flint model in mlx-lm 0.32.2: not run
FOREIGN_MAX = 5.0
MIN_ANSWER = 256  # a decode value needs an answer of at least this many tokens
SETUP = (
    "MacBook Pro M5 Pro (20-core GPU), 48 GB · macOS 26.5.2 · AC power, High Power mode · Flint 4-bit "
    "(Kapitsa: Recursive's MLX 4-bit; llama.cpp: Recursive's Q4_1 GGUF) · T 1.0, top_p 0.95, top_k 20"
)


def k(n):
    return f"{n // 1024}K"


def load():
    runs = []
    for p in sorted(glob.glob(str(HERE / "raw" / "*" / "runs.json"))):
        runs += [r for r in json.load(open(p)) if not r.get("skipped")]
    return runs


def summarize(runs):
    out = {}
    for e, _, _ in ENGINES:
        out[e] = {}
        for n in sorted({r["size"] for r in runs if r["engine"] == e}):
            rs = sorted(
                (r for r in runs if r["engine"] == e and r["size"] == n),
                key=lambda r: r["start_epoch"],
            )
            cell = {"prompt_tokens": sorted({r["prompt_tokens"] for r in rs})}
            for m, win in (
                ("prefill_tps", "foreign_gpu_prefill"),
                ("decode_tps", "foreign_gpu_decode"),
            ):
                kept, left = [], []
                for r in rs:
                    f = (r.get(win) or {}).get("mean_pct")
                    if r.get(m) is None:
                        left.append(
                            {
                                "leg": r["leg"],
                                "start_epoch": r["start_epoch"],
                                "reason": "no value",
                            }
                        )
                    elif m == "decode_tps" and r["completion_tokens"] < MIN_ANSWER:
                        left.append(
                            {
                                "leg": r["leg"],
                                "start_epoch": r["start_epoch"],
                                "value": round(r[m], 2),
                                "reason": f"the answer ended at {r['completion_tokens']} tokens",
                            }
                        )
                    elif f is not None and f > FOREIGN_MAX:
                        left.append(
                            {
                                "leg": r["leg"],
                                "start_epoch": r["start_epoch"],
                                "value": round(r[m], 2),
                                "reason": f"foreign GPU load {f}% mean ({(r.get(win) or {}).get('top')})",
                            }
                        )
                    else:
                        kept.append(r[m])
                c = {
                    "n": len(kept),
                    "values": [round(v, 2) for v in kept],
                    "left_out": left,
                }
                if kept:
                    med = statistics.median(kept)
                    c.update(
                        {
                            "median": round(med, 2),
                            "min": round(min(kept), 2),
                            "max": round(max(kept), 2),
                            "spread_pct": round(100 * (max(kept) - min(kept)) / med, 1),
                        }
                    )
                cell[m.replace("_tps", "")] = c
            cell["completion_tokens"] = sorted({r["completion_tokens"] for r in rs})
            out[e][str(n)] = cell
    return out


def table(s):
    sizes = sorted({int(n) for e in s for n in s[e]})
    lines = []
    for m, title, f in (
        ("prefill", "Prefill (tok/s)", "{:,.0f}"),
        ("decode", "Decode (tok/s)", "{:.1f}"),
    ):
        lines += [
            f"| {title} | " + " | ".join(k(n) for n in sizes) + " |",
            "| --- |" + " ---: |" * len(sizes),
        ]
        for e, name, _ in ENGINES:
            cells = []
            for n in sizes:
                c = s[e].get(str(n), {}).get(m, {})
                cells.append(f.format(c["median"]) if c.get("median") else "–")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        for name, why in UNSUPPORTED:
            lines.append(
                f"| {name} | "
                + " | ".join([f"<sub>{why}</sub>"] + ["×"] * (len(sizes) - 1))
                + " |"
            )
        for e, name, _ in ENGINES:
            cells = []
            for n in sizes:
                c = s[e].get(str(n), {}).get(m, {})
                cells.append(
                    f"<sub>{f.format(c['min'])}-{f.format(c['max'])} ({c['n']})</sub>"
                    if c.get("median")
                    else "–"
                )
            lines.append(
                f"| <sub>{name}: min-max (runs)</sub> | " + " | ".join(cells) + " |"
            )
        cells = []
        for n in sizes:
            a, b = (
                s[e].get(str(n), {}).get(m, {}).get("median")
                for e in ("kapitsa", "llamacpp")
            )
            cells.append(f"<sub>{a / b:.2f}×</sub>" if a and b else "–")
        lines += ["| <sub>Kapitsa ÷ llama.cpp</sub> | " + " | ".join(cells) + " |", ""]
    return "\n".join(lines)


def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, t, size=14, fill="#3A3F44", anchor="start", weight=400):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
        f'font-weight="{weight}">{esc(t)}</text>'
    )


def panel(s, m, title, x0, y0, w, h, fmt):
    sizes = sorted({int(n) for e in s for n in s[e]})
    lo, hi = math.log2(sizes[0]), math.log2(sizes[-1])
    top = max(s[e][str(n)][m].get("max", 0) for e in s for n in s[e])
    step = next(
        st for st in (10, 20, 25, 50, 100, 200, 250, 500, 1000) if top * 1.1 / st <= 8
    )
    ymax = math.ceil(top * 1.1 / step) * step
    pad = 28
    X = lambda n: x0 + pad + (w - 2 * pad) * (math.log2(n) - lo) / (hi - lo)  # noqa: E731
    Y = lambda v: y0 + h - h * v / ymax  # noqa: E731
    o = [txt(x0, y0 - 18, title, 18, "#15181B", weight=600)]
    for v in range(0, ymax + 1, step):
        o.append(
            f'<line x1="{x0}" x2="{x0 + w}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#E3E6E8"/>'
        )
        o.append(txt(x0 - 10, Y(v) + 5, f"{v:,}", 13, "#6B7278", "end"))
    for n in sizes:
        o.append(
            f'<line x1="{X(n):.1f}" x2="{X(n):.1f}" y1="{y0 + h}" y2="{y0 + h + 5}" stroke="#9AA1A6"/>'
        )
        o.append(txt(X(n), y0 + h + 22, k(n), 13, "#3A3F44", "middle"))
        a, b = (
            s[e].get(str(n), {}).get(m, {}).get("median")
            for e in ("kapitsa", "llamacpp")
        )
        if a and b:
            o.append(
                txt(
                    X(n),
                    y0 + h + 42,
                    f"{a / b:.2f}×",
                    13,
                    "#0E9F6E" if a > b else "#B4541F",
                    "middle",
                    600,
                )
            )
    o.append(txt(x0 - 10, y0 + h + 36, "Kapitsa ÷", 12, "#6B7278", "end"))
    o.append(txt(x0 - 10, y0 + h + 50, "llama.cpp", 12, "#6B7278", "end"))
    o.append(
        txt(
            x0 + w / 2,
            y0 + h + 66,
            "prompt length (tokens, log2 scale)",
            13,
            "#6B7278",
            "middle",
        )
    )
    for e, _, col in ENGINES:
        pts = [
            (n, X(int(n)), s[e][n][m])
            for n in sorted(s[e], key=int)
            if s[e][n][m].get("median")
        ]
        o.append(
            f'<polyline points="{" ".join(f"{x:.1f},{Y(c["median"]):.1f}" for _, x, c in pts)}" fill="none" '
            f'stroke="{col}" stroke-width="2.5" stroke-linejoin="round"/>'
        )
        for n, x, c in pts:
            o.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{Y(c["min"]):.1f}" y2="{Y(c["max"]):.1f}" stroke="{col}" '
                f'stroke-width="7" stroke-opacity="0.35" stroke-linecap="round"/>'
            )
            o.append(
                f'<circle cx="{x:.1f}" cy="{Y(c["median"]):.1f}" r="4.5" fill="{col}" stroke="#fff" stroke-width="1.5"/>'
            )
            other = max(
                (s[e2].get(n, {}).get(m, {}).get("median") or 0)
                for e2, _, _ in ENGINES
                if e2 != e
            )
            above = (
                c["median"] >= other
            )  # the higher engine's label above its point, the lower's below
            o.append(
                txt(
                    x,
                    Y(c["median"]) + (-12 if above else 22),
                    fmt.format(c["median"]),
                    12,
                    col,
                    "middle",
                    600,
                )
            )
    return "".join(o)


def svg(s):
    W, H = 1400, 640
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        "font-family=\"-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif\">",
        f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>',
        txt(
            40,
            42,
            "Recursive Flint 40B-A2.5B, 4-bit, on an M5 Pro 48 GB: speed by prompt length",
            22,
            "#15181B",
            weight=700,
        ),
        txt(40, 68, SETUP, 13, "#6B7278"),
    ]
    for j, (_, name, col) in enumerate(ENGINES):
        o += [
            f'<rect x="{40 + j * 190}" y="86" width="14" height="14" rx="3" fill="{col}"/>',
            txt(60 + j * 190, 98, name, 14, "#15181B", weight=600),
        ]
    for j, (name, why) in enumerate(UNSUPPORTED):
        x = 40 + (len(ENGINES) + j) * 190
        o += [
            f'<rect x="{x}" y="86" width="14" height="14" rx="3" fill="none" stroke="#9AA1A6" stroke-dasharray="3 2"/>',
            txt(x + 20, 98, f"{name}: {why}", 14, "#6B7278"),
        ]
    o.append(
        txt(
            W - 40, 98, "point: median of the runs · bar: min-max", 13, "#6B7278", "end"
        )
    )
    o.append(
        panel(
            s,
            "prefill",
            "Prefill: prompt tokens ÷ time to first token (tok/s)",
            100,
            150,
            540,
            370,
            "{:,.0f}",
        )
    )
    o.append(
        panel(
            s,
            "decode",
            "Decode: a 512-token answer after the prompt (tok/s)",
            820,
            150,
            540,
            370,
            "{:.1f}",
        )
    )
    o.append(
        txt(
            40,
            H - 14,
            "Each request cold (a random nonce opens the prompt; no prompt cache), one engine at a time, "
            "one request at a time, plain decoding (no speculative decoding). bench/results/2026-10-24-flint-m5pro-context",
            12,
            "#6B7278",
        )
    )
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true")
    a = ap.parse_args()
    s = summarize(load())
    json.dump(s, open(HERE / "summary.json", "w"), indent=1)
    (HERE / "table.md").write_text(table(s))
    (HERE / "flint-m5pro-context.svg").write_text(svg(s))
    print(table(s))
    if a.png:
        chrome = os.environ.get(
            "CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        )
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "c.html"
            page.write_text(
                '<!doctype html><html><body style="margin:0">'
                + svg(s)
                + "</body></html>"
            )
            subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--force-device-scale-factor=2",
                    f"--user-data-dir={tmp}/chrome",
                    "--window-size=1400,640",
                    f"--screenshot={HERE / 'flint-m5pro-context.png'}",
                    page.as_uri(),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    main()
