#!/usr/bin/env python3
# Renders the README's benchmark poster (docs/assets/benchmarks-{light,dark}.webp) and header (header-{light,dark,narrow}.svg)
# from readme_images.json beside it. Edit the numbers there, then: python3 readme_images.py --out <kapitsa repo>/docs/assets [--poster | --headers]
# The poster needs Google Chrome (run headless, CPU only) and cwebp; the headers need nothing but Python 3.
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ASSETS = HERE / "assets"           # engine logos and the header's glyph outlines
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
CWEBP = os.environ.get("CWEBP", "cwebp")
FONTS = "/System/Library/Fonts"           # macOS: New York, SF Pro, SF Mono
DASH, CROSS = "—", "×"


def num(v, nd=2):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def fmt(v):
    return str(round(v)) if isinstance(v, (int, float)) else DASH


def cell(row, engine):
    """(value, method, raw) of one cell: value is a number, None (a dash) or "x" (can't run)"""
    c = row["cells"].get(engine)
    if c is None or c == "x":
        return c, None, c
    if isinstance(c, (int, float)):
        return c, None, c
    return c.get("value"), c.get("method"), c


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check(data):
    names = [e["name"] for e in data["engines"]]
    if names[0] != "Kapitsa":
        sys.exit("readme_images.json: Kapitsa must be the first engine")
    for mac in data["poster"]["macs"]:
        for row in mac["rows"]:
            for e in row["cells"]:
                if e not in names:
                    sys.exit(f"readme_images.json: unknown engine {e!r} in {mac['name']} / {row['model']}")
    h = data["header"]
    for e in h["engines"]:
        if e not in names:
            sys.exit(f"readme_images.json: unknown engine {e!r} in header.engines")
    header_row(data)
    T = data.get("tables")
    if T:
        for mac in T["macs"]:
            for row in mac["rows"]:
                for e in row["cells"]:
                    if e not in names:
                        sys.exit(f"readme_images.json: unknown engine {e!r} in tables {mac['name']} / {row['model']}")
        for mac in data["poster"]["macs"] + T["macs"]:
            for row in mac["rows"]:
                if row["model"] not in T["models"]:
                    sys.exit(f"readme_images.json: {mac['name']} / {row['model']} has no entry in tables.models")
        for mac in data["poster"]["macs"]:
            if mac["name"] not in {m["name"] for m in T["macs"]}:
                sys.exit(f"readme_images.json: poster Mac {mac['name']} has no entry in tables.macs")


# ---------------------------------------------------------------- the README's per-Mac tables (Markdown)
NB = "\u00a0"
LEGEND = "<sub>Each cell: prefill / decode, tokens/s"     # where the tables start in README.md


def table_rows(data, tmac):
    """a Mac's rows for its table: its poster rows and its tables rows, in tables.models order, rows without a number
    dropped"""
    T = data["tables"]
    rows = [r for m in data["poster"]["macs"] if m["name"] == tmac["name"] for r in m["rows"]] + tmac["rows"]
    rows = [r for r in rows if any(is_num(cell(r, e)[0]) for e in r["cells"])]
    order = list(T["models"])
    return sorted(rows, key=lambda r: order.index(r["model"]))


def table_cell(row, engine, top, shared=None):
    v, meth, raw = cell(row, engine)
    if v is None:
        return DASH
    if v == "x":
        return CROSS
    tie = engine in row.get("tie", [])
    d = ("≈" if tie else "") + fmt(v)
    if v == top or tie:
        d = f"**{d}**"
    pre = raw.get("prefill") if isinstance(raw, dict) else None
    s = f"{fmt(pre)}{NB}/{NB}{d}" if pre else d
    if v == top:
        s = f"🟩{NB}{s}"
    detail = raw.get("detail") if isinstance(raw, dict) else None
    sub = " · ".join(x for x in (meth, None if detail == shared else detail) if x)
    return s + (f"<br><sub>{sub}</sub>" if sub else "")


def tables(data):
    """the README's per-Mac tables: a legend line and one <details> block per Mac"""
    T = data["tables"]
    out = [LEGEND + " (a lone number is decode: no prefill run) · 🟩 fastest decode · ≈ a decode tie with Kapitsa · "
           "small type: the decoding method · × can't run on that Mac · — not measured</sub>"]
    for tmac in T["macs"]:
        rows = table_rows(data, tmac)
        engines = [e for e in data["engines"] if any(is_num(cell(r, e["name"])[0]) for r in rows)]
        shared = {}                         # an engine whose every number on this Mac has one detail: it heads the column
        for e in engines:
            ds = {(cell(r, e["name"])[2] or {}).get("detail") for r in rows if is_num(cell(r, e["name"])[0])}
            shared[e["name"]] = ds.pop() if len(ds) == 1 else None
        head = []
        for e in engines:
            if e["name"] == "Kapitsa":
                head.append("<mark>**Kapitsa**</mark>")
                continue
            name = e["name"].replace(" ", "&nbsp;").replace("-", "\u2011")
            ver = shared[e["name"]] or T.get("versions", {}).get(e["name"], e.get("version", "")).replace("\n", " ")
            head.append(f"<sub>{name}<br>{ver}</sub>" if ver else f"<sub>{name}</sub>")
        n = len(rows)
        out += ["", "<details>", f"<summary><b>{tmac['title']}</b> · {n} model{'s' if n != 1 else ''}</summary>", "",
                "| Model | " + " | ".join(head) + " |", "| --- |" + " ---: |" * len(engines)]
        for r in rows:
            m = T["models"][r["model"]]
            top = max(cell(r, e["name"])[0] for e in engines if is_num(cell(r, e["name"])[0]))
            out.append(f"| **{m['name']}** · `{m['alias']}`<br><sub>{m['about']}</sub> | "
                       + " | ".join(table_cell(r, e["name"], top, shared[e["name"]]) for e in engines) + " |")
        if tmac.get("caption"):
            out += ["", f"<sub>{tmac['caption']}</sub>"]
        out += ["", "</details>"]
    return "\n".join(out) + "\n"


def readme_with_tables(readme, block):
    """README text with its Models section's tables replaced: from the legend line (or, the first time, the section's
    first <details>) to the section's last </details>"""
    start = readme.index("\n## Models\n")
    end = readme.find("\n## ", start + 1)
    end = len(readme) if end < 0 else end
    sec = readme[start:end]
    a = sec.find(LEGEND)
    a = sec.index("<details>") if a < 0 else a
    b = sec.rindex("</details>") + len("</details>\n")
    return readme[:start] + sec[:a] + block + sec[b:] + readme[end:]


def method_label(mid, base, meth, cls, size, fill):
    """the method under a number; a long one (no speculative decoding) on two lines in smaller type"""
    if len(meth) <= 9:
        return [text(mid, base, meth, cls, size, fill, "middle")]
    words = meth.split()
    cut = max(range(1, len(words)), key=lambda i: -abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))) * 100 - i)
    a, b = " ".join(words[:cut]), " ".join(words[cut:])
    small = round(size * 0.72)
    return [text(mid, base - 1, a, cls, small, fill, "middle"), text(mid, base - 1 + round(small * 1.15), b, cls, small, fill, "middle")]


def header_methods(data):
    """engine -> the method shown under its name, or None (shown under each number). Always None: every method sits under
    its number, so MTP and DFlash2 read in the same place in every column."""
    return {e["name"]: None for e in data["engines"]}
    out = {}
    for e in data["engines"]:
        ms = {cell(r, e["name"])[1] or "" for m in data["poster"]["macs"] for r in m["rows"]
              if is_num(cell(r, e["name"])[0])}
        if not ms:                          # no numbers yet: the engine's own method
            out[e["name"]] = e.get("method")
        elif len(ms) == 1:
            out[e["name"]] = ms.pop() or None
        else:
            out[e["name"]] = None
    return out


def header_row(data):
    h = data["header"]
    for mac in data["poster"]["macs"]:
        for row in mac["rows"]:
            if mac["name"] == h["mac"] and row["model"] == h["model"]:
                return row
    sys.exit(f"readme_images.json: header row {h['mac']} / {h['model']} is not in poster.macs")


# ---------------------------------------------------------------- the poster (HTML/SVG, rendered by Chrome)

POSTER = {
    "dark": dict(bg="#0B0F0D", panel="#111714", ink="#E8EEE9", ink2="#B9C4BE", muted="#7D8983",
                 kap="#35A852", blue="#6A86C9", method="#9AA59F", unit="#E2F5E6", badge="#26302B", dash="#3A4640",
                 greys=["#141A17", "#1A211E", "#212925", "#28312D", "#303A35", "#39443F"],
                 greens=["#173A22", "#1C4A2A", "#215C32", "#27703B", "#2E8A46", "#35A852"],
                 lap_edge="#C9D1CC", lap_body="#2A322E", lap_screen="#08100B", mark=("#9DF0B0", "#3FB55A", "#1E5A30")),
    "light": dict(bg="#F6F2EA", panel="#FFFDF8", ink="#1C2430", ink2="#3E4652", muted="#7E848C",
                  kap="#2E9A4A", blue="#4F72C8", method="#4B535D", unit="#F2FBF3", badge="#E4DED2", dash="#9A9FA6",
                  greys=["#F1EEE7", "#E6E2D9", "#D9D4CA", "#CBC5BA", "#BDB6AA", "#AEA79B"],
                  greens=["#DCEFDD", "#C3E5C6", "#A3D6A9", "#7DC487", "#55AF65", "#2E9A4A"],
                  lap_edge="#3A4048", lap_body="#D7DCE0", lap_screen="#101820", mark=("#1E5A30", "#3FB55A", "#B6E8C3")),
}
ORANGE = "#EE8E4E"
X0, LW, KW = 40, 360, 212           # left edge, the model-name column, Kapitsa's column (the others share the rest)
KW_PRE = 296                        # Kapitsa's column when it shows 'prefill / decode'
TILE, TILE_Y = 70, 202              # logo tiles
PANELS_Y, HEAD, PAD, GAP = 372, 74, 14, 30
FOOT_GAP, FOOT_LINE, FOOT_PAD = 44, 32, 30
RH_MIN, RH_MAX = 88, 150
F_MODEL, F_VAL, F_KVAL, F_UNIT, F_METH, F_KMETH, F_DASH = 32, 42, 44, 25, 20, 21, 28
F_CCOMB = 25                        # the other engines' 'prefill / decode'
F_COMB = 40                         # 'prefill / decode', both at one size and weight
M_GAP = 14                          # a number's baseline to the top of its method label
CAP = 0.705                         # cap height of SF Pro and SF Mono, per em

CSS = """@font-face{font-family:KSerif;src:url("file://%(f)s/NewYork.ttf");font-weight:1 1000}
@font-face{font-family:KSans;src:url("file://%(f)s/SFNS.ttf");font-weight:1 1000}
@font-face{font-family:KMono;src:url("file://%(f)s/SFNSMono.ttf");font-weight:1 1000}
html,body{margin:0;padding:0;background:%(bg)s}svg{display:block}
.serif{font-family:KSerif,Georgia,serif;font-variation-settings:"opsz" 256,"wght" 400}
.sans,.sans-m,.sans-sb,.sans-b{font-family:KSans,system-ui,sans-serif}
.sans{font-variation-settings:"opsz" 28,"wght" 400}.sans-m{font-variation-settings:"opsz" 28,"wght" 510}
.sans-sb{font-variation-settings:"opsz" 28,"wght" 590}.sans-b{font-variation-settings:"opsz" 28,"wght" 700}
.mono,.mono-b{font-family:KMono,Menlo,monospace}
.mono{font-variation-settings:"wght" 400,"YAXS" 324.3}.mono-b{font-variation-settings:"wght" 683.3,"YAXS" 335.8}"""


def text(x, y, s, cls, size, fill, anchor="start"):
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{num(x)}" y="{num(y)}" class="{cls}" font-size="{size}" fill="{fill}"{a}>{escape(s)}</text>'


def rect(x, y, w, h, r, fill, extra=""):
    rr = f' rx="{num(r)}"' if r else ""
    return f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}"{rr} fill="{fill}"{extra}/>'


def mark_body(fill, stroke, grad):
    """the beam mark of docs/assets/brand (144 x 104 around its own origin); fill/stroke map hi|main|dim -> attributes"""
    fr = "".join(f'<rect x="{x}" y="4" width="7.5" height="92" rx="3.75" {fill(c)}/>'
                 for x, c in ((46.25, "main"), (59.25, "dim"), (72.25, "main"), (85.25, "dim")))
    return (f'<path d="M0 50H46.75" stroke="url(#{grad})" stroke-width="9"/>{fr}'
            f'<path d="M46.25 50H95" {stroke("hi")} stroke-width="9"/>'
            f'<path d="M95 50L136 5M95 50L136 95" {stroke("main")} stroke-width="9" stroke-linecap="round"/>'
            f'<path d="M93 50H136" {stroke("hi")} stroke-width="9" stroke-linecap="round"/>')


def mark(x, y, width, colors, grad):
    hi, main, dim = colors
    c = {"hi": hi, "main": main, "dim": dim}
    s = width / 144
    return (f'<defs><linearGradient id="{grad}" x1="0" x2="46.25" y1="0" y2="0" gradientUnits="userSpaceOnUse">'
            f'<stop offset="0" stop-color="{hi}" stop-opacity="0"/><stop offset="1" stop-color="{hi}"/></linearGradient></defs>'
            f'<g transform="translate({num(x + 2 * s)} {num(y + 2 * s)}) scale({num(s, 4)})" fill="none">'
            + mark_body(lambda k: f'fill="{c[k]}"', lambda k: f'stroke="{c[k]}"', grad) + "</g>")


def laptop(x, y, w, t):
    """a plain MacBook drawing: lid with a notch, a base wider than the lid"""
    h = w * 0.62
    sw, sh = w * 0.84, h * 0.78
    sx, by = x + (w - sw) / 2, y + h * 0.78
    ins, nw, nh = w * 0.022, sw * 0.16, sh * 0.06
    pts = [(x, by + h * .06), (x + w, by + h * .06), (x + w * .97, by + h * .11), (x + w * .03, by + h * .11)]
    return "".join([
        rect(sx, y, sw, sh, w * 0.035, t["lap_body"], f' stroke="{t["lap_edge"]}" stroke-width="1"'),
        rect(sx + ins, y + ins, sw - 2 * ins, sh - 2 * ins, w * 0.02, t["lap_screen"]),
        rect(sx + sw / 2 - nw / 2, y + ins - 1, nw, nh + 1, nh * 0.5, t["lap_body"]),
        f'<polygon points="{" ".join(f"{num(a)},{num(b)}" for a, b in pts)}" fill="{t["lap_edge"]}"/>',
        rect(sx - w * .02, by, sw + w * .04, h * .06, 0, t["lap_edge"]),
        rect(x + w / 2 - w * .08, by, w * .16, h * .035, h * .02, t["lap_body"])])


def logo(e, x, y, size, t, i):
    r = size / 4
    src = e.get("logo")
    if src == "kapitsa":                    # the mark on a dark tile with a green edge, in both themes
        w = size * 0.78
        return (rect(x, y, size, size, r, "#0F1A13") + mark(x + (size - w) / 2, y + (size - w * 104 / 144) / 2, w,
                                                            POSTER["dark"]["mark"], f"mk{i}")
                + rect(x + 1.25, y + 1.25, size - 2.5, size - 2.5, r - 1.25, "none",
                       f' stroke="{t["kap"]}" stroke-width="2.5"'))
    if src:
        href = (ASSETS / src).as_uri()
        if e.get("tile"):                   # the logo inside a colored tile
            w = size * 0.74
            return (rect(x, y, size, size, r, e["tile"]) + f'<image href="{href}" x="{num(x + (size - w) / 2)}" '
                    f'y="{num(y + (size - w) / 2)}" width="{num(w)}" height="{num(w)}" preserveAspectRatio="xMidYMid meet"/>')
        return (f'<clipPath id="tile{i}"><rect x="{num(x)}" y="{num(y)}" width="{size}" height="{size}" rx="{num(r)}"/>'
                f'</clipPath><image href="{href}" x="{num(x)}" y="{num(y)}" width="{size}" height="{size}" '
                f'preserveAspectRatio="xMidYMid slice" clip-path="url(#tile{i})"/>')
    return (rect(x, y, size, size, r, t["badge"])   # no logo: its initial
            + text(x + size / 2, y + size / 2 + 0.352 * size * 0.41, e["name"][0].upper(), "sans-b", num(size * 0.41),
                   t["ink2"], "middle"))


def poster(data, theme):
    """the poster as an HTML page holding one SVG; returns (html, width, height, notes)"""
    t = POSTER[theme]
    P = data["poster"]
    W, H = P["size"]
    engines = data["engines"]
    names = [e["name"] for e in engines]
    under = header_methods(data)
    combined = any(isinstance(c, dict) and c.get('prefill') for m in P['macs'] for r in m['rows'] for c in r['cells'].values())
    kw = KW_PRE if combined else KW
    OW = (W - 44 - X0 - LW - kw) / (len(engines) - 1)
    colx = [(X0 + LW, kw)] + [(X0 + LW + kw + j * OW, OW) for j in range(len(engines) - 1)]
    notes = []

    rows = [r for m in P["macs"] for r in m["rows"]]
    beaten = any(is_num(cell(r, "Kapitsa")[0]) and any(is_num(cell(r, e)[0]) and cell(r, e)[0] > cell(r, "Kapitsa")[0]
                                                         for e in names[1:]) for r in rows)
    tied = any(r.get("tie") for r in rows)  # engines whose decode is level with Kapitsa's, inside the runs' spread
    footer = list(P["footer"]) + (["Blue outline: faster decode than Kapitsa · orange bar: rows where Kapitsa isn't alone at the fastest decode"
                                   + (" · ≈: a decode tie with Kapitsa (the gap is inside the runs' spread)" if tied else "")]
                                  if beaten or tied else [])
    foot_h = FOOT_GAP + len(footer) * FOOT_LINE + FOOT_PAD
    chrome = PANELS_Y + len(P["macs"]) * (HEAD + PAD) + (len(P["macs"]) - 1) * GAP + foot_h
    rh = min(RH_MAX, (H - chrome) // len(rows))
    if rh < RH_MIN:
        rh = RH_MIN
        notes.append(f"{len(rows)} rows don't fit {W} x {H} at {RH_MIN} px each: the poster grows")
    H = max(H, chrome + rh * len(rows))
    slack = H - chrome - rh * len(rows)       # leftover pixels go above the footer

    o = [rect(0, 0, W, H, 0, t["bg"])]
    o.append(mark(44, 43, 62, t["mark"], "mk"))
    o.append(text(122, 84, P["title"], "serif", 52, t["ink"]))
    o.append(text(44, 128, P["legend"], "sans-b", 29, t["ink"]))      # bold: what every cell reads
    for i, s in enumerate(P["subtitle"]):
        o.append(text(44, 162 + 28 * i, s, "sans", 22, t["ink2"]))

    for j, e in enumerate(engines):           # one header for every Mac: logo, name, method if it's the same everywhere
        cx, cw = colx[j]
        o.append(logo(e, cx + cw / 2 - TILE / 2, TILE_Y, TILE, t, j))
        if j == 0:
            o.append(rect(cx + 6, 281, cw - 12, 40, 10, t["kap"]))
            o.append(text(cx + cw / 2, 309, e["name"], "sans-b", 23, "#FFFFFF", "middle"))
        else:
            o.append(text(cx + cw / 2, 309, e["name"], "sans-m", 22, t["muted"], "middle"))
        if under[e["name"]]:
            o.append(text(cx + cw / 2, 339, under[e["name"]], "sans-m", F_METH, t["muted"], "middle"))
        elif j and e.get("version"):          # the competitor's version, one or two lines, under its name
            for i, line in enumerate(e["version"].split("\n")):
                o.append(text(cx + cw / 2, 336 + 19 * i, line, "sans", 17, t["muted"], "middle"))

    y = PANELS_Y
    for mi, mac in enumerate(P["macs"]):
        ph = HEAD + len(mac["rows"]) * rh + PAD
        o.append(rect(24, y, W - 48, ph, 16, t["panel"]))
        o.append(laptop(44, y + 20, 52, t))
        o.append(text(112, y + 45, mac["name"], "sans-b", 31, t["ink"]))
        ry = y + HEAD
        for row in mac["rows"]:
            cy = ry + rh / 2
            vals = [cell(row, e)[0] for e in names]
            nums = [v for v in vals if is_num(v)]
            top = max(nums + [1e-9])
            k = vals[0] if is_num(vals[0]) else None
            two = any(is_num(v) and under[e] is None and cell(row, e)[1] for e, v in zip(names, vals))
            vb = cy - (CAP * F_KVAL + M_GAP + CAP * F_METH) / 2 + CAP * F_KVAL if two else cy + CAP / 2 * F_VAL
            mb = vb + M_GAP + CAP * F_METH
            tie = set(row.get("tie", []))
            if k is not None and (k < top or tie):
                o.append(rect(30, ry + 14, 7, rh - 28, 3, ORANGE))
            o.append(text(X0 + 10, cy + CAP / 2 * F_MODEL, row["model"], "sans", F_MODEL, t["ink"]))
            for j, (e, v) in enumerate(zip(names, vals)):
                cx, cw = colx[j]
                mid = cx + cw / 2
                meth = cell(row, e)[1] if under[e] is None else None
                if not is_num(v):
                    o.append(text(mid, vb - CAP / 2 * F_VAL + 0.286 * F_DASH, CROSS if v == "x" else DASH, "sans", F_DASH,
                                  t["dash"], "middle"))
                    continue
                lvl = min(5, int(v / top * 5.999))
                if j == 0:
                    o.append(rect(cx + 3, ry + 5, cw - 6, rh - 10, 10, t["greens"][max(lvl, 4) if theme == "light" else lvl]))
                    pre = cell(row, e)[2].get("prefill") if isinstance(cell(row, e)[2], dict) else None
                    if combined:              # one value, prefill first, both the same size and weight
                        head = f"{fmt(pre)} / " if pre else f"{DASH} / "
                        o.append(f'<text x="{num(mid)}" y="{num(vb)}" text-anchor="middle" fill="#FFFFFF">'
                                 f'<tspan class="mono-b" font-size="{F_COMB}">{head}{fmt(v)}</tspan></text>')
                        if meth:
                            o += method_label(mid, mb, meth, "sans-sb", F_KMETH, t["unit"])
                        continue
                    o.append(f'<text x="{num(mid)}" y="{num(vb)}" text-anchor="middle" fill="#FFFFFF">'
                             f'<tspan class="mono-b" font-size="{F_KVAL}">{fmt(v)}</tspan>'
                             f'<tspan class="sans-sb" font-size="{F_UNIT}" dx="7" fill="{t["unit"]}">tok/s</tspan></text>')
                    if meth:
                        o += method_label(mid, mb, meth, "sans-sb", F_KMETH, t["unit"])
                    continue
                best = v == top or e in tie
                o.append(rect(cx + 3, ry + 5, cw - 6, rh - 10, 10, t["greys"][lvl]))
                tv = f"≈{fmt(v)}" if e in tie else fmt(v)
                ink = t["ink"] if best else (t["ink2"] if theme == "light" else t["muted"])
                if combined:                  # the same 'prefill / decode', the prefill from the engine's own run
                    cr = cell(row, e)[2]
                    pre = cr.get("prefill") if isinstance(cr, dict) else None
                    o.append(text(mid, vb, f"{fmt(pre) if pre else DASH} / {tv}", "mono-b" if best else "mono",
                                  F_CCOMB, ink, "middle"))
                else:
                    o.append(text(mid, vb, tv, "mono-b" if best else "mono", F_VAL, ink, "middle"))
                if meth:
                    o += method_label(mid, mb, meth, "sans-m", F_METH, t["method"])
                if k is not None and v > k:
                    o.append(rect(cx + 4.75, ry + 6.75, cw - 9.5, rh - 13.5, 8.5, "none",
                                  f' stroke="{t["blue"]}" stroke-width="3.5"'))
            ry += rh
        y += ph + (GAP if mi < len(P["macs"]) - 1 else 0)
    fy = y + FOOT_GAP + slack
    for i, s in enumerate(footer):
        o.append(text(44, fy + 16 + i * FOOT_LINE, s, "sans", 22, t["ink2"]))

    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
           + "".join(o) + "</svg>")
    html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="light">'
            f'<style>{CSS % {"f": FONTS, "bg": t["bg"]}}</style></head><body>{svg}</body></html>')
    return html, W, H, notes


def render(html, w, h, out_webp, tmp):
    """HTML -> PNG with headless Chrome on the CPU (no GPU), -> lossy WebP at quality 90"""
    page, png = tmp / f"{out_webp.stem}.html", tmp / f"{out_webp.stem}.png"
    page.write_text(html, encoding="utf-8")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                    "--no-default-browser-check", "--force-device-scale-factor=1", f"--user-data-dir={tmp / 'chrome'}",
                    f"--window-size={w},{h}", f"--screenshot={png}", page.as_uri()],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([CWEBP, "-quiet", "-q", "90", str(png), "-o", str(out_webp)], check=True)


# ---------------------------------------------------------------- the header (SVG, text as Geist Mono outlines)

HEADER = {
    "dark": dict(bg="#0B0F0D", card="#0B0F0D", edge="#1F2924", hi="#9DF0B0", main="#3FB55A", dim="#1E5A30",
                 text="#E8EEE9", muted="#8A9690", bar="#29332E", rule="#1F2924", method="#76827B", kmethod="#3FB55A"),
    "light": dict(bg="#FFFFFF", card="#F6F9F7", edge="#E1E8E3", hi="#1E5A30", main="#3FB55A", dim="#B6E8C3",
                  text="#0B0F0D", muted="#56615B", bar="#D9E0DB", rule="#E1E8E3", method="#6B766F", kmethod="#2A7A3C"),
}
# the bars: wide (1200 x 400) keeps the block's left edge on the caption's and fills to the right; narrow
# (800 x 816, phones) centers it. fs: names and numbers, mfs: methods, cy: the block's vertical center.
# mgap: a name's baseline to the top of the method under it, clear of the name's descenders.
WIDE = dict(left=655, right=1121.2, cy=168, fs=19, mfs=14, lgap=17, vgap=12, bh=18, pitch=39, rx=5, mgap=6, div=3.12,
            caption=(334, 358))
NARROW = dict(cx=400, maxbar=330, cy=503, fs=27, mfs=21, lgap=24, vgap=14, bh=26, pitch=56, rx=7, mgap=8, div=4.56,
              caption=(736, 769))


class Outlines:
    """text set in Geist Mono as <use>s of glyph outlines defined once per SVG (ids in order of first use)"""

    def __init__(self, table):
        self.t, self.ids, self.defs = table, {}, []

    def text(self, s, weight, x, y, size, paint):
        uses = []
        for i, ch in enumerate(s):
            if ch == " ":
                continue
            key = (weight, ch)
            if key not in self.ids:
                if ch not in self.t[weight]:
                    sys.exit(f"{ASSETS / 'geist-mono.json'} has no {weight} outline for {ch!r}")
                self.ids[key] = f"g{weight[0]}{len(self.ids):x}"
                self.defs.append(f'<path id="{self.ids[key]}" d="{self.t[weight][ch]}"/>')
            uses.append(f'<use href="#{self.ids[key]}"' + (f' x="{600 * i}"' if i else "") + "/>")
        k = num(size / 1000, 4)
        return f'<g transform="translate({num(x)} {num(y)}) scale({k} -{k})" {paint}>' + "".join(uses) + "</g>"


def header_rows(data):
    row, eng = header_row(data), {e["name"]: e for e in data["engines"]}
    out = []
    for e in data["header"]["engines"]:
        v, m, _ = cell(row, e)
        out.append(dict(name=e, value=v if is_num(v) else None, method=m or eng[e].get("method"), kap=e == "Kapitsa"))
    return out


def bars(g, rows, geo, p):
    adv, madv = 0.6 * geo["fs"], 0.6 * geo["mfs"]
    vmax = max(r["value"] for r in rows if r["value"] is not None)
    lw = max(max(len(r["name"]) * adv, len(r["method"] or "") * madv) for r in rows)
    vw = len(fmt(vmax)) * adv
    if "left" in geo:
        bx = geo["left"] + lw + geo["lgap"]
        maxbar = geo["right"] - geo["vgap"] - vw - bx
    else:
        maxbar = geo["maxbar"]
        bx = geo["cx"] - (lw + geo["lgap"] + maxbar + geo["vgap"] + vw) / 2 + lw + geo["lgap"]
    n, bh, pitch = len(rows), geo["bh"], geo["pitch"]
    y0 = geo["cy"] - ((n - 1) * pitch + bh) / 2
    cap, mcap = 0.72 * geo["fs"], 0.72 * geo["mfs"]
    out = []
    for i, r in enumerate(rows):
        yb = y0 + i * pitch
        c = yb + bh / 2
        right = bx - geo["lgap"]
        if r["value"] is not None:
            w = maxbar * r["value"] / vmax
            out.append(f'<rect x="{num(bx)}" y="{num(yb)}" width="{num(w)}" height="{bh}" rx="{geo["rx"]}" '
                       f'{p["kbar"] if r["kap"] else p["bar"]}/>')
            vx = bx + w + geo["vgap"]
        else:
            vx = bx + geo["vgap"]
        if r["method"]:
            nb = c - (cap + geo["mgap"] + mcap) / 2 + cap
            out.append(g.text(r["name"], "semibold" if r["kap"] else "regular", right - len(r["name"]) * adv, nb,
                              geo["fs"], p["text"] if r["kap"] else p["muted"]))
            out.append(g.text(r["method"], "regular", right - len(r["method"]) * madv, nb + geo["mgap"] + mcap,
                              geo["mfs"], p["kmethod"] if r["kap"] else p["method"]))
        else:
            out.append(g.text(r["name"], "semibold" if r["kap"] else "regular", right - len(r["name"]) * adv,
                              c + 0.36 * geo["fs"], geo["fs"], p["text"] if r["kap"] else p["muted"]))
        out.append(g.text(fmt(r["value"]), "semibold" if r["kap"] else "regular", vx, c + 0.36 * geo["fs"], geo["fs"],
                          p["hi"] if r["kap"] else p["muted"]))
    out.append(f'<path d="M{num(bx - 0.5)} {num(y0 - geo["div"])}V{num(y0 + (n - 1) * pitch + bh + geo["div"])}" '
               f'{p["rule"]} stroke-width="1.5"/>')
    return out


def header_desc(data, rows):
    parts = []
    for r in rows:
        how = f" with {r['method']}" if r["method"] and r["method"] != "none" else ""
        parts.append(f"{r['name']}{how} {fmt(r['value'])}" if r["value"] is not None
                     else f"{r['name']}{how} not measured yet")
    return data["header"]["desc"] + ": " + ", ".join(parts) + "."


def svg_open(w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
            f'aria-labelledby="t d"><title id="t">{escape(title)}</title><desc id="d">{escape(desc)}</desc>')


def header_wide(data, glyphs, theme):
    t, h, rows = HEADER[theme], data["header"], header_rows(data)
    g = Outlines(glyphs)
    P = lambda c: f'fill="{c}"'
    p = dict(bar=P(t["bar"]), kbar='fill="url(#hwk)"', text=P(t["text"]), muted=P(t["muted"]), hi=P(t["hi"]),
             method=P(t["method"]), kmethod=P(t["kmethod"]), rule=f'stroke="{t["rule"]}"')
    body = [f'<rect x="1" y="1" width="1198" height="398" rx="26" fill="{t["card"]}" stroke="{t["edge"]}" stroke-width="2"/>',
            '<g transform="translate(72 106) scale(0.82)" fill="none">'
            + mark_body(lambda k: f'fill="{t[k]}"', lambda k: f'stroke="{t[k]}"', "hwin") + "</g>",
            g.text("kapitsa", "semibold", 210.8, 171.03, 90, P(t["text"])),
            g.text(h["tagline"][0], "regular", 74, 246, 21, P(t["muted"])),
            g.text(h["tagline"][1], "regular", 74, 279, 21, P(t["muted"]))]
    body += bars(g, rows, WIDE, p)
    body += [g.text(s, "regular", WIDE["left"], y, 16, P(t["muted"])) for s, y in zip(h["caption"], WIDE["caption"])]
    defs = (f'<linearGradient id="hwin" x1="0" x2="46.25" y1="0" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" '
            f'stop-color="{t["hi"]}" stop-opacity="0"/><stop offset="1" stop-color="{t["hi"]}"/></linearGradient>'
            f'<linearGradient id="hwk" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="{t["main"]}"/>'
            f'<stop offset="1" stop-color="{t["hi"]}"/></linearGradient>' + "".join(g.defs))
    return svg_open(1200, 400, h["title"], header_desc(data, rows)) + f"<defs>{defs}</defs>" + "".join(body) + "</svg>\n"


def header_narrow(data, glyphs):
    """one file for both themes: GitHub drops <source media> that names a color scheme, so the colors follow
    prefers-color-scheme inside the SVG"""
    h, rows = data["header"], header_rows(data)
    g = Outlines(glyphs)
    keys = ["bg", "card", "edge", "hi", "main", "dim", "text", "muted", "bar", "rule", "method", "kmethod"]

    def classes(t):
        return "".join(f".f-{k}{{fill:{t[k]}}}.s-{k}{{stroke:{t[k]}}}.c-{k}{{stop-color:{t[k]}}}" for k in keys)

    style = f'<style>{classes(HEADER["light"])}@media (prefers-color-scheme:dark){{{classes(HEADER["dark"])}}}</style>'
    C = lambda k: f'class="f-{k}"'
    p = dict(bar=C("bar"), kbar='fill="url(#hnk)"', text=C("text"), muted=C("muted"), hi=C("hi"), method=C("method"),
             kmethod=C("kmethod"), rule='class="s-rule"')
    mid = lambda s, size: 400 - len(s) * 0.6 * size / 2
    body = ['<rect x="1" y="1" width="798" height="814" rx="30" class="f-card s-edge" stroke-width="2"/>',
            '<g transform="translate(112.6 70) scale(0.92)" fill="none">'
            + mark_body(lambda k: f'class="f-{k}"', lambda k: f'class="s-{k}"', "hnin") + "</g>",
            g.text("kapitsa", "semibold", 267.4, 142.7, 100, C("text")),
            g.text(h["tagline"][0], "regular", mid(h["tagline"][0], 27), 236, 27, C("muted")),
            g.text(h["tagline"][1], "regular", mid(h["tagline"][1], 27), 277, 27, C("muted"))]
    body += bars(g, rows, NARROW, p)
    body += [g.text(s, "regular", mid(s, 23), y, 23, C("muted")) for s, y in zip(h["caption"], NARROW["caption"])]
    defs = ('<linearGradient id="hnin" x1="0" x2="46.25" y1="0" y2="0" gradientUnits="userSpaceOnUse"><stop offset="0" '
            'class="c-hi" stop-opacity="0"/><stop offset="1" class="c-hi"/></linearGradient><linearGradient id="hnk" '
            'x1="0" x2="1" y1="0" y2="0"><stop offset="0" class="c-main"/><stop offset="1" class="c-hi"/></linearGradient>'
            + "".join(g.defs))
    return (svg_open(800, 816, h["title"], header_desc(data, rows)) + style + f"<defs>{defs}</defs>" + "".join(body)
            + "</svg>\n")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Render the README's benchmark poster and header from readme_images.json")
    ap.add_argument("--poster", action="store_true", help="only the poster (two WebP files)")
    ap.add_argument("--headers", action="store_true", help="only the header (three SVG files; no Chrome needed)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "assets", help="output folder (default: docs/assets)")
    ap.add_argument("--tables", action="store_true", help="only the README's per-Mac tables (rewrites README.md's Models section)")
    ap.add_argument("--check", action="store_true", help="with --tables: fail if README.md's tables are not what the data gives")
    ap.add_argument("--readme", type=Path, default=ROOT / "README.md")
    ap.add_argument("--data", type=Path, default=HERE / "readme_images.json")
    a = ap.parse_args()
    data = json.loads(a.data.read_text(encoding="utf-8"))
    check(data)
    if a.tables:
        old = a.readme.read_text(encoding="utf-8")
        new = readme_with_tables(old, tables(data))
        if a.check:
            if new != old:
                sys.exit(f"{a.readme}: the Models tables differ from readme_images.json; run python3 "
                         "tools/readme_images/readme_images.py --tables")
            print(f"{a.readme}: the Models tables match readme_images.json")
        elif new != old:
            a.readme.write_text(new, encoding="utf-8")
            print("wrote", a.readme)
        return
    a.out.mkdir(parents=True, exist_ok=True)
    written = []
    if not a.poster:
        glyphs = json.loads((ASSETS / "geist-mono.json").read_text(encoding="utf-8"))
        for theme in ("light", "dark"):
            f = a.out / f"header-{theme}.svg"
            f.write_text(header_wide(data, glyphs, theme), encoding="utf-8")
            written.append(f)
        f = a.out / "header-narrow.svg"
        f.write_text(header_narrow(data, glyphs), encoding="utf-8")
        written.append(f)
    if not a.headers:
        if not Path(CHROME).exists() or not shutil.which(CWEBP):
            sys.exit("the poster needs Google Chrome (set CHROME=...) and cwebp (brew install webp; or set CWEBP=...)")
        tmp = Path(tempfile.mkdtemp(prefix="readme-images-"))
        try:
            for theme in ("light", "dark"):
                html, w, h, notes = poster(data, theme)
                for n in notes:
                    print("note:", n)
                f = a.out / f"benchmarks-{theme}.webp"
                render(html, w, h, f, tmp)
                written.append(f)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    for f in written:
        print("wrote", f)
    for mac in data["poster"]["macs"]:
        for row in mac["rows"]:
            for e in row["cells"]:
                raw = cell(row, e)[2]
                if isinstance(raw, dict) and raw.get("todo"):
                    print(f"todo: {mac['name']} / {row['model']} / {e}: {raw['todo']}")
    rows = header_rows(data)
    print("README header alt text:", data["header"]["title"] + ". " + header_desc(data, rows[::-1]))
    words = "zero one two three four five six seven eight nine ten eleven twelve".split()
    others = len(data["engines"]) - 1
    chips = [m["name"].removeprefix("MacBook Pro ").rsplit(" ", 2)[0] for m in data["poster"]["macs"]]
    print("README poster alt text: Poster of prefill / decode speed for Kapitsa and "
          f"{words[others] if others < len(words) else others} other engines on a MacBook Pro "
          + ", ".join(chips[:-1]) + " and " + chips[-1]
          + "; the numbers are in the poster and in the Models section's tables")


if __name__ == "__main__":
    main()
