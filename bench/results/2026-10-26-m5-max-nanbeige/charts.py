# Reddit charts: Nanbeige4.3-33B prefill and decode speed from 512 to 32K tokens (flint_bench / mlx_lm.benchmark /
# llama-bench sweep), plus time to first token cold and on a follow-up turn (the harness at 7 sizes) when given.
# usage: uv run --with matplotlib python charts.py SWEEP_DIR OUT_DIR [HARNESS_RESULTS_DIR]
import glob, json, os, re, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

sweep, out = sys.argv[1], sys.argv[2]
harness = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)

# the reference palette's first three categorical slots (validated: CVD 9.2, normal 27.6; aqua under 3:1 on the
# surface, so every line is direct-labeled and the post carries the table)
ENG = [("Kapitsa 0.2.19", "#2a78d6"), ("mlx-lm 0.32.2", "#eb6834"), ("llama.cpp b11871", "#1baf7a")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f2328", "#5b6672", "#e6e8eb"

def rd(p):
    try:
        return open(p).read()
    except OSError:
        return ""

lens = sorted({int(re.search(r"-(\d+)\.txt$", f).group(1)) for f in glob.glob(os.path.join(sweep, "kapitsa-*.txt"))})
def kap(n):
    m = re.search(r"\((\d+) tok/s, GPU.*?\(([\d.]+) tok/s", rd(f"{sweep}/kapitsa-{n}.txt"))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
def mlx(n):
    m = re.search(r"Averages: prompt_tps=([\d.]+), generation_tps=([\d.]+)", rd(f"{sweep}/mlxlm-{n}.txt"))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
def llc(n):
    a = re.search(r"pp\d+ \|\s+([\d.]+)", rd(f"{sweep}/llamacpp-pp-{n}.txt"))
    b = re.search(r"tg\d+ @ d\d+ \|\s+([\d.]+)", rd(f"{sweep}/llamacpp-tg-{n}.txt"))
    return (float(a.group(1)) if a else None, float(b.group(1)) if b else None)
data = {e: [f(n) for n in lens] for (e, _), f in zip(ENG, (kap, mlx, llc))}

def lab(n):
    if n % 1024 == 0:
        return str(n) if n < 1024 else f"{n // 1024}K"
    return str(n) if n < 1000 else f"{n / 1000:.1f}K".replace(".0K", "K")

def style(ax, title, sub, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=12, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=12)
    ax.set_xlabel("prompt length (tokens)", color=INK2, fontsize=12)
    ax.set_title(title, loc="left", color=INK, fontsize=17, fontweight="bold", pad=34)
    ax.text(0, 1.03, sub, transform=ax.transAxes, color=INK2, fontsize=11.5)

def line_chart(xs, series, title, sub, ylabel, fname, fmt, ymin0=True):
    fig, ax = plt.subplots(figsize=(11, 6.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax, title, sub, ylabel)
    ends = []
    for name, color, ys in series:
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=2.2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, solid_capstyle="round")
        ends.append([pts[-1][1], name, color, pts[-1][0]])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab(x) for x in xs])
    ax.minorticks_off()
    if ymin0:
        top = max(y for _, _, ys in series for y in ys if y is not None)
        import math
        step = 10 ** math.floor(math.log10(top)) / 2
        ax.set_ylim(0, math.ceil(top * 1.08 / step) * step)
    lo, hi = ax.get_ylim()
    # direct labels at the right end, nudged apart so they never collide; text in ink, a colored dot beside it
    ends.sort(key=lambda e: e[0])
    gap = (hi - lo) * 0.055
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < gap:
            ends[i][0] = ends[i - 1][0] + gap
    for y, name, color, x in ends:
        ax.annotate(name, xy=(x, y), xytext=(14, 0), textcoords="offset points", va="center", fontsize=12, color=INK,
                    annotation_clip=False)
        ax.plot([x * 1.07], [y], marker="o", markersize=7, color=color, clip_on=False, zorder=4)
    ax.set_xlim(xs[0] / 1.25, xs[-1] * 1.25)
    fig.subplots_adjust(left=0.08, right=0.80, top=0.84, bottom=0.14)
    fig.text(0.08, 0.018, "Nanbeige4.3-33B, 4-bit (MLX AWQ pack; llama.cpp: Q4_K_M) · MacBook Pro M5 Max 128 GB · macOS 26.4",
             color=INK2, fontsize=10)
    fig.savefig(os.path.join(out, fname), facecolor=SURFACE)
    plt.close(fig)
    print("wrote", fname)

sub = "each engine's own benchmark tool (Kapitsa's, mlx_lm.benchmark, llama-bench), run in turn at each length"
line_chart(lens, [(e, c, [v[0] for v in data[e]]) for e, c in ENG],
           "Prefill speed vs context length", sub, "prompt tokens per second",
           "prefill.png", "{:.0f}")
line_chart(lens, [(e, c, [v[1] for v in data[e]]) for e, c in ENG],
           "Decode speed vs context length (no speculative decoding)", sub + " (64 tokens after the prompt)",
           "generated tokens per second", "decode.png", "{:.1f}")

if harness:
    recs = {}
    for f in glob.glob(os.path.join(harness, "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if not isinstance(j, dict) or "prefill" not in j:
            continue
        eng = {"kapitsa": "Kapitsa 0.2.19", "mlx-lm": "mlx-lm 0.32.2", "llama.cpp": "llama.cpp b11871"}.get(j.get("engine"), j.get("engine"))
        recs[eng] = j
    sizes = sorted({r["target"] for j in recs.values() for r in j["prefill"]})
    def ttft(j, n):
        p = next((r for r in j["prefill"] if r["target"] == n), {})
        return p.get("ttft_s"), p.get("turn2_ttft_s"), p.get("prompt_tokens")
    toks = [next((ttft(recs[e], n)[2] for e in recs if ttft(recs[e], n)[2]), n) for n in sizes]
    if sizes:
        line_chart(toks, [(e, c, [ttft(recs[e], n)[0] if e in recs else None for n in sizes]) for e, c in ENG],
                   "Time to first token, cold prompt", "each engine's own server, real chat prompts, thinking off",
                   "seconds", "ttft_cold.png", "{:.2f}")
        line_chart(toks, [(e, c, [ttft(recs[e], n)[1] if e in recs else None for n in sizes]) for e, c in ENG],
                   "Time to first token on a follow-up turn", "the conversation so far cached, a new user turn added",
                   "seconds", "ttft_followup.png", "{:.2f}")
        # the server process's peak physical footprint by the end of each prompt size (the harness runs them in
        # ascending order after decode, so each point is the high-water mark up to that context)
        def mem(j, n):
            v = (j.get("memory") or {}).get(f"prefill_{n}") or {}
            return v["phys_footprint_peak_mb"] / 1024 if "phys_footprint_peak_mb" in v else None
        line_chart(toks, [(e, c, [mem(recs[e], n) if e in recs else None for n in sizes]) for e, c in ENG],
                   "Peak memory as the context grows", "the server process's peak physical footprint, weights included",
                   "GB", "memory.png", "{:.1f}")
    json.dump({e: {str(n): ttft(recs[e], n) for n in sizes} for e in recs}, open(os.path.join(out, "harness.json"), "w"))
json.dump({e: dict(zip(map(str, lens), data[e])) for e in data}, open(os.path.join(out, "sweep.json"), "w"))
