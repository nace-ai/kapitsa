# Decode speed writing code: Kapitsa with KFlash, mlx-lm and llama.cpp (no speculation).
# usage: uv run --with matplotlib python chart_write.py results.jsonl OUT.png
import json, sys
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{") and json.loads(l)["task"] == "write code"]
ENG = [("kapitsa-auto", "Kapitsa 0.2.19 (KFlash)", "#2a78d6"),
       ("mlx-lm", "mlx-lm 0.32.2 (no spec)", "#eb6834"), ("llama.cpp", "llama.cpp b11871 (no spec)", "#1baf7a")]
TS = [(0.0, "greedy (T 0)"), (0.6, "sampled (T 0.6)")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f2328", "#5b6672", "#e6e8eb"

def val(e, T):
    return mean(r["tps"] for r in rows if r["engine"] == e and r["T"] == T)

fig, ax = plt.subplots(figsize=(11, 6.4), dpi=160)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=12, length=0)
ax.grid(axis="y", color=GRID, linewidth=1)
ax.set_axisbelow(True)
W, top = 0.25, max(r["tps"] for r in rows)
for i, (e, name, color) in enumerate(ENG):
    for j, (T, _) in enumerate(TS):
        v, x = val(e, T), j + (i - 1) * W
        ax.bar(x, v, W * 0.9, color=color, edgecolor=SURFACE, linewidth=1, zorder=3, label=name if j == 0 else None)
        ax.text(x, v + top * 0.012, f"{v:.1f}", ha="center", va="bottom", fontsize=11, color=INK)
ax.set_xticks(range(len(TS)))
ax.set_xticklabels([t for _, t in TS])
ax.set_ylim(0, top * 1.12)
ax.set_ylabel("generated tokens per second", color=INK2, fontsize=12)
fig.suptitle("Writing code with speculative decoding", x=0.08, ha="left", color=INK, fontsize=17,
             fontweight="bold", y=0.97)
fig.text(0.08, 0.885, "a 512-token Python function with unit tests, each engine's own server, mean of two runs",
         color=INK2, fontsize=11.5)
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=11, labelcolor=INK,
           bbox_to_anchor=(0.5, 0.05))
fig.text(0.08, 0.015, "Nanbeige4.3-33B, 4-bit (MLX AWQ pack; llama.cpp: Q4_K_M) · MacBook Pro M5 Max 128 GB · macOS 26.4",
         color=INK2, fontsize=10)
fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.24)
fig.savefig(sys.argv[2], facecolor=SURFACE)
