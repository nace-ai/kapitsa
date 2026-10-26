# Markdown tables of a sweep directory (sweep.sh's raw outputs): prefill and decode tok/s per context length
import glob, os, re, sys
d = sys.argv[1]
lens = sorted({int(re.search(r"-(\d+)\.txt$", f).group(1)) for f in glob.glob(os.path.join(d, "*-*.txt"))})
def rd(p):
    try: return open(p).read()
    except OSError: return ""
def kap(n):
    m = re.search(r"\((\d+) tok/s, GPU.*?\(([\d.]+) tok/s", rd(f"{d}/kapitsa-{n}.txt"))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
def mlx(n):
    m = re.search(r"Averages: prompt_tps=([\d.]+), generation_tps=([\d.]+)", rd(f"{d}/mlxlm-{n}.txt"))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)
def llc(n):
    a = re.search(r"pp\d+ \|\s+([\d.]+)", rd(f"{d}/llamacpp-pp-{n}.txt")); b = re.search(r"tg\d+ @ d\d+ \|\s+([\d.]+)", rd(f"{d}/llamacpp-tg-{n}.txt"))
    return (float(a.group(1)) if a else None, float(b.group(1)) if b else None)
f = lambda v, p=0: "–" if v is None else f"{v:.{p}f}"
lab = lambda n: f"{n // 1024}K"
for title, idx, p in (("Prefill (tokens/s)", 0, 0), ("Decode after the prompt (tokens/s)", 1, 1)):
    print(f"\n**{title}**\n\n| Context | Kapitsa | mlx-lm | llama.cpp |\n| ---: | ---: | ---: | ---: |")
    for n in lens:
        k, m, l = kap(n)[idx], mlx(n)[idx], llc(n)[idx]
        best = max(v for v in (k, m, l) if v is not None) if any(v is not None for v in (k, m, l)) else None
        cell = lambda v: ("**" + f(v, p) + "**") if v is not None and v == best else f(v, p)
        print(f"| {lab(n)} | {cell(k)} | {cell(m)} | {cell(l)} |")
