#!/usr/bin/env python3
"""Per-model comparison tables (markdown + CSV) from bench.py / compare_engines.py results. Stdlib only.

One table per model; one row per engine configuration (config label), from its latest clean run (no INVALID
reason, no contention flag), else its latest valid run, else its latest run (marked). Every number is the
harness's own:
  decode        median middle-window tok/s over the sampled runs (T=0.6, top_p 0.95, top_k 20), min-max
  greedy        same prompt at T=0
  prefill       prompt_tokens / TTFT on a cold, cache-busted prompt (~4.1k and ~16.3k tokens)
  cached TTFT   median TTFT of 3 append-only follow-up turns on that prefix
  peak memory   max phys_footprint_peak of the server process tree (+ resident mmap'd files, shown
                separately when > 1 GB: llama.cpp/Ollama GGUF weights mapped from disk)
Best value per column is bold; Kapitsa rows are marked. Results of the old harness (schema 1: max-of-2 decode,
no min_p/presence parity, no thinking guard) are left out unless --include-legacy.

usage: compare_table.py [results dir ...] [--out FILE_WITHOUT_EXT] [--include-legacy]
       writes <out>.md and <out>.csv, prints the markdown (default: the first results dir/compare_table)
"""
import argparse, collections, csv, glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

MODEL_ORDER = ["Qwen3.6-35B-A3B", "Qwen3.8-27B", "Qwen3.5-4B"]
OURS = ("kapitsa", "nace-infer")  # nace-infer: Kapitsa's former name, in older results

ENGINES = [  # fallback when a result has no "display" (label regex, engine name)
    (r"kapitsa|naceinfer", "Kapitsa"),
    (r"ollama-mlx", "Ollama (MLX runner)"),
    (r"ollama-nomtp", "Ollama (MTP off)"),
    (r"ollama", "Ollama"),
    (r"llamacpp-mtp|llamamtp", "llama.cpp + MTP"),
    (r"llamacpp|llama_", "llama.cpp"),
    (r"omlx", "oMLX"),
    (r"mtplx", "MTPLX"),
    (r"mlxlm", "mlx-lm"),
    (r"lmstudio", "LM Studio"),
]


def model_of(label):
    l = label.lower()
    if "35b" in l:
        return "Qwen3.6-35B-A3B"
    if "27b" in l:
        return "Qwen3.8-27B"
    if "4b" in l:
        return "Qwen3.5-4B"
    return None


def engine_of(label):
    for rx, name in ENGINES:
        if re.search(rx, label):
            return name
    return label


def _median(v):
    v = [x for x in v if x]
    return statistics.median(v) if v else None


def hardware(d):
    s = (d.get("provenance") or {}).get("system_start") or {}
    cpu, mem = s.get("cpu"), s.get("memsize")
    if not cpu:
        return None
    gb = f", {int(mem) / 2**30:.0f} GB" if str(mem or "").isdigit() else ""
    os_ = (s.get("sw_vers") or {}).get("ProductVersion")
    return f"{cpu}{gb}" + (f", macOS {os_}" if os_ else "")


def extract(d, path):
    label = d.get("label", os.path.basename(path))
    schema = d.get("schema", 1)
    pf = {p.get("target"): p for p in d.get("prefill", [])}

    def pref(size, key):
        p = pf.get(size) or {}
        if key == "cached":
            c = p.get("cached") or {}
            v = c.get("median_ttft_s", p.get("turn2_ttft_s"))
            bad = bool(c.get("invalid"))
        else:
            v = p.get(key)
            bad = bool(p.get("invalid"))
        return v, bad

    if schema >= 2:
        ds = d.get("decode_summary") or {}
        gs = d.get("decode_greedy_summary") or {}
        dec, dmin, dmax, dbad = ds.get("median_tps"), ds.get("min_tps"), ds.get("max_tps"), bool(ds.get("invalid"))
        gre, gmin, gmax, gbad = gs.get("median_tps"), gs.get("min_tps"), gs.get("max_tps"), bool(gs.get("invalid"))
        dec_e2e = ds.get("median_tps_e2e")
    else:  # legacy: median of whatever runs exist
        v = sorted(x["decode_tps"] for x in d.get("decode", []) if x.get("decode_tps"))
        dec, dmin, dmax, dbad = _median(v), (v[0] if v else None), (v[-1] if v else None), False
        gre = gmin = gmax = dec_e2e = None
        gbad = False
    tools = d.get("tools", [])
    tok = d.get("tools_pass")
    if tok is None and tools:
        tok = all(t["stream"].get("ok") and t["nonstream"].get("ok") and t.get("round_trip", {}).get("ok") for t in tools)
    flags = list(d.get("flags", []))
    cont = d.get("contention") or {}
    contended = any("CONTENTION" in f for f in flags) or bool((cont.get("overall") or {}).get("flagged"))
    if schema < 2:
        flags.append("legacy harness (schema 1)")
        c = cont if "max_foreign_gpu_pct" in cont else {}
        if c.get("max_foreign_gpu_pct", 0) > 10:
            contended = True
            flags.append(f"CONTENTION: max foreign GPU {c['max_foreign_gpu_pct']:.0f}%")
    p4, p4b = pref(4000, "prefill_tps")
    p16, p16b = pref(16000, "prefill_tps")
    c4, c4b = pref(4000, "cached")
    c16, c16b = pref(16000, "cached")
    mem = d.get("peak_memory_mb") or d.get("peak_footprint_mb")
    engine = d.get("engine") or ""
    return {
        "model": d.get("model_name") or model_of(label), "key": d.get("config_label") or label, "label": label,
        "display": d.get("display") or engine_of(label), "engine": engine,
        "version": d.get("version") or "", "when": d.get("when", ""), "file": os.path.basename(path),
        "hardware": hardware(d),
        "decode": dec, "decode_min": dmin, "decode_max": dmax, "decode_bad": dbad, "decode_e2e": dec_e2e,
        "greedy": gre, "greedy_min": gmin, "greedy_max": gmax, "greedy_bad": gbad,
        "p4k": p4, "p4k_bad": p4b, "p16k": p16, "p16k_bad": p16b,
        "c4k": c4, "c4k_bad": c4b, "c16k": c16, "c16k_bad": c16b,
        "pt4k": (pf.get(4000) or {}).get("prompt_tokens"), "pt16k": (pf.get(16000) or {}).get("prompt_tokens"),
        "mem_mb": mem, "mapped_mb": d.get("mapped_file_clean_mb"),
        "tools": tok, "invalid": list(d.get("invalid", [])), "flags": flags, "contended": contended,
        "foreign_max": (cont.get("overall") or cont).get("max_foreign_gpu_pct"),
        "ours": engine in OURS or bool(re.search(r"kapitsa|naceinfer", label)),
        "schema": d.get("schema", 1),
    }


def load(res_dirs, include_legacy):
    best = {}
    for res_dir in res_dirs:
        for f in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            if not isinstance(d, dict) or "label" not in d or d["label"].startswith("smoke"):
                continue
            if d.get("schema", 1) < 2 and not include_legacy:
                continue
            r = extract(d, f)
            if not r["model"]:
                continue
            r["rank"] = (not r["invalid"] and not r["contended"], not r["invalid"], r["when"])
            k = (r["model"], r["key"])
            if k not in best or r["rank"] >= best[k]["rank"]:
                best[k] = r
    return best


COLS = [  # key, header, fmt, better ("max"/"min")
    ("decode", "decode tok/s (median, T=0.6)", "%.1f", "max"),
    ("greedy", "decode greedy tok/s (T=0)", "%.1f", "max"),
    ("p4k", "prefill 4k tok/s", "%.0f", "max"),
    ("p16k", "prefill 16k tok/s", "%.0f", "max"),
    ("c4k", "cached TTFT 4k s", "%.2f", "min"),
    ("c16k", "cached TTFT 16k s", "%.2f", "min"),
    ("mem_mb", "peak memory GB", None, "min"),
]


def cell(r, key, fmt, best):
    v = r.get(key)
    if v is None:
        return "-"
    if key == "mem_mb":
        s = f"{v / 1024:.1f}"
        if (r.get("mapped_mb") or 0) > 1024:
            s += f" (+{r['mapped_mb'] / 1024:.1f} mmap)"
    else:
        s = fmt % v
    if r.get(key + "_bad"):
        return s + "!"
    if best is not None and v == best:
        s = f"**{s}**"
    if key in ("decode", "greedy") and r.get(key + "_min") is not None and r.get(key + "_max") is not None:
        s += f" ({r[key + '_min']:.1f}-{r[key + '_max']:.1f})"
    return s


def short_flags(r):
    out = []
    if r["invalid"]:
        out.append("INVALID: " + "; ".join(r["invalid"])[:160])
    if r["contended"]:
        out.append("CONTENTION" + (f" (max {r['foreign_max']:.0f}%)" if r.get("foreign_max") is not None else ""))
    other = [f for f in r["flags"] if "CONTENTION" not in f]
    if other:
        out.append("; ".join(other)[:160])
    return " / ".join(out) or "-"


def table(model, rows):
    rows = sorted(rows, key=lambda r: -(r["decode"] or 0))
    # cross-engine prompt-length consistency: every engine should see the same prompt (±1%)
    for size, key in ((4000, "pt4k"), (16000, "pt16k")):
        med = _median([r[key] for r in rows])
        for r in rows:
            if med and r[key] and abs(r[key] - med) > 0.01 * med:
                r["flags"].append(f"prompt {size}: {r[key]} tokens vs median {med:.0f} across engines")
    best = {}
    for key, _, _, how in COLS:
        vals = [r[key] for r in rows if r.get(key) is not None and not r.get(key + "_bad") and not r["invalid"] and not r["contended"]
                and not (key == "mem_mb" and (r.get("mapped_mb") or 0) > 1024)]  # mmap'd weights are outside phys_footprint
        best[key] = (max(vals) if how == "max" else min(vals)) if vals else None
    hw = collections.Counter(r["hardware"] for r in rows if r.get("hardware")).most_common(1)
    lines = [f"\n### {model}" + (f" ({hw[0][0]})" if hw else "") + "\n",
             "| engine | " + " | ".join(h for _, h, _, _ in COLS) + " | tools | flags |",
             "|---|" + "---|" * (len(COLS) + 2)]
    for r in rows:
        name = r["display"] + (" (ours)" if r["ours"] and "ours" not in r["display"] else "")
        tools = "pass" if r["tools"] else ("-" if r["tools"] is None else "FAIL")
        lines.append(f"| {name} | " + " | ".join(cell(r, k, f, best[k]) for k, _, f, _ in COLS) + f" | {tools} | {short_flags(r)} |")
    lines.append("")
    lines.append("Runs: " + "; ".join(f"{r['display']} = `{r['label']}`" + (f" {r['version']}" if r['version'] else "") + f" ({r['when']})" for r in rows))
    lines.append("")
    lines.append("`!` = that measurement is INVALID (see flags); bold = best valid value (memory: only rows whose weights "
                 "count in phys_footprint; `+N mmap` = resident mmap'd weight files on top); ranges are min-max over runs.")
    return "\n".join(lines)


CSV_COLS = ["model", "display", "key", "label", "engine", "version", "when", "decode", "decode_min", "decode_max",
            "decode_e2e", "greedy", "greedy_min", "greedy_max", "p4k", "p16k", "c4k", "c16k", "pt4k", "pt16k",
            "mem_mb", "mapped_mb", "tools", "invalid", "flags", "contended", "foreign_max", "ours", "schema", "file"]


def csv_val(v):
    if isinstance(v, list):
        return "; ".join(v)
    if isinstance(v, float):
        return round(v, 3)
    return "" if v is None else v


def main():
    ap = argparse.ArgumentParser(description="markdown + CSV comparison tables from results/*.json")
    ap.add_argument("results", nargs="*", help="results directories (default: $BENCH_OUT/results or bench/harness/runs/results)")
    ap.add_argument("--out", default=None, help="output path without extension (default <first results dir>/compare_table)")
    ap.add_argument("--include-legacy", action="store_true", help="also use schema-1 results (old harness)")
    a = ap.parse_args()
    dirs = a.results or [os.path.join(bench.out_dir(), "results")]
    best = load(dirs, a.include_legacy)
    out = a.out or os.path.join(dirs[0], "compare_table")
    models = sorted({m for m, _ in best}, key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))
    md, csv_rows = [], []
    for m in models:
        rows = [r for (mm, _), r in best.items() if mm == m]
        md.append(table(m, rows))
        csv_rows += sorted(rows, key=lambda r: -(r["decode"] or 0))
    text = "\n".join(md) if md else "(no results)"
    print(text)
    with open(out + ".md", "w") as fh:
        fh.write(text + "\n")
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLS)
        for r in csv_rows:
            w.writerow([csv_val(r.get(c)) for c in CSV_COLS])
    print(f"\n-> {out}.md, {out}.csv")


if __name__ == "__main__":
    main()
