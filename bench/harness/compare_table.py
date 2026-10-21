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

A second table per model covers the agent session (bench.py's "agent" phase, agent_replay.py), from each config's
latest run that has one (clean: no INVALID reason in the phase and no contention flag in it):
  engine time    the sum of the replayed requests' times (the gaps for the agent's tools are left out)
  prefill        the sum of the requests' times to first token; decode = the rest
  writing tok/s  all answers' tokens after their first streamed event over the time from first to last event
  TTFT           first request (cold), and the median / p90 of the follow-up requests

A third table per model covers the agent-harness metrics of runs made with --agentic (agentic.py): cold start
(spawn to first token; the first request's penalty over later ones), cold TTFT and decode tok/s at 1K / 16K / the
largest context, decode retention (largest over smallest), warm-prefix TTFT, whether a prefix survives another
conversation, the agent replay's cache hit rate, sustained-decode falloff, idle and peak memory.

usage: compare_table.py [results dir ...] [--out FILE_WITHOUT_EXT] [--include-legacy]
       writes <out>.md and <out>.csv, prints the markdown (default: the first results dir/compare_table)
"""
import argparse, collections, csv, glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench  # noqa: E402

MODEL_ORDER = ["Qwen3.6-35B-A3B", "Qwen3.8-27B", "Qwen3.5-4B", "Qwen3.5-4B (bf16)"]
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
    if "4b-bf16" in l:
        return "Qwen3.5-4B (bf16)"
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


def has_main_phases(d):
    return bool(d.get("decode") or d.get("decode_greedy") or d.get("prefill") or d.get("tools")) or d.get("schema", 1) < 2


def load(res_dirs, include_legacy):
    """(best run per (model, config) for the main table, best run per (model, config) with an agent session)"""
    best, best_agent, best_ax = {}, {}, {}
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
            k = (r["model"], r["key"])
            if has_main_phases(d):
                r["rank"] = (not r["invalid"] and not r["contended"], not r["invalid"], r["when"])
                if k not in best or r["rank"] >= best[k]["rank"]:
                    best[k] = r
            ax = extract_agentic(d, f, r)
            if ax:
                if k not in best_ax or ax["rank"] >= best_ax[k]["rank"]:
                    best_ax[k] = ax
            ag = extract_agent(d, f, r)
            if ag:
                if k not in best_agent or ag["rank"] >= best_agent[k]["rank"]:
                    best_agent[k] = ag
    return best, best_agent, best_ax


def sweep_point(sw, which):
    """(ttft_s, decode_tps, prompt_tokens) of the smallest / largest / 16000-target sweep point, or Nones."""
    pts = [p for p in (sw or {}).get("points", []) if not p.get("error")]
    if not pts:
        return None, None, None
    if which == "16k":
        pts = [p for p in pts if p.get("target") == 16000]
        if not pts:
            return None, None, None
        p = pts[0]
    else:
        p = pts[0] if which == "min" else pts[-1]
    return p.get("ttft_s"), p.get("decode_tps"), p.get("prompt_tokens")


def extract_agentic(d, path, base):
    """The --agentic row of a result (cold_start / sweep / cache / sustain / agent cache hit rate), or None."""
    cs, sw, ca, su = d.get("cold_start"), d.get("sweep"), d.get("cache"), d.get("sustain")
    hit = (d.get("agent") or {}).get("cache_hit_rate")
    if not (cs or sw or ca or su or hit is not None):
        return None
    cont_ph = (d.get("contention") or {}).get("phases") or {}
    contended = any((cont_ph.get(k) or {}).get("flagged") for k in ("sweep", "cache", "sustain"))
    invalid = [x for x in d.get("invalid", []) if x.split(":")[0] in ("sweep", "cache", "sustain", "cold start")]
    r = {k: base[k] for k in ("model", "key", "label", "display", "engine", "version", "when", "file", "hardware", "ours")}
    t1, d1, _ = sweep_point(sw, "min")
    t16, d16, _ = sweep_point(sw, "16k")
    tm, dm, ptm = sweep_point(sw, "max")
    idle = d.get("idle_footprint_mb")
    peak = (d.get("memory") or {}).get("sweep", {}).get("phys_footprint_peak_mb") or d.get("peak_memory_mb")
    r.update({"start_s": (cs or {}).get("spawn_to_first_token_s"), "penalty_s": (cs or {}).get("first_request_penalty_s"),
              "ttft_1k": t1, "ttft_16k": t16, "ttft_max": tm, "dec_1k": d1, "dec_max": dm, "max_tokens": ptm,
              "retention": (sw or {}).get("decode_retention"), "warm_ttft": (ca or {}).get("warm_ttft_s"), "followup_ttft": (ca or {}).get("followup_ttft_s"),
              "followup_hit_rate": (ca or {}).get("followup_hit_rate"),
              "warm_speedup": (ca or {}).get("warm_vs_cold_speedup"),
              "kept": (ca or {}).get("interleave_prefix_kept"), "interleave": (ca or {}).get("interleave_vs_cold"),
              "hit_rate": hit, "warm_hit_rate": (ca or {}).get("warm_hit_rate"),
              "falloff": (su or {}).get("thermal_falloff_pct"), "sustain_s": (su or {}).get("duration_s"),
              "idle_mb": idle, "peak_mb": peak,
              "invalid": invalid, "flags": [f for f in d.get("flags", []) if f.split(":")[0] in ("sweep", "cache", "sustain", "POWER")],
              "contended": contended, "foreign_max": None})
    r["rank"] = (not invalid and not contended, not invalid, r["when"])
    return r


AGENTIC_COLS = [  # key, header, fmt, better
    ("start_s", "spawn to first token s", "%.1f", "min"),
    ("penalty_s", "first-request penalty s", "%.2f", "min"),
    ("ttft_1k", "TTFT 1K s", "%.2f", "min"),
    ("ttft_16k", "TTFT 16K s", "%.2f", "min"),
    ("ttft_max", "TTFT max s", "%.2f", "min"),
    ("dec_1k", "decode 1K tok/s", "%.1f", "max"),
    ("dec_max", "decode max tok/s", "%.1f", "max"),
    ("retention", "decode retention", "%.2f", "max"),
    ("warm_ttft", "new-question TTFT s", "%.2f", "min"),
    ("followup_ttft", "follow-up TTFT s", "%.2f", "min"),
    ("hit_rate", "agent cache hit rate", "%.0f%%", "max"),
    ("falloff", "sustained falloff %", "%.1f", "min"),
]


def agentic_table(model, rows):
    rows = sorted(rows, key=lambda r: r["display"])
    best = {}
    for key, _, _, how in AGENTIC_COLS:
        vals = [r[key] for r in rows if r.get(key) is not None and not r["invalid"] and not r["contended"]]
        best[key] = (max(vals) if how == "max" else min(vals)) if vals else None
    hw = collections.Counter(r["hardware"] for r in rows if r.get("hardware")).most_common(1)
    lines = [f"\n### {model}: agent-harness metrics" + (f" ({hw[0][0]})" if hw else "") + "\n",
             "| engine | " + " | ".join(h for _, h, _, _ in AGENTIC_COLS) + " | prefix kept after other chat | idle / peak GB | flags |",
             "|---|" + "---|" * (len(AGENTIC_COLS) + 3)]
    for r in rows:
        name = r["display"] + (" (ours)" if r["ours"] and "ours" not in r["display"] else "")
        cells = []
        for k, _, f, _ in AGENTIC_COLS:
            v = r.get(k)
            if v is not None and k == "hit_rate":
                v = v * 100
            c = "-" if v is None else f % v
            if v is not None and r.get(k) == best[k]:
                c = f"**{c}**"
            cells.append(c)
        kept = "-" if r.get("kept") is None else ("yes" if r["kept"] else "no")
        if r.get("interleave") is not None:
            kept += f" ({r['interleave']:.2f}x cold)"
        mem = "-" if not r.get("peak_mb") else (f"{(r.get('idle_mb') or 0) / 1024:.1f} / {r['peak_mb'] / 1024:.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {kept} | {mem} | {short_flags(r)} |")
    lines.append("")
    lines.append("Runs: " + "; ".join(f"{r['display']} = `{r['file']}`" for r in rows))
    lines.append("")
    lines.append("`-` = not measured or not reported by the engine (a hit rate of 0% means it reported 0 cached tokens). "
                 "Max = the largest context that fits the server's window (`max tokens` in the CSV). Retention = decode at "
                 "the largest context over the smallest. Falloff = decode of the last third of the sustain run vs the first.")
    return "\n".join(lines)


AGENTIC_CSV_COLS = ["model", "display", "key", "engine", "version", "when"] + [k for k, *_ in AGENTIC_COLS] + [
    "max_tokens", "warm_speedup", "followup_hit_rate", "kept", "interleave", "warm_hit_rate", "sustain_s", "idle_mb", "peak_mb", "invalid",
    "flags", "contended", "ours", "file"]


def extract_agent(d, path, base):
    """The agent-session row of a result, or None."""
    a = d.get("agent")
    if not isinstance(a, dict) or not a.get("summary"):
        return None
    s = a["summary"]
    ph = ((d.get("contention") or {}).get("phases") or {}).get("agent") or {}
    contended = bool(ph.get("flagged"))
    invalid = list(a.get("invalid") or [])
    r = {k: base[k] for k in ("model", "key", "label", "display", "engine", "version", "when", "file", "hardware", "ours")}
    tr = a.get("trace") or {}
    r.update({"engine_s": s.get("engine_s"), "prefill_s": s.get("prefill_s"), "decode_s": s.get("decode_s"),
              "wall_s": s.get("wall_s"), "write_tps": s.get("write_tps"), "first_ttft": s.get("first_ttft_s"),
              "fu_med": s.get("followup_ttft_median_s"), "fu_p90": s.get("followup_ttft_p90_s"),
              "tokens_out": s.get("completion_tokens"), "tokens_cap": s.get("max_tokens"),
              "turns": s.get("turns"), "turns_ok": s.get("turns_ok"), "context_max": s.get("context_tokens_max"),
              "trace": tr.get("name"), "trace_sha": (tr.get("sha256") or "")[:12], "tool_mode": a.get("tool_mode"),
              "invalid": invalid, "flags": list(a.get("flags") or []) + [f for f in d.get("flags", []) if f.startswith("disk:")],
              "contended": contended, "foreign_max": ph.get("max_foreign_gpu_pct"),
              "mem_mb": (d.get("memory") or {}).get("agent", {}).get("phys_footprint_peak_mb")})
    r["rank"] = (not invalid and not contended, not invalid, r["when"])
    return r


AGENT_COLS = [  # key, header, fmt, better
    ("engine_s", "engine time s", "%.1f", "min"),
    ("prefill_s", "prefill s", "%.1f", "min"),
    ("decode_s", "decode s", "%.1f", "min"),
    ("write_tps", "writing tok/s", "%.1f", "max"),
    ("first_ttft", "first TTFT s", "%.2f", "min"),
    ("fu_med", "follow-up TTFT median s", "%.2f", "min"),
    ("fu_p90", "follow-up TTFT p90 s", "%.2f", "min"),
]


def agent_table(model, rows):
    rows = sorted(rows, key=lambda r: r["engine_s"] if r["engine_s"] is not None else 1e18)
    best = {}
    for key, _, _, how in AGENT_COLS:
        vals = [r[key] for r in rows if r.get(key) is not None and not r["invalid"] and not r["contended"]]
        best[key] = (max(vals) if how == "max" else min(vals)) if vals else None
    traces = sorted({f"{r['trace']} ({r['trace_sha']}, tool mode {r['tool_mode']})" for r in rows})
    hw = collections.Counter(r["hardware"] for r in rows if r.get("hardware")).most_common(1)
    lines = [f"\n### {model}: agent session" + (f" ({hw[0][0]})" if hw else "") + "\n",
             "Trace: " + "; ".join(traces) + "\n",
             "| engine | " + " | ".join(h for _, h, _, _ in AGENT_COLS) + " | tokens out | turns | flags |",
             "|---|" + "---|" * (len(AGENT_COLS) + 3)]
    for r in rows:
        name = r["display"] + (" (ours)" if r["ours"] and "ours" not in r["display"] else "")
        cells = []
        for k, _, f, _ in AGENT_COLS:
            v = r.get(k)
            c = "-" if v is None else f % v
            if v is not None and v == best[k]:
                c = f"**{c}**"
            cells.append(c)
        out = f"{r['tokens_out']}/{r['tokens_cap']}" if r.get("tokens_cap") else "-"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {out} | {r['turns_ok']}/{r['turns']} | {short_flags(r)} |")
    lines.append("")
    lines.append("Runs: " + "; ".join(f"{r['display']} = `{r['file']}`" for r in rows))
    return "\n".join(lines)


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


AGENT_CSV_COLS = ["model", "display", "key", "engine", "version", "when", "engine_s", "prefill_s", "decode_s", "wall_s",
                  "write_tps", "first_ttft", "fu_med", "fu_p90", "tokens_out", "tokens_cap", "turns", "turns_ok",
                  "context_max", "mem_mb", "trace", "trace_sha", "tool_mode", "invalid", "flags", "contended",
                  "foreign_max", "ours", "file"]


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
    best, best_agent, best_ax = load(dirs, a.include_legacy)
    out = a.out or os.path.join(dirs[0], "compare_table")
    models = sorted({m for m, _ in list(best) + list(best_agent) + list(best_ax)},
                    key=lambda m: (MODEL_ORDER.index(m) if m in MODEL_ORDER else len(MODEL_ORDER), m))
    md, csv_rows, agent_rows, ax_rows = [], [], [], []
    for m in models:
        rows = [r for (mm, _), r in best.items() if mm == m]
        if rows:
            md.append(table(m, rows))
            csv_rows += sorted(rows, key=lambda r: -(r["decode"] or 0))
        arows = [r for (mm, _), r in best_agent.items() if mm == m]
        if arows:
            md.append(agent_table(m, arows))
            agent_rows += arows
        xrows = [r for (mm, _), r in best_ax.items() if mm == m]
        if xrows:
            md.append(agentic_table(m, xrows))
            ax_rows += xrows
    text = "\n".join(md) if md else "(no results)"
    print(text)
    with open(out + ".md", "w") as fh:
        fh.write(text + "\n")
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLS)
        for r in csv_rows:
            w.writerow([csv_val(r.get(c)) for c in CSV_COLS])
    if agent_rows:
        with open(out + "_agent.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(AGENT_CSV_COLS)
            for r in agent_rows:
                w.writerow([csv_val(r.get(c)) for c in AGENT_CSV_COLS])
    if ax_rows:
        with open(out + "_agentic.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(AGENTIC_CSV_COLS)
            for r in ax_rows:
                w.writerow([csv_val(r.get(c)) for c in AGENTIC_CSV_COLS])
    print(f"\n-> {out}.md, {out}.csv" + (f", {out}_agent.csv" if agent_rows else "")
          + (f", {out}_agentic.csv" if ax_rows else ""))


if __name__ == "__main__":
    main()
