#!/usr/bin/env python3
"""kapitsa bench compare: Kapitsa against the other engines installed on this Mac, on one model, one table.

`kapitsa bench compare` runs this file. It adds no protocol: every engine goes through compare_engines.py and bench.py
(README.md in this folder), one at a time under the GPU lock, with the same corpus, sampler and thinking settings. This
file only picks the harness config each engine can run here, runs it and prints the result (--list, --json, --report).
An engine that is not installed, or whose weights are missing, is listed with how to get it. Nothing is downloaded,
installed or sent. Python 3.9 and the standard library only; colour only on a terminal (NO_COLOR is honoured).
"""
import argparse, copy, datetime, glob, json, os, platform, re, shutil, signal, subprocess, sys, tempfile, textwrap, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCHEMA = "kapitsa-bench-compare/1"
KAPITSA = "kapitsa"

# model shorthand -> label prefix in configs/*.json, Kapitsa's folder name in ~/.cache/kapitsa/models, served id
MODELS = {
    "4b": {"name": "Qwen3.5-4B", "prefix": "q35-4b", "folder": "qwen3.5-4b-4bit", "id": "qwen3.5-4b", "pull": "4b"},
    "27b": {"name": "Qwen3.8-27B", "prefix": "q38-27b", "folder": "qwen3.8-27b-4bit", "id": "qwen3.8-27b", "pull": "27b"},
    "35b": {"name": "Qwen3.6-35B-A3B", "prefix": "q36-35b", "folder": "qwen3.6-35b-a3b-4bit", "id": "qwen3.6-35b-a3b",
            "pull": "35b"},
}
DEFAULT_MODEL = "4b"

# engine = the configs/<name>.json file; shown name, and what to do when it is not on this Mac
ENGINES = [
    (KAPITSA, "Kapitsa", "this binary"),
    ("omlx", "oMLX", "release wheel from https://github.com/jundot/omlx/releases (OMLX_BIN=path/to/omlx)"),
    ("mlx-serve", "mlx-serve", "brew install mlx-serve (MLX_SERVE_BIN=path)"),
    ("mlx-lm", "mlx-lm", "uv tool install mlx-lm (MLX_LM_SERVER=path/to/mlx_lm.server)"),
    ("llama.cpp", "llama.cpp", "brew install llama.cpp (LLAMA_SERVER=path/to/llama-server)"),
    ("ollama", "Ollama", "brew install --cask ollama (OLLAMA_BIN=path)"),
    ("lmstudio", "LM Studio", "curl -fsSL https://lmstudio.ai/install.sh | bash (LMS_BIN=path/to/lms)"),
    ("mtplx", "MTPLX", "uv tool install mtplx (MTPLX_BIN=path)"),
    ("rapid-mlx", "Rapid-MLX", "see bench/harness/README.md (RAPID_MLX_BIN=path)"),
    ("mlx-vlm", "mlx-vlm", "uv pip install mlx-vlm (MLX_VLM_PYTHON=path/to/python)"),
]
ENGINE_NAME = {k: n for k, n, _ in ENGINES}
INSTALL_HINT = {k: h for k, _, h in ENGINES}

# what runs per profile (compare_engines.py --skip / --sizes / --bench-args)
PROFILES = {
    "default": {"sizes": "4000,16000", "skip": "greedy,tools,agent",
                "bench_args": "--decode-runs 3 --cached-samples 2", "minutes": 5},
    "quick": {"sizes": "4000", "skip": "greedy,tools,agent", "bench_args": "--decode-runs 2 --cached-samples 1",
              "minutes": 3},
}


class Term:
    def __init__(self, stream, color=None):
        self.tty = stream.isatty()
        self.color = (self.tty and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
                      if color is None else color)
        self.width = max(60, min(shutil.get_terminal_size((100, 24)).columns, 140)) if self.tty else 100

    def c(self, code, s):
        return f"\033[{code}m{s}\033[0m" if self.color else s

    bold, dim, green = (lambda self, s: self.c("1", s)), (lambda self, s: self.c("2", s)), (lambda self, s: self.c("32", s))
    red, yellow, accent = (lambda self, s: self.c("31", s)), (lambda self, s: self.c("33", s)), (lambda self, s: self.c("36", s))


def vlen(s):
    return len(re.sub(r"\033\[[0-9;]*m", "", s))


def pad(s, n, right=False):
    return " " * (n - vlen(s)) + s if right else s + " " * (n - vlen(s))


def clip(s, n):
    return s if len(s) <= n else s[: max(0, n - 1)] + "…"


def load_harness():
    import compare_engines as ce  # starts nothing until main()
    return ce


def machine():
    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    mem = sh("sysctl", "-n", "hw.memsize")
    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.machine(),
            "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else None,
            "macos": sh("sw_vers", "-productVersion") or platform.mac_ver()[0] or platform.platform()}


def kapitsa_home():
    return os.path.expanduser(os.environ.get("KAPITSA_HOME") or "~/.cache/kapitsa")

def rank_key(label):
    """Which of an engine's configs to try first: the tuned ones (best > MTP > DFlash), then the plain one."""
    return (0 if "best" in label else 1 if "mtp" in label else 2 if "dflash" in label else 3, label)


def candidates(ce, cfgs, engine, model):
    """The engine's harness configs for this model, best first. bf16 is another set of weights, and nomtp / mlx-q4
    runs are the harness's controls, so none of them stands for the engine here."""
    pre = MODELS[model]["prefix"] + "-"
    out = [c for c in cfgs if os.path.splitext(os.path.basename(c["_file"]))[0] == engine
           and c["label"].startswith(pre) and not re.search(r"bf16|nomtp|mlx-q4", c["label"])
           and c.get("enabled") is not False]
    return sorted(out, key=lambda c: rank_key(c["label"]))


def kapitsa_synth(cfgs, model):
    """Kapitsa on the files `kapitsa serve` left in ~/.cache/kapitsa/models, for a Mac with no harness weights."""
    m = MODELS[model]
    base = next((c for c in cfgs if c["label"] == m["prefix"] + "-kapitsa"), None)
    folder = os.path.join(kapitsa_home(), "models", m["folder"])
    if base is None:
        return None
    c = copy.deepcopy(base)
    c.update({"label": m["prefix"] + "-kapitsa-pulled", "display": "Kapitsa (as pulled)", "model_path": folder,
              "requires_paths": [folder + "/config.json"],
              "note": "Kapitsa with the companions that `kapitsa serve` fetched next to the weights (it finds them itself)."})
    c["cmd"][c["cmd"].index("--model") + 1] = folder
    if "--dflash" in c["cmd"]:  # the harness names the drafter; here serve finds what was pulled
        j = c["cmd"].index("--dflash")
        del c["cmd"][j:j + 2]
    return c


NOT_INSTALLED = re.compile(r"none of \[|not found on PATH|is not set|exists or is on PATH|not on PATH|No such file|\.lmstudio/bin")


def plan_engine(ce, cfgs, engine, model, args):
    """-> (config to run or None, why not, [variants tried]). Never starts anything."""
    cands = candidates(ce, cfgs, engine, model)
    if engine == KAPITSA:
        cands = cands + [x for x in [kapitsa_synth(cfgs, model)] if x]
    if not cands:
        return None, "no harness config for this model", []
    tried, missing = [], 0
    for c in cands:
        p, _ = ce.plan(c, args, "plan")
        errs = ([p["skip"]] if p["skip"] else []) + p["errors"]
        if not errs:
            return c, None, tried
        tried.append({"label": c["label"], "why": "; ".join(errs)})
        if any(NOT_INSTALLED.search(e) for e in errs):
            missing += 1
    if missing == len(cands):
        return None, "not installed", tried
    return None, "weights not here", tried


def skip_hint(engine, why, tried, model):
    if why == "not installed":
        return INSTALL_HINT.get(engine, "")
    first = tried[0]["why"] if tried else ""
    if engine == KAPITSA:
        return f"kapitsa pull {MODELS[model]['pull']}"
    return clip(first, 150) + "  (the files this config reads are listed in bench/harness/README.md)"


class Args:  # what compare_engines.plan()/run_config() read from argparse
    def __init__(self, prof, quiet_timeout, lock_timeout, quiet_s):
        self.sizes, self.skip, self.bench_args = prof["sizes"], prof["skip"], prof["bench_args"]
        self.agent_trace, self.quiet_max, self.quiet_timeout, self.agentic = "none", None, quiet_timeout, False
        self.lock_timeout, self.env_keys, self.quiet_s = lock_timeout, [], quiet_s


def run_engine(cfg, out_dir, prof, a, log_path, status):
    """One engine through compare_engines.py in its own process (its server, GPU lock and teardown end with it).
    Returns the result dict, or {"failed": why}."""
    cfg_path = os.path.join(out_dir, "configs", cfg["label"] + ".json")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    flat = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "extends"}
    json.dump([flat], open(cfg_path, "w"), indent=1)
    cmd = [sys.executable, "-u", os.path.join(HERE, "compare_engines.py"), cfg_path, "--out-dir", out_dir,
           "--sizes", prof["sizes"], "--skip", prof["skip"], "--bench-args", prof["bench_args"],
           "--agent-trace", "none", "--quiet-timeout", str(a.quiet_timeout), "--quiet-s", str(a.quiet_s)]
    if a.lock_timeout is not None:
        cmd += ["--lock-timeout", str(a.lock_timeout)]
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=os.environ)
        try:
            while proc.poll() is None:
                status(tail_status(log_path), time.time() - t0)
                time.sleep(0.4)
        except BaseException:
            proc.terminate()  # compare_engines stops its server on SIGTERM, then exits
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
    found = sorted(glob.glob(os.path.join(out_dir, "results", cfg["label"] + "-*.json")), key=os.path.getmtime)
    if proc.returncode != 0 and not found:
        return {"failed": f"compare_engines.py exited {proc.returncode}"}
    if not found:
        text = open(log_path, errors="replace").read()
        m = re.search(r"SKIP: (.*)", text)
        return {"failed": m.group(1).strip() if m else "no result was written"}
    return json.load(open(found[-1]))


def tail_status(path):
    """The last harness log line: what the engine is doing."""
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - 4000))
            lines = [re.sub(r"^\[?\d\d:\d\d:\d\d\]?\s*", "", l).strip() for l in f.read().decode("utf-8", "replace").splitlines()]
    except OSError:
        return ""
    return next((l for l in reversed(lines) if l), "")


def metrics_of(res):
    """The numbers the table shows, from one harness result (None where the run did not produce one)."""
    d = (res.get("decode_summary") or {})
    pre = {p.get("target"): p for p in (res.get("prefill") or []) if isinstance(p, dict)}
    sizes = sorted(pre)
    short, long_ = (sizes[0] if sizes else None), (sizes[-1] if len(sizes) > 1 else None)

    def pf(t, k):
        return (pre.get(t) or {}).get(k) if t is not None else None
    cached = lambda t: ((pre.get(t) or {}).get("cached") or {}).get("median_ttft_s") if t is not None else None
    return {"decode_tps": d.get("median_tps"), "decode_min_tps": d.get("min_tps"), "decode_max_tps": d.get("max_tps"),
            "decode_runs": d.get("valid_runs"),
            "prefill_short_tokens": pf(short, "prompt_tokens"), "prefill_short_tps": pf(short, "prefill_tps"),
            "prefill_long_tokens": pf(long_, "prompt_tokens"), "prefill_long_tps": pf(long_, "prefill_tps"),
            "ttft_cold_s": pf(short, "ttft_s"), "ttft_cached_s": cached(short),
            "memory_gb": round(res["peak_memory_mb"] / 1024, 2) if res.get("peak_memory_mb") else None,
            "tools_pass": res.get("tools_pass")}


def entry_of(engine, cfg, res):
    e = {"engine": engine, "name": ENGINE_NAME.get(engine, engine), "variant": cfg["label"],
         "display": res.get("display") or cfg.get("display"), "version": res.get("version"),
         "model_path": cfg.get("model_path"), "status": "ok", "flags": list(res.get("flags") or []),
         "invalid": list(res.get("invalid") or []), "result_file": None}
    e.update(metrics_of(res))
    if res.get("invalid") or e["decode_tps"] is None:
        e["status"] = "failed"
    prov = res.get("provenance") or {}
    e["command"] = prov.get("cmd_shell")
    return e


def f1(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def ttft(x):
    return "-" if x is None else (f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s")


def ranked(entries):
    ok = [e for e in entries if e["status"] == "ok"]
    return sorted(ok, key=lambda e: -(e["decode_tps"] or 0))


def table_rows(doc):
    ok = ranked(doc["engines"])
    return ok, (ok[0]["decode_tps"] if ok else 0), next((e for e in ok if e["engine"] == KAPITSA), None)


def skipped_line(s, t):
    return f"  {t.yellow('–')} {s['name']:10} {s['reason']}" + (f"\n      {t.dim(s['hint'])}" if s["hint"] else "")


def bar(frac, n=22):
    k = int(round(max(0.0, min(1.0, frac)) * n * 8))
    return ("█" * (k // 8) + " ▏▎▍▌▋▊▉"[k % 8]).rstrip().ljust(n)


def measures(two, sl, ll):
    """(heading, key, higher is better, formatter) for each column, in table order."""
    m = [("decode tok/s", "decode_tps", True, f1), (f"prefill {sl}", "prefill_short_tps", True, lambda x: f1(x, 0))]
    if two:
        m.append((f"prefill {ll}", "prefill_long_tps", True, lambda x: f1(x, 0)))
    return m + [("TTFT cold", "ttft_cold_s", False, ttft), ("TTFT cached", "ttft_cached_s", False, ttft),
                ("memory", "memory_gb", False, lambda x: f"{x:.1f} GB")]


def verdicts(ok, kap, ms, t):
    """One line per measure: where Kapitsa stands and by how much, against the best other engine."""
    rows, wins = [], 0
    for head, key, hi, fmt in ms:
        vals = [(e[key], e) for e in ok if e[key]]
        if kap[key] is None or len(vals) < 2:
            continue
        others = [(v, e) for v, e in vals if e is not kap]
        bv, be = max(others, key=lambda x: x[0]) if hi else min(others, key=lambda x: x[0])
        r = kap[key] / bv if hi else bv / kap[key]  # >1: Kapitsa is ahead
        place = 1 + sum(1 for v, _ in vals if (v > kap[key] if hi else v < kap[key]))
        first = place == 1
        wins += first
        what = ("faster" if hi else "lower") if first else ("slower" if hi else "higher")
        mark = t.green("✓") if first else t.red("✗")
        rows.append(f"  {mark} {head.replace(' tok/s', ''):15}{('1st' if first else f'{place}th' if place > 3 else ('2nd', '3rd')[place - 2]):>4}"
                    f"   {fmt(kap[key]):>9}   " + (t.green if first else t.red)(
                        f"{max(r, 1 / r):.2f}x {what}") + t.dim(f" than {be['name']}"))
    return wins, rows


def render(doc, t):
    m, mc = doc["model"], doc["machine"]
    out = ["", f"  {t.accent(t.bold('kapitsa bench compare'))}  {t.bold(m['name'])}"
           + t.dim(f"   {mc['chip']} · {mc['memory_gb']} GB · macOS {mc['macos']}"),
           t.dim(f"  {doc['profile']} profile · thinking off · one request at a time · sampler T=0.6 · "
                 f"prefill {doc['sizes'].replace(',', ' and ')} tokens"), ""]
    ok, top, kap = table_rows(doc)
    sizes = doc["sizes"].split(",")
    ms = measures(sizes[0] != sizes[-1], sizes[0], sizes[-1])
    if ok:
        best = {k: (max if hi else min)([e[k] for e in ok if e[k]] or [None], key=lambda x: x or 0) for _, k, hi, _ in ms}
        w = [max(len(e["name"]) for e in ok) + 3] + [max(len(h), *(len(f(e[k])) for e in ok if e[k] is not None)) + 2
                                                    for h, k, _, f in ms]
        bw = max(6, min(22, t.width - 92))  # the rest of the table is ~90 columns; shorter bars on a narrow terminal
        line = "  " + pad("", w[0]) + "".join(pad(h, w[i + 1], right=True) + (" " * (bw + 1) if i == 0 else "")
                                              for i, (h, _, _, _) in enumerate(ms))
        out += [t.dim(line), t.dim("  " + "─" * (len(line) - 2))]
        for e in ok:
            name = pad(("▸ " if e is kap else "  ") + e["name"], w[0])
            cells = [t.accent(t.bold(name)) if e is kap else name]
            for i, (h, k, hi, f) in enumerate(ms):
                s = pad(f(e[k]) if e[k] is not None else "-", w[i + 1] - 2, right=True)
                s = t.green(t.bold(s)) if len(ok) > 1 and e[k] is not None and e[k] == best[k] else s
                cells.append("  " + s)
                if i == 0:
                    cells.append(" " + (t.accent if e is kap else t.dim)(bar((e[k] or 0) / top if top else 0, bw)))
            out.append("  " + "".join(cells))
        if kap and len(ok) > 1:
            wins, rows = verdicts(ok, kap, ms, t)
            out += ["", t.bold(f"  Kapitsa is first on {wins} of {len(rows)} measures")] + rows
    else:
        out.append(t.red("  No engine produced a number."))
    for e in doc["engines"]:
        if e["status"] == "failed":
            out += ["", t.red(f"  ✗ {e['name']}") + f" ({e['variant']}) did not finish: "
                    + clip("; ".join(e["invalid"]) or "no decode result", 200) + t.dim(f"  log: {e.get('log')}")]
    out.append("")
    for e in ok:
        out.append(t.dim(f"  {e['name']:10} {clip(e['display'] or e['variant'], 38):38} {clip(e['version'] or '', 30)}"))
    for e in doc["engines"]:
        out += [t.yellow(f"  ! {e['name']}: {clip(f, t.width - 10)}") for f in e["flags"][:3]]
    if doc["skipped"]:
        out += ["", t.bold("  Not measured")] + [skipped_line(s, t) for s in doc["skipped"]]
    home = os.path.expanduser("~")
    out += ["", t.dim("  Results and logs: " + doc["out_dir"].replace(home, "~", 1)),
            t.dim("  `kapitsa bench compare --report` prints a Markdown block to keep or quote in a PR."), ""]
    return "\n".join(out)


def why(e, kap, decode_gap=True):
    """Facts from the two configs that could explain a gap. Not a measurement of the cause. Speculative decoding only
    speeds up decoding, so it is named only when decode is one of the gaps."""
    bits = []
    ed, kd = e.get("display") or "", kap.get("display") or ""
    spec = re.compile(r"MTP|DFlash|drafter|speculat|PLD", re.I)
    if decode_gap and (bool(spec.search(ed)) != bool(spec.search(kd)) or (spec.search(ed) and ed != kd)):
        bits.append(f"speculative decoding differs: {ed} against Kapitsa's {kd}")
    ep, kp = os.path.basename(e.get("model_path") or ""), os.path.basename(kap.get("model_path") or "")
    if ep and kp and ep != kp:
        bits.append(f"reads a different weights folder ({ep} against {kp}), so the quantization or files may differ")
    if e.get("version"):
        bits.append(f"version {e['version']}")
    return "; ".join(bits)


def scrub(text, doc):
    """Paths that name the user or this Mac's folders must not leave it in a report people share."""
    home = os.path.expanduser("~")
    for root, tag in ((doc.get("out_dir"), "<out>"), (os.environ.get("BENCH_MODELS"), "<models>"), (home, "~")):
        if root and root != "/":
            text = text.replace(root.rstrip("/"), tag)
    return re.sub(r"(/Users|/home)/[^/\s`]+", r"~", text)


def label(e):
    d = e.get("display") or e.get("variant") or ""
    return e["name"] if d.lower().startswith(e["name"].lower()) else f"{e['name']} ({d})"


def report(doc):
    m, mc = doc["model"], doc["machine"]
    ok, top, kap = table_rows(doc)
    sl, ll = doc["sizes"].split(",")[0], doc["sizes"].split(",")[-1]
    L = [f"### `kapitsa bench compare`: {m['name']} on {mc['chip']}, {mc['memory_gb']} GB, macOS {mc['macos']}", "",
         f"{doc['kapitsa']['version']}; harness "
         f"{doc['harness']['commit'] or doc['harness'].get('version') or 'unknown'} "
         f"(`bench/harness`, profile `{doc['profile']}`: thinking off, one request at a time, T=0.6, prefill "
         f"{doc['sizes']} tokens, medians). Measured {doc['when']}.", "",
         f"| # | engine | decode tok/s | prefill {sl} tok/s | prefill {ll} tok/s | TTFT cold | TTFT cached | memory | vs Kapitsa |",
         "|--:|---|--:|--:|--:|--:|--:|--:|--:|"]
    for i, e in enumerate(ok, 1):
        vs = "" if e is kap or not kap else f"{e['decode_tps'] / kap['decode_tps']:.2f}x"
        mem = f"{e['memory_gb']:.1f} GB" if e["memory_gb"] else "-"
        L.append(f"| {i} | {label(e)} | {f1(e['decode_tps'])} | {f1(e['prefill_short_tps'], 0)} | "
                 f"{f1(e['prefill_long_tps'], 0)} | {ttft(e['ttft_cold_s'])} | {ttft(e['ttft_cached_s'])} | "
                 f"{mem} | {vs} |")
    L += ["", "#### Who beats Kapitsa here, and why", ""]
    if not kap:
        L.append("Kapitsa did not produce a number in this run, so nothing is compared.")
    else:
        beat = []
        for e in (x for x in ok if x is not kap):
            wins = []
            for key, lab, low in (("decode_tps", "decode", False), ("prefill_short_tps", f"prefill {sl}", False),
                                  ("prefill_long_tps", f"prefill {ll}", False), ("ttft_cold_s", "cold TTFT", True),
                                  ("ttft_cached_s", "cached TTFT", True), ("memory_gb", "memory", True)):
                a, b = e.get(key), kap.get(key)
                r = (b / a if low else a / b) if a and b else 0
                if r > 1.03:
                    wins.append(f"{lab} {r:.2f}x")
            if wins:
                beat.append((e, wins))
        L += [] if beat else ["No engine measured here is more than 3% ahead of Kapitsa on any column."]
        for e, wins in beat:
            w = why(e, kap, any(x.startswith("decode") for x in wins))
            L.append(f"- **{e['name']}** is ahead on " + ", ".join(wins) + "." + (f" From the configs: {w}." if w else ""))
        if beat:
            L += ["", "The reasons above are read from the two configs, not measured. A gap is a place to look, "
                  "not a finding."]
    notes = [f"- {e['name']}: {f}" for e in doc["engines"] for f in e.get("flags") or []]
    if notes:
        L += ["", "#### Caveats from the run", ""] + notes
    if doc["skipped"]:
        L += ["", "Not measured: " + "; ".join(f"{s['name']} ({s['reason']})" for s in doc["skipped"]) + "."]
    L += ["", "<details><summary>Commands that ran</summary>", ""]
    L += [f"- {e['name']} `{e['variant']}`: `{e.get('command') or 'external server'}`" for e in doc["engines"]]
    L += ["", "</details>", "",
          "Kapitsa's maintainers wrote this benchmark; every config it ran is in `bench/harness/configs/` and is "
          "open to challenge. Re-run with `kapitsa bench compare --model " + doc["model"]["key"] + "`."]
    return scrub("\n".join(L) + "\n", doc)


def choose_model(args):
    ans = (args.model or "").lower()
    if not ans and sys.stdin.isatty() and sys.stderr.isatty() and not (args.json or args.yes):
        sys.stderr.write(f"Which model? {', '.join(MODELS)} [Enter = {DEFAULT_MODEL}]: ")
        sys.stderr.flush()
        ans = sys.stdin.readline().strip().lower()
    ans = ans or DEFAULT_MODEL
    if ans not in MODELS:
        raise SystemExit(f"kapitsa bench compare: no model {ans!r} (the harness has configs for {', '.join(MODELS)})")
    return ans


def parse(argv):
    ap = argparse.ArgumentParser(prog="kapitsa bench compare", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Engines are found from PATH and the harness variables (OMLX_BIN, MLX_SERVE_BIN, "
                                        "MLX_LM_SERVER, LLAMA_SERVER, OLLAMA_BIN, LMS_BIN, ...; "
                                        "bench/harness/local.env.example). Weights come from BENCH_MODELS (default ~/models).")
    ap.add_argument("--model", "-m", help="4b, 27b or 35b (default: ask on a terminal, else 4b)")
    ap.add_argument("--engines", help="comma list to measure, e.g. kapitsa,omlx (default: every engine found)")
    ap.add_argument("--quick", action="store_true", help="one prefill size and two decode runs (about 3 minutes per engine)")
    ap.add_argument("--list", action="store_true", help="show what was found and what is missing; start nothing")
    ap.add_argument("--json", action="store_true", help="print the result as JSON (progress goes to stderr)")
    ap.add_argument("--report", nargs="?", const="-", metavar="FILE",
                    help="print a Markdown block to keep or quote in a PR (and write it to FILE)")
    ap.add_argument("--out", help="folder for results and logs (default ~/.cache/kapitsa/bench/compare/<time>)")
    ap.add_argument("--ignore-running", action="store_true", help="measure even when other engines are running")
    ap.add_argument("--yes", "-y", action="store_true", help="do not ask before measuring")
    ap.add_argument("--quiet-timeout", type=float, default=300, help="seconds to wait for a quiet GPU per engine")
    ap.add_argument("--lock-timeout", type=float, default=None, help="seconds to wait for the GPU lock per engine")
    ap.add_argument("--quiet-s", type=float, default=15, help=argparse.SUPPRESS)
    ap.add_argument("--configs", default=os.path.join(HERE, "configs"), help=argparse.SUPPRESS)
    return ap.parse_args(argv)


SERVERS = {"omlx", "mlx-serve", "llama-server", "llmster", "mtplx", "rapid-mlx", "mlx_lm.server", "mlx_vlm.server"}


def others_running():
    """Engine or model servers already running: they share the GPU and memory, so a number beside them is not the engine's."""
    found = []
    for line in subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        t = cmd.split()
        if not t:
            continue
        names = [os.path.basename(x) for x in t[:4]]
        if names[0] in SERVERS or (names[0].startswith("python") and SERVERS & set(names[1:])) or \
                (names[0] == "kapitsa" and t[1:2] == ["serve"]) or (names[0] == "ollama" and t[1:2] in (["serve"], ["runner"])):
            found.append(f"{pid} {clip(cmd, 90)}")
    return found


def models_overlay(out_dir, model):
    """BENCH_MODELS for this run: the user's weights folder, plus the folder `kapitsa serve` made under the name the
    harness configs use, so the engines that read a plain MLX folder (mlx-lm, oMLX) work after a download alone."""
    base = os.path.abspath(os.path.expanduser(os.environ.get("BENCH_MODELS") or "~/models"))
    pulled = os.path.join(kapitsa_home(), "models", MODELS[model]["folder"])
    want = MODELS[model]["folder"] + "-mtp"
    if os.path.exists(os.path.join(base, want)) or not os.path.isdir(pulled):
        return base
    ov = os.path.join(out_dir, "models")
    os.makedirs(ov, exist_ok=True)
    if os.path.isdir(base):
        for n in os.listdir(base):
            if not os.path.lexists(os.path.join(ov, n)):
                os.symlink(os.path.join(base, n), os.path.join(ov, n))
    os.symlink(pulled, os.path.join(ov, want))
    return ov


def main(argv=None):
    args = parse(sys.argv[1:] if argv is None else argv)
    out_t, err_t = Term(sys.stdout), Term(sys.stderr)
    info = err_t if args.json else out_t
    model = choose_model(args)
    if args.json and args.report == "-":
        raise SystemExit("kapitsa bench compare: --json and --report both print to stdout; give --report a FILE")
    prof_name = "quick" if args.quick else "default"
    prof = PROFILES[prof_name]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.abspath(os.path.expanduser(args.out)) if args.out else os.path.join(
        kapitsa_home(), "bench", "compare", stamp)
    if not args.list:
        os.makedirs(out_dir, exist_ok=True)
    if "KAPITSA_BIN" not in os.environ and shutil.which("kapitsa"):
        os.environ["KAPITSA_BIN"] = shutil.which("kapitsa")
    bench_models = models_overlay(tempfile.mkdtemp(prefix="kbench-") if args.list else out_dir, model)
    os.environ["BENCH_MODELS"] = bench_models
    os.environ.setdefault("BENCH_OUT", out_dir)
    ce = load_harness()
    if os.path.exists(ce.bench.LOCAL_ENV):
        ce.bench.load_env_file(ce.bench.LOCAL_ENV)
    ce.set_out_dir(out_dir)
    cfgs = ce.load_configs(sorted(glob.glob(os.path.join(args.configs, "*.json"))))
    a = Args(prof, args.quiet_timeout, args.lock_timeout, args.quiet_s)
    a.quiet_max = ce.gpu_hogs.FOREIGN_MAX_PCT
    wanted = [x.strip() for x in args.engines.split(",")] if args.engines else None
    if wanted:
        unknown = [w for w in wanted if w not in ENGINE_NAME]
        if unknown:
            raise SystemExit(f"kapitsa bench compare: unknown engine {', '.join(unknown)} (known: {', '.join(ENGINE_NAME)})")
    todo, skipped = [], []
    for key, name, _ in ENGINES:
        if wanted and key not in wanted:
            continue
        cfg, reason, tried = plan_engine(ce, cfgs, key, model, a)
        if cfg:
            todo.append((key, cfg, tried))
        else:
            skipped.append({"engine": key, "name": name, "reason": reason, "hint": skip_hint(key, reason, tried, model),
                            "tried": tried})
    doc = {"schema": SCHEMA, "when": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "profile": prof_name,
           "model": dict(MODELS[model], key=model), "sizes": prof["sizes"], "machine": machine(),
           "kapitsa": {"version": kapitsa_version()}, "harness": harness_commit(ce), "out_dir": out_dir,
           "engines": [], "skipped": skipped}
    show = lambda s="": print(s, file=sys.stderr if args.json else sys.stdout)
    if args.list:
        show(info.bold(f"Engines for {MODELS[model]['name']}"))
        for key, cfg, tried in todo:
            show(f"  {info.green('✓')} {ENGINE_NAME[key]:10} {cfg['label']}" + info.dim(f"  ({cfg.get('display')})"))
        for s in skipped:
            show(skipped_line(s, info))
        if args.json:
            doc["engines"] = [{"engine": k, "variant": c["label"], "status": "ready"} for k, c, _ in todo]
            print(json.dumps(doc, indent=1))
        return 0
    if not todo:
        show(info.red("No engine can run here.") + " Run `kapitsa bench compare --list` to see what is missing.")
        return 1
    busy = [] if args.ignore_running else others_running()
    if busy:
        show(info.red("Other engines or model servers are running, which would skew every number:"))
        for b in busy:
            show("  " + b)
        show("Quit them and run again, or pass --ignore-running to measure anyway.")
        return 1
    show(info.bold(f"Measuring {len(todo)} engine{'s' if len(todo) != 1 else ''} on {MODELS[model]['name']}")
         + info.dim(f": {', '.join(ENGINE_NAME[k] for k, _, _ in todo)}; about {prof['minutes']} minutes each ({prof_name})."))
    show(info.dim("The GPU is taken for the whole run (lock " + ce.gpu_hogs.LOCK_PATH + "); close other GPU work first."))
    if sys.stdin.isatty() and sys.stderr.isatty() and not (args.yes or args.json):
        sys.stderr.write("Start? [Y/n] ")
        sys.stderr.flush()
        if sys.stdin.readline().strip().lower() in ("n", "no"):
            return 1
    interrupted = False
    for i, (key, cfg, tried) in enumerate(todo, 1):
        log = os.path.join(out_dir, "logs", f"compare-{cfg['label']}.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        name = ENGINE_NAME[key]
        head = f"[{i}/{len(todo)}] {name} ({cfg['label']})"
        last = ["", -30.0]

        def status(line, secs, head=head):
            if err_t.tty:
                txt = f"{head} {int(secs // 60)}:{int(secs % 60):02d} {line}"
                sys.stderr.write("\r\033[K" + clip(txt, err_t.width - 1))
                sys.stderr.flush()
            elif line != last[0] and secs - last[1] >= 30:
                last[0], last[1] = line, secs
                sys.stderr.write(f"{head} {int(secs // 60)}:{int(secs % 60):02d} {clip(line, 100)}\n")
        if not err_t.tty:
            sys.stderr.write(head + " ...\n")
        try:
            res = run_engine(cfg, out_dir, prof, a, log, status)
        except KeyboardInterrupt:
            interrupted = True
            break
        finally:
            if err_t.tty:
                sys.stderr.write("\r\033[K")
        if "failed" in res:
            e = {"engine": key, "name": name, "variant": cfg["label"], "display": cfg.get("display"), "version": None,
                 "model_path": cfg.get("model_path"), "status": "failed", "flags": [], "invalid": [res["failed"]],
                 "command": None}
            e.update(metrics_of({}))
        else:
            e = entry_of(key, cfg, res)
        e["log"] = log
        e["tried_first"] = tried
        doc["engines"].append(e)
        mark = err_t.green("✓") if e["status"] == "ok" else err_t.red("✗")
        sys.stderr.write(f"{mark} {name}: " + (f"{e['decode_tps']:.1f} tok/s decode" if e["status"] == "ok"
                                                else clip("; ".join(e["invalid"]), 120)) + "\n")
    doc["interrupted"] = interrupted
    json.dump(doc, open(os.path.join(out_dir, "compare.json"), "w"), indent=1)
    md = report(doc)
    open(os.path.join(out_dir, "report.md"), "w").write(md)
    if args.json:
        print(json.dumps(doc, indent=1))
    else:
        print(render(doc, out_t))
    if args.report:
        if args.report != "-":
            open(os.path.expanduser(args.report), "w").write(md)
        else:
            print("\n" + md)
    if interrupted:
        return 130
    return 0 if any(e["status"] == "ok" for e in doc["engines"]) else 1


def kapitsa_version():
    exe = os.environ.get("KAPITSA_BIN") or shutil.which("kapitsa")
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15).stdout.strip()
        return out.splitlines()[0] if out else "unknown"
    except Exception:
        return "unknown"


def harness_commit(ce):
    info = ce.bench.harness_info()
    return {"commit": (info.get("git_head") or "")[:7], "dirty": info.get("dirty"), "version": info.get("version")}


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
