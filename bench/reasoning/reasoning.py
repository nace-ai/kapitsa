#!/usr/bin/env python3
"""Math reasoning accuracy of a model on an OpenAI-compatible server (Kapitsa or any other engine).
Whether a quantization (4-bit, a 4/8-bit mix, 6-bit, 8-bit) answers competition math as well as another, thinking on.

  reasoning.py sets                                   the problem sets and their sizes
  reasoning.py fetch                                  download and verify the problems (Hugging Face, pinned)
  reasoning.py check                                  the grader on every MATH-500 reference solution (no server)
  reasoning.py run --set math500-hard --label 27b-4bit [--base URL] [--serve CMD] [--samples K] [--parallel N]
                   [--out FILE]
  reasoning.py report FILE... [--baseline LABEL] [--regrade] [--json OUT]

`run` sends each problem with Qwen's math instruction ("Please reason step by step, and put your final answer within
\\boxed{}."), thinking on, Qwen's sampling for thinking (T 0.6, top_p 0.95, top_k 20) and a per-sample seed, streams
the reply, grades its last \\boxed{} (grader.py) and appends one JSON line per sample to the results file: the reply's
text, answer, verdict, tokens and timing. A results file resumes: samples already graded are skipped. The GPU lock
(bench/harness/gpu_hogs.py, /tmp/kapitsa-gpu.lock) is held for the run unless --no-gpu-lock or NACE_GPU_LOCK_HELD=1.
`report` gives each label's accuracy with a 95% interval (bootstrap over problems), the replies cut short, without
an answer or looping, and each label against the baseline on the same problems (the paired difference, its interval
and, with one sample per problem, McNemar's exact test). Standard library only.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import queue
import random
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "harness"))
import grader  # noqa: E402
import bench  # noqa: E402  (bench/harness/bench.py: the streaming client)
import gpu_hogs  # noqa: E402  (bench/harness/gpu_hogs.py: the GPU lock)

PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
            "frequency_penalty": 0.0}

# The problems, pinned: Hugging Face dataset revisions and the files' sha256. Only the selection (sets/) is in this
# repository; the problems are downloaded on first use.
DATASETS = {
    "math500": {
        "repo": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "files": [{"name": "test.jsonl", "bytes": 446564,
                   "sha256": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"}],
        "license": "MIT: OpenAI's PRM800K subset of the MATH dataset (Hendrycks et al.)",
    },
    "aime2025": {
        "repo": "opencompass/AIME2025", "revision": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "files": [{"name": "aime2025-I.jsonl", "bytes": 5764, "prefix": "2025-I-",
                   "sha256": "b91b3c96f05d9635d2a0692b124ebe023c1ff59cb19c074275e6c4b349d0659e"},
                  {"name": "aime2025-II.jsonl", "bytes": 6285, "prefix": "2025-II-",
                   "sha256": "16a2dcfbbf9db1b11f8a69a3ba5e4cac73e3641b19a37e2307e9c12240bbed5e"}],
        "license": "the dataset card says MIT; the problems are the Mathematical Association of America's",
    },
}

# name: (dataset, the frozen selection in sets/ or None for all, default samples per problem, default max_tokens)
SETS = {
    "math500-quick": ("math500", "math500-quick.txt", 1, 32768),
    "math500-hard": ("math500", "math500-hard.txt", 1, 32768),
    "math500": ("math500", None, 1, 32768),
    "aime2025": ("aime2025", None, 4, 38912),
}
# How sets/*.txt were drawn (`sets --rebuild` redraws them): (levels, size); see select()
SELECTIONS = {"math500-quick.txt": ((5,), 50), "math500-hard.txt": ((4, 5), 150)}


def cache_dir(a=None):
    d = (getattr(a, "cache", None) or os.environ.get("KAPITSA_REASONING_CACHE")
         or os.path.expanduser("~/.cache/kapitsa/reasoning"))
    return os.path.abspath(os.path.expanduser(d))


# ---------------------------------------------------------------- problems


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_files(name, cache, download=True, log=print):
    """The dataset's files in the cache, downloaded and verified first if needed."""
    ds = DATASETS[name]
    base = os.path.join(cache, ds["repo"].replace("/", "--"), ds["revision"])
    out = []
    for f in ds["files"]:
        path = os.path.join(base, f["name"])
        if not (os.path.exists(path) and os.path.getsize(path) == f["bytes"] and _sha256(path) == f["sha256"]):
            if not download:
                raise FileNotFoundError(f"{name}: {path} missing; run `reasoning.py fetch`")
            endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
            url = f"{endpoint}/datasets/{ds['repo']}/resolve/{ds['revision']}/{f['name']}"
            log(f"downloading {url} ({f['bytes']:,} bytes)")
            os.makedirs(base, exist_ok=True)
            tmp = path + ".part"
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as w:
                w.write(r.read())
            got = _sha256(tmp)
            if got != f["sha256"]:
                os.remove(tmp)
                raise RuntimeError(f"{url}: sha256 {got}, expected {f['sha256']}")
            os.replace(tmp, path)
        out.append((f, path))
    return out


def load_dataset(name, cache, download=True, log=print):
    """Problems as {"id", "problem", "answer", "level", "subject"}, in the dataset's order."""
    rows = []
    for f, path in dataset_files(name, cache, download, log):
        with open(path, encoding="utf-8") as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        for k, r in enumerate(lines, 1):
            if name == "math500":
                rows.append({"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"],
                             "level": r["level"], "subject": r["subject"], "solution": r["solution"]})
            else:
                rows.append({"id": f["prefix"] + str(k), "problem": r["question"], "answer": str(r["answer"]),
                             "level": None, "subject": None})
    return rows


def select(rows, levels, n):
    """n problems of the given levels, spread over the subjects in proportion to their share (largest remainder),
    the first of each subject in sha256(id) order: a fixed draw that anyone can repeat."""
    pool = [r for r in rows if r["level"] in levels]
    groups = {}
    for r in pool:
        groups.setdefault(r["subject"], []).append(r)
    exact = {s: n * len(g) / len(pool) for s, g in groups.items()}
    quota = {s: int(math.floor(v)) for s, v in exact.items()}
    for s in sorted(exact, key=lambda s: (-(exact[s] - quota[s]), s))[:n - sum(quota.values())]:
        quota[s] += 1
    chosen = set()
    for s, g in groups.items():
        g = sorted(g, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        chosen.update(r["id"] for r in g[:quota[s]])
    return [r["id"] for r in rows if r["id"] in chosen]


def read_selection(fname):
    with open(os.path.join(HERE, "sets", fname), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def load_set(set_name, cache, download=True, log=print, problems=None):
    """The set's problems. problems: a JSONL file of {"id", "problem", "answer"} instead of a named set."""
    if problems:
        with open(problems, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        for r in rows:
            r.setdefault("level", None)
            r.setdefault("subject", None)
        return rows
    ds, fname, _, _ = SETS[set_name]
    rows = load_dataset(ds, cache, download, log)
    if fname is None:
        return rows
    ids = read_selection(fname)
    by_id = {r["id"]: r for r in rows}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(f"{fname}: {len(missing)} ids not in {ds} (first: {missing[0]})")
    return [by_id[i] for i in ids]


# ---------------------------------------------------------------- run


def seed_of(set_name, pid, k):
    return int(hashlib.sha256(f"{set_name}:{pid}:{k}".encode()).hexdigest()[:8], 16) % (1 << 31)


def server_info(base):
    """The server's /v1/models entry (Kapitsa's: id, context_length, context_fit, slots); {} when it has none."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=30) as r:
            data = json.load(r).get("data") or []
        return data[0] if data else {}
    except Exception as e:  # noqa: BLE001 (any engine: a missing or odd /models only loses provenance)
        return {"error": str(e)}


def machine():
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    mem = sh("sysctl", "-n", "hw.memsize")
    return {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "model": sh("sysctl", "-n", "hw.model"),
            "memory_gb": round(int(mem) / 2**30) if mem and mem.isdigit() else None,
            "macos": sh("sw_vers", "-productVersion")}


def done_keys(path):
    """(id, sample) of the samples a results file already graded (errors are retried)."""
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # a line cut by an interrupted write
                if r.get("type") == "sample" and not r.get("error"):
                    keys.add((r["id"], r["sample"]))
    return keys


def request_body(model, problem, seed, max_tokens):
    return {"model": model, "messages": [{"role": "user", "content": PROMPT.format(problem=problem)}],
            "max_tokens": max_tokens, "seed": seed, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": True}}


def solve(base, model, set_name, item, k, max_tokens, timeout):
    seed = seed_of(set_name, item["id"], k)
    rec = {"type": "sample", "set": set_name, "id": item["id"], "sample": k, "seed": seed,
           "level": item.get("level"), "subject": item.get("subject"), "gold": item["answer"]}
    try:
        r = bench.stream_chat(base, request_body(model, item["problem"], seed, max_tokens), timeout=timeout)
    except Exception as e:  # noqa: BLE001 (HTTP errors, timeouts, a server that went away: retried on resume)
        rec.update(error=str(e), correct=False)
        return rec
    usage = r.get("usage") or {}
    content, reasoning = r.get("content") or "", r.get("reasoning") or ""
    gen = r.get("gen_time")
    ct = usage.get("completion_tokens")
    rec.update(finish=r.get("finish"), prompt_tokens=usage.get("prompt_tokens"), completion_tokens=ct,
               ttft=r.get("ttft"), gen_time=gen, total=r.get("total"), tok_s=round(ct / gen, 2) if ct and gen else None,
               distinct4=bench.distinct4(reasoning + "\n" + content), content=content, reasoning=reasoning)
    return regrade(rec)


def cmd_run(a):
    cache = cache_dir(a)
    set_name = a.set if not a.problems else (a.set or os.path.splitext(os.path.basename(a.problems))[0])
    items = load_set(set_name, cache, problems=a.problems)
    if a.limit:
        items = items[:a.limit]
    _, _, samples_default, mt_default = SETS.get(set_name, (None, None, 1, 32768))
    samples = a.samples or samples_default
    held = os.environ.get("NACE_GPU_LOCK_HELD") or os.environ.get("NI_LOCK_HELD")
    lock = server = None
    try:
        # the lock before the server: loading the model is GPU work too (the harness holds it from server start)
        if not (a.no_gpu_lock or held):
            lock = gpu_hogs.GpuLock(log=lambda m: print("  " + m, flush=True)).acquire()
        if a.serve:
            server = start_server(a)
        return run_set(a, set_name, items, samples, mt_default, lock, held)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    finally:
        if server:
            stop_server(server)
        if lock:
            lock.release()


def start_server(a):
    """`--serve`: the server command, started in a session of its own (its log beside the results), once it answers
    /models: its model loaded."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log = (a.out or os.path.join(HERE, "runs", f"serve-{stamp}")) + ".server.log"
    os.makedirs(os.path.dirname(os.path.abspath(log)), exist_ok=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(shlex.split(a.serve), stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(f"started `{a.serve}` (pid {proc.pid}, log {log})", flush=True)
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"the server exited with code {proc.returncode} before it answered (see {log})")
        try:
            with urllib.request.urlopen(a.base.rstrip("/") + "/models", timeout=5) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001 (not listening yet, or still loading)
            pass
        if time.time() - t0 > a.serve_timeout:
            stop_server(proc)
            raise RuntimeError(f"the server did not answer {a.base}/models within {a.serve_timeout:.0f} s (see {log})")
        time.sleep(2)
    print(f"server ready after {time.time() - t0:.0f} s", flush=True)
    return proc


def stop_server(proc):
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 30)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_set(a, set_name, items, samples, mt_default, lock, held):
    info = server_info(a.base)
    model = a.model or info.get("id") or "default"
    label = a.label or model
    max_tokens = a.max_tokens or mt_default
    ctx = info.get("context_length")
    if isinstance(ctx, int) and max_tokens + 2048 > ctx:
        capped = max(1024, ctx - 2048)
        print(f"note: the server holds {ctx} tokens per conversation: max_tokens {max_tokens} -> {capped} "
              f"(start it with a larger --ctx for the full budget)", flush=True)
        max_tokens = capped
    slots = info.get("slots")
    if isinstance(slots, int) and a.parallel > slots:
        print(f"note: --parallel {a.parallel} but the server runs {slots} requests at once", flush=True)
    out = a.out or os.path.join(HERE, "runs", f"{label}-{set_name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    done = done_keys(out)
    todo = [(it, k) for it in items for k in range(samples) if (it["id"], k) not in done]
    print(f"{set_name}: {len(items)} problems x {samples} samples, {len(done)} already graded, {len(todo)} to run; "
          f"{label} at {a.base}, max_tokens {max_tokens}, parallel {a.parallel} -> {out}", flush=True)
    if not todo:
        return 0
    header = {"type": "run", "started": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "label": label, "set": set_name, "problems": len(items), "samples": samples, "base": a.base,
              "server": info, "serve": a.serve, "request": {k: v for k, v in request_body(model, "", 0, max_tokens).items()
                                                            if k not in ("messages", "seed")},
              "prompt": PROMPT, "parallel": a.parallel, "harness": bench.harness_info(), "machine": machine(),
              "dataset": None if a.problems else {k: DATASETS[SETS[set_name][0]][k] for k in ("repo", "revision")},
              "gpu_lock": "held by the caller" if held else (gpu_hogs.LOCK_PATH if lock else "none")}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
    right = graded = errors = 0
    t0 = time.time()
    jobs, results = queue.Queue(), queue.Queue()
    for job in todo:
        jobs.put(job)

    def worker():
        while True:
            try:
                it, k = jobs.get_nowait()
            except queue.Empty:
                return
            results.put(solve(a.base, model, set_name, it, k, max_tokens, a.timeout))

    # daemon threads: Ctrl+C stops the run at once (a reply in flight is dropped, and redone on resume)
    for _ in range(max(1, a.parallel)):
        threading.Thread(target=worker, daemon=True).start()
    for n in range(1, len(todo) + 1):
        while True:
            try:
                rec = results.get(timeout=1)
                break
            except queue.Empty:
                continue
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if rec.get("error"):
            errors += 1
            verdict = "error: " + rec["error"][:120]
        else:
            graded += 1
            right += bool(rec["correct"])
            verdict = (("right" if rec["correct"] else "wrong") + f" ({rec.get('answer')!s:.40})"
                       + (", cut short" if rec.get("finish") == "length" else "")
                       + (f", {rec['completion_tokens']} tokens" if rec.get("completion_tokens") else "")
                       + (f", {rec['tok_s']} tok/s" if rec.get("tok_s") else ""))
        print(f"[{n}/{len(todo)}] {rec['id']} #{rec['sample']}: {verdict}; "
              f"{right}/{graded} right so far, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {right}/{graded} right" + (f", {errors} errors (rerun to retry them)" if errors else ""), flush=True)
    return 1 if errors else 0


# ---------------------------------------------------------------- report


def regrade(r):
    """A sample's verdict from its reply with the grader as it is now: run() grades this way, and report --regrade
    again. A reply cut short is graded on what it answered, never on a box its unfinished thinking happened to write."""
    cut = r.get("finish") == "length"
    g = grader.grade(r.get("content"), r["gold"], None if cut else r.get("reasoning"))
    return dict(r, answer=g["answer"], correct=g["correct"], source=g["source"])


def load_results(paths, rename=None, regraded=False):
    """{(label, set): {id: [sample records]}} from results files; a file's label comes from its run header.
    regraded: every sample graded again from its saved reply."""
    runs = {}
    for path in paths:
        label = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "run":
                    label = (rename or {}).get(r["label"], r["label"])
                elif r.get("type") == "sample" and not r.get("error"):
                    if label is None:
                        raise ValueError(f"{path}: samples before a run header")
                    if regraded:
                        r = regrade(r)
                    per = runs.setdefault((label, r["set"]), {}).setdefault(r["id"], {})
                    per[r["sample"]] = r  # a sample graded twice (two runs into one file): the later counts
    return {k: {pid: [s[i] for i in sorted(s)] for pid, s in v.items()} for k, v in runs.items()}


def _mean_ci(values, reps=2000, seed=0):
    """Mean of per-problem values and its 95% bootstrap interval over problems."""
    if not values:
        return None, None, None
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, m, m
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(reps))
    return m, boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]


def _mcnemar(b, c):
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(per):
    acc = [sum(bool(s["correct"]) for s in v) / len(v) for v in per.values()]
    samples = [s for v in per.values() for s in v]
    m, lo, hi = _mean_ci(acc)
    tok = [s["completion_tokens"] for s in samples if s.get("completion_tokens")]
    rate = [s["tok_s"] for s in samples if s.get("tok_s")]
    loops = [s for s in samples if s.get("distinct4") is not None and s["distinct4"] < 0.5]
    levels = {}
    for pid, v in per.items():
        lv = v[0].get("level")
        if lv is not None:
            levels.setdefault(lv, []).append(sum(bool(s["correct"]) for s in v) / len(v))
    return {"problems": len(per), "samples": len(samples), "accuracy": m, "ci95": [lo, hi],
            "cut_short": sum(s.get("finish") == "length" for s in samples),
            "no_answer": sum(s.get("answer") is None for s in samples),
            "answered_in_thinking": sum(s.get("source") == "reasoning" for s in samples),
            "looping": len(loops),
            "completion_tokens_median": round(statistics.median(tok)) if tok else None,
            "tok_s_median": round(statistics.median(rate), 1) if rate else None,
            "by_level": {str(k): sum(v) / len(v) for k, v in sorted(levels.items())}}


def paired(base, other):
    """other against base on the problems both ran: the difference of per-problem accuracy, its 95% bootstrap
    interval, problems each got right more often, and McNemar's p when every problem has one sample."""
    ids = sorted(set(base) & set(other))
    if not ids:
        return None
    a = [sum(bool(s["correct"]) for s in base[i]) / len(base[i]) for i in ids]
    b = [sum(bool(s["correct"]) for s in other[i]) / len(other[i]) for i in ids]
    d = [y - x for x, y in zip(a, b)]
    m, lo, hi = _mean_ci(d, seed=1)
    out = {"problems": len(ids), "difference": m, "ci95": [lo, hi], "better": sum(x > 0 for x in d),
           "worse": sum(x < 0 for x in d)}
    if all(len(base[i]) == 1 and len(other[i]) == 1 for i in ids):
        out["mcnemar_p"] = _mcnemar(out["worse"], out["better"])
    return out


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def cmd_report(a):
    rename = dict(x.split("=", 1) for x in a.rename) if a.rename else None
    runs = load_results(a.files, rename, regraded=a.regrade)
    if not runs:
        print("no graded samples")
        return 1
    result = {"labels": {}, "paired": {}}
    for set_name in sorted({s for _, s in runs}):
        labels = [lab for lab, s in runs if s == set_name]
        base = a.baseline if a.baseline in labels else labels[0]
        print(f"\n### {set_name}\n")
        print("| label | problems | samples | accuracy (95% CI) | cut short | no answer | looping | tokens (median) "
              "| tok/s (median) |")
        print("| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
        for lab in labels:
            s = summarize(runs[(lab, set_name)])
            result["labels"].setdefault(set_name, {})[lab] = s
            lo, hi = s["ci95"]
            print(f"| {lab} | {s['problems']} | {s['samples']} | {_pct(s['accuracy'])} ({_pct(lo)}-{_pct(hi)}) "
                  f"| {s['cut_short']} | {s['no_answer']} | {s['looping']} | {s['completion_tokens_median'] or '-'} "
                  f"| {s['tok_s_median'] or '-'} |")
        levels = sorted({lv for lab in labels for lv in result["labels"][set_name][lab]["by_level"]})
        if levels:
            print("\n| label | " + " | ".join(f"level {lv}" for lv in levels) + " |")
            print("| --- |" + " ---: |" * len(levels))
            for lab in labels:
                bl = result["labels"][set_name][lab]["by_level"]
                print(f"| {lab} | " + " | ".join(_pct(bl.get(lv)) for lv in levels) + " |")
        others = [lab for lab in labels if lab != base]
        if others:
            print(f"\nAgainst {base}, on the problems both ran:\n")
            print("| label | problems | difference (95% CI) | better on | worse on | McNemar p |")
            print("| --- | ---: | --- | ---: | ---: | ---: |")
            for lab in others:
                p = paired(runs[(base, set_name)], runs[(lab, set_name)])
                if p is None:
                    continue
                result["paired"].setdefault(set_name, {})[lab] = dict(p, baseline=base)
                lo, hi = p["ci95"]
                mp = p.get("mcnemar_p")
                print(f"| {lab} | {p['problems']} | {100 * p['difference']:+.1f} pts ({100 * lo:+.1f} to "
                      f"{100 * hi:+.1f}) | {p['better']} | {p['worse']} | {'-' if mp is None else f'{mp:.3f}'} |")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


# ---------------------------------------------------------------- sets, fetch, check


def cmd_sets(a):
    cache = cache_dir(a)
    for name, (ds, fname, samples, mt) in SETS.items():
        size = len(read_selection(fname)) if fname else {"math500": 500, "aime2025": 30}[ds]
        print(f"{name:14s} {size:4d} problems of {ds}, {samples} sample(s) each by default, max_tokens {mt}"
              + (f" (sets/{fname})" if fname else ""))
    if not a.rebuild:
        return 0
    rows = load_dataset("math500", cache)
    bad = 0
    for fname, (levels, n) in SELECTIONS.items():
        ids = select(rows, levels, n)
        same = ids == read_selection(fname)
        bad += not same
        print(f"sets/{fname}: {'matches' if same else 'DIFFERS from'} the draw (levels {levels}, {n})")
        if a.write and not same:
            head = (f"# {fname[:-4]}: {n} MATH-500 problems of level {'/'.join(map(str, levels))}, spread over the "
                    f"subjects in proportion, the first of each in sha256(id) order (reasoning.py select)\n")
            with open(os.path.join(HERE, "sets", fname), "w", encoding="utf-8") as f:
                f.write(head + "".join(i + "\n" for i in ids))
            print("  written")
    return 1 if bad and not a.write else 0


def cmd_fetch(a):
    cache = cache_dir(a)
    for name in DATASETS:
        for f, path in dataset_files(name, cache):
            print(f"{name}: {path} ({f['bytes']:,} bytes, sha256 ok)")
    return 0


def cmd_check(a):
    """The grader on what it must get right: every MATH-500 reference solution's last box against its answer, and
    every AIME 2025 answer read as the integer it is."""
    cache = cache_dir(a)
    miss = 0
    rows = load_dataset("math500", cache)
    for r in rows:
        if not grader.equal(grader.extract(r["solution"]), r["answer"]):
            miss += 1
            print(f"MISS {r['id']}: {grader.extract(r['solution'])!r} vs {r['answer']!r}")
    print(f"MATH-500: {len(rows) - miss}/{len(rows)} reference solutions graded right")
    aime = load_dataset("aime2025", cache)

    def integer(s):
        try:
            return str(int(grader.normalize(s)))
        except ValueError:
            return None
    bad = [r["id"] for r in aime if integer(r["answer"]) is None or not grader.equal(integer(r["answer"]), r["answer"])]
    print(f"AIME 2025: {len(aime) - len(bad)}/{len(aime)} answers read as integers" + (f" (not: {bad})" if bad else ""))
    return 1 if miss or bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", help="where the problems are kept (default ~/.cache/kapitsa/reasoning, or "
                                    "$KAPITSA_REASONING_CACHE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sets", help="the problem sets")
    s.add_argument("--rebuild", action="store_true", help="redraw sets/*.txt from MATH-500 and compare")
    s.add_argument("--write", action="store_true", help="with --rebuild: write the redrawn lists")
    sub.add_parser("fetch", help="download and verify the problems")
    sub.add_parser("check", help="the grader on the reference solutions (no server)")
    r = sub.add_parser("run", help="run a set against a server")
    r.add_argument("--set", choices=sorted(SETS), help="the problem set (or --problems)")
    r.add_argument("--problems", help="a JSONL file of {id, problem, answer} instead of a named set")
    r.add_argument("--base", default="http://127.0.0.1:8000/v1", help="the server's OpenAI base URL")
    r.add_argument("--model", help="the request's model field (default: the server's first model)")
    r.add_argument("--label", help="what the report calls this run (default: the model)")
    r.add_argument("--samples", type=int, help="samples per problem (default: the set's)")
    r.add_argument("--max-tokens", type=int, help="per reply, thinking included (default: the set's)")
    r.add_argument("--parallel", type=int, default=1, help="requests at once (Kapitsa: up to its --slots)")
    r.add_argument("--timeout", type=float, default=1800, help="seconds without a byte before a request fails")
    r.add_argument("--limit", type=int, help="only the first N problems (a smoke test)")
    r.add_argument("--out", help="the results file, appended to and resumed (default: runs/<label>-<set>-<time>.jsonl)")
    r.add_argument("--no-gpu-lock", action="store_true", help="do not take the GPU lock")
    r.add_argument("--serve", help="start this server command for the run, under the GPU lock, and stop it after "
                                   "(e.g. \"kapitsa serve --model 27b --ctx 40960 --port 8199\" with --base "
                                   "http://127.0.0.1:8199/v1)")
    r.add_argument("--serve-timeout", type=float, default=3600, help="seconds for --serve's server to answer")
    p = sub.add_parser("report", help="accuracy per label and against a baseline")
    p.add_argument("files", nargs="+")
    p.add_argument("--baseline", help="the label the others are compared with (default: the first file's)")
    p.add_argument("--rename", nargs="*", help="OLD=NEW labels")
    p.add_argument("--regrade", action="store_true", help="grade the saved replies again with the current grader")
    p.add_argument("--json", help="also write the numbers here")
    a = ap.parse_args(argv)
    if a.cmd == "run" and not (a.set or a.problems):
        ap.error("run needs --set or --problems")
    return {"sets": cmd_sets, "fetch": cmd_fetch, "check": cmd_check, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
