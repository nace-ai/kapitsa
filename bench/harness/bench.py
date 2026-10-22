#!/usr/bin/env python3
"""Engine-agnostic benchmark for an OpenAI-compatible local LLM server (one engine, server already up).

Measures, through the server's /v1/chat/completions streaming API (what coding agents use):
  * decode tok/s, sampled  (LRU-cache prompt, T=0.6/top_p 0.95/top_k 20, thinking off, N runs, median)
  * decode tok/s, greedy   (same prompt at T=0; same weights -> nearly the same text on every engine)
  * prefill tok/s + TTFT   (cold ~4k and ~16k token prompts; cache-busting nonce after the system tag)
  * cached TTFT            (agent pattern: 3 append-only follow-up turns on the same prefix, median)
  * tool calling           (streaming + non-streaming, tools array, tool-result round trip)
  * agent session          (agent_replay.py: a recorded opencode session, agent-traces/*.json, replayed request
                            by request, greedy, max_tokens = each recorded answer's length; per-turn TTFT,
                            writing rate and time, session totals)
  * memory                 (footprint phys_footprint / peak / resident mapped files of the server
                            process tree at the end of every phase, plus swap)

Fairness rules built in (see README.md, "What makes a run count"):
  * thinking is turned off with BOTH chat_template_kwargs.enable_thinking=false and
    reasoning_effort="none" (Ollama and LM Studio only honour the latter, MTPLX and mlx-lm only the
    former); a run that returns reasoning text, or content with <think>/</think>, is INVALID
  * every request sends the same sampling parameters, including the neutral ones engines default
    differently (min_p 0: llama.cpp defaults to 0.05; presence_penalty 0: Ollama's qwen3.6 Modelfile
    has 1.5; repeat_penalty 1.0, the llama.cpp key). `repetition_penalty` is never sent: mlx-lm builds
    a logits processor for any non-zero value. --sampling-overrides adjusts this per engine.
  * decode rate = middle window of the streamed output (first ~16 and last ~4 tokens dropped), so
    engines that pack several tokens into one SSE event (oMLX ~10) are not flattered; the old
    end-to-end (tokens-1)/gen_time is reported too. Median over runs, with min/max.
  * finish_reason must be 'length' and completion_tokens == max_tokens (else flagged); the full text
    of every run is saved and its distinct-4-gram ratio flags repetition loops (they inflate
    speculative decoding).
  * prompts are cut from a frozen corpus file (corpus/bench_corpus.txt, sha256 recorded) so every
    engine sees byte-identical prompts; cold prompt_tokens must be within ±1% of the reference count
    and the engine must not report reused cache tokens on the cold request.
  * the GPU lock (gpu_hogs.LOCK_PATH, default ~/.cache/nace-gpu.lock) is held for the whole run unless
    the caller already holds it (compare_engines.py sets NACE_GPU_LOCK_HELD=1).

Stdlib only, so it runs with any python3 (even macOS's system 3.9).

usage:
  python3 bench.py --base-url http://127.0.0.1:8080/v1 --model NAME --label engine-model \
      --pid <server pid> [--decode-tokens 512] [--decode-runs 5] [--sizes 4000,16000] [--out x.json]
"""
import argparse, hashlib, json, os, random, re, statistics, subprocess, sys, threading, time
import urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gpu_hogs  # noqa: E402

SCHEMA = 2
CORPUS_FILE = os.path.join(HERE, "corpus", "bench_corpus.txt")
CORPUS_SHA256 = "1a9bb103b3f6d39ee5b9f5bf7207e304c5ce92788c273685f597508c7d2bde28"  # corpus/NOTICE.md
CHARS_PER_TOKEN = 4.25          # build_context() heuristic: target tokens -> characters cut from the corpus
CORPUS_TOKENS_MAX = 34000       # frozen corpus length in "heuristic tokens" (~144.5k chars)
WARM_TOKENS = 2700              # warm-up prompt ~2.9k tokens: > one 2048-token prefill chunk everywhere
WARM_OFFSET_CHARS = 100000      # warm-up text comes from a part of the corpus no timed prompt uses
COLD_CACHE_MAX = 16             # a cold prompt shares only "<|im_start|>system\n[session " (~5 tokens)
PROMPT_TOL = 0.01               # ±1% around the reference prompt_tokens
# usage.prompt_tokens of the cold prefill prompts (frozen corpus, official Qwen3.6 / Qwen3.8 chat templates,
# thinking off), keyed by corpus sha256. Counted with the Hugging Face tokenizers of Qwen3.6-35B-A3B and
# Qwen3.8-27B (transformers apply_chat_template, 8 nonces each: 35B 4130-4135 / 16284-16288, 27B 4130-4135 /
# 16285-16287). For another corpus pass --ref-prompt-tokens, or the check is reported as unchecked.
REF_PROMPT_TOKENS = {
    CORPUS_SHA256: {4000: 4133, 16000: 16286},
}

DECODE_USER = ("Write a complete, well-commented Python module that implements an LRU cache with TTL expiry, "
               "a thread-safe wrapper, and a pytest test suite. Write at least 600 lines. Do not stop early.")
DECODE_SYS = "You are a helpful coding assistant. Answer with code only."
FOLLOWUPS = ["Now name one function defined in that file.",
             "Name one class defined in that file, or say there is none.",
             "Which modules does that file import?"]


LOCAL_ENV = os.path.join(HERE, "local.env")  # this Mac's paths (git-ignored; see local.env.example)


def load_env_file(path):
    """KEY=VALUE lines (# comments, optional `export `, optional quotes; a leading ~ is expanded; an unquoted
    value ends at " #").
    Variables already set in the environment win. Returns the keys it set."""
    keys = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            k, eq, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if line.startswith("#") or not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
                continue
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            else:
                v = re.split(r"\s+#", v, maxsplit=1)[0]  # an unquoted value ends at " #" (a comment)
            if not os.environ.get(k):
                os.environ[k] = os.path.expanduser(v) if v.startswith("~") else v
                keys.append(k)
    gpu_hogs.configure()
    return keys


def out_dir():
    """Where runs go: $BENCH_OUT, else bench/harness/runs (git-ignored)."""
    return os.path.abspath(os.path.expanduser(os.environ.get("BENCH_OUT") or os.path.join(HERE, "runs")))


def harness_info():
    """The harness's own version: git HEAD of its checkout and whether bench/harness has local changes. A release
    ships a copy with no checkout (packaging/release/build.sh), which says which release it came with in a VERSION
    file instead, so a report made from an install names its harness too."""
    def git(*a):
        try:
            return subprocess.run(["git", "-C", HERE] + list(a), capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    head = git("rev-parse", "HEAD")
    if head:
        return {"git_head": head, "dirty": bool(git("status", "--porcelain", "--", "."))}
    try:
        with open(os.path.join(HERE, "VERSION")) as f:
            version = f.read().strip()
    except OSError:
        version = ""
    return {"git_head": None, "dirty": None, "version": version or None}


# ---------------------------------------------------------------- memory
def _tree(pid):
    return sorted(gpu_hogs.proc_tree([pid]))


def _rss_mb(pids):
    out = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True).stdout
    return sum(int(x) for x in out.split() if x.strip().isdigit()) / 1024.0


def _footprint_now_mb(pids):
    """Current phys_footprint of the pids, in MB (one `footprint -f bytes` call, byte precision)."""
    args = ["footprint", "-f", "bytes", "--noCategories"]
    for p in pids:
        args += ["-p", str(p)]
    out = subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    return sum(int(x) for x in re.findall(r"phys_footprint:\s*(\d+)\s*B", out)) / 2**20


def footprint_snapshot(pids):
    """phys_footprint, phys_footprint_peak and the resident mapped-file bytes of every pid.
    phys_footprint misses mmap'd weights (clean file-backed pages): llama.cpp without -lm none and
    Ollama's GGUF runner show them in mapped_file_clean instead, which is kept separately."""
    per = []
    for p in pids:
        try:
            out = subprocess.run(["footprint", "-f", "bytes", "-w", "-p", str(p)], capture_output=True, text=True,
                                 timeout=120).stdout
        except Exception:
            continue
        m1 = re.search(r"phys_footprint:\s*(\d+)\s*B", out)
        if not m1:
            continue
        m2 = re.search(r"phys_footprint_peak:\s*(\d+)\s*B", out)
        mf = re.search(r"^\s*(\d+) B\s+(\d+) B\s+(\d+) B\s+(\d+) B\s+(\d+) B\s+\d+\s+mapped file\s*$", out, re.M)
        nm = re.search(r"^(.*?) \[(\d+)\]:", out, re.M)
        per.append({"pid": p, "name": nm.group(1).strip() if nm else "?", "phys_footprint": int(m1.group(1)),
                    "phys_footprint_peak": int(m2.group(1)) if m2 else None,
                    "mapped_file_dirty": int(mf.group(1)) if mf else 0,
                    "mapped_file_clean": int(mf.group(3)) if mf else 0,
                    "mapped_file_wired": int(mf.group(5)) if mf else 0})
    mb = lambda k: round(sum((x.get(k) or 0) for x in per) / 2**20, 1)
    return {"pids": per, "phys_footprint_mb": mb("phys_footprint"), "phys_footprint_peak_mb": mb("phys_footprint_peak"),
            "mapped_file_clean_mb": mb("mapped_file_clean")}


def vm_snapshot():
    d = {}
    try:
        s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"used = ([\d.]+)([MG])", s)
        if m:
            d["swap_used_mb"] = float(m.group(1)) * (1024 if m.group(2) == "G" else 1)
        v = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
        for k, key in (("Swapouts", "swapouts"), ("Swapins", "swapins"), ("Pageouts", "pageouts")):
            m = re.search(rf"^{k}:\s+(\d+)", v, re.M)
            if m:
                d[key] = int(m.group(1))
        lvl = subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=10).stdout
        if lvl.strip().isdigit():
            d["memorystatus_level"] = int(lvl)
    except Exception as e:
        d["error"] = repr(e)[:200]
    return d


class MemSampler(threading.Thread):
    """Background sampler of RSS and current phys_footprint of the server process tree (2 s)."""

    def __init__(self, pid, interval=2.0):
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_rss = self.peak_fp = 0.0
        self.stop_ev = threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            try:
                pids = _tree(self.pid)
                self.peak_rss = max(self.peak_rss, _rss_mb(pids))
                self.peak_fp = max(self.peak_fp, _footprint_now_mb(pids))
            except Exception:
                pass
            self.stop_ev.wait(self.interval)

# ---------------------------------------------------------------- prompts
_CORPUS = None
_CORPUS_SHA = None


def set_corpus(path):
    global CORPUS_FILE, _CORPUS, _CORPUS_SHA
    CORPUS_FILE, _CORPUS, _CORPUS_SHA = os.path.abspath(os.path.expanduser(path)), None, None


def load_corpus():
    """The frozen prompt corpus (corpus/bench_corpus.txt). Prompts are cut from it by character offset, so the
    same file gives every engine byte-identical prompts; its sha256 keys the reference prompt_tokens."""
    global _CORPUS, _CORPUS_SHA
    if _CORPUS is None:
        if not os.path.exists(CORPUS_FILE):
            raise SystemExit(f"no prompt corpus at {CORPUS_FILE} (the frozen one is corpus/bench_corpus.txt)")
        with open(CORPUS_FILE, encoding="utf-8", newline="") as fh:
            _CORPUS = fh.read()
        _CORPUS_SHA = hashlib.sha256(_CORPUS.encode("utf-8")).hexdigest()
    return _CORPUS


def corpus_sha():
    load_corpus()
    return _CORPUS_SHA


def corpus_record():
    p = CORPUS_FILE
    rel = os.path.relpath(p, HERE)
    return {"path": p if rel.startswith("..") else rel, "sha256": corpus_sha(),
            "frozen": corpus_sha() == CORPUS_SHA256}


def build_context(target_tokens, chars_per_token=CHARS_PER_TOKEN, offset=0):
    want = int(target_tokens * chars_per_token)
    c = load_corpus()
    if offset + want > len(c):
        raise ValueError(f"corpus has {len(c)} chars; {target_tokens} tokens at offset {offset} need {offset + want}")
    return c[offset:offset + want]


SYS = ("You are Nace, a local coding agent. You help the user with their repository. "
       "Be precise. Use tools when you need to read files or run commands.")

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the user's workspace and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "start_line": {"type": "integer", "description": "Optional first line (1-based)"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command in the workspace and return stdout/stderr.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout_s": {"type": "integer"}},
            "required": ["command"]}}},
]


def prefill_messages(ctx, nonce, question):
    # the nonce sits right after "<|im_start|>system\n": no engine can reuse more than ~5 tokens
    return [{"role": "system", "content": f"[{nonce}] {SYS}\n\n# Repository context\n{ctx}"},
            {"role": "user", "content": question}]


def decode_messages():
    return [{"role": "system", "content": DECODE_SYS},
            {"role": "user", "content": f"[run {random.random():.6f}] {DECODE_USER}"}]

# ---------------------------------------------------------------- HTTP
def post(base, body, stream, timeout=1800, session=None):
    h = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    if session:  # stable per-conversation id; engines like MTPLX key their KV session bank on it
        h["X-Session-Id"] = session
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), headers=h)
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        try:
            msg = e.read().decode("utf-8", "ignore")[:500]
        except Exception:
            msg = ""
        raise RuntimeError(f"HTTP {e.code}: {msg}") from None


def stream_chat(base, body, session=None, timeout=1800):
    """Streams one chat completion. Returns ttft/gen_time/total (s), usage, timings (llama.cpp and
    Ollama send one), content, reasoning, tool_calls, n_chunks (content-bearing events), finish, and
    the per-event arrival times (ev_t, s since request), character counts (ev_c) and kinds (ev_k: "c" content,
    "r" reasoning, "t" tool call; an event can carry several)."""
    body = dict(body)
    body["stream"] = True
    body.setdefault("stream_options", {"include_usage": True})
    t0 = time.perf_counter()
    r = post(base, body, True, session=session, timeout=timeout)
    first = last = usage = timings = finish = None
    content, reasoning, tcs, ev_t, ev_c, ev_k = [], [], {}, [], [], []
    n = 0
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            ev = json.loads(data)
        except Exception:
            continue
        if ev.get("usage"):
            usage = ev["usage"]
        if ev.get("timings"):
            timings = ev["timings"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            got, nch, kind = False, 0, ""
            if d.get("content"):
                content.append(d["content"]); nch += len(d["content"]); got = True; kind += "c"
            rc = d.get("reasoning_content") or d.get("reasoning")
            if rc:
                reasoning.append(rc); nch += len(rc); got = True; kind += "r"
            for tc in d.get("tool_calls") or []:
                got = True
                kind += "" if "t" in kind else "t"
                i = tc.get("index", 0)
                e = tcs.setdefault(i, {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    e["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    e["name"] += fn["name"]; nch += len(fn["name"])
                if fn.get("arguments"):
                    s = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
                    e["arguments"] += s; nch += len(s)
            if got:
                n += 1
                now = time.perf_counter()
                if first is None:
                    first = now
                last = now
                ev_t.append(now - t0)
                ev_c.append(nch)
                ev_k.append(kind)
    t_end = time.perf_counter()
    return {"ttft": (first - t0) if first else None, "gen_time": (last - first) if first and last else None,
            "total": t_end - t0, "usage": usage, "timings": timings, "content": "".join(content),
            "reasoning": "".join(reasoning), "tool_calls": [tcs[k] for k in sorted(tcs)], "n_chunks": n,
            "finish": finish, "ev_t": ev_t, "ev_c": ev_c, "ev_k": ev_k}

# ---------------------------------------------------------------- request parameters
def _overrides(a):
    if not getattr(a, "_ovr", None):
        a._ovr = json.loads(a.sampling_overrides) if a.sampling_overrides else {}
    return a._ovr


def _apply(s, ovr):
    for k, v in ovr.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    return s


def sampling(a, greedy=False):
    """Every sampling field explicit, so no engine falls back to a hidden default. greedy: T=0 plus
    top_k 1 / top_p 1, which is greedy even on a server that treats temperature 0 as 'unset'."""
    s = {"temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p,
         "presence_penalty": a.presence_penalty, "frequency_penalty": a.frequency_penalty,
         "repeat_penalty": a.repeat_penalty}
    s = _apply(s, _overrides(a))
    if greedy:  # an override may drop a field the server rejects, but never undo greedy
        for k, v in (("temperature", 0.0), ("top_p", 1.0), ("top_k", 1)):
            if k in s or k == "temperature":
                s[k] = v
    s.pop("repetition_penalty", None)  # never: mlx-lm adds a logits processor for any non-zero value
    return s


def no_think(a):
    st = a.no_think_style
    if st == "both":
        return {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    if st == "kwargs":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if st == "effort":
        return {"reasoning_effort": "none"}
    return {}


def think_on(a):
    d = {"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else {}
    if a.no_think_style in ("both", "kwargs"):
        d["chat_template_kwargs"] = {"enable_thinking": True}
    return d


def body_of(a, msgs, max_tokens, greedy=False, think=False, **extra):
    b = {"model": a.model, "messages": msgs, "max_tokens": max_tokens, **sampling(a, greedy),
         **(think_on(a) if think else no_think(a)), **extra}
    if a.extra_body:
        b.update(json.loads(a.extra_body))
    return b

# ---------------------------------------------------------------- checks
def thinking_leak(r):
    """Reasons a non-thinking request actually thought (INVALID)."""
    out = []
    if (r.get("reasoning") or "").strip():
        out.append(f"reasoning text returned ({len(r['reasoning'])} chars)")
    c = (r.get("content") or "").lstrip()
    if c.startswith("<think>"):
        out.append("content starts with <think>")
    elif "</think>" in (r.get("content") or ""):
        out.append("content contains </think>")
    return out


def distinct4(text):
    w = re.findall(r"\w+|[^\w\s]", text or "")
    if len(w) < 8:
        return None
    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return round(len(set(g)) / len(g), 4)


def cached_of(usage, timings):
    """Cached prompt tokens as the engine reports them (OpenAI usage and/or llama.cpp timings.cache_n)."""
    vals = []
    c = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if isinstance(c, (int, float)):
        vals.append(int(c))
    c = (timings or {}).get("cache_n")
    if isinstance(c, (int, float)):
        vals.append(int(c))
    return max(vals) if vals else None


def decode_rates(r, ct, head=16, tail=4):
    """Middle-window decode rate. Each SSE event's token count is exact when the engine sends one
    token per event; otherwise (several tokens per event, held-back whitespace) tokens are assigned to
    events in proportion to their characters. The window runs from the first event that completes
    token `head` to the last event that ends at or before token ct-tail, measured between those two
    arrivals, so a k-token first event does not inflate the rate."""
    t, c = r["ev_t"], r["ev_c"]
    info = {"events": len(t)}
    if len(t) < 2 or not ct:
        return None, None, info
    e2e = (ct - 1) / (t[-1] - t[0]) if t[-1] > t[0] else None
    if len(t) == ct:
        cum = list(range(1, ct + 1)); method = "events"
    else:
        tot = sum(c)
        if tot <= 0:
            return None, e2e, info
        acc, cum = 0, []
        for x in c:
            acc += x
            cum.append(acc * ct / tot)
        method = "chars"
    i = next((k for k in range(len(t)) if cum[k] >= head), None)
    js = [k for k in range(len(t)) if cum[k] <= ct - tail]
    j = js[-1] if js else None
    info.update({"rate_method": method, "tokens_per_event": round(ct / len(t), 3)})
    if i is None or j is None or j <= i or t[j] <= t[i]:
        return None, e2e, info
    info.update({"window_tokens": round(cum[j] - cum[i], 1), "window_s": round(t[j] - t[i], 4)})
    return (cum[j] - cum[i]) / (t[j] - t[i]), e2e, info


def _summary(runs, key="decode_tps"):
    ok = [x for x in runs if not x.get("invalid") and x.get(key)]
    v = sorted(x[key] for x in ok)
    s = {"runs": len(runs), "valid_runs": len(ok)}
    if v:
        s.update({"median_tps": round(statistics.median(v), 2), "min_tps": round(v[0], 2), "max_tps": round(v[-1], 2)})
        e = [x["decode_tps_e2e"] for x in ok if x.get("decode_tps_e2e")]
        if e:
            s["median_tps_e2e"] = round(statistics.median(e), 2)
    s["invalid"] = sorted({r for x in runs for r in x.get("invalid", [])})
    s["flags"] = sorted({f for x in runs for f in x.get("flags", [])})
    return s

# ---------------------------------------------------------------- tests
def t_decode(a, greedy=False):
    body = body_of(a, decode_messages(), a.decode_tokens, greedy=greedy)
    res = {"greedy": greedy, "invalid": [], "flags": []}
    try:
        r = stream_chat(a.base_url, body)
    except Exception as e:
        res["invalid"].append(f"request failed: {e!r}"[:300])
        return res
    u = r["usage"] or {}
    ct = u.get("completion_tokens")
    mid, e2e, info = decode_rates(r, ct, a.window_head, a.window_tail)
    text = r["content"]
    res.update({"completion_tokens": ct, "chunks": r["n_chunks"], "ttft_s": r["ttft"], "decode_tps": mid,
                "decode_tps_e2e": e2e, "e2e_wall_tps": (ct / r["total"]) if ct else None, "finish": r["finish"],
                "usage_reported": bool(ct), "rate": info, "timings": r["timings"],
                "distinct4": distinct4(text or r["reasoning"]), "text": text, "reasoning": r["reasoning"],
                "events_t_ms": [round(x * 1000, 2) for x in r["ev_t"]], "events_chars": r["ev_c"]})
    res["invalid"] += thinking_leak(r)
    if not ct:
        res["invalid"].append("usage.completion_tokens missing (no fallback to event counts)")
    if mid is None and ct:
        res["invalid"].append("too few streamed events for a middle-window rate")
    if r["finish"] != "length":
        res["flags"].append(f"finish_reason={r['finish']} (expected length)")
    if ct and ct != a.decode_tokens:
        res["flags"].append(f"completion_tokens={ct} != {a.decode_tokens}")
    if res["distinct4"] is not None and res["distinct4"] < a.rep_threshold:
        res["flags"].append(f"repetition loop (distinct-4-gram {res['distinct4']})")
    tm = r["timings"] or {}
    if tm.get("draft_n"):
        res["draft_acceptance"] = round(tm.get("draft_n_accepted", 0) / tm["draft_n"], 4)
    return res


def parse_ref(a):
    ref = dict(REF_PROMPT_TOKENS.get(corpus_sha(), {}))
    for part in filter(None, (a.ref_prompt_tokens or "").split(",")):
        k, v = part.split(":")
        ref[int(k)] = int(v)
    return ref


def t_prefill(a, size, ref):
    ctx = build_context(size)
    nonce = f"session {random.getrandbits(64):016x}"
    msgs = prefill_messages(ctx, nonce, "In one sentence, what is the first file in the repository context about?")
    # max_tokens high enough that the answer ends naturally: a turn cut by max_tokens ("length") is not
    # reusable as a cached prefix in some engines (MTPLX), and real agent turns end with stop/tool_calls.
    body = body_of(a, msgs, 256)
    sid = f"bench-{random.getrandbits(48):012x}"
    res = {"target": size, "invalid": [], "flags": []}
    try:
        r = stream_chat(a.base_url, body, session=sid)
    except Exception as e:
        res["invalid"].append(f"request failed: {e!r}"[:300])
        return res
    u = r["usage"] or {}
    pt = u.get("prompt_tokens")
    exp = ref.get(size)
    cold_cached = cached_of(u, r["timings"])
    res.update({"prompt_tokens": pt, "expected_prompt_tokens": exp, "ttft_s": r["ttft"],
                "prefill_tps": (pt / r["ttft"]) if pt and r["ttft"] else None, "cold_cached_tokens": cold_cached,
                "first_event_chars": r["ev_c"][0] if r["ev_c"] else None, "timings": r["timings"],
                "finish": r["finish"], "answer": r["content"], "reasoning_chars": len(r["reasoning"])})
    res["invalid"] += thinking_leak(r)
    if not pt:
        res["invalid"].append("usage.prompt_tokens missing")
    elif exp:
        if abs(pt - exp) > PROMPT_TOL * exp:
            res["invalid"].append(f"prompt_tokens {pt} outside ±{PROMPT_TOL:.0%} of {exp} (truncated or template drift)")
    else:
        res["flags"].append("no reference prompt_tokens for this corpus/size (unchecked)")
    if cold_cached is not None and cold_cached > COLD_CACHE_MAX:
        res["invalid"].append(f"cold request reported {cold_cached} cached tokens")
    # agent pattern: same prefix + previous turns + new question -> should hit the prefix cache
    cached = {"samples": [], "invalid": [], "flags": []}
    res["cached"] = cached
    if not (r["content"] or "").strip():
        cached["invalid"].append("turn 1 returned no content; not substituting text, cached test skipped")
        return res
    if r["finish"] != "stop":
        cached["flags"].append(f"turn 1 finish_reason={r['finish']} (engines may not cache a 'length' turn)")
    hist = msgs + [{"role": "assistant", "content": r["content"]}]
    prev_pt = pt
    for q in FOLLOWUPS[:a.cached_samples]:
        time.sleep(a.turn_gap)  # emulate tool execution / client think time between agent turns
        msgs2 = hist + [{"role": "user", "content": q}]
        try:
            r2 = stream_chat(a.base_url, body_of(a, msgs2, 256), session=sid)
        except Exception as e:
            cached["invalid"].append(f"follow-up request failed: {e!r}"[:300])
            break
        u2 = r2["usage"] or {}
        c2 = cached_of(u2, r2["timings"])
        rec = {"ttft_s": r2["ttft"], "prompt_tokens": u2.get("prompt_tokens"), "cached_tokens": c2,
               "finish": r2["finish"], "answer": r2["content"]}
        cached["samples"].append(rec)
        for why in thinking_leak(r2):
            cached["invalid"].append(f"follow-up: {why}")
        if c2 is not None and prev_pt and c2 < 0.5 * prev_pt:
            cached["flags"].append(f"follow-up reused only {c2} of {prev_pt} prefix tokens")
        if not (r2["content"] or "").strip():
            cached["flags"].append("follow-up returned no content; stopping the cached series")
            break
        hist = msgs2 + [{"role": "assistant", "content": r2["content"]}]
        prev_pt = u2.get("prompt_tokens") or prev_pt
    tt = [s["ttft_s"] for s in cached["samples"] if s["ttft_s"]]
    cached["median_ttft_s"] = round(statistics.median(tt), 4) if tt else None
    # schema-1 field names, kept for older readers
    res["turn2_ttft_s"] = cached["median_ttft_s"]
    if cached["samples"]:
        res["turn2_cached_tokens"] = cached["samples"][0]["cached_tokens"]
        res["turn2_prompt_tokens"] = cached["samples"][0]["prompt_tokens"]
    res["turn1_finish"] = r["finish"]
    return res


def _check_tc(tcs):
    ok = False; parsed = None
    for tc in tcs:
        try:
            parsed = json.loads(tc["arguments"]) if tc["arguments"] else {}
            if tc["name"] == "read_file" and "path" in parsed and "main.py" in parsed["path"]:
                ok = True
        except Exception:
            pass
    return ok, parsed


def tools_messages():
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": "Open src/main.py and tell me what the entry point does."}]


def t_tools(a, think):
    msgs = tools_messages()
    body = body_of(a, msgs, 4096 if think else 512, think=think, tools=TOOLS, tool_choice="auto")
    out = {"thinking": think}
    # streaming
    try:
        r = stream_chat(a.base_url, body)
    except Exception as e:
        out["stream"] = {"ok": False, "error": repr(e)[:300], "tool_calls": []}
        out["nonstream"] = {"ok": False, "error": "skipped"}
        return out
    ok, parsed = _check_tc(r["tool_calls"])
    leaked = "<tool_call" in r["content"] or "<function" in r["content"]
    out["stream"] = {"ok": ok, "tool_calls": r["tool_calls"], "finish": r["finish"], "ttft_s": r["ttft"],
                     "total_s": r["total"], "content_leak": leaked, "reasoning_chars": len(r["reasoning"]),
                     "content_head": r["content"][:200]}
    if not think and thinking_leak(r):
        out["stream"]["thinking_leak"] = thinking_leak(r)
    # non-streaming
    try:
        t0 = time.perf_counter()
        resp = json.load(post(a.base_url, dict(body, stream=False), False))
        m = resp["choices"][0]["message"]
        tcs = [{"name": t["function"]["name"], "arguments": t["function"]["arguments"] if isinstance(t["function"]["arguments"], str) else json.dumps(t["function"]["arguments"]), "id": t.get("id")} for t in (m.get("tool_calls") or [])]
        ok2, _ = _check_tc(tcs)
        out["nonstream"] = {"ok": ok2, "tool_calls": tcs, "finish": resp["choices"][0].get("finish_reason"), "total_s": time.perf_counter() - t0}
    except Exception as e:
        out["nonstream"] = {"ok": False, "error": repr(e)[:300]}
    # round trip: feed a tool result back, expect a final answer (no crash on role=tool)
    if r["tool_calls"]:
        tc = r["tool_calls"][0]
        call_id = tc["id"] or "call_0"
        msgs2 = msgs + [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"] or "{}"}}]},
                        {"role": "tool", "tool_call_id": call_id, "content": "import sys\n\ndef main():\n    print('hello from nace')\n    return 0\n\nif __name__ == '__main__':\n    sys.exit(main())\n"}]
        try:
            r2 = stream_chat(a.base_url, dict(body, messages=msgs2))
            low = r2["content"].lower()
            out["round_trip"] = {"ok": bool(low.strip()) and ("hello" in low or "print" in low or "main" in low),
                                 "finish": r2["finish"], "content_head": r2["content"][:200], "more_tool_calls": len(r2["tool_calls"])}
        except Exception as e:
            out["round_trip"] = {"ok": False, "error": repr(e)[:300]}
    return out


def tools_pass(tools):
    return bool(tools) and all(t["stream"].get("ok") and t["nonstream"].get("ok") and t.get("round_trip", {}).get("ok")
                               for t in tools)

# ---------------------------------------------------------------- warm-up
def warmup(a, adaptive=False, log=print):
    """Pays one-time costs before anything is timed: lazy Metal pipeline/kernel creation for a
    multi-chunk prefill (> 2048 tokens), a decode and the tool path; for engines whose speculation
    adapts across requests (Ollama's MLX runner, MTPLX, oMLX's drafter), 3 full decodes so the depth
    controller settles. All requests use the benchmark's own sampling and thinking settings."""
    steps = []

    def step(name, fn):
        t0 = time.time()
        try:
            r = fn()
            u = r.get("usage") or {}
            rec = {"name": name, "ok": True, "s": round(time.time() - t0, 2), "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "finish": r.get("finish"),
                   "thinking_leak": thinking_leak(r)}
        except Exception as e:
            rec = {"name": name, "ok": False, "s": round(time.time() - t0, 2), "error": repr(e)[:300]}
        steps.append(rec)
        log(f"  warm-up {name}: {'ok' if rec['ok'] else 'FAILED ' + rec['error']} ({rec['s']}s)")
        return rec

    ctx = build_context(WARM_TOKENS, offset=WARM_OFFSET_CHARS)
    msgs = prefill_messages(ctx, f"warmup {random.getrandbits(64):016x}", "Summarize the first file in the repository context.")
    step("prefill~2.9k+decode64", lambda: stream_chat(a.base_url, body_of(a, msgs, 64)))
    step("tools", lambda: stream_chat(a.base_url, body_of(a, tools_messages(), 512, tools=TOOLS, tool_choice="auto")))
    if adaptive:
        for i in range(3):
            step(f"decode{a.decode_tokens}#{i + 1}", lambda: stream_chat(a.base_url, body_of(a, decode_messages(), a.decode_tokens)))
    return {"adaptive": adaptive, "steps": steps, "ok": all(s["ok"] for s in steps)}


def light_warmup(a):
    return stream_chat(a.base_url, body_of(a, [{"role": "user", "content": "Say hi."}], 8))

# ---------------------------------------------------------------- main
def f2(x):
    return 'NA' if x is None else f'{x:.2f}'


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="request model id (for Ollama: the tag; never taken from /v1/models)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--pid", type=int, default=0, help="server pid: its whole process tree is measured")
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--decode-runs", type=int, default=5)
    ap.add_argument("--greedy-runs", type=int, default=None, help="T=0 decode runs (default: --decode-runs; 0 = skip)")
    ap.add_argument("--window-head", type=int, default=16, help="tokens dropped at the start of the decode window")
    ap.add_argument("--window-tail", type=int, default=4, help="tokens dropped at the end of the decode window")
    ap.add_argument("--rep-threshold", type=float, default=0.5, help="flag runs whose distinct-4-gram ratio is below this")
    ap.add_argument("--sizes", default="4000,16000")
    ap.add_argument("--corpus", default=None, help="prompt corpus (default: the frozen corpus/bench_corpus.txt)")
    ap.add_argument("--ref-prompt-tokens", default="", help="override the reference prompt_tokens, e.g. 4000:4133,16000:16285")
    ap.add_argument("--cached-samples", type=int, default=3, help="follow-up turns in the cached-TTFT test (median)")
    ap.add_argument("--turn-gap", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--frequency-penalty", type=float, default=0.0)
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="llama.cpp key; neutral 1.0")
    ap.add_argument("--sampling-overrides", default="", help='JSON applied to every request\'s sampling fields; null drops a field, e.g. {"repeat_penalty": null}')
    ap.add_argument("--extra-body", default="", help="JSON merged into every request body")
    ap.add_argument("--reasoning-effort", default="low", help="effort for the thinking-on tools test")
    ap.add_argument("--no-think-style", default="both", choices=["both", "kwargs", "effort", "none"])
    ap.add_argument("--warmup", default="light", choices=["none", "light", "full"],
                    help="light: one tiny request; full: the compare_engines warm-up; none: caller warmed up")
    ap.add_argument("--adaptive-warm", action="store_true", help="with --warmup full: 3 extra full decodes")
    ap.add_argument("--no-gpu-lock", action="store_true", help=f"do not take the GPU lock {gpu_hogs.LOCK_PATH} (NACE_GPU_LOCK_HELD=1 also skips it)")
    ap.add_argument("--skip", default="", help="comma list of: decode,greedy,prefill,tools,agent,sweep,sustain,cache")
    ap.add_argument("--agentic", action="store_true",
                    help="also run the agent-harness phases (agentic.py): sweep, cache, sustain (last); --skip drops any")
    ap.add_argument("--sweep-sizes", default="1000,2000,4000,8000,16000,max", help="agentic sweep: token targets; max = the largest that fits --max-ctx")
    ap.add_argument("--sweep-repeats", type=int, default=2, help="agentic sweep: cold requests per size (median)")
    ap.add_argument("--max-ctx", type=int, default=32768, help="context the server runs with (agentic sweep 'max' fits it)")
    ap.add_argument("--cache-tokens", type=int, default=8000, help="agentic cache phase: shared prefix size")
    ap.add_argument("--server-slots", type=int, default=0,
                    help="conversations the server keeps live at once (its --slots); 0 = the engine's own default. "
                         "Stamped into the cache phase, whose A-after-B turn depends on it")
    ap.add_argument("--sustain-s", type=float, default=300, help="agentic sustain phase: seconds of back-to-back decodes")
    ap.add_argument("--agent-trace", default=None,
                    help="agent session to replay (default agent-traces/opencode-ledger.json; 'none' = no agent phase)")
    ap.add_argument("--agent-tool-mode", default="rendered", choices=["rendered", "api"],
                    help="rendered: tools in the system prompt as the Qwen chat template renders them; api: as recorded")
    ap.add_argument("--agent-gaps", default="recorded", help="pause before each agent turn: recorded (<= 5 s), none, or seconds")
    ap.add_argument("--agent-max-turns", type=int, default=None, help="replay only the first N requests of the trace")
    ap.add_argument("--agent-timeout", type=float, default=2400, help="start no agent turn after this many seconds")
    ap.add_argument("--out", default=None, help="result JSON (default: $BENCH_OUT or bench/harness/runs, results/<label>-<time>.json)")
    a = ap.parse_args(argv)
    if a.greedy_runs is None:
        a.greedy_runs = a.decode_runs
    if a.corpus:
        set_corpus(a.corpus)
    if a.agent_trace is None:
        a.agent_trace = os.path.join(HERE, "agent-traces", "opencode-ledger.json")
    elif a.agent_trace.lower() == "none":
        a.agent_trace = ""
    return a


def main():
    if os.path.exists(LOCAL_ENV):
        load_env_file(LOCAL_ENV)
    a = parse_args()
    skip = set(filter(None, a.skip.split(",")))
    lock = None
    if not a.no_gpu_lock and not os.environ.get("NACE_GPU_LOCK_HELD"):
        lock = gpu_hogs.GpuLock(log=lambda m: print("  " + m, flush=True)).acquire()
    mem = MemSampler(a.pid) if a.pid else None
    if mem:
        mem.start()
    load_corpus()
    res = {"schema": SCHEMA, "label": a.label, "model": a.model, "base_url": a.base_url,
           "when": time.strftime("%Y-%m-%d %H:%M:%S"), "temperature": a.temperature,
           "args": {k: v for k, v in vars(a).items() if not k.startswith("_")},
           "sampling": sampling(a), "sampling_greedy": sampling(a, greedy=True), "no_think": no_think(a),
           "corpus": corpus_record(), "harness": harness_info(), "phases": {}, "memory": {},
           "gpu_lock": {"held_by": "caller" if os.environ.get("NACE_GPU_LOCK_HELD") else ("bench" if lock else "none"),
                        "path": gpu_hogs.LOCK_PATH, "wait_s": round(lock.wait_s, 2) if lock else None}}
    if not res["corpus"]["frozen"]:
        print(f"  note: corpus sha256 {res['corpus']['sha256'][:12]} is not the frozen corpus; results are not "
              "comparable with published ones", flush=True)
    print(f"== {a.label}", flush=True)
    vm0 = vm_snapshot()
    res["vm_start"] = vm0
    if a.agentic:
        import agentic
        res["thermal_start"] = agentic.thermal_state()
        if a.pid:  # weights-only baseline: KV cost at the largest context = peak - idle
            res["idle_footprint_mb"] = footprint_snapshot(_tree(a.pid)).get("phys_footprint_mb")

    def phase(name, fn):
        v0 = vm_snapshot()
        t0 = time.time()
        out = fn()
        t1 = time.time()
        res["phases"][name] = [round(t0, 3), round(t1, 3)]
        m = footprint_snapshot(_tree(a.pid)) if a.pid else {}
        v1 = vm_snapshot()
        m.update({"swap_used_mb": v1.get("swap_used_mb"), "memorystatus_level": v1.get("memorystatus_level"),
                  "swapouts_delta": (v1.get("swapouts", 0) - v0.get("swapouts", 0)) if "swapouts" in v0 else None})
        res["memory"][name] = m
        return out

    try:
        if a.warmup == "light":
            phase("warmup", lambda: light_warmup(a))
        elif a.warmup == "full":
            res["warmup"] = phase("warmup", lambda: warmup(a, a.adaptive_warm))
        if "decode" not in skip and a.decode_runs > 0:
            res["decode"] = phase("decode", lambda: [t_decode(a) for _ in range(a.decode_runs)])
            res["decode_summary"] = _summary(res["decode"])
            for d in res["decode"]:
                print(f"  decode T={a.temperature}: {f2(d.get('decode_tps'))} tok/s (e2e {f2(d.get('decode_tps_e2e'))}; "
                      f"{d.get('completion_tokens')} tok, finish={d.get('finish')}, d4={d.get('distinct4')})"
                      + (f" INVALID {d['invalid']}" if d["invalid"] else "") + (f" flags {d['flags']}" if d["flags"] else ""), flush=True)
            s = res["decode_summary"]
            print(f"  decode median {s.get('median_tps')} (min {s.get('min_tps')}, max {s.get('max_tps')}) over {s['valid_runs']}/{s['runs']} valid", flush=True)
        if "greedy" not in skip and a.greedy_runs > 0:
            res["decode_greedy"] = phase("decode_greedy", lambda: [t_decode(a, greedy=True) for _ in range(a.greedy_runs)])
            res["decode_greedy_summary"] = _summary(res["decode_greedy"])
            s = res["decode_greedy_summary"]
            print(f"  decode T=0: median {s.get('median_tps')} (min {s.get('min_tps')}, max {s.get('max_tps')}) over {s['valid_runs']}/{s['runs']} valid"
                  + (f" INVALID {s['invalid']}" if s["invalid"] else "") + (f" flags {s['flags']}" if s["flags"] else ""), flush=True)
        if "prefill" not in skip:
            ref = parse_ref(a)
            res["prefill"] = []
            for sz in [int(x) for x in a.sizes.split(",") if x]:
                p = phase(f"prefill_{sz}", lambda: t_prefill(a, sz, ref))
                res["prefill"].append(p)
                c = p.get("cached", {})
                print(f"  prefill {sz}: {p.get('prompt_tokens')} tok (ref {p.get('expected_prompt_tokens')}), TTFT {f2(p.get('ttft_s'))}s -> "
                      f"{p.get('prefill_tps') and round(p['prefill_tps'])} tok/s, cold cached={p.get('cold_cached_tokens')} | "
                      f"cached TTFT median {f2(c.get('median_ttft_s'))}s over {len(c.get('samples', []))} turns "
                      f"(cached {[s['cached_tokens'] for s in c.get('samples', [])]})"
                      + (f" INVALID {p['invalid'] + c.get('invalid', [])}" if p["invalid"] or c.get("invalid") else "")
                      + (f" flags {p['flags'] + c.get('flags', [])}" if p["flags"] or c.get("flags") else ""), flush=True)
        if "tools" not in skip:
            res["tools"] = phase("tools", lambda: [t_tools(a, False), t_tools(a, True)])
            res["tools_pass"] = tools_pass(res["tools"])
            for t in res["tools"]:
                print(f"  tools(think={t['thinking']}): stream ok={t['stream'].get('ok')} {[(x['name'], x['arguments']) for x in t['stream'].get('tool_calls', [])]} leak={t['stream'].get('content_leak')} | nonstream ok={t['nonstream'].get('ok')} | roundtrip={t.get('round_trip', {}).get('ok')}", flush=True)
        if "agent" not in skip and a.agent_trace:
            import agent_replay
            if not os.path.exists(a.agent_trace):
                print(f"  agent session: no trace at {a.agent_trace}; skipped", flush=True)
            else:
                trace = agent_replay.load_trace(a.agent_trace)
                print(f"  agent session: {trace.get('name')} ({len(trace['requests'])} requests, tool mode "
                      f"{a.agent_tool_mode}, gaps {a.agent_gaps})", flush=True)
                res["agent"] = phase("agent", lambda: agent_replay.replay(
                    a, trace, a.agent_tool_mode, a.agent_gaps, a.agent_max_turns,
                    log=lambda m: print(m, flush=True), timeout_s=a.agent_timeout))
                ag = res["agent"]
                print("  agent session: " + agent_replay.line(ag["summary"])
                      + (f" INVALID {ag['invalid']}" if ag["invalid"] else "")
                      + (f" flags {ag['flags']}" if ag["flags"] else ""), flush=True)
        if a.agentic:
            import agentic
            if res.get("agent"):
                res["agent"]["cache_hit_rate"] = agentic.hit_rate(res["agent"].get("turns"))
            if "sweep" not in skip:
                res["sweep"] = phase("sweep", lambda: agentic.t_sweep(a, log=lambda m: print(m, flush=True)))
            if "cache" not in skip:
                res["cache"] = phase("cache", lambda: agentic.t_cache(a, log=lambda m: print(m, flush=True)))
            if "sustain" not in skip and a.sustain_s > 0:  # last: it heats the machine
                res["sustain"] = phase("sustain", lambda: agentic.t_sustain(a, log=lambda m: print(m, flush=True)))
            res["thermal_end"] = agentic.thermal_state()
    finally:
        if mem:
            mem.stop_ev.set(); mem.join(timeout=5)
            res["peak_rss_mb"] = round(mem.peak_rss); res["peak_footprint_mb"] = round(mem.peak_fp)
        if lock:
            res["gpu_lock"]["held_s"] = round(lock.release(), 1)
    # roll-up: INVALID reasons and warnings over all phases, peak memory
    inv, fl = [], []
    for name, key in (("decode", "decode_summary"), ("decode_greedy", "decode_greedy_summary")):
        s = res.get(key) or {}
        inv += [f"{name}: {x}" for x in s.get("invalid", [])]
        fl += [f"{name}: {x}" for x in s.get("flags", [])]
    for p in res.get("prefill", []):
        inv += [f"prefill_{p['target']}: {x}" for x in p.get("invalid", [])]
        fl += [f"prefill_{p['target']}: {x}" for x in p.get("flags", [])]
        inv += [f"cached_{p['target']}: {x}" for x in p.get("cached", {}).get("invalid", [])]
        fl += [f"cached_{p['target']}: {x}" for x in p.get("cached", {}).get("flags", [])]
    if res.get("agent"):
        inv += [f"agent: {x}" for x in res["agent"].get("invalid", [])]
        fl += [f"agent: {x}" for x in res["agent"].get("flags", [])]
    for name in ("sweep", "cache", "sustain"):
        if res.get(name):
            inv += [f"{name}: {x}" for x in res[name].get("invalid", [])]
            fl += [f"{name}: {x}" for x in res[name].get("flags", [])]
    th = res.get("thermal_start") or {}
    if th.get("battery") or th.get("low_power"):
        fl.append("POWER: run started on battery or in Low Power Mode (numbers are not comparable with plugged-in runs)")
    for name, m in res["memory"].items():
        if (m.get("swapouts_delta") or 0) > 1000:
            fl.append(f"{name}: {m['swapouts_delta']} pages swapped out during the phase")
    res["invalid"], res["flags"] = inv, fl
    peaks = [m.get("phys_footprint_peak_mb") for m in res["memory"].values() if m.get("phys_footprint_peak_mb")]
    res["peak_memory_mb"] = max(peaks) if peaks else None
    mapped = [m.get("mapped_file_clean_mb") for m in res["memory"].values() if m.get("mapped_file_clean_mb") is not None]
    res["mapped_file_clean_mb"] = max(mapped) if mapped else None
    res["vm_end"] = vm_snapshot()
    res["when_end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if res.get("peak_memory_mb"):
        print(f"  memory: phys_footprint_peak {res['peak_memory_mb'] / 1024:.2f} GB, resident mapped files "
              f"{(res['mapped_file_clean_mb'] or 0) / 1024:.2f} GB, swap used {res['vm_end'].get('swap_used_mb')} MB", flush=True)
    if inv:
        print(f"  INVALID: {inv}", flush=True)
    out = a.out or os.path.join(out_dir(), "results", f"{a.label}-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
