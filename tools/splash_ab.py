#!/usr/bin/env python3
"""Kapitsa's Splash backend against plain Splash 1.4.0, one engine at a time (never side by side), for the T=0 identity
check and the TTFT / decode tok/s overhead of the bridge. It takes the GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) itself and holds it for the run, unless
NI_LOCK_HELD or NACE_GPU_LOCK_HELD says the caller does (a `lockf -k ...` wrapper holds it on its own file description, so under one this script's flock would wait for ever: export NI_LOCK_HELD=1 too, or run it bare; it gives up after --lock-timeout-s); --wait-only stays outside it. It launches, measures and reaps each server itself, and refuses to start when a Splash or Kapitsa model process is up.

  python3 tools/splash_ab.py --legs K,S,K,S --out DIR [--reps 3] [--smoke]

K = `kapitsa serve --engine splash` (the bridge), S = `python -m server.server` (plain Splash). Both get the same engine
flags (--verify-rows all, --expert-memory, --idle-release off, 32768 context), so the child is the same process either way.
Per leg: one warm-up request, the 3 identity prompts, then REPS requests each at ~1K and ~4K prompt tokens (a nonce per
rep keeps every prompt cold, and identical on both engines). --tap runs the engine binary through tools/splash_tap.py on
both sides and adds each request's prompt-id md5, output-id md5 and first divergence to summary.json. Writes DIR/leg<N>-<engine>.json and DIR/summary.json."""
import argparse, fcntl, hashlib, json, os, re, signal, socket, statistics, subprocess, sys, threading, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# set from --splash-root / --binary / --model-dir (or $SPLASH_ROOT, $SPLASH_MODEL_DIR) in main(); Splash re-resolves a model to
# another hash folder, so no path of one Mac is built in
SPLASH_ROOT = os.path.expanduser(os.environ.get("SPLASH_ROOT", "~/Documents/nace/splash-1.4.0"))
SPLASH_BIN = SPLASH_PY = MODEL_DIR = ""
MODEL_ID = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0"
ENGINE_ARGS = ["--verify-rows", "all", "--expert-memory", "17179869184", "--idle-release", "off"]
DEFAULT_MEMORY_ARGS = ["--verify-rows", "all", "--idle-release", "off"]  # Splash's own memory plan: no --expert-memory
CTX = "32768"
TAP = False  # --tap: the engine binary is tools/splash_tap.py, which records every request's prompt and output ids
IDENTITY = [
    "Explain in two sentences why the sky is blue.",
    "Write a Python function that returns the nth Fibonacci number, with a short docstring.",
    "Translate 'The meeting is moved to Thursday afternoon' into French and German.",
]
IDENTITY_MAX = 160
PERF_MAX = 128
SIZES = {"1K": 4000, "4K": 16400}  # characters of context in the system message (~1.0K and ~4.0K tokens)


def corpus():
    text = open(ROOT + "/docs/architecture.md").read() + open(ROOT + "/docs/usage.md").read()
    return text * 3


def messages(nonce, chars, question):
    system = "[session %s] Answer in one short sentence." % nonce
    if chars:
        system += "\n\n# Repository context\n" + CORPUS[:chars]
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def running_model_processes():
    out = subprocess.run(["ps", "-axo", "pid,command"], capture_output=True, text=True).stdout
    return [l.strip() for l in out.splitlines()
            if re.search(r"serve-native|splash.*server\.server|kapitsa serve|build/kapitsa", l) and "splash_ab" not in l
            and "ps -axo" not in l]


def reclaimable_gib():
    """free + inactive + speculative + purgeable pages, the memory Splash's start check can count on."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    n = sum(int(re.search(r"%s:\s+(\d+)" % k, out).group(1)) for k in
            ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"))
    return n * page / 2**30


def wait_for_memory(need, cap, run_anyway=False):
    t0 = time.time()
    print("waiting for %.1f GiB free+reclaimable (cap %.0f s); now %.1f" % (need, cap, reclaimable_gib()), flush=True)
    while reclaimable_gib() < need:
        if time.time() - t0 > cap:
            if run_anyway:
                print("TIMEOUT: %.1f GiB free+reclaimable after %.0f s, need %.1f; running anyway" % (reclaimable_gib(), cap, need), flush=True)
                return
            print("TIMEOUT: %.1f GiB free+reclaimable after %.0f s, need %.1f; no run, no retry" % (reclaimable_gib(), cap, need), flush=True)
            raise SystemExit(3)
        time.sleep(5)
    print("memory ok: %.1f GiB after %.0f s" % (reclaimable_gib(), time.time() - t0), flush=True)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Leg:
    """One server: launched, its serve-native child's RSS sampled, torn down and the child reaped."""
    def __init__(self, engine, log):
        self.engine, self.port, self.peak_rss, self.child_pids = engine, free_port(), 0, set()
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        binary = SPLASH_BIN
        self.tap_log = log + ".tap.jsonl"
        if TAP:
            binary = ROOT + "/tools/splash_tap.py"
            env.update(SPLASH_TAP_REAL=SPLASH_BIN, SPLASH_TAP_LOG=self.tap_log, SPLASH_ROOT=SPLASH_ROOT)
            if os.path.exists(self.tap_log): os.remove(self.tap_log)
        if engine == "K":
            cmd = [ROOT + "/build/kapitsa", "serve", "--engine", "splash", "--splash-binary", binary, "--splash-model", MODEL_DIR,
                   "--context", "32k", "--private", "--port", str(self.port)]
            for a in ENGINE_ARGS: cmd += ["--splash-arg", a]
            self.model = None
        else:
            cmd = [SPLASH_PY, "-P", "-m", "server.server", MODEL_DIR, "--model", MODEL_ID, "--binary", binary,
                   "--port", str(self.port), "--max-context", CTX] + ENGINE_ARGS
            env["PYTHONPATH"] = SPLASH_ROOT
            self.model = MODEL_ID
        self.logf = open(log, "w")
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        self.stop = False
        self.sampler = threading.Thread(target=self.sample, daemon=True); self.sampler.start()
        try:
            for _ in range(1600):
                if self.proc.poll() is not None: raise SystemExit("leg %s exited early; see %s" % (engine, log))
                try:
                    urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % self.port, timeout=1).read(); break
                except Exception: time.sleep(0.25)
            else: raise SystemExit("leg %s did not come up" % engine)
        except BaseException:  # the server runs in its own session: nothing else would stop it (and its engine child) or free the port
            self.close()
            raise
        self.up_s = time.time() - self.t0

    def sample(self):
        while not self.stop:
            out = subprocess.run(["ps", "-axo", "pid,ppid,rss,command"], capture_output=True, text=True).stdout
            for l in out.splitlines():
                m = re.match(r"\s*(\d+)\s+(\d+)\s+(\d+)\s+(.*)", l)
                if m and "serve-native" in m.group(4) and "ps -axo" not in m.group(4) and self.descends(int(m.group(1))):
                    self.child_pids.add(int(m.group(1))); self.peak_rss = max(self.peak_rss, int(m.group(3)))
            time.sleep(0.3)

    def descends(self, pid):
        for _ in range(6):
            if pid == self.proc.pid: return True
            r = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
            if not r: return False
            pid = int(r)
        return False

    def chat(self, msgs, max_tokens):
        body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "seed": 1, "stream": True,
                "stream_options": {"include_usage": True}}
        if self.model: body["model"] = self.model
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % self.port, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time(); first = None; reasoning, content, usage, finish, chunks = [], [], None, None, 0
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    c = delta.get("content") or ""
                    if (rc or c) and first is None: first = time.time()
                    reasoning.append(rc); content.append(c); chunks += bool(rc or c)
                    finish = ch.get("finish_reason") or finish
        end = time.time()
        r_, c_ = "".join(reasoning), "".join(content)
        n = (usage or {}).get("completion_tokens")
        return {"reasoning": r_, "content": c_, "finish": finish, "usage": usage,
                "text_md5": hashlib.md5((r_ + "\x00" + c_).encode()).hexdigest(),
                "ttft_s": None if first is None else first - t0, "total_s": end - t0,
                "decode_tps": None if not n or n < 2 or first is None or end <= first else (n - 1) / (end - first)}

    def close(self):
        self.stop = True
        try: os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: self.proc.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL); self.proc.wait(10)
        for _ in range(100):  # the child drains and exits after its stdin closes
            if not any(subprocess.run(["ps", "-p", str(p)], capture_output=True).returncode == 0 for p in self.child_pids): break
            time.sleep(0.2)
        left = [p for p in self.child_pids if subprocess.run(["ps", "-p", str(p)], capture_output=True).returncode == 0]
        self.logf.close()
        return left


def run_leg(engine, idx, out, reps, smoke):
    leg = Leg(engine, "%s/leg%d-%s.log" % (out, idx, engine))
    res = {"engine": engine, "up_s": leg.up_s, "identity": [], "perf": []}
    try:
        if smoke:
            res["smoke"] = leg.chat([{"role": "user", "content": IDENTITY[0]}], IDENTITY_MAX)
        else:
            leg.chat(messages("warm", 0, "Say hi."), 16)
            for i, q in enumerate(IDENTITY):
                res["identity"].append(dict(leg.chat([{"role": "user", "content": q}], IDENTITY_MAX), prompt=i))
            for size, chars in SIZES.items():
                for rep in range(reps):
                    res["perf"].append(dict(leg.chat(messages("r%d" % rep, chars, "What does this repository do?"), PERF_MAX),
                                            size=size, rep=rep))
    finally:
        res["peak_rss_mib"] = leg.peak_rss / 1024
        res["child_left"] = leg.close()
        if TAP and os.path.exists(leg.tap_log):
            res["tap"] = [json.loads(l) for l in open(leg.tap_log)]
    json.dump(res, open("%s/leg%d-%s.json" % (out, idx, engine), "w"), indent=1)
    return res


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(legs):
    s = {"identity": [], "perf": {}}
    for i in range(len(IDENTITY)):
        rows = [(l["engine"], l["identity"][i]) for l in legs if l["identity"]]
        s["identity"].append({"prompt": i, "md5": [(e, r["text_md5"][:12], (r["usage"] or {}).get("completion_tokens"),
                                                  (r["usage"] or {}).get("prompt_tokens")) for e, r in rows],
                              "all_equal": len({r["text_md5"] for _, r in rows}) == 1})
    for eng in sorted({l["engine"] for l in legs}):
        for size in SIZES:
            rows = [p for l in legs if l["engine"] == eng for p in l["perf"] if p["size"] == size]
            s["perf"]["%s/%s" % (eng, size)] = {
                "n": len(rows), "prompt_tokens": sorted({(p["usage"] or {}).get("prompt_tokens") for p in rows}),
                # prompt tokens / TTFT: TTFT holds the fixed per-request cost, so this is not prefill throughput
                "prefill_tps_median": med([(p["usage"] or {}).get("prompt_tokens", 0) / p["ttft_s"] for p in rows if p["ttft_s"]]),
                "ttft_s_median": med([p["ttft_s"] for p in rows]), "ttft_s_all": [round(p["ttft_s"], 2) for p in rows if p["ttft_s"]],
                "decode_tps_median": med([p["decode_tps"] for p in rows]),
                "decode_tps_all": [round(p["decode_tps"], 2) for p in rows if p["decode_tps"]],
                "text_md5": sorted({p["text_md5"][:12] for p in rows})}
    taps = {}
    for l in legs:
        if "tap" in l: taps.setdefault(l["engine"], l["tap"])  # the first leg of each engine
    if len(taps) == 2:
        s["ids"] = compare_ids(*taps.values(), *taps.keys())
    s["peak_rss_mib"] = {"%d-%s" % (i + 1, l["engine"]): round(l["peak_rss_mib"]) for i, l in enumerate(legs)}
    return s


def compare_ids(a, b, name_a, name_b):
    """Request i of one engine against request i of the other: same prompt ids, same output ids, first divergence."""
    rows = []
    for i, (x, y) in enumerate(zip(a, b)):
        div = next((k for k, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), None)
        if div is None and len(x["ids"]) != len(y["ids"]): div = min(len(x["ids"]), len(y["ids"]))
        rows.append({"request": i, "same_prompt": x["prompt_md5"] == y["prompt_md5"], "prompt_tokens": [x["prompt_tokens"], y["prompt_tokens"]],
                     "ids_md5": {name_a: x["ids_md5"], name_b: y["ids_md5"]}, "n_ids": [len(x["ids"]), len(y["ids"])],
                     "finish": [x["finish"], y["finish"]], "identical": bool(x["ids"]) and x["ids"] == y["ids"], "first_divergence": div})
    # two empty id lists, or a tap that logged a different number of requests per side, are not an identity result
    return {"requests": rows, "count": [len(a), len(b)], "all_identical": bool(rows) and len(a) == len(b) and all(r["identical"] for r in rows)}


def main():
    global CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--legs", default="K,S,K,S"); ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--default-memory", action="store_true", help="no --expert-memory: Splash's own plan")
    ap.add_argument("--speed-only", action="store_true", help="skip the identity prompts")
    ap.add_argument("--wait-gib", type=float, default=0, help="read-only wait until free+reclaimable >= this, then run")
    ap.add_argument("--run-anyway", action="store_true", help="after the wait cap, run instead of exiting")
    ap.add_argument("--tap", action="store_true", help="record prompt and output token ids through tools/splash_tap.py")
    ap.add_argument("--wait-only", action="store_true", help="only the wait (run it before taking the GPU lock)")
    ap.add_argument("--wait-cap-s", type=float, default=1200)
    ap.add_argument("--lock-timeout-s", type=int, default=600, help="how long to wait for the GPU lock before exiting (like lockf -t)")
    ap.add_argument("--splash-root", default=os.environ.get("SPLASH_ROOT", "~/Documents/nace/splash-1.4.0"), help="Splash 1.4.0's folder (server/, python/, engine/); default $SPLASH_ROOT")
    ap.add_argument("--binary", help="Splash's engine binary; default <splash-root>/engine/splash")
    ap.add_argument("--model-dir", default=os.environ.get("SPLASH_MODEL_DIR"),
                    help="Splash's resolved model folder (.../models/.resolved/<hash>); default $SPLASH_MODEL_DIR")
    a = ap.parse_args()
    global ENGINE_ARGS, IDENTITY, TAP, SPLASH_ROOT, SPLASH_BIN, SPLASH_PY, MODEL_DIR
    SPLASH_ROOT = os.path.abspath(os.path.expanduser(a.splash_root))
    SPLASH_BIN = os.path.abspath(os.path.expanduser(a.binary)) if a.binary else SPLASH_ROOT + "/engine/splash"
    SPLASH_PY = SPLASH_ROOT + "/python/bin/python3"
    MODEL_DIR = os.path.abspath(os.path.expanduser(a.model_dir)) if a.model_dir else ""
    if not a.wait_only:
        for what, path, isdir in (("--binary", SPLASH_BIN, False), ("--splash-root's python", SPLASH_PY, False), ("--model-dir (or $SPLASH_MODEL_DIR)", MODEL_DIR, True)):
            if not (os.path.isdir(path) if isdir else os.access(path, os.X_OK)):
                sys.exit("splash_ab: %s is not usable: %r" % (what, path))
    TAP = a.tap
    if a.default_memory: ENGINE_ARGS = DEFAULT_MEMORY_ARGS
    if a.speed_only: IDENTITY = []
    if a.wait_gib: wait_for_memory(a.wait_gib, a.wait_cap_s, a.run_anyway)
    if a.wait_only: return
    if not (os.environ.get("NI_LOCK_HELD") or os.environ.get("NACE_GPU_LOCK_HELD")):
        lock = open(os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock"), "a")  # held until this process ends
        print("waiting for the GPU lock (up to %d s) ..." % a.lock_timeout_s, flush=True)
        end = time.time() + a.lock_timeout_s
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
            except OSError:
                if time.time() >= end:
                    raise SystemExit("could not take the GPU lock in %d s: another run holds it. If you started this under `lockf -k ...`"
                                     " (the wrapper holds it, and this script's own flock then waits behind it for good), export NI_LOCK_HELD=1 "
                                     "for the script, or drop the wrapper." % a.lock_timeout_s)
                time.sleep(2)
    CORPUS = corpus()
    os.makedirs(a.out, exist_ok=True)
    busy = running_model_processes()
    if busy: raise SystemExit("a model process is already running, not starting:\n" + "\n".join(busy))
    free_at_start = reclaimable_gib()
    print("free+reclaimable at start: %.1f GiB" % free_at_start, flush=True)
    legs = []
    for i, e in enumerate(a.legs.split(","), 1):
        print("leg %d (%s) ..." % (i, e), flush=True)
        r = run_leg(e, i, a.out, a.reps, a.smoke); legs.append(r)
        print("  up %.1fs, peak RSS %.0f MiB, child left: %s" % (r["up_s"], r["peak_rss_mib"], r["child_left"]), flush=True)
        if r["child_left"]: raise SystemExit("child not reaped: %s" % r["child_left"])
    if a.smoke:
        print(json.dumps(legs[0]["smoke"], indent=1)); return
    s = summarize(legs)
    s["free_reclaimable_gib_at_start"] = round(free_at_start, 1)
    json.dump(s, open(a.out + "/summary.json", "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
