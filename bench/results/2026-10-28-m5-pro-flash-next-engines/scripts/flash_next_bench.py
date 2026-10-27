#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (Q2_0) on one Mac, four engines, one at a time, through the same OpenAI chat endpoint and the same
frozen prompts (prompts/ctx_*.txt). Run each leg under the GPU lock (lockf -k "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" ...): a leg launches the
server, measures, reaps it; it refuses to start when another model server is running.

  flash_next_bench.py leg --arm K|S|N|L --idx I --out DIR [--reps 3] [--smoke]
  flash_next_bench.py summarize DIR

Arms: K = `kapitsa serve --engine splash` (Splash 1.4.0 child), S = plain Splash 1.4.0 (`python -m server.server`),
N = Kapitsa's native engine (MTP), L = llama.cpp llama-server on the same Q2_0 GGUFs Splash uses.
Per leg: warm-up, then REPS each of: decode 256 tokens at T=0 and T=0.6; cold prefill at ~1K/4K/16K prompt tokens (a nonce
at the start of every prompt, so no prefix-cache hit); the second turn of a ~4K tool-call chat (the first turn's reasoning is
not sent back, as clients do). TTFT is the time to the first streamed token (reasoning or content); decode tok/s is
(completion_tokens-1)/(end-first token); "prompt tokens / TTFT" (tok/s) is uncached prompt tokens / TTFT, including the fixed per-request cost.
Memory: peak RSS of the server's process tree (it does not count GPU-wired pages) and, system-wide, the lowest
free+reclaimable memory, highest wired memory, swap and pressure level while the leg ran."""
import tempfile, argparse, hashlib, json, os, re, signal, socket, statistics, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
NACE = HOME + "/Documents/nace"
SPLASH_ROOT = NACE + "/splash-1.4.0"
SPLASH_BIN = SPLASH_ROOT + "/engine/splash"
SPLASH_PY = SPLASH_ROOT + "/python/bin/python3"
SPLASH_MODEL = HOME + "/Library/Application Support/Splash/models/.resolved/652799347ee912b7acd050042b20d3677ca2163103dc90e7d85138daa15e0c3e"
SPLASH_MODEL_ID = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF:Q2_0"
GGUF = SPLASH_MODEL + "/target/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
NATIVE_MODEL = NACE + "/kapitsa-m5pro/models/flash-next-q20"
BIN = {  # overridable with --kbin / --nbin / --llama
    "K": NACE + "/kapitsa-splash-ba/after/build/kapitsa",   # splash-6-ondemand 7e478053
    "N": NACE + "/kapitsa-bench-main/build/kapitsa",        # main e840651c
    "L": NACE + "/llama.cpp/build/bin/llama-server",        # c7474afc
}
SPLASH_FLAGS = ["--expert-memory", "17179869184", "--idle-release", "off"]  # K and S alike; every other Splash flag is its default
CTX = "32768"
SIZES = {"1K": 4000, "4K": 16400, "16K": 65600}  # characters of context in the system message
DECODE_PROMPT = "Write the whole numbers from 1 to 1000 in order, separated by spaces. Output only the numbers, nothing else."
DECODE_TOKENS = 256
PREFILL_TOKENS = 16
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository and return its text.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def ctx_text(size):
    return open("%s/prompts/ctx_%s.txt" % (RES, size)).read()


def messages(nonce, size, question):
    system = "[session %s] Answer in one short sentence." % nonce
    if size: system += "\n\n# Repository context\n" + ctx_text(size)
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def running_model_processes():
    busy = []
    for l in sh("ps", "-axo", "pid,command").splitlines()[1:]:
        pid, _, cmd = l.strip().partition(" ")
        words = cmd.split()
        if not words or "flash_next_bench" in cmd: continue
        exe = os.path.basename(words[0])
        if (exe == "splash" and "serve-native" in words[:3]) or (exe == "kapitsa" and words[1:2] in (["serve"], ["run"], ["bench"])) \
                or exe in ("llama-server", "llama-cli", "ni_run") or (exe.startswith(("python", "Python")) and "server.server" in words[1:5]):
            busy.append(l.strip())
    return busy


def vm():
    out = sh("vm_stat")
    page = int(re.search(r"page size of (\d+)", out).group(1))
    g = lambda k: int(re.search(r"%s:\s+(\d+)" % k, out).group(1)) * page / 2**30
    return {"reclaimable_gib": sum(g(k) for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")),
            "wired_gib": g("Pages wired down")}


def sysctl_int(name):
    try: return int(sh("sysctl", "-n", name).strip())
    except ValueError: return None


def swap_gb():
    m = re.search(r"used = ([\d.]+)([MG])", sh("sysctl", "-n", "vm.swapusage"))
    return 0.0 if not m else float(m.group(1)) / (1024 if m.group(2) == "M" else 1)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Leg:
    def __init__(self, arm, log, a):
        self.arm, self.port, self.a = arm, free_port(), a
        self.rss_peak, self.aborted = 0, None
        self.fp_peak, self.sys_used_at_fp_peak, self.fp_samples, self.fp_pids = 0, None, 0, 0
        self.memsize_gib = sysctl_int('hw.memsize') / 2**30
        self.swap0 = self.swap_max = swap_gb()
        v = vm(); self.free_start, self.free_min, self.wired_max = v["reclaimable_gib"], v["reclaimable_gib"], v["wired_gib"]
        self.level_max, self.pct_min = 1, 100
        env = dict(os.environ, HF_HUB_OFFLINE="1")
        if arm == "N":  # pinned like the 2026-10-24 bench, for memory headroom; --nenv none = Kapitsa's auto size
            for kv in a.nenv.split(","):
                if kv != "none": k, _, v = kv.partition("="); env[k] = v
        self.model = None
        if arm == "K":
            cmd = [a.kbin, "serve", "--engine", "splash", "--splash-binary", SPLASH_BIN, "--splash-model", SPLASH_MODEL,
                   "--context", "32k", "--private", "--port", str(self.port)]
            for f in SPLASH_FLAGS: cmd += ["--splash-arg", f]
        elif arm == "S":
            cmd = [SPLASH_PY, "-P", "-m", "server.server", SPLASH_MODEL, "--model", SPLASH_MODEL_ID, "--binary", SPLASH_BIN,
                   "--port", str(self.port), "--max-context", CTX] + SPLASH_FLAGS
            env["PYTHONPATH"] = SPLASH_ROOT; self.model = SPLASH_MODEL_ID
        elif arm == "N":
            cmd = [a.nbin, "serve", "--model", NATIVE_MODEL, "--context", "32k", "--private", "--port", str(self.port)]
        else:
            cmd = [a.llama, "-m", GGUF, "-ngl", "99", "-fa", "on", "-c", "33280", "-np", "1", "--jinja",
                   "--host", "127.0.0.1", "--port", str(self.port), "--no-webui"]
        self.cmd = cmd
        self.logf = open(log, "w")
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, env=env, stdout=self.logf, stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=SPLASH_ROOT if arm == "S" else RES)
        self.stop = False
        self.sampler = threading.Thread(target=self.sample, daemon=True); self.sampler.start()
        for _ in range(3600):
            if self.proc.poll() is not None: raise SystemExit("arm %s exited early (status %s); see %s" % (arm, self.proc.returncode, log))
            if self.aborted: raise SystemExit("aborted while loading: %s" % self.aborted)
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/v1/models" % self.port, timeout=1).read(); break
            except Exception: time.sleep(0.25)
        else: raise SystemExit("arm %s did not come up in 900 s" % arm)
        self.up_s = time.time() - self.t0

    def tree_footprint_bytes(self):
        """phys_footprint of the serving process tree (`footprint`), one number for every arm."""
        pids = sorted(self.tree_pids())
        out = os.path.join(tempfile.gettempdir(), "fnb-fp-%d.json" % os.getpid())
        subprocess.run(["footprint", "-j", out] + [str(p) for p in pids], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        self.fp_pids = max(self.fp_pids, len(pids))
        return int(json.load(open(out))["total footprint"])

    def tree_pids(self):
        return self._tree(None)

    def tree_rss_kib(self):
        return self._tree("rss")

    def _tree(self, what):
        rows = {}
        for l in sh("ps", "-axo", "pid=,ppid=,rss=").splitlines():
            p = l.split()
            if len(p) == 3: rows[int(p[0])] = (int(p[1]), int(p[2]))
        kids, total = {self.proc.pid}, 0
        for _ in range(8):
            kids |= {pid for pid, (pp, _) in rows.items() if pp in kids}
        if what is None: return {p for p in kids if p in rows}
        return sum(rows[p][1] for p in kids if p in rows)

    def sample(self):
        n = 0
        while not self.stop:
            n += 1
            try: self.rss_peak = max(self.rss_peak, self.tree_rss_kib())
            except Exception: pass
            if n % 2 == 0:  # about every 1 s
                try:
                    fp = self.tree_footprint_bytes(); self.fp_samples += 1
                    if fp > self.fp_peak:
                        self.fp_peak = fp; self.sys_used_at_fp_peak = self.memsize_gib - vm()["reclaimable_gib"]
                except Exception: pass
            if n % 4 == 0: self.watch()
            time.sleep(0.5)

    def watch(self):
        """Relaxed guard: stop the leg only at pressure level 4, free < 3% or swap > 16 GB."""
        v = vm()
        self.free_min = min(self.free_min, v["reclaimable_gib"]); self.wired_max = max(self.wired_max, v["wired_gib"])
        self.swap_max = max(self.swap_max, swap_gb())
        lvl, pct = sysctl_int("kern.memorystatus_vm_pressure_level"), sysctl_int("kern.memorystatus_level")
        self.level_max = max(self.level_max, lvl or 1)
        if pct is not None: self.pct_min = min(self.pct_min, pct)
        why = ("pressure level 4" if lvl == 4 else "free %d%% < 3%%" % pct if pct is not None and pct < 3
               else "swap %.1f GB > 16" % self.swap_max if self.swap_max > 16 else None)
        if why and not self.aborted:
            self.aborted = why
            print("ABORT: %s" % why, flush=True)
            try: os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass

    def chat(self, msgs, max_tokens, temperature=0.0, tools=None, cache=True):
        body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "stream": True,
                "stream_options": {"include_usage": True}}
        if temperature == 0: body["seed"] = 1
        if self.model: body["model"] = self.model
        if tools: body["tools"] = tools
        if self.arm == "L" and not cache: body["cache_prompt"] = False
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % self.port, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = time.time(); first = None; reasoning, content, usage, timings, finish = [], [], None, None, None
        calls = {}
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if d.get("usage"): usage = d["usage"]
                if d.get("timings"): timings = d["timings"]
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    c = delta.get("content") or ""
                    for tc in delta.get("tool_calls") or []:
                        e = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        e["id"] = tc.get("id") or e["id"]
                        f = tc.get("function") or {}
                        e["name"] += f.get("name") or ""; e["arguments"] += f.get("arguments") or ""
                        if first is None: first = time.time()
                    if (rc or c) and first is None: first = time.time()
                    reasoning.append(rc); content.append(c)
                    finish = ch.get("finish_reason") or finish
        end = time.time()
        r_, c_ = "".join(reasoning), "".join(content)
        n = (usage or {}).get("completion_tokens")
        pt = (usage or {}).get("prompt_tokens")
        cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached is None and timings: cached = timings.get("cache_n")
        return {"reasoning": r_, "content": c_, "finish": finish, "tool_calls": [calls[k] for k in sorted(calls)],
                "prompt_tokens": pt, "completion_tokens": n, "cached_tokens": cached or 0,
                "ttft_s": None if first is None else first - t0, "total_s": end - t0,
                "decode_tps": None if not n or n < 2 or first is None or end <= first else (n - 1) / (end - first),
                "text_md5": hashlib.md5((r_ + "\x00" + c_).encode()).hexdigest(), "engine_timings": timings}

    def two_turn(self, nonce, size):
        msgs = messages(nonce, size, "Use the read_file tool to read docs/usage.md, then tell me in one sentence what it covers.")
        msgs[0]["content"] = msgs[0]["content"].replace("Answer in one short sentence.", "Always call a tool when the user asks you to read a file.")
        t1 = self.chat(msgs, 512, 0.0, TOOLS)
        out = {"turn1": {k: t1[k] for k in ("finish", "ttft_s", "prompt_tokens", "completion_tokens", "cached_tokens")},
               "turn1_reasoning_chars": len(t1["reasoning"])}
        if not t1["tool_calls"]: out["error"] = "turn 1 made no tool call (finish %s)" % t1["finish"]; return out
        tc = t1["tool_calls"][0]
        msgs += [{"role": "assistant", "content": t1["content"] or None,
                  "tool_calls": [{"id": tc["id"] or "call_0", "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}]},
                 {"role": "tool", "tool_call_id": tc["id"] or "call_0", "content": "# Usage\nRun `kapitsa serve` to start the server; `kapitsa run` for one prompt."}]
        t2 = self.chat(msgs, 128, 0.0, TOOLS)
        out["turn2"] = {k: t2[k] for k in ("finish", "ttft_s", "prompt_tokens", "completion_tokens", "cached_tokens", "decode_tps")}
        return out

    def close(self):
        self.stop = True
        try: os.killpg(self.proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError): pass  # already killed by the memory watch
        try: self.proc.wait(60)
        except subprocess.TimeoutExpired:
            try: os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError): pass
            self.proc.wait(10)
        time.sleep(2)


def slim(r):
    return {k: r[k] for k in ("finish", "prompt_tokens", "completion_tokens", "cached_tokens", "ttft_s", "total_s", "decode_tps", "text_md5")} \
        | ({"engine_timings": r["engine_timings"]} if r.get("engine_timings") else {})


def attempt(label, fn, errors):
    try: return fn()
    except Exception as e:
        errors.append("%s: %r" % (label, e)); print("  ERROR %s: %r" % (label, e), flush=True); return None


def run_leg(a):
    busy = running_model_processes()
    if busy: raise SystemExit("a model process is already running, not starting:\n" + "\n".join(busy))
    os.makedirs(a.out, exist_ok=True)
    tag = "leg%d-%s" % (a.idx, a.arm)
    leg = Leg(a.arm, "%s/%s.log" % (a.out, tag), a)
    res = {"arm": a.arm, "group": a.group or a.arm, "binary": binary_rev(a), "idx": a.idx, "cmd": leg.cmd, "env": a.nenv if a.arm == "N" else None, "up_s": leg.up_s, "free_reclaimable_gib_at_start": leg.free_start,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "decode": {"T0": [], "T06": []}, "prefill": {s: [] for s in SIZES},
           "twoturn": [], "errors": []}
    err = res["errors"]
    try:
        print("  up %.1fs" % leg.up_s, flush=True)
        w = attempt("warmup", lambda: leg.chat(messages("warm%d" % a.idx, "1K", "Hello."), 8, cache=False), err)
        print("  warm-up ttft %s" % (w and round(w["ttft_s"], 2)), flush=True)
        if a.smoke:
            r = attempt("smoke", lambda: leg.chat(messages("smoke", None, "Explain in two sentences why the sky is blue."), 64, cache=False), err)
            res["smoke"] = r; print(json.dumps(r, indent=1)[:1500]); return finish(leg, res, a, tag)
        for rep in range(a.reps):
            nonce = "b%d-%d" % (a.idx, rep)
            for key, temp in (("T0", 0.0), ("T06", 0.6)):
                r = attempt("decode %s rep %d" % (key, rep), lambda: leg.chat(messages(nonce + key, None, DECODE_PROMPT), DECODE_TOKENS, temp, cache=False), err)
                if r: res["decode"][key].append(slim(r)); print("  decode %s %.1f tok/s (%s tok, %s)" % (key, r["decode_tps"] or 0, r["completion_tokens"], r["finish"]), flush=True)
            for size in SIZES:
                r = attempt("prefill %s rep %d" % (size, rep), lambda: leg.chat(messages(nonce + size, size, "What does this repository do?"), PREFILL_TOKENS, 0.0, cache=False), err)
                if r:
                    r = slim(r); pt, c = r["prompt_tokens"], r["cached_tokens"]
                    r["prefill_tps"] = (pt - c) / r["ttft_s"] if pt and r["ttft_s"] else None
                    res["prefill"][size].append(r); print("  prefill %s: %s tok, ttft %.2fs, %.0f tok/s, cached %s" % (size, pt, r["ttft_s"] or 0, r["prefill_tps"] or 0, c), flush=True)
            t = attempt("twoturn rep %d" % rep, lambda: leg.two_turn(nonce + "tt", "4K"), err)
            if t:
                res["twoturn"].append(t)
                print("  twoturn: %s" % (t.get("error") or "turn2 ttft %.2fs cached %s/%s" % (t["turn2"]["ttft_s"], t["turn2"]["cached_tokens"], t["turn2"]["prompt_tokens"])), flush=True)
            if leg.aborted: break
    except BaseException:
        finish(leg, res, a, tag)
        raise
    return finish(leg, res, a, tag)


def finish(leg, res, a, tag):
    leg.close()
    res["memory"] = {"peak_rss_mib": leg.rss_peak / 1024, "peak_footprint_mib": leg.fp_peak / 2**20, "system_used_gib_at_footprint_peak": leg.sys_used_at_fp_peak,
                     "footprint_samples": leg.fp_samples, "footprint_max_processes": leg.fp_pids, "free_reclaimable_gib_start": leg.free_start, "free_reclaimable_gib_min": leg.free_min,
                     "wired_gib_max": leg.wired_max, "swap_gb_start": leg.swap0, "swap_gb_max": leg.swap_max,
                     "pressure_level_max": leg.level_max, "memorystatus_level_min": leg.pct_min, "aborted": leg.aborted,
                     "swap_grew_over_1gb": leg.swap_max - leg.swap0 > 1.0}
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(res, open("%s/%s.json" % (a.out, tag), "w"), indent=1)
    print("  peak footprint %.0f MiB (%d samples), peak RSS %.0f MiB, free+reclaimable min %.1f GiB, wired max %.1f GiB, swap +%.2f GB, pressure %d%s" % (
        res["memory"]["peak_footprint_mib"], leg.fp_samples, res["memory"]["peak_rss_mib"], leg.free_min, leg.wired_max, leg.swap_max - leg.swap0, leg.level_max,
        ", ABORTED " + leg.aborted if leg.aborted else ""), flush=True)
    return res


def binary_rev(a):
    exe = {"K": a.kbin, "N": a.nbin, "L": a.llama}.get(a.arm)
    if not exe: return {"path": SPLASH_BIN, "version": "Splash 1.4.0 (no --version flag; folder splash-1.4.0)",
                      "sha256": hashlib.sha256(open(SPLASH_BIN, "rb").read()).hexdigest()}
    out = subprocess.run([exe, "--version"], capture_output=True, text=True)
    return {"path": exe, "version": (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else None}


def stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return {"n": len(vals), "median": statistics.median(vals), "min": min(vals), "max": max(vals), "values": vals}


def summarize(d, meta=None):
    legs = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if re.match(r"leg\d+-[KSNL]\.json$", f)]
    legs = [l for l in legs if "smoke" not in l]  # the smoke leg is kept as raw data only
    legs.sort(key=lambda l: l["idx"])
    out = {"meta": json.load(open(d + "/meta.json")) if os.path.exists(d + "/meta.json") else {}, "order": [l["arm"] for l in legs], "arms": {}}
    for arm in ("K", "K2", "S", "S2", "N", "L"):
        ls = [l for l in legs if l.get("group", l["arm"]) == arm]
        if not ls: continue
        c = {}
        c["decode_T0_tps"] = stat([r["decode_tps"] for l in ls for r in l["decode"]["T0"] if (r["completion_tokens"] or 0) >= 250])
        c["decode_T06_tps"] = stat([r["decode_tps"] for l in ls for r in l["decode"]["T06"] if (r["completion_tokens"] or 0) >= 250])
        for s in SIZES:
            rows = [r for l in ls for r in l["prefill"][s]]
            c["ttft_%s_s" % s] = stat([r["ttft_s"] for r in rows])
            c["prefill_%s_tps" % s] = stat([r["prefill_tps"] for r in rows])
            c["prompt_tokens_%s" % s] = stat([r["prompt_tokens"] for r in rows])
            c["cached_tokens_%s" % s] = stat([r["cached_tokens"] for r in rows])
        tt = [t for l in ls for t in l["twoturn"] if "turn2" in t]
        c["turn2_ttft_s"] = stat([t["turn2"]["ttft_s"] for t in tt])
        c["turn2_prompt_tokens"] = stat([t["turn2"]["prompt_tokens"] for t in tt])
        c["turn2_cached_tokens"] = stat([t["turn2"]["cached_tokens"] for t in tt])
        mem = [l["memory"] for l in ls]
        c["memory"] = {"peak_rss_mib_max": max(m["peak_rss_mib"] for m in mem), "free_reclaimable_gib_min": min(m["free_reclaimable_gib_min"] for m in mem),
                       "free_reclaimable_gib_start": [m["free_reclaimable_gib_start"] for m in mem], "wired_gib_max": max(m["wired_gib_max"] for m in mem),
                       "swap_gb_growth_max": max(m["swap_gb_max"] - m["swap_gb_start"] for m in mem), "pressure_level_max": max(m["pressure_level_max"] for m in mem),
                       "aborted": [m["aborted"] for m in mem if m["aborted"]], "legs": len(ls)}
        fpm = [m for m in mem if m.get("peak_footprint_mib")]
        c["memory"]["footprint_legs"] = len(fpm)
        c["memory"]["peak_footprint_mib_per_leg"] = [round(m["peak_footprint_mib"]) for m in fpm]
        c["memory"]["peak_footprint_mib_max"] = max((m["peak_footprint_mib"] for m in fpm), default=None)
        c["memory"]["system_used_gib_at_footprint_peak_max"] = max((m["system_used_gib_at_footprint_peak"] for m in fpm), default=None)
        c["up_s"] = stat([l["up_s"] for l in ls])
        c["errors"] = [e for l in ls for e in l["errors"]]
        c["cmd"] = ls[0]["cmd"]
        c["binary"] = ls[0].get("binary")
        # Splash legs: --verify-rows is not passed, so the engine default applies; its log line says which mode ran
        vr = set()
        for l in ls:
            try: vr |= set(re.findall(r"--verify-rows (\w+)", open("%s/leg%d-%s.log" % (d, l["idx"], l["arm"])).read()[:20000]))
            except (OSError, KeyError): pass
        if arm in ("K", "K2", "S", "S2"): c["verify_rows_in_engine_log"] = sorted(vr)
        out["arms"][arm] = c
    json.dump(out, open(d + "/summary.json", "w"), indent=1)
    names = {"K": "Kapitsa --engine splash @7e478053", "K2": "Kapitsa --engine splash @ff6151c8", "S": "plain Splash 1.4.0", "S2": "plain Splash 1.4.0 (paired with K2)", "N": "Kapitsa native (MTP)", "L": "llama.cpp"}
    f = lambda c, k, p=1: "not measured" if not c.get(k) else "%.*f (n %d)" % (p, c[k]["median"], c[k]["n"])
    rows = ["| arm | decode T=0 tok/s | decode T=0.6 tok/s | " + " | ".join("TTFT %s s | prompt tokens / TTFT %s (tok/s)" % (s, s) for s in SIZES) + " | turn-2 TTFT s (4K chat) | peak footprint GiB (serving tree) | system used GiB at that peak | system wired GiB (max) | swap growth GB (max) | memory pressure (max) | peak ps RSS MiB (not a memory comparison) |",
            "|" + "---|" * (10 + 2 * len(SIZES))]
    for arm, c in out["arms"].items():
        rows.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (names[arm], f(c, "decode_T0_tps"), f(c, "decode_T06_tps"),
                    " | ".join("%s | %s" % (f(c, "ttft_%s_s" % s, 2), f(c, "prefill_%s_tps" % s, 0)) for s in SIZES),
                    f(c, "turn2_ttft_s", 2), ("not measured" if not c["memory"]["peak_footprint_mib_max"] else "%.1f (%d legs)" % (c["memory"]["peak_footprint_mib_max"] / 1024, c["memory"]["footprint_legs"])),
                    ("not measured" if not c["memory"]["peak_footprint_mib_max"] else "%.1f" % c["memory"]["system_used_gib_at_footprint_peak_max"]),
                    "%.1f" % c["memory"]["wired_gib_max"], "%.2f" % c["memory"]["swap_gb_growth_max"], c["memory"]["pressure_level_max"], "%.0f" % c["memory"]["peak_rss_mib_max"]))
    open(d + "/table.md", "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("leg")
    l.add_argument("--arm", required=True, choices=list("KSNL")); l.add_argument("--idx", type=int, required=True)
    l.add_argument("--out", required=True); l.add_argument("--reps", type=int, default=3); l.add_argument("--smoke", action="store_true")
    l.add_argument("--group", default=None, help="summary group, e.g. K2 for a second Kapitsa build"); l.add_argument("--kbin", default=BIN["K"]); l.add_argument("--nbin", default=BIN["N"]); l.add_argument("--llama", default=BIN["L"])
    l.add_argument("--nenv", default="NI_EXPERT_STORE=24", help="env for the N arm, KEY=VAL[,KEY=VAL] or none")
    s = sub.add_parser("summarize"); s.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "summarize": summarize(a.dir)
    else: run_leg(a)


if __name__ == "__main__":
    main()
