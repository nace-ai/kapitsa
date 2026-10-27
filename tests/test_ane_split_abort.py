#!/usr/bin/env python3
"""Engine::wait_split's bound on a split prefill chunk (src/engine/engine.cpp, issue #426): a chunk whose Neural Engine
job never reports must not keep the GPU - and the request - waiting on it. Needs a dense model, the GPU and the Neural
Engine (`kapitsa serve` prints whether the split is on for the model): it takes the GPU lock ($BENCH_GPU_LOCK, default
/tmp/kapitsa-gpu.lock) unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1 says the caller holds it. Every leg is a fresh
server with KAPITSA_HOME in a temporary folder (the split is per process and the RAM and SSD prefix caches would answer
a leg's prompt from an earlier leg's), greedy replies, thinking off. A leg first sends a one-token request: the engine
builds the split from its chat path (Engine::chat's ane_poll), not when the model loads.
  0. NI_ANE=0: the two prompts' replies on the GPU alone, the reference for leg 2's rerun (leg 1's are not compared:
     the split runs the FFN in int8 on the Neural Engine, so its rows are an approximation)
  1. the split on: both prompts run split chunks, and the log's "prefill: Neural Engine busy ... evaluations" says
     they held (no chunk was given up)
  2. the fault (NI_ANE_FAULT=0: the ANE job of layer 0 of the first split chunk never starts, NI_ANE_VERIFY=0 so the
     startup check does not consume it) with NI_ANE_SPLIT_WAIT_MS=250: the bound runs out long before the per-job
     watchdog, the split is given up ("has not finished in 250 ms"), the chunk reruns on the GPU alone, and both
     prompts give the control's replies again. Each of the two lines is logged once; the chunk's rerun is logged
     right after the bound (the ~2 s in between is Handoff::finish's own bound for a job that never reports); and
     after the first prompt the split is off for the rest of the process - no split chunk and no second bound -
     which is what #426 asks the branch to leave behind.
The bound is shortened because the shipped 60 s is not a test's to wait out; with no fault it does not fire at all (the
2 s per-job watchdog retires a job that started and never reported, so the bound is the backstop for a chunk stuck
before that watchdog).
usage: test_ane_split_abort.py <kapitsa> <model dir> [--port 8199] [--rows 700] [--tokens 24] [--split-ms 250] [--keep]
  NI_ANE_H4096=1: also exercise the experimental 4096-hidden split (for example Qwen3.5-9B)
  --rows: tokens of each leg's prompt (over the split's floor, under one prefill chunk)
  --keep: keep the temporary folder (the three servers' logs)
With no arguments it prints "SKIP" and exits 0: CI has no model folder, GPU or Neural Engine, so it must not fail
there (a run that typed the arguments wrong still gets this usage message and exit 1)."""
import fcntl, json, os, re, shutil, signal, subprocess, sys, tempfile, time, urllib.error, urllib.request

ARGS = sys.argv[1:]


def opt(name, default=None, flag=False):
    if name in ARGS:
        i = ARGS.index(name)
        if flag:
            del ARGS[i]
            return True
        v = ARGS[i + 1]
        del ARGS[i:i + 2]
        return v
    return default


PORT = int(opt("--port", "8199"))
ROWS = int(opt("--rows", "700"))
TOKENS = int(opt("--tokens", "24"))
SPLIT_MS = opt("--split-ms", "250")
KEEP = opt("--keep", flag=True)
if not SPLIT_MS.isdigit():
    sys.exit(__doc__)
if not ARGS:  # CI has no model folder, GPU or Neural Engine: skip, do not fail a run that has nothing to test
    print("SKIP: this test needs a binary and a model folder, on a Mac with the GPU and its Neural Engine", flush=True)
    sys.exit(0)
if len(ARGS) != 2:
    sys.exit(__doc__)
KAPITSA, MODEL = os.path.abspath(ARGS[0]), os.path.abspath(os.path.expanduser(ARGS[1]))
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "bench", "harness", "corpus", "bench_corpus.txt")
BASE = f"http://127.0.0.1:{PORT}"
fails = 0


def check(ok, name, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else "\n     " + str(detail)), flush=True)
    fails += 0 if ok else 1


def gpu_lock():
    if os.environ.get("NACE_GPU_LOCK_HELD") == "1" or os.environ.get("NI_LOCK_HELD") == "1":
        return None
    path = os.environ.get("BENCH_GPU_LOCK") or os.path.expanduser("/tmp/kapitsa-gpu.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = open(path, "a")
    print(f"waiting for the GPU lock {path} ...", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


SERVERS = []  # every server this run started, stopped again in the finally: a failed check must not leave one behind


class Server:
    """a `kapitsa serve` on BASE with its own KAPITSA_HOME; `env` is added to the leg's environment"""

    def __init__(self, root, leg, env):
        self.home = os.path.join(root, leg)
        self.log_path = os.path.join(root, leg + ".log")
        os.makedirs(self.home, exist_ok=True)
        e = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
        if "NI_ANE_H4096" in os.environ:
            e["NI_ANE_H4096"] = os.environ["NI_ANE_H4096"]
        e.update({"KAPITSA_HOME": self.home, "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1"})
        e.update(env)
        self.log = open(self.log_path, "w")
        self.stopped = False
        self.start_at = 0
        self.p = subprocess.Popen([KAPITSA, "serve", "--model", MODEL, "--port", str(PORT), "--ctx", "32768",
                                   "--slots", "1", "--disk-cache", "0"], env=e, stdout=self.log,
                                  stderr=subprocess.STDOUT)
        SERVERS.append(self)
        t0 = time.time()
        while time.time() - t0 < 600:
            if self.p.poll() is not None:
                raise SystemExit(f"the server of leg {leg} exited ({self.p.returncode}); its log: {self.log_path}")
            try:
                if urllib.request.urlopen(BASE + "/health", timeout=2).status == 200:
                    return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        raise SystemExit(f"the server of leg {leg} did not start in 600 s")

    def text(self):
        with open(self.log_path, errors="replace") as f:
            return f.read()

    def mark(self):
        return len(self.text())

    def wait_log(self, pattern, timeout, since=None):
        """the first match of `pattern` after offset `since` (default: now), or None after timeout s"""
        start = self.mark() if since is None else since
        t0 = time.time()
        while time.time() - t0 < timeout:
            m = re.search(pattern, self.text()[start:])
            if m:
                return m
            time.sleep(0.25)
        return None

    def stop(self):
        if self.stopped:
            return
        self.stopped = True
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def chat(messages, max_tokens=TOKENS, timeout=180):
    """(content, completion tokens, seconds) of a streamed greedy chat completion"""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0, text, usage = time.time(), "", None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            c = json.loads(line[6:])
            if c.get("usage"):
                usage = c["usage"]
            for ch in c.get("choices", []):
                text += ch.get("delta", {}).get("content") or ""
    return text, (usage or {}).get("completion_tokens"), time.time() - t0


with open(CORPUS, errors="replace") as f:
    CORPUS_TEXT = f.read()


def prompt(k):
    """prompt k: about ROWS tokens (4.25 characters each) of the corpus and a question"""
    n = int(ROWS * 4.25)
    off = (k * 7919) % max(1, len(CORPUS_TEXT) - n)
    return [{"role": "user", "content": "Here is some source code:\n\n" + CORPUS_TEXT[off:off + n] +
                                        "\n\nIn two sentences: what does the first class in it do?"}]


def warm(server):
    """the request that makes the engine build the split (Engine::chat calls ane_poll): small enough not to split, so an
    injected fault stays unused"""
    chat([{"role": "user", "content": "hi"}], max_tokens=1)


def ane_line(server, since, timeout):
    """waits for the split to be on (the server compiles its program in the background)"""
    m = server.wait_log(r"NI_ANE: Neural Engine FFN split[^:]*: \d+/\d+ units", timeout, since=since)
    if m is None:
        for why in (r"NI_ANE: (Neural Engine unavailable[^\n]*)", r"NI_ANE: (no calibration[^\n]*)",
                    r"NI_ANE: (releasing the Neural Engine split[^\n]*)"):
            if re.search(why, server.text()[since:]):
                raise SystemExit("SKIP: this Mac or model has no Neural Engine split: " +
                                 re.search(why, server.text()[since:]).group(1))
        raise SystemExit("the split was not on within %d s; the log: %s" % (timeout, server.log_path))


# 512 rows (the default calibration) so the 700-row prompt's chunk splits and takes the single-lane schedule: on that
# schedule the GPU's command buffer waits on the job, which is what a bound has to release. The wavefront schedule (from
# 256 rows: two row halves) does not wait on the GPU side, so the chunk's own Handoff::finish bound gives it up instead.
MIN_ROWS = {"NI_ANE_MIN_ROWS": "512"}
lock, root = gpu_lock(), tempfile.mkdtemp(prefix="kap-ane-abort-")
print(f"logs in {root}", flush=True)
try:
    # 0. the GPU alone: the reference replies
    s0 = Server(root, "gpu", dict(MIN_ROWS, NI_ANE="0"))
    want = []
    for k in range(2):
        text, n, secs = chat(prompt(k))
        want.append((text, n))
        check(bool(text) and n >= 4, f"gpu alone: prompt {k} replies ({n} tokens, {secs:.1f} s)", text[:200])
    check(not re.search(r"Neural Engine busy", s0.text()), "gpu alone: no split chunk in the log")
    s0.stop()

    # 1. the split on: both prompts hold
    s1 = Server(root, "split", MIN_ROWS)
    m1 = s1.mark()
    warm(s1)
    ane_line(s1, m1, 300)
    for k in range(2):
        mark = s1.mark()
        text, n, secs = chat(prompt(k))
        busy = s1.wait_log(r"prefill: Neural Engine busy [\d.]+% of [\d.]+ ms over (\d+) split chunks "
                           r"\((\d+) evaluations\)", 60, since=mark)
        check(busy is not None and int(busy.group(2)) > 0,
              f"split: prompt {k} ran split chunks that held ({secs:.1f} s)",
              busy.group(0) if busy else s1.text()[mark:][-1500:])
    check(not re.search(r"has not finished|split stopped", s1.text()),
          "split: no chunk was given up",
          "\n     ".join(re.findall(r"NI_ANE: [^\n]*", s1.text())))
    s1.stop()

    # 2. the fault: the bound gives up the split, the chunk reruns, the replies do not change
    s2 = Server(root, "fault", dict(MIN_ROWS, NI_ANE_FAULT="0", NI_ANE_VERIFY="0",
                                    NI_ANE_SPLIT_WAIT_MS=SPLIT_MS))
    m2 = s2.mark()
    warm(s2)
    ane_line(s2, m2, 300)
    check(re.search(r"NI_ANE_FAULT: the ANE job of layer 0 ", s2.text()[m2:]) is not None,
          "fault: the injected fault is armed")
    for k in range(2):
        mark = s2.mark()
        try:
            text, n, secs = chat(prompt(k))
        except (urllib.error.URLError, OSError) as e:
            check(False, f"fault: prompt {k} returned ({type(e).__name__}: {e})", s2.text()[mark:][-2000:])
            continue
        check((text, n) == want[k],
              f"fault: prompt {k} gives the GPU-alone reply again ({n} tokens, {secs:.1f} s)",
              f"got {text[:300]!r} ({n} tokens)\n     want {want[k][0][:300]!r} ({want[k][1]} tokens)")
        # the given-up chunk costs the bound plus Handoff::finish's ~2 s and the GPU rerun, not the shipped 60 s
        check(secs < 10, f"fault: prompt {k} came back right after the bound ({secs:.1f} s)")
        if k == 0:
            bound = s2.wait_log(r"NI_ANE: a split prefill chunk has not finished in %s ms; giving up the Neural Engine "
                                r"split" % SPLIT_MS, 30, since=mark)
            check(bound is not None, f"fault: the {SPLIT_MS} ms bound gave the chunk up",
                  s2.text()[mark:][-2000:])
            rerun = s2.wait_log(r"Neural Engine FFN split stopped \(a split chunk did not finish within %s ms\); this "
                                r"chunk reruns on the GPU" % SPLIT_MS, 30, since=mark)
            check(rerun is not None, "fault: the abort, not the per-job watchdog, retired the split",
                  "\n     ".join(re.findall(r"NI_ANE: [^\n]*", s2.text()[mark:])))
            busy = s2.wait_log(r"prefill: Neural Engine busy [\d.]+% of [\d.]+ ms over (\d+) split chunks "
                               r"\(0 evaluations\)", 30, since=mark)
            check(busy is not None, "fault: the given-up chunk counted no evaluation",
                  s2.text()[mark:][-1500:])
            check(s2.text().count("has not finished in %s ms" % SPLIT_MS) == 1 and
                  s2.text().count("split stopped") == 1,
                  "fault: the bound and the rerun were reported once each",
                  "\n     ".join(re.findall(r"NI_ANE: [^\n]*", s2.text()[mark:])))
        else:
            check(not re.search(r"has not finished|split stopped", s2.text()[mark:]),
                  "fault: the second prompt does not split (and so does not bound) again",
                  "\n     ".join(re.findall(r"NI_ANE: [^\n]*", s2.text()[mark:])))
            check(not re.search(r"Neural Engine busy", s2.text()[mark:]),
                  "fault: no split chunk after the split was given up",
                  s2.text()[mark:][-1500:])
    s2.stop()
finally:
    for server in SERVERS:
        server.stop()
    if KEEP:
        print(f"kept {root}")
    else:
        shutil.rmtree(root, ignore_errors=True)
    if lock:
        lock.close()

print(("FAILED: %d check(s)" % fails) if fails else "OK", flush=True)
sys.exit(1 if fails else 0)
