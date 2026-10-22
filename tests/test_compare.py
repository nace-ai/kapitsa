#!/usr/bin/env python3
"""`kapitsa bench compare` on the CPU: no model, no GPU, no engine installed, nothing sent anywhere.

usage: test_compare.py [BUILD_DIR]   (default: build; needs BUILD_DIR/kapitsa)

1. The command finds the harness (KAPITSA_HARNESS, the installed <prefix>/share/kapitsa/bench/harness, or
   bench/harness in the binary's own folder), hands over its arguments, forwards the exit code, and says what to do
   when the harness is not there.
2. bench bench/harness/kbench.py against two scripted OpenAI-compatible servers (configs with no command, so nothing is
   started; the GPU lock is a test file): --list, the run, the table, --json, --report, the saved files, an engine that
   fails (a closed port) and does not stop the others, and an engine that is not installed, listed with how to get it.
3. The choice of configs: tuned variants first, controls (nomtp, bf16, mlx-q4) never, engines whose binary is missing
   told apart from weights that are missing.
"""
import http.server, io, json, os, random, re, shutil, socketserver, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "build"))
KAPITSA = os.path.join(BUILD, "kapitsa")
HARNESS = os.path.join(REPO, "bench", "harness")
sys.path.insert(0, HARNESS)
FAILS = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


def run(args, **kw):
    env = dict(os.environ, **kw.pop("env", {}))
    return subprocess.run(args, capture_output=True, text=True, timeout=kw.pop("timeout", 300), env=env, **kw)


# ---------------------------------------------------------------- scripted engines
class Script(http.server.BaseHTTPRequestHandler):
    """Answers like an OpenAI-compatible server (SSE in HTTP/1.1 chunks); `speed` sets the pause between events."""
    protocol_version = "HTTP/1.1"
    speed = 0.0005

    def log_message(self, *a):
        pass

    def events(self, evs):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        for ev in evs + ["[DONE]"]:
            data = ("data: " + (ev if isinstance(ev, str) else json.dumps(ev)) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
            time.sleep(self.speed)
        self.wfile.write(b"0\r\n\r\n")

    def do_GET(self):
        out = b'{"data": [{"id": "m"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        msgs = body["messages"]
        chunk = lambda d, f=None, u=None: dict({"choices": [{"index": 0, "delta": d, "finish_reason": f}]}, **({"usage": u} if u else {}))
        role = [chunk({"role": "assistant", "content": ""})]
        if body.get("tools"):
            call = {"index": 0, "id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}
            return self.events(role + [chunk({"tool_calls": [call]}), chunk({}, "tool_calls", {"prompt_tokens": 9, "completion_tokens": 3})])
        if body["max_tokens"] == 512:
            rng = random.Random(time.time())
            words = [f"w{rng.randrange(10**6)}" for _ in range(600)]
            evs, k = list(role), 0
            for i in range(436):
                take = 2 if i < 76 else 1
                evs.append(chunk({"content": " ".join(words[k:k + take]) + " "}))
                k += take
            evs.append(chunk({}, "length", {"prompt_tokens": 64, "completion_tokens": 512}))
            return self.events(evs)
        sys_len = len(msgs[0]["content"])
        if len(msgs) == 2:
            pt, cached = (4133 if 15000 < sys_len < 20000 else 16286 if sys_len > 60000 else 2900), 0
        else:
            cached, pt = 4139, 4169
        answer = "ok" if body["max_tokens"] == 64 else "The first file implements a speech model."
        self.events(role + [chunk({"content": w + " "}) for w in answer.split()] +
                    [chunk({}, "stop", {"prompt_tokens": pt, "completion_tokens": len(answer.split()),
                                        "prompt_tokens_details": {"cached_tokens": cached}})])


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve(speed):
    handler = type("H", (Script,), {"speed": speed})
    s = Threaded(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, s.server_address[1]


def closed_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def cfg(label, display, port, engine, **kw):
    return dict({"label": label, "display": display, "engine": engine, "version": "1.2.3", "model_name": "Qwen3.5-4B",
                 "model_path": "/tmp/" + label, "cmd": None, "lock": "harness", "base_url": f"http://127.0.0.1:{port}/v1",
                 "model": "m", "ready_timeout": 5}, **kw)


tmp = tempfile.mkdtemp(prefix="test-compare-")
env0 = {"BENCH_GPU_LOCK": os.path.join(tmp, "gpu.lock"), "KAPITSA_HOME": os.path.join(tmp, "home"),
        "BENCH_MODELS": os.path.join(tmp, "models"), "NO_COLOR": "1", "KAPITSA_BIN": KAPITSA}
env0.pop("NACE_GPU_LOCK_HELD", None)
os.environ.pop("NACE_GPU_LOCK_HELD", None)
os.environ.pop("NI_LOCK_HELD", None)

# ---------------------------------------------------------------- 1. the command
stub = os.path.join(tmp, "stub", "bench", "harness")
os.makedirs(stub)
open(os.path.join(stub, "kbench.py"), "w").write(
    "import sys, os\nprint('ARGS', *sys.argv[1:])\nprint('BIN', os.environ.get('KAPITSA_BIN'))\nsys.exit(7)\n")
r = run([KAPITSA, "bench", "compare", "--model", "4b", "--list"], env=dict(env0, KAPITSA_HARNESS=stub))
check("ARGS --model 4b --list" in r.stdout, "the arguments reach kbench.py", r.stdout + r.stderr)
check(r.returncode == 7, "the exit code is forwarded", str(r.returncode))
check(f"BIN {KAPITSA}" in r.stdout or "BIN " + os.path.realpath(KAPITSA) in r.stdout, "KAPITSA_BIN names this binary", r.stdout)
r = run([KAPITSA, "bench", "compare"], env=dict(env0, KAPITSA_HARNESS=os.path.join(tmp, "nowhere")), cwd=tmp)
check(r.returncode == 2 and "KAPITSA_HARNESS" in r.stderr and "install.sh" in r.stderr,
      "no harness: says how to get it, downloads nothing, starts nothing", r.stderr)

# the installed layout: <prefix>/bin/kapitsa with <prefix>/share/kapitsa/bench/harness beside it, which install.sh and
# the Homebrew formula make. No KAPITSA_HARNESS, and a folder with no bench/harness of its own: the reported bug.
inst = os.path.join(tmp, "inst")
os.makedirs(os.path.join(inst, "bin"))
shutil.copy2(KAPITSA, os.path.join(inst, "bin", "kapitsa"))
shutil.copytree(stub, os.path.join(inst, "share", "kapitsa", "bench", "harness"))
elsewhere = os.path.join(tmp, "elsewhere")
os.makedirs(elsewhere)
r = run([os.path.join(inst, "bin", "kapitsa"), "bench", "compare", "--model", "4b", "--list"],
        env={k: v for k, v in env0.items() if k != "KAPITSA_BIN"}, cwd=elsewhere)
check(r.returncode == 7 and "ARGS --model 4b --list" in r.stdout and
      "BIN " + os.path.realpath(os.path.join(inst, "bin", "kapitsa")) in r.stdout,
      "an installed layout: share/kapitsa/bench/harness beside the binary is found, with no env var",
      r.stdout + r.stderr)

# an unpacked release: kapitsa and bench/harness in the same folder, run from anywhere else (kapitsa update puts the
# harness there when the folder itself holds the notices)
unpacked = os.path.join(tmp, "unpacked")
os.makedirs(unpacked)
shutil.copy2(KAPITSA, os.path.join(unpacked, "kapitsa"))
shutil.copytree(stub, os.path.join(unpacked, "bench", "harness"))
r = run([os.path.join(unpacked, "kapitsa"), "bench", "compare", "--model", "4b", "--list"],
        env={k: v for k, v in env0.items() if k != "KAPITSA_BIN"}, cwd=elsewhere)
check(r.returncode == 7 and "ARGS --model 4b --list" in r.stdout,
      "an unpacked release: bench/harness in the binary's own folder is found, with no env var", r.stdout + r.stderr)

# ---------------------------------------------------------------- 2. kbench.py against scripted engines
fast, p_fast = serve(0.0004)
slow, p_slow = serve(0.002)
cdir = os.path.join(tmp, "configs")
os.makedirs(cdir)
json.dump([cfg("q35-4b-kapitsa-fake", "Kapitsa (scripted)", p_fast, "kapitsa"),
           cfg("q35-4b-kapitsa-fake-nomtp", "Kapitsa control", p_slow, "kapitsa")], open(os.path.join(cdir, "kapitsa.json"), "w"))
json.dump([cfg("q35-4b-omlx-fake", "oMLX plain", p_slow, "omlx"),
           cfg("q35-4b-omlx-fake-best", "oMLX + MTP (scripted)", p_fast, "omlx", model_path="/tmp/other")],
          open(os.path.join(cdir, "omlx.json"), "w"))
json.dump([cfg("q35-4b-mlx-lm-fake", "mlx-lm", closed_port(), "mlx-lm")], open(os.path.join(cdir, "mlx-lm.json"), "w"))
json.dump([cfg("q35-4b-llamacpp-fake", "llama.cpp", p_slow, "llama.cpp",
               cmd=["{first:/nonexistent/llama-server-for-test|llama-server-for-test-nope}"], requires_paths=[])], open(os.path.join(cdir, "llama.cpp.json"), "w"))
common = ["--configs", cdir, "--quiet-s", "0", "--quiet-timeout", "2", "--model", "4b", "--yes", "--ignore-running"]

os.symlink("/bin/sleep", os.path.join(tmp, "omlx"))
rival = subprocess.Popen([os.path.join(tmp, "omlx"), "60"])
r = run([sys.executable, os.path.join(HARNESS, "kbench.py")] + [c for c in common if c != "--ignore-running"], env=env0)
check(r.returncode == 1 and "skew" in r.stdout + r.stderr and str(rival.pid) in r.stdout + r.stderr,
      "another engine already running: refuses, names it, measures nothing", r.stdout + r.stderr)
rival.kill()
r = run([sys.executable, os.path.join(HARNESS, "kbench.py"), "--list"] + common, env=env0)
check(r.returncode == 0 and "oMLX" in r.stdout and "q35-4b-omlx-fake-best" in r.stdout,
      "--list picks the tuned variant (best) of an engine", r.stdout + r.stderr)
check(re.search(r"llama\.cpp\s+not installed", r.stdout) is not None, "a missing binary reads 'not installed'", r.stdout)
check("brew install llama.cpp" in r.stdout, "and says how to get it", r.stdout)

out = os.path.join(tmp, "run1")
t0 = time.time()
r = run([sys.executable, os.path.join(HARNESS, "kbench.py"), "--quick", "--out", out, "--engines", "kapitsa,omlx,mlx-lm"] + common,
        env=env0, timeout=600)
print(f"     (run took {time.time() - t0:.0f}s)")
check(r.returncode == 0, "a run with one engine failing still succeeds", r.stdout[-800:] + r.stderr[-800:])
check("Kapitsa" in r.stdout and "oMLX" in r.stdout and "decode tok/s" in r.stdout, "the table names the engines", r.stdout)
check("mlx-lm" in r.stdout.split("did not finish")[0] or "did not finish" in r.stdout, "the failed engine is reported", r.stdout)
doc = json.load(open(os.path.join(out, "compare.json")))
by = {e["engine"]: e for e in doc["engines"]}
check(by["kapitsa"]["status"] == "ok" and by["omlx"]["status"] == "ok" and by["mlx-lm"]["status"] == "failed",
      "statuses", str({k: v["status"] for k, v in by.items()}))
check(by["kapitsa"]["variant"] == "q35-4b-kapitsa-fake" and by["omlx"]["variant"] == "q35-4b-omlx-fake-best",
      "the tuned variants ran", str({k: v["variant"] for k, v in by.items()}))
check(by["kapitsa"]["ttft_cached_s"] is not None, "cached TTFT was measured", str(by["kapitsa"]))
check(doc["sizes"] == "4000" and doc["profile"] == "quick", "--quick: one prefill size")
md = open(os.path.join(out, "report.md")).read()
check("Who beats Kapitsa here" in md and "oMLX" in md and "<details>" in md, "the Markdown report has its parts", md)

out2 = os.path.join(tmp, "run2")
r = run([sys.executable, os.path.join(HARNESS, "kbench.py"), "--quick", "--json", "--out", out2, "--engines", "kapitsa,llama.cpp"] + common,
        env=env0, timeout=600)
try:
    d = json.loads(r.stdout)
    check(d["schema"] == "kapitsa-bench-compare/1" and d["engines"][0]["engine"] == "kapitsa", "--json prints one document")
    check([s["engine"] for s in d["skipped"]] == ["llama.cpp"] and d["skipped"][0]["reason"] == "not installed",
          "a missing engine is in `skipped` with its reason", str(d["skipped"]))
except Exception as e:
    check(False, "--json prints one document", f"{e!r}: {r.stdout[:300]} {r.stderr[-300:]}")

check(tmp not in md and os.path.expanduser("~") not in md.replace("~/", ""), "the report holds no path of this Mac", md)

r = run([sys.executable, os.path.join(HARNESS, "kbench.py"), "--json", "--report", "--out", out2] + common, env=env0)
check(r.returncode != 0 and "--report a FILE" in r.stderr + r.stdout, "--json with --report to stdout is refused")

r = run([sys.executable, os.path.join(HARNESS, "kbench.py"), "--engines", "nope"] + common, env=env0)
check(r.returncode != 0 and "unknown engine" in r.stderr, "an unknown engine is refused", r.stderr)
r = run([sys.executable, os.path.join(HARNESS, "kbench.py"), "--configs", cdir, "--model", "13b"], env=env0)
check(r.returncode != 0 and "4b, 27b, 35b" in r.stderr, "an unknown model names the choices", r.stderr)

# ---------------------------------------------------------------- 3. the real configs
import compare_engines as ce  # noqa: E402
import kbench  # noqa: E402

# a release ships a copy of the harness with no git checkout: its VERSION says which release, and harness_info reads it
copy = os.path.join(tmp, "harness-copy")
shutil.copytree(HARNESS, copy, ignore=shutil.ignore_patterns("__pycache__"))
with open(os.path.join(copy, "VERSION"), "w") as f:
    f.write("9.9.9\n")
r = run([sys.executable, "-c", "import sys; sys.path.insert(0, %r); import bench; print(bench.harness_info())" % copy],
        env=env0)
check(r.returncode == 0 and "'git_head': None" in r.stdout and "'version': '9.9.9'" in r.stdout,
      "a harness copy with no checkout names its release in reports", r.stdout + r.stderr)


class Tty(io.StringIO):
    def isatty(self):
        return True


plain, colour = kbench.render(doc, kbench.Term(Tty(), color=False)), kbench.render(doc, kbench.Term(Tty(), color=True))
check("\033" not in plain and "\033" in colour, "colour only when asked for; the plain render has no escape codes")
check(re.search(r"Kapitsa is first on \d of \d measures", plain) and re.search(r"[✓✗] decode\s+\d(st|nd|rd)", plain),
      "a verdict line per measure says where Kapitsa stands", plain)
check(re.search(r"x (faster|slower|lower|higher) than (oMLX|mlx-lm)", plain) and "▸ Kapitsa" in plain and "█" in plain,
      "the verdict names the engine and the factor; the table marks Kapitsa and draws bars", plain)
check(kbench.bar(0.5, 8) == "████    " and kbench.bar(1, 8) == "████████" and kbench.bar(1 / 16, 8) == "▌       ",
      "bars have eighth-block resolution", repr(kbench.bar(1 / 16, 8)))
os.environ.update(BENCH_MODELS=os.path.join(tmp, "models"))
cfgs = ce.load_configs(sorted(__import__("glob").glob(os.path.join(HARNESS, "configs", "*.json"))))
for model in kbench.MODELS:
    for key, name, _ in kbench.ENGINES:
        c = kbench.candidates(ce, cfgs, key, model)
        bad = [x["label"] for x in c if re.search(r"bf16|nomtp|mlx-q4", x["label"])]
        check(not bad, f"{model} {key}: no control config is a candidate", str(bad))
        labels = [x["label"] for x in c]
        check(labels == sorted(labels, key=kbench.rank_key), f"{model} {key}: tuned first", str(labels))
check(any("best" in x["label"] or "mtp" in x["label"] for x in kbench.candidates(ce, cfgs, "omlx", "4b")), "oMLX has a tuned 4B variant")
synth = kbench.kapitsa_synth(cfgs, "4b")
check(synth and "--dflash" not in synth["cmd"] and any("qwen3.5-4b-4bit" in x for x in synth["cmd"]),
      "Kapitsa as pulled: its folder, no drafter named", str(synth and synth["cmd"]))

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: " + "; ".join(FAILS))
    sys.exit(1)
print("all passed")
