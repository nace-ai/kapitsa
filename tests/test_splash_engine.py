#!/usr/bin/env python3
"""`kapitsa serve --engine splash` against tests/fake_splash.py (a protocol-v11 stand-in for the Splash binary), CPU only:
no model, no GPU. It needs Splash's resolved model folder for its tokenizer and chat template (default: the one under
~/Library/Application Support/Splash/models/.resolved, or give SPLASH_MODEL_DIR); without one it uses the small stand-in
folder tests/fixtures/splash_model, so it needs no model (the CI PR, #743, adds it to ci.yml).
usage: test_splash_engine.py build/kapitsa"""
import glob, json, os, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request

CLI = sys.argv[1] if len(sys.argv) > 1 else "build/kapitsa"
HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = os.path.join(HERE, "fake_splash.py")
dirs = [os.environ["SPLASH_MODEL_DIR"]] if os.environ.get("SPLASH_MODEL_DIR") else sorted(
    glob.glob(os.path.expanduser("~/Library/Application Support/Splash/models/.resolved/*/tokenizer/tokenizer.json")))
failures = 0
def check(c, what):
    global failures
    if not c:
        failures += 1
        print("FAIL:", what)

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

# no model needed: what the flags refuse and what discovery says
# the flags of the native path do not apply: refused at once, not overwritten or passed on
for extra in (["--model", "x"], ["--allow-switch"], ["--split", "127.0.0.1:1"]):
    r = subprocess.run([CLI, "serve", "--engine", "splash", "--splash-binary", FAKE, "--splash-model", tempfile.gettempdir(), *extra,
                        "--private", "--port", str(free_port())], capture_output=True, text=True, timeout=60)
    check(r.returncode != 0 and "do not apply" in r.stderr + r.stdout, "native flag %s refused: %d %r" % (extra, r.returncode, (r.stderr + r.stdout)[-300:]))

# no flags: the binary and the model are looked for where Splash installs them, and what is missing is named
with tempfile.TemporaryDirectory() as home:
    env = {k: v for k, v in os.environ.items() if k != "SPLASH_HOME"}
    env.update(HOME=home, PATH="/usr/bin:/bin")
    r = subprocess.run([CLI, "serve", "--engine", "splash", "--private", "--port", str(free_port())], capture_output=True, text=True, timeout=60, env=env)
    if not os.path.isfile("/Applications/Splash/engine/splash"):  # a system-wide install is looked at too, and HOME= does not hide it
        check(r.returncode != 0 and "no Splash engine binary found" in r.stderr and "SPLASH_HOME" in r.stderr, "no binary: %r" % r.stderr[-300:])
    if not os.path.isfile("/Applications/Splash/engine/splash"):  # same reason: a system-wide install would be found after this one
        os.makedirs(home + "/Applications/Splash/engine")
        open(home + "/Applications/Splash/engine/splash", "w").close()
        r = subprocess.run([CLI, "serve", "--engine", "splash", "--private", "--port", str(free_port())], capture_output=True, text=True, timeout=60, env=env)
        check(r.returncode != 0 and "no Splash engine binary found" in r.stderr, "a file that cannot be run is not the engine: %r" % r.stderr[-300:])
    os.makedirs(home + "/Applications/Splash/engine", exist_ok=True)
    open(home + "/Applications/Splash/engine/splash", "a").close()
    os.chmod(home + "/Applications/Splash/engine/splash", 0o755)
    r = subprocess.run([CLI, "serve", "--engine", "splash", "--private", "--port", str(free_port())], capture_output=True, text=True, timeout=60, env=env)
    check(r.returncode != 0 and "no Splash model found" in r.stderr, "no model: %r" % r.stderr[-300:])


# no Splash installed (CI): the small stand-in folder of tests/fixtures/make_splash_model.py
MODEL = (dirs[0] if os.environ.get("SPLASH_MODEL_DIR") else os.path.dirname(os.path.dirname(dirs[0]))) if dirs else os.path.join(HERE, "fixtures/splash_model")
TOK = json.load(open(MODEL + "/tokenizer/tokenizer.json"))
VOCAB = TOK["model"]["vocab"]
ADDED = {a["content"]: a["id"] for a in TOK["added_tokens"]}
def tid(piece): return ADDED[piece] if piece in ADDED else VOCAB[piece]
THINK_OPEN, THINK_CLOSE, EOS = tid("<think>"), tid("</think>"), tid("<|im_end|>")
NL = tid("Ċ")
# "ok\n</think>\n\nHi there" then the end of turn
REPLY = [tid("ok"), NL, THINK_CLOSE, tid("ĊĊ"), tid("Hi"), tid("Ġthere"), EOS]
last_headers = {}
def post(base, body, timeout=30):
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            last_headers.clear(); last_headers.update(r.headers)
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        last_headers.clear(); last_headers.update(e.headers)
        return e.code, json.loads(e.read() or b"{}")

class Server:
    def __init__(self, *fake_args, context=None):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = self.tmp.name + "/prompts.jsonl"
        self.port = free_port()
        self.base = "http://127.0.0.1:%d" % self.port
        cmd = [CLI, "serve", "--engine", "splash", "--splash-binary", FAKE, "--splash-model", MODEL, "--private",
               "--port", str(self.port), "--splash-arg", "--prompt-log", "--splash-arg", self.log]
        for a in fake_args: cmd += ["--splash-arg", a]
        env = dict(os.environ, HOME=self.tmp.name)
        self.proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(100):
            if self.proc.poll() is not None: raise SystemExit("server exited: " + self.proc.stderr.read().decode()[-2000:])
            try:
                urllib.request.urlopen(self.base + "/v1/models", timeout=1).read(); return
            except Exception: time.sleep(0.1)
        raise SystemExit("server did not start")
    def prompts(self):
        return [json.loads(l) for l in open(self.log)] if os.path.exists(self.log) else []
    def close(self):
        self.proc.terminate()
        try: self.proc.wait(10)
        except subprocess.TimeoutExpired: self.proc.kill()
        self.tmp.cleanup()

MSG = [{"role": "user", "content": "hello"}]
s = Server("--reply", ",".join(map(str, REPLY)))
try:
    st, r = post(s.base, {"messages": MSG, "max_tokens": 50, "temperature": 0})
    check(st == 200, "chat 200: %s %s" % (st, r))
    m = r.get("choices", [{}])[0].get("message", {})
    check(m.get("content") == "Hi there", "content %r" % m.get("content"))
    check("ok" in (m.get("reasoning_content") or m.get("reasoning") or ""), "reasoning %r" % m)
    check(r["choices"][0]["finish_reason"] == "stop", "finish %r" % r["choices"][0]["finish_reason"])
    check(r["usage"]["completion_tokens"] == len(REPLY), "completion_tokens %r" % r["usage"])
    p = s.prompts()
    check(len(p) == 1 and p[0]["prompt"][-2:] == [THINK_OPEN, NL], "prompt ends with <think>\\n: %r" % (p[0]["prompt"][-4:] if p else p))
    check(p and p[0]["max"] == 50, "max_output_tokens %r" % (p and p[0]["max"]))
    # max_tokens ends the reply as length and cancels the engine's request
    st, r = post(s.base, {"messages": MSG, "max_tokens": 4, "temperature": 0})
    check(st == 200 and r["choices"][0]["finish_reason"] == "length", "length: %s %s" % (st, r))
    # a stop string ends it at the string
    st, r = post(s.base, {"messages": MSG, "max_tokens": 50, "stop": ["there"], "temperature": 0})
    check(st == 200 and r["choices"][0]["message"]["content"] == "Hi ", "stop string: %s %s" % (st, r))
    # streaming
    req = urllib.request.Request(s.base + "/v1/chat/completions", json.dumps({"messages": MSG, "max_tokens": 50, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    body = urllib.request.urlopen(req, timeout=30).read().decode()
    check("Hi" in body and "[DONE]" in body, "stream: %r" % body[-300:])
    # refused up front
    st, r = post(s.base, {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}]}]})
    check(st == 400, "image refused with 400, got %s %s" % (st, r))
    st, r = post(s.base, {"messages": MSG, "response_format": {"type": "json_object"}})
    check(st == 400 and "structured output" in json.dumps(r), "json mode refused with 400: %s %s" % (st, r))
    check(s.proc.poll() is None, "server still running")

    # sampling: an absent field is Splash's own default (1.0 / 0.95 / 20), a given one goes through, penalties and min_p
    # included, ignore_eos is the engine's flag; an out-of-range value is a 400 naming the field and queues nothing
    n0 = len(s.prompts())
    post(s.base, {"messages": MSG, "max_tokens": 3})
    post(s.base, {"messages": MSG, "max_tokens": 3, "temperature": 0.3, "top_p": 0.5, "top_k": 7, "presence_penalty": 0.5,
                  "frequency_penalty": -0.5, "repetition_penalty": 1.25, "min_p": 0.125, "ignore_eos": True})
    p = s.prompts()[n0:]
    check(len(p) == 2, "two sampling requests reached the engine: %r" % p)
    if len(p) == 2:
        d, g = p[0]["sampling"], p[1]["sampling"]
        check((d["temperature"], round(d["top_p"], 4), d["top_k"]) == (1.0, 0.95, 20) and p[0]["flags"] == 0, "defaults %r" % p[0])
        check((round(g["temperature"], 4), g["top_p"], g["top_k"], g["presence"], g["frequency"], g["repetition"], g["min_p"]) ==
              (0.3, 0.5, 7, 0.5, -0.5, 1.25, 0.125) and p[1]["flags"] == 1, "given sampling %r" % p[1])
    for field, value in [("temperature", 3), ("top_p", 0), ("presence_penalty", 5), ("frequency_penalty", -3),
                         ("repetition_penalty", 0), ("min_p", 2)]:
        st, r = post(s.base, {"messages": MSG, "max_tokens": 3, field: value})
        check(st == 400 and field in json.dumps(r), "%s=%s is a 400 naming it: %s %s" % (field, value, st, r))
    check(len(s.prompts()) == n0 + 2, "refused requests were not queued")
finally:
    s.close()

# the engine's request capacity (the stand-in's is two) answers the third at once: 503 with Retry-After, before a stream starts
s = Server("--reply", ",".join(map(str, REPLY * 40)), "--slow", "50")
try:
    held = []
    for _ in range(2):
        req = urllib.request.Request(s.base + "/v1/chat/completions", json.dumps({"messages": MSG, "max_tokens": 200, "stream": True}).encode(),
                                     {"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=30); r.read(20); held.append(r)
    st, r = post(s.base, {"messages": MSG, "max_tokens": 3})
    check(st == 503 and last_headers.get("Retry-After"), "full engine: 503 + Retry-After, got %s %s %r" % (st, r, dict(last_headers)))
    req = urllib.request.Request(s.base + "/v1/chat/completions", json.dumps({"messages": MSG, "max_tokens": 3, "stream": True}).encode(),
                                 {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=30).read()
        check(False, "a streamed request to a full engine is a 503 before the stream")
    except urllib.error.HTTPError as e:
        check(e.code == 503 and e.headers.get("Retry-After"), "streamed: 503 + Retry-After, got %s" % e.code)
    for r in held: r.close()
    time.sleep(1.0)
    st, r = post(s.base, {"messages": MSG, "max_tokens": 3, "temperature": 0}, timeout=30)
    check(st == 200, "served once the held ones are gone: %s %s" % (st, r))
finally:
    s.close()

# the client going away cancels the engine's request: the next one is served (the pool is two, a leak would block)
s = Server("--reply", ",".join(map(str, REPLY * 40)), "--slow", "20")
try:
    for _ in range(4):
        req = urllib.request.Request(s.base + "/v1/chat/completions", json.dumps({"messages": MSG, "max_tokens": 200, "stream": True}).encode(),
                                     {"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=30); r.read(200); r.close()
        time.sleep(0.5)  # the server sees the client gone at its next write and cancels; a full engine answers 503
    time.sleep(0.5)
    st, r = post(s.base, {"messages": MSG, "max_tokens": 3, "temperature": 0}, timeout=30)
    check(st == 200, "served after four abandoned streams: %s %s" % (st, r))
finally:
    s.close()

# an engine that cannot start ends the server with its reason and a plain exit status, not an abort
r = subprocess.run([CLI, "serve", "--engine", "splash", "--splash-binary", "/nonexistent/splash", "--splash-model", MODEL,
                    "--private", "--port", str(free_port())], capture_output=True, text=True, timeout=60)
check(r.returncode == 1 and "did not start" in r.stderr + r.stdout, "start failure: %d %r" % (r.returncode, (r.stderr + r.stdout)[-300:]))

# a model name is no folder: refused at once, nothing looked up or downloaded
r = subprocess.run([CLI, "serve", "--engine", "splash", "--splash-binary", FAKE, "--splash-model", "/nonexistent/qwen3.5-35b-a3b",
                    "--private", "--port", str(free_port())], capture_output=True, text=True, timeout=60)
check(r.returncode != 0 and "is not a folder" in r.stderr + r.stdout, "model name refused: %d %r" % (r.returncode, (r.stderr + r.stdout)[-300:]))

print("test_splash_engine: %s" % ("ok" if not failures else "%d FAILED" % failures))
sys.exit(1 if failures else 0)
