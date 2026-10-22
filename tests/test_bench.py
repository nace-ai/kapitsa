#!/usr/bin/env python3
"""`kapitsa bench` against the harness it ports, on the CPU: no model, no GPU, nothing sent anywhere.

usage: test_bench.py [BUILD_DIR]   (default: build; needs BUILD_DIR/bench_test and BUILD_DIR/kapitsa)

1. bench_test self: the Mac table, the names and formats, the embedded corpus.
2. One protocol, two clients: bench/harness/bench.py and the C++ port run the same steps against one scripted server
   in this process, with the same nonces. Every request must be the same, body for body and key for key in order (the
   chat template renders the tools as JSON), and their records of the replies alike.
3. The rates and checks on the published runs' own data: decode_rates on every decode run's recorded events, distinct4
   on every saved text, Python's rounding, bench.py's summaries; with edge cases.
4. The result file: bench/community/schema.json (with jsonschema, when installed) and tools/community_check.py.
5. Submitting: the plan, `kapitsa bench submit --dry-run` with and without gh (a pull request is the only way in: no
   issue, no browser), and a stand-in gh that records what it is asked: nothing without an answer at a terminal, one question (Enter is a yes),
   then the pull request's calls.
6. `kapitsa bench` fails before loading anything when the model can't be had (a home without models, no network).
7. The GPU lock, on a test file: it waits for lockf's holder and names it, and lockf waits for it.
8. The whole command against the scripted server (KAPITSA_BENCH_SERVER, a test hook: no model, no lock): plain,
   --json and on a terminal; the result saved, valid against the schema, refused for submission.
"""
import glob, http.server, json, os, pty, random, re, select, shutil, socketserver, subprocess, sys, tempfile, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "build"))
BENCH_TEST, KAPITSA = os.path.join(BUILD, "bench_test"), os.path.join(BUILD, "kapitsa")
sys.path.insert(0, os.path.join(REPO, "bench", "harness"))
import bench  # noqa: E402  (bench/harness/bench.py)

FAILS = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


def run(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, timeout=kw.pop("timeout", 300), **kw)


# ---------------------------------------------------------------- 1. self checks
r = run([BENCH_TEST, "self"])
check(r.returncode == 0, "bench_test self", (r.stdout + r.stderr)[-500:])

# ---------------------------------------------------------------- 2. one protocol, two clients


class Script(http.server.BaseHTTPRequestHandler):
    """A server that answers like Kapitsa's /v1/chat/completions (SSE in HTTP/1.1 chunks) and records each request."""
    protocol_version = "HTTP/1.1"
    log = []
    prompt = {}  # session -> last prompt_tokens

    def log_message(self, *a):
        pass

    def send_events(self, events):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        for ev in events + ["[DONE]"]:
            data = ("data: " + (ev if isinstance(ev, str) else json.dumps(ev)) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()
            time.sleep(0.0005)
        self.wfile.write(b"0\r\n\r\n")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        body = json.loads(raw)
        Script.log.append({"path": self.path, "session": self.headers.get("X-Session-Id"), "body": body})
        msgs = body["messages"]
        chunk = lambda delta, finish=None, usage=None: dict({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]},
                                                            **({"usage": usage} if usage else {}))
        role = [chunk({"role": "assistant", "content": ""})]
        if body.get("tools"):
            if msgs[-1]["role"] == "tool":
                events = role + [chunk({"content": "It prints hello "}), chunk({"content": "from nace."}),
                                 chunk({}, "stop", {"prompt_tokens": 210, "completion_tokens": 7})]
            else:
                call = {"index": 0, "id": "call_7", "type": "function",
                        "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}
                think = body.get("chat_template_kwargs", {}).get("enable_thinking")
                events = role + ([chunk({"reasoning_content": "I should open it."})] if think else []) + \
                    [chunk({"tool_calls": [call]}), chunk({}, "tool_calls", {"prompt_tokens": 180, "completion_tokens": 20})]
            if not body.get("stream"):
                msg = {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_8", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"src/main.py\"}"}}]}
                out = json.dumps({"choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                                  "usage": {"prompt_tokens": 180, "completion_tokens": 20}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            return self.send_events(events)
        if body["max_tokens"] == 512:  # a decode: 512 tokens in 436 events of 1 or 2 tokens, distinct words
            rng = random.Random(len(Script.log))
            words = [f"w{rng.randrange(10**6)}" for _ in range(600)]
            events, k = role, 0
            for i in range(436):
                take = 2 if i < 76 else 1
                events.append(chunk({"content": " ".join(words[k:k + take]) + " "}))
                k += take
            events.append(chunk({}, "length", {"prompt_tokens": 64, "completion_tokens": 512}))
            return self.send_events(events)
        sid = self.headers.get("X-Session-Id") or "-"
        sys_len = len(msgs[0]["content"])
        if len(msgs) == 2:  # a cold prompt: its size by its characters
            pt = 4133 if 15000 < sys_len < 20000 else 16286 if sys_len > 60000 else 2900
            cached = 0
        else:  # a follow-up: the conversation so far, from the cache
            cached = Script.prompt.get(sid, 0) + 6
            pt = cached + 30
        Script.prompt[sid] = pt
        answer = "ok" if body["max_tokens"] == 64 else "The first file implements a speech recognition model."
        events = role + [chunk({"content": w + " "}) for w in answer.split()] + \
            [chunk({}, "stop", {"prompt_tokens": pt, "completion_tokens": len(answer.split()),
                                "prompt_tokens_details": {"cached_tokens": cached}})]
        self.send_events(events)


class Threaded(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


srv = Threaded(("127.0.0.1", 0), Script)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()

# the C++ port (fixed nonces) ...
Script.log, Script.prompt = [], {}
r = run([BENCH_TEST, "run", str(port)], timeout=120)
cpp = json.loads(r.stdout) if r.returncode == 0 else {}
cpp_log = Script.log
check(r.returncode == 0 and cpp, "the C++ protocol runs against the scripted server", r.stderr[-300:])
# ... and bench.py with the same nonces
Script.log, Script.prompt = [], {}
bench.random.random = lambda: 0.123456
bench.random.getrandbits = lambda n: 0x0123456789abcdef & ((1 << n) - 1)
a = bench.parse_args(["--base-url", f"http://127.0.0.1:{port}/v1", "--model", "qwen3.6-35b-a3b", "--label", "t",
                      "--turn-gap", "0.2"])
py = {"warmup": bench.warmup(a, log=lambda m: None), "decode": bench.t_decode(a), "greedy": bench.t_decode(a, greedy=True),
      "prefill": bench.t_prefill(a, 4000, bench.parse_ref(a)), "tools_off": bench.t_tools(a, False),
      "tools_on": bench.t_tools(a, True)}
py_log = Script.log
# warm-up 2, decode 1, greedy 1, the 4K prompt and its 3 follow-ups, and 3 per tool test (streamed, whole, round trip)
check(len(cpp_log) == len(py_log) == 14, "both send the same 14 requests", f"{len(cpp_log)} vs {len(py_log)}")
for i, (c, p) in enumerate(zip(cpp_log, py_log)):
    same = c["path"] == p["path"] and c["session"] == p["session"] and json.dumps(c["body"]) == json.dumps(p["body"])
    if not same:
        diff = [k for k in set(c["body"]) | set(p["body"]) if json.dumps(c["body"].get(k)) != json.dumps(p["body"].get(k))]
        check(False, f"request {i + 1} is the same", f"keys differ: {diff}; order: {list(c['body'])} vs {list(p['body'])}")
        break
else:
    check(bool(cpp_log), "every request is the same, key for key, in order (bodies, paths, session ids)")
# the prompts themselves: byte for byte the harness's
four_k = [x for x in py_log if x["body"]["max_tokens"] == 256 and len(x["body"]["messages"]) == 2][0]["body"]["messages"][0]["content"]
check(four_k == f"[session 0123456789abcdef] {bench.SYS}\n\n# Repository context\n{bench.build_context(4000)}",
      "the 4K prompt is the harness's, from the frozen corpus")


def pick(d, keys):
    return {k: d.get(k) for k in keys}


for k in ("decode", "greedy"):
    keys = ("greedy", "invalid", "flags", "completion_tokens", "chunks", "finish", "usage_reported", "distinct4")
    check(pick(cpp[k], keys) == pick(py[k], keys), f"t_decode ({k}) records alike", f"{pick(cpp[k], keys)} vs {pick(py[k], keys)}")
    check(cpp[k]["rate"]["rate_method"] == py[k]["rate"]["rate_method"] == "chars" and
          cpp[k]["rate"]["events"] == py[k]["rate"]["events"], f"t_decode ({k}): the same rate method over the same events")
keys = ("target", "invalid", "flags", "prompt_tokens", "expected_prompt_tokens", "cold_cached_tokens", "finish")
check(pick(cpp["prefill"], keys) == pick(py["prefill"], keys), "t_prefill records alike",
      f"{pick(cpp['prefill'], keys)} vs {pick(py['prefill'], keys)}")
cs = [pick(s, ("prompt_tokens", "cached_tokens", "finish")) for s in cpp["prefill"]["cached"]["samples"]]
ps = [pick(s, ("prompt_tokens", "cached_tokens", "finish")) for s in py["prefill"]["cached"]["samples"]]
check(cs == ps and len(cs) == 3, "the follow-ups alike", f"{cs} vs {ps}")
check(cpp["prefill"]["cached"]["invalid"] == py["prefill"]["cached"]["invalid"] and
      cpp["prefill"]["cached"]["flags"] == py["prefill"]["cached"]["flags"], "the follow-ups' checks alike")
for k in ("tools_off", "tools_on"):
    c, p = cpp[k], py[k]
    st = lambda x: {kk: x["stream"].get(kk) for kk in ("ok", "finish", "content_leak")} | {
        "calls": [(t["name"], t["arguments"], t["id"]) for t in x["stream"]["tool_calls"]]}
    check(st(c) == st(p), f"t_tools ({k}): the streamed call alike", f"{st(c)} vs {st(p)}")
    ns = lambda x: (x["nonstream"]["ok"], [(t["name"], t["arguments"], t["id"]) for t in x["nonstream"].get("tool_calls", [])])
    check(ns(c) == ns(p), f"t_tools ({k}): the whole reply alike")
    check(c["round_trip"]["ok"] == p["round_trip"]["ok"] == True, f"t_tools ({k}): the round trip passes in both")
check(bench.tools_pass([py["tools_off"], py["tools_on"]]), "the scripted tool checks pass")
wk = lambda w: [pick(s, ("name", "ok", "prompt_tokens", "completion_tokens", "finish", "thinking_leak")) for s in w["steps"]]
check(wk(cpp["warmup"]) == wk(py["warmup"]), "the warm-up alike", f"{wk(cpp['warmup'])} vs {wk(py['warmup'])}")

# ---------------------------------------------------------------- 3. rates and checks on the published runs' data
cases = {"model": "qwen3.6-35b-a3b", "nonce": "session 0123456789abcdef", "run": "0.123456",
         "contexts": [{"size": 4000}, {"size": 16000}, {"size": 2700, "offset": 100000}], "rates": [], "distinct4": [],
         "leaks": [], "rounds": [], "summaries": []}
runs = []
for f in sorted(glob.glob(os.path.join(REPO, "bench", "results", "*", "*.json"))):
    try:
        res = json.load(open(f))
    except Exception:
        continue
    if not isinstance(res, dict):  # not a run's record (2026-10-21-kflash-trees/kflash-merge.json lists its arms)
        continue
    for d in (res.get("decode") or []) + (res.get("decode_greedy") or []):
        if d.get("events_t_ms") and d.get("completion_tokens"):
            runs.append(d)
runs = runs[:400]
for d in runs:
    cases["rates"].append({"t": [x / 1000 for x in d["events_t_ms"]], "c": d["events_chars"], "ct": d["completion_tokens"]})
    cases["distinct4"].append(d.get("text") or d.get("reasoning") or "")
# edge cases: one token per event, too few events, all text in one event, no tokens
cases["rates"] += [{"t": [0.1 * i for i in range(512)], "c": [3] * 512, "ct": 512},
                   {"t": [0.5], "c": [10], "ct": 512}, {"t": [0.1, 0.2], "c": [0, 0], "ct": 10},
                   {"t": [0.1, 0.2, 0.3], "c": [100, 1, 1], "ct": 30, "head": 16, "tail": 4},
                   {"t": [0.01 * i for i in range(100)], "c": [5] * 100, "ct": 0}]
cases["distinct4"] += ["café naïve façade_1 x² 東京 東京 🚨🚨 été ٣٤ … — «quote»" * 3, "a b c", "", "x" * 50,
                       " ".join(["same words again"] * 40), "tab\tsep\nnew line nbsp　ideo"]
cases["leaks"] = [{"content": "<think>x</think>y"}, {"content": "  \n<think>"}, {"content": "a</think>b"}, {"reasoning": " \n"},
                  {"reasoning": "hmm", "content": "ok"}, {"content": "plain"}]
cases["rounds"] = [[2.675, 2], [0.125, 2], [0.375, 2], [1169.45, 1], [0.06035, 4], [153.7449999, 1], [-0.5, 0], [1e-9, 4]]
cases["summaries"] = [[{"invalid": [], "flags": [], "decode_tps": x, "decode_tps_e2e": x - 1} for x in (150.1, 147.25, 160.9, 152.3)],
                      [{"invalid": ["x"], "flags": ["f1"], "decode_tps": 10}, {"invalid": [], "flags": ["f2"], "decode_tps": None},
                       {"invalid": [], "flags": [], "decode_tps": 0}, {"invalid": [], "flags": ["f1"], "decode_tps": 99.999}]]
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump(cases, fh)
r = run([BENCH_TEST, "protocol", fh.name], timeout=300)
os.unlink(fh.name)
out = json.loads(r.stdout) if r.returncode == 0 else {}
check(r.returncode == 0, "bench_test protocol", r.stderr[-300:])
ctxs = out.get("prompts", {})
import hashlib  # noqa: E402
for c in cases["contexts"]:
    s = bench.build_context(c["size"], offset=c.get("offset", 0)).encode()
    got = ctxs.get(f"context_{c['size']}_{c.get('offset', 0)}", {})
    check(got.get("sha256") == hashlib.sha256(s).hexdigest() and got.get("bytes") == len(s),
          f"build_context({c['size']}, offset {c.get('offset', 0)}) is the harness's, byte for byte")
p = out.get("prompts", {})
check(json.dumps(p.get("tools")) == json.dumps(bench.TOOLS), "TOOLS, key for key")
check(p.get("followups") == bench.FOLLOWUPS, "the follow-up questions")
check(json.dumps(p.get("tools_messages")) == json.dumps(bench.tools_messages()), "the tool test's messages")
bench.random.random = lambda: 0.123456
check(json.dumps(p.get("decode_messages")) == json.dumps(bench.decode_messages()), "the decode prompt (with the same nonce)")
check(json.dumps(p.get("prefill_messages_4000")) == json.dumps(bench.prefill_messages(bench.build_context(4000), "session 0123456789abcdef",
      "In one sentence, what is the first file in the repository context about?")), "the 4K prefill messages")
check(json.dumps(p.get("warm_messages")) == json.dumps(bench.prefill_messages(bench.build_context(bench.WARM_TOKENS, offset=bench.WARM_OFFSET_CHARS),
      "warmup 0123456789abcdef", "Summarize the first file in the repository context.")), "the warm-up messages")
check(json.dumps(p.get("body_decode")) == json.dumps(bench.body_of(a, bench.decode_messages(), 512)), "the decode request body")
check(json.dumps(p.get("body_greedy")) == json.dumps(bench.body_of(a, bench.decode_messages(), 512, greedy=True)), "the greedy request body")
check(json.dumps(p.get("body_tools_on")) == json.dumps(bench.body_of(a, bench.tools_messages(), 4096, think=True, tools=bench.TOOLS,
      tool_choice="auto")), "the thinking-on tool request body")
bad = 0
for i, (c, got) in enumerate(zip(cases["rates"], out.get("rates", []))):
    mid, e2e, info = bench.decode_rates({"ev_t": c["t"], "ev_c": c["c"]}, c["ct"], c.get("head", 16), c.get("tail", 4))
    same = (mid is None) == (got["mid"] is None) and (mid is None or abs(mid - got["mid"]) <= 1e-9 * abs(mid)) and \
        (e2e is None) == (got["e2e"] is None) and (e2e is None or abs(e2e - got["e2e"]) <= 1e-9 * abs(e2e)) and \
        json.dumps(info, sort_keys=True) == json.dumps(got["info"], sort_keys=True)
    if not same:
        bad += 1
        if bad <= 3:
            print(f"   rate case {i}: py {mid} {e2e} {info} / c++ {got}")
check(bad == 0 and len(out.get("rates", [])) == len(cases["rates"]),
      f"decode_rates on {len(runs)} published runs' events and the edge cases ({bad} differ)")
bad = [i for i, (t, got) in enumerate(zip(cases["distinct4"], out.get("distinct4", []))) if bench.distinct4(t) != got]
check(not bad and len(out.get("distinct4", [])) == len(cases["distinct4"]), f"distinct4 on {len(cases['distinct4'])} texts",
      f"differ at {bad[:5]}: {[(bench.distinct4(cases['distinct4'][i]), out['distinct4'][i]) for i in bad[:3]]}")
leaks = [bench.thinking_leak(x) for x in cases["leaks"]]
check(leaks == out.get("leaks"), "thinking_leak", f"{leaks} vs {out.get('leaks')}")
rounds = [round(x, n) for x, n in cases["rounds"]]
check(rounds == out.get("rounds"), "Python's round()", f"{rounds} vs {out.get('rounds')}")
sums = [bench._summary(s) for s in cases["summaries"]]
check(json.dumps(sums, sort_keys=True) == json.dumps(out.get("summaries"), sort_keys=True), "bench.py's _summary",
      f"{sums} vs {out.get('summaries')}")

# ---------------------------------------------------------------- 4. the result file
tmp = tempfile.mkdtemp(prefix="kapitsa-bench-test-")
results = {}
for mode in ("full", "quick"):
    r = run([BENCH_TEST, "result"] + (["quick"] if mode == "quick" else []))
    results[mode] = r.stdout
    with open(os.path.join(tmp, f"{mode}.json"), "w") as fh:
        fh.write(r.stdout)
schema = json.load(open(os.path.join(REPO, "bench", "community", "schema.json")))
try:
    import jsonschema
    jsonschema.Draft202012Validator.check_schema(schema)
    for mode, text in results.items():
        errs = list(jsonschema.Draft202012Validator(schema).iter_errors(json.loads(text)))
        check(not errs, f"the {mode} result validates against schema.json", "; ".join(e.message for e in errs[:3]))
    broken = json.loads(results["full"])
    broken["mac"]["serial"] = "C02XXXX"
    broken["results"]["decode"]["median_tps"] = "fast"
    check(len(list(jsonschema.Draft202012Validator(schema).iter_errors(broken))) == 2,
          "schema.json refuses an extra field (a serial number) and a wrong type")
except ImportError:
    print("SKIP schema validation: pip install jsonschema", flush=True)
r = run([sys.executable, os.path.join(REPO, "tools", "community_check.py"), "--no-git", os.path.join(tmp, "full.json"),
         os.path.join(tmp, "quick.json")])
check(r.returncode == 0, "tools/community_check.py accepts both", (r.stdout + r.stderr)[-600:])
full = json.loads(results["full"])
check(len(json.dumps(full, separators=(",", ":"))) < 2500, "the result is small enough to read and to travel in an address",
      str(len(json.dumps(full))))
import getpass  # noqa: E402


def private_names():
    """This Mac's user name and its names on the network (scutil), and the words for a serial number and a path"""
    out = [("user name", getpass.getuser()), ("home path", "/Users/"), ("serial number", "serial")]
    for k in ("LocalHostName", "ComputerName", "HostName"):
        v = run(["/usr/sbin/scutil", "--get", k]).stdout.strip()
        if len(v) >= 6:
            out.append((k, v))
    return out


for what, k in private_names():
    check(k.lower() not in results["full"].lower(), f"no {what} in the result")

# ---------------------------------------------------------------- 5. submitting
r = run([BENCH_TEST, "plan", os.path.join(tmp, "full.json"), "tester"])
plan = json.loads(r.stdout)
check(plan["title"] == "Benchmark: MacBook Pro (16-inch, 2024) · M4 Pro · 48 GB · Qwen3.6 35B-A3B 4-bit", "the pull request's title",
      plan["title"])
check(plan["path"].startswith("bench/community/results/kapitsa-bench-2026-10-18-m4-pro-qwen3.6-35b-a3b-4bit-") and
      plan["path"].endswith(".json"), "the file's path", plan["path"])
check(plan["fork"] == "tester/kapitsa" and plan["submit_problem"] == "", "the fork, and nothing in the way")
check("issue_url" not in plan, "the plan has no issue form: a pull request is the only way in")

# a stand-in gh: records its calls, answers like GitHub
gh_dir = tempfile.mkdtemp(prefix="kapitsa-fake-gh-")
gh_log = os.path.join(gh_dir, "calls.jsonl")
with open(os.path.join(gh_dir, "gh"), "w") as fh:
    fh.write(f"""#!{sys.executable}
import json, sys
a = sys.argv[1:]
rec = {{"args": a}}
if "--input" in a:
    rec["input"] = json.load(open(a[a.index("--input") + 1]))
open({gh_log!r}, "a").write(json.dumps(rec) + "\\n")
path = next((x for x in a if x.startswith("repos/") or x == "user"), "")
if path == "user":
    print("tester" if "--jq" in a else json.dumps({{"login": "tester"}}))
elif path.endswith("/forks"):
    print(json.dumps({{"full_name": "tester/kapitsa"}}))
elif path == "repos/nace-ai/kapitsa":
    print(json.dumps({{"default_branch": "main"}}))
elif "/git/ref/heads/" in path:
    print(json.dumps({{"object": {{"sha": "0123456789abcdef0123456789abcdef01234567"}}}}))
elif path.endswith("/git/refs"):
    print(json.dumps({{"ref": "refs/heads/x"}}))
elif "/contents/" in path:
    print(json.dumps({{"content": {{"path": path}}}}))
elif path.endswith("/pulls"):
    print(json.dumps({{"html_url": "https://github.com/nace-ai/kapitsa/pull/4242"}}))
else:
    sys.exit(1)
""")
os.chmod(os.path.join(gh_dir, "gh"), 0o755)
# no model can be had: an empty Kapitsa home, a HOME without the Hugging Face cache or the app's models, no network to
# Hugging Face; and a temporary GPU lock (never the shared /tmp/kapitsa-gpu.lock, which another session's benchmark may hold
# for hours), so that even a run that got further could not load a model outside a lock
home = tempfile.mkdtemp(prefix="kapitsa-home-")
env = dict(os.environ, PATH=gh_dir + ":/usr/bin:/bin", KAPITSA_HOME=home, HOME=os.path.join(home, "user"),
           HF_ENDPOINT="http://127.0.0.1:9", HF_HUB_CACHE=os.path.join(home, "hf"), NO_COLOR="1",
           BENCH_GPU_LOCK=os.path.join(home, "gpu.lock"))
os.makedirs(env["HOME"])
env.pop("NACE_GPU_LOCK_HELD", None)
env.pop("GH_TOKEN", None)
env.pop("GITHUB_TOKEN", None)


def calls():
    return [json.loads(l) for l in open(gh_log)] if os.path.exists(gh_log) else []


r = run([KAPITSA, "bench", "submit", os.path.join(tmp, "full.json"), "--dry-run"], env=env)
check(r.returncode == 0 and "gh api -X POST repos/nace-ai/kapitsa/forks" in r.stdout and "issues/new" not in r.stdout,
      "submit --dry-run with gh: prints the gh commands and no issue form", (r.stdout + r.stderr)[-400:])
check([c["args"][:4] for c in calls()] == [["api", "--hostname", "github.com", "user"]], "the dry run only asked gh who is logged in",
      json.dumps(calls()))
open(gh_log, "w").close()
r = run([KAPITSA, "bench", "--dry-run"], env=env)
check(r.returncode == 2, "--dry-run without --submit is a usage error")
nogh = dict(env, PATH="/usr/bin:/bin")
r = run([KAPITSA, "bench", "submit", os.path.join(tmp, "full.json"), "--dry-run"], env=nogh)
check(r.returncode == 0 and "gh is not installed" in r.stdout and "gh api -X POST repos/nace-ai/kapitsa/pulls" in r.stdout
      and "issues/new" not in r.stdout, "submit --dry-run without gh: says so and shows the pull request, no issue form", r.stdout[-300:])
r = run([KAPITSA, "bench", "submit", os.path.join(tmp, "full.json")], env=nogh, stdin=subprocess.DEVNULL, start_new_session=True)
check(r.returncode == 1 and "gh is not installed" in r.stderr and "pull request" in r.stderr and "issue" not in r.stderr.lower()
      and "browser" not in r.stderr.lower(), "submit without gh: explains how to get gh, offers no other path", r.stderr[-300:])
r = run([KAPITSA, "bench", "submit", os.path.join(tmp, "full.json")], env=env, stdin=subprocess.DEVNULL, start_new_session=True)
check(r.returncode == 1 and "terminal" in r.stderr, "without a terminal nothing is submitted", r.stderr[-200:])
check(not [c for c in calls() if c["args"][:2] != ["api", "--hostname"]], "and gh was asked for nothing else")


def with_terminal(args, answers, timeout=60, environ=None):
    """Runs args on a pseudo-terminal, typing each answer when a [Y/n] question appears; returns (code, output)."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execve(args[0], args, environ or env)
    out, left = b"", list(answers)
    t0 = time.time()
    while time.time() - t0 < timeout:
        rl, _, _ = select.select([fd], [], [], 0.2)
        if rl:
            try:
                b = os.read(fd, 4096)
            except OSError:
                break
            if not b:
                break
            out += b
            if out.count(b"[Y/n]") > len(answers) - len(left) and left:
                os.write(fd, (left.pop(0) + "\n").encode())
    _, st = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(st), out.decode("utf-8", "replace")


open(gh_log, "w").close()
code, text = with_terminal([KAPITSA, "bench", "submit", os.path.join(tmp, "full.json")], ["n"])
check(code == 0 and f"Not sent. kapitsa bench submit {os.path.join(tmp, 'full.json')} sends it later" in " ".join(text.split()),
      "a no sends nothing and names the saved file", text[-300:])
check(all(c["args"][:2] == ["api", "--hostname"] for c in calls()), "nothing but the login was asked of gh", json.dumps(calls())[:300])
check(os.path.exists(os.path.join(tmp, "full.json")), "the result stays where it was saved")
busy = json.loads(results["full"])
busy["conditions"].update(gpu_other_mean_pct=12.4, gpu_other_max_pct=40.2, gpu_users=["Google Chrome", "Slack"],
                          contention=["decode", "prefill_4000"])
busy["flags"] = busy["flags"] + ["CONTENTION: sustained foreign GPU load > 5% in ['decode', 'prefill_4000'] (mean 12.4%, top "
                                 "['Google Chrome', 'Slack'])"]
with open(os.path.join(tmp, "busy.json"), "w") as fh:
    json.dump(busy, fh, indent=2)
tenv = dict(env, TERM="xterm-256color", COLUMNS="100")
tenv.pop("NO_COLOR", None)
code, text = with_terminal([KAPITSA, "bench", "submit", os.path.join(tmp, "busy.json")], ["n"], environ=tenv)
plain = " ".join(re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text).replace("│", " ").split())
check(code == 0 and "Writing" in plain and "Google Chrome and Slack used the GPU: 12% on average, up to 40%. Close them for a clean run."
      in plain and "CONTENTION" not in plain and plain.count("[Y/n]") == 1,
      "submitting a saved result shows its card first, with who used the GPU in words", plain[-500:])
open(gh_log, "w").close()
code, text = with_terminal([KAPITSA, "bench", "submit", os.path.join(tmp, "full.json")], [""])
got = calls()
check(code == 0 and "pull/4242" in text and text.count("[Y/n]") == 1, "one question, and Enter opens the pull request",
      text[-300:])
paths = [next((x for x in c["args"] if x.startswith("repos/") or x == "user"), "") for c in got]
check(paths == ["user", "repos/nace-ai/kapitsa/forks", "repos/nace-ai/kapitsa", "repos/nace-ai/kapitsa/git/ref/heads/main",
                "repos/tester/kapitsa/git/refs", f"repos/tester/kapitsa/contents/{plan['path']}", "repos/nace-ai/kapitsa/pulls"],
      "the calls: login, fork, base, its head, the branch, the file, the pull request", json.dumps(paths))
put = next(c for c in got if "/contents/" in " ".join(c["args"]))
import base64  # noqa: E402
check(base64.b64decode(put["input"]["content"]).decode() == results["full"], "the file sent is the saved file, byte for byte")
pr = next(c for c in got if c["args"][-3:-1] == ["repos/nace-ai/kapitsa/pulls", "--input"] or "pulls" in " ".join(c["args"]))
check(pr["input"]["head"] == f"tester:{plan['branch']}" and pr["input"]["base"] == "main" and pr["input"]["title"] == plan["title"]
      and "| Writing, T = 0.6 |" in pr["input"]["body"], "the pull request: head, base, title and the summary table")

# ---------------------------------------------------------------- 6. no model: fails before loading anything
r = run([KAPITSA, "bench", "--model", "4b", "--yes"], env=nogh, timeout=120)
check(r.returncode != 0 and not glob.glob(os.path.join(home, "logs", "bench-*.log")),
      "a model that can't be downloaded stops the run before any server starts", (r.stdout + r.stderr)[-300:])
r = run([KAPITSA, "bench", "--model", os.path.join(home, "nothing-here")], env=nogh)
check(r.returncode == 2, "a folder that is not a model is refused", (r.stdout + r.stderr)[-200:])
r = run([KAPITSA, "bench", "--help"], env=nogh)
check(r.returncode == 0 and "kapitsa bench [--model MODEL]" in r.stdout, "--help")

# ---------------------------------------------------------------- 7. the GPU lock (a test file, immutable like this Mac's)
if not shutil.which("lockf"):  # lockf(1) came with macOS 15
    print("SKIP the GPU lock against lockf: this Mac has no lockf (macOS 14)", flush=True)
else:
    lockfile = os.path.join(tmp, "gpu.lock")
    open(lockfile, "w").close()
    subprocess.run(["chflags", "uchg", lockfile], check=True)
    lenv = dict(os.environ, BENCH_GPU_LOCK=lockfile)
    holder = subprocess.Popen(["lockf", "-k", lockfile, "sleep", "2"])
    time.sleep(0.5)
    r = run([BENCH_TEST, "lock"], env=lenv)
    info = json.loads(r.stdout or "{}")
    holder.wait()
    check(info.get("ok") and info.get("waited") and info.get("wait_s", 0) >= 1.0 and "lockf" in info.get("holders", ""),
          "waits for lockf's holder of an immutable lock file, and names it", r.stdout)
    p = subprocess.Popen([BENCH_TEST, "lock", "2"], env=lenv, stdout=subprocess.PIPE, text=True)
    first = p.stdout.readline()
    r2 = run(["lockf", "-k", "-t", "0", lockfile, "true"])
    p.wait()
    r3 = run(["lockf", "-k", "-t", "0", lockfile, "true"])
    check(json.loads(first).get("ok") and r2.returncode != 0 and r3.returncode == 0,
          "while it holds the lock lockf can't take it; released after", f"{first} {r2.returncode} {r3.returncode}")
    subprocess.run(["chflags", "nouchg", lockfile])

# ---------------------------------------------------------------- 8. the whole command, against the scripted server
# KAPITSA_BENCH_SERVER: kapitsa bench measures the scripted server instead of starting one (no download, lock or quiet
# wait); everything else is the real command: the protocol, the monitor, the result, the screens. Its result is invalid.
xhome = tempfile.mkdtemp(prefix="kapitsa-bench-e2e-")
xenv = dict(env, KAPITSA_BENCH_SERVER=str(port), KAPITSA_HOME=xhome, HOME=os.path.join(xhome, "user"))
os.makedirs(xenv["HOME"])
Script.log, Script.prompt = [], {}
r = run([KAPITSA, "bench", "--model", "4b", "--quick"], env=xenv, timeout=300)
saved = glob.glob(os.path.join(xhome, "bench", "kapitsa-bench-*-qwen3.5-4b-4bit-quick.json"))
check(r.returncode == 1 and len(saved) == 1, "plain: a quick run against the scripted server completes and is saved", (r.stdout + r.stderr)[-500:])
check(len(Script.log) == 18, "the quick protocol's 18 requests (warm-up 2, writing 3 + 3, the 4K prompt and 3 follow-ups, tools 6)",
      str(len(Script.log)))
res = json.load(open(saved[0])) if saved else {}
check(res.get("invalid") == ["server: a server kapitsa bench did not start (KAPITSA_BENCH_SERVER), so its settings are unknown"],
      "its result is invalid (an external server), so it can't be submitted", str(res.get("invalid")))
check(res.get("mode") == "quick" and res.get("model", {}).get("name") == "qwen3.5-4b" and res["model"]["variant"] == "4bit" and
      res["model"]["precision"] == "4-bit", "the model and the mode", str(res.get("model")))
check(res["protocol"]["sizes"] == [4000] and res["protocol"]["decode_runs"] == 3 and len(res["results"]["decode"]["runs_tps"]) == 3,
      "the quick protocol: 3 + 3 runs, the 4K prompt")
check(res["results"]["prefill"][0]["prompt_tokens"] == 4133 and len(res["results"]["prefill"][0]["followup_s"]) == 3 and
      res["results"]["tool_calls"] == "pass", "the prompt, the follow-ups and the tool check as the server answered")
hw = run(["sysctl", "-n", "hw.model"]).stdout.strip()
check(res["mac"]["identifier"] == hw and res["mac"]["memory_gb"] > 0 and res["mac"]["chip"].startswith("Apple"), "this Mac")
try:
    import jsonschema
    errs = list(jsonschema.Draft202012Validator(schema).iter_errors(res))
    check(not errs, "the run's result validates against schema.json", "; ".join(e.message for e in errs[:3]))
except ImportError:
    pass
check("| Writing, T = 0.6 |" in r.stdout and "decode T=0.6: median" in r.stderr and "Saved " in r.stderr,
      "plain: the summary table on stdout, the progress on stderr")
r = run([KAPITSA, "bench", "--model", "4b", "--quick", "--submit", "--dry-run"], env=xenv, timeout=300)
check(r.returncode == 1 and "invalid measurements" in r.stderr, "--submit refuses an invalid result", r.stderr[-300:])
r = run([KAPITSA, "bench", "--model", "4b", "--quick", "--json"], env=xenv, timeout=300)
try:
    js = json.loads(r.stdout)
except Exception:
    js = {}
check(js.get("kind") == "kapitsa-bench" and "decode T=0.6" in r.stderr, "--json: the result on stdout, the progress on stderr")
code, text = with_terminal([KAPITSA, "bench", "--model", "4b", "--quick"], [], timeout=300,
                           environ=dict(xenv, TERM="xterm-256color", COLUMNS="100"))
plain = __import__("re").sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
check(code == 1 and "Writing, T = 0.6" in plain and "Reading 4K" in plain and "Follow-up at 4K" in plain and "Tool calls" in plain
      and "╭" in plain and "Saved" in plain, "a terminal: the phases and the summary card", plain[-600:])
shutil.rmtree(xhome, ignore_errors=True)

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(gh_dir, ignore_errors=True)
shutil.rmtree(home, ignore_errors=True)
srv.shutdown()
print(f"\n{'FAIL' if FAILS else 'ok'}: {len(FAILS)} failed" + (f": {FAILS}" if FAILS else ""))
sys.exit(1 if FAILS else 0)
