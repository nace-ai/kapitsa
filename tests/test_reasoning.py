#!/usr/bin/env python3
"""bench/reasoning on the CPU: no model, no GPU, nothing sent anywhere.

usage: test_reasoning.py

1. The grader: answers written differently that are the same, near misses that are not, and the last \\boxed{}.
2. The frozen sets: their sizes, no repeats, and select() drawing in proportion and repeatably.
3. A run against a scripted OpenAI-compatible server in this process: the requests (Qwen's thinking sampling, a seed
   per sample, thinking on, max_tokens within the server's context), the grading of right, wrong and cut-short replies,
   an HTTP error recorded and retried on resume, the GPU lock taken on a test lock file.
4. The report: accuracy, its interval, replies cut short or without an answer, and a label against the baseline.
5. With the problems already downloaded ($KAPITSA_REASONING_CACHE or ~/.cache/kapitsa/reasoning): `check` and the
   frozen sets redrawn from MATH-500. Skipped otherwise (this test does not download).
"""
import http.server
import json
import os
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
RDIR = os.path.join(REPO, "bench", "reasoning")
SCRIPT = os.path.join(RDIR, "reasoning.py")
sys.path.insert(0, RDIR)
import grader  # noqa: E402
import reasoning  # noqa: E402

FAILS = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- 1. the grader
SAME = [
    ("\\frac{1}{2}", "0.5"), ("1/2", "\\frac12"), ("\\dfrac{3}{4}", "\\frac{3}{4}"),
    ("\\frac{\\sqrt{3}}{3}", "\\frac{1}{\\sqrt{3}}"), ("3\\sqrt{13}", "\\sqrt{117}"), ("11\\sqrt2", "11 \\sqrt{2}"),
    ("90", "90^\\circ"), ("10\\%", "10"), ("\\$18.90", "18.9"), ("10080", "10,\\!080"),
    ("11111111100", "11,\\! 111,\\! 111,\\! 100"), ("864", "864 \\mbox{ inches}^2"), ("5.4", "5.4 \\text{ cents}"),
    ("\\frac{270}{7}", "\\frac{270}7\\text{ degrees}"), ("\\frac{9}{5}", "1\\frac{4}{5}"),
    ("137.5", "137 \\frac{1}{2}"),
    ("-5i+6", "6 - 5i"), ("x^2+2x+1", "(x+1)^2"), ("ab+2a+5b+10", "(a+5)(b+2)"), ("2(k+1)", "2k+2"),
    ("20000/\\pi", "\\frac{20000}{\\pi}"), ("(3, \\pi/2)", "\\left( 3, \\frac{\\pi}{2} \\right)"),
    ("(-\\infty, 0]", "(-\\infty,0]"), ("(9,36) \\cup (0,9)", "(0,9) \\cup (9,36)"), ("[-2,7]", "x \\in [-2,7]"),
    ("-2, 1", "1,-2"), ("1+\\sqrt{19}, 1-\\sqrt{19}", "1 \\pm \\sqrt{19}"),
    ("\\{-2, 1+\\sqrt5, 1-\\sqrt5\\}", "\\{1\\pm\\sqrt{5},-2\\}"), ("5", "x=5"), ("x=5", "5"), ("2x+3=y", "y = 2x + 3"),
    ("x=1, x=-2", "1,-2"), ("(-1/3, 2/3, 5/3)", "\\begin{pmatrix} -1/3 \\\\ 2/3 \\\\ 5/3 \\end{pmatrix}"),
    ("\\begin{bmatrix}-1&0\\\\0&-1\\end{bmatrix}", "\\begin{pmatrix} -1 & 0 \\\\ 0 & -1 \\end{pmatrix}"),
    ("52", "52_8"), ("4210_5", "4210_{5}"), ("C", "\\text{(C)}"), ("evelyn", "\\text{Evelyn}"),
    ("an ellipse", "\\text{ellipse}"), ("\\frac{\\cos x}{\\sin x}", "\\cot x"), ("6.72\\times10^{-5}", ".0000672"),
    ("070", "70"), ("336", "336^\\circ"), ("\\sqrt[3]{-8}", "-2"), ("2^{10}", "1024"), ("\\binom{5}{2}", "10"),
    ("5!", "120"), ("e^{i\\pi}", "-1"), ("\\sin^2 x + \\cos^2 x", "1"), ("\\log_2 8", "3"),
    ("2\\theta", "\\theta+\\theta"),
    ("5x-7y+11z+4=0", "5x - 7y + 11z + 4 = 0"),
]
DIFFERENT = [
    ("\\frac{1}{3}", "0.333"), ("(3,4)", "(3,4]"), ("[3,4]", "(3,4]"), ("(1,2)", "(2,1)"), ("1,2", "1,2,3"),
    ("1", "1,-2"), ("x^2+1", "x^2-1"), ("2k+1", "2k+2"), ("52_7", "52_8"), ("53", "52_8"),
    ("\\text{(B)}", "\\text{(C)}"), ("Navin", "\\text{Evelyn}"), ("6+5i", "6 - 5i"), ("\\sqrt{2}", "1.414"),
    ("y=2x+4", "y = 2x + 3"), ("x=6", "5"), ("0.0000673", ".0000672"), ("(0,9)\\cup(9,37)", "(0,9) \\cup (9,36)"),
    ("2^{10}", "1023"), (None, "5"), ("", "5"), ("3r^2", "3R^2"), ("(1,-2)", "1,-2"), ("\\infty", "-\\infty"),
    ("x+y=5", "5"), ("11111111101", "11,\\! 111,\\! 111,\\! 100"), ("1000001", "1000000"),
    ("\\frac{10000001}{3}", "3333333.6667"),
]
wrong = [(p, g) for p, g in SAME if not grader.equal(p, g)]
check(not wrong, f"grader: {len(SAME)} pairs written differently are the same", repr(wrong[:3]))
wrong = [(p, g) for p, g in DIFFERENT if grader.equal(p, g)]
check(not wrong, f"grader: {len(DIFFERENT)} near misses are not", repr(wrong[:3]))
EXTRACT = [("So $\\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}"), ("\\boxed{1} then \\boxed{2}", "2"),
           ("\\boxed{\\{1, 2\\}}", "\\{1, 2\\}"), ("\\fbox{7}", "7"), ("no box", None),
           ("cut \\boxed{\\frac{1}{", None), ("\\boxed{}", None)]
wrong = [(t, grader.extract(t), w) for t, w in EXTRACT if grader.extract(t) != w]
check(not wrong, "grader: the last box, braces balanced; none when cut or empty", repr(wrong))
g = grader.grade("", "3", reasoning="... so \\boxed{3}")
check(g == {"answer": "3", "correct": True, "source": "reasoning"}, "grader: an answer only in the thinking", repr(g))
check(grader.grade(None, "3") == {"answer": None, "correct": False, "source": None}, "grader: no reply, no answer")

# ---------------------------------------------------------------- 2. the sets
for name in ("math500-quick", "math500-hard"):
    ids = reasoning.read_selection(name + ".txt")
    want = reasoning.SELECTIONS[name + ".txt"][1]
    check(len(ids) == want and len(set(ids)) == want, f"sets/{name}.txt: {want} problems, no repeats", str(len(ids)))
rows = [{"id": f"{s}/{i}", "level": 1 + i % 5, "subject": s} for s, n in (("A", 60), ("B", 30), ("C", 10))
        for i in range(n)]
pick = reasoning.select(rows, (1, 2, 3, 4, 5), 20)
by = {s: sum(p.startswith(s + "/") for p in pick) for s in "ABC"}
check(by == {"A": 12, "B": 6, "C": 2}, "select: 20 of 60/30/10 by subject are 12/6/2", repr(by))
check(sorted(pick) == sorted(reasoning.select(list(reversed(rows)), (1, 2, 3, 4, 5), 20)),
      "select: the same draw whatever the order of the rows")
check(all(int(p.split("/")[1]) % 5 + 1 in (4, 5) for p in reasoning.select(rows, (4, 5), 10)),
      "select: only the levels asked for")
check(reasoning.seed_of("s", "p", 0) != reasoning.seed_of("s", "p", 1) == reasoning.seed_of("s", "p", 1),
      "seeds: one per sample, the same every run")

# ---------------------------------------------------------------- 3. a run against a scripted server
PROBLEMS = [{"id": "p1", "problem": "What is 1+1?", "answer": "2"},
            {"id": "p2", "problem": "Halve 1.", "answer": "\\frac12"},
            {"id": "p3", "problem": "Solve x^2 = 1.", "answer": "1,-1"},
            {"id": "p4", "problem": "Think forever.", "answer": "5"}]
STATE = {"ctx": 40960, "requests": [], "fail_once": {("p1", 1)}, "p3": "\\boxed{1}"}


class Scripted(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake-model", "context_length": STATE["ctx"], "slots": 2}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        STATE["requests"].append(req)
        text = req["messages"][0]["content"]
        pid = next(p["id"] for p in PROBLEMS if text.startswith(p["problem"]))
        k = next(k for k in range(8) if reasoning.seed_of("reasoning-test", pid, k) == req["seed"])
        if (pid, k) in STATE["fail_once"]:
            STATE["fail_once"].discard((pid, k))
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        thinking, content, finish = {"p1": ("1 and 1", "It is \\boxed{2}.", "stop"),
                                     "p2": ("half", "\\boxed{0.5}", "stop"),
                                     "p3": ("squares", STATE["p3"], "stop"),
                                     "p4": ("so far \\boxed{5} maybe, and on", "", "length")}[pid]
        chunks = [{"role": "assistant", "content": ""}, {"content": ""}, {"reasoning_content": thinking}]
        if content:
            chunks.append({"content": content})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for i, d in enumerate(chunks + [{}]):
            ev = {"choices": [{"index": 0, "delta": d, "finish_reason": finish if i == len(chunks) else None}]}
            if i == len(chunks):
                ev["usage"] = {"prompt_tokens": 30, "completion_tokens": 100}
            self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


srv = Server(("127.0.0.1", 0), Scripted)
threading.Thread(target=srv.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{srv.server_address[1]}/v1"
tmp = tempfile.mkdtemp(prefix="reasoning-test-")
pfile = os.path.join(tmp, "reasoning-test.jsonl")
with open(pfile, "w") as f:
    f.writelines(json.dumps(p) + "\n" for p in PROBLEMS)
lock = os.path.join(tmp, "gpu.lock")
env = dict(os.environ, BENCH_GPU_LOCK=lock)
env.pop("NACE_GPU_LOCK_HELD", None)
env.pop("NI_LOCK_HELD", None)


def run(*args, **kw):
    return subprocess.run([sys.executable, SCRIPT, "--cache", os.path.join(tmp, "cache")] + list(args),
                          capture_output=True, text=True, timeout=120, env=kw.get("env", env))


out = os.path.join(tmp, "a.jsonl")
r = run("run", "--problems", pfile, "--base", BASE, "--samples", "2", "--parallel", "2", "--out", out,
        "--label", "base")
recs = [json.loads(line) for line in open(out)] if os.path.exists(out) else []
samples = [x for x in recs if x["type"] == "sample"]
check(r.returncode == 1 and len(samples) == 8 and sum(bool(x.get("error")) for x in samples) == 1,
      "run: 4 problems x 2 samples, the HTTP error recorded and the run exit 1", r.stdout[-400:] + r.stderr[-400:])
reqs = STATE["requests"]
q = reqs[0] if reqs else {}
check(q.get("temperature") == 0.6 and q.get("top_p") == 0.95 and q.get("top_k") == 20 and q.get("stream") is True
      and q.get("chat_template_kwargs") == {"enable_thinking": True} and q.get("max_tokens") == 32768
      and q["messages"][0]["content"].endswith("put your final answer within \\boxed{}."),
      "run: thinking on, Qwen's thinking sampling, the math instruction, max_tokens 32768", json.dumps(q)[:300])
seeds = {(x["id"], x["sample"]): x["seed"] for x in samples}
check(all(s == reasoning.seed_of("reasoning-test", pid, k) for (pid, k), s in seeds.items())
      and len(set(seeds.values())) == 8,
      "run: a seed of its own for every sample")
byid = {(x["id"], x["sample"]): x for x in samples if not x.get("error")}
check(byid[("p1", 0)]["correct"] and byid[("p2", 0)]["correct"] and not byid[("p3", 0)]["correct"],
      "run: right, right written differently, and wrong")
p4 = byid[("p4", 0)]
check(p4["finish"] == "length" and p4["answer"] is None and not p4["correct"],
      "run: a reply cut short is not graded on the box in its unfinished thinking",
      repr({k: p4.get(k) for k in ("finish", "answer")}))
hdr = recs[0] if recs else {}
check(hdr.get("type") == "run" and hdr.get("gpu_lock") == lock and os.path.exists(lock)
      and hdr.get("server", {}).get("id") == "fake-model" and hdr.get("label") == "base",
      "run: the header records the server and the GPU lock (a test lock file)", json.dumps(hdr)[:300])
n0 = len(reqs)
r = run("run", "--problems", pfile, "--base", BASE, "--samples", "2", "--out", out, "--label", "base")
check(r.returncode == 0 and len(reqs) == n0 + 1, "run: a resumed run redoes only the failed sample",
      f"{len(reqs) - n0} requests; " + r.stdout[-300:])
STATE["ctx"] = 16384
r = run("run", "--problems", pfile, "--base", BASE, "--samples", "1", "--limit", "1", "--no-gpu-lock",
        "--out", os.path.join(tmp, "c.jsonl"))
check(r.returncode == 0 and reqs[-1].get("max_tokens") == 16384 - 2048 and "max_tokens 32768 -> 14336" in r.stdout,
      "run: max_tokens fits the server's context", r.stdout[-300:])
STATE["ctx"] = 40960

# --serve: a static server (python -m http.server) answers /v1/models; the run starts it, sees it ready, and stops it
static = os.path.join(tmp, "static")
os.makedirs(os.path.join(static, "v1"))
with open(os.path.join(static, "v1", "models"), "w") as f:
    json.dump({"data": [{"id": "static", "context_length": 40960, "slots": 1}]}, f)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
sout = os.path.join(tmp, "served.jsonl")
r = run("run", "--problems", pfile, "--base", f"http://127.0.0.1:{port}/v1", "--limit", "1", "--samples", "1",
        "--out", sout, "--serve", f"{sys.executable} -m http.server {port} --bind 127.0.0.1 --directory {static}")
hdr = json.loads(open(sout).readline()) if os.path.exists(sout) else {}
probe = socket.socket()
closed = probe.connect_ex(("127.0.0.1", port)) != 0
probe.close()
check(r.returncode == 1 and "server ready" in r.stdout and hdr.get("server", {}).get("id") == "static"
      and hdr.get("serve", "").endswith(static) and hdr.get("gpu_lock") == lock and closed,
      "run --serve: the server started under the lock, used once it answers, and stopped after",
      r.stdout[-400:] + r.stderr[-300:])
r = run("run", "--problems", pfile, "--base", "http://127.0.0.1:9/v1", "--limit", "1", "--no-gpu-lock",
        "--out", os.path.join(tmp, "dead.jsonl"), "--serve", f"{sys.executable} -c \"raise SystemExit(3)\"")
check(r.returncode == 2 and "exited with code 3" in r.stderr, "run --serve: a server that dies is an error",
      r.stderr[-300:])

# ---------------------------------------------------------------- 4. the report
STATE["p3"] = "\\boxed{-1, 1}"
out2 = os.path.join(tmp, "b.jsonl")
r = run("run", "--problems", pfile, "--base", BASE, "--samples", "2", "--out", out2, "--label", "other",
        "--no-gpu-lock")
check(r.returncode == 0, "run: a second label", r.stdout[-300:])
js = os.path.join(tmp, "report.json")
r = run("report", out, out2, "--json", js)
rep = json.load(open(js)) if os.path.exists(js) else {}
a = rep.get("labels", {}).get("reasoning-test", {})
base, other = a.get("base", {}), a.get("other", {})
check(base.get("accuracy") == 0.5 and other.get("accuracy") == 0.75 and base.get("samples") == 8,
      "report: accuracy is the mean of per-problem accuracy", json.dumps(a)[:300])
check(base.get("cut_short") == 2 and base.get("no_answer") == 2 and base.get("problems") == 4,
      "report: replies cut short and without an answer counted", json.dumps(base)[:300])
lo, hi = base.get("ci95", [None, None])
check(lo is not None and lo <= 0.5 <= hi, "report: the interval holds the accuracy", repr((lo, hi)))
pp = rep.get("paired", {}).get("reasoning-test", {}).get("other", {})
check(pp.get("baseline") == "base" and pp.get("difference") == 0.25 and pp.get("better") == 1 and pp.get("worse") == 0
      and "mcnemar_p" not in pp, "report: the second label against the baseline, problem by problem", json.dumps(pp))
check("| other | 4 | +25.0 pts" in r.stdout, "report: the markdown table", r.stdout[-500:])
check(reasoning._mcnemar(0, 5) == 0.0625 and reasoning._mcnemar(3, 3) == 1.0, "report: McNemar's exact test")
stale = os.path.join(tmp, "stale.jsonl")
with open(stale, "w") as f:
    f.write(json.dumps({"type": "run", "label": "old", "set": "s"}) + "\n")
    f.write(json.dumps({"type": "sample", "set": "s", "id": "p", "sample": 0, "gold": "\\frac12", "finish": "stop",
                        "content": "so \\boxed{0.5}", "reasoning": "", "answer": "0.5", "correct": False}) + "\n")
js2 = os.path.join(tmp, "stale.json")
run("report", stale, "--json", js2)
before = json.load(open(js2))["labels"]["s"]["old"]["accuracy"]
run("report", stale, "--regrade", "--json", js2)
after = json.load(open(js2))["labels"]["s"]["old"]["accuracy"]
check(before == 0.0 and after == 1.0, "report --regrade: saved replies graded again with the current grader",
      repr((before, after)))

# ---------------------------------------------------------------- 5. with the problems downloaded
cache = reasoning.cache_dir()
try:
    reasoning.load_dataset("math500", cache, download=False)
    reasoning.load_dataset("aime2025", cache, download=False)
    have = True
except (FileNotFoundError, RuntimeError):
    have = False
if have:
    r = subprocess.run([sys.executable, SCRIPT, "check"], capture_output=True, text=True, timeout=120)
    check(r.returncode == 0, "check: every MATH-500 reference solution graded right, every AIME answer an integer",
          r.stdout[-500:])
    r = subprocess.run([sys.executable, SCRIPT, "sets", "--rebuild"], capture_output=True, text=True, timeout=120)
    check(r.returncode == 0 and r.stdout.count("matches the draw") == 2, "sets: the frozen lists are the draw",
          r.stdout[-300:])
else:
    print(f"SKIP check and sets --rebuild: the problems are not in {cache} (reasoning.py fetch)")

srv.shutdown()
print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all passed'}")
sys.exit(1 if FAILS else 0)
