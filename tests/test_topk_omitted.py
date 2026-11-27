#!/usr/bin/env python3
"""A request that leaves top_k out keeps its speculative decoding (kapitsa#1286; model + GPU required).

  python3 tests/test_topk_omitted.py --cli build/kapitsa --model ~/models/qwen3.6-35b-a3b-4bit

  python3 tests/test_topk_omitted.py --cli build/kapitsa --model ~/models/qwen3.8-27b-4bit --review

Starts `kapitsa serve` on the model and sends sampled requests (temperature 0.7) that leave top_k out on the
OpenAI chat completions, Ollama /api/chat and /api/generate, Anthropic /v1/messages and OpenAI /v1/responses
routes, then the same with an explicit top_k 0 on chat completions and Ollama, and top_k 40 and 64. Since #1286 a
request's top_k <= 0 or above 64 is sampled over the whole vocabulary (src/model/sampler.h; in plain steps, or in
Nemotron-H's KFlash tree verify: tests/test_topk_full_tree.py), which the
engine logs as "sampled over the whole vocabulary". The check: no request without a top_k, or with top_k 40 or 64,
logs it (each keeps the two-pass sampler and the model's speculative loop), and each explicit top_k 0 does.

--review adds the routes the review of #1287 named, each at top_k 40 and 0 (and 100 for structured output and tools): structured output (json_schema) and tool
calls; a short suffix after 1024+ cached rows (the verify-pass prefill, NI_VPF: a dense target with DFlash2 takes it,
the 27B; the server's log names it at NI_LOG=2) at top_k 0 and 40, with the server still up after each; a greedy turn
on the speculative loop after a turn in plain steps (its tok/cycle against the same turn after a re-prefill of the
same conversation); and, on a second server with 2 slots, a top_k 0 request decoded while a batch-eligible top_k 20
request runs. Prints each request's decode tok/s from the server's log. Takes the shared GPU lock, or respects
NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench/harness"))
from gpu_hogs import GpuLock, held_by_caller  # noqa: E402

FULL = "sampled over the whole vocabulary"
PROMPT = "Write a short story about a lighthouse keeper and a storm."


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def requests(n):
    msgs = [{"role": "user", "content": PROMPT}]
    oa = {"model": "m", "messages": msgs, "temperature": 0.7, "max_tokens": n, "stream": False,
          "chat_template_kwargs": {"enable_thinking": False}}
    ol = {"model": "m", "messages": msgs, "stream": False, "think": False,
          "options": {"temperature": 0.7, "num_predict": n}}
    og = {"model": "m", "prompt": PROMPT, "stream": False, "think": False,
          "options": {"temperature": 0.7, "num_predict": n}}
    an = {"model": "m", "messages": msgs, "temperature": 0.7, "max_tokens": n}
    rs = {"model": "m", "input": PROMPT, "temperature": 0.7, "max_output_tokens": n}
    return [
        ("chat completions, top_k omitted", "/v1/chat/completions", oa, False),
        ("Ollama /api/chat, top_k omitted", "/api/chat", ol, False),
        ("Ollama /api/generate, top_k omitted", "/api/generate", og, False),
        ("Anthropic /v1/messages, top_k omitted", "/v1/messages", an, False),
        ("OpenAI /v1/responses, top_k omitted", "/v1/responses", rs, False),
        ("chat completions, top_k 0", "/v1/chat/completions", dict(oa, top_k=0), True),
        ("Ollama /api/chat, top_k 0", "/api/chat", dict(ol, options=dict(ol["options"], top_k=0)), True),
        ("chat completions, top_k 40", "/v1/chat/completions", dict(oa, top_k=40), False),
        ("Ollama /api/chat, top_k 64", "/api/chat", dict(ol, options=dict(ol["options"], top_k=64)), False),
    ]


SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "required": ["name", "age"], "additionalProperties": False}
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a text file from disk.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
LONG = " ".join(f"Item {i}: the keeper logged wind {i % 9} and swell {i % 5} at hour {i % 24}." for i in range(110))


class Server:
    def __init__(self, a, tmp, slots, name):
        self.port = a.port + slots
        self.log_path = Path(os.environ.get("TOPK_LOG_DIR") or tmp) / f"serve-{name}.log"
        env = dict(os.environ, NI_LOCK_HELD="1", NACE_GPU_LOCK_HELD="1", NI_LOG="2")
        if a.expect_vpf:  # (forced: the default leaves some dense targets out, Engine::chat's vpf_default)
            env["NI_VPF"] = "1"
        with open(self.log_path, "w") as log:
            self.p = subprocess.Popen([a.cli, "serve", "--model", a.model, "--port", str(self.port), "--ctx", "8192",
                                       "--slots", str(slots), "--disk-cache", "0"], stdout=log,
                                      stderr=subprocess.STDOUT, env=env)
        for _ in range(900):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2)
                return
            except Exception:
                if self.p.poll() is not None:
                    sys.exit(f"server exited: {self.log_path.read_text()[-2000:]}")
                time.sleep(1)

    def text(self):
        return self.log_path.read_text(errors="replace")

    def up(self):
        if self.p.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return True
        except Exception:
            return False

    def chat(self, body):
        """(the reply's message, the log lines it wrote)"""
        before = len(self.text())
        r = json.loads(post(self.port, "/v1/chat/completions", dict({"model": "m", "stream": False}, **body)))
        time.sleep(0.5)
        return r["choices"][0]["message"], self.text()[before:]

    def close(self):
        self.p.terminate()
        self.p.wait(60)


def cycles(new):
    m = re.findall(r"([\d.]+) tok/cycle", new)
    return float(m[-1]) if m else 0.0


def review(a, tmp):
    fails = 0

    def check(ok, what):
        nonlocal fails
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}: {what}")

    nt = {"chat_template_kwargs": {"enable_thinking": False}}
    s = Server(a, tmp, 1, "review")
    try:
        for k in (40, 0, 100):
            m, new = s.chat(dict(nt, messages=[{"role": "user", "content": "Invent a person. Reply with JSON."}],
                                 temperature=0.7, top_k=k, max_tokens=128,
                                 response_format={"type": "json_schema",
                                                  "json_schema": {"name": "p", "schema": SCHEMA, "strict": True}}))
            try:
                obj = json.loads(m["content"])
                ok = set(obj) == {"name", "age"} and isinstance(obj["age"], int)
            except Exception:
                ok = False
            check(ok and (FULL in new) == (k != 40), f"structured output at top_k {k}: {m['content']!r:.80}"
                  f"{', whole vocabulary' if FULL in new else ''}")
            m, new = s.chat(dict(nt, messages=[{"role": "user", "content": "Read /etc/hosts with the tool."}],
                                 temperature=0.7, top_k=k, max_tokens=256, tools=TOOLS))
            calls = m.get("tool_calls") or []
            ok = bool(calls) and calls[0]["function"]["name"] == "read_file" and \
                "path" in json.loads(calls[0]["function"]["arguments"])
            check(ok and (FULL in new) == (k != 40), f"tool call at top_k {k}: "
                  f"{calls[0]['function'] if calls else m.get('content')!r:.100}")
        # a short suffix after 1024+ cached rows
        conv = [{"role": "user", "content": LONG + "\nSummarize the log in one sentence."}]
        # (the template drops an earlier reply's think tags, so the cached prefix ends at the user's turn and the
        # suffix holds the reply too: replies of 8 tokens keep it inside vpf's 25 to 64 rows)
        m, new = s.chat(dict(nt, messages=conv, temperature=0.7, top_k=20, max_tokens=8))
        conv += [{"role": "assistant", "content": m["content"]}]
        for k in (0, 40):
            # (25 to 64 suffix rows with the turn boundary inside: vpf's window, NI_VPF_SPLIT_MIN / NI_VPF_MAX)
            conv += [{"role": "user", "content": "And the wind and the swell at hours 5 and 6? One line, numbers only."}]
            m, new = s.chat(dict(nt, messages=conv, temperature=0.7, top_k=k, max_tokens=8))
            conv += [{"role": "assistant", "content": m["content"]}]
            vp = re.findall(r"prefill: (\d+) suffix rows after (\d+) cached as verify passes", new)
            why = re.findall(r"prefill: \d+ suffix rows after \d+ cached, not as verify passes:.*", new)
            check(s.up() and (FULL in new) == (k == 0) and (not vp if k == 0 else bool(vp) or not a.expect_vpf),
                  f"short suffix after the cached prompt, top_k {k}: server up, "
                  f"{'verify-pass prefill ' + str(vp) if vp else 'the prefill path'}{' (' + why[-1] + ')' if why else ''}"
                  f"{', whole vocabulary' if FULL in new else ''}")
        # a greedy speculative turn after a turn in plain steps, against the same turn after a re-prefill
        c1 = [{"role": "user", "content": PROMPT}]
        m, new = s.chat(dict(nt, messages=c1, temperature=0.7, top_k=0, max_tokens=128))
        check(FULL in new, "turn 1 at top_k 0: " + ("KFlash's tree verify" if "tree verify" in new else "plain steps"))
        c2 = c1 + [{"role": "assistant", "content": m["content"]}, {"role": "user", "content": "Now tell it in verse."}]
        m2, new2 = s.chat(dict(nt, messages=c2, temperature=0, max_tokens=256))
        s.chat(dict(nt, messages=[{"role": "user", "content": "Name three rivers."}], temperature=0, max_tokens=8))
        m3, new3 = s.chat(dict(nt, messages=c2, temperature=0, max_tokens=256))
        t2, t3 = cycles(new2), cycles(new3)
        check(t2 > 1.0 and t2 >= 0.7 * t3, f"greedy turn 2 after a plain-step turn: {t2:.2f} tok/cycle, "
              f"re-prefilled {t3:.2f}; replies {'identical' if m2['content'] == m3['content'] else 'differ'}")
    finally:
        up = s.up()
        s.close()
    check(up, "the 1-slot server stayed up through every request")
    # a top_k 0 request decoded while a batch-eligible one runs
    s = Server(a, tmp, 2, "two-slots")
    try:
        out = {}

        def run(k):
            out[k] = s.chat(dict(nt, messages=[{"role": "user", "content": PROMPT}], temperature=0.7, top_k=k,
                                 max_tokens=256))

        th = [threading.Thread(target=run, args=(k,)) for k in (20, 0)]
        for t in th:
            t.start()
            time.sleep(0.3)
        for t in th:
            t.join()
        n = len(re.findall(FULL, s.text()))
        check(set(out) == {0, 20} and all(out[k][0]["content"] for k in out) and n == 1 and s.up(),
              f"top_k 0 next to a top_k 20 request (2 slots): both replied, whole vocabulary logged {n}x")
    finally:
        s.close()
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18590)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--review", action="store_true", help="also the routes #1287's review named (see the top)")
    ap.add_argument("--review-only", action="store_true", help="--review without the omitted-top_k requests")
    ap.add_argument("--expect-vpf", action="store_true",
                    help="--review: NI_VPF=1, and the top_k 40 suffix must take the verify-pass prefill (a target with DFlash2)")
    a = ap.parse_args()
    lock = nullcontext() if held_by_caller() else GpuLock(
        path=str(Path(os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock").expanduser().resolve()))
    failures = 0
    with lock, tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "serve.log"
        env = dict(os.environ, NI_LOCK_HELD="1", NACE_GPU_LOCK_HELD="1")
        with open(log_path, "w") as log:
            p = subprocess.Popen([a.cli, "serve", "--model", a.model, "--port", str(a.port), "--ctx", "8192",
                                  "--slots", "1", "--disk-cache", "0"], stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            for _ in range(900):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
                    break
                except Exception:
                    if p.poll() is not None:
                        sys.exit(f"server exited: {log_path.read_text()[-2000:]}")
                    time.sleep(1)
            for name, path, body, full in [] if a.review_only else requests(a.tokens):
                before = log_path.read_text()
                post(a.port, path, body)
                time.sleep(0.5)  # (the request's log lines)
                new = log_path.read_text()[len(before):]
                rates = re.findall(r"tokens in \d+ ms \(([\d.]+) tok/s\)", new)
                spec = re.findall(r"tok/cycle", new)
                took_full = FULL in new
                ok = took_full == full
                failures += not ok
                how = "two-pass" if not took_full else (
                    "whole vocabulary, " + ("tree verify" if "tree verify" in new else "plain steps"))
                print(f"{'PASS' if ok else 'FAIL'}: {name}: {how}"
                      f"{', speculative' if spec else ''}, {rates[-1] if rates else '?'} tok/s")
        finally:
            p.terminate()
            p.wait(60)
        if a.review or a.review_only:
            failures += review(a, tmp)
    print(f"test_topk_omitted: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
