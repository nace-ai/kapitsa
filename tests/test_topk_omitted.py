#!/usr/bin/env python3
"""A request that leaves top_k out keeps its speculative decoding (kapitsa#1286; model + GPU required).

  python3 tests/test_topk_omitted.py --cli build/kapitsa --model ~/models/qwen3.6-35b-a3b-4bit

Starts `kapitsa serve` on the model and sends sampled requests (temperature 0.7) that leave top_k out on the
OpenAI chat completions, Ollama /api/chat and /api/generate, Anthropic /v1/messages and OpenAI /v1/responses
routes, then the same with an explicit top_k 0 on chat completions and Ollama. Since #1286 a request's top_k <= 0 or
above 32 is sampled over the whole vocabulary in plain steps (src/model/sampler.h), which the engine logs as
"sampled over the whole vocabulary". The check: no request without a top_k logs it (each keeps the two-pass sampler
and the model's speculative loop), and each explicit top_k 0 does. Prints each request's decode tok/s from the
server's log. Takes the shared GPU lock, or respects NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1.
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
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18590)
    ap.add_argument("--tokens", type=int, default=256)
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
            for name, path, body, full in requests(a.tokens):
                before = log_path.read_text()
                post(a.port, path, body)
                time.sleep(0.5)  # (the request's log lines)
                new = log_path.read_text()[len(before):]
                rates = re.findall(r"tokens in \d+ ms \(([\d.]+) tok/s\)", new)
                spec = re.findall(r"tok/cycle", new)
                took_full = FULL in new
                ok = took_full == full
                failures += not ok
                print(f"{'PASS' if ok else 'FAIL'}: {name}: {'whole vocabulary, plain steps' if took_full else 'two-pass'}"
                      f"{', speculative' if spec else ''}, {rates[-1] if rates else '?'} tok/s")
        finally:
            p.terminate()
            p.wait(60)
    print(f"test_topk_omitted: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
