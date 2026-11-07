#!/usr/bin/env python3
"""Completed prefix checkpoints survive a cancelled continuation prefill.

Needs an existing model and a Mac GPU. Holds BENCH_GPU_LOCK, uses temporary
KAPITSA_HOME directories and disables the SSD cache and ANE. The reference leg
finishes a long continuation; the candidate cancels it twice during prefill,
then retries it. A cold cancellation also crosses a new system checkpoint and
must not publish that unfinished request's state. Save each run in a new output
folder. This is a cache/output regression, not a throughput benchmark or logit gate.

Usage: test_prefill_cancel.py --cli build/kapitsa --model MODEL --output NEW_DIR
       [--reference OTHER_BINARY] [--port 8298]
"""
import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def request(port, output, name, messages, cancel_after=None):
    body = {"messages": messages, "max_tokens": 32, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "return_progress": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    (output / (name + "-request.json")).write_text(json.dumps(body) + "\n")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    result = {"name": name, "progress": [], "text": "", "reasoning": "", "cancelled": False}
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{name}: HTTP {response.status}: {response.read().decode()}")
        with (output / (name + "-events.txt")).open("wb") as events:
            for line in response:
                events.write(line)
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                data = json.loads(line[6:])
                progress = data.get("prompt_progress")
                if progress:
                    result["progress"].append(progress)
                    if (cancel_after is not None and
                            progress["processed"] > max(progress["cache"], cancel_after) and
                            progress["processed"] < progress["total"]):
                        if conn.sock:
                            conn.sock.shutdown(socket.SHUT_RDWR)
                        response.close()
                        result["cancelled"] = True
                        break
                if data.get("usage"):
                    result["usage"] = data["usage"]
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    result["text"] += delta.get("content") or ""
                    result["reasoning"] += delta.get("reasoning_content") or ""
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
    finally:
        conn.close()
    (output / (name + "-result.json")).write_text(json.dumps(result, indent=2) + "\n")
    if cancel_after is not None and not result["cancelled"]:
        raise RuntimeError(f"{name}: no mid-prefill cancellation")
    if cancel_after is None and not result.get("usage"):
        raise RuntimeError(f"{name}: missing terminal usage")
    return result


class Server:
    def __init__(self, cli, model, port, home, log):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))  # refuse a port another server already owns
        self.path = log
        self.log = log.open("w")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "KAPITSA_"))}
        env.update(KAPITSA_HOME=str(home), KAPITSA_NO_AUTO_UPDATE="1", KAPITSA_NO_PROMPT="1",
                   NI_ANE="0", NI_LOG="2", NI_LOCK_HELD="1")
        self.command = [str(cli), "serve", "--model", model, "--host", "127.0.0.1", "--private",
                        "--port", str(port), "--ctx", "32768", "--slots", "1", "--disk-cache", "0"]
        self.process = subprocess.Popen(self.command, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"server exited; see {log}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            return
                except (urllib.error.URLError, OSError):
                    time.sleep(0.25)
            raise RuntimeError(f"server readiness timeout; see {log}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()

    def cancel(self, port, output, name, messages, after=0):
        mark = len(self.path.read_text(errors="replace"))
        result = request(port, output, name, messages, cancel_after=after)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            text = self.path.read_text(errors="replace")[mark:]
            match = re.search(r"cancelled during prefill at (\d+) of (\d+) tokens", text)
            if match:
                if not 0 < int(match[1]) < int(match[2]):
                    raise RuntimeError("server did not cancel inside the prompt")
                (output / (name + "-cancel.log")).write_text(text)
                return result
            time.sleep(0.1)
        raise RuntimeError(f"{name}: server did not confirm prefill cancellation")


def cached(result):
    return result["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)


def main():
    if len(sys.argv) == 1:
        print("SKIP: needs --cli, --model and --output on a Mac with a GPU")
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8298)
    args = parser.parse_args()
    cli = args.cli.resolve()
    reference = (args.reference or cli).resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    lock = None
    if not any(os.environ.get(k) == "1" for k in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD")):
        path = Path(os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock"))
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.open("a")
        fcntl.flock(lock, fcntl.LOCK_EX)
    corpus = (Path(__file__).resolve().parents[1] / "bench/harness/corpus/bench_corpus.txt").read_text()
    first = [{"role": "system", "content": "You are a careful code reviewer, cancellation test."},
             {"role": "user", "content": "Read this source:\n" + corpus[:8500] +
              "\nName one function in one sentence."}]
    checks = []
    results = {}
    try:
        with tempfile.TemporaryDirectory(prefix="kapitsa-prefill-cancel-") as tmp:
            for leg, binary in (("control", reference), ("cancel", cli)):
                server = Server(binary, args.model, args.port, Path(tmp) / leg, args.output / (leg + "-server.log"))
                try:
                    seed = request(args.port, args.output, leg + "-seed", first)
                    prefix = first + [{"role": "assistant", "content": seed["text"]}]
                    time.sleep(1)  # allow the normal idle history prefill
                    short = request(args.port, args.output, leg + "-short", prefix +
                                    [{"role": "user", "content": "Name one possible failure in one sentence."}])
                    checks.append((leg + " has a cached prefix", cached(short) >= 64))
                    long_request = prefix + [{"role": "user", "content": "Review these additional files:\n" +
                                             corpus[10000:65000] + "\nName another function in one sentence."}]
                    if leg == "cancel":
                        server.cancel(args.port, args.output, "cancel-first", long_request)
                        server.cancel(args.port, args.output, "cancel-second", long_request)
                    reply = request(args.port, args.output, leg + "-reply", long_request)
                    results[leg] = {"seed": seed, "short": short, "reply": reply}
                    if leg == "cancel":
                        checks.append(("retry keeps the completed prefix", cached(reply) >= cached(short)))
                        again = request(args.port, args.output, "cancel-repeat", long_request)
                        checks.append(("completed retry can be reused", cached(again) >= cached(reply)))
                        checks.append(("repeat output matches", (again["text"], again["reasoning"]) ==
                                       (reply["text"], reply["reasoning"])))
                        cold = [{"role": "system", "content": "A separate review.\n" + corpus[:12000]},
                                {"role": "user", "content": corpus[10000:65000] + "\nName a function."}]
                        server.cancel(args.port, args.output, "cold-cancel", cold, after=6000)
                        cold_reply = request(args.port, args.output, "cold-retry", cold)
                        checks.append(("cold cancellation discards unfinished checkpoints", cached(cold_reply) == 0))
                finally:
                    server.stop()
            checks.append(("seed output matches reference", results["control"]["seed"]["text"] ==
                           results["cancel"]["seed"]["text"]))
            a, b = results["control"]["reply"], results["cancel"]["reply"]
            checks.append(("retry output matches uninterrupted reference", (a["text"], a["reasoning"], a["finish_reason"]) ==
                           (b["text"], b["reasoning"], b["finish_reason"])))
    finally:
        if lock:
            lock.close()
    report = {"cli": str(cli), "reference": str(reference), "model": args.model,
              "flags": {"NI_ANE": "0", "disk_cache": "0", "slots": 1, "ctx": 32768},
              "checks": [{"name": name, "pass": ok} for name, ok in checks]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, ok in checks:
        print(("PASS " if ok else "FAIL ") + name, flush=True)
    return int(any(not ok for _, ok in checks))


if __name__ == "__main__":
    sys.exit(main())
