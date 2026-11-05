#!/usr/bin/env python3
"""Model-switch allocation, rollback and request-draining regression (issue #932).

Needs a local Qwen model and a Mac GPU; a 4B model is enough. Two temporary names
share its weights, so repeated switches exercise unload without downloading a
second model. A broken tokenizer fails after weights have loaded. The server uses
one slot, 256-row prefill, no drafter or disk cache, and keeps history prefill on.
This checks memory and outputs, not speed. Example:
  python3 tests/test_model_switch_memory.py build/kapitsa /path/to/4b --out /tmp/switch-check
The GPU lock is held for the server lifetime; NI_LOCK_HELD=1 or
NACE_GPU_LOCK_HELD=1 means the caller already holds it. No arguments skips the GPU.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

MEMORY = re.compile(r"memory: ([0-9.]+) GB allocated.*?KV room ([0-9.]+) GB")
# The log rounds to 10 MB; allow that and small lazy pipeline allocations, not a
# growing model allocation (the original 4B leak is about 430 MB per switch).
TOLERANCE_GB = 0.05


def check(condition, description):
    if not condition:
        raise AssertionError(description)
    print("PASS " + description, flush=True)


def run(args, scratch, out):
    models = [scratch / "model-a", scratch / "model-b"]
    for model in models:
        model.symlink_to(args.model.resolve(), target_is_directory=True)
    broken = scratch / "broken-tokenizer"
    broken.mkdir()
    for entry in args.model.resolve().iterdir():
        if entry.name != "tokenizer.json":
            (broken / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    (broken / "tokenizer.json").write_text("{broken tokenizer\n")
    log_path = out / "server.log"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"

    def request(path, body=None, timeout=180):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)

    def text():
        return log_path.read_text(errors="replace")

    def allocations():
        return [float(allocated) for allocated, _ in MEMORY.findall(text())]

    def payload(model, content="Reply with exactly one word: Hello", tokens=16):
        return {"model": str(model), "messages": [{"role": "user", "content": content}],
                "max_tokens": tokens, "temperature": 0, "seed": 42,
                "chat_template_kwargs": {"enable_thinking": False}}

    def reply(model):
        code, data = request("/v1/chat/completions", payload(model))
        check(code == 200 and data.get("usage", {}).get("completion_tokens", 0) > 0, "nonempty reply")
        return data["choices"][0]["message"], data["usage"]["completion_tokens"]

    env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "NACE_", "KAPITSA_"))}
    env.update(KAPITSA_HOME=str(scratch / "home"), KAPITSA_NO_AUTO_UPDATE="1", KAPITSA_NO_PROMPT="1",
               HF_ENDPOINT="http://127.0.0.1:9", KAPITSA_UPDATE_URL="http://127.0.0.1:9",
               NI_LOCK_HELD="1", NI_LOG="1", NI_ANE="0")
    command = [str(args.cli.resolve()), "serve", "--model", str(models[0]), "--host", "127.0.0.1", "--private",
               "--port", str(port), "--allow-switch", "--context", "2048", "--slots", "1", "--prefill-chunk", "256",
               "--spec", "0", "--dflash", "none", "--disk-cache", "0"]
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    stop = threading.Event()
    metadata_errors = []
    metadata_reads = []
    snapshots = []
    watcher = None
    with log_path.open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if server.poll() is not None:
                    raise RuntimeError(text())
                try:
                    if request("/health", timeout=1)[0] == 200:
                        break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup")
                time.sleep(0.2)

            def read_metadata():
                # /cluster uses current_engine(), the metadata handle that also has
                # to leave the old engine before its allocations may be reclaimed.
                while not stop.wait(0.01):
                    try:
                        code, _ = request("/cluster", timeout=30)
                        if code != 200:
                            raise RuntimeError(f"/cluster returned {code}")
                        metadata_reads.append(1)
                    except Exception as exc:
                        metadata_errors.append(str(exc))
                        return

            watcher = threading.Thread(target=read_metadata)
            watcher.start()
            expected = reply(models[0])
            initial = allocations()[0]

            def stable(label):
                rows = allocations()
                check(bool(rows), "allocation was logged")
                current = rows[-1]
                snapshots.append({"label": label, "allocated_gb": current})
                check(current <= initial + TOLERANCE_GB,
                      f"{label}: {current:.2f} GB (initial {initial:.2f} GB)")

            stable("initial")
            for index in range(1, 5):
                model = models[index % 2]
                before = len(allocations())
                if index % 2:
                    code, data = request("/v1/models/switch", {"model": str(model)})
                    check(code == 200 and data.get("status") == "ok", "explicit switch")
                check(reply(model) == expected, "same reply after switch")
                check(len(allocations()) == before + 1, "switch really reloaded the model")
                stable(f"switch {index}")

            before = len(allocations())
            code, _ = request("/v1/models/switch", {"model": str(scratch / "missing")})
            check(code == 400 and len(allocations()) == before, "missing model leaves the running engine alone")

            for index in range(2):
                mark = len(text())
                code, data = request("/v1/models/switch", {"model": str(broken)})
                check(code == 400 and "parse_error" in data.get("error", {}).get("message", ""),
                      "tokenizer failure is returned to the client")
                check("loaded " + str(broken) in text()[mark:] and "reloading " in text()[mark:],
                      "failure happened after loading weights, then rolled back")
                code, active = request("/v1/models")
                check(code == 200 and active["data"][0]["id"] == models[0].name, "old model restored")
                check(reply(models[0]) == expected, "same reply after failed switch")
                stable(f"rollback {index + 1}")

            # Start a switch while the old model is still producing a streamed
            # answer. The log orders the completed request before the unload.
            mark = len(text())
            body = {**payload(models[0], "Count upwards from one to ten thousand, without skipping numbers.", 128),
                    "stream": True, "stream_options": {"include_usage": True}}
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with ThreadPoolExecutor(max_workers=1) as pool, urllib.request.urlopen(req, timeout=180) as response:
                switched = None
                done = False
                usage = {}
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    if line.strip() == b"data: [DONE]":
                        done = True
                        break
                    event = json.loads(line[6:])
                    if "error" in event:
                        raise AssertionError(f"stream failed: {event['error']}")
                    if event.get("usage"):
                        usage = event["usage"]
                    if switched is None and any(c.get("delta", {}).get("content") for c in event.get("choices", [])):
                        switched = pool.submit(request, "/v1/models/switch", {"model": str(models[1])})
                check(done and usage.get("completion_tokens") == 128, "in-flight stream completes all 128 tokens")
                check(switched is not None and switched.result()[0] == 200, "concurrent switch completes")
            tail = text()[mark:]
            check(tail.index(" chat: ") < tail.index(" switching model from "), "old request finishes before unload")
            stable("switch after streaming")
            check(reply(models[1]) == expected, "new model answers after concurrent switch")
            stop.set()
            watcher.join(timeout=31)
            check(not watcher.is_alive(), "metadata reader finished")
            check(bool(metadata_reads) and not metadata_errors, "metadata reads survived switches and rollbacks")
            print("PASS model-switch memory regression", flush=True)
        finally:
            stop.set()
            if watcher:
                watcher.join(timeout=31)
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            (out / "allocations.json").write_text(json.dumps({"snapshots": snapshots,
                "all_loads_gb": allocations(), "metadata_reads": len(metadata_reads),
                "metadata_errors": metadata_errors}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cli", type=Path, nargs="?")
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.cli is None and args.model is None:
        print("SKIP: needs a kapitsa binary, a local Qwen model and a Mac GPU")
        return
    if args.cli is None or not args.cli.is_file() or args.model is None or not (args.model / "config.json").is_file():
        parser.error("give an existing binary and a local model folder")
    with tempfile.TemporaryDirectory(prefix="kapitsa-switch-") as directory:
        scratch = Path(directory)
        out = args.out.resolve() if args.out else scratch / "evidence"
        out.mkdir(parents=True, exist_ok=False)
        lock_path = Path(os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock"))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            if os.environ.get("NI_LOCK_HELD") != "1" and os.environ.get("NACE_GPU_LOCK_HELD") != "1":
                fcntl.flock(lock, fcntl.LOCK_EX)
            run(args, scratch, out)


if __name__ == "__main__":
    main()
