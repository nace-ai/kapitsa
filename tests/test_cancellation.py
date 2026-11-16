import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cli", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--port", type=int, default=8198)
    parser.add_argument("--server-args", default="")
    args = parser.parse_args()
    lock_path = Path(os.environ.get("BENCH_GPU_LOCK", "/tmp/kapitsa-gpu.lock"))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o644), "rb") as lock, tempfile.TemporaryDirectory(prefix="kapitsa-cancellation-") as directory:
        if os.environ.get("NI_LOCK_HELD") != "1" and os.environ.get("NACE_GPU_LOCK_HELD") != "1":
            fcntl.flock(lock, fcntl.LOCK_EX)
        root = Path(directory)
        log_path = root / "server.log"
        env = {**os.environ, "KAPITSA_HOME": str(root), "KAPITSA_NO_AUTO_UPDATE": "1", "NI_ANE": "0",
               "NI_HISTORY_PREFILL": "0", "NI_LOCK_HELD": "1"}
        with log_path.open("w") as log:
            server = subprocess.Popen([
                str(args.cli.resolve()), "serve", "--model", str(args.model.resolve()),
                "--model-id", "test", "--port", str(args.port), "--context", "32768", "--slots", "1",
                "--disk-cache", "0", "--prefill-chunk", "256",
                *shlex.split(args.server_args),
            ], env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 180
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(log_path.read_text())
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1):
                            break
                    except (OSError, urllib.error.URLError):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Server did not become healthy")
                        time.sleep(0.1)
                prompt = " ".join(f"item{index}: read this line carefully." for index in range(1200))
                cases = [
                    ("/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}],
                                               "chat_template_kwargs": {"enable_thinking": False}}),
                    ("/v1/messages", {"messages": [{"role": "user", "content": prompt + " messages"}]}),
                    ("/v1/responses", {"input": prompt + " responses", "max_output_tokens": 1024}),
                ]
                for route, body in cases:
                    offset = log_path.stat().st_size
                    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=60)
                    try:
                        connection.request("POST", route, json.dumps({"model": "test", "max_tokens": 1024,
                                                                     **body, "stream": True}),
                                           {"Content-Type": "application/json"})
                        response = connection.getresponse()
                        if response.status != 200:
                            raise RuntimeError(response.read().decode())
                        if not response.readline():
                            raise RuntimeError("The stream ended before prefill cancellation")
                        time.sleep(0.2)
                        response.close()
                    finally:
                        connection.close()
                    cancelled_at = time.monotonic()
                    deadline = cancelled_at + 15
                    while "cancelled during prefill" not in log_path.read_text()[offset:]:
                        if server.poll() is not None:
                            raise RuntimeError(log_path.read_text())
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"{route}: prefill did not stop within 15 seconds: "
                                               + log_path.read_text()[offset:])
                        time.sleep(0.1)
                    print(f"PASS {route}: prefill stopped in {time.monotonic() - cancelled_at:.2f} seconds", flush=True)
                offset = log_path.stat().st_size
                connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=60)
                try:
                    connection.request("POST", "/v1/chat/completions", json.dumps({
                        "model": "test", "messages": [{"role": "user", "content": "Count upwards to 10000."}],
                        "stream": True, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False},
                    }), {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    if response.status != 200:
                        raise RuntimeError(response.read().decode())
                    while True:
                        line = response.readline()
                        if not line:
                            raise RuntimeError("The stream ended before decode cancellation")
                        if line.startswith(b"data: {"):
                            event = json.loads(line[6:])
                            if any(choice.get("delta", {}).get("content") for choice in event.get("choices", [])):
                                break
                    response.close()
                finally:
                    connection.close()
                cancelled_at = time.monotonic()
                deadline = cancelled_at + 15
                while ", cancelled" not in log_path.read_text()[offset:]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Decode did not stop within 15 seconds: " + log_path.read_text()[offset:])
                    time.sleep(0.1)
                print(f"PASS decode stopped in {time.monotonic() - cancelled_at:.2f} seconds", flush=True)
                request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                                 data=json.dumps({"model": "test", "max_tokens": 1,
                                                     "messages": [{"role": "user", "content": "Say hello."}]}).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    if not json.load(response).get("choices"):
                        raise RuntimeError("The server did not answer after cancellation")
                print("PASS server answers after cancellation", flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
