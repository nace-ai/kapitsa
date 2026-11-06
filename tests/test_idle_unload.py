#!/usr/bin/env python3
"""kapitsa serve --idle-unload (#373, issue #1020): flag checks (CPU), then on a Mac a small model is released
while idle and loaded again by the next request.

  python3 tests/test_idle_unload.py [--model 4b] [--no-gpu]

The GPU part runs the unconditional form (--idle-unload 0.5: 30 s idle, whatever the pressure); `auto` adds only the
Gpu::memory_pressed() check, which NI_RESIDENCY_PRESSURE_TEST ("normal" or "warn" in a file) drives.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

BINARY = os.path.abspath("build/kapitsa")
PORT = 8933
BASE = f"http://127.0.0.1:{PORT}"


def check(name, ok, extra=""):
    print(("PASS: " if ok else "FAIL: ") + name + (f" {extra}" if not ok and extra else ""))
    if not ok:
        sys.exit(1)


def chat():
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
                       "temperature": 0}).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(r, timeout=300) as resp:
        out = json.loads(resp.read())
    return out, time.time() - t


def models():
    with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as resp:
        return json.loads(resp.read())["data"][0]


def log_has(path, text):
    return text in open(path).read()


def wait_log(path, text, secs):
    t = time.time()
    while time.time() - t < secs:
        if log_has(path, text):
            return True
        time.sleep(1)
    return False


def flags():
    for bad in ("soon", "-3", "0.1", "20000"):
        r = subprocess.run([BINARY, "serve", "--model", "nonexistent-model-373", "--idle-unload", bad],
                           capture_output=True, text=True, timeout=60)
        check(f"--idle-unload {bad} is refused", r.returncode != 0 and "--idle-unload takes" in r.stderr + r.stdout,
              (r.stdout + r.stderr)[-200:])
    r = subprocess.run([BINARY, "serve", "--help"], capture_output=True, text=True, timeout=60)
    check("--help names --idle-unload", "--idle-unload" in r.stdout + r.stderr)


def gpu(model):
    tmp = tempfile.mkdtemp()
    log = os.path.join(tmp, "serve.log")
    env = dict(os.environ, KAPITSA_HOME=tmp)
    # the unconditional form: 30 s idle, whatever the pressure
    proc = subprocess.Popen([BINARY, "serve", "--model", model, "--port", str(PORT), "--slots", "1", "--context", "4k",
                             "--disk-cache", "0", "--idle-unload", "0.5"], stdout=open(log, "w"),
                            stderr=subprocess.STDOUT, env=env)
    try:
        t = time.time()
        while time.time() - t < 180:
            try:
                models()
                break
            except Exception:
                time.sleep(1)
        out, warm = chat()
        check("a request is answered", bool(out["choices"][0]["message"]["content"] is not None))
        check("released after the idle time", wait_log(log, "released ", 90), open(log).read()[-400:])
        m = models()
        check("/v1/models still lists the model while released", m["active"] and m["context_length"] > 0 and m["slots"] == 1)
        out, back = chat()
        check("the next request loads it again and is answered", log_has(log, "again in") and out["choices"][0]["message"] is not None)
        print(f"first request after release {back:.1f} s, warm {warm:.1f} s")
        # a second request does not load again
        n = open(log).read().count("again in")
        chat()
        check("the following request does not load again", open(log).read().count("again in") == n)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    flags()
    if "--no-gpu" not in sys.argv:
        gpu(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "4b")
    print("test_idle_unload: ok")
