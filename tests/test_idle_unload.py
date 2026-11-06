#!/usr/bin/env python3
"""kapitsa serve --idle-unload (#373, issue #1020): flag checks (CPU), then on a Mac a small model is released
while idle and loaded again by the next request.

  python3 tests/test_idle_unload.py [--model 4b] [--no-gpu]

The GPU part takes the GPU lock (unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1 says the caller holds it) and runs:
  1. --idle-unload 0.5: released after 30 s idle whatever the pressure, loaded again by the next request.
  2. the default `auto` gate: NI_RESIDENCY_PRESSURE_TEST ("normal" / "warn" in a file) stands in for macOS's pressure
     and NI_IDLE_UNLOAD_AUTO_S=20 shortens its 10 minutes. Idle past 20 s at "normal": stays loaded. The file says
     "warn": released by the next check (15 s). Back to "normal" and a request: loaded again.
  The --no-gpu run (CPU, CI) checks only the flag.
"""
import json
import fcntl
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


def chat_status():
    """(status, Retry-After header) of one chat request."""
    import urllib.error
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8,
                       "temperature": 0}).encode()
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            resp.read()
            return resp.status, None
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Retry-After")


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


def gpu_lock():
    if os.environ.get("NACE_GPU_LOCK_HELD") == "1" or os.environ.get("NI_LOCK_HELD") == "1":
        return None
    path = os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock"
    f = open(path, "a")
    print(f"waiting for the GPU lock {path} ...", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def flags():
    for bad in ("soon", "-3", "0.1", "20000", "nan", "inf"):
        r = subprocess.run([BINARY, "serve", "--model", "nonexistent-model-373", "--idle-unload", bad],
                           capture_output=True, text=True, timeout=60)
        check(f"--idle-unload {bad} is refused", r.returncode != 0 and "--idle-unload takes" in r.stderr + r.stdout,
              (r.stdout + r.stderr)[-200:])
    r = subprocess.run([BINARY, "serve", "--help"], capture_output=True, text=True, timeout=60)
    check("--help names --idle-unload", "--idle-unload" in r.stdout + r.stderr)


def start(model, tmp, extra, env_extra=None):
    log = os.path.join(tmp, "serve.log")
    env = dict(os.environ, KAPITSA_HOME=tmp, **(env_extra or {}))
    proc = subprocess.Popen([BINARY, "serve", "--model", model, "--port", str(PORT), "--slots", "1", "--context", "4k",
                             "--disk-cache", "0"] + extra, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
    t = time.time()
    while time.time() - t < 180:
        try:
            models()
            break
        except Exception:
            time.sleep(1)
    return proc, log


def stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except Exception:
        proc.kill()


def count(log, text):
    return open(log).read().count(text)


def gpu(model):
    # 1. the unconditional form
    tmp = tempfile.mkdtemp()
    proc, log = start(model, tmp, ["--idle-unload", "0.5"])
    try:
        out, warm = chat()
        check("a request is answered", out["choices"][0]["message"]["content"] is not None)
        check("released after the idle time", wait_log(log, "released ", 90), open(log).read()[-400:])
        m = models()
        check("/v1/models still lists the model while released", m["active"] and m["context_length"] > 0 and m["slots"] == 1)
        out, back = chat()
        check("the next request loads it again and is answered", log_has(log, "again in") and out["choices"][0]["message"] is not None)
        print(f"first request after release {back:.1f} s, warm {warm:.1f} s")
        n = count(log, "again in")
        chat()
        check("the following request does not load again", count(log, "again in") == n)
        # two requests at once after a release: one loads the model, the other is answered or told to retry (503 +
        # Retry-After) at once; /v1/models never waits on the load
        check("released again", wait_log(log, "released ", 90) and count(log, "released ") == 2)
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(3) as ex:
            f1, f2 = ex.submit(chat_status), ex.submit(chat_status)
            t = time.time()
            models()
            listed = time.time() - t
            r1, r2 = f1.result(), f2.result()
        check("two requests at once: one reload, each answered or 503 + Retry-After",
              count(log, "again in") == n + 1 and all(r[0] == 200 or (r[0] == 503 and r[1]) for r in (r1, r2))
              and 200 in (r1[0], r2[0]), (r1, r2, count(log, "again in")))
        check("/v1/models answered at once while it loaded", listed < 5, f"{listed:.1f} s")
    finally:
        stop(proc)

    # 2. the default `auto` gate: 20 s idle needed, and only while the (stand-in) pressure is not normal
    tmp = tempfile.mkdtemp()
    pfile = os.path.join(tmp, "pressure")
    open(pfile, "w").write("normal")
    proc, log = start(model, tmp, [], {"NI_RESIDENCY_PRESSURE_TEST": pfile, "NI_IDLE_UNLOAD_AUTO_S": "20"})
    try:
        chat()
        time.sleep(50)  # past the 20 s and three of the watcher's 15 s checks
        check("auto: idle at normal pressure stays loaded", count(log, "released ") == 0, open(log).read()[-300:])
        open(pfile, "w").write("warn")
        check("auto: released once the pressure is not normal", wait_log(log, "released ", 40), open(log).read()[-300:])
        open(pfile, "w").write("normal")
        out, back = chat()
        check("auto: the next request loads it again", count(log, "again in") == 1 and out["choices"][0]["message"] is not None)
        time.sleep(20)  # the load just ended; a new idle period starts at its end
        check("auto: the request restarts the idle time", count(log, "released ") == 1)
    finally:
        stop(proc)


if __name__ == "__main__":
    flags()
    if "--no-gpu" not in sys.argv:
        lock = gpu_lock()
        gpu(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "4b")
    print("test_idle_unload: ok")
