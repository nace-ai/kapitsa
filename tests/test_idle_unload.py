#!/usr/bin/env python3
"""kapitsa serve --idle-unload (#373, issue #1020): flag checks (CPU), then on a Mac a small model is released
while idle and loaded again by the next request.

  python3 tests/test_idle_unload.py [--model 4b] [--no-gpu]

The GPU part takes the GPU lock (unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1 says the caller holds it) and runs:
  1. --idle-unload 0.5 with the SSD prefix cache at its default (on): released after 30 s idle whatever the pressure,
     loaded again by the next request, the same (greedy) reply before and after.
  2. the default `auto` gate: NI_RESIDENCY_PRESSURE_TEST ("normal" / "warn" in a file) stands in for macOS's pressure
     and NI_IDLE_UNLOAD_AUTO_S=20 shortens its 10 minutes. Idle past 20 s at "normal": stays loaded. The file says
     "warn": released by the next check (15 s). Back to "normal" and a request: loaded again.
  3. a load made to last 6 s (NI_IDLE_UNLOAD_TEST_LOAD_MS, for this test only): /v1/models answers within 1 s, a second
     request gets 503 with a short Retry-After, the first is answered. Without the delay a small model reloads in
     about a second and these checks could not fail.
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


def text(out):
    return out["choices"][0]["message"]["content"]


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
    proc = subprocess.Popen([BINARY, "serve", "--model", model, "--port", str(PORT), "--slots", "1", "--context", "4k"] + extra, stdout=open(log, "w"), stderr=subprocess.STDOUT, env=env)
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
    # 1. the unconditional form, with the SSD prefix cache at its default (on): the release saves it under the lock
    tmp = tempfile.mkdtemp()
    proc, log = start(model, tmp, ["--idle-unload", "0.5"])
    try:
        out, warm = chat()
        before = text(out)
        check("a request is answered", before is not None)
        check("released after the idle time", wait_log(log, "released ", 90), open(log).read()[-400:])
        m = models()
        check("/v1/models still lists the model while released", m["active"] and m["context_length"] > 0 and m["slots"] == 1)
        out, back = chat()
        check("the next request loads it again and is answered", log_has(log, "again in") and text(out) is not None)
        check("the same reply before and after the release (greedy)", text(out) == before, (before, text(out)))
        print(f"first request after release {back:.1f} s, warm {warm:.1f} s")
        n = count(log, "again in")
        chat()
        check("the following request does not load again", count(log, "again in") == n)
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

    # 3. a load that lasts (NI_IDLE_UNLOAD_TEST_LOAD_MS delays it by 6 s, so the checks below can fail): while it runs,
    # /v1/models answers at once, a second request gets 503 + a short Retry-After, and the first is answered
    tmp = tempfile.mkdtemp()
    proc, log = start(model, tmp, ["--idle-unload", "0.5", "--disk-cache", "0"], {"NI_IDLE_UNLOAD_TEST_LOAD_MS": "6000"})
    try:
        chat()
        check("slow load: released", wait_log(log, "released ", 90))
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(2) as ex:
            first = ex.submit(chat)
            time.sleep(1.5)  # the first request is inside the load now
            t = time.time()
            m = models()
            listed = time.time() - t
            code, retry = chat_status()
            still_loading = count(log, "again in") == 0
            out, _ = first.result()
        check("slow load: the load was still running when /v1/models and the second request were answered", still_loading)
        check("slow load: /v1/models answered within 1 s during the load", listed < 1.0 and m["active"], f"{listed:.2f} s")
        check("slow load: a second request got 503 + Retry-After of a few seconds", code == 503 and retry and int(retry) <= 5, (code, retry))
        check("slow load: the first request was answered", text(out) is not None and count(log, "again in") == 1)
    finally:
        stop(proc)


if __name__ == "__main__":
    flags()
    if "--no-gpu" not in sys.argv:
        lock = gpu_lock()
        gpu(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "4b")
    print("test_idle_unload: ok")
