#!/usr/bin/env python3
"""Pins gpu.py's reading of a probe: who holds the lock, who waits, and when the GPU is expected free (stdlib only)."""
import importlib.util, os, subprocess, sys

spec = importlib.util.spec_from_file_location("gpu", os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu.py"))
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)
NOW = 1_000_000
fails = 0


def check(name, got, want):
    global fails
    ok = got == want
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got!r}" + ("" if ok else f" (want {want!r})"))


def run(lines):
    d = gpu.parse("\n".join([f"now {NOW}", "chip Apple M5 Pro", "macos 26.5.2"] + lines))
    return d, gpu.estimate(d)


# lockf 100 holds (10 min in, declared 30 min), its kapitsa child 101 inherits the fd, lockf 200 waits with a 5 min ETA,
# logit_dump 300 waits in flock (holds an fd, no ETA)
d, e = run([
    "open 100 1 10:00 lockf -k /tmp/kapitsa-gpu.lock kapitsa serve",
    "open 101 100 09:59 kapitsa serve --model m",
    "open 300 1 01:00 logit_dump m",
    "lockf 100 1 10:00 lockf -k /tmp/kapitsa-gpu.lock kapitsa serve",
    "lockf 200 1 02:00 lockf -k /tmp/kapitsa-gpu.lock bash leg.sh",
    f"eta 100 {NOW - 900} 30 serve 35B",
    f"held 100 {NOW - 600}",
    f"eta 200 {NOW - 120} 5 leg B",
])
check("holder is the lockf, not its child", e["holder"]["pid"], 100)
check("holder ETA left", round(e["holder"]["left_min"]), 20)
check("waiters oldest first, child excluded", [w["pid"] for w in e["waiters"]], [200, 300])
check("waiter without ETA gets the default", e["waiters"][1]["known"], False)
check("free in = 20 left + waiter's full 5 + default", round(e["wait_min"]), 25 + round(gpu.DEFAULT_ETA))

# gpu run -- bash leg.sh: lockf 500 holds, bash 501 and its kapitsa grandchild 502 inherit the fd; lockf 600 waits
d, e = run([
    "open 500 1 03:00 lockf -k L bash leg.sh", "anc 500",
    "open 501 500 03:00 bash leg.sh", "anc 501 500",
    "open 502 501 02:50 kapitsa serve --model m", "anc 502 501 500",
    "lockf 500 1 03:00 lockf -k L bash leg.sh", "lockf 600 1 01:00 lockf -k L bash other.sh",
])
check("holder's grandchild is not a waiter", (e["holder"]["pid"], [w["pid"] for w in e["waiters"]]), (500, [600]))

# the holder runs under detach.py job 499 (an ancestor): named, with its log's last line; swap is read in GB
d, e = run(["open 500 499 03:00 lockf -k L bash leg.sh", "anc 500 499", "lockf 500 499 03:00 lockf -k L bash leg.sh",
            "job 499 df2-leg", "jobtail 499 step 3/8 ok", "job 777 other", "swap total = 7168.00M  used = 5969.00M  free = 1199.00M"])
check("holder's detach job and log tail", (e["holder"]["job"], e["holder"]["log_tail"]), ("df2-leg", "step 3/8 ok"))
check("holder's start time", e["holder"]["start"], NOW - 180)
check("swap used", round(d["swap_gb"], 2), 5.83)

# run is local only: no --host, and a bad --cwd stops before any queue entry
r = subprocess.run([sys.executable, gpu.__file__, "run", "--host", "m3pro", "--eta", "1", "--", "true"], capture_output=True, text=True)
check("run --host is refused", (r.returncode, "unrecognized arguments: --host" in r.stderr), (2, True))
r = subprocess.run([sys.executable, gpu.__file__, "run", "--eta", "1", "--cwd", "/nonexistent-vk", "--", "true"],
                   capture_output=True, text=True, env={**os.environ, "BENCH_GPU_LOCK": "/tmp/vk-test-never.lock"})
check("bad --cwd exits before queueing", (r.returncode, "--cwd /nonexistent-vk" in r.stderr), (1, True))
check("queue script has no remote or cd step", gpu.queue_script(5, "x", ["true"], "/tmp/L").startswith("mkdir -p"), True)

# past its ETA: assumed OVERDUE more minutes, marked overdue
d, e = run(["open 100 1 1-02:00:00 lockf -k L sleep", "lockf 100 1 1-02:00:00 lockf -k L sleep", f"eta 100 {NOW - 7200} 30 x"])
check("day-long etime parses", e["holder"]["age_s"], 93600)
check("overdue holder", (e["holder"]["known"], e["holder"]["left_min"]), ("overdue", gpu.OVERDUE))

# free lock, but an 18 GB kapitsa process runs outside it: not counted as free
d, e = run(["big 18874368 555 30:00 ./build/kapitsa serve --model x", "pagesize 16384", "free 65536"])
check("model outside the lock is not free", (e["holder"], e["wait_min"] > 0, len(e["outside"])), (None, True, 1))
check("available GB from vm_stat", d["avail_gb"], 1.0)
d, e = run(["big 3500000 556 10:00 python3 bench/harness/compare_engines.py --runs 3"])
check("a big python process is not a loaded model", (e["wait_min"], len(e["outside"])), (0, 0))

# a flock-only holder (kapitsa bench, no lockf): the oldest fd holds
d, e = run(["open 400 1 05:00 kapitsa bench", "open 401 1 00:30 logit_dump"])
check("flock-only holder is the oldest", (e["holder"]["pid"], [w["pid"] for w in e["waiters"]]), (400, [401]))

check("model filter matches a floor stem", gpu.has_model({"models": ["qwen3.5-4b-4bit-mtp"]}, "qwen3.5-4b-4bit"), True)
check("model filter rejects a prefix that is not a stem", gpu.has_model({"models": ["qwen3.5-4b-4bitx"]}, "qwen3.5-4b-4bit"), False)
check("macOS compare", gpu.macos_tuple("26.5.2") >= gpu.macos_tuple("26"), True)

print(f"{'all passed' if not fails else f'{fails} failed'}")
sys.exit(1 if fails else 0)
