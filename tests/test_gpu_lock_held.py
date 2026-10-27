#!/usr/bin/env python3
"""The Python side of "the caller already holds the GPU lock": NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1.

usage: test_gpu_lock_held.py

No model, no Metal, no GPU: every lock here is a temporary file ($BENCH_GPU_LOCK points at it), never /tmp/kapitsa-gpu.lock.

1. gpu_hogs.held_by_caller() reads both names with the rule of env_on() in src/util/log.h: unset, empty and "0" are
   off, anything else is on (so "00", "false" and " " are on).
2. A lock this process holds really blocks a second taker (GpuLock times out), and a caller that says it holds the
   lock is not made to take it again: each name, in a child process, skips the acquire and returns at once.
3. tools/qwen4_exp_pack.py's own GpuLock.take() does the same on a held temp lock (needs numpy; skipped without).
4. compare_engines.engine_env(): the environment of a launched engine drops both names, whatever the caller or the
   config's own env says.
5. The scripts that gate on the lock go through held_by_caller(), not their own spelling of the variable.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
HARNESS = os.path.join(REPO, "bench", "harness")
sys.path.insert(0, HARNESS)
import gpu_hogs  # noqa: E402

FAILS = []
NAMES = ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD")


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items() if k not in NAMES}
    env.update(extra)
    return env


# 1. the predicate
saved = {n: os.environ.pop(n, None) for n in NAMES}
try:
    check(not gpu_hogs.held_by_caller(), "neither name set: not held")
    for name in NAMES:
        for value, want in (("1", True), ("yes", True), ("00", True), ("false", True), (" ", True), ("", False), ("0", False)):
            os.environ[name] = value
            check(gpu_hogs.held_by_caller() == want, f"{name}={value!r} -> {want}", str(gpu_hogs.held_by_caller()))
            del os.environ[name]
    os.environ["NI_LOCK_HELD"], os.environ["NACE_GPU_LOCK_HELD"] = "0", "1"
    check(gpu_hogs.held_by_caller(), "one name off, the other on: held")
    os.environ["NI_LOCK_HELD"], os.environ["NACE_GPU_LOCK_HELD"] = "1", "0"
    check(gpu_hogs.held_by_caller(), "the other way round: held")
    os.environ["NI_LOCK_HELD"], os.environ["NACE_GPU_LOCK_HELD"] = "0", ""
    check(not gpu_hogs.held_by_caller(), "both off (0 and empty): not held")
finally:
    for n in NAMES:
        os.environ.pop(n, None)
        if saved[n] is not None:
            os.environ[n] = saved[n]

with tempfile.TemporaryDirectory() as tmp:
    lock_path = os.path.join(tmp, "gpu.lock")
    quiet = {"log": lambda m: None}

    # 2. a held lock blocks a second taker; a caller that says it holds the lock does not take it
    with gpu_hogs.GpuLock(path=lock_path, **quiet):
        try:
            gpu_hogs.GpuLock(path=lock_path, timeout=1, **quiet).acquire()
            check(False, "a held lock blocks a second taker")
        except TimeoutError:
            check(True, "a held lock blocks a second taker")

        # what a gated script does: take the lock unless the caller holds it
        snippet = (
            "import os, sys; sys.path.insert(0, sys.argv[1]); import gpu_hogs\n"
            "if gpu_hogs.held_by_caller():\n"
            "    print('skipped'); sys.exit(0)\n"
            "gpu_hogs.GpuLock(path=os.environ['BENCH_GPU_LOCK'], timeout=1, log=lambda m: None).acquire()\n"
            "print('acquired')\n")

        def child(**extra):
            p = subprocess.run([sys.executable, "-c", snippet, HARNESS], env=clean_env(BENCH_GPU_LOCK=lock_path, **extra),
                               capture_output=True, text=True, timeout=30)
            return p.returncode, p.stdout.strip()

        rc, out = child()
        check(rc != 0 and "acquired" not in out, "held lock, no env: the child cannot take it", f"rc={rc} out={out!r}")
        for name in NAMES:
            rc, out = child(**{name: "1"})
            check(rc == 0 and out == "skipped", f"held lock, {name}=1: not taken again", f"rc={rc} out={out!r}")
            rc, out = child(**{name: "0"})
            check(rc != 0, f"held lock, {name}=0: still a taker (blocked)", f"rc={rc} out={out!r}")

        # 3. tools/qwen4_exp_pack.py takes the lock through its own GpuLock.take() (module imports numpy)
        try:
            import numpy  # noqa: F401
            have_numpy = True
        except ImportError:
            have_numpy = False
        if have_numpy:
            qsnippet = (
                "import os, signal, sys\n"
                "sys.path.insert(0, os.path.join(sys.argv[1], 'tools')); import qwen4_exp_pack as q\n"
                "q.GPU_LOCK = os.environ['BENCH_GPU_LOCK']\n"
                "signal.alarm(5)\n"
                "q.GpuLock.take()\n"
                "print('returned')\n")

            def qchild(**extra):
                p = subprocess.run([sys.executable, "-c", qsnippet, REPO], env=clean_env(BENCH_GPU_LOCK=lock_path, **extra),
                                   capture_output=True, text=True, timeout=30)
                return p.returncode, p.stdout.strip()

            rc, out = qchild()
            check(rc != 0 and "returned" not in out, "qwen4_exp_pack: held lock, no env: take() blocks", f"rc={rc}")
            for name in NAMES:
                rc, out = qchild(**{name: "1"})
                check(rc == 0 and out == "returned", f"qwen4_exp_pack: {name}=1 does not re-take", f"rc={rc} out={out!r}")
            rc, out = qchild(NI_LOCK_HELD="0")
            check(rc != 0, "qwen4_exp_pack: NI_LOCK_HELD=0 is not held", f"rc={rc}")
        else:
            print("SKIP qwen4_exp_pack.GpuLock.take() (numpy is not installed)")

# 4. the engine child env (compare_engines imports bench, stdlib only)
import compare_engines  # noqa: E402

os.environ["NI_LOCK_HELD"] = os.environ["NACE_GPU_LOCK_HELD"] = "1"
os.environ["KEEP_ME_X"], os.environ["DROP_ME_X"] = "a", "b"
try:
    env, cleared = compare_engines.engine_env({"env_clear_prefixes": ["DROP_ME_"]}, {"EXTRA_X": "1", "NI_LOCK_HELD": "1"})
    check("NI_LOCK_HELD" not in env and "NACE_GPU_LOCK_HELD" not in env, "engine env drops both held names", str(sorted(env.keys() & set(NAMES))))
    check(env.get("KEEP_ME_X") == "a" and env.get("EXTRA_X") == "1" and "DROP_ME_X" not in env and cleared == ["DROP_ME_X"],
          "engine env keeps the rest, applies the config's env and clear prefixes", f"cleared={cleared}")
    check(os.environ["NI_LOCK_HELD"] == "1" and os.environ["NACE_GPU_LOCK_HELD"] == "1", "our own env is left alone")
finally:
    for n in NAMES + ("KEEP_ME_X", "DROP_ME_X"):
        os.environ.pop(n, None)

# 5. no script keeps its own spelling of the variable
SITES = ("bench/harness/bench.py", "bench/quality/eval_mlx.py", "bench/quality/eval_stream.py",
         "bench/quality/speed_variants.py", "bench/quality/speed_mlx.py", "bench/harness/agent_replay.py",
         "bench/harness/compare_engines.py", "bench/reasoning/reasoning.py", "tools/qwen4_exp_pack.py")
own = re.compile(r"""environ(\.get\(|\[)\s*["'](NI_LOCK_HELD|NACE_GPU_LOCK_HELD)["']|getenv\(\s*["'](NI_LOCK_HELD|NACE_GPU_LOCK_HELD)["']""")
for rel in SITES:
    with open(os.path.join(REPO, rel)) as f:
        text = f.read()
    bad = [i + 1 for i, line in enumerate(text.splitlines()) if own.search(line) and "environ[" not in line.split("=")[0]]
    check("held_by_caller()" in text and not bad, f"{rel} uses held_by_caller()", f"own reads on lines {bad}")

if FAILS:
    print(f"\n{len(FAILS)} FAILED")
    sys.exit(1)
print("\nall passed")
