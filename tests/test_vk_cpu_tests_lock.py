#!/usr/bin/env python3
"""`vk cpu-tests` (.dsh/skills/verify-kapitsa/scripts/vk) runs `tests/test_server.py --cli`, which takes the GPU lock and
waits for it with no bound. With `vk serve` up that is until `vk stop`, so the test_server_cli entry fails at once when
another process holds the lock, the way `vk serve` does (lockf -t 0), and runs as before when the lock is free or the
caller says it holds it (NI_LOCK_HELD / NACE_GPU_LOCK_HELD).

No build, no model, no GPU: VK_ROOT is a throwaway repo that only has src/engine, `cmake` is a stub that succeeds, every
test of the list then fails for lack of a binary, and only the test_server_cli line is read. The lock is a temporary file
($BENCH_GPU_LOCK) held by a child process, never /tmp/kapitsa-gpu.lock. stdlib only; needs lsof, as vk does.

usage: test_vk_cpu_tests_lock.py
"""
import fcntl, os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
VK = os.path.normpath(os.path.join(HERE, "..", ".dsh", "skills", "verify-kapitsa", "scripts", "vk"))
FAILED = []
NAMES = ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD")
HOLDER = ("import fcntl, os, sys, time\n"
          "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_CREAT)\n"
          "fcntl.flock(fd, fcntl.LOCK_EX)\n"
          "print('held', flush=True)\n"
          "time.sleep(60)\n")


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


def run_vk(root, lock, evid, **extra):
    env = {k: v for k, v in os.environ.items() if k not in NAMES}
    env.update(VK_ROOT=root, VK_EVIDENCE=evid, VK_RUN="t", BENCH_GPU_LOCK=lock, PATH=os.path.join(root, "bin") + os.pathsep + env["PATH"], **extra)
    p = subprocess.run(["bash", VK, "cpu-tests"], env=env, capture_output=True, text=True, timeout=60)
    line = next((x for x in p.stdout.splitlines() if x.startswith(("PASS test_server_cli", "FAIL test_server_cli"))), "")
    log = os.path.join(evid, "t", subprocess.run(["git", "-C", root, "rev-parse", "--short=8", "HEAD"], capture_output=True,
                                                 text=True).stdout.strip(), "cpu-tests", "test_server_cli.log")
    return p, line, open(log).read() if os.path.exists(log) else ""


if shutil.which("lsof") is None:
    print("FAIL lsof is missing (vk needs it to see the lock holder)")
    sys.exit(1)

with tempfile.TemporaryDirectory() as d:
    root = os.path.join(d, "repo")
    os.makedirs(os.path.join(root, "src", "engine"))
    os.makedirs(os.path.join(root, "bin"))
    os.makedirs(os.path.join(root, "build"))
    open(os.path.join(root, "build", "CMakeCache.txt"), "w").close()  # vk does not configure again
    with open(os.path.join(root, "bin", "cmake"), "w") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(os.path.join(root, "bin", "cmake"), 0o755)
    git = ["git", "-C", root, "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "x"], check=True)
    evid = os.path.join(d, "evidence")
    lock = os.path.join(d, "gpu.lock")

    # free lock: the entry runs (and fails here only because the fake checkout has no tests/test_server.py)
    p, line, log = run_vk(root, lock, evid)
    check(line.startswith("FAIL test_server_cli") and "not run" not in line and "needs the GPU lock free" not in log,
          "a free lock: test_server_cli runs", f"{line!r} {log[:200]!r}")

    holder = subprocess.Popen([sys.executable, "-c", HOLDER, lock], stdout=subprocess.PIPE, text=True)
    try:
        check(holder.stdout.readline().strip() == "held", "the stand-in holder took the lock")

        # held by someone else: fails at once, names the holder, does not run the test, counts as a failure
        t0 = time.time()
        p, line, log = run_vk(root, lock, evid)
        took = time.time() - t0
        check("not run: GPU lock held" in line and "needs the GPU lock free" in log and f"pid {holder.pid}" in log,
              "a held lock: test_server_cli fails at once and names the holder", f"{line!r} {log[:300]!r}")
        check(took < 30 and p.returncode != 0 and "failed" in p.stdout.splitlines()[-1],
              "the run ends (nonzero) without waiting for the lock", f"{took:.1f}s rc {p.returncode}")

        # the caller holds it: no check, the entry runs
        for name in NAMES:
            p, line, log = run_vk(root, lock, evid, **{name: "1"})
            check("not run" not in line and "needs the GPU lock free" not in log, f"{name}=1: the caller's claim is believed",
                  f"{line!r}")
        p, line, log = run_vk(root, lock, evid, NI_LOCK_HELD="0")
        check("not run: GPU lock held" in line, "NI_LOCK_HELD=0 is not a claim to hold it", f"{line!r}")
    finally:
        holder.kill()
        holder.wait()

    # the holder is gone: free again
    p, line, log = run_vk(root, lock, evid)
    check("not run" not in line and "needs the GPU lock free" not in log, "after the holder exits: test_server_cli runs",
          f"{line!r}")

print("FAILED: %d" % len(FAILED) if FAILED else "ALL PASSED")
sys.exit(1 if FAILED else 0)
