#!/usr/bin/env python3
"""tests/test_server.py --cli starts `kapitsa serve` on stand-in models, so it takes the shared GPU lock itself and waits for
it (AGENTS.md "GPU lock"): a raw run cannot overlap someone's timed GPU leg. Uses a throwaway lock file held by this
test, and the script's --lock-only switch (take the lock, say so, stop); no model, no GPU. stdlib only.

usage: test_server_lock.py
"""
import fcntl, os, subprocess, sys, tempfile, time

BOTH = "children see NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1"
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "test_server.py")
FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


def start(lock, **extra):
    env = {k: v for k, v in os.environ.items() if k not in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD")}
    env.update(BENCH_GPU_LOCK=lock, **extra)
    return subprocess.Popen([sys.executable, SCRIPT, "--cli", "/nonexistent/kapitsa", "--lock-only"], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def lines_within(p, seconds):
    """whether the child is still running after `seconds`"""
    t0 = time.time()
    while time.time() - t0 < seconds and p.poll() is None:
        time.sleep(0.05)
    return p.poll() is None


with tempfile.TemporaryDirectory() as d:
    lock = os.path.join(d, "gpu.lock")

    # free lock: taken at once
    p = start(lock)
    out, _ = p.communicate(timeout=20)
    check(p.returncode == 0 and "GPU lock held" in out and "waiting" not in out, "a free lock is taken without waiting",
          f"rc {p.returncode}, {out!r}")
    # the children of the run (every serve / doctor it starts) are told, or they take the lock again and block on it
    check(BOTH in out, "after the take, a child sees NI_LOCK_HELD=1 and NACE_GPU_LOCK_HELD=1", f"{out!r}")

    # held by someone else: says it waits, stays blocked, goes on when the holder lets go
    fd = os.open(lock, os.O_RDONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    p = start(lock)
    running = lines_within(p, 2.0)
    check(running, "a held lock keeps the run waiting", f"rc {p.poll()}")
    fcntl.flock(fd, fcntl.LOCK_UN)
    out, _ = p.communicate(timeout=20)
    check(out.splitlines()[:1] == [f"waiting for GPU lock ({lock})"] and "GPU lock held" in out and p.returncode == 0,
          "it prints one waiting line, then takes the lock when the holder releases", f"rc {p.returncode}, {out!r}")
    check(BOTH in out, "a lock taken after waiting tells the children too", f"{out!r}")

    # the caller says it holds the lock (lockf -k ... env NI_LOCK_HELD=1): no second take, even while the file is locked
    fcntl.flock(fd, fcntl.LOCK_EX)
    for name in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD"):
        p = start(lock, **{name: "1"})
        out, _ = p.communicate(timeout=20)
        check(p.returncode == 0 and "GPU lock not taken" in out, f"{name}=1 does not take the lock again", f"rc {p.returncode}, {out!r}")
        check(f"{name}=1" in out, f"{name}=1 reaches the children as it came", f"{out!r}")
    p = start(lock, NI_LOCK_HELD="0")
    time.sleep(1.0)
    check(p.poll() is None, "NI_LOCK_HELD=0 is not a claim to hold it", f"rc {p.poll()}")
    p.kill()
    p.communicate()
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)

    # without --cli there is no serve to protect: the base-URL mode takes no lock (nothing to start)
    fd = os.open(lock, os.O_RDONLY | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    env = dict(os.environ, BENCH_GPU_LOCK=lock)
    env.pop("NI_LOCK_HELD", None); env.pop("NACE_GPU_LOCK_HELD", None)
    p = subprocess.run([sys.executable, SCRIPT, "--lock-only"], env=env, capture_output=True, text=True, timeout=20)
    check(p.returncode == 0 and "GPU lock not taken" in p.stdout, "the base-URL mode takes no lock", f"rc {p.returncode}, {p.stdout!r}")
    os.close(fd)

print("FAILED: %d" % len(FAILED) if FAILED else "ALL PASSED")
sys.exit(1 if FAILED else 0)
