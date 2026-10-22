#!/usr/bin/env python3
"""GPU contention helper and GPU mutual-exclusion lock for macOS (no sudo). Stdlib only.

Contention is measured by diffing each process's accumulatedGPUTime in ioreg (AGXDeviceUserClient).

  gpu_hogs.py [secs]                         -> print processes using the GPU now
  gpu_hogs.py wait [maxpct] [quiet_s] [timeout_s] [rootpid ...]
                                             -> block until no *foreign* process uses > maxpct GPU
                                                (default 5%) for quiet_s seconds (default 15); the
                                                process trees of the given root pids are not foreign
  gpu_hogs.py monitor <outfile> <rootpid> [rootpid ...]
                                             -> every 2 s append "epoch total_foreign_pct top_name";
                                                foreign = everything except WindowServer, the ambient
                                                processes below and the whole process tree of the root
                                                pids, recomputed on every sample (Ollama's GPU work runs
                                                in a child)

Library use (compare_engines.py, bench.py):
  GpuLock         exclusive flock on the GPU lock file, the same lock `lockf -k <file> <cmd>` takes
                  (BSD flock(2)); every timed benchmark phase must hold it
  proc_tree       pid set of one or more process trees (one `ps` call)
  ContentionMonitor, wait_quiet, summarize_samples

Environment:
  BENCH_GPU_LOCK     lock file (default /tmp/kapitsa-gpu.lock). Every GPU job on the Mac must use the same file,
                     e.g. `lockf -k /tmp/kapitsa-gpu.lock <cmd>` for anything that is not this harness.
  BENCH_GPU_AMBIENT  comma-separated process names to treat as ambient desktop load (added to AMBIENT below):
                     never blocks the quiet wait, never flags a result, recorded separately as "ambient".
"""
import fcntl, os, re, subprocess, sys, threading, time

IGNORE = {"WindowServer"}
# Ambient desktop GPU load that no benchmark can switch off and that is the same for every engine: macOS's photo and
# media analysis daemons (busy for hours after an OS update). Add the app you drive the benchmark from (a terminal or
# a desktop app whose window redraws while it runs) with BENCH_GPU_AMBIENT. The monitor records ambient load
# separately ("ambient_pct"), so the numbers stay transparent.
AMBIENT_DEFAULT = ("mediaanalysisd", "photoanalysisd", "VisualIntelligenceCore")
AMBIENT = set(AMBIENT_DEFAULT)
LOCK_PATH = os.path.expanduser("/tmp/kapitsa-gpu.lock")
FOREIGN_MAX_PCT = 5.0     # foreign GPU load above this, sustained, flags a result
SUSTAIN_SAMPLES = 3       # consecutive samples (2 s each) that count as "sustained"
COMM_LEN = 16             # ioreg reports the process name truncated to MAXCOMLEN (16) characters


def configure():
    """(Re)reads BENCH_GPU_LOCK and BENCH_GPU_AMBIENT, e.g. after an env file was loaded."""
    global AMBIENT, LOCK_PATH
    AMBIENT = set(AMBIENT_DEFAULT) | {x.strip() for x in os.environ.get("BENCH_GPU_AMBIENT", "").split(",") if x.strip()}
    LOCK_PATH = os.path.expanduser(os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock")


configure()


def _comm(name):
    return name[:COMM_LEN]


def is_ambient(name):
    return _comm(name) in {_comm(a) for a in AMBIENT}


def is_ignored(name):
    return _comm(name) in {_comm(a) for a in IGNORE}


# ---------------------------------------------------------------- process trees
def proc_tree(roots):
    """Set of pids in the process trees rooted at `roots` (roots included even if already gone)."""
    roots = [int(r) for r in roots if r]
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return set(roots)
    kids = {}
    for line in out.split("\n"):
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    res, st = set(), list(roots)
    while st:
        x = st.pop()
        if x in res:
            continue
        res.add(x)
        st += kids.get(x, [])
    return res


# ---------------------------------------------------------------- GPU time per process
def snap():
    out = subprocess.run(["ioreg", "-r", "-c", "AGXDeviceUserClient", "-l", "-w", "0"],
                         capture_output=True, text=True).stdout
    res = {}
    for block in out.split("+-o AGXDeviceUserClient")[1:]:
        m = re.search(r'"IOUserClientCreator" = "pid (\d+), ([^"]*)"', block)
        if not m:
            continue
        t = sum(int(x) for x in re.findall(r'"accumulatedGPUTime"=(\d+)', block))
        k = (int(m.group(1)), m.group(2))
        res[k] = res.get(k, 0) + t
    return res


def diff_rows(a, b, iv, exclude=()):
    rows = []
    for k in b:
        pct = (b[k] - a.get(k, 0)) / (iv * 1e9) * 100
        if pct > 0.5 and k[0] not in exclude and not is_ignored(k[1]) and not is_ambient(k[1]):
            rows.append((pct, k))
    return sorted(rows, reverse=True)


def usage(iv, exclude=(), exclude_roots=None):
    """Foreign GPU users over the next `iv` seconds: [(pct, (pid, name))], highest first.
    exclude_roots: pids whose whole process tree (computed at the end of the interval) is not foreign."""
    a = snap()
    time.sleep(iv)
    b = snap()
    excl = set(exclude)
    if exclude_roots:
        excl |= proc_tree(exclude_roots() if callable(exclude_roots) else exclude_roots)
    return diff_rows(a, b, iv, excl)


def wait_quiet(maxpct=FOREIGN_MAX_PCT, quiet_s=15, timeout_s=3600, exclude_roots=None, log=None, iv=2.0, own_max=None):
    """Block until no foreign process uses more than maxpct GPU for quiet_s seconds. With own_max,
    the excluded trees (the server under test) must also stay at or below own_max in total, so
    background work left over from the warm-up does not run into the timed phases.
    Returns (ok, waited_s, last_busy_rows)."""
    t0 = time.time()
    ok_since = None
    last_busy = []
    last_log = t0
    while True:
        if own_max is None:
            rows = usage(iv, exclude_roots=exclude_roots)
            busy = [r for r in rows if r[0] > maxpct]
        else:
            a = snap(); time.sleep(iv); b = snap()
            own = proc_tree(exclude_roots() if callable(exclude_roots) else (exclude_roots or []))
            rows = diff_rows(a, b, iv)
            busy = [r for r in rows if r[0] > maxpct and r[1][0] not in own]
            own_pct = sum(r[0] for r in rows if r[1][0] in own)
            if own_pct > own_max:
                busy.append((own_pct, (0, "server under test (still busy after warm-up)")))
        now = time.time()
        if busy:
            ok_since = None
            last_busy = busy
        elif ok_since is None:
            ok_since = now
        if ok_since and now - ok_since >= quiet_s:
            return True, now - t0, []
        if now - t0 > timeout_s:
            return False, now - t0, last_busy[:3]
        if log and busy and now - last_log > 60:
            log(f"gpu busy ({now - t0:.0f}s): " + ", ".join(f"{n}[{p}] {pct:.0f}%" for pct, (p, n) in busy[:3]))
            last_log = now


class ContentionMonitor(threading.Thread):
    """Samples foreign GPU load every `interval` s until stopped. The server's process tree is
    recomputed on every sample from roots() (a callable or a list), so children started later
    (Ollama's runner) and daemonized listeners are never counted as foreign."""

    def __init__(self, roots, interval=2.0):
        super().__init__(daemon=True)
        self.roots = roots if callable(roots) else (lambda: list(roots))
        self.interval = interval
        self.samples = []  # [epoch_end, total_foreign_pct, top_name, top_pct, ambient_pct]
        self.stop_ev = threading.Event()

    def run(self):
        a = snap()
        ta = time.time()
        while not self.stop_ev.wait(self.interval):
            try:
                b = snap()
                tb = time.time()
                rows = diff_rows(a, b, tb - ta, proc_tree(self.roots()))
                tot = sum(r[0] for r in rows)
                top = rows[0] if rows else (0.0, (0, "-"))
                # a process that closes one of its GPU clients loses that client's time: clamp, or the sample goes negative
                amb = sum(max(0, b[k] - a.get(k, 0)) / ((tb - ta) * 1e9) * 100 for k in b if is_ambient(k[1]))
                self.samples.append([round(tb, 3), round(tot, 2), top[1][1], round(top[0], 2), round(amb, 2)])
                a, ta = b, tb
            except Exception:
                pass

    def stop(self):
        self.stop_ev.set()
        self.join(timeout=10)


def summarize_samples(samples, t0=None, t1=None, maxpct=FOREIGN_MAX_PCT, sustain=SUSTAIN_SAMPLES):
    """Contention summary for the samples whose end time lies in [t0, t1]."""
    s = [x for x in samples if (t0 is None or x[0] >= t0) and (t1 is None or x[0] <= t1 + 2.5)]
    if not s:
        return {"samples": 0, "flagged": False}
    v = [x[1] for x in s]
    run = best = 0
    for x in v:
        run = run + 1 if x > maxpct else 0
        best = max(best, run)
    mean = sum(v) / len(v)
    top = {}
    for x in s:
        if x[1] > maxpct:
            top[x[2]] = top.get(x[2], 0) + 1
    amb = [x[4] for x in s if len(x) > 4]
    return {"samples": len(s), "mean_foreign_gpu_pct": round(mean, 2), "max_foreign_gpu_pct": round(max(v), 2),
            "mean_ambient_gpu_pct": round(sum(amb) / len(amb), 2) if amb else None,
            "busy_samples": sum(1 for x in v if x > maxpct), "longest_busy_run": best,
            "top": sorted(top, key=lambda k: -top[k])[:5],
            "flagged": mean > maxpct or best >= sustain}


# ---------------------------------------------------------------- GPU lock
def lock_holders(path=None):
    path = path or LOCK_PATH
    try:
        pids = subprocess.run(["lsof", "-t", path], capture_output=True, text=True, timeout=10).stdout.split()
        if not pids:
            return "unknown holder"
        out = subprocess.run(["ps", "-o", "pid=,command=", "-p", ",".join(pids)], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return "; ".join(l.strip()[:160] for l in out.split("\n") if l.strip())
    except Exception as e:
        return f"holder lookup failed: {e}"


class GpuLock:
    """Exclusive BSD flock on the GPU lock file (interoperates with `lockf -k <file> <cmd>`).
    The file is never unlinked: removing it would let two holders lock different inodes.
    The descriptor is not inherited by child processes (PEP 446), so servers started while the
    lock is held do not keep it after the holder releases it."""

    def __init__(self, path=None, timeout=None, log=print):
        self.path, self.timeout, self.log = path or LOCK_PATH, timeout, log
        self.fd = None
        self.wait_s = None
        self.t_acquired = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.fd = os.open(self.path, os.O_RDONLY | os.O_CREAT, 0o644)
        t0 = time.time()
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.log(f"GPU lock {self.path} is held by: {lock_holders(self.path)}; waiting...")
            if self.timeout is None:
                # Block in flock() like `lockf` does: blocked waiters are queued by the kernel, so a
                # polling loop (LOCK_NB + sleep) would lose every release to them and starve.
                done = threading.Event()

                def report():
                    while not done.wait(300):
                        self.log(f"still waiting for the GPU lock ({time.time() - t0:.0f}s): {lock_holders(self.path)}")
                threading.Thread(target=report, daemon=True).start()
                try:
                    fcntl.flock(self.fd, fcntl.LOCK_EX)
                finally:
                    done.set()
            else:
                last = t0
                while True:
                    try:
                        fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        now = time.time()
                        if now - t0 > self.timeout:
                            os.close(self.fd)
                            self.fd = None
                            raise TimeoutError(f"GPU lock not acquired within {self.timeout:.0f}s")
                        if now - last > 300:
                            self.log(f"still waiting for the GPU lock ({now - t0:.0f}s): {lock_holders(self.path)}")
                            last = now
                        time.sleep(0.5)
        self.t_acquired = time.time()
        self.wait_s = self.t_acquired - t0
        self.log(f"GPU lock acquired after {self.wait_s:.1f}s wait")
        return self

    def release(self):
        if self.fd is None:
            return 0.0
        held = time.time() - self.t_acquired
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = None
        self.log(f"GPU lock released after {held:.1f}s held")
        return held

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()


# ---------------------------------------------------------------- CLI
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "3"
    if mode == "wait":
        mx = float(sys.argv[2]) if len(sys.argv) > 2 else FOREIGN_MAX_PCT
        quiet = float(sys.argv[3]) if len(sys.argv) > 3 else 15
        to = float(sys.argv[4]) if len(sys.argv) > 4 else 3600
        roots = [int(x) for x in sys.argv[5:]]
        ok, waited, busy = wait_quiet(mx, quiet, to, exclude_roots=roots or None)
        if ok:
            print(f"gpu quiet after {waited:.0f}s", flush=True)
            return 0
        print(f"gpu still busy after {to:.0f}s: {busy}", flush=True)
        return 1
    elif mode == "monitor":
        out, roots = sys.argv[2], [int(x) for x in sys.argv[3:]]
        while True:
            rows = usage(2, exclude_roots=roots)
            tot = sum(r[0] for r in rows)
            top = rows[0] if rows else (0.0, (0, "-"))
            with open(out, "a") as f:
                f.write(f"{time.time():.0f} {tot:.1f} {top[1][1]}\n")
    else:
        for pct, (pid, name) in usage(float(mode)):
            print(f"{pct:6.1f}%  pid {pid} {name}")


if __name__ == "__main__":
    sys.exit(main() or 0)
