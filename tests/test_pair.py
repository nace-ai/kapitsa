#!/usr/bin/env python3
"""Two Macs that run Flash-Next pair on their own (src/server/pair.h), on this Mac: two `kapitsa serve` of the same
synthetic Flash-Next pack find each other over Bonjour, the second becomes the stage of the first one's split, and both
answer as a whole server does; when either goes away, the other runs the model alone again. NI_CLUSTER_PAIR=1 lets the
two pair although this Mac holds the pack whole, and shortens the waits. Needs the GPU, the local network (Bonjour) and
a Qwen tokenizer (as tests/test_split.py): it takes the GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock)
unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1 says the caller holds it.
  1. A starts alone; B, started next, becomes A's stage, and A splits the model with it
  2. the same replies from a whole server (--private: no pairing), from A, and from B (which sends its requests to A)
  3. B killed: A runs the model alone again, with the same replies; B started again pairs again
  4. A killed: B, its head gone, starts again on its own, with the same replies
usage: test_pair.py <build dir> [--tokenizer QWEN_MODEL_DIR] [--keep]"""
import json, os, shutil, signal, subprocess, sys, tempfile, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_split as ts  # noqa: E402 (its argument parsing, lock, tokenizer search, chat and checks)

PORTS = {"whole": 8431, "a": 8432, "b": 8433}


class Proc(ts.Proc):
    """a kapitsa process with extra environment"""

    def __init__(self, tmp, name, args, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
        env.update({"KAPITSA_HOME": os.path.join(tmp, "home"), "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1"})
        env.update(extra or {})
        self.log_path = os.path.join(tmp, name + ".log")
        self.p = subprocess.Popen([ts.KAPITSA] + args, stdout=open(self.log_path, "a"), stderr=subprocess.STDOUT, env=env)


def up(port):
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except (urllib.error.URLError, OSError):
        return False


def cluster(port):
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/cluster", timeout=2))
    except (urllib.error.URLError, OSError, ValueError):
        return {}


def server(tmp, name, private=False):
    args = ["serve", "--model", ts.PACK, "--port", str(PORTS[name]), "--ctx", "16384", "--disk-cache", "0"]
    s = Proc(tmp, name, args + (["--private"] if private else []), None if private else {"NI_CLUSTER_PAIR": "1"})
    s.wait_for(lambda: up(PORTS[name]), name)
    return s


def until(ok, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if ok():
            return True
        time.sleep(0.5)
    return False


def same(label, port):
    for name, msgs in ts.CONVS.items():
        a, b = ts.chat(PORTS["whole"], msgs), ts.chat(port, msgs)
        ts.check(a == b, f"{label}: {name}", f"whole {a[:120]!r}\n     here  {b[:120]!r}")


def main():
    lock = ts.gpu_lock()
    tmp = tempfile.mkdtemp(prefix="kapitsa-pair-")
    ts.PACK = os.path.join(tmp, "flash-next-synthetic")
    procs = []
    try:
        tok = ts.TOKENIZER or ts.find_tokenizer()
        subprocess.run([sys.executable, os.path.join(ts.ROOT, "tools", "fn_synth.py"), ts.PACK, "--layers", "8", "--seed",
                        "11", "--tokenizer", tok], check=True, stdout=subprocess.DEVNULL)
        procs.append(server(tmp, "whole", private=True))
        # 1. A alone, then B its stage
        a = server(tmp, "a")
        procs.append(a)
        ts.check(not cluster(PORTS["a"]).get("whole", True), "A alone counts as streamed (NI_CLUSTER_PAIR)")
        b = server(tmp, "b")
        procs.append(b)
        bi = cluster(PORTS["b"])
        ts.check(bi.get("stage_port", 0) > 0 and bi.get("stage_of") == cluster(PORTS["a"]).get("id"),
                 "B started next: the stage of A", bi)
        ts.check(until(lambda: cluster(PORTS["a"]).get("whole")), "A splits the model with B",
                 a.log()[-1500:])
        # 2. the same replies everywhere
        same("A, split with B", PORTS["a"])
        same("B, its requests sent to A", PORTS["b"])
        # 3. B goes away: A alone again; B back: paired again
        b.stop(signal.SIGKILL)
        ts.check(until(lambda: cluster(PORTS["a"]).get("whole") is False), "B killed: A runs the model alone again",
                 a.log()[-1500:])
        same("A alone again", PORTS["a"])
        b = server(tmp, "b")
        procs.append(b)
        ts.check(until(lambda: cluster(PORTS["a"]).get("whole")), "B started again: A splits with it again",
                 a.log()[-1500:])
        same("A, split again", PORTS["a"])
        # 4. A goes away: B starts again on its own
        a.stop(signal.SIGKILL)
        ts.check(until(lambda: cluster(PORTS["b"]).get("stage_port", 0) == 0 and up(PORTS["b"])
                       and cluster(PORTS["b"]).get("model")),
                 "A killed: B starts again on its own", b.log()[-1500:])
        same("B alone", PORTS["b"])
    finally:
        for p in procs:
            p.stop()
        if ts.KEEP:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        if lock:
            lock.close()
    print("FAIL" if ts.fails else "PASS", f"({ts.fails} failed)")
    sys.exit(1 if ts.fails else 0)


if __name__ == "__main__":
    main()
