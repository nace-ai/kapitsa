#!/usr/bin/env python3
"""Flash-Next split across Macs by layers (src/cluster/pipe.h), on this Mac: the same requests to a whole server and to a
split one (`kapitsa serve --split` with `kapitsa stage` processes over 127.0.0.1) must give the same replies, and a
stage that goes away must end the request in flight with an error that says so. Needs the GPU and numpy: it takes the
GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless NACE_GPU_LOCK_HELD=1 or NI_LOCK_HELD=1 says the caller
holds it. The model is a synthetic Flash-Next pack (tools/fn_synth.py: 8 layers at hidden 512, random weights) with a
Qwen model's tokenizer, so its replies are random tokens, the same ones every time; greedy, thinking off.
  0. `kapitsa stage`'s own flags, before any model: a prefill chunk below one is a usage error, as bench's is
     (tests/test_bench.py), and a good one still parses and goes on to look for its model. No GPU, no tokenizer and no
     peer: it runs first and in a second, or alone with --flags-only
  1. one stage (layers 5-8): short prompts, a long one (two prefill chunks), a follow-up turn, a streamed reply, and two
     requests at once on the split server against each alone on the whole one
  2. two stages (layers 3-4 and 5-8): the ring back through the head's own port
  3. the stage killed in the middle of a reply: that request ends with an error naming the split, the next one at once
usage: test_split.py <build dir> [--tokenizer QWEN_MODEL_DIR] [--keep] [--flags-only]
  --tokenizer: a downloaded Qwen3.5-family folder whose tokenizer the pack takes (default: the first under
               ~/.cache/kapitsa/models with a tokenizer.json and a vocabulary of 248320)
  --keep: keep the temporary folder (the pack and the logs)
  --flags-only: run section 0 alone (the stage's own flags) and stop before the GPU lock, the tokenizer and the pack"""
import fcntl, json, os, shutil, signal, subprocess, sys, tempfile, threading, time, urllib.error, urllib.request

ARGS = sys.argv[1:]
KEEP = "--keep" in ARGS
if KEEP:
    ARGS.remove("--keep")
FLAGS_ONLY = "--flags-only" in ARGS
if FLAGS_ONLY:
    ARGS.remove("--flags-only")
TOKENIZER = ""
if "--tokenizer" in ARGS:
    i = ARGS.index("--tokenizer")
    TOKENIZER = ARGS[i + 1]
    del ARGS[i : i + 2]
if len(ARGS) != 1:
    sys.exit(__doc__)
BUILD = os.path.abspath(ARGS[0])
KAPITSA = os.path.join(BUILD, "kapitsa")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fails = 0


def check(ok, name, detail=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else "\n     " + str(detail)), flush=True)
    fails += 0 if ok else 1


def gpu_lock():
    if os.environ.get("NACE_GPU_LOCK_HELD") == "1" or os.environ.get("NI_LOCK_HELD") == "1":
        return None
    path = os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock"
    f = open(path, "a")
    print(f"waiting for the GPU lock {path} ...", flush=True)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def find_tokenizer():
    base = os.path.expanduser("~/.cache/kapitsa/models")
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        d = os.path.join(base, name)
        try:
            c = json.load(open(os.path.join(d, "config.json")))
        except (OSError, ValueError):
            continue
        if os.path.exists(os.path.join(d, "tokenizer.json")) and c.get("text_config", c).get("vocab_size") == 248320:
            return d
    sys.exit("no Qwen3.5-family model with a tokenizer under ~/.cache/kapitsa/models: pass --tokenizer DIR")


class Proc:
    """a kapitsa process with its log; ready() waits for its line or its port"""

    def __init__(self, tmp, name, args):
        env = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
        env.update({"KAPITSA_HOME": os.path.join(tmp, "home"), "KAPITSA_NO_AUTO_UPDATE": "1", "KAPITSA_NO_PROMPT": "1"})
        self.log_path = os.path.join(tmp, name + ".log")
        self.p = subprocess.Popen([KAPITSA] + args, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT, env=env)

    def log(self):
        return open(self.log_path, errors="replace").read()

    def wait_for(self, ok, what, timeout=180):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if ok():
                return
            if self.p.poll() is not None:
                sys.exit(f"{what}: the process ended\n{self.log()[-2000:]}")
            time.sleep(0.3)
        sys.exit(f"{what}: not ready in {timeout} s\n{self.log()[-2000:]}")

    def stop(self, sig=signal.SIGTERM):
        if self.p.poll() is None:
            self.p.send_signal(sig)
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                self.p.kill()


def stage(tmp, name, layers, port):
    s = Proc(tmp, name, ["stage", "--model", PACK, "--layers", layers, "--port", str(port)])
    s.wait_for(lambda: "listening" in s.log(), name)
    return s


def stage_flags(tmp):
    """The stage command's own flags, before any model is loaded: a chunk below one is a usage error, as `kapitsa bench`
    says of its own --prefill-chunk, and a good one still parses and reaches the model lookup. No GPU, no tokenizer, no
    peer, no lock: a usage error exits in the parse loop (status 2), before anything is loaded. The model is a folder
    that does not exist and the network is off, so a chunk that parses cannot start anything: the stage stops at the
    lookup (status 1, "No folder at") and never listens. Its own HOME and KAPITSA_HOME keep the caller's caches, the
    catalog and any saved settings out of these runs, as tests/test_bench.py does for its scripted server."""
    home = os.path.join(tmp, "home")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NI_") and not k.startswith("KAPITSA_")}
    env.update({"KAPITSA_HOME": home, "HOME": os.path.join(home, "user"), "KAPITSA_NO_AUTO_UPDATE": "1",
                "KAPITSA_NO_PROMPT": "1", "HF_HUB_OFFLINE": "1"})
    os.makedirs(env["HOME"])
    model = os.path.join(tmp, "no-such-model")

    def run(args):
        return subprocess.run([KAPITSA] + args, capture_output=True, text=True, timeout=10, env=env)

    missing = run(["stage", "--model", model, "--prefill-chunk"])
    check(missing.returncode == 2 and "needs a value" in missing.stderr, "prefill chunk requires a value",
          missing.stderr[:300])
    for value, why in (("0", "must be a positive integer"), ("-1", "needs a whole number"),
                       ("abc", "needs a whole number"), ("4096x", "needs a whole number"),
                       ("99999999999999999", "needs a whole number")):
        bad = run(["stage", "--model", model, "--prefill-chunk", value])
        check(bad.returncode == 2 and "--prefill-chunk" in bad.stderr and why in bad.stderr,
              f"invalid prefill chunk {value!r} is a usage error naming the flag", bad.stderr[:300])
    good = run(["stage", "--model", model, "--prefill-chunk", "2048"])
    check(good.returncode == 1 and "No folder at" in good.stderr and "--prefill-chunk" not in good.stderr,
          "a positive prefill chunk parses and stops at the model lookup", good.stderr[:300])


def server(tmp, name, port, split=None):
    args = ["serve", "--model", PACK, "--port", str(port), "--ctx", "16384", "--disk-cache", "0"]
    s = Proc(tmp, name, args + (["--split", split] if split else []))

    def up():
        try:
            return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
        except (urllib.error.URLError, OSError):
            return False

    s.wait_for(up, name)
    return s


def chat(port, messages, max_tokens=64, stream=False):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            if not stream:
                j = json.load(r)
                m = j["choices"][0]["message"]
                return (m.get("content") or "") + "|" + j["choices"][0]["finish_reason"]
            out = []
            for line in r:
                line = line.decode().strip()
                if line.startswith("data:") and line != "data: [DONE]":
                    for c in json.loads(line[5:]).get("choices", []):
                        out.append(c.get("delta", {}).get("content") or "")
            return "".join(out)
    except urllib.error.HTTPError as e:
        return "HTTP %d %s" % (e.code, e.read().decode(errors="replace"))


LONG = " ".join(f"line {i}: the quick brown fox jumps over the lazy dog {i * 7 % 13}" for i in range(260))
CONVS = {
    "a short prompt": [{"role": "user", "content": "Hello there, how are you?"}],
    "a system prompt": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "List three colors."}],
    "a long prompt (two prefill chunks)": [{"role": "user", "content": LONG + "\nSummarize."}],
}


def compare(whole, split, label):
    for name, msgs in CONVS.items():
        a, b = chat(whole, msgs), chat(split, msgs)
        check(a == b, f"{label}: {name}", f"whole {a[:120]!r}\n     split {b[:120]!r}")
    m1 = CONVS["a system prompt"]
    m2 = m1 + [{"role": "assistant", "content": chat(whole, m1).split("|")[0]}, {"role": "user", "content": "Three more."}]
    a, b = chat(whole, m2), chat(split, m2)
    check(a == b, f"{label}: a follow-up turn", f"whole {a[:120]!r}\n     split {b[:120]!r}")
    a, b = chat(whole, CONVS["a short prompt"], stream=True), chat(split, CONVS["a short prompt"], stream=True)
    check(a == b, f"{label}: a streamed reply", f"whole {a[:120]!r}\n     split {b[:120]!r}")
    got = {}
    keys = ["a short prompt", "a system prompt"]
    ts = [threading.Thread(target=lambda k=k: got.__setitem__(k, chat(split, CONVS[k], max_tokens=96))) for k in keys]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for k in keys:
        a = chat(whole, CONVS[k], max_tokens=96)
        check(a == got[k], f"{label}: two requests at once ({k})", f"whole {a[:120]!r}\n     split {got[k][:120]!r}")


def main():
    global PACK
    tmp = tempfile.mkdtemp(prefix="kapitsa-split-")
    PACK = os.path.join(tmp, "flash-next-synthetic")
    # 0. the stage command's flags alone: no model, no GPU, no lock
    stage_flags(tmp)
    if FLAGS_ONLY:
        print("FAIL" if fails else "PASS", f"({fails} failed)")
        if KEEP:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(1 if fails else 0)
    lock = gpu_lock()
    procs = []
    try:
        tok = TOKENIZER or find_tokenizer()
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "fn_synth.py"), PACK, "--layers", "8", "--seed", "11",
                        "--tokenizer", tok], check=True, stdout=subprocess.DEVNULL)
        whole = server(tmp, "whole", 8411)
        procs.append(whole)
        # 1. one stage
        s1 = stage(tmp, "stage-5-8", "5-8", 9411)
        procs.append(s1)
        sp = server(tmp, "split-1", 8412, "127.0.0.1:9411")
        procs.append(sp)
        compare(8411, 8412, "one stage")
        sp.stop()
        s1.stop()
        # 2. two stages
        s2 = stage(tmp, "stage-3-4", "3-4", 9412)
        s3 = stage(tmp, "stage-5-8b", "5-8", 9413)
        procs += [s2, s3]
        sp2 = server(tmp, "split-2", 8413, "127.0.0.1:9412,127.0.0.1:9413")
        procs.append(sp2)
        compare(8411, 8413, "two stages")
        # 3. the last stage goes away in the middle of a reply
        killer = threading.Timer(1.0, lambda: s3.stop(signal.SIGKILL))
        killer.start()
        t0 = time.time()
        r = chat(8413, [{"role": "user", "content": "Write a long story."}], max_tokens=4000)
        check("split across Macs" in r and time.time() - t0 < 30, "a lost stage ends the reply in flight with an error", r[:200])
        t0 = time.time()
        r = chat(8413, CONVS["a short prompt"])
        check("split across Macs" in r and time.time() - t0 < 5, "and the next request at once", r[:200])
    finally:
        for p in procs:
            p.stop()
        if KEEP:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        if lock:
            lock.close()
    print("FAIL" if fails else "PASS", f"({fails} failed)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
