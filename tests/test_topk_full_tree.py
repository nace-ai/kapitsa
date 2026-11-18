#!/usr/bin/env python3
"""top_k 0 keeps Nemotron-H's drafter and samples the plain sampler's law (kapitsa#1286; model + GPU required).

  python3 tests/test_topk_full_tree.py --cli build/kapitsa --model ~/models/nemotron-3.5-lightning-nvfp4

A request whose top_k the two-pass sampler cannot hold (top_k 0: no truncation; above 64) is sampled over the whole
vocabulary (src/model/sampler.h). On a Nemotron-H model with a block drafter (LiLiCorr, DSpark) KFlash's tree verify
samples each row that way at the RNG step its position's plain step takes (NemotronH::encode_accept_tree), and the
acceptance keeps the longest path whose drafts equal those samples. So with the drafter on, a seed must give what plain
steps give, up to the verify's rows differing from a one-row step in the last bits of a logit.

Two servers in turn on the model: the default (the drafter on), then `--dflash none` (plain steps). Each answers the
same --n requests (temperature 1.0, top_p 1.0, top_k 0, --tokens tokens, no reasoning) at seeds 1..n, and n more at
seeds n+1..2n. The checks:
  - every drafter-on request logs the tree verify's line, drafts, and its server's tokens per cycle averages at least 1.5;
  - same seeds: the share of replies identical to plain steps' (at least --min-same, 0.8 by default);
  - the law: a two-sample chi-square (cells under 5 expected pooled) between the drafter's replies at seeds n+1..2n and
    plain steps' at seeds 1..n, on each of the first three words' histograms (p > 1e-3 each); plain steps against
    plain steps (seeds 1..n against n+1..2n) is printed beside it as the reference.
Takes the shared GPU lock, or respects NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench/harness"))
from gpu_hogs import GpuLock, held_by_caller  # noqa: E402

TREE = "sampled over the whole vocabulary, in KFlash's tree verify"
PROMPT = "Name one animal, then say in one short sentence where it lives."


def post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def serve(a, tmp, name, extra):
    log_path = Path(tmp) / f"serve-{name}.log"
    env = dict(os.environ, NI_LOCK_HELD="1", NACE_GPU_LOCK_HELD="1")
    with open(log_path, "w") as log:
        p = subprocess.Popen([a.cli, "serve", "--model", a.model, "--port", str(a.port), "--ctx", "4096", "--slots", "1",
                              "--disk-cache", "0", *extra], stdout=log, stderr=subprocess.STDOUT, env=env)
    for _ in range(900):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2)
            return p, log_path
        except Exception:
            if p.poll() is not None:
                sys.exit(f"server exited: {log_path.read_text()[-2000:]}")
            time.sleep(1)
    sys.exit("server did not answer /health")


def replies(a, tmp, name, extra):
    """{seed: reply text} for seeds 1..2n, and the server's log"""
    p, log_path = serve(a, tmp, name, extra)
    out = {}
    try:
        for seed in range(1, 2 * a.n + 1):
            body = {"model": "m", "messages": [{"role": "user", "content": PROMPT}], "temperature": 1.0, "top_p": 1.0,
                    "top_k": 0, "seed": seed, "max_tokens": a.tokens, "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False}}
            out[seed] = post(a.port, body)["choices"][0]["message"].get("content") or ""
    finally:
        p.terminate()
        p.wait(60)
    return out, log_path.read_text(errors="replace")


def gamma_q(s, x):  # regularized upper incomplete gamma Q(s, x): the chi-square tail
    if x <= 0:
        return 1.0
    if x < s + 1:
        term = total = 1.0 / s
        k = s
        for _ in range(10000):
            k += 1
            term *= x / k
            total += term
            if term < total * 1e-15:
                break
        return max(0.0, 1.0 - total * math.exp(-x + s * math.log(x) - math.lgamma(s)))
    b, c, d = x + 1 - s, 1e300, 1.0 / (x + 1 - s)
    h = d
    for i in range(1, 10000):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return math.exp(-x + s * math.log(x) - math.lgamma(s)) * h


def two_sample(xs, ys):
    """chi-square homogeneity of two samples of labels: (statistic, cells, p)"""
    cx, cy = Counter(xs), Counter(ys)
    nx, ny = len(xs), len(ys)
    cells, pool = [], [0, 0]
    for k in set(cx) | set(cy):
        o = (cx[k], cy[k])
        if (o[0] + o[1]) * min(nx, ny) / (nx + ny) < 5:
            pool[0] += o[0]
            pool[1] += o[1]
        else:
            cells.append(o)
    if sum(pool):
        cells.append(tuple(pool))
    stat = 0.0
    for ox, oy in cells:
        t = ox + oy
        for o, n in ((ox, nx), (oy, ny)):
            e = t * n / (nx + ny)
            stat += (o - e) ** 2 / e
    df = len(cells) - 1
    return stat, len(cells), gamma_q(df / 2, stat / 2) if df > 0 else 1.0


def word(text, i):
    w = re.findall(r"\w+|[^\w\s]", text.lower())
    return w[i] if len(w) > i else "<end>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=18596)
    ap.add_argument("--n", type=int, default=300, help="requests per seed set (each server answers 2n)")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--min-same", type=float, default=0.8)
    a = ap.parse_args()
    lock = nullcontext() if held_by_caller() else GpuLock(
        path=str(Path(os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock").expanduser().resolve()))
    failures = 0

    def check(ok, what):
        nonlocal failures
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}: {what}")

    with lock, tempfile.TemporaryDirectory() as tmp:
        tree, tlog = replies(a, tmp, "drafter", [])
        plain, plog = replies(a, tmp, "plain", ["--dflash", "none"])
    n = a.n
    lines = tlog.count(TREE)
    tpc = [float(x) for x in re.findall(r"([\d.]+) tok/cycle", tlog)]
    mean_tpc = sum(tpc) / len(tpc) if tpc else 0.0
    check(lines == 2 * n, f"every drafter request took the tree verify ({lines} of {2 * n})")
    check(mean_tpc >= 1.5, f"the drafter drafts: {mean_tpc:.2f} tokens per cycle on average over {len(tpc)} requests")
    check("tok/cycle" not in plog, "--dflash none decoded in plain steps")
    same = sum(tree[s] == plain[s] for s in range(1, n + 1))
    check(same >= a.min_same * n, f"same seeds, the same reply as plain steps: {same} of {n} ({same / n:.1%})")
    for i in range(3):
        t2 = [word(tree[s], i) for s in range(n + 1, 2 * n + 1)]
        p1 = [word(plain[s], i) for s in range(1, n + 1)]
        p2 = [word(plain[s], i) for s in range(n + 1, 2 * n + 1)]
        st, cells, pv = two_sample(t2, p1)
        rs, rcells, rpv = two_sample(p2, p1)
        print(f"  word {i + 1}: drafter (seeds {n + 1}..{2 * n}) vs plain (1..{n}): chi-square {st:.1f} over {cells} "
              f"cells, p = {pv:.3f}; plain vs plain: {rs:.1f} over {rcells}, p = {rpv:.3f}; "
              f"top: {Counter(t2).most_common(4)} / {Counter(p1).most_common(4)}")
        check(pv > 1e-3, f"word {i + 1}'s histogram: the drafter's matches plain steps' (p > 1e-3)")
    print(f"test_topk_full_tree: {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
