#!/usr/bin/env python3
"""MLX (mlx-lm) speed on a local model, the yardstick for Kapitsa's numbers on the same weights.

  speed_mlx.py <model dir> @prefill_ids @decode_prompt_ids [--gen 256] [--reps 3]

Prefill: time to the first token of the prefill prompt (whole prompt, mlx-lm's default 2048-token steps).
Decode: tokens/s of `gen` greedy tokens after the decode prompt (the first token excluded). Median of reps, after a
warm-up. Holds /tmp/kapitsa-gpu.lock (NI_LOCK_HELD=1: the caller holds it).
"""
import fcntl, json, os, statistics, sys, time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

args = sys.argv[1:]
gen = int(args[args.index("--gen") + 1]) if "--gen" in args else 256
reps = int(args[args.index("--reps") + 1]) if "--reps" in args else 3
model_dir, pf, dp = args[0], args[1], args[2]
ids = lambda a: [int(x) for x in open(a[1:]).read().split(",") if x.strip()]
pf_ids, dp_ids = ids(pf), ids(dp)
if not os.environ.get("NI_LOCK_HELD"):
    lock = os.fdopen(os.open(os.path.expanduser("/tmp/kapitsa-gpu.lock"), os.O_RDONLY | os.O_CREAT, 0o644))
    fcntl.flock(lock, fcntl.LOCK_EX)
model, tok = load(model_dir)
greedy = make_sampler(0.0)


def run(prompt, n):
    t0 = time.perf_counter()
    times = []
    for i, (t, _) in enumerate(generate_step(mx.array(prompt), model, max_tokens=n, sampler=greedy)):
        mx.eval(t)
        times.append(time.perf_counter())
    return times[0] - t0, (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 else 0.0


run(pf_ids[:256], 2)  # warm-up
pre = [len(pf_ids) / run(pf_ids, 1)[0] for _ in range(reps)]
dec = [run(dp_ids, gen)[1] for _ in range(reps)]
print(json.dumps({"model": model_dir, "prefill_tokens": len(pf_ids), "prefill_tok_s": round(statistics.median(pre), 1),
                  "prefill_runs": [round(x, 1) for x in pre], "decode_tokens": gen,
                  "decode_tok_s": round(statistics.median(dec), 2), "decode_runs": [round(x, 2) for x in dec],
                  "peak_gb": round(mx.get_peak_memory() / 1e9, 2)}))
