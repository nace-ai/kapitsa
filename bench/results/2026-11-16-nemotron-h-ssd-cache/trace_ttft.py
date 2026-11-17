#!/usr/bin/env python3
"""tests/test_disk_cache.py --ttft with NI_TRACE_TOKENS set for every server it starts (one file per server, in start
order: per round the cache-on server of turn 1, the one that answers turn 2 from the SSD, then the one without the
cache: warm-up, turn 2 cold, turn 1, warm-up, turn 2 from RAM). Then compares turn 2's trace lines (the decode's tokens
and each token's top-1 minus top-2 logit margin, src/engine/engine.cpp) from the SSD against RAM, and against cold.
usage: trace_ttft.py OUT_DIR <test_disk_cache.py's arguments, with --ttft>   (under the GPU lock, as that test)"""
import json, os, subprocess, sys

OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
sys.argv = ["test_disk_cache.py"] + sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "tests") if os.path.isdir(os.path.join(here, "tests")) else here)
import test_disk_cache as tdc  # noqa: E402

starts = [0]
_popen = subprocess.Popen


def popen(cmd, *a, env=None, **kw):
    if len(cmd) > 1 and cmd[1] == "serve":
        starts[0] += 1
        env = dict(env or os.environ, NI_TRACE_TOKENS=os.path.join(OUT, f"server-{starts[0]:02d}.jsonl"))
    return _popen(cmd, *a, env=env, **kw)


tdc.subprocess.Popen = popen
rc = tdc.main()


def lines(i):
    p = os.path.join(OUT, f"server-{i:02d}.jsonl")
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


bad = 0
for r in range(starts[0] // 3):
    ssd = [x for x in lines(3 * r + 2) if x["prompt_tokens"] > 1000]
    nc = [x for x in lines(3 * r + 3) if x["prompt_tokens"] > 1000]  # cold turn 2, turn 1, RAM turn 2
    if len(ssd) != 1 or len(nc) != 3:
        print(f"round {r + 1}: unexpected trace lines (SSD {len(ssd)}, no-cache server {len(nc)})")
        bad += 1
        continue
    s, cold, ram = ssd[0], nc[0], nc[2]
    for name, o in (("RAM", ram), ("cold", cold)):
        same_p = s["prompt_hash"] == o["prompt_hash"]
        same_t = s["first"] == o["first"] and s["tokens"] == o["tokens"]
        same_m = s["margins"] == o["margins"]
        diff = [i for i, (a, b) in enumerate(zip(s["margins"], o["margins"])) if a != b]
        mx = max((abs(a - b) for a, b in zip(s["margins"], o["margins"])), default=0.0)
        print(f"round {r + 1}: turn 2 ({s['prompt_tokens']} prompt tokens, {len(s['tokens'])} traced tokens), SSD "
              f"against {name}: same prompt {same_p}, same tokens {same_t}, every margin identical {same_m}"
              + ("" if same_m else f" ({len(diff)} differ, first at {diff[0]}, largest difference {mx:.6g})"),
              flush=True)
        if name == "RAM" and not (same_p and same_t and same_m and s["tokens"]):
            bad += 1
print("TRACE: SSD == RAM in every round" if not bad else f"TRACE: {bad} round(s) differ", flush=True)
sys.exit(rc or (1 if bad else 0))
