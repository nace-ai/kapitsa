#!/usr/bin/env python3
"""tests/vision/check_ink2.py's tower gate on invalid numbers, with CPU stand-ins (no model, no GPU).

usage: test_check_ink2.py   (needs numpy, as check_ink2.py does; skipped without it)

The stand-ins play `vit_check` (prep and tower) and the reference interpreter (`--python`): they write small arrays
in the layouts check_ink2.py reads, the candidate within the bf16 reference's error of the float32 one, so the
control passes. STUB_BAD="where:stage:kind" puts a NaN or an infinity into one array: where is cand (Kapitsa's
output), ref (the float32 reference) or bf16 (the bf16 reference), stage is h0, h1, hL or emb. Every such run must
fail with the stage named INVALID: a non-finite stage once passed because max() kept the last finite ratio (review
6073858784 on #1431). The GPU lock is a temporary file.
"""
import os
import subprocess
import sys
import tempfile

try:
    import numpy  # noqa: F401
except ImportError:
    print("SKIP test_check_ink2.py (numpy is not installed)")
    sys.exit(0)

HERE = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(HERE, "vision", "check_ink2.py")
FAILS = []

COMMON = r'''
import os, sys, numpy as np
H, TOKENS, DEPTH = 8, 1, 2
ROWS = 1 + TOKENS * 4
def base(stage):
    shape = (TOKENS, H) if stage == "emb" else (ROWS, H)
    return np.random.default_rng({"h0": 1, "h1": 2, "hL": 3, "emb": 4}[stage]).normal(size=shape).astype(np.float32)
def noisy(values, scale, seed):
    return (values * (1 + scale * np.random.default_rng(seed).normal(size=values.shape))).astype(np.float32)
def spoil(where, stage, values):
    bad = os.environ.get("STUB_BAD", "")
    if bad:
        w, s, kind = bad.split(":")
        if w == where and s == stage:
            values = values.copy()
            values.flat[min(3, values.size - 1)] = np.nan if kind == "nan" else np.inf
    return values
'''

VIT = COMMON + r'''
cmd = sys.argv[1]
if cmd == "prep":
    np.zeros(40 * 40 * 3, np.float32).tofile(sys.argv[4])
    print(1, 1, TOKENS, 40, 40)
elif cmd == "tower":
    out, layers = sys.argv[4], (int(sys.argv[5]) if len(sys.argv) > 5 else None)
    stage = "emb" if layers is None else {0: "h0", 1: "h1", DEPTH: "hL"}[layers]
    values = spoil("cand", stage, noisy(base(stage), 1e-4, 7))
    if stage == "emb":  # bf16, as the encoder writes its rows
        (values.view(np.uint32) >> 16).astype(np.uint16).tofile(out)
    else:
        values.tofile(out)
'''

REF = COMMON + r'''
args = [a for a in sys.argv[1:] if a != "-I"]
out = args[3]
os.makedirs(out, exist_ok=True)
np.zeros(40 * 40 * 3, np.float32).tofile(os.path.join(out, "pv.f32"))
if "--prep" not in args:
    bf16 = "--bf16" in args
    for stage in ("h0", "h1", "hL", "emb"):
        values = noisy(base(stage), 1e-2, 11) if bf16 else base(stage)
        spoil("bf16" if bf16 else "ref", stage, values).tofile(os.path.join(out, stage + ".f32"))
print(1, 1, TOKENS)
'''


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


with tempfile.TemporaryDirectory() as tmp:
    model = os.path.join(tmp, "model")
    os.makedirs(model)
    with open(os.path.join(model, "config.json"), "w") as f:
        f.write('{"vision_config": {"num_hidden_layers": 2}}')
    stubs = {}
    for name, body in (("vit_check", VIT), ("ref_python", REF)):
        stubs[name] = os.path.join(tmp, name)
        with open(stubs[name], "w") as f:
            f.write(f"#!{sys.executable}\n{body}")
        os.chmod(stubs[name], 0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("NI_LOCK_HELD", "NACE_GPU_LOCK_HELD", "STUB_BAD")}
    env["BENCH_GPU_LOCK"] = os.path.join(tmp, "gpu.lock")

    def run(bad=None):
        e = dict(env, **({"STUB_BAD": bad} if bad else {}))
        r = subprocess.run([sys.executable, CHECK, model, stubs["vit_check"], os.path.join(tmp, "x.png"), "--gpu",
                            "--python", stubs["ref_python"]], env=e, capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    rc, out = run()
    check(rc == 0 and "0 failed" in out and "ok   tower" in out, "control: finite arrays within the bf16 error pass",
          out[-400:])
    for where in ("cand", "ref", "bf16"):
        for stage in ("h0", "h1", "hL", "emb"):
            rc, out = run(f"{where}:{stage}:nan")
            check(rc == 1 and f"INVALID {stage}:" in out and "FAIL tower" in out, f"NaN in {where} {stage} fails",
                  out[-400:])
    rc, out = run("cand:h1:inf")
    check(rc == 1 and "INVALID h1:" in out, "an infinity in a hidden stage fails", out[-400:])

if FAILS:
    print(f"\n{len(FAILS)} FAILED")
    sys.exit(1)
print("\nall passed")
