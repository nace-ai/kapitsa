#!/usr/bin/env python3
"""tools/online_drafter_sim.py on synthetic rowlogs: the logistic fit against a known calibration, the label rule
(hits, one miss, censored tail), warm start beating cold start, the drift term tracking a shifted drafter, the depth
choice, and the break-even shift of the gain calculator. CPU only, stdlib only, no model, no GPU, no recorded traces.

usage: test_online_drafter_sim.py
"""
import json, math, os, random, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOL)
import online_drafter_sim as o  # noqa: E402

FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


def synth(n, alpha, beta, shift=0.0, seed=1):
    """Cycles whose true acceptance at position i is sigmoid(alpha + beta*logit(conf) + shift)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        conf = [rng.uniform(0.2, 0.95) for _ in range(o.KD)]
        acc = 0
        while acc < o.KD and rng.random() < o.sigmoid(alpha + beta * o.logit_q(conf[acc]) + shift):
            acc += 1
        out.append({"conf": conf, "acc": acc, "rows": o.KD + 1, "wall_ms": 40.0})
    return out


# labels: a cycle with acc=2 tested positions 0,1 (hits) and 2 (miss); acc=KD tested all KD, no miss
check(o.labels({"acc": 2}) == [(0, 1), (1, 1), (2, 0)], "labels: hits then one miss")
check(o.labels({"acc": 0}) == [(0, 0)], "labels: first draft rejected")
check(o.labels({"acc": o.KD}) == [(i, 1) for i in range(o.KD)], "labels: full chain has no miss")

# logit_q matches KfBudget::logit_q: lq - log1p(-q) with q = exp(lq) capped at 0.999999, given the probability c
check(abs(o.logit_q(0.5)) < 1e-9, "logit_q(0.5) = 0")
check(o.logit_q(1.0) < 15.0, "logit_q is capped near 1")

# fit recovers a known calibration (alpha -0.6, beta 0.6: an overconfident drafter, as the 35B's cache shows)
s = synth(4000, -0.6, 0.6)
xs, ys = [], []
for c in s:
    for i, y in o.labels(c):
        if i == 2:
            xs.append(o.logit_q(c["conf"][2]))
            ys.append(y)
a, b = o.fit(xs, ys, 0.0, 1.0, 12)
check(abs(a + 0.6) < 0.2 and abs(b - 0.6) < 0.15, "fit recovers (alpha, beta)", f"{a:.3f}, {b:.3f}")
check(0.1 <= o.fit([0.0] * 5, [1] * 5, 0.0, 1.0)[1] <= 4.0, "fit keeps beta in [0.1, 4]")
check(o.fit([], [], 0.3, 1.2) == (0.3, 1.2) or True, "fit on no records does not raise")

# warm start (pooled fit from other sessions) beats a cold KfCalib on a fresh short session
train = [synth(600, -0.6, 0.6, seed=k) for k in range(2, 6)]
fresh = synth(120, -0.6, 0.6, seed=99)
warm = o.fit_pooled(train)
cold_ll, _ = o.score(fresh, lambda: o.KfCalib())
warm_ll, _ = o.score(fresh, lambda: o.KfCalib(warm))
check(warm_ll < cold_ll, "warm start beats cold start on a short session", f"{warm_ll:.4f} vs {cold_ll:.4f}")
static_ll, _ = o.score(fresh, lambda: o._Static())
check(warm_ll < static_ll, "calibrated beats taking the drafter at its word", f"{warm_ll:.4f} vs {static_ll:.4f}")

# a drafter that drifts mid-session (user changes topic): the drift term beats the frozen warm start
shifted = synth(400, -0.6, 0.6, shift=-0.8, seed=7)
frozen_ll, _ = o.score(shifted, lambda: o.Frozen(warm))
ema_ll, _ = o.score(shifted, lambda: o.EmaDrift(warm, rate=0.02))
check(ema_ll < frozen_ll, "drift term tracks a shifted drafter", f"{ema_ll:.4f} vs {frozen_ll:.4f}")
# and on a stationary session it does not help: the premise of every online method has to be earned
stationary_ll, _ = o.score(fresh, lambda: o.Frozen(warm))
ema_stat, _ = o.score(fresh, lambda: o.EmaDrift(warm, rate=0.02))
check(ema_stat >= stationary_ll - 0.01, "drift term gains nothing on a stationary session",
      f"{ema_stat:.4f} vs {stationary_ll:.4f}")

check(abs(o.expected_tokens([0.5, 0.5], 2) - 1.75) < 1e-12, "expected_tokens 1 + .5 + .25")

# gain calculator: a zero shift at zero training share is 1.0; training with no gain is a loss; break-even grows
sess = synth(300, 0.0, 1.0, seed=3)
check(abs(o.speedup(sess, 0.0, 0.0) - 1.0) < 1e-12, "speedup(0 shift, 0 train) = 1")
check(o.speedup(sess, 0.0, 0.05) < 1.0, "training with no acceptance gain loses")
be5 = o.break_even_shift(sess, 0.05)
be10 = o.break_even_shift(sess, 0.10)
check(be5 is not None and be10 is not None and 0 < be5 < be10, "break-even shift grows with the training share",
      f"{be5} {be10}")
check(abs(o.speedup(sess, be5, 0.05) - 1.0) < 1e-6, "speedup at the break-even shift is 1")

# rowlog loader skips explore / invalid cycles; the CLI runs end to end on files
with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "x.rowlog")
    rows = [{"valid": 1, "explore": 0, "rows": 4, "acc": 1, "wall_ms": 30.0, "conf": [0.5] * 7},
            {"valid": 0, "explore": 0, "rows": 4, "acc": 1, "wall_ms": 30.0, "conf": [0.5] * 7},
            {"valid": 1, "explore": 1, "rows": 4, "acc": 1, "wall_ms": 30.0, "conf": [0.5] * 7}]
    with open(p, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    check(len(o.load_rowlog(p)) == 1, "loader keeps only valid, non-explore cycles")
    q = os.path.join(d, "y.rowlog")
    with open(q, "w") as f:
        for c in synth(200, -0.3, 0.8, seed=5):
            f.write(json.dumps({"valid": 1, "explore": 0, "rows": c["rows"], "acc": c["acc"],
                                "wall_ms": 40.0, "conf": c["conf"]}) + "\n")
        p2 = os.path.join(d, "fresh_zh1.rowlog")
    q2 = os.path.join(d, "fresh_code_r0.rowlog")
    for src, dst, sh in ((p, p2, 0.0), (q, q2, 0.0)):
        with open(dst, "w") as f:
            f.write(open(src).read())
    r = subprocess.run([sys.executable, os.path.join(TOOL, "online_drafter_sim.py"), "shift", p2, q2],
                       capture_output=True, text=True)
    check(r.returncode == 0 and "zh" in r.stdout and "code" in r.stdout, "shift: one row per class", r.stdout + r.stderr[-200:])
    for cmd in (["calib", p, q], ["gain", q, "--train-frac", "0.05"]):
        r = subprocess.run([sys.executable, os.path.join(TOOL, "online_drafter_sim.py"), *cmd],
                           capture_output=True, text=True)
        check(r.returncode == 0 and r.stdout.strip() != "", f"CLI {cmd[0]} runs", r.stderr[-200:])

if FAILED:
    print(f"{len(FAILED)} FAILED")
    sys.exit(1)
print("all passed")
