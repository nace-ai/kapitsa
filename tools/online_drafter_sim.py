#!/usr/bin/env python3
"""Replay logged speculative-decoding cycles against online-adaptation policies. CPU only, stdlib only.

Question it answers: before anyone writes a backward pass for the drafter, how much room is there in adapting
what the engine already logs per cycle? Two parts.

1. Calibration replay (`calib`). A rowlog is one JSON object per verify cycle (`conf[7]` the drafter's confidence at
   draft positions 0-6, `acc` accepted drafts, `rows` rows verified, `wall_ms`). Position i was accepted when
   i < acc, missed when i == acc, and never tested beyond (censored). Each predictor sees only what the engine
   could see online (labels of cycles already finished) and predicts p(accept at i) for the next cycle. Scored
   by log-loss on the tested positions. Predictors:
     static    p = conf                                  (a drafter taken at its word)
     kf        per-depth logistic p = sigmoid(a_d + b_d logit(conf)), Newton refit every 16 cycles on the last 2048
               records, prior toward (0, 1) with weight ~4 records. Same algorithm as KfBudget::fit
               (src/model/kflash_budget.cpp:153, refit_tick :311).
     kf_warm   `kf` started from the parameters fitted on all OTHER sessions (leave-one-session-out): what a
               persisted per-model cache gives a fresh engine.
     frozen    the other sessions' parameters, never refit (a cache that is read and never written).
     kf_ema    `frozen` plus ONE session parameter: a logit shift s tracked by an exponential moving average of
               (hit - predicted) residuals, no refit; the cheapest "online learning" that is not a weight update.

2. Gain calculator (`gain`). A weight update that moves the drafter's per-position acceptance by a logit shift
   delta changes tokens per cycle; training steals a fraction f of the GPU. Net decode speed ratio =
   (tok'/tok) * (1 - f) at a fixed verify depth (the cycle's cost is held constant: an upper bound, rows cost
   ~1 ms each on the 35B). Prints the shift needed to break even for a given f, so a candidate method can be
   rejected on the acceptance it must deliver before anything is built.

usage: online_drafter_sim.py calib ROWLOG [ROWLOG ...]
       online_drafter_sim.py shift ROWLOG [ROWLOG ...] [--group REGEX]   (leave-one-class-out calibration shift)
       online_drafter_sim.py gain ROWLOG [ROWLOG ...] [--train-frac 0.05]
"""
import argparse
import json
import math

KD = 7  # draft positions in a rowlog's conf[]
WIN = 2048
REFIT_EVERY = 16
MIN_RECORDS = 32
LAM = 1.0


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def logit_q(c):
    """KfBudget::logit_q: the lattice probability c, capped just under 1, as a log-odds."""
    c = min(max(c, 1e-6), 0.999999)
    return math.log(c) - math.log1p(-c)


def load_rowlog(path):
    cycles = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("valid", 1) and not r.get("explore", 0):
                cycles.append({"conf": r["conf"], "acc": int(r["acc"]), "rows": int(r["rows"]),
                               "wall_ms": float(r.get("wall_ms", 0.0))})
    return cycles


def labels(cyc):
    """[(position, hit)] for the positions this cycle actually tested: hits 0..acc-1, then one miss unless acc
    reached the last draft position."""
    acc = cyc["acc"]
    out = [(i, 1) for i in range(min(acc, KD))]
    if acc < KD:
        out.append((acc, 0))
    return out


def fit(xs, ys, alpha, beta, iters=4):
    """L2-regularised logistic regression of y on (1, x) toward (0, 1); Newton steps from (alpha, beta). A port of
    KfBudget::fit (kflash_budget.cpp:153): beta clamped to [0.1, 4], alpha to [-8, 8]."""
    a, b = alpha, beta
    for _ in range(iters):
        ga, gb, haa, hab, hbb = LAM * a, LAM * (b - 1.0), LAM, 0.0, LAM
        for x, y in zip(xs, ys):
            p = sigmoid(a + b * x)
            w = p * (1 - p)
            r = p - y
            ga += r
            gb += r * x
            haa += w
            hab += w * x
            hbb += w * x * x
        det = haa * hbb - hab * hab
        if not det > 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a = min(max(a - da, -8.0), 8.0)
        b = min(max(b - db, 0.1), 4.0)
        if abs(da) + abs(db) < 1e-4:
            break
    return a, b


class KfCalib:
    """Per-depth online logistic calibrator (the shipped KfBudget design)."""

    def __init__(self, init=None):
        self.ab = [list(p) for p in (init or [(0.0, 1.0)] * KD)]
        self.xs = [[] for _ in range(KD)]
        self.ys = [[] for _ in range(KD)]
        self.head = [0] * KD
        self.since = 0

    def predict(self, conf):
        return [sigmoid(self.ab[i][0] + self.ab[i][1] * logit_q(conf[i])) for i in range(KD)]

    def observe(self, cyc):
        for i, y in labels(cyc):
            x = logit_q(cyc["conf"][i])
            if len(self.xs[i]) < WIN:
                self.xs[i].append(x)
                self.ys[i].append(y)
            else:
                self.xs[i][self.head[i]] = x
                self.ys[i][self.head[i]] = y
                self.head[i] = (self.head[i] + 1) % WIN
        self.since += 1
        if self.since >= REFIT_EVERY:
            self.since = 0
            for i in range(KD):
                if len(self.xs[i]) >= MIN_RECORDS:
                    self.ab[i] = list(fit(self.xs[i], self.ys[i], *self.ab[i]))


class Frozen(KfCalib):
    """The parameters it starts with, never refit (a cache that is read and never written)."""

    def observe(self, cyc):
        pass


class EmaDrift(Frozen):
    """Frozen parameters plus ONE session parameter, a logit shift s: p = sigmoid(a + b x + s), s moved by an
    exponential moving average of the residual (hit - predicted). No refit, so it is not double counting."""

    def __init__(self, init=None, rate=0.02, gain=4.0):
        super().__init__(init)
        self.s, self.rate, self.gain = 0.0, rate, gain

    def predict(self, conf):
        return [sigmoid(self.ab[i][0] + self.ab[i][1] * logit_q(conf[i]) + self.s) for i in range(KD)]

    def observe(self, cyc):
        pred = self.predict(cyc["conf"])
        lab = labels(cyc)
        if lab:
            resid = sum(y - pred[i] for i, y in lab) / len(lab)
            self.s = min(max(self.s + self.rate * self.gain * resid, -3.0), 3.0)


def logloss(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(math.log(p) if y else math.log(1 - p))


def score(session, make):
    """Mean log-loss over the tested positions of one session, predicting each cycle before observing it."""
    m, tot, n = make(), 0.0, 0
    for cyc in session:
        pred = m.predict(cyc["conf"])
        for i, y in labels(cyc):
            tot += logloss(pred[i], y)
            n += 1
        m.observe(cyc)
    return tot / max(n, 1), n


def fit_pooled(sessions):
    """Per-depth (alpha, beta) on every record of the given sessions (what a cache from other sessions holds)."""
    ab = []
    for i in range(KD):
        xs, ys = [], []
        for s in sessions:
            for cyc in s:
                for j, y in labels(cyc):
                    if j == i:
                        xs.append(logit_q(cyc["conf"][i]))
                        ys.append(y)
        ab.append(fit(xs[-WIN * 4:], ys[-WIN * 4:], 0.0, 1.0, 12) if len(xs) >= MIN_RECORDS else (0.0, 1.0))
    return ab


def expected_tokens(p, k):
    e, run = 1.0, 1.0
    for j in range(k):
        run *= p[j]
        e += run
    return e


def mean_tokens(session, shift=0.0, ab=None):
    """Mean tokens per cycle with the full chain verified, under per-position p = sigmoid(a_i + b_i logit(conf) +
    shift); (a_i, b_i) default to the identity, a calibrated baseline comes from fit_pooled."""
    ab = ab or [(0.0, 1.0)] * KD
    tot = 0.0
    for cyc in session:
        p = [sigmoid(ab[i][0] + ab[i][1] * logit_q(cyc["conf"][i]) + shift) for i in range(KD)]
        tot += expected_tokens(p, KD)
    return tot / max(len(session), 1)


def speedup(session, shift, train_frac, ab=None):
    """Net decode-speed ratio of a drafter whose acceptance logit moved by `shift` from the calibrated baseline,
    with train_frac of the GPU spent on the update, at a fixed verify depth (the cost per cycle does not change)."""
    return mean_tokens(session, shift, ab) / mean_tokens(session, 0.0, ab) * (1.0 - train_frac)


def break_even_shift(session, train_frac, ab=None):
    lo, hi = 0.0, 6.0
    if speedup(session, hi, train_frac, ab) < 1.0:
        return None
    for _ in range(50):
        mid = (lo + hi) / 2
        if speedup(session, mid, train_frac, ab) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def sessions_from(paths):
    return [(p, load_rowlog(p)) for p in paths]


def cmd_calib(args):
    ss = sessions_from(args.rowlogs)
    print(f"{'session':28} {'cyc':>5} {'static':>8} {'frozen':>8} {'kf':>8} {'kf_warm':>8} {'kf_ema':>8}")
    tot = {"static": 0, "frozen": 0, "kf": 0, "kf_warm": 0, "kf_ema": 0}
    cnt = 0
    for idx, (path, sess) in enumerate(ss):
        others = [s for j, (_, s) in enumerate(ss) if j != idx]
        warm = fit_pooled(others) if others else None
        mk = {"static": lambda: _Static(), "kf": lambda: KfCalib(), "kf_warm": lambda: KfCalib(warm),
              "frozen": lambda: Frozen(warm), "kf_ema": lambda: EmaDrift(warm)}
        ll = {k: score(sess, f) for k, f in mk.items()}
        n = ll["static"][1]
        for k in tot:
            tot[k] += ll[k][0] * n
        cnt += n
        print(f"{path.rsplit('/', 1)[-1]:28} {len(sess):5d} " + " ".join(f"{ll[k][0]:8.4f}" for k in
              ("static", "frozen", "kf", "kf_warm", "kf_ema")))
    print("pooled log-loss (position-weighted): " + "  ".join(f"{k} {v / max(cnt, 1):.4f}" for k, v in tot.items()))


class _Static:
    def predict(self, conf):
        return [min(max(c, 1e-6), 1 - 1e-6) for c in conf[:KD]]

    def observe(self, cyc):
        pass


def cmd_gain(args):
    ss = [s for _, s in sessions_from(args.rowlogs)]
    sess = [c for s in ss for c in s]
    ab = fit_pooled(ss)
    base_tok = mean_tokens(sess, 0.0, ab)
    print(f"cycles {len(sess)}; chain tokens/cycle: raw conf {mean_tokens(sess):.3f}, calibrated {base_tok:.3f}, "
          f"logged {sum(c['acc'] + 1 for c in sess) / len(sess):.3f}")
    print("calibration (alpha, beta) by position:", " ".join(f"({a:+.2f},{b:.2f})" for a, b in ab))
    print(f"a +shift in the acceptance logit at every position -> tokens/cycle ratio and net speed ratio "
          f"({args.train_frac:.0%} of the GPU spent training)")
    for s in (0.1, 0.25, 0.5, 0.75, 1.0):
        print(f"  shift {s:+.2f}: tok x{mean_tokens(sess, s, ab) / base_tok:.3f}, "
              f"net x{speedup(sess, s, args.train_frac, ab):.3f}")
    be = break_even_shift(sess, args.train_frac, ab)
    print("break-even logit shift at that training share:", "none up to +6" if be is None else f"{be:+.3f}")


def cmd_shift(args):
    """Leave-one-group-out: calibrate on the other groups (files grouped by --group, a regex whose group 1 names the
    class), then find the logit shift that fits the held-out group best. A user whose domain was not in the
    calibration mix, in one number; tokens/cycle is the chain's expected value under the shifted probabilities."""
    import re
    groups = {}
    for path, sess in sessions_from(args.rowlogs):
        m = re.search(args.group, path.rsplit("/", 1)[-1])
        groups.setdefault(m.group(1) if m else "?", []).append(sess)
    print(f"{'group':8} {'sessions':>8} {'cycles':>7} {'best shift':>10} {'logloss':>8} {'at shift':>9} {'tok/cyc x':>10}")
    for g, held in groups.items():
        ab = fit_pooled([s for k, v in groups.items() if k != g for s in v])
        flat = [c for s in held for c in s]

        def ll(sh):
            t = n = 0
            for c in flat:
                for i, y in labels(c):
                    t += logloss(sigmoid(ab[i][0] + ab[i][1] * logit_q(c["conf"][i]) + sh), y)
                    n += 1
            return t / n

        best = min((ll(x / 20), x / 20) for x in range(-60, 61))
        print(f"{g:8} {len(held):8d} {len(flat):7d} {best[1]:+10.2f} {ll(0.0):8.4f} {best[0]:9.4f} "
              f"{mean_tokens(flat, best[1], ab) / mean_tokens(flat, 0.0, ab):10.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calib", "gain", "shift"):
        p = sub.add_parser(name)
        p.add_argument("rowlogs", nargs="+")
        if name == "shift":
            p.add_argument("--group", default=r"fresh_([a-z]+?)\d*(?:_r\d)?\.", help="regex; group 1 = the class")
        p.add_argument("--train-frac", type=float, default=0.05)
    args = ap.parse_args()
    {"calib": cmd_calib, "gain": cmd_gain, "shift": cmd_shift}[args.cmd](args)


if __name__ == "__main__":
    main()
