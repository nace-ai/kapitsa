#!/usr/bin/env python3
"""Tree-verify simulator for DFlash2 / DFlash drafters, on ni_run's lattice recordings (NI_DFLASH_LAT).

Record the lattice at every position of a plain output (one token per cycle), then replay draft policies on it:

  NI_DFLASH_LAT=x.lat NI_DFLASH_STEP1=1 ni_run <model> @prompt.ids 512 0 --dflash auto --ignore-eos
  tree_sim.py x.lat [y.lat ...]   (recordings of one model: the calibration is fitted leave-one-out)

Replays the walk's chain, the engine's tree (DFlash2 tree drafting: best-first, 7 nodes, score temperature tau (1 +
slope d) at position d), flat temperatures (--taus), a tree on per-depth calibrated probabilities, and the lattice's
own bound (the output's depth inside the top-16 lattice).

At T = 0 (and at T > 0 with tree acceptance by the verify rows' own samples, which draw position-keyed uniforms), the
tokens a cycle emits are exactly the recorded output: a draft tree accepts the longest root path that spells the output
after the anchor, plus one token. So any policy can be replayed exactly, cycle boundaries included, from the lattice at
every position: replay(policy) = tokens per cycle of a decode that drafts with it.

Record layout (int32 / float32, little endian): header [magic 'LAT1', version 1, prompt length n, T x 1000, K, v1,
step1, vocab]; per cycle: P, a, verified, tokens[P]; samples[8]; cand_i[7][16]; cand_v[7][16]; edges[7][16][16];
walk_s[7][16]; walk_k[7]; qprob[7][16]; per verify row r < 8: top-16 ids, top-16 logits, log-sum-exp, the logits of
cand_i[r] (rows 0-6).
"""
import argparse
import heapq
import math
import sys

import numpy as np

REC = np.dtype([("P", "<i4"), ("a", "<i4"), ("nv", "<i4"), ("anchor", "<i4"), ("samp", "<i4", 8),
                ("ci", "<i4", (7, 16)), ("cv", "<f4", (7, 16)), ("E", "<f4", (7, 16, 16)), ("ws", "<f4", (7, 16)),
                ("wk", "<i4", 7), ("q", "<f4", (7, 16)),
                ("rows", [("ti", "<i4", 16), ("tv", "<f4", 16), ("lse", "<f4"), ("cl", "<f4", 16)], 8)])


def load(path):
    raw = open(path, "rb").read()
    hdr = np.frombuffer(raw[:32], "<i4")
    if hdr[0] != 0x3154414C:
        sys.exit(f"{path}: not a lattice recording")
    recs = np.frombuffer(raw[32:], REC, count=(len(raw) - 32) // REC.itemsize)
    info = dict(n=int(hdr[2]), T=hdr[3] / 1000.0, K=int(hdr[4]), v1=bool(hdr[5]), step1=bool(hdr[6]), V=int(hdr[7]))
    return info, recs


class Run:
    """One step1 recording: the lattice at every position P and the output tokens seq[P] (the anchor at P)."""

    def __init__(self, path):
        self.path = path
        self.info, self.recs = load(path)
        if not self.info["step1"]:
            sys.exit(f"{path}: recorded without NI_DFLASH_STEP1 (the replay needs the lattice at every position)")
        self.P0 = int(self.recs[0]["P"])
        self.by_P = {int(r["P"]): i for i, r in enumerate(self.recs)}
        last = self.recs[-1]
        # tokens: anchors, then the last cycle's row-0 sample
        self.seq = {int(r["P"]): int(r["anchor"]) for r in self.recs}
        self.seq[int(last["P"]) + 1] = int(last["samp"][0])
        self.end = int(last["P"]) + 1  # last known token position

    def cont(self, P, k):
        """Output tokens P+1 .. P+k (fewer at the end of the recording)."""
        return [self.seq[q] for q in range(P + 1, min(P + k, self.end) + 1)]


def cond_logits(rec, d, parent_k):
    """Scores of the 16 candidates at position d given the parent's candidate parent_k (-1 = the anchor)."""
    s = rec["cv"][d].astype(np.float64).copy()
    if d == 0:
        s += np.nan_to_num(rec["E"][0][0].astype(np.float64))
    else:
        s += np.nan_to_num(rec["E"][d][parent_k].astype(np.float64))
    return s


def softmax(x, tau=1.0):
    z = (x - x.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def chain_policy(rec, depth=7):
    """The walk: argmax of U + E along the path (DFlash2's greedy chain)."""
    nodes = []  # (parent index, depth position d, candidate k, token)
    parent, pk = -1, -1
    for d in range(depth):
        s = cond_logits(rec, d, pk)
        k = int(np.argmax(s))
        nodes.append((parent, d, k, int(rec["ci"][d][k])))
        parent, pk = len(nodes) - 1, k
    return nodes


def best_first(rec, budget=7, prob=None, max_depth=7):
    """Best-first tree: repeatedly add the frontier node with the largest path probability (product of the
    conditional probabilities prob(rec, d, parent_k) along its path)."""
    nodes, est = [], 0.0
    heap = []
    p0 = prob(rec, 0, -1)
    for k in range(16):
        heapq.heappush(heap, (-p0[k], len(heap), -1, 0, k))
    tie = 1000
    while heap and len(nodes) < budget:
        negp, _, parent, d, k = heapq.heappop(heap)
        nodes.append((parent, d, k, int(rec["ci"][d][k])))
        est += -negp
        me = len(nodes) - 1
        if d + 1 < max_depth:
            pc = prob(rec, d + 1, k)
            for kk in range(16):
                tie += 1
                heapq.heappush(heap, (negp * pc[kk], tie, me, d + 1, kk))
    return nodes, est


def accepted(nodes, cont):
    """Depth of the deepest node whose root path equals cont (tokens after the anchor)."""
    cur, depth = -1, 0
    while depth < len(cont):
        nxt = None
        for i, (par, d, k, tok) in enumerate(nodes):
            if par == cur and tok == cont[depth]:
                nxt = i
                break
        if nxt is None:
            break
        cur, depth = nxt, depth + 1
    return depth


def replay(run, policy, max_tok=None):
    """Decode the recording with `policy` (rec -> nodes): (cycles, tokens) from the first anchor to the end."""
    P, cycles, toks = run.P0, 0, 0
    lim = run.end - 8 if max_tok is None else min(run.end - 8, run.P0 + max_tok)
    while P <= lim and P in run.by_P:
        rec = run.recs[run.by_P[P]]
        nodes = policy(rec)
        a = accepted(nodes, run.cont(P, 7))
        cycles += 1
        toks += a + 1
        P += a + 1
    return cycles, toks


def per_position(run, policy):
    """Mean tokens a cycle would emit, over every recorded anchor (no cycle structure)."""
    tot, n = 0, 0
    for r in run.recs:
        P = int(r["P"])
        if P + 7 > run.end:
            break
        tot += accepted(policy(r), run.cont(P, 7)) + 1
        n += 1
    return tot / max(n, 1)


def calib_rows(run, tau=1.0):
    """(depth, raw conditional probability at temperature tau, hit) of every candidate whose parent lies on the
    output path: what a calibration of node probabilities is fitted to."""
    out = []
    for r in run.recs:
        P = int(r["P"])
        cont = run.cont(P, 7)
        pk = -1
        for d in range(min(7, len(cont))):
            p = softmax(cond_logits(r, d, pk), tau)
            hit = r["ci"][d] == cont[d]
            for k in range(16):
                out.append((d, p[k], 1.0 if hit[k] else 0.0))
            w = np.where(hit)[0]
            if len(w) == 0:
                break
            pk = int(w[0])
    return np.array(out, dtype=np.float64)


CAL_BINS = 32  # calibration table: per depth, the hit rate at raw probabilities 0, 1/32, .., 1 (linear in between)


def fit_calib(rows, prior=4.0):
    """Per depth, a monotone (pool-adjacent-violators) table of hit rates over CAL_BINS + 1 knots of the raw
    probability, each knot's estimate shrunk toward the raw probability itself with `prior` pseudo-counts."""
    tab = np.zeros((7, CAL_BINS + 1))
    knots = np.linspace(0, 1, CAL_BINS + 1)
    for d in range(7):
        s = rows[rows[:, 0] == d]
        if len(s) == 0:
            tab[d] = knots
            continue
        # soft assignment of each sample to its two neighbouring knots
        x = s[:, 1] * CAL_BINS
        i0 = np.clip(np.floor(x).astype(int), 0, CAL_BINS - 1)
        w1 = x - i0
        num = np.zeros(CAL_BINS + 1)
        den = np.zeros(CAL_BINS + 1)
        np.add.at(num, i0, (1 - w1) * s[:, 2])
        np.add.at(den, i0, 1 - w1)
        np.add.at(num, i0 + 1, w1 * s[:, 2])
        np.add.at(den, i0 + 1, w1)
        est = (num + prior * knots) / (den + prior)
        wt = den + prior
        # pool adjacent violators (non-decreasing)
        vals, wts, cnt = [], [], []
        for v, w in zip(est, wt):
            vals.append(v); wts.append(w); cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
                vals[-1] = (vals[-1] * wts[-1] + v2 * w2) / (wts[-1] + w2)
                wts[-1] += w2
                cnt[-1] += c2
        tab[d] = np.repeat(vals, cnt)
    return tab


def calib_prob(tab, tau=1.0):
    def prob(rec, d, pk):
        p = softmax(cond_logits(rec, d, pk), tau)
        x = p * CAL_BINS
        i0 = np.clip(np.floor(x).astype(int), 0, CAL_BINS - 1)
        w1 = x - i0
        return tab[d][i0] * (1 - w1) + tab[d][i0 + 1] * w1
    return prob


class OnlineCalib:
    """A per-position temperature learner (an alternative the engine does not use; KfCalib is the shared one): per
    children position p,
    a log-likelihood over a temperature grid (tau_g = 2^((g - 8) / 4), g = 0..16: 0.25 .. 4) and the coverage (how
    often the target's token is among the 16 candidates), learned from the rows on each cycle's accepted path (the
    anchor's and every accepted node's: the target's token after a correct prefix), all 16 candidates labeled by that
    token; exponential forgetting over a window of W samples per position. Node probability = coverage_p *
    softmax(scores / tau_p)[k]; tau_p = the grid's maximum of the log-likelihood minus a weak prior (log2 tau)^2 / 2;
    before any data, tau 1 and coverage 1 (the drafter's own probabilities)."""

    G = 17

    def __init__(self, window=256.0, prior=0.5):
        self.lam = 1.0 - 1.0 / window
        self.prior = prior
        self.taus = np.array([2.0 ** ((g - 8) / 4.0) for g in range(self.G)])
        self.ll = np.zeros((7, self.G))
        self.hit = np.zeros(7)
        self.tot = np.zeros(7)

    def tau(self, p):
        score = self.ll[p] - self.prior * np.log2(self.taus) ** 2
        return self.taus[int(np.argmax(score))]

    def cover(self, p):
        return (self.hit[p] + 1.0) / (self.tot[p] + 1.0)

    def prob(self, rec, p, pk):
        return self.cover(p) * softmax(cond_logits(rec, p, pk), self.tau(p))

    def learn(self, rec, p, pk, y):
        """one row: children position p under candidate pk (-1: the anchor), target token y"""
        self.ll[p] *= self.lam
        self.hit[p] *= self.lam
        self.tot[p] *= self.lam
        self.tot[p] += 1.0
        k = np.where(rec["ci"][p] == y)[0]
        if len(k) == 0:
            return
        self.hit[p] += 1.0
        s = cond_logits(rec, p, pk)
        for g, t in enumerate(self.taus):
            z = (s - s.max()) / t
            self.ll[p][g] += z[k[0]] - np.log(np.exp(z).sum())


class KfCalib:
    """The shared KFlash calibration (src/model/kflash_budget.cpp, KfBudget): the probability that the target accepts a
    node of depth d whose drafter probability is q is sigmoid(alpha_d + beta_d logit(q)), fitted online per depth
    (L2-regularized logistic regression toward alpha 0, beta 1, Newton steps on the last 2048 records, every 16 cycles
    once a depth holds 32 records) from the records of every cycle: each tree node whose parent is on the accepted path,
    and whether it was accepted. Before any records p = q (the drafter's own probabilities)."""

    WIN, EVERY, MIN_REC, ITERS = 2048, 16, 32, 4

    def __init__(self):
        self.alpha = np.zeros(8)
        self.beta = np.ones(8)
        self.x = [[] for _ in range(8)]
        self.y = [[] for _ in range(8)]
        self.head = [0] * 8
        self.since = 0

    @staticmethod
    def logit_q(lq):
        q = min(np.exp(lq), 0.999999)
        return lq - np.log1p(-q)

    @staticmethod
    def fit(x, y, a, b, iters):
        lam = 1.0
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        for _ in range(iters):
            z = a + b * x
            p = 1.0 / (1.0 + np.exp(-z))
            w = p * (1.0 - p)
            r = p - y
            ga, gb = lam * a + r.sum(), lam * (b - 1.0) + (r * x).sum()
            haa, hab, hbb = lam + w.sum(), (w * x).sum(), lam + (w * x * x).sum()
            det = haa * hbb - hab * hab
            if not det > 1e-12:
                break
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
            a = float(np.clip(a - da, -8.0, 8.0))
            b = float(np.clip(b - db, 0.1, 4.0))
            if abs(da) + abs(db) < 1e-4:
                break
        return a, b

    def prob(self, rec, p, pk):
        """calibrated acceptance probabilities of the 16 candidates at children position p (depth p + 1)"""
        s = cond_logits(rec, p, pk)
        lq = np.log(np.maximum(softmax(s, 1.0), 1e-30))
        z = self.alpha[p + 1] + self.beta[p + 1] * np.array([self.logit_q(v) for v in lq])
        return 1.0 / (1.0 + np.exp(-z))

    def observe(self, records):
        """records: [(depth 1..7, log q, hit)] of one cycle"""
        for d, lq, hit in records:
            if not 1 <= d <= 7:
                continue
            x = self.logit_q(lq)
            if len(self.x[d]) < self.WIN:
                self.x[d].append(x)
                self.y[d].append(hit)
            else:
                self.x[d][self.head[d]] = x
                self.y[d][self.head[d]] = hit
                self.head[d] = (self.head[d] + 1) % self.WIN
        self.since += 1
        if self.since >= self.EVERY:
            self.since = 0
            for d in range(1, 8):
                if len(self.x[d]) >= self.MIN_REC:
                    self.alpha[d], self.beta[d] = self.fit(self.x[d], self.y[d], self.alpha[d], self.beta[d], self.ITERS)


def replay_kf(run, calib, budget=7, max_tok=None, sizes=None, cost=None):
    """replay() with the shared calibration (KfCalib): each cycle's tree from the calibration so far; the cycle's records
    (every tree node whose parent is on the accepted path; hit = on it) teach it. sizes / cost: the verify's allowed row
    counts and their cost (ms): the rows used are the n with the most expected tokens per ms, E(n) = 1 + the summed
    calibrated path probabilities of the first n - 1 nodes (default: all budget + 1 rows)."""
    P, cycles, toks = run.P0, 0, 0
    lim = run.end - 8 if max_tok is None else min(run.end - 8, run.P0 + max_tok)
    ns = []
    while P <= lim and P in run.by_P:
        rec = run.recs[run.by_P[P]]
        nodes, _ = best_first(rec, budget, calib.prob)
        # path probabilities in insertion order (best-first: a prefix is the best tree of its size)
        n = len(nodes) + 1
        if sizes is not None:
            pp = []
            for i, (par, d, k, tok) in enumerate(nodes):
                pk = nodes[par][2] if par >= 0 else -1
                pr = calib.prob(rec, d, pk if par >= 0 else -1)[k]
                pp.append((pp[par] if par >= 0 else 1.0) * pr)
            E = 1.0 + np.cumsum([0.0] + pp)
            best, n = -1.0, None
            for m in sizes:
                if m <= len(nodes) + 1 and E[m - 1] / cost[m] > best:
                    best, n = E[m - 1] / cost[m], m
            if n is None:
                n = min(sizes)
        use = nodes[: n - 1]
        cont = run.cont(P, 8)
        a = accepted(use, cont[:7])
        # records: nodes whose parent is on the accepted path
        onpath = {-1}
        cur = -1
        for j in range(a):
            for i, (par, d, k, tok) in enumerate(use):
                if par == cur and tok == cont[j]:
                    onpath.add(i)
                    cur = i
                    break
        records = []
        for i, (par, d, k, tok) in enumerate(use):
            if par in onpath:
                pk = use[par][2] if par >= 0 else -1
                s = cond_logits(rec, d, pk if par >= 0 else -1)
                lq = float(np.log(max(softmax(s, 1.0)[k], 1e-30)))
                records.append((d + 1, lq, 1 if i in onpath else 0))
        calib.observe(records)
        cycles += 1
        toks += a + 1
        ns.append(n)
        P += a + 1
    return cycles, toks


def replay_online(run, calib, budget=7, max_tok=None):
    """replay() with the online calibration: each cycle's tree from the calibration so far, then the cycle's accepted
    path rows teach it"""
    P, cycles, toks = run.P0, 0, 0
    lim = run.end - 8 if max_tok is None else min(run.end - 8, run.P0 + max_tok)
    while P <= lim and P in run.by_P:
        rec = run.recs[run.by_P[P]]
        nodes, _ = best_first(rec, budget, calib.prob)
        cont = run.cont(P, 8)
        a = accepted(nodes, cont[:7])
        # rows on the accepted path: the anchor (children position 0) and the accepted nodes at depths 1..a
        pk = -1
        for j in range(min(a, 6) + 1):
            if j < len(cont):
                calib.learn(rec, j, pk, cont[j])
            if j < a:
                w = np.where(rec["ci"][j] == cont[j])[0]
                pk = int(w[0])
        cycles += 1
        toks += a + 1
        P += a + 1
    return cycles, toks


def lattice_oracle(rec, cont):
    """How deep the output stays inside the top-16 lattice (the best any tree over this lattice can do)."""
    d = 0
    while d < min(7, len(cont)) and cont[d] in rec["ci"][d]:
        d += 1
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lat", nargs="+", help="recordings of one model (calibration is fitted on the others: leave one out)")
    ap.add_argument("--tau", type=float, help="the tree's score temperature (default: DFlash2's, 1.3 / v1 1.2)")
    ap.add_argument("--slope", type=float, help="its growth per position (default: DFlash2's, 0.1 / v1 0)")
    ap.add_argument("--taus", default="", help="more flat temperatures to replay, comma-separated")
    ap.add_argument("--cal-tau", type=float, default=1.0, help="temperature of the raw probabilities the table maps")
    ap.add_argument("--budget", type=int, default=7)
    ap.add_argument("--table", help="write the calibration table fitted on all recordings (C++ initializer) here")
    a = ap.parse_args()
    taus = [float(x) for x in a.taus.split(",") if x]
    runs = [Run(p) for p in a.lat]
    rows = [calib_rows(r, a.cal_tau) for r in runs]
    tot = {}

    def add(key, c, t):
        tc = tot.setdefault(key, [0, 0])
        tc[0] += c
        tc[1] += t

    for i, run in enumerate(runs):
        nrec = len(run.recs)
        walk_ok = sum(int(all(chain_policy(r)[d][2] == r["wk"][d] for d in range(7))) for r in run.recs)
        c, t = replay(run, chain_policy)
        add("chain", c, t)
        line = [f"{run.path.split('/')[-1]}: {nrec} pos (n={run.info['n']}, T={run.info['T']}), walk ok {walk_ok}/{nrec}; "
                f"chain {t / c:.3f} tok/cycle ({c} cycles)"]
        # the engine's tree (DFlash2::set_tree): tau (1 + slope d) at position d
        tau = a.tau if a.tau is not None else (1.2 if run.info["v1"] else 1.3)
        slope = a.slope if a.slope is not None else (0.0 if run.info["v1"] else 0.1)
        pol = lambda r: best_first(r, a.budget, lambda rr, d, pk: softmax(cond_logits(rr, d, pk), tau * (1 + slope * d)))[0]
        c2, t2 = replay(run, pol)
        add(f"tree", c2, t2)
        line.append(f"tree (tau {tau} + {slope:.0%}/pos) {t2 / c2:.3f} ({100 * (t2 / c2) / (t / c) - 100:+.1f}%)")
        for tau in taus:
            pol = lambda r, tau=tau: best_first(r, a.budget, lambda rr, d, pk: softmax(cond_logits(rr, d, pk), tau))[0]
            c2, t2 = replay(run, pol)
            add(f"tree tau {tau}", c2, t2)
            line.append(f"tree tau {tau} {t2 / c2:.3f} ({100 * (t2 / c2) / (t / c) - 100:+.1f}%)")
        if len(runs) > 1:
            tab = fit_calib(np.concatenate([rows[j] for j in range(len(runs)) if j != i]))
            prob = calib_prob(tab, a.cal_tau)
            c2, t2 = replay(run, lambda r: best_first(r, a.budget, prob)[0])
            add("tree calibrated (LOO)", c2, t2)
            line.append(f"calibrated (LOO) {t2 / c2:.3f} ({100 * (t2 / c2) / (t / c) - 100:+.1f}%)")
        orc = []
        P = run.P0
        while P <= run.end - 8 and P in run.by_P:
            d = lattice_oracle(run.recs[run.by_P[P]], run.cont(P, 7))
            orc.append(d + 1)
            P += d + 1
        add("lattice oracle", len(orc), sum(orc))
        line.append(f"lattice oracle {np.mean(orc):.3f}")
        print("; ".join(line))
    if len(runs) > 1:
        base = tot["chain"][1] / tot["chain"][0]
        print("all: " + "; ".join(f"{k} {v[1] / v[0]:.3f} ({100 * (v[1] / v[0]) / base - 100:+.1f}%, {v[0]} cycles)"
                                  for k, v in tot.items()))
    if a.table:
        tab = fit_calib(np.concatenate(rows))
        with open(a.table, "w") as f:
            f.write("{" + ",\n ".join("{" + ", ".join(f"{v:.4f}f" for v in row) + "}" for row in tab) + "}\n")


if __name__ == "__main__":
    main()
