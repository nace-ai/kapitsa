#!/usr/bin/env python3
"""Did a change move the model's outputs beyond noise? Compares two logit dumps and gates the result.

  logit_compare.py REFERENCE CANDIDATE [--floor FILE] [--against SOURCE] [--no-gate] [--min-top1 X]
                   [--max-mean-kl X] [--max-p99-kl X] [--near-tie X] [--max-far-flips N] [--flips N] [--json FILE]
                   [--vocab V]

REFERENCE and CANDIDATE are dumps of build/logit_dump, or of another engine that writes the same layout: a directory
holding sequences.txt (token ids, one sequence per line), records.txt (per scored prefix its sequence's line and
length) and logits.f32 (per record the vocabulary's fp32 logits, little-endian), meta.json optional; or a directory of
such directories (logit_dump's one per path: every path both hold is compared); or, with --vocab V, raw files of fp32
rows [rows][V] (nh_eval --vcheck, ni_run --dump-all-logits), whose rows are matched by index.

Records are matched by their prefix tokens, so the dumps may list them in any order; the candidate must hold every
prefix of the reference. Per record, in fp64 over the whole vocabulary: the KL divergence of the candidate's softmax
from the reference's, whether their top-1 tokens agree (a flip otherwise: its margin is the reference's logit of its
own top-1 minus of the candidate's), how much the reference's top-2 margin changed, the change in the negative
log-likelihood of the token the sequence goes on with (candidate minus reference, nats), the largest |logit
difference| and whether the row is bitwise the same. Printed: bitwise rows, top-1 agreement and its flips (how many
were near-ties), KL mean / median / p99 / max, the mean NLL change.

The gate (exit status 1 past it): top-1 agreement at least --min-top1, mean and p99 KL at most --max-mean-kl and
--max-p99-kl, and at most --max-far-flips flips (default 0) whose margin is --near-tie or more (near-ties: margins
that twice the floor's own differences can flip, compare()'s tie_margin). The bounds come from the model's noise
floor on this Mac, tools/logit_floors/<model>.json (meta.json names the model; tools/logit_floor.py measures it), or
--floor FILE: its "gate", twice the worst of the conditions the engine legitimately runs a request under; but when
the dumps' prefill paths are bitwise the same (a change confined to the passes after the prefill), the decode paths
are held to its "decode_gate", twice what moves their rows when only the prefill stays the same (alone against
batched). The prefill path itself (one row per window) is reported, not gated: a change to the prefill moves the
decode rows after it, which are. --against SOURCE takes twice one source of the floor instead; a flag overrides its
own bound; --no-gate only reports. Non-finite logits always fail. Exit status 2: the dumps cannot be compared. numpy
when it is installed, else the standard library (the same numbers, slower).
"""
import argparse
import array
import itertools
import json
import math
import operator
import os
import sys

try:
    import numpy as np
except ImportError:  # the standard library computes the same numbers, ~50x slower
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
FLOORS = os.path.join(HERE, "logit_floors")
GATE_KEYS = ("min_top1", "max_mean_kl", "max_p99_kl", "near_tie", "max_far_flips")


class Dump:
    """A dump's records (key -> row index and the token the sequence goes on with), its rows and meta.json."""

    def __init__(self, path, vocab=None):
        self.path = path
        self.meta = {}
        if os.path.isdir(path):
            self.logits = os.path.join(path, "logits.f32")
            with open(os.path.join(path, "sequences.txt")) as f:
                seqs = [line.split() for line in f.read().splitlines()]
            with open(os.path.join(path, "records.txt")) as f:
                recs = [tuple(int(x) for x in line.split()) for line in f.read().splitlines() if line.strip()]
            if not recs:
                raise ValueError(f"{path}: no records")
            self.keys, self.nexts, self.labels = [], [], []
            for i, r in enumerate(recs):
                if len(r) != 2 or not 0 <= r[0] < len(seqs) or not 0 < r[1] < len(seqs[r[0]]):
                    raise ValueError(f"{path}: record {i} {r} is no proper prefix of a sequence")
                s, n = r
                self.keys.append((s, n))  # resolved to tokens only when two dumps' lines differ (match())
                self.nexts.append(int(seqs[s][n]))
                self.labels.append(f"sequence {s} prefix {n}")
            self.seqs = seqs
            size = os.path.getsize(self.logits)
            if size % (4 * len(recs)):
                raise ValueError(f"{path}: logits.f32 ({size} bytes) does not hold one row per record ({len(recs)})")
            self.vocab = size // (4 * len(recs))
            meta = os.path.join(path, "meta.json")
            if os.path.exists(meta):
                with open(meta) as f:
                    self.meta = json.load(f)
        else:
            if not vocab:
                raise ValueError(f"{path}: a raw file needs --vocab")
            self.logits, self.vocab, self.seqs = path, vocab, None
            size = os.path.getsize(path)
            if size % (4 * vocab):
                raise ValueError(f"{path}: {size} bytes are not rows of {vocab} fp32 values")
            n = size // (4 * vocab)
            self.keys = [("row", i) for i in range(n)]
            self.nexts = [None] * n
            self.labels = [f"row {i}" for i in range(n)]
        self.rows = len(self.keys)
        self._f = None
        self._mm = None

    def token_key(self, i):
        s, n = self.keys[i]
        return tuple(self.seqs[s][:n]) if self.seqs is not None else self.keys[i]

    def row(self, i):
        """Row i: (raw bytes, values) -- a float64 numpy array, or an array('f') without numpy."""
        if np is not None:
            if self._mm is None:
                self._mm = np.memmap(self.logits, dtype="<f4", mode="r", shape=(self.rows, self.vocab))
            r = self._mm[i]
            return r.tobytes(), r.astype(np.float64)
        if self._f is None:
            self._f = open(self.logits, "rb")
        self._f.seek(i * self.vocab * 4)
        raw = self._f.read(self.vocab * 4)
        a = array.array("f")
        a.frombytes(raw)
        if sys.byteorder == "big":
            a.byteswap()
        return raw, a


def match(ref, cand):
    """Pairs (reference row, candidate row) for every reference record."""
    if ref.vocab != cand.vocab:
        raise ValueError(f"the dumps have different vocabularies ({ref.vocab} and {cand.vocab})")
    if ref.seqs is None or cand.seqs is None:
        if ref.rows > cand.rows:
            raise ValueError(f"the candidate has {cand.rows} rows, the reference {ref.rows}")
        return [(i, i) for i in range(ref.rows)]
    if ref.seqs == cand.seqs and ref.keys == cand.keys:  # the same files (logit_dump --from, or the same windows)
        return [(i, i) for i in range(ref.rows)]
    index = {cand.token_key(j): j for j in range(cand.rows)}
    pairs, missing = [], 0
    for i in range(ref.rows):
        j = index.get(ref.token_key(i))
        if j is None:
            missing += 1
        else:
            pairs.append((i, j))
    if missing:
        raise ValueError(f"the candidate lacks {missing} of the reference's {ref.rows} prefixes")
    return pairs


def row_metrics(ra, rb, nxt):
    """One record's numbers: kl, top-1 of each, the reference's top-1 margin over the candidate's top-1, the change of
    the reference's top-2 margin, the NLL change of token nxt, max |diff|, bitwise; None for kl if not finite."""
    (raw_a, a), (raw_b, b) = ra, rb
    out = {"bitwise": raw_a == raw_b}
    if np is not None:
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return dict(out, kl=None)
        ta, tb = int(np.argmax(a)), int(np.argmax(b))
        a2 = a.copy()
        a2[ta] = -np.inf
        t2 = int(np.argmax(a2))
        lse_a = a[ta] + math.log(float(np.exp(a - a[ta]).sum()))
        lse_b = b[tb] + math.log(float(np.exp(b - b[tb]).sum()))
        c = lse_a - lse_b
        d = a - b
        kl = float(np.dot(np.exp(a - lse_a), d - c))
        maxdiff = float(np.abs(d).max())
    else:
        if not all(map(math.isfinite, itertools.chain(a, b))):
            return dict(out, kl=None)
        ma, mb = max(a), max(b)
        ta, tb = a.index(ma), b.index(mb)
        a2 = array.array("f", a)
        a2[ta] = -math.inf
        t2 = a2.index(max(a2))
        lse_a = ma + math.log(math.fsum(map(math.exp, map(operator.sub, a, itertools.repeat(ma)))))
        lse_b = mb + math.log(math.fsum(map(math.exp, map(operator.sub, b, itertools.repeat(mb)))))
        c = lse_a - lse_b
        d = list(map(operator.sub, a, b))
        p = map(math.exp, map(operator.sub, a, itertools.repeat(lse_a)))
        kl = math.fsum(map(operator.mul, p, map(operator.sub, d, itertools.repeat(c))))
        maxdiff = max(map(abs, d))
    out.update(kl=kl, top_ref=ta, top_cand=tb, maxdiff=maxdiff,
               margin=float(a[ta] - a[tb]),
               margin2=float(a[ta] - a[t2]),
               margin_change=float((b[ta] - b[t2]) - (a[ta] - a[t2])),
               nll_change=None if nxt is None else float((a[nxt] - b[nxt]) - c))
    return out


def quantile(sorted_values, q):
    """numpy.percentile's default (linear interpolation between the closest ranks)."""
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def compare(ref, cand, near_tie=None):
    """The comparison of two Dumps, summarized; flips are listed with their margins. tie_margin: the largest top-2
    margin of the reference that the candidate moved by half of it or more (a flip's margin included): twice such a
    difference flips margins up to it, which makes it the near-tie bound of a gate at twice this comparison."""
    pairs = match(ref, cand)
    kls, nll, flips, bad = [], [], [], []
    bitwise, maxdiff, dmargin, tie = 0, 0.0, 0.0, 0.0
    for i, j in pairs:
        m = row_metrics(ref.row(i), cand.row(j), ref.nexts[i])
        bitwise += m["bitwise"]
        if m["kl"] is None:
            bad.append(ref.labels[i])
            continue
        kls.append(m["kl"])
        maxdiff = max(maxdiff, m["maxdiff"])
        dmargin = max(dmargin, abs(m["margin_change"]))
        if 2 * abs(m["margin_change"]) >= m["margin2"]:
            tie = max(tie, m["margin2"])
        if m["nll_change"] is not None:
            nll.append(m["nll_change"])
        if m["top_ref"] != m["top_cand"]:
            tie = max(tie, m["margin"])
            flips.append({"record": ref.labels[i], "ref_top": m["top_ref"], "cand_top": m["top_cand"],
                          "margin": m["margin"]})
    kls.sort()
    n = len(pairs)
    s = {
        "records": n,
        "bitwise": bitwise,
        "non_finite": bad,
        "top1": (n - len(flips) - len(bad)) / n,
        "flips": len(flips),
        "kl_mean": math.fsum(kls) / len(kls) if kls else 0.0,
        "kl_median": quantile(kls, 0.5),
        "kl_p99": quantile(kls, 0.99),
        "kl_max": kls[-1] if kls else 0.0,
        "margin_change_max": dmargin,
        "tie_margin": tie,
        "nll_change_mean": math.fsum(nll) / len(nll) if nll else None,
        "max_abs_diff": maxdiff,
        "flip_list": sorted(flips, key=lambda f: -f["margin"]),
    }
    if near_tie is not None:
        s["near_tie"] = near_tie
        s["near_tie_flips"] = sum(f["margin"] < near_tie for f in flips)
    return s


def up3(x):
    """x rounded up to 3 significant digits (a bound is never below twice the floor)."""
    if x <= 0:
        return 0.0
    e = math.floor(math.log10(x)) - 2
    return float(f"{math.ceil(round(x / 10 ** e, 9)) * 10 ** e:.3g}")


MEASURES = ("flips", "kl_mean", "kl_p99", "kl_max", "margin_change_max", "tie_margin")


def worst(summaries):
    """The worst of comparison summaries, per measure (a floor's sources, a source's paths)."""
    w = {k: max(s[k] for s in summaries) for k in MEASURES}
    w["records"] = min(s["records"] for s in summaries)
    return w


def gate_from(f):
    """The gate twice a floor f (worst()): at most twice its flips (at least two), each at a margin under its
    tie_margin (what twice its differences can flip), mean and p99 KL at most twice its."""
    return {
        "min_top1": math.floor((1 - 2 * max(f["flips"], 1) / f["records"]) * 1e6) / 1e6,
        "max_mean_kl": up3(2 * f["kl_mean"]),
        "max_p99_kl": up3(2 * f["kl_p99"]),
        "near_tie": up3(f["tie_margin"]),
        "max_far_flips": 0,
    }


def source_gate(floor_json, name):
    """The gate twice one source of a floor file (--against): e.g. batch, for a change to the decode passes alone."""
    src = floor_json["sources"].get(name)
    if src is None:
        raise ValueError(f"the floor has no source {name} (it has {', '.join(floor_json['sources'])})")
    return gate_from(worst(list(src["paths"].values()) if "paths" in src else [src]))


def floor_file(ref, cand, given):
    if given:
        return given
    model = ref.meta.get("model") or cand.meta.get("model")
    if model:
        path = os.path.join(FLOORS, model + ".json")
        if os.path.exists(path):
            return path
    return None


def gate(s, bounds):
    """The failures of summary s against the bounds (None: unchecked)."""
    fails = []
    if s["non_finite"]:
        fails.append(f"{len(s['non_finite'])} records with non-finite logits (first: {s['non_finite'][0]})")
    if bounds.get("min_top1") is not None and s["top1"] < bounds["min_top1"]:
        fails.append(f"top-1 agreement {s['top1']:.4f} < {bounds['min_top1']:.4f}")
    if bounds.get("max_mean_kl") is not None and s["kl_mean"] > bounds["max_mean_kl"]:
        fails.append(f"mean KL {s['kl_mean']:.3e} > {bounds['max_mean_kl']:.3e}")
    if bounds.get("max_p99_kl") is not None and s["kl_p99"] > bounds["max_p99_kl"]:
        fails.append(f"p99 KL {s['kl_p99']:.3e} > {bounds['max_p99_kl']:.3e}")
    if bounds.get("near_tie") is not None:
        far = s["flips"] - s["near_tie_flips"]
        allowed = bounds.get("max_far_flips") or 0
        if far > allowed:
            fails.append(f"{far} flips at a reference margin of {bounds['near_tie']:.4g} or more (allowed {allowed})")
    return fails


def describe(meta):
    if not meta:
        return "no meta.json"
    bits = [meta.get("build", "?")]
    for k in ("verify", "batch", "prefill_chunk", "ane", "env"):
        if k in meta and meta[k] not in ({}, None):
            bits.append(f"{k} {json.dumps(meta[k], separators=(',', ':'))}")
    return ", ".join(bits)


def report(name, ref, cand, s, show_flips):
    print(f"{name}: {s['records']} records ({ref.meta.get('model', '?')})")
    print(f"  reference: {describe(ref.meta)}")
    print(f"  candidate: {describe(cand.meta)}")
    models = (ref.meta.get("model"), cand.meta.get("model"))
    if all(models) and models[0] != models[1]:
        print(f"  warning: dumps of different models ({models[0]} and {models[1]})")
    if s["bitwise"] == s["records"]:
        print("  every row bitwise identical")
        return
    flips = f"{s['flips']} flip{'s' if s['flips'] != 1 else ''}"
    if "near_tie" in s:
        flips += f", {s['near_tie_flips']} of them near-ties (reference margin under {s['near_tie']:.4g})"
    print(f"  bitwise identical rows {s['bitwise']} of {s['records']}; max |logit diff| {s['max_abs_diff']:.4g}")
    print(f"  top-1 agreement {s['top1']:.4f} ({flips})")
    print(f"  KL mean {s['kl_mean']:.3e} median {s['kl_median']:.3e} p99 {s['kl_p99']:.3e} max {s['kl_max']:.3e}; "
          f"top-2 margin change up to {s['margin_change_max']:.4g}")
    if s["nll_change_mean"] is not None:
        print(f"  next-token NLL change {s['nll_change_mean']:+.3e} nats per record (candidate minus reference)")
    for f in s["flip_list"][:show_flips]:
        print(f"    flip at {f['record']}: top-1 {f['ref_top']} -> {f['cand_top']}, reference margin {f['margin']:.4g}")


def pairs_of(ref_path, cand_path, vocab):
    """(name, reference dump, candidate dump) for every path both hold."""
    def is_dump(p):
        return os.path.isfile(os.path.join(p, "records.txt"))
    if os.path.isdir(ref_path) and not is_dump(ref_path):
        if not os.path.isdir(cand_path) or is_dump(cand_path):
            raise ValueError(f"{ref_path} holds dumps per path; {cand_path} does not")
        names = sorted(n for n in os.listdir(ref_path)
                       if is_dump(os.path.join(ref_path, n)) and is_dump(os.path.join(cand_path, n)))
        if not names:
            raise ValueError(f"{ref_path} and {cand_path} hold no path in common")
        return [(n, Dump(os.path.join(ref_path, n)), Dump(os.path.join(cand_path, n))) for n in names]
    ref = Dump(ref_path, vocab)
    return [(ref.meta.get("path", os.path.basename(os.path.normpath(ref_path))), ref, Dump(cand_path, vocab))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--floor", help="the noise floor's JSON (default: tools/logit_floors/<model>.json)")
    ap.add_argument("--against", metavar="SOURCE",
                    help="gate at twice this source of the floor (e.g. batch) instead of the worst of its sources")
    ap.add_argument("--no-gate", action="store_true", help="report only")
    ap.add_argument("--min-top1", type=float)
    ap.add_argument("--max-mean-kl", type=float)
    ap.add_argument("--max-p99-kl", type=float)
    ap.add_argument("--near-tie", type=float, help="a flip is a near-tie under this reference margin (logits)")
    ap.add_argument("--max-far-flips", type=int)
    ap.add_argument("--flips", type=int, default=10, help="flips to list (largest margin first)")
    ap.add_argument("--json", help="also write the summaries and the verdict here")
    ap.add_argument("--vocab", type=int, help="raw fp32 files of rows of this many logits")
    a = ap.parse_args(argv)
    try:
        comparisons = pairs_of(a.reference, a.candidate, a.vocab)
    except (OSError, ValueError) as e:
        print(f"logit_compare: {e}", file=sys.stderr)
        return 2
    # bounds: the floor's gate (or one source's, --against); decode: the decode passes' own gate, for the decode paths
    # of two dumps whose prefill rows are bitwise the same (only the passes after the prefill changed)
    bounds, decode, source = {}, None, None
    if not a.no_gate:
        path = floor_file(comparisons[0][1], comparisons[0][2], a.floor)
        if path:
            with open(path) as f:
                floor = json.load(f)
            try:
                bounds = source_gate(floor, a.against) if a.against else dict(floor["gate"])
            except ValueError as e:
                print(f"logit_compare: {path}: {e}", file=sys.stderr)
                return 2
            source = os.path.relpath(path) + (f", twice its {a.against} source" if a.against else "")
            if not a.against and "decode_gate" in floor:
                decode = dict(floor["decode_gate"])
        elif a.against:
            print("logit_compare: --against needs a noise floor (--floor FILE)", file=sys.stderr)
            return 2
        flags = [k for k in GATE_KEYS if getattr(a, k) is not None]
        for b in (bounds, decode):
            if b is not None:
                b.update((k, getattr(a, k)) for k in flags)
        if flags:
            source = f"{source} and flags" if source else "flags"
    out, failed, prefill_same = {"paths": {}}, False, False
    comparisons.sort(key=lambda c: c[0] != "prefill")  # the prefill first: it decides the decode paths' gate
    for name, ref, cand in comparisons:
        use_decode = decode is not None and prefill_same and name != "prefill"
        # the prefill path (one row per window) only decides the decode paths' gate: a change to the prefill moves
        # every decode row after it, which the decode paths are gated on
        b = {} if name == "prefill" and len(comparisons) > 1 else decode if use_decode else bounds
        try:
            s = compare(ref, cand, b.get("near_tie"))
        except (OSError, ValueError) as e:
            print(f"logit_compare: {name}: {e}", file=sys.stderr)
            return 2
        if name == "prefill":
            prefill_same = s["bitwise"] == s["records"]
        report(name, ref, cand, s, a.flips)
        fails = gate(s, b)
        failed |= bool(fails)
        for f in fails:
            print(f"  FAIL: {f}")
        label = source and source + (": the decode passes' gate (the prefill rows are bitwise the same)"
                                      if use_decode else "")
        if label and b:
            print(f"  {'FAIL' if fails else 'PASS'} ({label})")
        s.update(failures=fails, gate=dict(b, source=label), reference=ref.meta, candidate=cand.meta)
        out["paths"][name] = s
    if not source and not a.no_gate:
        print("no noise floor for this model and no bounds given: reported only (tools/logit_floor.py measures one)")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
