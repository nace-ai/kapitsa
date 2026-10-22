#!/usr/bin/env python3
"""tools/logit_compare.py and tools/logit_floor.py's gate on synthetic dumps: the numbers (KL, flips and their
margins, the NLL change, percentiles) against a direct computation, records matched by their prefix tokens in any
order, the gate's verdicts and exit status, refusals of dumps that cannot be compared, and the standard library and
numpy giving the same numbers. CPU only, stdlib only (numpy too when it is installed).

usage: test_logit_compare.py
"""
import array, json, math, os, random, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
SCRIPT = os.path.join(TOOLS, "logit_compare.py")
sys.path.insert(0, TOOLS)
import logit_compare as lc  # noqa: E402
import logit_floor as lf  # noqa: E402

FAILED = []
V = 300  # vocabulary


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


def write_dump(d, seqs, recs, rows, meta=None):
    """A dump: sequences (lists of ids), records (sequence, prefix length), rows (lists of V floats, one per record)."""
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "sequences.txt"), "w") as f:
        f.write("".join(" ".join(map(str, s)) + "\n" for s in seqs))
    with open(os.path.join(d, "records.txt"), "w") as f:
        f.write("".join(f"{s} {n}\n" for s, n in recs))
    a = array.array("f", [x for r in rows for x in r])
    if sys.byteorder == "big":
        a.byteswap()
    with open(os.path.join(d, "logits.f32"), "wb") as f:
        f.write(a.tobytes())
    if meta is not None:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f)
    return d


def f32(x):
    return array.array("f", [x])[0]


def fsrc(kl):
    """A floor source's summary with this mean KL (the measures logit_floor.py keeps)."""
    return {"records": 192, "bitwise": 0, "top1": 1.0, "flips": 0, "kl_mean": kl, "kl_median": kl / 10,
            "kl_p99": 10 * kl, "kl_max": 20 * kl, "margin_change_max": 0.1, "tie_margin": 0.2,
            "nll_change_mean": 0.0, "max_abs_diff": 0.1}


# A stand-in for build/logit_dump (its arguments and its dump layout): 3 windows of --rows rows (default 24), the
# prefill path one row a window; deterministic, so a repeat is bitwise and so are chunks; step, NI_QMT=0 and
# NI_MATH_SAFE=1 each move the rows by their own noise (NI_QMT=0's step the most), so every source differs.
FAKE_DUMP = r'''#!/usr/bin/env python3
import array, json, os, random, sys
a = sys.argv[1:]
out, paths = a[1], a[a.index("--paths") + 1].split(",")
rows = int(a[a.index("--rows") + 1]) if "--rows" in a else 24
env = {k: os.environ.get(k, "") for k in ("NI_QMT", "NI_MATH_SAFE")}
V = 50
for p in ["prefill"] + paths:
    d = os.path.join(out, p)
    os.makedirs(d, exist_ok=True)
    n = 1 if p == "prefill" else rows
    seqs = [[(w * 7 + i) % V for i in range(rows + 8)] for w in range(3)]
    recs = [(w, 4 + i) for w in range(3) for i in range(n)]
    data = []
    for w, k in recs:
        r = random.Random(w * 1000 + k)
        row = [r.gauss(0, 2) for _ in range(V)]
        for name, amp in [("step" if p == "step" else "", 0.3 if env["NI_QMT"] == "0" else 0.1),
                          ("qmt0" if env["NI_QMT"] == "0" else "", 0.05), ("safe" if env["NI_MATH_SAFE"] else "", 0.04)]:
            if name:
                q = random.Random(f"{name} {w} {k}")
                row = [x + amp * q.gauss(0, 1) for x in row]
        data += row
    with open(os.path.join(d, "sequences.txt"), "w") as f:
        f.write("".join(" ".join(map(str, s)) + "\n" for s in seqs))
    with open(os.path.join(d, "records.txt"), "w") as f:
        f.write("".join(f"{s} {k}\n" for s, k in recs))
    with open(os.path.join(d, "logits.f32"), "wb") as f:
        f.write(array.array("f", data).tobytes())
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"model": "fm", "path": p, "build": "test", "mac": "Apple M5 Pro", "macos": "26", "date": "now",
                   "records": len(recs), "source": "fake", "env": {k: v for k, v in env.items() if v},
                   "ane": {"evals": 0}, "verify": "chain"}, f)
'''


def run(*args):
    p = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def kl_direct(a, b):
    """KL(softmax a || softmax b), term by term in fp64 (the definition, no shortcuts)."""
    ma, mb = max(a), max(b)
    za = math.fsum(math.exp(x - ma) for x in a)
    zb = math.fsum(math.exp(x - mb) for x in b)
    return math.fsum(math.exp(x - ma) / za * ((x - ma - math.log(za)) - (y - mb - math.log(zb))) for x, y in zip(a, b))


rnd = random.Random(7)
# 3 sequences of 40 tokens; 12 records each (prefixes 20 .. 31)
seqs = [[rnd.randrange(V) for _ in range(40)] for _ in range(3)]
recs = [(s, n) for s in range(3) for n in range(20, 32)]
ref = []
for _ in recs:  # a confident top-1 (margin >= 2) over noise
    r = [f32(rnd.gauss(0, 1.5)) for _ in range(V)]
    r[rnd.randrange(V)] = f32(max(r) + 2 + rnd.random())
    ref.append(r)
NEAR = 5  # record 5: its top-2 a near-tie (margin 0.002)
t1 = max(range(V), key=ref[NEAR].__getitem__)
t2 = max((i for i in range(V) if i != t1), key=ref[NEAR].__getitem__)
ref[NEAR][t2] = f32(ref[NEAR][t1] - 0.002)

with tempfile.TemporaryDirectory() as tmp:
    meta = {"model": "test-model-without-floor", "build": "test", "path": "verify"}
    R = write_dump(os.path.join(tmp, "ref"), seqs, recs, ref, meta)

    # ---- identical ----
    rc, out = run(R, write_dump(os.path.join(tmp, "same"), seqs, recs, ref, meta))
    check(rc == 0 and "every row bitwise identical" in out and "no noise floor" in out,
          "identical dumps: bitwise, and no floor for the model means a report only", out)

    # ---- rounding noise: no flips, small KL; the numbers against a direct computation ----
    noisy = [[f32(x + rnd.uniform(-1e-3, 1e-3)) for x in r] for r in ref]
    noisy[NEAR][t1], noisy[NEAR][t2] = f32(ref[NEAR][t1]), f32(ref[NEAR][t2])  # the near-tie stays unflipped here
    N = write_dump(os.path.join(tmp, "noisy"), seqs, recs, noisy, meta)
    s = lc.compare(lc.Dump(R), lc.Dump(N), near_tie=0.01)
    kls = sorted(kl_direct(a, b) for a, b in zip(ref, noisy))
    check(s["records"] == 36 and s["bitwise"] == 0 and s["flips"] == 0 and s["top1"] == 1.0 and s["tie_margin"] == 0,
          "noise: no flips, no margin halved", s)
    check(abs(s["kl_mean"] - math.fsum(kls) / len(kls)) <= 1e-12 + 1e-9 * s["kl_mean"]
          and abs(s["kl_max"] - kls[-1]) <= 1e-12 + 1e-9 * kls[-1], "KL mean and max as defined (fp64)",
          (s["kl_mean"], math.fsum(kls) / len(kls)))
    p99 = kls[34] + (kls[35] - kls[34]) * 0.65  # numpy.percentile's linear rule: rank 0.99 x 35 = 34.65
    check(abs(s["kl_p99"] - p99) <= 1e-12 + 1e-9 * p99 and abs(s["kl_median"] - (kls[17] + kls[18]) / 2) < 1e-12,
          "p99 and median by linear interpolation", (s["kl_p99"], p99))
    md = max(abs(x - y) for a, b in zip(ref, noisy) for x, y in zip(a, b))
    check(abs(s["max_abs_diff"] - md) < 1e-12, "max |logit diff|", (s["max_abs_diff"], md))

    # ---- the NLL change: the next token's logit lowered by 0.5 in every row ----
    nxt = [seqs[q][n] for q, n in recs]
    lower = [list(r) for r in ref]
    for r, t in zip(lower, nxt):
        r[t] = f32(r[t] - 0.5)
    s = lc.compare(lc.Dump(R), lc.Dump(write_dump(os.path.join(tmp, "lower"), seqs, recs, lower, meta)))

    def nll(r, t):
        m = max(r)
        return m + math.log(math.fsum(math.exp(x - m) for x in r)) - r[t]
    want = math.fsum(nll(b, t) - nll(a, t) for a, b, t in zip(ref, lower, nxt)) / len(nxt)
    check(want > 0.4 and abs(s["nll_change_mean"] - want) < 1e-9, "NLL change (candidate minus reference)",
          (s["nll_change_mean"], want))

    # ---- flips: a near-tie, then a confident one ----
    flip = [list(r) for r in noisy]
    flip[NEAR][t1], flip[NEAR][t2] = f32(ref[NEAR][t2]), f32(ref[NEAR][t1])
    F = write_dump(os.path.join(tmp, "flip"), seqs, recs, flip, meta)
    rc, out = run(R, F, "--near-tie", "0.01", "--max-mean-kl", "1")
    check(rc == 0 and "1 flip, 1 of them near-ties" in out and "PASS (flags)" in out, "a near-tie flip passes", out)
    rc, out = run(R, F, "--near-tie", "0.001")
    check(rc == 1 and "1 flips at a reference margin of 0.001 or more" in out, "under a smaller bound it does not", out)
    s = lc.compare(lc.Dump(R), lc.Dump(F), near_tie=0.01)
    check(s["flips"] == 1 and abs(s["flip_list"][0]["margin"] - (ref[NEAR][t1] - ref[NEAR][t2])) < 1e-9
          and s["flip_list"][0]["record"] == "sequence 0 prefix 25", "the flip's record and reference margin", s["flip_list"])
    check(abs(s["tie_margin"] - s["flip_list"][0]["margin"]) < 1e-12, "tie_margin: the flip's margin", s["tie_margin"])
    halved = [list(r) for r in ref]
    u3 = max(range(V), key=ref[3].__getitem__)
    k2 = max((i for i in range(V) if i != u3), key=ref[3].__getitem__)  # record 3's runner-up, halfway up to its top
    halved[3][k2] = f32((ref[3][u3] + ref[3][k2]) / 2 + 1e-3)
    s = lc.compare(lc.Dump(R), lc.Dump(write_dump(os.path.join(tmp, "halved"), seqs, recs, halved, meta)))
    check(s["flips"] == 0 and abs(s["tie_margin"] - (ref[3][u3] - ref[3][k2])) < 1e-5,
          "tie_margin: a margin moved by half of it", (s["tie_margin"], ref[3][u3] - ref[3][k2]))
    rc, out = run(R, F, "--min-top1", "0.99")
    check(rc == 1 and "top-1 agreement 0.9722 < 0.9900" in out, "--min-top1", out)
    far = [list(r) for r in noisy]
    k = 9
    u = max(range(V), key=ref[k].__getitem__)
    far[k][(u + 1) % V] = f32(ref[k][u] + 3)  # another token jumps 3 logits over the confident top-1
    rc, out = run(R, write_dump(os.path.join(tmp, "far"), seqs, recs, far, meta), "--near-tie", "0.01")
    check(rc == 1 and "FAIL: 1 flips at a reference margin" in out, "a confident flip fails", out)

    # ---- KL bounds ----
    rc, out = run(R, N, "--max-mean-kl", "1e-12")
    check(rc == 1 and "FAIL: mean KL" in out, "--max-mean-kl", out)
    rc, out = run(R, N, "--max-p99-kl", "1e-12", "--no-gate")
    check(rc == 0 and "FAIL" not in out, "--no-gate only reports", out)

    # ---- a floor file, found by the model's name or given ----
    floor_json = os.path.join(tmp, "floor.json")
    with open(floor_json, "w") as f:
        json.dump({"gate": {"min_top1": 0.9, "max_mean_kl": 1.0, "max_p99_kl": 1.0, "near_tie": 0.01,
                            "max_far_flips": 0}}, f)
    rc, out = run(R, F, "--floor", floor_json)
    check(rc == 0 and "PASS (" in out, "--floor FILE", out)
    rc, out = run(R, F, "--floor", floor_json, "--max-mean-kl", "1e-12")
    check(rc == 1 and "FAIL: mean KL" in out, "a flag overrides the floor's bound", out)

    # ---- records in another order, sequences on other lines: matched by their prefix tokens ----
    order = list(range(len(recs)))
    rnd.shuffle(order)
    perm = [2, 0, 1]  # sequence line s moves to perm[s]
    seqs2 = [None] * 3
    for q in range(3):
        seqs2[perm[q]] = seqs[q]
    recs2 = [(perm[recs[i][0]], recs[i][1]) for i in order]
    S = write_dump(os.path.join(tmp, "shuffled"), seqs2, recs2, [flip[i] for i in order], meta)
    s2 = lc.compare(lc.Dump(R), lc.Dump(S), near_tie=0.01)
    s1 = lc.compare(lc.Dump(R), lc.Dump(F), near_tie=0.01)
    check(all(s1[k] == s2[k] for k in ("records", "bitwise", "flips", "kl_mean", "kl_p99", "kl_max", "nll_change_mean")),
          "the same comparison whatever the order", (s1, s2))
    rc, out = run(S, R, "--no-gate")
    check(rc == 0 and "36 records" in out, "and the other way round", out)

    # ---- dumps that cannot be compared ----
    rc, out = run(R, write_dump(os.path.join(tmp, "short"), seqs, recs[:30], ref[:30], meta))
    check(rc == 2 and "lacks 6 of the reference's 36 prefixes" in out, "a candidate without some prefixes", out)
    rc, out = run(R, write_dump(os.path.join(tmp, "vocab"), seqs, recs, [r[:-1] for r in ref], meta))
    check(rc == 2 and "different vocabularies" in out, "another vocabulary", out)
    bad = write_dump(os.path.join(tmp, "trunc"), seqs, recs, ref, meta)
    with open(os.path.join(bad, "logits.f32"), "ab") as f:
        f.write(b"\0\0\0\0")
    rc, out = run(R, bad)
    check(rc == 2 and "one row per record" in out, "a logits file that is not one row per record", out)
    rc, out = run(R, write_dump(os.path.join(tmp, "badrec"), seqs, [(0, 40)], [ref[0]], meta))
    check(rc == 2 and "no proper prefix" in out, "a record past its sequence", out)
    nan = [list(r) for r in ref]
    nan[3][7] = float("nan")
    rc, out = run(R, write_dump(os.path.join(tmp, "nan"), seqs, recs, nan, meta), "--max-mean-kl", "1")
    check(rc == 1 and "non-finite logits" in out, "non-finite logits fail", out)

    # ---- logit_dump's layout: one dump per path; every path both hold ----
    for side, rows_v in (("A", ref), ("B", flip)):
        write_dump(os.path.join(tmp, side, "verify"), seqs, recs, rows_v, meta)
        write_dump(os.path.join(tmp, side, "step"), seqs, recs, ref, dict(meta, path="step"))
    os.makedirs(os.path.join(tmp, "A", "batch"))  # not a dump: ignored
    rc, out = run(os.path.join(tmp, "A"), os.path.join(tmp, "B"), "--near-tie", "0.001",
                  "--json", os.path.join(tmp, "out.json"))
    with open(os.path.join(tmp, "out.json")) as f:
        js = json.load(f)
    check(rc == 1 and out.index("step:") < out.index("verify:") and "every row bitwise identical" in out,
          "a directory of paths: each compared, any failure fails", out)
    check(set(js["paths"]) == {"step", "verify"} and js["paths"]["verify"]["failures"]
          and not js["paths"]["step"]["failures"] and js["paths"]["verify"]["gate"]["near_tie"] == 0.001, "--json", js)

    # ---- raw fp32 rows (nh_eval --vcheck), matched by index ----
    rc, out = run(os.path.join(R, "logits.f32"), os.path.join(N, "logits.f32"), "--vocab", str(V), "--no-gate")
    check(rc == 0 and "36 records" in out and "top-1 agreement 1.0000" in out, "raw files with --vocab", out)
    rc, out = run(os.path.join(R, "logits.f32"), os.path.join(N, "logits.f32"), "--vocab", "7")
    check(rc == 2, "raw files of another row size", out)

    # ---- the standard library and numpy: the same numbers ----
    if lc.np is not None:
        np_mod = lc.np
        a = lc.compare(lc.Dump(R), lc.Dump(F), near_tie=0.01)
        lc.np = None
        b = lc.compare(lc.Dump(R), lc.Dump(F), near_tie=0.01)
        lc.np = np_mod
        same = all(abs(a[k] - b[k]) <= 1e-12 + 1e-9 * abs(a[k]) for k in
                   ("kl_mean", "kl_median", "kl_p99", "kl_max", "margin_change_max", "nll_change_mean", "max_abs_diff"))
        check(same and a["flips"] == b["flips"] and a["bitwise"] == b["bitwise"], "numpy and the standard library agree",
              (a, b))
    else:
        print("(numpy not installed: the standard library's numbers only)")

    # ---- --against: twice one source of the floor ----
    def src(**kw):  # a source's summary in a floor file
        base = {"records": 192, "bitwise": 0, "top1": 1.0, "flips": 0, "kl_mean": 1e-5, "kl_median": 1e-6,
                "kl_p99": 1e-4, "kl_max": 2e-4, "margin_change_max": 0.01, "tie_margin": 0.005,
                "nll_change_mean": 0.0, "max_abs_diff": 0.1}
        base.update(kw)
        return base
    check(lc.up3(2.0004e-5) == 2.01e-5 and lc.up3(4e-5) == 4e-5 and lc.up3(0) == 0, "bounds round up", lc.up3(2.0004e-5))
    with open(floor_json, "w") as f:
        json.dump({"sources": {"batch": src(kl_mean=1e-9, kl_p99=1e-9, margin_change_max=0.5),
                               "ane": src(kl_mean=1.0, kl_p99=1.0, margin_change_max=10)},
                   "gate": {"max_mean_kl": 2.0, "max_p99_kl": 2.0, "near_tie": 20, "max_far_flips": 0}}, f)
    rc, out = run(R, N, "--floor", floor_json)
    check(rc == 0, "the floor's own gate", out)
    rc, out = run(R, N, "--floor", floor_json, "--against", "batch")
    check(rc == 1 and "FAIL: mean KL" in out and "twice its batch source" in out, "--against a stricter source", out)
    rc, out = run(R, N, "--floor", floor_json, "--against", "chunk")
    check(rc == 2 and "no source chunk" in out, "--against a source the floor lacks", out)

    # ---- the prefill path decides: bitwise the same, the decode paths are held to the decode passes' gate ----
    with open(floor_json, "w") as f:
        json.dump({"gate": {"max_mean_kl": 1.0, "max_p99_kl": 1.0, "near_tie": 5, "max_far_flips": 0},
                   "decode_gate": {"max_mean_kl": 1e-12, "max_p99_kl": 1.0, "near_tie": 5, "max_far_flips": 0}}, f)
    pre_recs = [(q, 20) for q in range(3)]
    for side, pre_rows in (("P", ref[:3]), ("Q", ref[:3]), ("Q2", noisy[:3])):
        write_dump(os.path.join(tmp, side, "prefill"), seqs, pre_recs, pre_rows, dict(meta, path="prefill"))
        write_dump(os.path.join(tmp, side, "verify"), seqs, recs, ref if side == "P" else noisy, meta)
    rc, out = run(os.path.join(tmp, "P"), os.path.join(tmp, "Q"), "--floor", floor_json)
    check(rc == 1 and "FAIL: mean KL" in out and "decode passes' gate" in out and out.index("prefill:") < out.index("verify:"),
          "the same prefill: the decode paths held to the decode gate", out)
    rc, out = run(os.path.join(tmp, "P"), os.path.join(tmp, "Q2"), "--floor", floor_json)
    check(rc == 0 and "decode passes' gate" not in out, "another prefill: the whole floor's gate", out)
    rc, out = run(os.path.join(tmp, "P"), os.path.join(tmp, "Q"), "--floor", floor_json, "--max-mean-kl", "1")
    check(rc == 0, "a flag overrides the decode gate's bound too", out)

    # ---- logit_floor.py: the floor is the worst source in it, the gate twice that ----
    sources = {
        "repeat": {"in_floor": True, "paths": {"verify": src(bitwise=192, kl_mean=0, kl_p99=0, kl_max=0,
                                                             margin_change_max=0, tie_margin=0)}},
        "batch": dict(src(kl_mean=2e-5, flips=1, top1=1 - 1 / 192), in_floor=True),
        "chunk": dict(src(kl_p99=3e-4, margin_change_max=0.04, tie_margin=0.04), in_floor=True),
        "step": dict(src(kl_mean=1.0, kl_p99=1.0, flips=50), in_floor=False),
    }
    fl, gate = lf.floor_and_gate(sources)
    check(fl["kl_mean"] == 2e-5 and fl["set_by"]["kl_mean"] == "batch" and fl["kl_p99"] == 3e-4
          and fl["set_by"]["kl_p99"] == "chunk" and fl["flips"] == 1, "the floor: the worst source in it", fl)
    check(gate == {"min_top1": 0.989583, "max_mean_kl": 4e-5, "max_p99_kl": 6e-4, "near_tie": 0.04, "max_far_flips": 0},
          "the gate: twice the floor", gate)
    sources["batch"]["flips"] = 0
    check(lf.floor_and_gate(sources)[1]["min_top1"] == 0.989583, "a floor without flips still allows two in 192")
    sources["repeat"]["paths"]["prefill"] = src(records=8, flips=8, kl_mean=9.0)  # (a check, not a measure)
    check(lf.floor_and_gate(sources)[0]["kl_mean"] == 2e-5, "the prefill path's rows are not in the floor")
    mixed = dict(sources, step=dict(src(records=768, flips=6, kl_mean=1e-3), in_floor=True))
    mfl, mgate = lf.floor_and_gate(mixed)
    check(mfl["flips"] == 6 and mfl["records"] == 768 and mgate["min_top1"] == 0.984375,
          "a source of more rows: its flips a share of its own rows", (mfl, mgate))
    sources["batch"]["decode"] = sources["repeat"]["decode"] = True
    dfl, dgate = lf.floor_and_gate(sources, "decode")
    check(dfl["kl_p99"] == 1e-4 and dgate["max_p99_kl"] == 2e-4, "the decode floor: its own sources alone", dfl)

    # ---- --reorder: the order-change gate, for a change that only reassociates the K reduction ----
    with open(floor_json, "w") as f:
        json.dump({"gate": {"min_top1": 0.999, "max_mean_kl": 1e-9, "max_p99_kl": 4e-6, "near_tie": 1e-4,
                            "max_far_flips": 0}}, f)
    rc, out = run(R, N, "--floor", floor_json)
    check(rc == 1, "a floor a reordering cannot pass", out)
    rc, out = run(R, N, "--reorder")
    check(rc == 0 and "order-change gate" in out and "PASS (the order-change gate" in out,
          "--reorder passes rounding noise the floor refuses", out)
    rc, out = run(R, N, "--reorder", "--no-gate")
    check(rc == 2 and "--reorder sets its own bounds" in out, "--reorder takes no --no-gate", out)
    rc, out = run(R, N, "--reorder", "--floor", floor_json)
    check(rc == 2, "--reorder takes no --floor", out)
    rc, out = run(R, F, "--reorder", "--min-top1", "0.9", "--max-nll-change", "1")
    check(rc == 0 and "1 flip, 1 of them near-ties" in out, "--reorder: a near-tie flip inside its margin", out)
    # a drift of the size an expert GEMM reordering causes passes; a bigger one does not
    reordered = [[f32(x + rnd.gauss(0, 0.02)) for x in r] for r in ref]
    rc, out = run(R, write_dump(os.path.join(tmp, "reordered"), seqs, recs, reordered, meta), "--reorder")
    check(rc == 0, "--reorder: drift of the order a reordering causes", out)
    drifted = [[f32(x + rnd.gauss(0, 0.25)) for x in r] for r in ref]
    rc, out = run(R, write_dump(os.path.join(tmp, "drifted"), seqs, recs, drifted, meta), "--reorder")
    check(rc == 1 and "FAIL: mean KL" in out, "--reorder: a drift beyond it fails", out)
    rc, out = run(R, write_dump(os.path.join(tmp, "lower2"), seqs, recs, lower, meta), "--reorder")
    check(rc == 1 and "FAIL: mean NLL change" in out, "--reorder: the next token's NLL is bounded", out)

    # ---- the floor per chip: an M5 or later takes <model>.m<N>.json when there is one, else <model>.json ----
    m4 = {"model": "fm", "mac": "MacBook Pro (16-inch, 2024), M4 Pro (20-core GPU), 48 GB"}
    m5 = {"model": "fm", "mac": "MacBook Pro (14-inch, 2026), M5 Pro (20-core GPU), 48 GB"}
    check(lc.chip_gen(m4) == 4 and lc.chip_gen(m5) == 5 and lc.chip_gen({"mac": "?"}) is None, "chip_gen from meta's mac")
    saved, lc.FLOORS = lc.FLOORS, os.path.join(tmp, "floors")
    try:
        os.makedirs(lc.FLOORS)
        base, own = os.path.join(lc.FLOORS, "fm.json"), os.path.join(lc.FLOORS, "fm.m5.json")
        check(lc.floor_name("fm", m4) == base and lc.floor_name("fm", m5) == own and lc.floor_name("fm") == base,
              "floor_name: M5 its own file, M1-M4 and unknown the model's")
        open(base, "w").close()
        D4, D5 = type("D", (), {"meta": m4}), type("D", (), {"meta": m5})
        check(lc.floor_file(D5, D5, None) == base, "an M5 without its own floor falls back to the model's")
        open(own, "w").close()
        check(lc.floor_file(D5, D5, None) == own and lc.floor_file(D4, D4, None) == base,
              "an M5 takes its own floor, an M4 the model's")
        check(lc.floor_file(D5, D5, "given.json") == "given.json", "--floor wins")
        # --reorder: the model's own "reorder_gate" (here one no noise passes) instead of REORDER_GATE
        R5 = write_dump(os.path.join(tmp, "r5"), seqs, recs, ref, m5)
        N5 = write_dump(os.path.join(tmp, "n5"), seqs, recs, reordered, m5)
        with open(own, "w") as f:
            json.dump({"gate": {}, "reorder_gate": {"max_mean_kl": 1e-12, "max_median_kl": 1e-12}}, f)
        check(lc.main([R5, N5, "--reorder"]) == 1, "--reorder takes the model floor's own reorder_gate")
        with open(own, "w") as f:
            json.dump({"gate": {}}, f)
        check(lc.main([R5, N5, "--reorder"]) == 0, "--reorder without one: REORDER_GATE")
        s = lc.compare(lc.Dump(R5), lc.Dump(N5))
        check(lc.gate(s, {"max_median_kl": s["kl_median"] / 2}) and not lc.gate(s, {"max_median_kl": s["kl_median"]}),
              "max_median_kl bounds the median KL")
    finally:
        lc.FLOORS = saved

    # ---- logit_floor.py end to end on a stand-in logit_dump (CPU): --rows, --also, --reorder, --rescore ----
    fake = os.path.join(tmp, "fake_dump.py")
    with open(fake, "w") as f:
        f.write(FAKE_DUMP)
    os.chmod(fake, 0o755)
    mdir = os.path.join(tmp, "fm")
    os.makedirs(mdir)
    with open(os.path.join(mdir, "config.json"), "w") as f:
        json.dump({"model_type": "flint"}, f)
    out = os.path.join(tmp, "fm-floor.json")
    p = subprocess.run([sys.executable, os.path.join(TOOLS, "logit_floor.py"), mdir, "--bin", fake, "--work",
                        os.path.join(tmp, "fw"), "--out", out, "--rows", "6", "--also", "NI_QMT=0", "--reorder"],
                       capture_output=True, text=True)
    check(p.returncode == 0, "logit_floor.py --rows --also --reorder runs", p.stdout[-800:] + p.stderr[-800:])
    if p.returncode == 0:
        with open(out) as f:
            fl = json.load(f)
        names = set(fl["sources"])
        check({"repeat", "chunk", "step", "repeat NI_QMT=0", "chunk NI_QMT=0", "step NI_QMT=0"} <= names,
              "--also: every source again with the path's env", names)
        check(fl["measured"]["records"] == 18 and "--rows 6" in fl["measured"]["command"]
              and "--reorder" in fl["measured"]["command"], "--rows reaches logit_dump and the file", fl["measured"])
        check(fl["floor"]["set_by"]["kl_mean"] == "step NI_QMT=0", "the floor: the worst of both paths",
              fl["floor"]["set_by"])
        check(set(fl["reorder_sources"]) == {"NI_QMT=0", "NI_QMT=0 NI_MATH_SAFE=1"}
              and fl["reorder_gate"]["max_median_kl"] > 0 and "max_nll_change" in fl["reorder_gate"],
              "--reorder: the default against each --also path, fast and safe math", fl.get("reorder_gate"))
        rg = dict(fl["reorder_gate"])
        fl["reorder_gate"] = {"max_mean_kl": 1.0}
        with open(out, "w") as f:
            json.dump(fl, f)
        p = subprocess.run([sys.executable, os.path.join(TOOLS, "logit_floor.py"), mdir, "--rescore", "--out", out],
                           capture_output=True, text=True)
        with open(out) as f:
            check(p.returncode == 0 and json.load(f)["reorder_gate"] == rg, "--rescore derives the reorder gate again")
    lc_saved, lf_mac = lc.FLOORS, lf.this_mac
    try:  # --rescore without --out: this Mac's file
        lc.FLOORS = os.path.join(tmp, "floors2")
        os.makedirs(lc.FLOORS)
        for name in ("fm.json", "fm.m5.json"):
            with open(os.path.join(lc.FLOORS, name), "w") as f:
                json.dump({"model": "fm", "sources": {"step": fsrc(1e-3 if name == "fm.json" else 2e-3)}}, f)
        lf.this_mac = lambda: {"mac": "Apple M5 Pro"}
        old_argv, sys.argv = sys.argv, ["logit_floor.py", mdir.replace("fm", "x") + "/fm", "--rescore"]
        try:
            lf.main()
        finally:
            sys.argv = old_argv
        with open(os.path.join(lc.FLOORS, "fm.m5.json")) as f5, open(os.path.join(lc.FLOORS, "fm.json")) as f4:
            check("gate" in json.load(f5) and "gate" not in json.load(f4), "--rescore on an M5: its own file")
    finally:
        lc.FLOORS, lf.this_mac = lc_saved, lf_mac

print("\nALL PASSED" if not FAILED else f"\nFAIL: {len(FAILED)} failed: {FAILED}")
sys.exit(1 if FAILED else 0)
