"""27B verdict: means over valid legs (six4 + the shipping-shape leg), with and without 6-A. Valid = 'ok' in legs.log. B = KQuant, A = standard."""
import json, re, sys, os
R = os.path.dirname(os.path.abspath(__file__))  # this record directory
legs = []  # (dir, tag, arm, status)
for d in ("legs", "legs-void-topup", "legs-shipping-shape"):  # void-topup is void-foreign (in no mean); shipping-shape is the third valid KQuant leg
    p = f"{R}/{d}/legs.log"
    if not os.path.exists(p): continue
    for l in open(p):
        m = re.search(r"\s(\d+-[AB]) rc=\d+ load \S+ (\S+(?:\([^)]*\))?) prefill4k_ms", l)
        if m: legs.append((d, m.group(1), m.group(1)[-1], m.group(2)))
def get(d, tag):
    r = json.load(open(f"{R}/{d}/{tag}.json"))["results"]
    pf = [x for x in r["prefill"] if x["size"] == 4000][0]
    return r["decode"]["median_tps"], r["decode_greedy"]["median_tps"], pf["tps"], pf["ttft_s"] * 1000
def mean(a): return sum(a) / len(a)
def table(excl):
    out = {}
    for arm in "AB":
        L = [(d, t) for d, t, a, s in legs if a == arm and s == "ok" and not (d, t) in excl]
        V = [get(d, t) for d, t in L]
        out[arm] = dict(n=len(L), legs=[(t if d == 'legs' else d.replace('legs-', '') + ':' + t) for d, t in L], samp=mean([v[0] for v in V]), greedy=mean([v[1] for v in V]),
                        pooled=mean([x for v in V for x in v[:2]]), pf=mean([v[2] for v in V]), ttft=mean([v[3] for v in V]))
    return out
for name, excl in (("as run (6-A kept)", set()), ("without 6-A", {("legs", "6-A")})):
    t = table(excl); A, B = t["A"], t["B"]; pc = lambda k: 100 * (B[k] / A[k] - 1)
    print(f"{name}: A legs {A['legs']} B legs {B['legs']}")
    print(f"  decode sampled {B['samp']:.1f} vs {A['samp']:.1f} ({pc('samp'):+.1f}%), greedy {B['greedy']:.1f} vs {A['greedy']:.1f} ({pc('greedy'):+.1f}%), pooled {B['pooled']:.1f} vs {A['pooled']:.1f} ({pc('pooled'):+.1f}%)")
    print(f"  4K prefill {B['pf']:.1f} vs {A['pf']:.1f} tok/s ({pc('pf'):+.1f}%), TTFT {B['ttft']:.0f} vs {A['ttft']:.0f} ms")
    print(f"  rule: decode pooled >= -2%: {'PASS' if pc('pooled') >= -2 else 'FAIL'}; prefill >= -5%: {'PASS' if pc('pf') >= -5 else 'FAIL'}; KL lower (0.0349 vs 0.0544) PASS; GSM8K-100 0.94 vs 0.94 PASS")
