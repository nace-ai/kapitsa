#!/usr/bin/env python3
"""Tables and output comparison for tool_loop_ttft.py results.

usage: compare.py <results.jsonl> [--ref TAG_PREFIX] [tag prefixes...]

Prints, per tag, every conversation's tool-call count, the new query's TTFT / prompt / cached tokens and
the tool-loop requests' TTFTs; then compares the replies (every request's reasoning, content and tool
calls) of conversations with the same nonce across tags against the reference tag prefix.
"""
import json, sys, statistics
from collections import defaultdict

path = sys.argv[1]
args = sys.argv[2:]
ref = None
if "--ref" in args:
    i = args.index("--ref")
    ref = args[i + 1]
    del args[i:i + 2]
prefixes = args
recs = [json.loads(l) for l in open(path) if l.strip()]
convs = [r for r in recs if r.get("kind") == "conversation" and (not prefixes or any(r["tag"].startswith(p) for p in prefixes))]

by_tag = defaultdict(list)
for r in convs:
    by_tag[r["tag"]].append(r)
for tag, rs in by_tag.items():
    print(f"== {tag}")
    for r in rs:
        loop = [s for s in r["steps"] if s["kind"] == "tool"]
        q2 = (f"q2 ttft {r['q2_ttft']:7.3f} s  prompt {r['q2_prompt']:>6} cached {r['q2_cached']:>6}"
              if r.get("q2_ttft") is not None else "no new query")
        tc = f"  calls left in reasoning {r['think_calls']}" if r.get("think_calls") else ""
        print(f"  size {r['size']:>5} calls {r['calls']}/{r['calls_asked']}  {q2}  | q1 {r['steps'][0]['ttft']:.3f} s, "
              "loop " + " ".join(f"{s['ttft']:.3f}({s['cached']})" for s in loop) + tc)

# same conversation (nonce) across tags: replies must be identical
groups = defaultdict(dict)
for r in convs:
    groups[(r["size"], r["calls_asked"], r["nonce"])][r["tag"]] = r
diffs = same = 0
for key, tags in sorted(groups.items()):
    if len(tags) < 2:
        continue
    names = sorted(tags)
    base = next((t for t in names if ref and t.startswith(ref)), names[0])
    for t in names:
        if t == base:
            continue
        a, b = tags[base]["texts"], tags[t]["texts"]
        if a == b:
            same += 1
            continue
        diffs += 1
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                for f in ("reasoning", "content", "tool_calls"):
                    if x[f] != y[f]:
                        xs, ys = json.dumps(x[f]), json.dumps(y[f])
                        k = next((j for j in range(min(len(xs), len(ys))) if xs[j] != ys[j]), min(len(xs), len(ys)))
                        print(f"DIFF {key} {base} vs {t}: request {i} {f} differs at char {k}: "
                              f"{xs[max(0, k - 40):k + 60]!r} vs {ys[max(0, k - 40):k + 60]!r}")
                break
        if len(a) != len(b):
            print(f"DIFF {key} {base} vs {t}: {len(a)} vs {len(b)} requests")
print(f"replies compared: {same} identical, {diffs} different")

# medians per (tag group, size, calls)
med = defaultdict(list)
for r in convs:
    if r.get("q2_ttft") is not None:
        med[(r["tag"].rsplit("-r", 1)[0], r["size"], r["calls"])].append(r["q2_ttft"])
print("median q2 TTFT (s) per tag group / size / tool calls:")
for k in sorted(med):
    v = med[k]
    print(f"  {k[0]:<28} {k[1]:>5} {k[2]}  {statistics.median(v):.3f}  ({', '.join(f'{x:.3f}' for x in v)})")
