#!/usr/bin/env python3
"""gtexts.py BASE MODE OUT [N] [TOKENS]: bench/batch/concurrency.py's greedy plan (N requests, TOKENS each), one at a
time (alone) or all at once (together); every reply's text, token count and finish go to OUT (JSON)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "batch"))
import concurrency as c  # noqa: E402

base, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 300
c.warm(base)
c.evict(base)
plan = c.greedy_plan(n, tokens)
t0 = time.perf_counter()
if mode == "alone":
    outs = [c.stream(base, b, t0) for _, b in plan]
else:
    outs = c.run_threads([lambda b=b: c.stream(base, b, t0) for _, b in plan])
res = {name: {"text": o["text"], "tokens": c.ntok(o), "finish": o["finish"], "error": o["error"]}
       for (name, _), o in zip(plan, outs)}
json.dump(res, open(out, "w"), indent=1)
span = max(o["done"] for o in outs) - min(o["sent"] for o in outs)
print(json.dumps({"mode": mode, "n": n, "tokens": [c.ntok(o) for o in outs], "span_s": round(span, 2),
                  "errors": [o["error"] for o in outs if o["error"]]}))
