#!/usr/bin/env python3
"""sampled.py BASE MODE OUT PLAN: requests of bench/batch/concurrency.py's kinds, one at a time (alone) or all at once
(together); each reply's text, tokens, finish, time to first token and decode tok/s go to OUT (JSON).
PLAN seeded: the exact plan's 2 seeded T = 0.6 requests (code, 400 tokens, seed 7; 2K-token context, 300, seed 3).
PLAN ctx6k: 2 requests on ~6K-token contexts, 300 tokens each, one greedy and one seeded (T = 0.6, seed 5)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", "batch"))
import concurrency as c  # noqa: E402

base, mode, out, which = sys.argv[1:5]
q = "In one sentence, what is the first file in the repository context about?"
plans = {
    "seeded": [
        ("code_t06_seed7_400", c.body(c.code_messages("E2 77c1"), 400, 0.6, 7)),
        ("ctx2k_t06_seed3", c.body(c.context_messages(2000, "E4 1b7f", "List the classes defined there, briefly.",
                                                      offset=60000), 300, 0.6, 3)),
    ],
    "ctx6k": [
        ("ctx6k_greedy_300", c.body(c.context_messages(6000, "E5 c44d", "Describe what the code above does, file by file.",
                                                       offset=80000), 300)),
        ("ctx6k_t06_seed5_300", c.body(c.context_messages(6000, "E8 5e6b", "Describe what the code above does, file by file.",
                                                          offset=100000), 300, 0.6, 5)),
    ],
}
plan = plans[which]
c.warm(base)
c.evict(base)
t0 = time.perf_counter()
if mode == "alone":
    outs = [c.stream(base, b, t0) for _, b in plan]
else:
    outs = c.run_threads([lambda b=b: c.stream(base, b, t0) for _, b in plan])
res = {name: {"text": o["text"], "tokens": c.ntok(o), "finish": o["finish"], "error": o["error"],
              "prompt_tokens": (o["usage"] or {}).get("prompt_tokens"),
              "ttft": round(o["first"] - o["sent"], 3) if o["first"] else None,
              "decode_tps": round(c.decode_rate(o), 2)} for (name, _), o in zip(plan, outs)}
span = max(o["done"] for o in outs) - min(o["sent"] for o in outs)
tot = sum(c.ntok(o) for o in outs)
res["_run"] = {"mode": mode, "plan": which, "span_s": round(span, 3), "total_tps": round(tot / span, 2)}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (v if k == "_run" else {x: v[x] for x in ("tokens", "prompt_tokens", "ttft", "decode_tps", "error")})
                  for k, v in res.items()}))
