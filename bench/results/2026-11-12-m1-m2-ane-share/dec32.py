#!/usr/bin/env python3
"""#520: greedy 400-token code answers after cold ~32K prompts cut at several corpus offsets (one content sample per
offset), so a decode-speed gap between two packs can be told apart from one prompt's luck.
usage: dec32.py BASE_URL MODEL OUT.json [target_tokens] [offsets,comma,separated]"""
import json, os, random, sys, time

H = os.path.expanduser("~/kapitsa-0.2.35-darwin-arm64/bench/harness")
sys.path.insert(0, H)
import bench  # noqa: E402

base, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
target = int(sys.argv[4]) if len(sys.argv) > 4 else 32000
offsets = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,120000,240000,360000,480000").split(",")]
res = []
for off in offsets:
    ctx = bench.build_context(target, offset=off)
    msgs = bench.prefill_messages(ctx, f"session {random.getrandbits(64):016x}", bench.DECODE_USER)
    body = {"model": model, "messages": msgs, "max_tokens": 400, "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
    t0 = time.time()
    r = bench.stream_chat(base, body, session=f"bench-{random.getrandbits(48):012x}")
    u = r["usage"] or {}
    ct = u.get("completion_tokens")
    mid, e2e, _ = bench.decode_rates(r, ct, 16, 4)
    rec = {"offset": off, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": ct, "finish": r["finish"],
           "ttft_s": r["ttft"], "decode_tps": mid, "cached": bench.cached_of(u, r["timings"]),
           "text_head": (r["content"] or "")[:120], "wall_s": round(time.time() - t0, 1)}
    print(json.dumps({k: rec[k] for k in ("offset", "prompt_tokens", "completion_tokens", "finish", "decode_tps")}), flush=True)
    res.append(rec)
json.dump({"base": base, "model": model, "target": target, "runs": res}, open(out, "w"), indent=1)
