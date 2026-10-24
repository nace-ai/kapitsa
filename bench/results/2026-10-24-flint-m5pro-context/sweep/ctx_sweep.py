"""ctx_sweep.py: Flint prompt reading and writing speed against the prompt's length, one engine per run.

  ctx_sweep.py --url URL --key KEY --out OUT.json --tok tokenizer.json --src DIR --plan 1024,2048,...
               [--max-tokens 512] [--leg-start EPOCH] [--budget-s 570] [--foreign FILE] [--engine NAME] [--rep-base N]

Each request in --plan is cold: the prompt is a coding agent's context (the repo's bench/corpus.txt, then its src/,
tools/ and docs/ files in path order) cut to the size in Flint tokens, with a random nonce at its very start so no
engine reuses an earlier request's cache. The reply is streamed at Flint's card's sampling (T 1.0, top_p 0.95, top_k
20) up to --max-tokens tokens. Prefill = prompt tokens / time to the first streamed token. Decode = the rate between
the event that completes token 16 and the last event ending at or before token n-4 (bench/harness/bench.py's
decode_rates). Every request's raw event times are stored, so other windows can be recomputed.

A request is started only if the leg (from --leg-start, the server's launch) is projected to stay under --budget-s:
its duration is estimated from the slowest earlier request of the same size in this leg, else from a conservative
default. --foreign is gpu_hogs.py monitor's file; each request records the foreign GPU load in its prefill and decode
windows. Results are rewritten to --out after every request."""
import argparse, glob, json, os, random, subprocess, time, urllib.request
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
for a in ("--url", "--key", "--out", "--tok", "--src", "--plan"):
    ap.add_argument(a, required=True)
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--leg-start", type=float, default=time.time())
ap.add_argument("--budget-s", type=float, default=570)
ap.add_argument("--foreign", default="")
ap.add_argument("--engine", default="")
ap.add_argument("--leg", default="")
a = ap.parse_args()
base = a.url.rstrip("/")
plan = [int(s) for s in a.plan.split(",")]
tok = Tokenizer.from_file(a.tok)
parts = [open(os.path.join(a.src, "bench/corpus.txt")).read()]
for pat in ("src/**/*.cpp", "src/**/*.h", "src/**/*.mm", "src/**/*.metal", "tools/*.py", "docs/*.md"):
    for f in sorted(glob.glob(os.path.join(a.src, pat), recursive=True)):
        parts.append(f"\n### FILE: {os.path.relpath(f, a.src)}\n" + open(f, errors="replace").read())
ids = tok.encode("".join(parts), add_special_tokens=False).ids
print(f"context source: {len(ids)} tokens", flush=True)
assert len(ids) > max(plan), "prompt source shorter than the largest size"
QUESTION = "Walk through this code base file by file and explain in detail what each part does and how they connect."


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(s.split("used = ")[1].split("M")[0])


def foreign(t0, t1):
    """Mean and max foreign GPU %% over the monitor's 2 s samples whose window (epoch-2, epoch] overlaps [t0, t1]."""
    if not a.foreign or not os.path.exists(a.foreign):
        return None
    v = []
    for line in open(a.foreign):
        p = line.split()
        if len(p) >= 2:
            try:
                e, pct = float(p[0]), float(p[1])
            except ValueError:
                continue
            if e > t0 and e - 2.0 < t1:
                v.append((pct, p[2] if len(p) > 2 else ""))
    if not v:
        return {"samples": 0}
    top = max(v)
    return {"samples": len(v), "mean_pct": round(sum(x for x, _ in v) / len(v), 1), "max_pct": top[0],
            "top": top[1]}


def decode_rate(ev_t, ev_c, ct, head=16, tail=4):
    if len(ev_t) < 2 or not ct:
        return None, {}
    if len(ev_t) == ct:
        cum, method = list(range(1, ct + 1)), "events"
    else:
        tot = sum(ev_c) or 1
        acc, cum = 0, []
        for c in ev_c:
            acc += c
            cum.append(acc * ct / tot)
        method = "chars"
    i = next((k for k in range(len(ev_t)) if cum[k] >= head), None)
    js = [k for k in range(len(ev_t)) if cum[k] <= ct - tail]
    j = js[-1] if js else None
    if i is None or j is None or j <= i or ev_t[j] <= ev_t[i]:
        return None, {"rate_method": method}
    return (cum[j] - cum[i]) / (ev_t[j] - ev_t[i]), {"rate_method": method, "window_tokens": round(cum[j] - cum[i], 1),
                                                       "window_s": round(ev_t[j] - ev_t[i], 4)}


DEFAULT_EST = lambda n: n / 350 + a.max_tokens / 30 + 5   # noqa: E731  (slower than any engine measured on this Mac)
results, dur, reps = [], {}, {}
for size in plan:
    est = dur.get(size, DEFAULT_EST(size)) * 1.1
    el = time.time() - a.leg_start
    if el + est > a.budget_s:
        print(f"skip {size}: leg at {el:.0f}s, request estimated {est:.0f}s, budget {a.budget_s:.0f}s", flush=True)
        results.append({"size": size, "skipped": f"budget: leg at {el:.0f}s, estimate {est:.0f}s"})
        continue
    reps[size] = reps.get(size, 0) + 1
    body_ids = ids[: max(0, size - 60)]
    nonce = f"session {random.getrandbits(64):016x}"
    msgs = [{"role": "system", "content": f"[{nonce}] You are a coding assistant working in this repository.\n\n"
                                          + tok.decode(body_ids)},
            {"role": "user", "content": QUESTION}]
    body = {"model": "flint", "stream": True, "max_tokens": a.max_tokens, "temperature": 1.0, "top_p": 0.95,
            "top_k": 20, "messages": msgs, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + a.key})
    sw0 = swap_mb()
    t0 = time.time()
    ev_t, ev_c, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                txt = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                if txt:
                    ev_t.append(time.time() - t0)
                    ev_c.append(len(txt))
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    t1 = time.time()
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or len(ev_t)
    ttft = ev_t[0] if ev_t else None
    dec, info = decode_rate(ev_t, ev_c, ct)
    rec = {"engine": a.engine, "leg": a.leg, "size": size, "rep_in_leg": reps[size], "start_epoch": round(t0, 3),
           "prompt_tokens": pt, "ttft_s": ttft, "prefill_tps": pt / ttft if pt and ttft else None,
           "completion_tokens": ct, "finish_reason": finish, "events": len(ev_t), "decode_tps": dec,
           "decode_tps_e2e": (ct - 1) / (ev_t[-1] - ev_t[0]) if len(ev_t) > 1 and ev_t[-1] > ev_t[0] else None,
           **info, "wall_s": round(t1 - t0, 3), "swap_used_mb": [sw0, swap_mb()],
           "foreign_gpu_prefill": foreign(t0, t0 + (ttft or 0)), "foreign_gpu_decode": foreign(t0 + (ttft or 0), t1),
           "event_t": [round(x, 4) for x in ev_t], "event_chars": ev_c}
    dur[size] = max(dur.get(size, 0), t1 - t0)
    results.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("event_t", "event_chars")}), flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
json.dump(results, open(a.out, "w"), indent=1)
