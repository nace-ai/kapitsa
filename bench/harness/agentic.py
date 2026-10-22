#!/usr/bin/env python3
"""Agent-harness metrics for bench.py / compare_engines.py (opt in with bench.py --agentic). Stdlib only.

What a coding agent feels beyond one chat decode, measured through the same streaming /v1/chat/completions
requests, sampling, no-think settings and frozen corpus as the other phases (bench.body_of, build_context,
stream_chat, decode_rates):

  cold_start   first request after the server is ready (lazy kernel/pipeline build), then repeats      -> cold_start()
  sweep        cold prefill TTFT and decode tok/s at 1K .. max context, one request gives both         -> t_sweep()
  sustain      back-to-back decodes for minutes with the thermal / power state                          -> t_sustain()
  cache        warm-prefix TTFT (same prefix, different suffix) and two-conversation interleave         -> t_cache()
  hit_rate     cached / prompt tokens over the agent replay's follow-up turns                           -> hit_rate()

The summary functions (summarize_*, hit_rate, sweep_targets, parse_thermal, ...) are pure so the tests need no model.
"""
import random, re, statistics, subprocess, time

import bench

MAX_CTX = 32768          # context the configs serve (--ctx 32768); the sweep's "max" point is cut to fit it
TOKEN_SAFETY = 1.05      # build_context's 4.25 chars/token is 2-3 % low for the reference tokenizer; engines differ
SWEEP_DECODE = 192       # tokens each sweep request decodes (middle window needs >= ~40)
SWEEP_Q = ("Write a long, detailed technical summary of the repository context above: every file, its purpose "
           "and how the parts fit together. Use at least 600 words and do not stop early.")
SHORT_Q = ["In one sentence, what is the first file in the repository context about?",
           "In one sentence, what is the second file in the repository context about?",
           "In one sentence, name one function that appears in the repository context.",
           "In one sentence, what language is most of the repository context written in?",
           "In one sentence, which file in the repository context looks the most complex?"]
FALLOFF_FLAG_PCT = 10.0  # last-third vs first-third decode drop that is flagged
KEPT_RATIO = 0.5         # interleave: TTFT after the other conversation / cold TTFT below this = prefix kept


# ---------------------------------------------------------------- pure helpers
def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def sweep_targets(spec, max_ctx=MAX_CTX, decode=SWEEP_DECODE):
    """Token targets for build_context from '1000,2000,max'. 'max' (and any size that does not fit) becomes the
    largest target that, with the tokenizer margin and the answer, fits max_ctx; duplicates collapse."""
    cap = int((max_ctx - decode - 128) / TOKEN_SAFETY)
    out = []
    for part in filter(None, (x.strip() for x in spec.split(","))):
        n = cap if part == "max" else min(int(part), cap)
        if n not in out:
            out.append(n)
    return sorted(out)


def hit_rate(turns):
    """Engine-reported cached / prompt tokens over follow-up turns (agent replay turn records). None when the engine
    reported no cached count on any of them (not 0: unknown)."""
    c = p = 0
    seen = False
    for t in turns or []:
        if not t.get("follow_up") or not t.get("prompt_tokens"):
            continue
        if t.get("cached_tokens") is None:
            continue
        seen = True
        c += t["cached_tokens"]
        p += t["prompt_tokens"]
    return round(c / p, 4) if seen and p else None


def summarize_sweep(points):
    """points: per-size records {target, prompt_tokens, ttft_s, decode_tps, error}. Adds decode retention (largest
    point over the smallest, both valid) and the prefill rate curve."""
    ok = [p for p in points if not p.get("error") and p.get("decode_tps")]
    s = {"points": [{k: p.get(k) for k in ("target", "prompt_tokens", "ttft_s", "prefill_tps", "decode_tps", "error")}
                    for p in points]}
    if len(ok) >= 2:
        lo, hi = ok[0], ok[-1]
        s["decode_retention"] = round(hi["decode_tps"] / lo["decode_tps"], 3)
        s["retention_from_tokens"], s["retention_to_tokens"] = lo.get("prompt_tokens"), hi.get("prompt_tokens")
    s["max_prompt_tokens"] = max((p.get("prompt_tokens") or 0 for p in points if not p.get("error")), default=None)
    return s


def summarize_sustain(runs, min_runs=3):
    """runs: [{tps, t_s, speed_limit}]. Falloff = median tps of the first third vs the last third, in %."""
    v = [r for r in runs if r.get("tps")]
    s = {"runs": len(runs), "valid_runs": len(v)}
    if not v:
        return s
    s.update({"first_tps": round(v[0]["tps"], 2), "min_tps": round(min(r["tps"] for r in v), 2),
              "median_tps": round(statistics.median(r["tps"] for r in v), 2),
              "duration_s": round(runs[-1].get("t_s", 0.0), 1)})
    if len(v) >= min_runs:
        k = max(1, len(v) // 3)
        a, b = statistics.median(r["tps"] for r in v[:k]), statistics.median(r["tps"] for r in v[-k:])
        s["first_third_tps"], s["last_third_tps"] = round(a, 2), round(b, 2)
        s["thermal_falloff_pct"] = round(100.0 * (a - b) / a, 1)
    lims = [r["speed_limit"] for r in runs if r.get("speed_limit") is not None]
    s["min_speed_limit"] = min(lims) if lims else None
    return s


def summarize_cache(c):
    """Derived numbers of the cache phase record."""
    cold, warm = c.get("cold_ttft_s"), c.get("warm_ttft_s")
    out = {}
    if cold and warm:
        out["warm_vs_cold_speedup"] = round(cold / warm, 2)
    after = c.get("interleave_ttft_s")
    if cold and after:
        out["interleave_vs_cold"] = round(after / cold, 2)
        out["interleave_prefix_kept"] = (after / cold) < KEPT_RATIO
    return out


def parse_thermal(therm_out, batt_out="", all_out=""):
    """pmset -g therm / -g batt / -g text -> {speed_limit, scheduler_limit, battery, low_power}. Missing -> None."""
    d = dict(re.findall(r"(\w+)\s*=\s*(\d+)", therm_out or ""))
    n = lambda k: int(d[k]) if k in d else None
    return {"speed_limit": n("CPU_Speed_Limit"), "scheduler_limit": n("CPU_Scheduler_Limit"),
            "battery": "Battery Power" in (batt_out or "") if batt_out else None,
            "low_power": bool(re.search(r"lowpowermode\s+1", all_out or "")) if all_out else None}


def thermal_state():
    def pm(*a):
        try:
            return subprocess.run(["pmset", "-g", *a], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return ""
    return parse_thermal(pm("therm"), pm("batt"), pm())


# ---------------------------------------------------------------- requests
def _cold(a, target, question, max_tokens, nonce=None):
    """One cold, cache-busted request with a ~target-token context. -> (record, stream result or None)."""
    rec = {"target": target}
    try:
        ctx = bench.build_context(target)
        nonce = nonce or f"session {random.getrandbits(64):016x}"
        msgs = bench.prefill_messages(ctx, nonce, question)
        r = bench.stream_chat(a.base_url, bench.body_of(a, msgs, max_tokens), session=f"bench-{random.getrandbits(48):012x}")
    except Exception as e:
        rec["error"] = f"{e!r}"[:300]
        return rec, None
    u = r["usage"] or {}
    pt = u.get("prompt_tokens")
    rec.update({"prompt_tokens": pt, "completion_tokens": u.get("completion_tokens"), "ttft_s": r["ttft"],
                "prefill_tps": (pt / r["ttft"]) if pt and r["ttft"] else None, "finish": r["finish"],
                "cached_tokens": bench.cached_of(u, r["timings"])})
    return rec, r


def cold_start(a, repeats=3, target=1000):
    """First request after ready (call before any warm-up), then `repeats` more of the same size with new nonces.
    Penalty = first TTFT - median of the repeats. The file cache is warm (no purge): a daemon-restart number."""
    first, r = _cold(a, target, "Say hello in one sentence.", 8)
    out = {"target": target, "os_cache": "warm", "first": first, "repeats": []}
    for _ in range(repeats):
        rec, _ = _cold(a, target, "Say hello in one sentence.", 8)
        out["repeats"].append(rec)
    m = med([x.get("ttft_s") for x in out["repeats"]])
    out["repeat_median_ttft_s"] = m
    if first.get("ttft_s") is not None and m is not None:
        out["first_request_penalty_s"] = round(first["ttft_s"] - m, 3)
    out["invalid"] = [f"cold start: first request failed: {first['error']}"] if first.get("error") else []
    return out


def t_sweep(a, log=print):
    """Cold prefill TTFT + decode tok/s at each size (median over --sweep-repeats), longest answer 192 tokens."""
    targets = sweep_targets(a.sweep_sizes, a.max_ctx)
    points, invalid, flags = [], [], []
    for tg in targets:
        runs = []
        for _ in range(a.sweep_repeats):
            rec, r = _cold(a, tg, SWEEP_Q, SWEEP_DECODE)
            if r is not None:
                ct = rec.get("completion_tokens")
                mid, e2e, _ = bench.decode_rates(r, ct, a.window_head, a.window_tail)
                rec["decode_tps"] = mid
                for why in [] if a.always_reasons else bench.thinking_leak(r):  # (a model that always reasons: Flint)
                    invalid.append(f"sweep {tg}: {why}")
                if r["finish"] != "length":
                    flags.append(f"sweep {tg}: finish_reason={r['finish']} (expected length)")
                if rec.get("cached_tokens") and rec["cached_tokens"] > bench.COLD_CACHE_MAX:
                    invalid.append(f"sweep {tg}: cold prompt reported {rec['cached_tokens']} cached tokens")
            runs.append(rec)
        good = [x for x in runs if not x.get("error")]
        p = {"target": tg, "runs": runs}
        if good:
            for k in ("prompt_tokens", "ttft_s", "prefill_tps", "decode_tps"):
                p[k] = med([x.get(k) for x in good])
        else:
            p["error"] = runs[-1].get("error", "no result")
            invalid.append(f"sweep {tg}: request failed: {p['error']}"[:300])
        points.append(p)
        log(f"  sweep {tg}: {p.get('prompt_tokens')} tok, TTFT {bench.f2(p.get('ttft_s'))}s, "
            f"{p.get('prefill_tps') and round(p['prefill_tps'])} tok/s prefill, decode {bench.f2(p.get('decode_tps'))} tok/s"
            + (f" ERROR {p['error']}" if p.get("error") else ""))
    res = summarize_sweep(points)
    res.update({"runs": points, "max_ctx": a.max_ctx, "invalid": invalid, "flags": flags})
    return res


def t_sustain(a, log=print):
    """Back-to-back sampled decodes (bench.t_decode: same prompt family, validity and window) for --sustain-s."""
    t0 = time.time()
    st0 = thermal_state()
    runs, invalid, flags = [], [], []
    while True:
        d = bench.t_decode(a)
        th = thermal_state()
        rec = {"tps": d.get("decode_tps"), "t_s": round(time.time() - t0, 1), "speed_limit": th["speed_limit"],
               "ttft_s": d.get("ttft_s"), "finish": d.get("finish")}
        runs.append(rec)
        invalid += [f"sustain run {len(runs)}: {x}" for x in d.get("invalid", [])]
        log(f"  sustain {rec['t_s']:6.1f}s: {bench.f2(rec['tps'])} tok/s, speed limit {rec['speed_limit']}")
        if d.get("invalid") and len(runs) >= 3 and all(not x["tps"] for x in runs[-3:]):
            invalid.append("sustain: 3 failed runs in a row, stopped")
            break
        if time.time() - t0 >= a.sustain_s:
            break
    s = summarize_sustain(runs)
    if (s.get("thermal_falloff_pct") or 0) > FALLOFF_FLAG_PCT:
        flags.append(f"THERMAL: decode fell {s['thermal_falloff_pct']}% from the first to the last third of {s['duration_s']} s")
    if s.get("min_speed_limit") is not None and s["min_speed_limit"] < 100:
        flags.append(f"THROTTLED: pmset CPU_Speed_Limit reached {s['min_speed_limit']}")
    s.update({"run_records": runs, "thermal_start": st0, "thermal_end": thermal_state(), "invalid": invalid, "flags": flags})
    return s


def t_cache(a, log=print):
    """Warm-prefix TTFT and the two-conversation interleave on a shared prefix of --cache-tokens."""
    tg = min(a.cache_tokens, sweep_targets("max", a.max_ctx, 256)[-1])
    ctx_a = ctx_b = bench.build_context(tg)  # the per-conversation nonce right after the system start keeps them apart
    sid = {k: f"bench-{k}-{random.getrandbits(40):010x}" for k in "ab"}
    sysmsg = {"a": f"[session {random.getrandbits(64):016x}] {bench.SYS}\n\n# Repository context\n{ctx_a}",
              "b": f"[session {random.getrandbits(64):016x}] {bench.SYS}\n\n# Repository context\n{ctx_b}"}
    res, invalid, flags = {"target": tg, "turns": [],
                           "server_slots": getattr(a, "server_slots", 0) or 0}, [], []
    slots = res["server_slots"]  # conversations the server keeps live at once: the A-after-B turn depends on it

    def ask(who, i, label, history=None):
        msgs = [{"role": "system", "content": sysmsg[who]}, {"role": "user", "content": SHORT_Q[i % len(SHORT_Q)]}]
        if history:  # a grown conversation: the earlier question and its answer, then a new question
            msgs = [msgs[0]] + history + [msgs[1]]
        # 256 tokens: the answer ends by itself (a turn cut by max_tokens is not reusable in some engines); 1024 for a
        # model that always reasons (Flint), so it gets through its reasoning to the answer, as bench.py's phases do
        try:
            r = bench.stream_chat(a.base_url, bench.body_of(a, msgs, 1024 if a.always_reasons else 256),
                                  session=sid[who])
        except Exception as e:
            invalid.append(f"cache {label}: request failed: {e!r}"[:300])
            return None
        u = r["usage"] or {}
        t = {"label": label, "conversation": who, "prompt_tokens": u.get("prompt_tokens"),
             "cached_tokens": bench.cached_of(u, r["timings"]), "ttft_s": r["ttft"], "finish": r["finish"]}
        for why in [] if a.always_reasons else bench.thinking_leak(r):
            invalid.append(f"cache {label}: {why}")
        t["answer"] = r["content"]
        res["turns"].append(t)
        log(f"  cache {label}: prompt {t['prompt_tokens']} (cached {t['cached_tokens']}), TTFT {bench.f2(t['ttft_s'])}s")
        return t

    cold = ask("a", 0, "A cold")
    # Follow-up: same conversation grown by one turn (what a chat or tool loop sends). Engines that keep only the live
    # state or turn-boundary checkpoints reuse this but not a new question on the shared prefix (the warm turns below).
    fol = None
    if cold and cold.get("answer"):
        time.sleep(a.turn_gap)
        hist = [{"role": "user", "content": SHORT_Q[0]}, {"role": "assistant", "content": cold["answer"]}]
        fol = ask("a", 5, "A follow-up", history=hist)
        if fol and not fol.get("answer"):
            flags.append("cache: the follow-up turn returned no content (cut inside its reasoning?)")
    elif cold:
        flags.append("cache: the cold turn returned no content (cut inside its reasoning?); the follow-up turn was skipped")
    warm = []
    for i in (1, 2, 3):
        time.sleep(a.turn_gap)
        w = ask("a", i, f"A warm {i}")
        if w:
            warm.append(w)
    ask("b", 0, "B cold")
    time.sleep(a.turn_gap)
    # The label says how many conversations the server keeps: with one slot B evicts A, with several A's prefix survives
    back = ask("a", 4, f"A after B (slots {slots})" if slots else "A after B")
    if cold:
        res["cold_ttft_s"] = cold["ttft_s"]
        res["cold_cached_tokens"] = cold["cached_tokens"]
        if cold["cached_tokens"] and cold["cached_tokens"] > bench.COLD_CACHE_MAX:
            invalid.append(f"cache: cold prompt reported {cold['cached_tokens']} cached tokens")
    if fol:
        res["followup_ttft_s"] = fol["ttft_s"]
        res["followup_cached_tokens"] = fol["cached_tokens"]
        res["followup_hit_rate"] = (round(fol["cached_tokens"] / fol["prompt_tokens"], 4)
                                    if fol["cached_tokens"] is not None and fol["prompt_tokens"] else None)
    res["warm_ttft_s"] = med([w["ttft_s"] for w in warm])
    res["warm_cached_tokens"] = med([w["cached_tokens"] for w in warm])
    wp = sum(w["prompt_tokens"] or 0 for w in warm)
    wc = [w["cached_tokens"] for w in warm]
    res["warm_hit_rate"] = round(sum(wc) / wp, 4) if wp and all(c is not None for c in wc) else None
    if back:
        res["interleave_ttft_s"] = back["ttft_s"]
        res["interleave_cached_tokens"] = back["cached_tokens"]
        if slots == 1 and not (back["cached_tokens"] or 0):
            res["interleave_note"] = (
                "A re-prefilled: --slots 1 keeps one conversation live, so B evicted A. The engine's own default "
                "(Kapitsa 4) keeps several; the *-4slots config measures that, and there A hits its prefix")
            flags.append(f"cache: A after B re-prefilled with --slots {slots} (one live conversation, not a prefix loss)")
    for t in res["turns"]:
        t.pop("answer", None)
    res.update(summarize_cache(res))
    res.update({"invalid": invalid, "flags": flags})
    return res
