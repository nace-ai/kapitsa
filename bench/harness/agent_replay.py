#!/usr/bin/env python3
"""Replays a recorded coding-agent session against an OpenAI-compatible server and times every turn. Stdlib only.

A trace (agent-traces/*.json, written by tools/agent_record.py) holds every /v1/chat/completions request a coding
agent sent while it solved a task, with the length of each recorded answer. The replay sends those requests in order,
with the recorded messages (never the engine's own answers), so every engine reads exactly the same conversation:
the same system prompt and tools, the same file contents and test output, growing turn by turn the way an agent's
context grows. Each request is streamed, greedy, with thinking off (bench.py's fields, so per-engine overrides
apply), and max_tokens equal to the recorded answer's length: an engine that writes the same answer stops where the
recording stopped, one that stops earlier is recorded as such.

Tool definitions (--tool-mode):
  rendered  (default) the tools go into the system message exactly as the Qwen3.5-family chat template renders them
            (Qwen3.5, Qwen3.6, Qwen3.8: "# Tools ... <tools>{json per tool}</tools> ..." before the system prompt),
            and the request carries no `tools` field. The prompt is token for token the one the template builds
            from the request with tools, the same on every engine, and every generated token streams: engines that
            parse tool calls hold a call back until its closing tag, which would hide when the answer started.
  api       the requests as recorded, with `tools` (and tool_choice): the engine renders the tools itself and parses
            tool calls. Time to first token then includes the whole call on engines that hold calls back.

Measured per turn: time to first token (TTFT: the prefill of whatever the engine could not reuse from its cache,
plus the first token), the writing rate (tokens after the first event over the time from the first to the last
event), and the total time. Per session: engine time (the sum of the turns), wall time (with the gaps the agent
spent running tools, replayed from the recording, at most 5 s each), prefill time (sum of TTFTs), decode time (the
rest), the mean writing rate (all turns' window tokens over all turns' window time), and the median and p90 TTFT of
the follow-up turns (each request that continues the previous one).

usage:
  agent_replay.py --base-url http://127.0.0.1:8091/v1 --model qwen3.6-35b-a3b [--trace agent-traces/X.json]
                  [--label L] [--tool-mode rendered|api] [--gaps recorded|none|SECONDS] [--max-turns N] [--out x.json]
The GPU lock is taken for the replay unless the caller holds it (NI_LOCK_HELD=1 or NACE_GPU_LOCK_HELD=1) or --no-gpu-lock. bench.py
runs the same replay as its "agent" phase (with the server's memory and, under compare_engines.py, the contention
monitor and the quiet-GPU wait).
"""
import argparse, copy, hashlib, json, math, os, random, re, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench, gpu_hogs  # noqa: E402

DEFAULT_TRACE = os.path.join(HERE, "agent-traces", "opencode-ledger.json")
MAX_GAP_S = 5.0          # a recorded gap (the agent running tools) is replayed up to this long
PROMPT_TOL = 0.02        # engine prompt_tokens vs the trace's reference: flag outside ±2%
PROMPT_TRUNCATED = 0.90  # ... and INVALID below 90% (the engine cut the conversation)
MAX_FAILED_IN_A_ROW = 3
DEFAULT_TIMEOUT_S = 2400  # no new turn after 40 minutes: the session is then incomplete (INVALID)

# The Qwen3.5-family chat template's tools block (Qwen3.5, Qwen3.6 and Qwen3.8 render it identically): the system
# turn starts with it and continues with "\n\n" + the system prompt. tojson is json.dumps(ensure_ascii=False).
QWEN_TOOLS_HEAD = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
QWEN_TOOLS_TAIL = (
    "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>")


def render_tools(tools):
    return QWEN_TOOLS_HEAD + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + QWEN_TOOLS_TAIL


def content_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


# ---------------------------------------------------------------- trace
def load_trace(path):
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    t = json.loads(raw)
    if t.get("schema") != 1 or "requests" not in t or "messages" not in t:
        raise SystemExit(f"{path}: not an agent trace (schema 1)")
    t["_path"], t["_sha256"] = path, hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return t


def trace_record(t):
    p = t["_path"]
    rel = os.path.relpath(p, HERE)
    s = t.get("stats") or {}
    return {"path": p if rel.startswith("..") else rel, "sha256": t["_sha256"], "name": t.get("name"),
            "requests": len(t["requests"]), "main_turns": s.get("main_turns"),
            "peak_context_tokens": s.get("peak_context_tokens"), "completion_tokens": s.get("completion_tokens")}


def request_parts(t, rq, mode):
    """(messages, extra body fields) of one recorded request in the given tool mode."""
    msgs = [copy.deepcopy(t["messages"][i]) for i in rq["messages"]]
    tools = t["tool_sets"][rq["tools"]] if rq.get("tools") is not None else None
    extra = {}
    if tools:
        if mode == "rendered":
            block = render_tools(tools)
            if msgs and msgs[0].get("role") == "system":
                text = content_text(msgs[0].get("content")).strip()
                msgs[0] = {"role": "system", "content": block + ("\n\n" + text if text else "")}
            else:
                msgs.insert(0, {"role": "system", "content": block})
        else:
            extra["tools"] = tools
            for k in ("tool_choice", "parallel_tool_calls"):
                if k in (rq.get("settings") or {}):
                    extra[k] = rq["settings"][k]
    return msgs, extra


def recorded_gaps(t):
    """Seconds between the end of request i-1 and the start of request i in the recording (0 for the first)."""
    out, prev_end = [], None
    for rq in t["requests"]:
        start = rq.get("t_start_s") or 0.0
        out.append(0.0 if prev_end is None else max(0.0, start - prev_end))
        prev_end = start + ((rq.get("response") or {}).get("total_s") or 0.0)
    return out


TOOL_CALL_RX = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RX = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def answer_form(content, tool_calls):
    """An answer as (text, [(tool, {argument: value text})]), whether its tool calls came as Qwen's raw
    <tool_call> text (rendered mode) or parsed by the server, so answers can be compared across engines."""
    content = content or ""
    calls = [(m.group(1).strip(), {k.strip(): v for k, v in PARAM_RX.findall(m.group(2))})
             for m in TOOL_CALL_RX.finditer(content)]
    text = TOOL_CALL_RX.sub("", content).strip()
    for tc in tool_calls or []:
        try:
            args = json.loads(tc.get("arguments") or "{}")
        except ValueError:
            args = {"_raw": tc.get("arguments")}
        calls.append((tc.get("name"), {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                                       for k, v in (args.items() if isinstance(args, dict) else [])}))
    return text, calls


def ref_prompt_tokens(rq, mode):
    r = rq.get("ref_prompt_tokens") or {}
    return r.get(mode) if isinstance(r, dict) else None


# ---------------------------------------------------------------- stats
def pct(v, q):
    """Nearest-rank percentile (q in 0..100) of a non-empty list."""
    v = sorted(v)
    return v[max(0, min(len(v) - 1, math.ceil(q / 100 * len(v)) - 1))]


def window(ev_t, ev_c, ct, total):
    """(tokens, seconds, tail_unstreamed) written between the first and the last streamed event of one answer: the
    first event's tokens are not counted (their time is in the TTFT). One token per event when the event count
    matches the reported tokens (with or without the end token), else tokens are spread over events by characters
    (bench.decode_rates). A stream that carries fewer characters than its answer has tokens and ends on a small event
    never streamed the end of its answer (Ollama drops a tool call that max_tokens cut short): its window runs to the
    end of the stream, where the last tokens were written."""
    if len(ev_t) < 2 or not ct:
        if len(ev_t) == 1 and ct and ct >= 8 and sum(ev_c) < ct and ev_c[0] < 0.1 * max(1, sum(ev_c)) + 4:
            return float(ct - 1), max(0.0, total - ev_t[0]), True
        return 0.0, 0.0, False
    chars = sum(ev_c)
    if ct >= 8 and chars < ct and ev_c[-1] < 0.1 * chars:
        return float(ct - 1), max(0.0, total - ev_t[0]), True
    if len(ev_t) in (ct, ct - 1):
        return float(len(ev_t) - 1), ev_t[-1] - ev_t[0], False
    _, _, info = bench.decode_rates({"ev_t": ev_t, "ev_c": ev_c}, ct, head=1, tail=0)
    return float(info.get("window_tokens") or 0.0), float(info.get("window_s") or 0.0), False


HELD_SHARE = 0.5  # a first event with at least this share of an answer's characters held the answer back


def held_back(x):
    """True if a turn's first streamed event came only after most of the answer was written: a parsed tool call
    (an engine that holds a call until its closing tag, even without tools in the request), or one event carrying at
    least half of the answer's characters (e.g. an engine that replaces a tool call it will not run with a notice).
    Such a turn has no time to first token of its own."""
    if x.get("first_event") == "tool_call":
        return True
    ch = x.get("events_chars") or []
    return bool(ch) and (x.get("completion_tokens") or 0) >= 8 and ch[0] >= HELD_SHARE * sum(ch)


def summarize(turns, wall_s, gaps_s):
    """Session totals. A turn whose answer was held back (held_back) has no time to first token of its own: its TTFT
    is left out of the TTFT statistics, and its prefill is estimated as its time minus its tokens at the session's
    writing rate (counted in prefill_estimated_turns)."""
    ok = [x for x in turns if not x.get("error")]
    tok = sum(x.get("completion_tokens") or 0 for x in ok)
    eng = sum(x["total_s"] for x in ok)
    wt = sum(x["window_tokens"] for x in ok)
    ws = sum(x["window_s"] for x in ok)
    rate = wt / ws if ws > 0 else None
    held = [x for x in ok if held_back(x)]
    pre, est = 0.0, 0
    for x in ok:
        if x.get("ttft_s") is None:
            pre += x["total_s"]
        elif held_back(x):
            dec = ((x.get("completion_tokens") or 1) - 1) / rate if rate else 0.0
            pre += max(0.0, min(x["total_s"], x["total_s"] - dec))
            est += 1
        else:
            pre += x["ttft_s"]
    fu_all = [x["ttft_s"] for x in ok if x.get("follow_up") and x.get("ttft_s") is not None]
    fu = [x["ttft_s"] for x in ok if x.get("follow_up") and x.get("ttft_s") is not None and not held_back(x)]
    main = [x for x in ok if x["kind"] == "main"]
    first = turns[0] if turns else {}
    s = {"turns": len(turns), "turns_ok": len(ok), "follow_ups": len(fu_all), "followup_ttft_n": len(fu),
         "engine_s": round(eng, 3), "wall_s": round(wall_s, 3), "gaps_s": round(gaps_s, 3),
         "prefill_s": round(pre, 3), "decode_s": round(eng - pre, 3), "prefill_estimated_turns": est,
         "completion_tokens": tok, "max_tokens": sum(x["max_tokens"] for x in turns),
         "write_tps": round(rate, 2) if rate else None, "window_tokens": round(wt, 1), "window_s": round(ws, 3),
         "e2e_tps": round(tok / eng, 2) if eng > 0 else None,
         "first_ttft_s": round(first["ttft_s"], 4) if first.get("ttft_s") is not None and not held_back(first) else None,
         "followup_ttft_median_s": round(statistics.median(fu), 4) if fu else None,
         "followup_ttft_p90_s": round(pct(fu, 90), 4) if fu else None,
         "followup_ttft_max_s": round(max(fu), 4) if fu else None,
         "followup_ttft_all_median_s": round(statistics.median(fu_all), 4) if fu_all else None,
         "prompt_tokens_max": max((x.get("prompt_tokens") or 0 for x in ok), default=None),
         "context_tokens_max": max(((x.get("prompt_tokens") or 0) + (x.get("completion_tokens") or 0) for x in ok),
                                   default=None),
         "prompt_tokens_main_sum": sum(x.get("prompt_tokens") or 0 for x in main),
         "stopped_early": sum(1 for x in ok if (x.get("completion_tokens") or 0) < x["max_tokens"] - 1),
         "held_tool_calls": len(held),
         "after_last_token_s": round(sum(x["total_s"] - (x.get("events_t_ms") or [0])[-1] / 1000 for x in ok
                                         if x.get("events_t_ms") and not x.get("tail_unstreamed")), 3),
         "tail_unstreamed_turns": sum(1 for x in ok if x.get("tail_unstreamed")),
         "same_answers": sum(1 for x in ok if x.get("same_answer")),
         "cached_tokens_reported": sum(x.get("cached_tokens") or 0 for x in ok) if any(
             x.get("cached_tokens") is not None for x in ok) else None}
    return s


# ---------------------------------------------------------------- replay
def replay(a, t, mode="rendered", gaps="recorded", max_turns=None, log=print, timeout_s=None):
    """Replays trace `t` against a.base_url with bench.py's request fields (a = bench.parse_args namespace).
    Returns the phase result: per-turn records, the session summary, invalid reasons and flags. With timeout_s, no
    new turn starts after that many seconds (the session is then INVALID: incomplete)."""
    reqs = t["requests"][:max_turns] if max_turns else t["requests"]
    rec_gaps = recorded_gaps(t)
    sid = f"agent-{random.getrandbits(48):012x}"
    res = {"trace": trace_record(t), "tool_mode": mode, "gaps": gaps, "session_id": sid, "turns": [],
           "invalid": [], "flags": []}
    prev_main_ids, failed_in_a_row, gaps_s = None, 0, 0.0
    t0 = time.perf_counter()
    for k, rq in enumerate(reqs):
        if timeout_s and time.perf_counter() - t0 > timeout_s:
            res["invalid"].append(f"incomplete: stopped after {k} of {len(reqs)} requests (time limit {timeout_s:.0f} s)")
            res["timed_out"] = True
            log(f"  agent session: time limit {timeout_s:.0f} s reached after {k} of {len(reqs)} requests")
            break
        if k:
            g = rec_gaps[k] if gaps == "recorded" else (0.0 if gaps == "none" else float(gaps))
            g = min(max(0.0, g), MAX_GAP_S)
            if g:
                time.sleep(g)
                gaps_s += g
        msgs, extra = request_parts(t, rq, mode)
        rec = rq.get("response") or {}
        mt = max(1, int(rec.get("completion_tokens") or 1))
        body = bench.body_of(a, msgs, mt, greedy=True, **extra)
        follow = bool(rq["kind"] == "main" and prev_main_ids is not None
                      and rq["messages"][:len(prev_main_ids)] == prev_main_ids)
        turn = {"i": rq.get("i", k), "kind": rq["kind"], "follow_up": follow, "messages": len(msgs),
                "max_tokens": mt, "recorded_completion_tokens": rec.get("completion_tokens"),
                "ref_prompt_tokens": ref_prompt_tokens(rq, mode)}
        try:
            r = bench.stream_chat(a.base_url, body, session=sid)
        except Exception as e:
            turn.update({"error": f"{e!r}"[:400], "total_s": 0.0, "ttft_s": None, "window_tokens": 0.0,
                         "window_s": 0.0})
            res["turns"].append(turn)
            res["invalid"].append(f"turn {turn['i']}: request failed: {e!r}"[:300])
            log(f"  agent turn {turn['i']}: FAILED {e!r}"[:300])
            failed_in_a_row += 1
            if failed_in_a_row >= MAX_FAILED_IN_A_ROW:
                res["invalid"].append(f"stopped after {failed_in_a_row} failed turns in a row")
                break
            continue
        failed_in_a_row = 0
        u = r["usage"] or {}
        ct, pt = u.get("completion_tokens"), u.get("prompt_tokens")
        wtok, wsec, tail = window(r["ev_t"], r["ev_c"], ct, r["total"])
        k0 = r["ev_k"][0] if r.get("ev_k") else ""
        first_kind = None if not r["ev_t"] else ("text" if "c" in k0 else "reasoning" if "r" in k0 else "tool_call")
        turn.update({"prompt_tokens": pt, "cached_tokens": bench.cached_of(u, r["timings"]), "completion_tokens": ct,
                     "finish": r["finish"], "ttft_s": round(r["ttft"], 4) if r["ttft"] is not None else None,
                     "total_s": round(r["total"], 4),
                     "decode_s": round(r["total"] - r["ttft"], 4) if r["ttft"] is not None else None,
                     "events": r["n_chunks"], "first_event": first_kind, "tool_call_events": len(r["tool_calls"]),
                     "window_tokens": round(wtok, 1), "window_s": round(wsec, 4), "tail_unstreamed": tail,
                     "write_tps": round(wtok / wsec, 2) if wsec > 0 else None,
                     "timings": r["timings"], "content": r["content"], "reasoning": r["reasoning"],
                     "tool_calls": r["tool_calls"], "events_t_ms": [round(x * 1000, 1) for x in r["ev_t"]],
                     "events_chars": r["ev_c"], "events_kind": ",".join(r["ev_k"])})
        turn["same_answer"] = answer_form(r["content"], r["tool_calls"]) == answer_form(rec.get("content"),
                                                                                       rec.get("tool_calls"))
        for why in bench.thinking_leak(r):
            res["invalid"].append(f"turn {turn['i']}: {why}")
        if not ct:
            res["invalid"].append(f"turn {turn['i']}: usage.completion_tokens missing")
        ref = turn["ref_prompt_tokens"]
        if pt and ref:
            if pt < PROMPT_TRUNCATED * ref:
                res["invalid"].append(f"turn {turn['i']}: prompt_tokens {pt} < {PROMPT_TRUNCATED:.0%} of the reference {ref} "
                                      "(conversation cut)")
            elif abs(pt - ref) > PROMPT_TOL * ref:
                res["flags"].append(f"turn {turn['i']}: prompt_tokens {pt} vs reference {ref}")
        res["turns"].append(turn)
        if rq["kind"] == "main":
            prev_main_ids = rq["messages"]
        log(f"  agent turn {turn['i']:2d}{'' if rq['kind'] == 'main' else ' (' + rq['kind'] + ')'}: prompt {pt} "
            f"(cached {turn['cached_tokens']}), TTFT {bench.f2(turn['ttft_s'])}s, {ct}/{mt} tok "
            f"({r['finish']}), {bench.f2(turn['write_tps'])} tok/s, {turn['total_s']:.2f}s")
    wall = time.perf_counter() - t0
    res["summary"] = s = summarize(res["turns"], wall, gaps_s)
    res["flags"] += session_flags(s, mode)
    return res


def session_flags(s, mode):
    out = []
    if s["held_tool_calls"]:
        out.append(f"{s['held_tool_calls']} answers held back until (most of) the answer was written"
                   + (" although the request had no tools" if mode == "rendered" else "")
                   + ": their TTFT includes the answer (left out of the TTFT statistics; their prefill time is "
                   "estimated from the session's writing rate)")
    if s["max_tokens"] and s["completion_tokens"] < 0.9 * s["max_tokens"]:
        out.append(f"wrote {s['completion_tokens']} of the recorded {s['max_tokens']} tokens "
                   f"({s['stopped_early']} turns stopped early)")
    return out


def resummarize(res):
    """Recomputes a stored phase result's summary and session flags from its per-turn records (e.g. after a change
    to the summary rules); invalid reasons and per-turn flags are kept."""
    keep = [f for f in res.get("flags", []) if f.startswith("turn ")]
    for x in res["turns"]:
        if x.get("error") or not x.get("events_t_ms"):
            continue
        wtok, wsec, tail = window([t / 1000 for t in x["events_t_ms"]], x.get("events_chars") or [],
                                  x.get("completion_tokens"), x["total_s"])
        x.update({"window_tokens": round(wtok, 1), "window_s": round(wsec, 4), "tail_unstreamed": tail,
                  "write_tps": round(wtok / wsec, 2) if wsec > 0 else None})
    s = summarize(res["turns"], res["summary"]["wall_s"], res["summary"]["gaps_s"])
    res["summary"] = s
    res["flags"] = keep + session_flags(s, res.get("tool_mode", "rendered"))
    return res


def line(s):
    """One-line summary of a replay."""
    return (f"{s['turns_ok']}/{s['turns']} turns, engine {s['engine_s']:.1f}s (prefill {s['prefill_s']:.1f}s, decode "
            f"{s['decode_s']:.1f}s), wall {s['wall_s']:.1f}s, {s['completion_tokens']}/{s['max_tokens']} tok out, "
            f"writing {bench.f2(s['write_tps'])} tok/s, first TTFT {bench.f2(s['first_ttft_s'])}s, follow-up TTFT median "
            f"{bench.f2(s['followup_ttft_median_s'])}s p90 {bench.f2(s['followup_ttft_p90_s'])}s, "
            f"peak context {s['context_tokens_max']}, {s['same_answers']}/{s['turns_ok']} answers as recorded")


def main():
    if os.path.exists(bench.LOCAL_ENV):
        bench.load_env_file(bench.LOCAL_ENV)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--label", default="agent-replay")
    ap.add_argument("--tool-mode", default="rendered", choices=["rendered", "api"])
    ap.add_argument("--gaps", default="recorded", help="recorded (default, at most 5 s each), none, or seconds")
    ap.add_argument("--max-turns", type=int, default=None, help="replay only the first N requests (a quick check)")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="start no turn after this many seconds")
    ap.add_argument("--sampling-overrides", default="")
    ap.add_argument("--no-think-style", default="both", choices=["both", "kwargs", "effort", "none"])
    ap.add_argument("--extra-body", default="")
    ap.add_argument("--no-gpu-lock", action="store_true")
    ap.add_argument("--out", default=None)
    o = ap.parse_args()
    a = bench.parse_args(["--base-url", o.base_url, "--model", o.model, "--label", o.label,
                          "--no-think-style", o.no_think_style] +
                         (["--sampling-overrides", o.sampling_overrides] if o.sampling_overrides else []) +
                         (["--extra-body", o.extra_body] if o.extra_body else []))
    t = load_trace(o.trace)
    lock = None
    if not o.no_gpu_lock and not gpu_hogs.held_by_caller():
        lock = gpu_hogs.GpuLock(log=lambda m: print("  " + m, flush=True)).acquire()
    try:
        print(f"== {o.label}: {t.get('name')} ({len(t['requests'])} requests), tool mode {o.tool_mode}", flush=True)
        r = replay(a, t, o.tool_mode, o.gaps, o.max_turns, log=lambda m: print(m, flush=True), timeout_s=o.timeout)
    finally:
        if lock:
            lock.release()
    r.update({"label": o.label, "model": o.model, "base_url": o.base_url, "when": time.strftime("%Y-%m-%d %H:%M:%S"),
              "sampling": bench.sampling(a, greedy=True), "no_think": bench.no_think(a), "harness": bench.harness_info()})
    print("  agent session: " + line(r["summary"]), flush=True)
    if r["invalid"]:
        print(f"  INVALID: {r['invalid']}", flush=True)
    if r["flags"]:
        print(f"  flags: {r['flags']}", flush=True)
    out = o.out or os.path.join(bench.out_dir(), "results", f"{o.label}-agent-{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(r, fh, indent=1)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
