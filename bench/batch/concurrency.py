#!/usr/bin/env python3
"""Requests in flight at once against a running `kapitsa serve` (OpenAI chat completions, streamed; stdlib only).

  concurrency.py ttft    --base URL [--reply 1000] [--prompt 2000] [--after-s 1.5] [--a-prompt N]
      A long reply streams (or, with --a-prompt, an N-token prompt prefills); --after-s later a cold --prompt-token
      request arrives. Reports the second request's time to first token, the first one's stall, and both requests'
      tokens over the whole run.
  concurrency.py streams --base URL --n N [--tokens 512] [--serial] [--temperature 0]
      N decode streams sent at once (or one after another with --serial): per-stream and total tok/s.
  concurrency.py exact   --base URL --out FILE [--mode alone|together|staggered]
      A fixed set of requests (greedy and seeded T > 0; short and 2K-token prompts; replies cut by max_tokens and
      replies that stop) run one at a time or all at once; their texts go to FILE.
  concurrency.py turns   --base URL --out FILE [--mode alone|together]
      Two-turn conversations (turn 2 reuses turn 1's slot, checkpoint and drafter ring), one after another or at once.
  concurrency.py memory  --base URL [--n 2] [--prompt 16000] [--tokens 64]
      N cold --prompt-token requests at once against a server with a small --mem-budget: all must be answered.
  concurrency.py memtake --base URL --log SERVER_LOG [--after-s 6]
      A reply in progress needs KV cache that a request waiting for memory holds: the reply takes it and runs to
      max_tokens; the waiting request is answered after it (on a server with a small --mem-budget).
  concurrency.py greedy  --base URL [--mode alone|together] [--n 8] [--tokens 300]
      Greedy requests of different kinds, one at a time or all at once (with NI_TRACE_TOKENS set on the server,
      match.py compares the batched outputs with the single-stream ones).
  concurrency.py single  --base URL [--reps 5] [--tokens 512] [--sizes 4000,16000] [--prefill-reps 2]
      One request at a time with the same prompts in every run (a single-stream build A/B): greedy decode tok/s and
      the time to first token of cold prompts.
  concurrency.py compare A.json B.json
      The same requests' texts and token counts in two runs (exactness: alone vs together, main vs this build).

Prompts are cut from the frozen corpus (bench/corpus.txt) behind a nonce, so no request reuses another's cache. Every
request has thinking off (chat_template_kwargs.enable_thinking = false)."""
import argparse, base64, json, os, random, struct, sys, threading, time, urllib.error, urllib.request, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "..", "corpus.txt")
CHARS_PER_TOKEN = 4.25
SYS = ("You are Nace, a local coding agent. You help the user with their repository. "
       "Be precise. Use tools when you need to read files or run commands.")
CODE = ("Write a complete, well-commented Python module that implements an LRU cache with TTL expiry, a thread-safe "
        "wrapper, and a pytest test suite. Write at least 600 lines. Do not stop early.")


def png(w, h, px):
    """an RGB PNG of w x h pixels, px(x, y) -> (r, g, b)"""
    raw = b"".join(b"\x00" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def image_messages(tag):
    """a picture (four colored quadrants and a white diagonal) and a question about it"""
    def px(x, y):
        if abs(x - y) < 6:
            return (255, 255, 255)
        return [(220, 30, 30), (30, 160, 60), (40, 60, 220), (240, 200, 30)][(x >= 160) + 2 * (y >= 160)]
    url = "data:image/png;base64," + base64.b64encode(png(320, 320, px)).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": f"[{tag}] Describe this image in two sentences."}]}]


def corpus():
    with open(CORPUS, encoding="utf-8") as f:
        return f.read()


def corpus_cut(tokens, offset=0):
    """`tokens` worth of the corpus from `offset`; exits rather than return a short cut"""
    c, want = corpus(), int(tokens * CHARS_PER_TOKEN)
    if offset + want > len(c):
        sys.exit(f"a {tokens:,}-token prompt at offset {offset:,} needs {offset + want:,} characters of the corpus, "
                 f"and {CORPUS} has {len(c):,}: lower the size")
    return c[offset:offset + want]


def context_messages(tokens, nonce, question, offset=0):
    ctx = corpus_cut(tokens, offset)
    return [{"role": "system", "content": f"[{nonce}] {SYS}\n\n# Repository context\n{ctx}"},
            {"role": "user", "content": question}]


def code_messages(nonce):
    return [{"role": "system", "content": f"[{nonce}] You are a helpful coding assistant. Answer with code only."},
            {"role": "user", "content": CODE}]


def body(msgs, max_tokens, temperature=0.0, seed=None):
    b = {"messages": msgs, "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
         "chat_template_kwargs": {"enable_thinking": False}, "temperature": temperature,
         "top_p": 1.0 if temperature == 0 else 0.95, "top_k": 1 if temperature == 0 else 20}
    if seed is not None:
        b["seed"] = seed
    return b


def stream(base, b, t_ref=None, timeout=3600):
    """One streamed completion: send/first/last times (s since t_ref), event times, text, usage, finish."""
    t_ref = t_ref if t_ref is not None else time.perf_counter()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
    out = {"sent": time.perf_counter() - t_ref, "first": None, "last": None, "ev": [], "text": "", "usage": None,
           "finish": None, "error": None}
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        parts = []
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            if "error" in ev:
                out["error"] = ev["error"]
                continue
            if ev.get("usage"):
                out["usage"] = ev["usage"]
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    out["finish"] = ch["finish_reason"]
                txt = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if txt:
                    now = time.perf_counter() - t_ref
                    if out["first"] is None:
                        out["first"] = now
                    out["last"] = now
                    out["ev"].append(now)
                    parts.append(txt)
        out["text"] = "".join(parts)
    except urllib.error.HTTPError as e:
        out["error"] = f"HTTP {e.code}: {e.read()[:300]!r}"
    except Exception as e:  # noqa: BLE001 (report, don't die: other streams are in flight)
        out["error"] = repr(e)
    out["done"] = time.perf_counter() - t_ref
    return out


def ntok(o):
    return (o["usage"] or {}).get("completion_tokens", len(o["ev"]))


def decode_rate(o):
    """tok/s from the first to the last token (time to first token excluded)"""
    n = ntok(o)
    if o["first"] is None or o["last"] is None or o["last"] <= o["first"] or n < 2:
        return 0.0
    return (n - 1) / (o["last"] - o["first"])


def run_threads(fns):
    res = [None] * len(fns)

    def go(i):
        res[i] = fns[i]()

    th = [threading.Thread(target=go, args=(i,)) for i in range(len(fns))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res


def warm(base):
    stream(base, body(context_messages(300, f"warm {random.getrandbits(32):08x}", "Say hi."), 8))


def evict(base, k=8):
    """one-off requests that take every slot (up to k), so the measured requests prefill cold: runs with the same prompts
    then give the same texts (the same tokens alone and together), whatever ran before"""
    for i in range(k):
        stream(base, body([{"role": "user", "content": f"[evict {i} {random.getrandbits(32):08x}] " + "Say ok. " * 40}], 1))


# ---------------------------------------------------------------- ttft
def cmd_ttft(a):
    warm(a.base)
    evict(a.base)
    t0 = time.perf_counter()
    nonce = a.nonce
    ra = {}

    def first():
        msgs = (context_messages(a.a_prompt, f"A {nonce:012x}", "Summarize the repository context in detail.",
                                 offset=60000) if a.a_prompt else code_messages(f"A {nonce:012x}"))
        ra["o"] = stream(a.base, body(msgs, a.reply), t0)

    th = threading.Thread(target=first)
    th.start()
    time.sleep(a.after_s)  # the first reply is streaming by then
    rb = stream(a.base, body(context_messages(a.prompt, f"B {nonce:012x}",
                                              "In one sentence, what is the first file in the repository context about?",
                                              offset=20000), a.b_tokens), t0)
    th.join()
    oa = ra["o"]
    gaps = [y - x for x, y in zip(oa["ev"], oa["ev"][1:])]
    tot = ntok(oa) + ntok(rb)
    span = max(oa["done"], rb["done"]) - oa["sent"]
    res = {
        "A": {"ttft": oa["first"] - oa["sent"] if oa["first"] else None, "tokens": ntok(oa),
              "decode_tps": decode_rate(oa), "done": oa["done"], "max_gap_s": max(gaps) if gaps else None,
              "prompt_tokens": (oa["usage"] or {}).get("prompt_tokens"), "finish": oa["finish"], "error": oa["error"]},
        "B": {"sent": rb["sent"], "ttft": rb["first"] - rb["sent"] if rb["first"] else None, "tokens": ntok(rb),
              "prompt_tokens": (rb["usage"] or {}).get("prompt_tokens"),
              "cached": ((rb["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
              "done": rb["done"], "finish": rb["finish"], "error": rb["error"]},
        "A_tokens_before_B": sum(1 for t in oa["ev"] if t < rb["sent"]),
        "total_tokens": tot, "span_s": span, "total_tps": tot / span if span > 0 else 0,
    }
    print(json.dumps(res))


# ---------------------------------------------------------------- streams
def cmd_streams(a):
    warm(a.base)
    evict(a.base)
    nonce = a.nonce
    bodies = [body(code_messages(f"S{i} {nonce:012x}"), a.tokens, a.temperature, seed=1000 + i if a.temperature else None)
              for i in range(a.n)]
    t0 = time.perf_counter()
    if a.serial:
        outs = [stream(a.base, b, t0) for b in bodies]
    else:
        outs = run_threads([lambda b=b: stream(a.base, b, t0) for b in bodies])
    errs = [o["error"] for o in outs if o["error"]]
    tot = sum(ntok(o) for o in outs)
    span = max(o["done"] for o in outs) - min(o["sent"] for o in outs)
    # decode-only aggregate: tokens after each stream's first, over the window from the first first-token to the end
    dec_win = max(o["last"] or 0 for o in outs) - min(o["first"] or 1e9 for o in outs)
    res = {"n": a.n, "serial": a.serial, "temperature": a.temperature, "tokens": [ntok(o) for o in outs],
           "per_stream_tps": [round(decode_rate(o), 2) for o in outs],
           "ttft": [round(o["first"] - o["sent"], 3) if o["first"] else None for o in outs],
           "total_tokens": tot, "span_s": round(span, 3), "total_tps": round(tot / span, 2) if span > 0 else 0,
           "decode_window_tps": round((tot - len(outs)) / dec_win, 2) if dec_win > 0 else 0,
           "errors": errs}
    print(json.dumps(res))


# ---------------------------------------------------------------- single
def cmd_single(a):
    """One request at a time, the same prompts in every run, so two builds that give the same tokens do the same GPU
    work (a single-stream A/B without the prompt-to-prompt spread of speculative decoding): decode tok/s of --reps
    greedy replies, then the time to first token of --prefill-reps cold prompts of each --sizes"""
    warm(a.base)
    evict(a.base)
    dec, ttft, ptok = [], {}, {}
    for i in range(a.reps):
        o = stream(a.base, body(code_messages(f"D{i} {a.nonce:012x}"), a.tokens))
        dec.append(round(decode_rate(o), 2) if not o["error"] and ntok(o) == a.tokens else None)
    q = "In one sentence, what is the first file in the repository context about?"
    for size in [int(x) for x in a.sizes.split(",") if x]:
        ttft[size], ptok[size] = [], []
        for i in range(a.prefill_reps):
            o = stream(a.base, body(context_messages(size, f"P{size} {i} {a.nonce:012x}", q), 4))
            ttft[size].append(round(o["first"] - o["sent"], 3) if o["first"] and not o["error"] else None)
            ptok[size].append((o["usage"] or {}).get("prompt_tokens"))
    med = lambda v: sorted(x for x in v if x is not None)[len([x for x in v if x is not None]) // 2] if any(
        x is not None for x in v) else None
    print(json.dumps({"decode_tps": dec, "decode_median": med(dec), "ttft": ttft, "ttft_median":
                      {k: med(v) for k, v in ttft.items()}, "prompt_tokens": ptok}))


# ---------------------------------------------------------------- exact
def exact_plan():
    q = "In one sentence, what is the first file in the repository context about?"
    return [
        ("code_greedy_400", body(code_messages("E1 3f2a"), 400)),
        ("code_t06_seed7_400", body(code_messages("E2 77c1"), 400, 0.6, 7)),
        ("ctx2k_greedy_stop", body(context_messages(2000, "E3 9d0e", q, offset=40000), 200)),
        ("ctx2k_t06_seed3", body(context_messages(2000, "E4 1b7f", "List the classes defined there, briefly.",
                                                 offset=60000), 300, 0.6, 3)),
        ("ctx6k_greedy", body(context_messages(6000, "E5 c44d", q, offset=80000), 120)),
        ("short_greedy_stop", body([{"role": "user", "content": "[E6] Name three prime numbers, comma separated."}], 64)),
        ("image_greedy", body(image_messages("E7"), 80)),
    ]


def cmd_exact(a):
    warm(a.base)
    evict(a.base)  # the plan's prompts are fixed: an earlier run's caches must not be reused
    plan = exact_plan()
    if a.only:
        plan = [p for p in plan if p[0] in a.only.split(",")]
    t0 = time.perf_counter()
    if a.mode == "alone":
        outs = [stream(a.base, b, t0) for _, b in plan]
    elif a.mode == "together":
        outs = run_threads([lambda b=b: stream(a.base, b, t0) for _, b in plan])
    else:  # staggered: each request 0.4 s after the previous one (they meet in prefill and in decode)
        def later(i, b):
            def f():
                time.sleep(0.4 * i)
                return stream(a.base, b, t0)
            return f
        outs = run_threads([later(i, b) for i, (_, b) in enumerate(plan)])
    res = {name: {"text": o["text"], "tokens": ntok(o), "finish": o["finish"], "error": o["error"],
                  "ttft": (o["first"] - o["sent"]) if o["first"] else None, "decode_tps": decode_rate(o)}
           for (name, _), o in zip(plan, outs)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(f"{k:22s} {v['tokens']:5d} tok {v['finish']!s:7s} ttft {v['ttft'] or 0:6.2f} s  {v['decode_tps']:6.1f} tok/s"
              + (f"  ERROR {v['error']}" if v["error"] else ""))


# ---------------------------------------------------------------- memory
def cmd_memory(a):
    """N cold requests of --prompt tokens sent together to a server whose memory budget holds fewer of their KV caches
    at once: each must be answered in full (one waits for another's cache when the budget is short), none may fail."""
    warm(a.base)
    nonce = random.getrandbits(48)
    t0 = time.perf_counter()
    bodies = [body(context_messages(a.prompt, f"M{i} {nonce:012x}", "In one sentence, what is the first file about?",
                                    offset=(i * 7919) % 20000), a.tokens) for i in range(a.n)]
    outs = run_threads([lambda b=b: stream(a.base, b, t0) for b in bodies])
    res = [{"prompt_tokens": (o["usage"] or {}).get("prompt_tokens"), "tokens": ntok(o), "finish": o["finish"],
            "ttft": round(o["first"] - o["sent"], 2) if o["first"] else None, "done": round(o["done"], 2),
            "error": o["error"]} for o in outs]
    print(json.dumps(res))
    ok = all(r["error"] is None and r["finish"] in ("stop", "length") and r["tokens"] > 0 for r in res)
    print("MEMORY OK" if ok else "MEMORY FAILED")
    sys.exit(0 if ok else 1)


def kv_room(log):
    """the tokens of KV cache the server's budget holds (its log's memory line), less what DFlash2 takes at the first
    prefill, and at most a conversation's context"""
    import re
    txt = open(log, encoding="utf-8", errors="ignore").read()
    m = re.search(r"KV room [\d.]+ GB = (\d+) tokens \((\d+) KB per token\)", txt)
    room, per_kb = int(m.group(1)), int(m.group(2))
    d = re.search(r"DFlash2: .* and ([\d.]+) GB at the first prefill", txt)
    if d:
        room -= int(float(d.group(1)) * 1e9 / (per_kb * 1024))
    c = re.search(r"one conversation up to \d+ of (\d+) tokens", txt)
    return min(room, int(c.group(1)) - 64) if c else room


def cmd_memtake(a):
    """A reply in progress needs more KV cache while another request waits for memory with its conversation cached
    (before its prefill): the reply takes that cache and runs to max_tokens (it does not end early), and the waiting
    request reads its prompt again from the start once the reply has ended. Sizes follow the budget's KV room (the
    server's log, --log): conversation W holds 45% of it (two turns, so a new conversation takes another slot), then a
    reply R (prompt 10%, 35% written) and, --after-s later, W's next turn (85%: it fits alone, not beside R). (A cache
    grows by a copy, old and new side by side: one conversation grows to about half the room of such a small budget.)"""
    room = kv_room(a.log)
    warm(a.base)
    evict(a.base)
    nonce = random.getrandbits(48)
    msgs = context_messages(int(room * 0.45), f"W {nonce:012x}", "In one sentence, what is the first file about?")
    for follow in ["And in two words?", None]:
        o = stream(a.base, body(msgs, 16))
        msgs = msgs + [{"role": "assistant", "content": o["text"]}]
        if follow:
            msgs.append({"role": "user", "content": follow})
    w_cached = (o["usage"] or {}).get("prompt_tokens")
    msgs.append({"role": "user", "content": "More of the repository:\n" +
                 corpus_cut(room * 0.40, 90000) + "\n\nIn one sentence, what does it do?"})
    r_tokens = int(room * 0.35)
    rmsgs = context_messages(int(room * 0.10), f"R {nonce:012x}",
                             "Ignore the context above. Count from 1 to 100000, one number per line, nothing else.",
                             offset=60000)
    t0 = time.perf_counter()
    out = {}
    th = threading.Thread(target=lambda: out.__setitem__("r", stream(a.base, body(rmsgs, r_tokens), t0)))
    th.start()
    time.sleep(a.after_s)  # R holds its slot and its KV cache by then
    ow = stream(a.base, body(msgs, 16), t0)
    th.join()
    orr = out["r"]
    res = {"room": room, "W_cached_before": w_cached,
           "R": {"tokens": ntok(orr), "max_tokens": r_tokens, "finish": orr["finish"], "error": orr["error"],
                 "first": round(orr["first"], 1) if orr["first"] else None, "done": round(orr["done"], 1)},
           "W": {"prompt_tokens": (ow["usage"] or {}).get("prompt_tokens"),
                 "cached": ((ow["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                 "tokens": ntok(ow), "finish": ow["finish"], "error": ow["error"], "sent": round(ow["sent"], 1),
                 "first": round(ow["first"], 1) if ow["first"] else None}}
    # the scenario happened: W waited before its prefill and R took W's cache (the server's log says so)
    txt = open(a.log, encoding="utf-8", errors="ignore").read()
    res["W_waited"] = "KV cache before its prefill" in txt
    res["R_took_W_cache"] = "its request waits for memory" in txt
    print(json.dumps(res))
    # R writes all its tokens; W is answered, after R (it waited for R's memory)
    ok = (orr["error"] is None and ntok(orr) == r_tokens and ow["error"] is None and ntok(ow) > 0 and
          ow["first"] is not None and orr["done"] is not None and ow["first"] > orr["done"] - 1)
    if ok and not (res["W_waited"] and res["R_took_W_cache"]):
        print("MEMTAKE NOT EXERCISED (the budget left room for both)")
        sys.exit(2)
    print("MEMTAKE OK" if ok else "MEMTAKE FAILED")
    sys.exit(0 if ok else 1)


def greedy_plan(n, tokens):
    """n greedy requests of different kinds (code, prose, a long context, counting), fixed nonces"""
    kinds = [
        lambda i: code_messages(f"G{i} c0d3"),
        lambda i: [{"role": "user", "content": f"[G{i}] Explain how a B-tree keeps itself balanced, in detail."}],
        lambda i: context_messages(1500, f"G{i} ctx", "Describe what the code above does, file by file.",
                                   offset=10000 + 3000 * i),
        lambda i: [{"role": "user", "content": f"[G{i}] Write a short story about a lighthouse keeper and a storm."}],
        lambda i: [{"role": "user", "content": f"[G{i}] List the planets of the solar system with one fact each."}],
        lambda i: [{"role": "system", "content": f"[G{i}] You are a terse assistant."},
                   {"role": "user", "content": "Write a Python function that merges overlapping intervals, with tests."}],
    ]
    return [(f"g{i}", body(kinds[i % len(kinds)](i), tokens)) for i in range(n)]


def cmd_greedy(a):
    """greedy requests one at a time or all at once (with NI_TRACE_TOKENS on the server: match.py compares them)"""
    warm(a.base)
    evict(a.base)
    plan = greedy_plan(a.n, a.tokens)
    t0 = time.perf_counter()
    if a.mode == "alone":
        outs = [stream(a.base, b, t0) for _, b in plan]
    else:
        outs = run_threads([lambda b=b: stream(a.base, b, t0) for _, b in plan])
    span = max(o["done"] for o in outs) - min(o["sent"] for o in outs)
    tot = sum(ntok(o) for o in outs)
    print(json.dumps({"mode": a.mode, "n": a.n, "tokens": [ntok(o) for o in outs], "span_s": round(span, 2),
                      "total_tps": round(tot / span, 2) if span > 0 else 0,
                      "errors": [o["error"] for o in outs if o["error"]]}))


def cmd_turns(a):
    """two-turn conversations: turn 2 continues turn 1 on its slot (prefix reuse, the checkpoint and the drafter ring a
    finished turn 1 left), so a wrong state after turn 1 shows in turn 2. Conversations one after another, or at once."""
    warm(a.base)
    evict(a.base)
    # system prompts well over 64 tokens: turn 2 then reuses its own conversation's slot whatever ran before (a shorter
    # prefix is no reason to pick a slot, and a reused prefix and a cold prefill can split at near-ties)
    rules = ("Follow the user's instructions carefully, keep answers focused, prefer clear names, explain trade-offs "
             "briefly, and mention edge cases when they matter. ") * 4
    convs = [
        ("conv_code", [{"role": "system", "content": "[T1 51a] You are a helpful coding assistant. " + rules},
                       {"role": "user", "content": "Write a Python function that parses ISO dates, with tests."}],
         "Now make it accept timezones too.", 0.0, None),
        ("conv_t06", [{"role": "system", "content": "[T2 77b] You are a helpful assistant. " + rules},
                      {"role": "user", "content": "Explain how a hash map handles collisions, in detail."}],
         "And how does resizing work?", 0.6, 5),
    ]

    def run(c):
        name, msgs, follow, temp, seed = c
        r1 = stream(a.base, body(msgs, a.tokens, temp, seed))
        msgs2 = msgs + [{"role": "assistant", "content": r1["text"]}, {"role": "user", "content": follow}]
        r2 = stream(a.base, body(msgs2, a.tokens, temp, seed))
        return r1, r2

    outs = [run(c) for c in convs] if a.mode == "alone" else run_threads([lambda c=c: run(c) for c in convs])
    res = {}
    for c, (r1, r2) in zip(convs, outs):
        for k, r in (("turn1", r1), ("turn2", r2)):
            res[f"{c[0]}_{k}"] = {"text": r["text"], "tokens": ntok(r), "finish": r["finish"], "error": r["error"],
                                  "cached": ((r["usage"] or {}).get("prompt_tokens_details") or {}).get("cached_tokens"),
                                  "ttft": (r["first"] - r["sent"]) if r["first"] else None, "decode_tps": decode_rate(r)}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(f"{k:18s} {v['tokens']:5d} tok {v['finish']!s:7s} cached {v['cached']!s:6s} {v['decode_tps']:6.1f} tok/s"
              + (f"  ERROR {v['error']}" if v["error"] else ""))


def cmd_compare(a):
    x, y = json.load(open(a.a)), json.load(open(a.b))
    bad = 0
    for k in x:
        if k not in y:
            continue
        same = x[k]["text"] == y[k]["text"] and x[k]["tokens"] == y[k]["tokens"]
        if not same:
            bad += 1
            i = next((j for j, (c, d) in enumerate(zip(x[k]["text"], y[k]["text"])) if c != d),
                     min(len(x[k]["text"]), len(y[k]["text"])))
            print(f"DIFF {k}: {x[k]['tokens']} vs {y[k]['tokens']} tokens, first difference at char {i}: "
                  f"{x[k]['text'][i:i + 40]!r} vs {y[k]['text'][i:i + 40]!r}")
        else:
            print(f"SAME {k}: {x[k]['tokens']} tokens ({x[k]['finish']})")
    print("ALL IDENTICAL" if not bad else f"{bad} DIFFER")
    sys.exit(1 if bad else 0)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("ttft")
    t.add_argument("--base", required=True)
    t.add_argument("--reply", type=int, default=1000)
    t.add_argument("--prompt", type=int, default=2000)
    t.add_argument("--b-tokens", type=int, default=32)
    t.add_argument("--after-s", type=float, default=1.5, help="the second request's delay after the first's send")
    t.add_argument("--nonce", type=int, default=0x5eed, help="prompt nonce (the same prompts in every run)")
    t.add_argument("--a-prompt", type=int, default=0, help="the first request's prompt (tokens; 0 = the coding prompt)")
    s = sub.add_parser("streams")
    s.add_argument("--base", required=True)
    s.add_argument("--n", type=int, default=2)
    s.add_argument("--tokens", type=int, default=512)
    s.add_argument("--temperature", type=float, default=0.0)
    s.add_argument("--serial", action="store_true")
    s.add_argument("--nonce", type=int, default=0x5eed, help="prompt nonce (the same prompts in every run)")
    e = sub.add_parser("exact")
    e.add_argument("--base", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--mode", default="alone", choices=["alone", "together", "staggered"])
    e.add_argument("--only", default="")
    tt = sub.add_parser("turns")
    tt.add_argument("--base", required=True)
    tt.add_argument("--out", required=True)
    tt.add_argument("--mode", default="alone", choices=["alone", "together"])
    tt.add_argument("--tokens", type=int, default=300)
    mm = sub.add_parser("memory")
    mm.add_argument("--base", required=True)
    mm.add_argument("--n", type=int, default=2)
    mm.add_argument("--prompt", type=int, default=16000)
    mm.add_argument("--tokens", type=int, default=64)
    mt = sub.add_parser("memtake")
    mt.add_argument("--base", required=True)
    mt.add_argument("--log", required=True, help="the server's log (its KV room)")
    mt.add_argument("--after-s", type=float, default=6.0)
    gr = sub.add_parser("greedy")
    gr.add_argument("--base", required=True)
    gr.add_argument("--mode", default="alone", choices=["alone", "together"])
    gr.add_argument("--n", type=int, default=8)
    gr.add_argument("--tokens", type=int, default=300)
    sg = sub.add_parser("single")
    sg.add_argument("--base", required=True)
    sg.add_argument("--reps", type=int, default=5)
    sg.add_argument("--tokens", type=int, default=512)
    sg.add_argument("--sizes", default="4000,16000", help="cold prompt sizes (tokens) for the time to first token")
    sg.add_argument("--prefill-reps", type=int, default=2)
    sg.add_argument("--nonce", type=int, default=0x5eed, help="prompt nonce (the same prompts in every run)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    {"ttft": cmd_ttft, "streams": cmd_streams, "exact": cmd_exact, "turns": cmd_turns, "memory": cmd_memory,
     "single": cmd_single, "memtake": cmd_memtake, "greedy": cmd_greedy, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
