#!/usr/bin/env python3
"""Cached-turn and short-prompt TTFT against a running server, with the request pattern of the
cross-engine benchmark harness (warmup + t_prefill). Stdlib only.

usage: cached_ttft.py [--base http://127.0.0.1:8097/v1] [--sizes 4000,16000] [--greedy] [--cold-check]
                      [--short] [--no-warmup] [--tag T] [--out results.jsonl] [--seed N] [--corpus FILE]

--corpus: the text the prompts are cut from (default bench/corpus.txt: the harness's frozen corpus, see
bench/harness/corpus/NOTICE.md; another text works, but its numbers are not comparable with the harness's).

Per size: turn 1 = system "[session <nonce>] SYS + repository context" (frozen corpus, 4.25 chars per
token) + the harness question, max_tokens 256; then 3 follow-up user turns (the harness FOLLOWUPS)
appended to the growing history after a 1 s gap each; TTFT = first content/reasoning/tool event.
Sampling as the harness (T 0.6, top_p 0.95, top_k 20, min_p 0, neutral penalties; thinking off via
chat_template_kwargs and reasoning_effort none). --greedy: T 0 / top_k 1 / top_p 1.
--cold-check (with --greedy): after the follow-ups an unrelated request evicts the conversation
(run the server with --slots 1) and the last follow-up is sent again cold; its answer must equal
the cached one. --short: cold TTFT of three short prompts (no system context).
"""
import argparse, json, os, random, statistics, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


DEFAULT_CORPUS = os.path.join(HERE, "..", "corpus.txt")


ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8097/v1")
ap.add_argument("--model", default="x")
ap.add_argument("--sizes", default="4000,16000")
ap.add_argument("--corpus", default=DEFAULT_CORPUS)
ap.add_argument("--samples", type=int, default=3)
ap.add_argument("--turn-gap", type=float, default=1.0)
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--cold-check", action="store_true")
ap.add_argument("--short", action="store_true")
ap.add_argument("--short-reps", type=int, default=3)
ap.add_argument("--no-warmup", action="store_true")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
ap.add_argument("--seed", type=int, default=None, help="seed the nonces (paired A/B runs see the same prompts)")
a = ap.parse_args()
if a.seed is not None:
    random.seed(a.seed)

CHARS_PER_TOKEN = 4.25
WARM_TOKENS, WARM_OFFSET_CHARS = 2700, 100000
SYS = ("You are Nace, a local coding agent. You help the user with their repository. "
       "Be precise. Use tools when you need to read files or run commands.")
QUESTION = "In one sentence, what is the first file in the repository context about?"
FOLLOWUPS = ["Now name one function defined in that file.",
             "Name one class defined in that file, or say there is none.",
             "Which modules does that file import?"]
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a UTF-8 text file from the user's workspace and return its contents.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Workspace-relative path"},
        "start_line": {"type": "integer", "description": "Optional first line (1-based)"}}, "required": ["path"]}}}]
SHORT = ["Write a Python function that reverses a string.",
         "What is the difference between a process and a thread? Answer in two sentences.",
         "Explain what this shell command does: find . -name '*.py' -mtime -1 | xargs grep -n TODO"]
if not os.path.exists(a.corpus):
    sys.exit(f"no corpus at {a.corpus}: pass --corpus FILE (a large text the prompts are cut from)")
corpus = open(a.corpus, encoding="utf-8", newline="").read()
out_f = open(a.out, "a") if a.out else None


def ctx_of(tokens, offset=0):
    want = int(tokens * CHARS_PER_TOKEN)
    if offset + want > len(corpus):  # never a silently short prompt
        sys.exit(f"a {tokens:,}-token prompt at offset {offset:,} needs {offset + want:,} characters of the corpus, "
                 f"and {a.corpus} has {len(corpus):,}: ask for a smaller size or pass a longer --corpus")
    return corpus[offset:offset + want]


def msgs_of(ctx, nonce, q):
    return [{"role": "system", "content": f"[{nonce}] {SYS}\n\n# Repository context\n{ctx}"},
            {"role": "user", "content": q}]


def body(msgs, max_tokens, **extra):
    s = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0,
         "frequency_penalty": 0.0, "repeat_penalty": 1.0}
    if a.greedy:
        s.update(temperature=0.0, top_p=1.0, top_k=1)
    return {"model": a.model, "messages": msgs, "max_tokens": max_tokens, **s,
            "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none",
            "stream": True, "stream_options": {"include_usage": True}, **extra}


def chat(b, session=None):
    h = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    if session:
        h["X-Session-Id"] = session
    req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", data=json.dumps(b).encode(), headers=h)
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=1800)
    first, usage, finish, content, reasoning = None, None, None, [], []
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        ev = json.loads(d)
        if "error" in ev:
            raise RuntimeError(ev["error"])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            dl = ch.get("delta") or {}
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
            got = False
            if dl.get("content"):
                content.append(dl["content"]); got = True
            if dl.get("reasoning_content"):
                reasoning.append(dl["reasoning_content"]); got = True
            if dl.get("tool_calls"):
                got = True
            if got and first is None:
                first = time.perf_counter()
    return {"ttft": (first - t0) if first else None, "total": time.perf_counter() - t0, "usage": usage or {},
            "finish": finish, "content": "".join(content), "reasoning": "".join(reasoning)}


def emit(rec):
    rec["tag"] = a.tag
    print(json.dumps(rec), flush=True)
    if out_f:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()


def cached_of(u):
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens")


if not a.no_warmup:
    w = msgs_of(ctx_of(WARM_TOKENS, WARM_OFFSET_CHARS), f"warmup {random.getrandbits(64):016x}",
                "Summarize the first file in the repository context.")
    r = chat(body(w, 64))
    t = chat(body([{"role": "system", "content": SYS},
                   {"role": "user", "content": "Open src/main.py and tell me what the entry point does."}], 512,
                  tools=TOOLS, tool_choice="auto"))
    emit({"kind": "warmup", "prompt": r["usage"].get("prompt_tokens"), "s": round(r["total"] + t["total"], 2)})

summary = {}
for size in [int(x) for x in a.sizes.split(",") if x]:
    nonce = f"session {random.getrandbits(64):016x}"
    sid = f"bench-{random.getrandbits(48):012x}"
    msgs = msgs_of(ctx_of(size), nonce, QUESTION)
    r = chat(body(msgs, 256), sid)
    emit({"kind": "turn1", "size": size, "prompt": r["usage"].get("prompt_tokens"), "cached": cached_of(r["usage"]),
          "ttft": round(r["ttft"], 4), "finish": r["finish"], "answer": r["content"][:120]})
    hist = msgs + [{"role": "assistant", "content": r["content"]}]
    tt = []
    last = None
    for q in FOLLOWUPS[:a.samples]:
        time.sleep(a.turn_gap)
        m2 = hist + [{"role": "user", "content": q}]
        r2 = chat(body(m2, 256), sid)
        tt.append(r2["ttft"])
        emit({"kind": "followup", "size": size, "prompt": r2["usage"].get("prompt_tokens"),
              "cached": cached_of(r2["usage"]), "ttft": round(r2["ttft"], 4), "finish": r2["finish"],
              "answer": r2["content"][:120]})
        last = (m2, r2)
        hist = m2 + [{"role": "assistant", "content": r2["content"]}]
    med = statistics.median(tt)
    summary[size] = med
    emit({"kind": "cached_median", "size": size, "ttft": round(med, 4), "samples": [round(x, 4) for x in tt]})
    if a.cold_check and last:
        chat(body([{"role": "user", "content": f"Unrelated {random.getrandbits(32)}: say ok."}], 4))
        m2, r2 = last
        r3 = chat(body(m2, 256), f"cold-{random.getrandbits(48):012x}")
        same = r3["content"] == r2["content"]
        emit({"kind": "cold_check", "size": size, "prompt": r3["usage"].get("prompt_tokens"),
              "cached": cached_of(r3["usage"]), "ttft": round(r3["ttft"], 4), "equal": same,
              "cached_answer": r2["content"], "cold_answer": r3["content"]})

if a.short:
    for q in SHORT:
        tt = []
        for i in range(a.short_reps):
            m = [{"role": "user", "content": f"[{random.getrandbits(32):08x}] {q}"}]
            r = chat(body(m, 16))
            tt.append(r["ttft"])
            pt = r["usage"].get("prompt_tokens")
        emit({"kind": "short", "prompt": pt, "ttft_median": round(statistics.median(tt), 4),
              "samples": [round(x, 4) for x in tt]})
print("SUMMARY", a.tag, json.dumps({str(k): round(v, 4) for k, v in summary.items()}), flush=True)
