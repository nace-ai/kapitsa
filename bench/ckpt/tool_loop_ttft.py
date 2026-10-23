#!/usr/bin/env python3
"""TTFT of a user query that follows an agent tool loop, against a running server. Stdlib only.

usage: tool_loop_ttft.py [--base http://127.0.0.1:8097/v1] [--sizes 4000,16000] [--calls 1,2,3] [--reps 1]
                         [--seed S] [--think] [--echo-reasoning] [--temp T] [--tool-tokens 600]
                         [--no-warmup] [--side N] [--tag T] [--out results.jsonl]

Per (size, calls, rep), one conversation in the agent pattern
  system ("[session <nonce>] SYS + repository context", frozen corpus, 4.25 chars per token) + tools,
  user query -> assistant tool call -> tool response -> ... -> final answer -> NEW user query,
where the tool loop is scripted through the tool results: the query asks the model to read
notes/step1.txt and follow the instruction at its end; each file holds a corpus excerpt
(--tool-tokens) and names the next file, the last one says to stop and answer. The model's own
replies are sent back as they came (content + tool_calls; reasoning_content too with
--echo-reasoning), as an agent harness does. Recorded per request: TTFT (first content / reasoning /
tool event), prompt and cached tokens; the record's "calls" is how many tool-call turns the model
actually made. The new query's TTFT is the measurement ("q2"); it is sent --q2-gap seconds (1) after
the final answer, the harness's turn gap.

Sampling: greedy (T 0, top_k 1, top_p 1) unless --temp; thinking off (chat_template_kwargs and
reasoning_effort none) unless --think. --seed fixes the session nonces, so runs against two servers
see the same conversations and their texts can be compared ("texts" holds every reply).
"""
import argparse, hashlib, json, os, random, statistics, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def find_corpus():
    d = HERE
    for _ in range(6):
        c = os.path.join(d, "research", "inference", "bench_corpus.txt")
        if os.path.exists(c):
            return c
        d = os.path.dirname(d)
    return "bench_corpus.txt"


ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8097/v1")
ap.add_argument("--model", default="x")
ap.add_argument("--sizes", default="4000,16000")
ap.add_argument("--calls", default="1,2,3")
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--think", action="store_true")
ap.add_argument("--echo-reasoning", action="store_true")
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--tool-tokens", type=int, default=600)
ap.add_argument("--q2-tokens", type=int, default=128)
ap.add_argument("--q2-gap", type=float, default=1.0)
ap.add_argument("--q2", default="", help="the new user query (default: a one-sentence question)")
ap.add_argument("--no-q2", action="store_true", help="end each conversation after the tool loop")
ap.add_argument("--corpus", default=find_corpus())
ap.add_argument("--no-warmup", action="store_true")
ap.add_argument("--side", type=int, default=0,
                help="side requests (a short title or summary prompt with its own system message, as harnesses send them "
                     "beside the main conversation) sent before every request after the first: how many each time")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()

CHARS_PER_TOKEN = 4.25
SYS = ("You are Nace, a local coding agent. You help the user with their repository. "
       "Be precise. Use tools when you need to read files or run commands.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a UTF-8 text file from the user's workspace and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Workspace-relative path"},
            "start_line": {"type": "integer", "description": "Optional first line (1-based)"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_shell", "description": "Run a shell command in the workspace and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
Q1 = ("Use read_file to open notes/step1.txt and follow the instruction at the end of it. Read one file per "
      "tool call. When a file says it was the last one, stop calling tools and answer as it says.")
Q2 = "Thanks. In one sentence: which programming language was the code in those files written in?"
WORDS = ["amber", "basalt", "cobalt", "dune", "ember", "fjord", "garnet", "harbor"]
corpus = open(a.corpus, encoding="utf-8", newline="").read()
out_f = open(a.out, "a") if a.out else None
rng = random.Random(a.seed)


def excerpt(tokens, offset):
    want = int(tokens * CHARS_PER_TOKEN)
    if offset + want > len(corpus):  # never a silently short prompt
        sys.exit(f"a {tokens:,}-token excerpt at offset {offset:,} needs {offset + want:,} characters of the corpus, "
                 f"and {a.corpus} has {len(corpus):,}: ask for fewer tokens or pass a longer --corpus")
    return corpus[offset:offset + want]


def tool_file(step, calls, word):
    body = excerpt(a.tool_tokens, 118000 + (step - 1) * 4000)
    if step < calls:
        tail = f"Instruction: next, call read_file on notes/step{step + 1}.txt."
    else:
        tail = (f"Instruction: this was the last file. Stop calling tools and reply to the user with the code "
                f"word {word.upper()} and one sentence about what these files contain.")
    return f"# notes/step{step}.txt\n{body}\n\n{tail}"


def sampling():
    if a.temp is None:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": 1}
    return {"temperature": a.temp, "top_p": 0.95, "top_k": 20}


def body(msgs, max_tokens, seed=None):
    b = {"model": a.model, "messages": msgs, "tools": TOOLS, "tool_choice": "auto", "max_tokens": max_tokens,
         "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_penalty": 1.0, **sampling(),
         "stream": True, "stream_options": {"include_usage": True}}
    if a.temp is not None and seed is not None:
        b["seed"] = seed  # sampled runs repeat request for request across servers
    if not a.think:
        b["chat_template_kwargs"] = {"enable_thinking": False}
        b["reasoning_effort"] = "none"
    return b


def chat(b):
    h = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    req = urllib.request.Request(a.base.rstrip("/") + "/chat/completions", data=json.dumps(b).encode(), headers=h)
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=1800)
    first, usage, finish, content, reasoning, calls = None, None, None, [], [], {}
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
            for tc in dl.get("tool_calls") or []:
                got = True
                i = tc.get("index", len(calls))
                c = calls.setdefault(i, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    c["id"] = tc["id"]
                f = tc.get("function") or {}
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""
            if got and first is None:
                first = time.perf_counter()
    u = usage or {}
    return {"ttft": round(first - t0, 4) if first else None, "total": round(time.perf_counter() - t0, 3),
            "prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion": u.get("completion_tokens"), "finish": finish, "content": "".join(content),
            "reasoning": "".join(reasoning), "tool_calls": [calls[i] for i in sorted(calls)]}


def emit(rec):
    rec["tag"] = a.tag
    print(json.dumps(rec), flush=True)
    if out_f:
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()


def step_rec(kind, r):
    # think_call: the reply is empty but its reasoning holds a tool call (the model wrote the call inside an
    # unclosed think block and the server left it there)
    think_call = not r["tool_calls"] and not r["content"].strip() and "<tool_call>" in r["reasoning"]
    return {"kind": kind, "ttft": r["ttft"], "prompt": r["prompt"], "cached": r["cached"],
            "completion": r["completion"], "finish": r["finish"], "think_call": think_call}


def seed_of(nonce, turn):
    return int(hashlib.sha256(f"{nonce}/{turn}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def side_requests(nonce):
    """The harness's side calls: each its own system prompt and a distinct text, so each wants a slot of its own."""
    for k in range(a.side):
        sysmsg = f"[{nonce}/side{k}/{random.getrandbits(32):08x}] Write a title of at most five words for this conversation."
        chat({"model": a.model, "messages": [{"role": "system", "content": sysmsg},
                                              {"role": "user", "content": excerpt(150, 90000 + 1000 * k)}],
              "max_tokens": 12, "temperature": 0.0, "top_k": 1, "stream": True,
              "stream_options": {"include_usage": True},
              "chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"})


def conversation(size, calls, nonce, word):
    sysmsg = f"[{nonce}] {SYS}\n\n# Repository context\n{excerpt(size, 0)}"
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": Q1}]
    steps, texts, made = [], [], 0
    for turn in range(calls + 3):
        if a.side and turn > 0:
            side_requests(nonce)
        r = chat(body(msgs, 4096 if a.think else 1024, seed_of(nonce, turn)))
        steps.append(step_rec("q1" if turn == 0 else "tool", r))
        texts.append({"reasoning": r["reasoning"], "content": r["content"], "tool_calls": [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in r["tool_calls"]]})
        if r["tool_calls"]:
            m = {"role": "assistant", "content": r["content"], "tool_calls": r["tool_calls"]}
            if a.echo_reasoning and r["reasoning"]:
                m["reasoning_content"] = r["reasoning"]
            msgs.append(m)
            made += 1
            for c in r["tool_calls"]:
                try:
                    path = json.loads(c["function"]["arguments"] or "{}").get("path", "")
                except ValueError:
                    path = ""
                step = None
                for s in range(1, calls + 1):
                    if path.endswith(f"step{s}.txt"):
                        step = s
                out = tool_file(step, calls, word) if step else \
                    f"error: {path or 'that file'} does not exist. The next file to read is notes/step{min(made, calls)}.txt."
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
            continue
        m = {"role": "assistant", "content": r["content"]}
        if a.echo_reasoning and r["reasoning"]:
            m["reasoning_content"] = r["reasoning"]
        msgs.append(m)
        break
    if a.no_q2:
        return steps, texts, made
    msgs.append({"role": "user", "content": a.q2 or Q2})
    time.sleep(a.q2_gap)  # the user reads the answer and types (the harness's turn gap)
    if a.side:
        side_requests(nonce)
    r = chat(body(msgs, a.q2_tokens, seed_of(nonce, "q2")))
    steps.append(step_rec("q2", r))
    texts.append({"reasoning": r["reasoning"], "content": r["content"], "tool_calls": [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in r["tool_calls"]]})
    return steps, texts, made


if not a.no_warmup:
    w = [{"role": "system", "content": f"[warmup {random.getrandbits(64):016x}] {SYS}\n\n# Repository context\n"
                                       f"{excerpt(2700, 100000)}"},
         {"role": "user", "content": "Summarize the first file in the repository context."}]
    t0 = time.perf_counter()
    chat(body(w, 64))
    chat(body([{"role": "system", "content": SYS},
               {"role": "user", "content": "Open src/main.py and tell me what the entry point does."}], 256))
    emit({"kind": "warmup", "s": round(time.perf_counter() - t0, 2)})

q2 = {}
totals = {"conversations": 0, "complete": 0, "think_calls": 0}
for size in [int(x) for x in a.sizes.split(",") if x]:
    for calls in [int(x) for x in a.calls.split(",") if x]:
        for rep in range(a.reps):
            nonce = f"session {rng.getrandbits(64):016x}"
            word = WORDS[rng.randrange(len(WORDS))]
            steps, texts, made = conversation(size, calls, nonce, word)
            last = steps[-1]
            if last["kind"] == "q2":
                q2.setdefault((size, calls), []).append(last["ttft"])
            think_calls = sum(1 for st in steps if st.get("think_call"))
            totals["conversations"] += 1
            totals["complete"] += made == calls
            totals["think_calls"] += think_calls
            q = last if last["kind"] == "q2" else {}
            emit({"kind": "conversation", "size": size, "calls_asked": calls, "calls": made, "rep": rep,
                  "nonce": nonce, "q2_ttft": q.get("ttft"), "q2_prompt": q.get("prompt"), "q2_cached": q.get("cached"),
                  "think_calls": think_calls, "steps": steps, "texts": texts})
summary = {f"{s}/{c}": round(statistics.median(v), 4) for (s, c), v in q2.items()}
print("SUMMARY", a.tag, json.dumps(summary), json.dumps(totals), flush=True)
