#!/usr/bin/env python3
"""Cached-turn TTFT against a running nace-infer-server (stdlib only).

usage: ttft.py <base_url> <text_file> [--reps N] [--max-chars C] [--cold-check] [--msg TEXT] [--tag T]

Per rep: turn 1 = a fresh (nonce-prefixed) ~4k/16k-token user message, max_tokens 16, thinking off;
turn 2 = same messages + the assistant reply + a short new user message (cached prefix).
With --cold-check, turn 2 is then re-sent cold (after an unrelated request evicts the slot) and
its output compared with the cached one (temperature 0).
Prints one line per request: tag, turn, prompt/cached tokens, client TTFT, total time, output.
"""
import json, sys, time, urllib.request, random, argparse

ap = argparse.ArgumentParser()
ap.add_argument("base")
ap.add_argument("text")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--max-chars", type=int, default=0)
ap.add_argument("--cold-check", action="store_true")
ap.add_argument("--msg", default="Thanks. Which class is defined first in that file?")
ap.add_argument("--tag", default="")
ap.add_argument("--max-tokens", type=int, default=16)
a = ap.parse_args()

text = open(a.text).read()
if a.max_chars:
    text = text[: a.max_chars]


def post(messages, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "chat_template_kwargs": {"enable_thinking": False}, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(a.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=900)
    out = {"content": "", "usage": None, "ttft": None}
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        c = json.loads(line[6:])
        if "error" in c:
            raise RuntimeError(c["error"])
        ch = c["choices"][0]
        d = ch["delta"]
        if (d.get("content") or d.get("reasoning_content")) and out["ttft"] is None:
            out["ttft"] = time.time() - t0
        out["content"] += d.get("content") or ""
        if ch.get("finish_reason"):
            out["usage"] = c.get("usage")
    out["total"] = time.time() - t0
    return out


def show(turn, r):
    u = r["usage"] or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    print(f"{a.tag}\t{turn}\tprompt={u.get('prompt_tokens')}\tcached={cached}\tttft={r['ttft']:.3f}\t"
          f"total={r['total']:.3f}\tout={r['content'][:60]!r}", flush=True)


post([{"role": "user", "content": "Say hi."}], 2)  # warm-up (pipelines, allocator)
for rep in range(a.reps):
    nonce = f"[session {random.randrange(10**9)}]\n"
    m1 = [{"role": "user", "content": nonce + text + "\n\nWhat does this file implement? One sentence."}]
    r1 = post(m1, a.max_tokens)
    show("turn1", r1)
    m2 = m1 + [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": a.msg}]
    r2 = post(m2, a.max_tokens)
    show("turn2", r2)
    if a.cold_check:
        post([{"role": "user", "content": f"Unrelated {random.randrange(10**9)}: say ok."}], 2)
        r3 = post(m2, a.max_tokens)
        show("turn2cold", r3)
        print(f"{a.tag}\tcheck\tcached==cold: {r2['content'] == r3['content']}", flush=True)
