#!/usr/bin/env python3
"""Stress test of requests in flight at once against a running `kapitsa serve` (stdlib only).

  stress.py --base http://127.0.0.1:8131 [--workers 5] [--seconds 300] [--images]

Each worker loops over random operations until the time is up: streamed and non-streamed chat completions, tool loops
(OpenAI and Anthropic Messages, streamed and not), long prompts (2-6K tokens: several prefill chunks), clients that
disconnect mid-stream, and with --images requests with a picture. Every request has a deadline (--deadline, 600 s): one
that does not finish is a hang. At the end the server must answer a fresh request at once. Prints a summary and exits 1
on any hang, error or wrong answer."""
import argparse, json, os, random, struct, sys, threading, time, urllib.error, urllib.request, zlib, base64

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a text file from disk.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
A_TOOLS = [{"name": "read_file", "description": "Read a text file from disk.",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
HOSTS = "127.0.0.1 localhost\n255.255.255.255 broadcasthost\n::1 localhost\n"


def png(w, h, rgb):
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.ops, self.errors, self.hangs, self.lat = {}, [], [], {}

    def add(self, op, dt, err=None):
        with self.lock:
            self.ops[op] = self.ops.get(op, 0) + 1
            self.lat.setdefault(op, []).append(dt)
            if err:
                self.errors.append(f"{op}: {err}")


def http(base, path, body, deadline, stream=False):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "x-api-key": "k"})
    return urllib.request.urlopen(req, timeout=deadline)


def sse(r):
    """(event, data) pairs of a stream (OpenAI: event None)"""
    name = None
    for raw in r:
        line = raw.decode("utf-8", "ignore").rstrip("\n")
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            d = line[6:]
            if d == "[DONE]":
                return
            yield name, json.loads(d)


def no_think(b):
    b["chat_template_kwargs"] = {"enable_thinking": False}
    return b


def corpus_ctx(tokens):
    with open(os.path.join(HERE, "..", "corpus.txt"), encoding="utf-8") as f:
        c = f.read()
    if int(tokens * 4.25) + 1 >= len(c):  # never a silently short prompt
        sys.exit(f"a {tokens:,}-token prompt needs {int(tokens * 4.25):,} characters of the corpus, which has {len(c):,}")
    off = random.randrange(0, len(c) - int(tokens * 4.25) - 1)
    return c[off:off + int(tokens * 4.25)]


# ---- operations: each returns None or an error string
def op_stream(a):
    n = random.choice([16, 64, 200, 400])
    b = no_think({"messages": [{"role": "user", "content": f"[{random.random()}] Count from 1 to 300, comma separated."}],
                  "max_tokens": n, "stream": True, "temperature": random.choice([0, 0.7]), "seed": random.randrange(1000)})
    got, finish = "", None
    for _, ev in sse(http(a.base, "/v1/chat/completions", b, a.deadline)):
        if "error" in ev:
            return f"stream error {ev['error']}"
        for ch in ev.get("choices") or []:
            got += (ch.get("delta") or {}).get("content") or ""
            finish = ch.get("finish_reason") or finish
    if finish not in ("length", "stop") or not got:
        return f"finish {finish}, {len(got)} chars"
    return None


def op_tool_loop(a):
    msgs = [{"role": "system", "content": "You are a coding agent. Use tools to answer."},
            {"role": "user", "content": f"[{random.random()}] Read the file /etc/hosts and tell me how many lines it has."}]
    for _ in range(3):
        r = json.loads(http(a.base, "/v1/chat/completions",
                            no_think({"messages": msgs, "tools": TOOLS, "max_tokens": 512, "temperature": 0, "top_k": 1}),
                            a.deadline).read())
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        if not calls:
            return None if r["choices"][0]["finish_reason"] in ("stop", "length") and (m.get("content") or "") else \
                f"empty answer {r['choices'][0]['finish_reason']}"
        msgs += [{"role": "assistant", "content": m.get("content") or "", "tool_calls": calls[:1]},
                 {"role": "tool", "tool_call_id": calls[0]["id"], "content": HOSTS}]
    return None


def op_anthropic(a):
    body = {"model": "m", "max_tokens": 1024, "temperature": 0, "top_k": 1, "tools": A_TOOLS, "stream": True,
            "system": "You are a coding agent. Use tools to answer.",
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": f"[{random.random()}] Read /etc/hosts and count its lines."}]}
    for _ in range(3):
        blocks, stop, inputs = [], None, {}
        for name, d in sse(http(a.base, "/v1/messages", body, a.deadline)):
            if name == "content_block_start":
                blocks.append(dict(d["content_block"]))
            elif name == "content_block_delta":
                x = d["delta"]
                if x["type"] == "text_delta":
                    blocks[d["index"]]["text"] += x["text"]
                elif x["type"] == "input_json_delta":
                    inputs[d["index"]] = inputs.get(d["index"], "") + x["partial_json"]
            elif name == "content_block_stop" and blocks[d["index"]]["type"] == "tool_use":
                blocks[d["index"]]["input"] = json.loads(inputs.get(d["index"], "{}"))
            elif name == "message_delta":
                stop = d["delta"]["stop_reason"]
            elif name == "error":
                return f"anthropic error {d}"
        uses = [b for b in blocks if b["type"] == "tool_use"]
        if stop != "tool_use" or not uses:
            return None if stop in ("end_turn", "max_tokens") else f"stop_reason {stop}"
        body["messages"] = body["messages"] + [
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": uses[0]["id"], "content": HOSTS}]}]
    return None


def op_long(a):
    n = random.choice([2000, 4000, 6000])
    b = no_think({"messages": [{"role": "system", "content": f"[{random.random()}] You read code."},
                               {"role": "user", "content": corpus_ctx(n) + "\n\nName one function defined above."}],
                  "max_tokens": 48, "temperature": 0, "top_k": 1})
    r = json.loads(http(a.base, "/v1/chat/completions", b, a.deadline).read())
    if r["usage"]["prompt_tokens"] < n * 0.8:
        return f"prompt_tokens {r['usage']['prompt_tokens']} for {n}"
    return None if r["choices"][0]["message"].get("content") else "empty answer"


def op_abandon(a):
    b = no_think({"messages": [{"role": "user", "content": f"[{random.random()}] Write a long essay about rivers."}],
                  "max_tokens": 2000, "stream": True})
    r = http(a.base, "/v1/chat/completions", b, a.deadline)
    k = random.randint(1, 40)
    for i, _ in enumerate(sse(r)):
        if i >= k:
            break
    r.close()
    return None


def op_abandon_early(a):
    """a client gone right after the request was accepted (maybe still waiting for a slot): it must not run to the end"""
    b = no_think({"messages": [{"role": "user", "content": f"[{random.random()}] Write a long essay about mountains."}],
                  "max_tokens": 2000, "stream": True})
    r = http(a.base, "/v1/chat/completions", b, a.deadline)
    r.readline()  # the first event (the role), sent before the engine takes the request
    r.close()
    return None


def op_image(a):
    color, rgb = random.choice([("red", (220, 20, 20)), ("green", (20, 180, 40)), ("blue", (30, 40, 220))])
    url = "data:image/png;base64," + base64.b64encode(png(64, 64, rgb)).decode()
    b = no_think({"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": f"[{random.random()}] What color is this image? Answer with one word."}]}],
        "max_tokens": 8, "temperature": 0, "top_k": 1})
    r = json.loads(http(a.base, "/v1/chat/completions", b, a.deadline).read())
    ans = (r["choices"][0]["message"].get("content") or "").lower()
    return None if color in ans else f"answered {ans!r} for {color}"


def worker(a, stats, stop_at, ops):
    while time.time() < stop_at:
        name, fn = random.choice(ops)
        t0 = time.time()
        err = None
        try:
            err = fn(a)
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code} {e.read()[:200]!r}"
        except Exception as e:  # noqa: BLE001
            err = repr(e)
        dt = time.time() - t0
        if dt > a.deadline * 0.95 or (err and "timed out" in err):
            stats.hangs.append(f"{name} took {dt:.0f} s")
        stats.add(name, dt, err)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="server root, e.g. http://127.0.0.1:8131")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--deadline", type=float, default=600)
    p.add_argument("--images", action="store_true")
    a = p.parse_args()
    a.base = a.base.rstrip("/").removesuffix("/v1")
    ops = [("stream", op_stream), ("tool_loop", op_tool_loop), ("anthropic", op_anthropic), ("long", op_long),
           ("abandon", op_abandon), ("abandon_early", op_abandon_early)]
    if a.images:
        ops.append(("image", op_image))
    stats = Stats()
    stop_at = time.time() + a.seconds
    th = [threading.Thread(target=worker, args=(a, stats, stop_at, ops)) for _ in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    # the server is still responsive: a fresh request is answered at once
    t0 = time.time()
    err = op_stream(a)
    fresh = time.time() - t0
    print(json.dumps({"ops": stats.ops, "errors": stats.errors[:20], "n_errors": len(stats.errors), "hangs": stats.hangs,
                      "latency_s": {k: [round(min(v), 1), round(sorted(v)[len(v) // 2], 1), round(max(v), 1)]
                                    for k, v in stats.lat.items()},
                      "fresh_request_s": round(fresh, 2), "fresh_error": err}, indent=1))
    ok = not stats.errors and not stats.hangs and err is None and fresh < 30
    print(f"summary: {sum(stats.ops.values())} operations {stats.ops}, {len(stats.errors)} errors, {len(stats.hangs)} hangs, "
          f"a fresh request after {fresh:.2f} s")
    print("STRESS PASSED" if ok else "STRESS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
