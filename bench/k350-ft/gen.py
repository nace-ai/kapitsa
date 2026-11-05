#!/usr/bin/env python3
"""Sends items.jsonl to a running `kapitsa serve` (greedy, thinking off), P requests at once; appends each answer to
results.jsonl (resumable: ids already there are skipped). Stops at --deadline (unix seconds).
  gen.py <items.jsonl> <results.jsonl> <port> <parallel> <max_tokens> <deadline>"""
import json, sys, threading, time, urllib.request

items_f, res_f, port, par, maxtok, deadline = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
done = set()
try:
    for l in open(res_f):
        done.add(json.loads(l)["id"])
except FileNotFoundError:
    pass
todo = [json.loads(l) for l in open(items_f)]
todo = [it for it in todo if it["id"] not in done]
lock = threading.Lock()
out = open(res_f, "a")
stats = {"n": 0, "tok": 0, "err": 0}
t0 = time.time()

def body(it):
    msgs = []
    for m in it["messages"]:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {"name": c["function"]["name"],
                                                  "arguments": json.dumps(c["function"]["arguments"])}} for c in m["tool_calls"]]
        msgs.append(m)
    b = {"model": "x", "messages": msgs, "temperature": 0, "max_tokens": maxtok, "stream": False,
         "chat_template_kwargs": {"enable_thinking": False}}
    if it.get("tools"):
        b["tools"] = it["tools"]
    return json.dumps(b).encode()

def worker():
    while True:
        with lock:
            if not todo or time.time() > deadline:
                return
            it = todo.pop(0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body(it),
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            msg = r["choices"][0]["message"]
            rec = {"id": it["id"], "message": msg, "finish": r["choices"][0].get("finish_reason"), "usage": r.get("usage")}
            with lock:
                out.write(json.dumps(rec) + "\n"); out.flush()
                stats["n"] += 1; stats["tok"] += (r.get("usage") or {}).get("completion_tokens", 0)
                if stats["n"] % 25 == 0:
                    dt = time.time() - t0
                    print(f"gen: {stats['n']} done, {stats['tok']} tok, {stats['tok'] / dt:.0f} tok/s, {dt:.0f} s, err {stats['err']}", flush=True)
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["err"] += 1
                print("gen error", it["id"], repr(e)[:200], flush=True)
                if stats["err"] > 50:
                    todo.clear()

ts = [threading.Thread(target=worker) for _ in range(par)]
for t in ts: t.start()
for t in ts: t.join()
print(f"gen: finished {stats['n']} answers, {stats['tok']} tok in {time.time() - t0:.0f} s, errors {stats['err']}", flush=True)
