#!/usr/bin/env python3
"""Records a coding agent's traffic to an OpenAI-compatible server and builds a replayable trace. Stdlib only.

  agent_record.py proxy --listen 127.0.0.1:8099 --upstream http://127.0.0.1:8091 --log rec.jsonl [--greedy] [--no-think]
      A recording proxy: point the agent at http://127.0.0.1:8099/v1. Every /v1/chat/completions request is forwarded
      to the upstream server and logged with the exact body the agent sent, the overrides the proxy applied, the
      response (streamed back to the agent as it arrives) and its timings. --greedy and --no-think apply the
      harness's greedy sampling and thinking-off fields (bench.py) to the forwarded request, so the server answers
      the way agent_replay.py later asks every engine to; stream_options.include_usage is always added, so the log
      has the token counts. Other paths are passed through. --mock answers every chat request with "Done." without
      an upstream (to see what an agent sends). --max-requests N answers HTTP 503 after N chat requests (a runaway
      agent stops instead of filling the context).
      --rewrite REAL=SHOWN (repeatable) keeps local paths away from the model: REAL (in all its spellings) becomes
      SHOWN in every request the server gets, and SHOWN becomes REAL again in the tool calls the agent gets back (so
      its tools work on the real files). With --rewrite "$PWD=/work/task" the model sees the conversation exactly as
      the trace stores it, so a replay asks every engine for the same answers the recording got. --anonymize also
      turns the home folder, the user name and the Mac's name into /home/user, user and host on the way in. The log
      holds the requests as the server got them.

  agent_record.py build rec.jsonl --workdir DIR [--as /work/task] --out trace.json [--name N] [--meta JSON] [--forbid S,...]
      The trace for agent_replay.py: the chat requests in the order they arrived (messages and tools stored once and
      referenced by index), each with its settings, the overrides applied when it was recorded, the response's token
      counts, text and timings, and the gap before it. Local details are rewritten: DIR (and its /private and
      symlink-resolved spellings) becomes --as, each --replace OLD=NEW pair is applied, the home folder becomes
      /home/user, the user name user and the host name host; the build fails if a --forbid string (default: the user
      name, the host name and the home folder, in any case) is still in the trace.
"""
import argparse, copy, getpass, hashlib, http.client, json, os, re, socket, sys, threading, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TRACE_SCHEMA = 1
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "min_p": 0.0, "presence_penalty": 0.0,
          "frequency_penalty": 0.0, "repeat_penalty": 1.0}
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}
HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "content-length", "host", "accept-encoding"}


def deep_replace(v, pairs):
    """v (any JSON value) with every string's occurrences of each pair's first item replaced by its second."""
    if not pairs:
        return v
    if isinstance(v, str):
        for a, b in pairs:
            if a in v:
                v = v.replace(a, b)
        return v
    if isinstance(v, list):
        return [deep_replace(x, pairs) for x in v]
    if isinstance(v, dict):
        return {deep_replace(k, pairs): deep_replace(x, pairs) for k, x in v.items()}
    return v


def back_translate_event(ev, pairs):
    """Rewrites the tool-call arguments of one streamed (delta) or complete (message) chat event in place; True if
    anything changed."""
    changed = False
    for ch in ev.get("choices") or []:
        for part in (ch.get("delta"), ch.get("message")):
            for tc in (part or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                a = fn.get("arguments")
                if isinstance(a, str):
                    b = deep_replace(a, pairs)
                    if b != a:
                        fn["arguments"] = b
                        changed = True
    return changed


# ---------------------------------------------------------------- proxy
class Recorder:
    def __init__(self, log_path, upstream, greedy, no_think, mock, max_requests=0, rewrites=(), anonymize=False):
        self.log_path, self.greedy, self.no_think, self.mock = log_path, greedy, no_think, mock
        self.max_requests = max_requests
        self.fwd, self.back = [], []
        for real, shown in rewrites:
            self.fwd += [(x, shown) for x in sorted(spellings(real), key=len, reverse=True)]
            self.back.append((shown, os.path.realpath(real)))
        self.fwd.sort(key=lambda p: len(p[0]), reverse=True)
        if anonymize:
            self.fwd += anonymizing_pairs()
        u = urllib.parse.urlparse(upstream or "http://127.0.0.1:1")
        self.host, self.port, self.prefix = u.hostname, u.port or 80, u.path.rstrip("/")
        self.lock = threading.Lock()
        self.seq = 0

    def next_seq(self):
        with self.lock:
            self.seq += 1
            return self.seq

    def write(self, rec):
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def overrides(self, body):
        """The fields this proxy sets on the forwarded request (recorded with it)."""
        o = {}
        if self.greedy:
            o.update(GREEDY)
        if self.no_think:
            o.update(copy.deepcopy(NO_THINK))
        if body.get("stream"):
            o["stream_options"] = dict(body.get("stream_options") or {}, include_usage=True)
        return o


class StreamCapture:
    """Accumulates an SSE chat stream: text, tool calls, finish, usage and event times."""

    def __init__(self, t0):
        self.t0 = t0
        self.content, self.reasoning, self.tcs = [], [], {}
        self.finish = self.usage = self.timings = None
        self.first = self.last = None
        self.events = 0
        self.buf = b""

    def feed(self, chunk):
        self.buf += chunk
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            self.line(line.decode("utf-8", "ignore").strip())

    def line(self, line):
        if not line.startswith("data:"):
            return
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            ev = json.loads(data)
        except Exception:
            return
        if ev.get("usage"):
            self.usage = ev["usage"]
        if ev.get("timings"):
            self.timings = ev["timings"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]
            got = False
            if d.get("content"):
                self.content.append(d["content"]); got = True
            rc = d.get("reasoning_content") or d.get("reasoning")
            if rc:
                self.reasoning.append(rc); got = True
            for tc in d.get("tool_calls") or []:
                got = True
                e = self.tcs.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    e["id"] = tc["id"]
                fn = tc.get("function") or {}
                e["name"] += fn.get("name") or ""
                a = fn.get("arguments")
                if a:
                    e["arguments"] += a if isinstance(a, str) else json.dumps(a)
            if got:
                now = time.time()
                self.events += 1
                self.first = self.first or now
                self.last = now

    def result(self):
        return {"content": "".join(self.content), "reasoning": "".join(self.reasoning),
                "tool_calls": [self.tcs[k] for k in sorted(self.tcs)], "finish": self.finish, "usage": self.usage,
                "timings": self.timings, "events": self.events,
                "ttft_s": round(self.first - self.t0, 4) if self.first else None,
                "gen_s": round(self.last - self.first, 4) if self.first else None}


def make_handler(rec):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _fwd_headers(self):
            return {k: v for k, v in self.headers.items() if k.lower() not in HOP_HEADERS}

        def _relay(self, method, body):
            """Pass a request through unchanged (models list, health, ...)."""
            if rec.mock:
                if self.path.rstrip("/").endswith("/models"):
                    out = json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]}).encode()
                    self._send(200, out, "application/json")
                else:
                    self._send(404, b'{"error":"mock"}', "application/json")
                return
            c = http.client.HTTPConnection(rec.host, rec.port, timeout=3600)
            c.request(method, rec.prefix + self.path, body=body or None, headers=self._fwd_headers())
            r = c.getresponse()
            data = r.read()
            self._send(r.status, data, r.getheader("Content-Type") or "application/octet-stream")
            c.close()

        def _send(self, status, data, ctype):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _back(self, data):
            """SSE lines with the tool calls' arguments translated back to the real paths."""
            out = []
            for line in data.split(b"\n"):
                t = line.strip()
                if t.startswith(b"data:") and t[5:].strip() not in (b"", b"[DONE]"):
                    try:
                        ev = json.loads(t[5:].strip())
                        if back_translate_event(ev, rec.back):
                            line = b"data: " + json.dumps(ev, ensure_ascii=False).encode()
                    except Exception:
                        pass
                out.append(line)
            return b"\n".join(out)

        def do_GET(self):
            self._relay("GET", None)

        def do_POST(self):
            raw = self._body()
            if not re.search(r"/chat/completions/?$", self.path.split("?")[0]):
                self._relay("POST", raw)
                return
            seq = rec.next_seq()
            t0 = time.time()
            if rec.max_requests and seq > rec.max_requests:
                print(f"[{time.strftime('%H:%M:%S')}] #{seq} refused: --max-requests {rec.max_requests}", flush=True)
                self._send(503, json.dumps({"error": {"message": "recording limit reached", "type": "server_error"}}).encode(),
                           "application/json")
                return
            try:
                body = json.loads(raw)
            except Exception:
                self._relay("POST", raw)
                return
            body = deep_replace(body, rec.fwd)  # what the server (the model) gets: local paths rewritten
            ovr = rec.overrides(body)
            fwd = dict(body)
            for k, v in ovr.items():
                fwd[k] = v
            entry = {"seq": seq, "t_start": t0, "path": self.path,
                     "headers": deep_replace({k: v for k, v in self.headers.items()
                                              if k.lower() in ("user-agent", "x-session-id", "openai-organization")
                                              or k.lower().startswith("x-")}, rec.fwd),
                     "body": body, "overrides": ovr, "rewritten": bool(rec.fwd)}
            if rec.mock:
                self._mock(fwd, entry, t0)
                return
            c = http.client.HTTPConnection(rec.host, rec.port, timeout=3600)
            hdr = self._fwd_headers()
            hdr["Content-Type"] = "application/json"
            data = json.dumps(fwd).encode()
            try:
                c.request("POST", rec.prefix + self.path, body=data, headers=hdr)
                r = c.getresponse()
            except Exception as e:
                entry.update({"status": 502, "error": repr(e)[:300], "t_end": time.time()})
                rec.write(entry)
                self._send(502, json.dumps({"error": {"message": repr(e)}}).encode(), "application/json")
                return
            entry["status"] = r.status
            ctype = r.getheader("Content-Type") or ""
            if fwd.get("stream") and r.status == 200 and "event-stream" in ctype:
                cap = StreamCapture(t0)
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                client_ok = True
                pending = b""
                while True:
                    chunk = r.read1(65536) if hasattr(r, "read1") else r.read(4096)
                    if not chunk:
                        out, pending = (self._back(pending) if rec.back else pending), b""
                    else:
                        cap.feed(chunk)
                        if rec.back:  # whole lines only, so each event can be rewritten
                            pending += chunk
                            cut = pending.rfind(b"\n") + 1
                            out, pending = self._back(pending[:cut]), pending[cut:]
                        else:
                            out = chunk
                    if out and client_ok:
                        try:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(out), out))
                            self.wfile.flush()
                        except Exception:
                            client_ok = False  # the agent hung up; keep reading so the record is complete
                    if not chunk:
                        break
                if client_ok:
                    try:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
                    except Exception:
                        pass
                entry["response"] = cap.result()
                entry["client_disconnected"] = not client_ok
            else:
                out = r.read()
                entry["response_raw"] = out.decode("utf-8", "ignore")[:200000]
                if rec.back:
                    try:
                        j = json.loads(out)
                        if back_translate_event(j, rec.back):
                            out = json.dumps(j, ensure_ascii=False).encode()
                    except Exception:
                        pass
                try:
                    j = json.loads(out)
                    m = (j.get("choices") or [{}])[0].get("message") or {}
                    entry["response"] = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
                                         "tool_calls": [{"id": t.get("id"), "name": t["function"]["name"],
                                                         "arguments": t["function"]["arguments"]}
                                                        for t in m.get("tool_calls") or []],
                                         "finish": (j.get("choices") or [{}])[0].get("finish_reason"),
                                         "usage": j.get("usage"), "events": 0, "ttft_s": None, "gen_s": None}
                except Exception:
                    pass
                self._send(r.status, out, ctype or "application/json")
            c.close()
            entry["t_end"] = time.time()
            rec.write(entry)
            u = (entry.get("response") or {}).get("usage") or {}
            print(f"[{time.strftime('%H:%M:%S')}] #{seq} {entry['status']} msgs={len(body.get('messages') or [])} "
                  f"tools={len(body.get('tools') or [])} prompt={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
                  f"finish={(entry.get('response') or {}).get('finish')} {entry['t_end'] - t0:.1f}s", flush=True)

        def _mock(self, fwd, entry, t0):
            text = "Done."
            if fwd.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                evs = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
                       {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2}}]
                for ev in evs:
                    s = ("data: " + json.dumps(dict(ev, id="mock", object="chat.completion.chunk", model="mock")) + "\n\n").encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(s), s))
                s = b"data: [DONE]\n\n"
                self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(s), s))
            else:
                self._send(200, json.dumps({"id": "mock", "object": "chat.completion", "model": "mock",
                                            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                         "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 0, "completion_tokens": 2}}).encode(),
                           "application/json")
            entry.update({"status": 200, "mock": True, "t_end": time.time(),
                          "response": {"content": text, "finish": "stop", "usage": {"completion_tokens": 2}}})
            rec.write(entry)
            print(f"[{time.strftime('%H:%M:%S')}] #{entry['seq']} mock msgs={len(fwd.get('messages') or [])} "
                  f"tools={len(fwd.get('tools') or [])}", flush=True)

    return H


def cmd_proxy(a):
    host, _, port = a.listen.rpartition(":")
    rec = Recorder(a.log, a.upstream, a.greedy, a.no_think, a.mock, a.max_requests,
                   [tuple(x.split("=", 1)) for x in a.rewrite], a.anonymize)
    srv = ThreadingHTTPServer((host or "127.0.0.1", int(port)), make_handler(rec))
    srv.daemon_threads = True
    print(f"recording proxy on http://{a.listen}/v1 -> {'mock' if a.mock else a.upstream}; log {a.log}"
          + (" (greedy)" if a.greedy else "") + (" (thinking off)" if a.no_think else "")
          + (f" (the model sees {', '.join(sh for sh, _ in rec.back)}{'; anonymized' if a.anonymize else ''})"
             if rec.fwd else ""),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- build
def host_names():
    """This Mac's distinctive names (host name, local host name), skipping short generic ones such as "Mac"."""
    names = {socket.gethostname().split(".")[0]}
    try:
        import subprocess
        for key in ("LocalHostName", "ComputerName"):
            names.add(subprocess.run(["scutil", "--get", key], capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        pass
    return sorted((n for n in names if len(n) >= 6 and n.lower() not in ("localhost", "macbook", "mac-mini")),
                  key=len, reverse=True)


def spellings(path):
    """A path as macOS may print it: as given, symlinks resolved, with and without /private."""
    out = set()
    for p in (path, os.path.realpath(path)):
        p = p.rstrip("/")
        out.add(p)
        if p.startswith("/private/"):
            out.add(p[len("/private"):])
        elif p.startswith(("/tmp/", "/var/")):
            out.add("/private" + p)
    return out


def anonymizing_pairs():
    """home folder -> /home/user, this Mac's names -> host, user name -> user (in that order)."""
    pairs = [(os.path.expanduser("~").rstrip("/"), "/home/user")]
    pairs += [(h, "host") for h in host_names()]
    if getpass.getuser():
        pairs.append((getpass.getuser(), "user"))
    return pairs


def sanitizer(workdir, as_path, home, replace=()):
    """A function that rewrites local details in any JSON value: the work dir (all its spellings) -> as_path, then
    the --replace pairs (paths among them in all spellings), the home folder -> /home/user, the user name -> user and
    the host name -> host."""
    pairs = [(s, as_path) for s in sorted(spellings(workdir), key=len, reverse=True)]
    for old, new in replace:
        olds = spellings(old) if old.startswith("/") else {old}
        pairs += [(s, new) for s in sorted(olds, key=len, reverse=True)]
    if home:
        pairs.append((home.rstrip("/"), "/home/user"))
    pairs += [(h, "host") for h in host_names()]
    user = getpass.getuser()
    if user:
        pairs.append((user, "user"))

    def fix(v):
        if isinstance(v, str):
            for s, r in pairs:
                v = v.replace(s, r)
            return v
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {fix(k): fix(x) for k, x in v.items()}
        return v
    return fix


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def classify(reqs):
    """main = requests that carry the agent's tools and its system prompt (the agent loop); everything else (title or
    summary requests with their own prompts) is aux."""
    def sys_of(b):
        m = (b.get("messages") or [{}])[0]
        return m.get("content") if m.get("role") == "system" else None
    main_sys = None
    for r in reqs:
        if r["body"].get("tools"):
            main_sys = canon(sys_of(r["body"]))
            break
    out = []
    for r in reqs:
        b = r["body"]
        out.append("main" if b.get("tools") and canon(sys_of(b)) == main_sys else "aux")
    return out


def cmd_build(a):
    recs = [json.loads(l) for l in open(a.log, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("status") == 200 and isinstance(r.get("body"), dict)]
    if not recs:
        raise SystemExit("no successful chat requests in the log")
    recs.sort(key=lambda r: r["t_start"])
    home = os.path.expanduser("~")
    fix = sanitizer(a.workdir, a.as_path, home, [tuple(x.split("=", 1)) for x in a.replace])
    t_first = recs[0]["t_start"]
    kinds = classify(recs)
    msg_pool, msg_ix, tool_pool, tool_ix = [], {}, [], {}
    requests = []
    prev_main_end = None
    for r, kind in zip(recs, kinds):
        b = fix(r["body"])
        ids = []
        for m in b.get("messages") or []:
            k = canon(m)
            if k not in msg_ix:
                msg_ix[k] = len(msg_pool)
                msg_pool.append(m)
            ids.append(msg_ix[k])
        tset = None
        if b.get("tools") is not None:
            k = canon(b["tools"])
            if k not in tool_ix:
                tool_ix[k] = len(tool_pool)
                tool_pool.append(b["tools"])
            tset = tool_ix[k]
        settings = {k: v for k, v in b.items() if k not in ("messages", "tools")}
        resp = fix(r.get("response") or {})
        u = resp.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
        rq = {"i": len(requests), "kind": kind, "messages": ids, "tools": tset, "settings": settings,
              "overrides": r.get("overrides") or {}, "t_start_s": round(r["t_start"] - t_first, 3),
              "gap_before_s": round(r["t_start"] - prev_main_end, 3) if kind == "main" and prev_main_end else None,
              "response": {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                           "cached_tokens": cached, "finish": resp.get("finish"),
                           "ttft_s": resp.get("ttft_s"), "total_s": round(r["t_end"] - r["t_start"], 3),
                           "content": resp.get("content") or "", "reasoning": resp.get("reasoning") or "",
                           "tool_calls": resp.get("tool_calls") or []}}
        if r.get("headers"):
            rq["headers"] = fix(r["headers"])
        requests.append(rq)
        if kind == "main":
            prev_main_end = r["t_end"]
    main = [q for q in requests if q["kind"] == "main"]
    pt = [q["response"]["prompt_tokens"] or 0 for q in main]
    ct = [q["response"]["completion_tokens"] or 0 for q in main]
    peak = max((p + c for p, c in zip(pt, ct)), default=0)
    meta = json.loads(a.meta) if a.meta else {}
    trace = {"schema": TRACE_SCHEMA, "name": a.name or os.path.splitext(os.path.basename(a.out))[0],
             **meta,
             "workdir": a.as_path,
             "stats": {"requests": len(requests), "main_turns": len(main), "aux_requests": len(requests) - len(main),
                       "peak_context_tokens": peak, "max_prompt_tokens": max(pt, default=0),
                       "first_prompt_tokens": pt[0] if pt else None,
                       "completion_tokens": sum(ct), "prompt_tokens_sum": sum(pt),
                       "recorded_engine_s": round(sum(q["response"]["total_s"] for q in requests), 2),
                       "recorded_wall_s": round(recs[-1]["t_end"] - t_first, 2),
                       "finish": {f: sum(1 for q in main if q["response"]["finish"] == f)
                                  for f in sorted({q["response"]["finish"] or "none" for q in main})}},
             "messages": msg_pool, "tool_sets": tool_pool, "requests": requests}
    text = json.dumps(trace, ensure_ascii=False, indent=1)
    forbid = [x for x in (a.forbid.split(",") if a.forbid else [getpass.getuser(), home] + host_names()) if x]
    leaks = sorted({f for f in forbid if f.lower() in text.lower()})
    if leaks:
        for f in leaks:
            i = text.lower().find(f.lower())
            print(f"leak: {f!r} near ...{text[max(0, i - 80):i + 80]!r}...", file=sys.stderr)
        raise SystemExit(f"refusing to write {a.out}: it still contains {leaks}")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    s = trace["stats"]
    print(f"{a.out}: {s['main_turns']} turns (+{s['aux_requests']} other requests), first prompt "
          f"{s['first_prompt_tokens']} tokens, peak context {s['peak_context_tokens']}, {s['completion_tokens']} tokens out, "
          f"recorded in {s['recorded_wall_s']} s; sha256 {hashlib.sha256((text + chr(10)).encode()).hexdigest()[:12]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proxy")
    p.add_argument("--listen", default="127.0.0.1:8099")
    p.add_argument("--upstream", default="http://127.0.0.1:8091")
    p.add_argument("--log", required=True)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--max-requests", type=int, default=0)
    p.add_argument("--rewrite", action="append", default=[], metavar="REAL=SHOWN")
    p.add_argument("--anonymize", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("log")
    b.add_argument("--workdir", required=True, help="the agent's working directory while recording")
    b.add_argument("--as", dest="as_path", default="/work/task", help="what the work dir is called in the trace")
    b.add_argument("--out", required=True)
    b.add_argument("--name", default=None)
    b.add_argument("--meta", default=None, help="JSON merged into the trace's top level (task, agent, recording)")
    b.add_argument("--forbid", default=None, help="comma list of strings that must not appear in the trace")
    b.add_argument("--replace", action="append", default=[], metavar="OLD=NEW",
                   help="another rewrite, e.g. the agent's temp dir: --replace /var/folders/x/T=/tmp (repeatable)")
    a = ap.parse_args()
    cmd_proxy(a) if a.cmd == "proxy" else cmd_build(a)


if __name__ == "__main__":
    main()
