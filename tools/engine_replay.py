#!/usr/bin/env python3
"""The engine view (src/server/engine.html) against a stand-in server: a recorded session, no model, no GPU.

It serves the page and the control socket's read endpoints (docs/control-socket.md) over HTTP on 127.0.0.1, as
`kapitsa serve` serves them at /engine for this Mac. GET /engine/status and GET /engine/events (SSE, Last-Event-ID
replay and resets) answer from the recording. /engine is read-only: POST /_standin/stop and /_standin/drop are this
stand-in's test hooks (the real server has none; a stop goes through its control socket). A stop ends the run and a
new one starts (the view sees `run_changed`).

Access as on the real server: every request but the page's needs this
process's random token as `Authorization: Bearer` (the page reads it from the URL's #t=; `--token-file` also writes it,
0600), and a foreign Host, Origin or proxy header gets 403 forbidden_peer first.

The recording is built here, tick by tick, in the shapes of schema/control/; its numbers are made up, not measured.

    python3 tools/engine_replay.py              # prints http://127.0.0.1:8199/engine#t=<token>
    python3 tools/engine_replay.py --print      # the frames, one JSON object per line (tests/test_events_schema.py)
"""

import argparse
import hmac
import json
import os
import random
import re
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "src", "server", "engine.html")
TICK_MS = 500
BUFFER = 1024  # events a run keeps for replay, as the real bus does
MAX_BODY = 65536
HOST_RE = re.compile(r"(127\.0\.0\.1|localhost|\[::1\])(:[0-9]{1,5})?", re.I)
PROXY_HEADERS = ("X-Forwarded-For", "Forwarded", "Via")
CURSOR_RE = re.compile(r"([0-9a-f]{16,64})\.([0-9]{1,19})")
# /engine/* is read-only, as on the real server: stop and switch go through the control socket there (Kapitan, or
# Ctrl-C where kapitsa serve runs). /_standin/* are this stand-in's own test hooks, so tests can drive a stop (and with
# {"then": "exit"} one that no new run follows) and a drop (every event stream ends cleanly, the run goes on); the real
# server has no such routes over HTTP.
ROUTES = {"/": "GET", "/engine": "GET", "/engine/status": "GET", "/engine/events": "GET",
          "/_standin/stop": "POST", "/_standin/drop": "POST"}
PAGE_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; "
            "img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
GB = 1_000_000_000
MODELS = {  # id: (context_length, slots, weights_bytes, kv_bytes per slot, decode tokens per second)
    "qwen3.6-27b": (131072, 2, 17_400_000_000, 5_950_000_000, 41.0),
    "qwen3.5-4b": (65536, 4, 2_900_000_000, 1_100_000_000, 128.0),
}
CLIENTS = [("openai", "Kapitan/0.9"), ("anthropic", "claude-cli/2.1.4"), ("openai", "opencode/0.15"),
           ("openai", "codex_cli_rs/0.58")]
LOGS = ["SSD prefix cache: {cached} tokens of the conversation read back from disk",
        "KFlash: verify budget {rows} rows (this Mac's cost per row count)",
        "request {rid}: prompt {prompt} tokens, {cached} cached",
        "KFlash: drafter calibration saved"]


class Bus:
    """A run's events: numbered from 1, the newest `cap` kept (all of them when cap is None)."""

    def __init__(self, run_id, cap):
        self.run_id, self.cap, self.seq, self.frames = run_id, cap, 0, []

    @property
    def first_seq(self):
        return self.seq - len(self.frames) + 1

    def publish(self, type_, data):
        self.seq += 1
        self.frames.append({"id": f"{self.run_id}.{self.seq}", "type": type_, "data": data})
        if self.cap and len(self.frames) > self.cap:
            del self.frames[:len(self.frames) - self.cap]

    def after(self, seq):
        """the held frames newer than event `seq`"""
        return self.frames[max(0, seq + 1 - self.first_seq):]

    def cursor(self, last):
        """(reset reason or None, seq) for a Last-Event-ID header, as docs/control-socket.md, Events, says"""
        m = CURSOR_RE.fullmatch(last)
        if not m:
            return "invalid_cursor", None
        if m[1] != self.run_id:
            return "run_changed", None
        seq = int(m[2])
        if seq < 1 or seq > self.seq:  # seq counts from 1; a later one never existed
            return "invalid_cursor", None
        if seq < self.first_seq - 1:  # the event after it is gone
            return "expired", None
        return None, seq


class Run(Bus):
    """One run of a stand-in instance: its state, requests and engine, advanced one tick at a time. Every change is an
    event on its bus; status is read from the same fields."""

    def __init__(self, run_id, model="qwen3.6-27b", seed=7, cap=None):
        super().__init__(run_id, cap)
        self.rng = random.Random(seed)
        self.t, self.state, self.model_id, self.target = 0, "starting", None, model
        self.load_left, self.requests, self.slots, self.drafts = 0, {}, [], {"cycles": 0, "tokens": 0}
        self.next_req, self.next_arrival, self.started = 1, 4.0, time.time()
        self.publish("state", {"state": "starting", "model": None})

    def model(self, mid=None):
        mid = mid or self.model_id
        if not mid:
            return None
        ctx, slots, *_ = MODELS[mid]
        return {"id": mid, "context_length": ctx, "slots": slots}

    def load(self, mid, state):
        # loading at start: no model yet; switching: the model that served before (docs/control-socket.md, Status)
        self.target, self.state, self.load_left = mid, state, 6
        self.publish("state", {"state": state, "model": self.model()})

    def ready(self):
        self.model_id, self.state = self.target, "ready"
        self.slots = [{"index": i, "busy": False, "tokens": 0} for i in range(MODELS[self.model_id][1])]
        self.drafts = {"cycles": 0, "tokens": 0}  # counted since the model loaded
        self.publish("state", {"state": "ready", "model": self.model()})
        self.publish("log", {"level": "info", "message": f"{self.model_id} ready: {self.model()['context_length']} "
                                                        f"tokens of context, {len(self.slots)} slots"})

    def engine(self):
        ctx, nslots, weights, kv_slot, _ = MODELS[self.model_id]
        kv = sum(min(kv_slot, kv_slot * max(s["tokens"], 4096) // ctx * 4) for s in self.slots)
        return {"memory": {"budget_bytes": 40 * GB, "allocated_bytes": weights + kv + 1_350_000_000,
                           "weights_bytes": weights, "kv_bytes": kv},
                "slots": [dict(s) for s in self.slots], "drafts": dict(self.drafts)}

    def arrive(self):
        free = [s for s in self.slots if not s["busy"]]
        if not free:
            return
        api, client = CLIENTS[(self.next_req - 1) % len(CLIENTS)]
        rid = f"req-{self.next_req}"
        self.next_req += 1
        slot = free[0]
        cached = self.rng.choice([0, 0, 8192, 16384, 24576])
        prompt = cached + self.rng.randint(300, 9000)
        slot.update(busy=True, request_id=rid, tokens=cached)
        self.requests[rid] = {"request_id": rid, "api": api, "client": client, "phase": "prefill",
                              "prompt_tokens": prompt, "cached_tokens": cached, "prefilled_tokens": 0,
                              "completion_tokens": 0, "elapsed_ms": 0, "tokens_per_second": 0.0,
                              "_slot": slot["index"], "_want": self.rng.randint(180, 900), "_prefill_ms": 0}
        self.publish("request_started", {"request_id": rid, "api": api, "client": client})
        if cached:
            self.publish("log", {"level": "info", "message": LOGS[0].format(cached=cached)})

    def step_request(self, r, dt):
        slot = self.slots[r["_slot"]]
        r["elapsed_ms"] += int(dt * 1000)
        if r["phase"] == "prefill":
            left = r["prompt_tokens"] - r["cached_tokens"] - r["prefilled_tokens"]
            n = min(left, int(self.rng.uniform(780, 920) * dt))
            r["prefilled_tokens"] += n
            slot["tokens"] += n
            r["_prefill_ms"] += int(dt * 1000)
            r["tokens_per_second"] = 0.0
            if r["prefilled_tokens"] >= r["prompt_tokens"] - r["cached_tokens"]:
                r["phase"] = "writing"
            return False
        rate = MODELS[self.model_id][4] * self.rng.uniform(0.85, 1.12) / (1 + 0.18 * (self.busy() - 1))
        n = min(r["_want"] - r["completion_tokens"], max(1, int(rate * dt)))
        r["completion_tokens"] += n
        slot["tokens"] += n
        r["tokens_per_second"] = round(n / dt, 1)
        cycles = max(1, round(n / self.rng.uniform(2.2, 3.1)))
        self.drafts["cycles"] += cycles
        self.drafts["tokens"] += n
        return r["completion_tokens"] >= r["_want"]

    def busy(self):
        return sum(s["busy"] for s in self.slots) or 1

    def finish(self, r, reason="stop"):
        slot = self.slots[r["_slot"]]
        slot.update(busy=False)
        slot.pop("request_id", None)
        decode_ms = max(1, r["elapsed_ms"] - r["_prefill_ms"])
        prefill_ms = max(1, r["_prefill_ms"])
        self.publish("request_finished", {
            "request_id": r["request_id"], "finish_reason": reason, "prompt_tokens": r["prompt_tokens"],
            "cached_tokens": r["cached_tokens"], "completion_tokens": r["completion_tokens"], "queue_ms": 0,
            "prefill_ms": prefill_ms, "decode_ms": decode_ms,
            "tokens_per_second": round(r["completion_tokens"] * 1000 / decode_ms, 1),
            "prefill_tokens_per_second": round(r["prefilled_tokens"] * 1000 / prefill_ms, 1)})
        del self.requests[r["request_id"]]

    def tick(self):
        dt = TICK_MS / 1000
        self.t += dt
        if self.state == "starting" and self.t >= 0.5:
            self.load(self.target, "loading")
        elif self.state in ("loading", "switching"):
            self.load_left -= dt
            if self.load_left <= 0:
                self.ready()
        elif self.state == "ready":
            if self.t >= self.next_arrival:
                self.arrive()
                self.next_arrival = self.t + self.rng.uniform(3, 11)
            for r in list(self.requests.values()):
                if self.step_request(r, dt):
                    self.finish(r)
            for r in self.requests.values():
                self.publish("request_progress", {k: v for k, v in r.items() if not k.startswith("_")})
            if self.rng.random() < 0.04:
                self.publish("log", {"level": "info", "message": LOGS[1].format(rows=self.rng.choice([3, 4, 6]))})
            self.publish("engine", self.engine())

    def status(self, instance):
        s = {"protocol": 1, "instance_id": "default", "run_id": self.run_id, "name": "default", "pid": os.getpid(),
             "version": instance.version, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime(self.started)),
             "lifecycle": "foreground", "state": self.state, "api_url": "http://127.0.0.1:8100",
             "model": self.model() if self.state != "loading" else None,
             "config": {"model": self.target, "port": 8100, "private": True, "context": "128k", "slots": 2},
             "tick_ms": TICK_MS, "last_event_id": f"{self.run_id}.{self.seq}",
             "requests": [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.requests.values()]}
        if self.state == "ready":
            s["engine"] = self.engine()
        return s


class Instance:
    """The stand-in server: its current run, advanced by a clock thread, and the waiters on its events."""

    def __init__(self, args):
        self.args, self.version, self.cond, self.runs = args, "0.2.33", threading.Condition(), 0
        self.token = secrets.token_urlsafe(24)  # one per process: a restarted run keeps it
        self.run = self.new_run()
        self.stop_at, self.exit_on_stop, self.drops = None, False, 0

    def new_run(self):
        self.runs += 1
        rid = "%016x" % random.Random(self.runs * 7919).getrandbits(64)
        return Run(rid, seed=self.runs, cap=self.args.buffer)

    def clock(self):
        while True:
            time.sleep(TICK_MS / 1000 / self.args.speed)
            with self.cond:
                if self.stop_at is not None:
                    if time.time() >= self.stop_at and self.exit_on_stop:  # the process ends, nothing restarts it
                        os._exit(0)
                    if time.time() >= self.stop_at:  # the old run is gone: a new one starts, as launchd would
                        self.run, self.stop_at = self.new_run(), None
                else:
                    self.run.tick()
                self.cond.notify_all()

    def stop(self):
        with self.cond:
            r = self.run
            if r.state != "stopping":
                for req in list(r.requests.values()):
                    if hasattr(r, "finish"):
                        r.finish(req, "cancelled")
                r.state = "stopping"  # status reads it (without `engine`) until the new run starts
                r.publish("state", {"state": "stopping", "model": r.model()})
                self.stop_at = time.time() + self.args.stop_seconds / self.args.speed
            self.cond.notify_all()
            return {"run_id": r.run_id, "state": "stopping"}


def handler(inst):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def head(self, code, ctype, extra=()):
            """status line and the headers every answer carries"""
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in extra:
                self.send_header(k, v)

        def send_json(self, code, body, close=False, extra=()):
            """close: the request's body was not read, so the connection cannot carry another request"""
            raw = json.dumps(body).encode()
            self.head(code, "application/json", extra)
            self.send_header("Content-Length", str(len(raw)))
            if close:
                self.close_connection = True
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def fail(self, code, name, message, close=False, extra=()):
            self.send_json(code, {"error": {"code": name, "message": message}}, close, extra)

        def peer_problem(self):
            """why this request cannot come from this Mac's own browser tab, or None"""
            hosts = self.headers.get_all("Host") or []
            if len(hosts) != 1 or not HOST_RE.fullmatch(hosts[0]):
                return "the Host is not this Mac's loopback address"
            if any(o != "http://" + hosts[0] for o in self.headers.get_all("Origin") or []):
                return "the Origin is another site's"
            if any(self.headers.get(h) is not None for h in PROXY_HEADERS):
                return "the request came through a proxy"
            return None

        def authorized(self):
            auth = self.headers.get("Authorization") or ""
            return auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].encode(), inst.token.encode())

        def do_request(self):
            method, path = self.command, urlsplit(self.path).path
            why = self.peer_problem()
            if why:
                return self.fail(403, "forbidden_peer", why, close=True)
            # as the real gate (src/server/engine_view.cpp): 404 and 405 before the token, which all but the page need
            allowed = ROUTES.get(path)
            if allowed is None:
                return self.fail(404, "not_found", f"no {path} here", close=method != "GET")
            if method != allowed:
                return self.fail(405, "method_not_allowed", f"{path} takes {allowed}", close=method != "GET",
                                 extra=[("Allow", allowed)])
            if path not in ("/", "/engine") and not self.authorized():
                return self.fail(403, "forbidden_peer", "a missing or wrong token", close=True)
            if path in ("/", "/engine"):
                self.page()
            elif path == "/engine/status":
                with inst.cond:
                    self.send_json(200, inst.run.status(inst))
            elif path == "/engine/events":
                self.events()
            else:
                self.post(path)

        do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_request

        def page(self):
            with open(PAGE, "rb") as f:  # with a band that says the numbers are a stand-in's
                raw = f.read().replace(b"<body>", b'<body><div style="background:#c27c0e;color:#fff;font:600 12px '
                                       b'system-ui;padding:4px 22px;letter-spacing:.04em">MOCK: tools/engine_replay.py '
                                       b'replays a made-up session, no model is running</div>', 1)
            self.head(200, "text/html; charset=utf-8", [("Content-Security-Policy", PAGE_CSP),
                                                        ("X-Frame-Options", "DENY")])
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def body(self):
            """the POST's JSON object, or None after answering why not"""
            if self.headers.get_content_type() != "application/json":  # text/plain is what a cross-site form sends
                return self.fail(400, "bad_request", "Content-Type must be application/json", close=True)
            n = self.headers.get("Content-Length") or "0"
            if "Transfer-Encoding" in self.headers or not re.fullmatch(r"[0-9]{1,12}", n):
                return self.fail(400, "bad_request", "the body needs a Content-Length", close=True)
            if int(n) > MAX_BODY:
                return self.fail(400, "bad_request", f"the body is over {MAX_BODY} bytes", close=True)
            try:
                body = json.loads(self.rfile.read(int(n)) or b"{}")
            except ValueError:
                return self.fail(400, "bad_request", "the body is not JSON")
            if not isinstance(body, dict):
                return self.fail(400, "bad_request", "the body is not a JSON object")
            return body

        def post(self, path):
            body = self.body()
            if body is None:
                return
            # The socket's schemas make run_id optional; these hooks require it, as a browser's click would carry the
            # run it saw, and one that names no run must not act on whatever run holds the port.
            if "run_id" not in body:
                return self.fail(400, "bad_request", "run_id is required")
            if body["run_id"] != inst.run.run_id:
                return self.send_json(409, {"error": {"code": "run_id_mismatch", "message": "another run",
                                                      "details": {"run_id": inst.run.run_id}}})
            if path == "/_standin/stop":
                inst.exit_on_stop = body.get("then") == "exit"
                self.send_json(202, inst.stop())
            elif path == "/_standin/drop":
                with inst.cond:
                    inst.drops += 1
                    inst.cond.notify_all()
                self.send_json(202, {})

        def events(self):
            # no length: the stream ends when the connection closes, which is how a viewer sees the run end
            self.close_connection = True
            self.head(200, "text/event-stream", extra=[("Connection", "close")])
            self.end_headers()
            last = self.headers.get("Last-Event-ID")
            with inst.cond:
                run, drops = inst.run, inst.drops
                sent = run.seq  # no cursor: from the next event
                if last:
                    reason, seq = run.cursor(last)
                    if reason:  # a fresh status, whose last_event_id is the cursor from here on
                        st = run.status(inst)
                        self.write([{"id": st["last_event_id"], "type": "reset",
                                     "data": {"reason": reason, "status": st}}])
                    else:
                        sent = seq
            while True:
                with inst.cond:
                    if inst.run is not run or inst.drops != drops:  # the run ended (or a drop): the stream closes
                        return
                    out = run.after(sent)
                    if not out:
                        inst.cond.wait(15)
                        out = run.after(sent)
                    sent = run.seq
                if not self.write(out or None):
                    return

        def write(self, frames):
            try:
                if frames is None:
                    self.wfile.write(b": keepalive\n\n")
                for f in frames or []:
                    self.wfile.write(f"id: {f['id']}\nevent: {f['type']}\ndata: {json.dumps(f['data'])}\n\n".encode())
                self.wfile.flush()
                return True
            except OSError:
                return False

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--speed", type=float, default=1.0, help="ticks run this many times faster than real time")
    ap.add_argument("--token-file", help="also write the access token here (mode 0600)")
    ap.add_argument("--buffer", type=int, default=BUFFER, help="events kept for Last-Event-ID replay")
    ap.add_argument("--stop-seconds", type=float, default=2, help="how long a stopped run reads `stopping`, on its clock")
    ap.add_argument("--print", action="store_true", help="print the default recording's frames and exit")
    ap.add_argument("--seconds", type=float, default=90, help="with --print: this much of the recording")
    args = ap.parse_args()
    if args.print:
        r = Run("5c1e0a7d93b24f18")
        while r.t < args.seconds:
            r.tick()
        for f in r.frames:
            print(json.dumps(f))
        return 0
    inst = Instance(args)
    threading.Thread(target=inst.clock, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler(inst))
    srv.daemon_threads = True
    if args.token_file:
        fd = os.open(args.token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)  # also when the file was there with another mode
        with os.fdopen(fd, "w") as f:
            f.write(inst.token)
    print(f"engine view: http://127.0.0.1:{args.port}/engine#t={inst.token}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
