#!/usr/bin/env python3
"""The engine view's stand-in server speaks the control socket's protocol, and the page asks nothing of the internet.

tools/engine_replay.py is what the view (src/server/engine.html) is built and checked against before `kapitsa serve`
serves it, so its answers must be the real server's: status, events with Last-Event-ID resume and the three resets,
errors, each checked against schema/control/, and the access contract of the HTTP mirror (a per-run bearer token, Host,
Origin and proxy-header refusals, security headers, read-only /engine). The stand-in's own /_standin/stop and
/_standin/model hooks drive a stop and a switch. tests/test_events_schema.py checks its events.
CPU only, no model. Needs: jsonschema (4.18 or later, for `referencing`).

    python3 tests/test_engine_view.py
"""

import http.client
import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from urllib.parse import urljoin

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(ROOT, "schema", "control")
ROOT_ID = "https://github.com/nace-ai/kapitsa/blob/main/schema/control/control.schema.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def validators():
    schemas = {n: load(os.path.join(SCHEMA_DIR, n)) for n in os.listdir(SCHEMA_DIR) if n.endswith(".schema.json")}
    resources = [(urljoin(ROOT_ID, "../../docs/serve-config.schema.json"),
                  load(os.path.join(ROOT, "docs", "serve-config.schema.json")))]
    resources += [(s["$id"], s) for s in schemas.values()]
    reg = Registry().with_resources((u, Resource.from_contents(s, default_specification=DRAFT202012)) for u, s in resources)
    return {n.split(".")[0]: Draft202012Validator(s, registry=reg) for n, s in schemas.items()}


def req(port, method, path, token=None, body=None, headers=None, ctype="application/json", raw=None, stream=False):
    """One request as (status, headers, body); stream: the answer's head only, for an event stream"""
    h = {"Authorization": f"Bearer {token}"} if token else {}
    if method == "POST":
        h["Content-Type"] = ctype
    h.update(headers or {})
    data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=40)
    try:
        c.request(method, path, body=data, headers=h)
        r = c.getresponse()
        return r.status, {k.title(): v for k, v in r.getheaders()}, b"" if stream else r.read()
    finally:
        c.close()


def events(port, token, last=None, seconds=2.0):
    """The frames of GET /engine/events read for `seconds`, as (id, type, data)"""
    s = socket.create_connection(("127.0.0.1", port), timeout=seconds)
    head = (f"GET /engine/events HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nAuthorization: Bearer {token}\r\n"
            + (f"Last-Event-ID: {last}\r\n" if last is not None else "") + "\r\n")
    s.sendall(head.encode())
    raw, end = b"", time.time() + seconds
    try:
        while time.time() < end and (chunk := s.recv(65536)):
            raw += chunk
    except OSError:
        pass
    s.close()
    out = []
    for block in raw.partition(b"\r\n\r\n")[2].decode().split("\n\n"):
        f = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "id" in f:
            out.append((f["id"], f["event"], json.loads(f["data"])))
    return out


def wait_for(fn, timeout=20.0):
    """fn()'s first truthy value, polled for `timeout` seconds, else None"""
    end = time.time() + timeout
    while time.time() < end:
        if v := fn():
            return v
        time.sleep(0.1)
    return None


class Standin:
    """tools/engine_replay.py as a child process, started with a token file in `tmp`"""

    def __init__(self, tmp, name, *args, fails):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        tokfile = os.path.join(tmp, name + ".token")
        self.p = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "engine_replay.py"), "--port",
                                   str(self.port), "--token-file", tokfile, *args], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        line = self.p.stdout.readline()
        with open(tokfile) as f:
            self.token = f.read()
        if len(self.token) < 32 or line != f"engine view: http://127.0.0.1:{self.port}/engine#t={self.token}\n":
            fails.append(f"{name}: start line {line!r}, token {len(self.token)} chars")
        if stat.S_IMODE(os.stat(tokfile).st_mode) != 0o600:
            fails.append(f"{name}: token file mode {oct(stat.S_IMODE(os.stat(tokfile).st_mode))}, want 0o600")

    def get(self, path, **kw):
        return req(self.port, "GET", path, token=self.token, **kw)

    def post(self, path, body=None, **kw):
        return req(self.port, "POST", path, token=self.token, body=body, **kw)

    def status(self):
        code, _, raw = self.get("/engine/status")
        return json.loads(raw) if code == 200 else {}

    def ready(self):
        return wait_for(lambda: (st := self.status()).get("state") == "ready" and st)

    def close(self):
        self.p.terminate()
        self.p.wait(5)


def outside_links(page):
    """what the page names that is not this server: absolute, protocol-relative and WebSocket URLs, CSS url() that is
    not data: or a fragment"""
    found = set(re.findall(rb"https?://[^\s\"'<>)]+", page))
    found |= set(re.findall(rb"[\"'(=]\s*//[^\s\"')]+", page))
    found |= set(re.findall(rb"(?<![\w.$-])wss?:[^\s\"'<>)]*", page))
    found |= set(re.findall(rb"url\(\s*[\"']?(?!data:|#)[^)]*\)", page, re.I))
    return sorted(found)


def main():
    v, fails = validators(), []

    def check(name, kind, body):
        for e in v[kind].iter_errors(body):
            fails.append(f"{name}: {e.message} at {'/'.join(map(str, e.absolute_path))}")

    def expect(name, resp, code, error=None):
        """the answer has this HTTP code and, for an error, this code in an error.schema.json body"""
        got, _, raw = resp
        if got != code:
            fails.append(f"{name}: {got}, want {code}")
        if error:
            try:
                body = json.loads(raw)
            except ValueError:
                return fails.append(f"{name}: body {raw[:80]!r} is not JSON")
            check(name, "error", body)
            if body.get("error", {}).get("code") != error:
                fails.append(f"{name}: error {body.get('error')}, want {error}")

    tmp = tempfile.mkdtemp(prefix="engine-view-")
    servers = []

    def start(name, *args):
        servers.append(Standin(tmp, name, *args, fails=fails))
        return servers[-1]

    try:
        # a long stop window (3 s here), so status can be read while the run reads `stopping`
        a = start("main", "--speed", "20", "--stop-seconds", "60")
        port, tok = a.port, a.token
        run_a = a.status().get("run_id", "")

        # --- the page: no token, nothing outside, the headers
        code, ph, page = req(port, "GET", "/engine")
        left = outside_links(page)
        if code != 200 or left:
            fails.append(f"GET /engine without a token: {code}, outside URLs {left} (the page asks only this server)")
        for ok_host in (f"localhost:{port}", f"[::1]:{port}", "127.0.0.1"):
            if req(port, "GET", "/engine", headers={"Host": ok_host})[0] != 200:
                fails.append(f"Host {ok_host} on the page: refused, want 200")
        every = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"}
        answers = {"page": ph, "status": a.get("/engine/status")[1], "403": req(port, "GET", "/engine/status")[1],
                   "404": a.get("/engine/nope")[1], "events": req(port, "GET", "/engine/events", token=tok,
                                                                  stream=True)[1]}
        for what, h in answers.items():
            for k, want in every.items():
                if h.get(k) != want:
                    fails.append(f"{what} answer: {k} is {h.get(k)!r}, want {want!r}")
        csp = ph.get("Content-Security-Policy", "")
        if "frame-ancestors 'none'" not in csp or "default-src 'none'" not in csp or ph.get("X-Frame-Options") != "DENY":
            fails.append(f"page: CSP {csp!r}, X-Frame-Options {ph.get('X-Frame-Options')!r}")
        if "Content-Security-Policy" in answers["status"]:
            fails.append("status answer carries the page's CSP")

        # --- the token: every other /engine path and every hook
        for path, method in (("/engine/status", "GET"), ("/engine/events", "GET"), ("/_standin/stop", "POST"),
                             ("/_standin/model", "POST")):
            for why, t in (("no token", None), ("a wrong token", "x" * 32), ("a prefix of the token", tok[:-1])):
                expect(f"{method} {path} with {why}", req(port, method, path, token=t, body={"run_id": run_a}), 403,
                       "forbidden_peer")
        expect("Basic scheme", req(port, "GET", "/engine/status", headers={"Authorization": f"Basic {tok}"}), 403,
               "forbidden_peer")
        expect("token in the URL", req(port, "GET", f"/engine/status?t={tok}"), 403, "forbidden_peer")
        # as the real gate: a path or method the view does not have answers 404 or 405 before the token
        expect("GET /engine/nope without the token", req(port, "GET", "/engine/nope"), 404, "not_found")
        expect("POST /engine without the token", req(port, "POST", "/engine", body={}), 405, "method_not_allowed")

        # --- the peer: Host, Origin, proxy headers, refused before the token is looked at, on every path
        bad = [("Host evil.example", {"Host": "evil.example"}), ("Host 127.0.0.1.evil.example",
               {"Host": "127.0.0.1.evil.example"}), ("Host with a foreign port suffix", {"Host": f"localhost:{port}.evil"}),
               ("Origin http://evil.example", {"Origin": "http://evil.example"}),
               ("Origin null", {"Origin": "null"}),
               ("Origin of another port", {"Origin": f"http://127.0.0.1:{port + 1}"}),
               ("X-Forwarded-For", {"X-Forwarded-For": "203.0.113.9"}), ("Forwarded", {"Forwarded": "for=203.0.113.9"}),
               ("Via", {"Via": "1.1 proxy"})]
        for why, h in bad:
            for path in ("/engine", "/engine/status"):
                expect(f"{why} on {path} with the token", req(port, "GET", path, token=tok, headers=h), 403,
                       "forbidden_peer")
            expect(f"{why} without the token", req(port, "GET", "/engine/status", headers=h), 403, "forbidden_peer")
        for why, h in (("the matching Origin", {"Origin": f"http://127.0.0.1:{port}"}),
                       ("Origin and Host localhost", {"Host": f"localhost:{port}", "Origin": f"http://localhost:{port}"})):
            expect(why, a.get("/engine/status", headers=h), 200)
            expect(why + " on the page", req(port, "GET", "/engine", headers=h), 200)

        # --- /engine is read-only; unknown paths and wrong methods answer in the error shape
        for path in ("/engine/stop", "/engine/model", "/engine/x"):
            for method in ("POST", "GET"):
                expect(f"{method} {path}", req(port, method, path, token=tok, body={"run_id": run_a}), 404, "not_found")
        for path in ("/engine", "/engine/status", "/engine/events"):
            expect(f"POST {path}", a.post(path, {"run_id": run_a}), 405, "method_not_allowed")
        expect("GET /_standin/stop", a.get("/_standin/stop"), 405, "method_not_allowed")
        expect("GET /nothing", req(port, "GET", "/nothing"), 404, "not_found")

        # --- the stand-in's hooks: what a cross-site form or a stray click would send
        for why, kw in (("text/plain (a form's, no preflight)", {"ctype": "text/plain", "body": {"run_id": run_a}}),
                        ("no Content-Type value", {"ctype": "", "body": {"run_id": run_a}}),
                        ("no run_id", {"body": {}}), ("a JSON array", {"raw": b"[]"}), ("not JSON", {"raw": b"{"}),
                        ("an oversized Content-Length", {"headers": {"Content-Length": "70000"}})):
            expect(f"stop with {why}", a.post("/_standin/stop", **kw), 400, "bad_request")
        expect("charset parameter", a.post("/_standin/model", {"model": "nope", "run_id": run_a},
                                           ctype="application/json; charset=utf-8"), 404, "model_not_found")
        if a.status().get("run_id") != run_a or a.status().get("state") == "stopping":
            fails.append("a refused POST stopped the run")

        # --- status and events
        st = a.ready()
        if not st:
            fails.append("the stand-in was not ready in 20 s")
            return 1
        check("status", "status", st)
        frames = events(port, tok)
        for fid, t, d in frames:
            check(f"event {fid}", "event", {"id": fid, "type": t, "data": d})
        if not frames:
            fails.append("GET /engine/events: no frames in 2 s")
        else:  # a resume from the middle: exactly the frames after it
            mid = frames[len(frames) // 2][0]
            again = events(port, tok, mid, 1.0)
            want = [f[0] for f in frames[len(frames) // 2 + 1:]]
            if [f[0] for f in again[:len(want)]] != want:
                fails.append(f"resume after {mid}: got {[f[0] for f in again[:3]]}..., want {want[:3]}...")
        st = a.status()

        # --- the resets: every frame is checked against event.schema.json and carries a fresh status
        def reset_case(name, last, reason, srv=a, status=None):
            got = events(srv.port, srv.token, last, 1.0)
            if not got or got[0][1] != "reset" or got[0][2]["reason"] != reason:
                return fails.append(f"reset {name} ({last!r}): {[f[:2] for f in got[:2]]}, want a {reason} reset")
            for fid, t, d in got:
                check(f"reset case {name} {fid}", "event", {"id": fid, "type": t, "data": d})
            rid, _, d = got[0]
            if rid != d["status"]["last_event_id"]:
                fails.append(f"reset {name}: id {rid} is not its status.last_event_id {d['status']['last_event_id']}")
            seqs = [int(f[0].rpartition(".")[2]) for f in got]
            if seqs != sorted(seqs) or any(f[0].rpartition(".")[0] != d["status"]["run_id"] for f in got):
                fails.append(f"reset {name}: what follows is not this run's later events: {[f[0] for f in got[:4]]}")

        run = st["run_id"]
        for name, last in (("garbage", "garbage"), ("a dotted word", "abc.5"), ("a run part that is not hex",
                           "zzzzzzzzzzzzzzzz.5"), ("no seq", run), ("an empty seq", run + "."),
                           ("a signed seq", run + ".-5"), ("an event that never existed", run + ".99999")):
            reset_case(name, last, "invalid_cursor")
        reset_case("another run", "0123456789abcdef.5", "run_changed")

        # --- at most 3 event streams: the next gets 503 busy with Retry-After: 5, and a closed one frees its place
        # (the stand-in notices a closed viewer at its next write, a tick or two later: the opens retry for 5 s)
        def open_stream():
            c = socket.create_connection(("127.0.0.1", port), timeout=5)
            c.sendall(f"GET /engine/events HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {tok}\r\n\r\n".encode())
            if c.recv(64).startswith(b"HTTP/1.1 200"):
                return c
            c.close()

        held = [c for c in (wait_for(open_stream, 5) for _ in range(3)) if c]
        if len(held) != 3:
            fails.append(f"{len(held)} of the first 3 event streams opened")
        time.sleep(2)  # (the earlier checks' streams gone: these 3 are all there are)
        code, h, raw = a.get("/engine/events", stream=True)
        if code == 503:
            code, h, raw = a.get("/engine/events")
        expect("a fourth event stream", (code, h, raw), 503, "busy")
        if code == 503 and h.get("Retry-After") != "5":
            fails.append(f"a fourth event stream: Retry-After {h.get('Retry-After')!r}, want '5'")
        if held:
            held.pop().close()
        if not wait_for(lambda: a.get("/engine/events", stream=True)[0] == 200, 10):
            fails.append("a closed event stream did not free its place in 10 s")
        for c in held:
            c.close()

        # --- a switch (polled to ready first, not slept for), then a stop whose window reads `stopping`
        expect("switch to a missing model", a.post("/_standin/model", {"model": "nope", "run_id": run}), 404,
               "model_not_found")
        expect("stop with an old run_id", a.post("/_standin/stop", {"run_id": "0000000000000000"}), 409,
               "run_id_mismatch")
        a.ready()
        code, _, raw = a.post("/_standin/model", {"model": "qwen3.5-4b", "run_id": run})
        check("switch", "model-switch-response", json.loads(raw))
        if code != 200 or json.loads(raw)["model"]["id"] != "qwen3.5-4b":
            fails.append(f"switch: {code} {raw[:200]!r}")
        a.ready()
        pre = a.status()["last_event_id"]
        code, _, raw = a.post("/_standin/stop", {"run_id": run})
        check("stop", "stop-response", json.loads(raw))
        if code != 202:
            fails.append(f"stop: {code}, want 202")
        during = a.status()
        check("status while stopping", "status", during)
        if during.get("state") != "stopping" or "engine" in during or during.get("run_id") != run:
            fails.append(f"status in the stop window: state {during.get('state')}, engine {'engine' in during}")
        states = [f[2]["state"] for f in events(port, tok, pre, 0.5) if f[1] == "state"]
        if "stopping" not in states:
            fails.append(f"events after the stop: states {states}, want a stopping event")
        if not wait_for(lambda: a.status().get("run_id") not in (None, run)):
            fails.append("no new run 20 s after the stop")
        reset_case("after the stop, the old run's cursor", pre, "run_changed")

        # --- expired: a cursor older than what the (small) buffer holds
        b = start("small", "--speed", "20", "--buffer", "8")
        sb = b.ready()
        if wait_for(lambda: int(b.status().get("last_event_id", "x.0").rpartition(".")[2]) > 30):
            reset_case("older than the buffer", f"{sb['run_id']}.1", "expired", srv=b)
        else:
            fails.append("the small-buffer stand-in published no 30 events")

    finally:
        for srv in servers:
            srv.close()
        shutil.rmtree(tmp, ignore_errors=True)
    for f in fails:
        print("FAIL", f)
    print(f"engine view stand-in: {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
