#!/usr/bin/env python3
"""The control socket's answers match its schema.

`control_test --print` drives the control socket (src/server/control.h) over a real Unix socket and prints every
answer it got as {"schema", "message"}: statuses, a stop and a model switch response, each error code it can give
without a model, and a reset event. This checks each against schema/control/<schema>.schema.json. With build/kapitsa
too, it starts `kapitsa serve` on a model folder that does not exist: in a home where `run/` cannot be a folder and in
one whose socket path is over macOS's 103 bytes it goes on without a socket, and held in `starting` with SIGINT
ignored, POST /stop still ends it with status 0 and removes the socket. CPU only, no model.
Needs: jsonschema (4.18 or later, for `referencing`).

    python3 tests/test_control_socket.py build/control_test [build/kapitsa]
"""

import json
import os
import pty
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urljoin

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(ROOT, "schema", "control")
ROOT_ID = "https://github.com/nace-ai/kapitsa/blob/main/schema/control/control.schema.json"
EXPECTED = {"status", "stop-response", "model-switch-response", "error", "event"}


def load(path):
    with open(path) as f:
        return json.load(f)


def ask(path, method, target, body=b""):
    """One request over the Unix socket at `path`: (status, JSON body)"""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10)
    s.connect(path)
    s.sendall(f"{method} {target} HTTP/1.1\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
    raw = b""
    while chunk := s.recv(65536):
        raw += chunk
    s.close()
    head, _, rest = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(rest)


def serve(kapitsa, home, hold_ms=0, sigint="ignored"):
    """`kapitsa serve` on a missing model. sigint: "ignored" (a script's background job), "default", or "terminal"
    (stdout and stderr on a pseudo-terminal: the terminal UI and its signal handlers; its output in p.screen)"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KAPITSA_", "NI_"))}
    env["KAPITSA_HOME"] = home
    if hold_ms:
        env["NI_CONTROL_TEST_HOLD_MS"] = str(hold_ms)
    args = [kapitsa, "serve", "--model", "/nonexistent/model", "--port", "18731", "--private"]
    ignore = (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) if sigint == "ignored" else None
    if sigint != "terminal":
        return subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, preexec_fn=ignore)
    master, slave = pty.openpty()
    env["TERM"] = "xterm-256color"
    p = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave)
    os.close(slave)
    p.screen = bytearray()

    def drain():  # (a full pty buffer would block the server)
        try:
            while chunk := os.read(master, 65536):
                p.screen += chunk
        except OSError:
            pass
        os.close(master)

    p.drain = threading.Thread(target=drain, daemon=True)
    p.drain.start()
    return p


def stop_while_starting(kapitsa, check_status, sigint):
    failures = []
    with tempfile.TemporaryDirectory() as home:
        p = serve(kapitsa, home, hold_ms=30000, sigint=sigint)
        sock = os.path.join(home, "run", "default.sock")
        for _ in range(200):
            if os.path.exists(sock) or p.poll() is not None:
                break
            time.sleep(0.05)
        try:
            status, body = ask(sock, "GET", "/status")
            failures += check_status(body)
            if status != 200 or body["state"] != "starting":
                failures.append(f"status while starting: {status} {body}")
            busy, answer = ask(sock, "POST", "/model", b'{"model": "qwen3.6-27b"}')
            if busy != 503 or answer["error"]["code"] != "busy":
                failures.append(f"POST /model before the first model loaded: {busy} {answer}")
            status, body = ask(sock, "POST", "/stop", b'{"run_id": "%s"}' % body["run_id"].encode())
            if status != 202:
                failures.append(f"POST /stop while starting: {status} {body}")
            rc = p.wait(timeout=15)
            key = os.path.join(home, "run", "default.token")
            if rc != 0 or os.path.exists(sock) or os.path.exists(key):
                failures.append(f"stop while starting (SIGINT {sigint}): exit {rc}, want 0 (a stop is not a crash), "
                                f"socket left: {os.path.exists(sock)}, engine view key left: {os.path.exists(key)}")
        except (OSError, subprocess.TimeoutExpired) as e:
            p.kill()
            out = bytes(p.screen) if sigint == "terminal" else p.communicate()[0]
            failures.append(f"stop while starting (SIGINT {sigint}): {e!r}: {out!r}")
    return failures


def serve_checks(kapitsa, check_status):
    failures = []
    with tempfile.TemporaryDirectory() as home:
        open(os.path.join(home, "run"), "w").close()  # run/ cannot be a folder
        out = serve(kapitsa, home).communicate(timeout=60)[0]
        if "no control socket" not in out or "cannot listen" in out or "No folder at" not in out:
            failures.append(f"run/ not a folder: the server should go on without a socket and fail on the model: {out!r}")
    with tempfile.TemporaryDirectory() as tmp:
        home = os.path.join(tmp, "h" * max(1, 110 - len(tmp)))  # run/default.sock over 103 bytes
        os.mkdir(home)
        out = serve(kapitsa, home).communicate(timeout=60)[0]
        if "longer than macOS allows" not in out or "no control socket" not in out or "No folder at" not in out:
            failures.append(f"a socket path over 103 bytes: the server should warn, go on and fail on the model: {out!r}")
    # held in `starting`: POST /stop ends it with status 0 however SIGINT is set, in a terminal too (the terminal UI's
    # handlers installed; `starting` draws no status block, so there is no cursor state to check)
    for sigint in ("ignored", "default", "terminal"):
        failures += stop_while_starting(kapitsa, check_status, sigint)
    return failures


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    names = [n for n in sorted(os.listdir(SCHEMA_DIR)) if n.endswith(".schema.json")]
    schemas = {n[: -len(".schema.json")]: load(os.path.join(SCHEMA_DIR, n)) for n in names}
    resources = [(urljoin(ROOT_ID, "../../docs/serve-config.schema.json"),
                  load(os.path.join(ROOT, "docs", "serve-config.schema.json")))]
    resources += [(s["$id"], s) for s in schemas.values()]
    registry = Registry().with_resources(
        (uri, Resource.from_contents(s, default_specification=DRAFT202012)) for uri, s in resources)

    out = subprocess.run([sys.argv[1], "--print"], capture_output=True, text=True, timeout=120, check=True).stdout
    messages = [json.loads(line) for line in out.splitlines() if line.strip()]
    failures = []
    for i, m in enumerate(messages):
        validator = Draft202012Validator(schemas[m["schema"]], registry=registry)
        for e in validator.iter_errors(m["message"]):
            failures.append(f"#{i} {m['schema']}: {e.message} at {'/'.join(map(str, e.absolute_path))}")
    if EXPECTED - {m["schema"] for m in messages}:
        failures.append(f"no message for {sorted(EXPECTED - {m['schema'] for m in messages})}")
    if len(sys.argv) == 3:
        status = Draft202012Validator(schemas["status"], registry=registry)
        failures += serve_checks(sys.argv[2], lambda body: [f"status: {e.message}" for e in status.iter_errors(body)])
    for line in failures:
        print("FAIL", line)
    print(f"{len(messages)} messages, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
