#!/usr/bin/env python3
"""The control socket's answers match its schema.

`control_test --print` drives the control socket (src/server/control.h) over a real Unix socket and prints every
answer it got as {"schema", "message"}: statuses, each error code it can give, and a reset event. This checks each
against schema/control/<schema>.schema.json. With build/kapitsa too, it starts `kapitsa serve` on a model folder that
does not exist, in a home where `run/` cannot be a folder and in one whose socket path is over macOS's 103 bytes: each
time the server goes on without a socket. CPU only, no model.
Needs: jsonschema (4.18 or later, for `referencing`).

    python3 tests/test_control_socket.py build/control_test [build/kapitsa]
"""

import json
import os
import subprocess
import sys
import tempfile
from urllib.parse import urljoin

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(ROOT, "schema", "control")
ROOT_ID = "https://github.com/nace-ai/kapitsa/blob/main/schema/control/control.schema.json"
EXPECTED = {"status", "error", "event"}


def load(path):
    with open(path) as f:
        return json.load(f)


def serve(kapitsa, home):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KAPITSA_", "NI_"))}
    env["KAPITSA_HOME"] = home
    args = [kapitsa, "serve", "--model", "/nonexistent/model", "--port", "18731", "--private"]
    return subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def serve_checks(kapitsa):
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
        failures += serve_checks(sys.argv[2])
    for line in failures:
        print("FAIL", line)
    print(f"{len(messages)} messages, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
