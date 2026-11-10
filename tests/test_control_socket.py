#!/usr/bin/env python3
"""The control socket's answers match its schema.

`control_test --print` drives the control socket (src/server/control.h) over a real Unix socket and prints every
answer it got as {"schema", "message"}: a status, a stop and a model switch response, each error code it can give
without a model, and a reset event. This checks each against schema/control/<schema>.schema.json. CPU only, no model.
Needs: jsonschema (4.18 or later, for `referencing`).

    python3 tests/test_control_socket.py build/control_test
"""

import json
import os
import subprocess
import sys
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


def main():
    if len(sys.argv) != 2:
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
    for line in failures:
        print("FAIL", line)
    print(f"{len(messages)} messages, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
