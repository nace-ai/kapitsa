#!/usr/bin/env python3
"""The server's events match the control socket's schema.

`events_test --print` publishes every kind of event `kapitsa serve` puts on its bus (src/server/events.h,
src/server/activity.h) and prints each as one SSE frame in schema/control/event.schema.json's JSON. This checks every
frame against that schema, and that every event type the bus publishes appears. CPU only, no model.
Needs: jsonschema (4.18 or later, for `referencing`).

    python3 tests/test_events_schema.py build/events_test
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
PUBLISHED = {"state", "request_started", "request_progress", "request_finished", "log"}


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    schemas = [load(os.path.join(SCHEMA_DIR, n)) for n in sorted(os.listdir(SCHEMA_DIR)) if n.endswith(".schema.json")]
    resources = [(urljoin(ROOT_ID, "../../docs/serve-config.schema.json"),
                  load(os.path.join(ROOT, "docs", "serve-config.schema.json")))]
    resources += [(s["$id"], s) for s in schemas]
    registry = Registry().with_resources(
        (uri, Resource.from_contents(s, default_specification=DRAFT202012)) for uri, s in resources)
    event = load(os.path.join(SCHEMA_DIR, "event.schema.json"))
    validator = Draft202012Validator(event, registry=registry)

    out = subprocess.run([sys.argv[1], "--print"], capture_output=True, text=True, timeout=60, check=True).stdout
    frames = [json.loads(line) for line in out.splitlines() if line.strip()]
    failures = []
    for f in frames:
        for e in validator.iter_errors(f):
            failures.append(f"{f['id']} {f['type']}: {e.message} at {'/'.join(map(str, e.absolute_path))}")
    seen = {f["type"] for f in frames}
    if PUBLISHED - seen:
        failures.append(f"no frame of type {sorted(PUBLISHED - seen)}")
    seqs = [int(f["id"].rsplit(".", 1)[1]) for f in frames]
    if seqs != list(range(1, len(seqs) + 1)):
        failures.append(f"event ids are not 1, 2, 3, ...: {seqs}")
    for line in failures:
        print("FAIL", line)
    print(f"{len(frames)} frames, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
