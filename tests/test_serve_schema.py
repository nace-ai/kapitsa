#!/usr/bin/env python3
"""kapitsa serve's config schema (docs/serve-config.schema.json) and its validator (src/server/service_config.cpp)
give the same answer on the same values, and `--config -` reads the same JSON after the server execs itself again.

Usage: python3 tests/test_serve_schema.py build/serve_options_test   (needs: jsonschema; no model, no GPU)
"""
import json
import os
import subprocess
import sys

import jsonschema

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (field, value, valid): the answer both must give
CASES = [
    ("idle_unload", "auto", True), ("idle_unload", "off", True), ("idle_unload", "0", True), ("idle_unload", 0, True),
    ("idle_unload", 0.5, True), ("idle_unload", 5, True), ("idle_unload", 10080, True),
    ("idle_unload", "soon", False), ("idle_unload", "5", False), ("idle_unload", 0.1, False),
    ("idle_unload", 20000, False), ("idle_unload", -3, False), ("idle_unload", True, False),
    ("context", 1, True), ("context", 131072, True), ("context", 2147483647, True), ("context", "131072", True),
    ("context", "128k", True), ("context", "1M", True), ("context", "max", True), ("context", "MAX", True),
    ("context", 0, False), ("context", -1, False), ("context", 2147483648, False), ("context", "0", False),
    ("context", "0k", False), ("context", "128 k", False), ("context", "1g", False), ("context", "", False),
    ("context", 1.5, False),
    ("disk_cache", "auto", True), ("disk_cache", "AUTO", True), ("disk_cache", "0", True), ("disk_cache", "off", True),
    ("disk_cache", "none", True), ("disk_cache", "no", True), ("disk_cache", "16G", True), ("disk_cache", "16GB", True),
    ("disk_cache", "16GiB", True), ("disk_cache", "512m", True), ("disk_cache", "1.5T", True),
    ("disk_cache", ".5g", True), ("disk_cache", "1048576", True),
    ("disk_cache", "lots", False), ("disk_cache", "16X", False), ("disk_cache", "-1G", False),
    ("disk_cache", "non", False), ("disk_cache", "", False), ("disk_cache", 16, False),
]
# What a JSON Schema pattern cannot say: the token count's upper bound behind a k or m (the server refuses these)
SCHEMA_WIDER = [("context", "9999999k"), ("context", "4096m"), ("context", "99999999999")]


def main():
    binary = sys.argv[1]
    with open(os.path.join(ROOT, "docs", "serve-config.schema.json")) as f:
        schema = jsonschema.Draft202012Validator(json.load(f))
    rows = CASES + [(k, v, None) for k, v in SCHEMA_WIDER]
    lines = "".join(json.dumps({k: v}) + "\n" for k, v, _ in rows)
    out = subprocess.run([binary, "--check"], input=lines, capture_output=True, text=True, timeout=60, check=True)
    verdicts = out.stdout.splitlines()
    assert len(verdicts) == len(rows), out.stdout + out.stderr
    failed = 0
    for (key, value, want), verdict in zip(rows, verdicts):
        by_schema, by_validator = schema.is_valid({key: value}), verdict == "ok"
        ok = (by_schema, by_validator) == ((True, False) if want is None else (want, want))
        if not ok:
            failed += 1
            print(f"FAIL {key}={value!r}: schema {by_schema}, validator {verdict}")
    # --config -: the same read and execv as serve's restart_server(), three times over
    config = {"model": "4b", "port": 8123, "api_key": "k", "peer": ["mac2"]}
    r = subprocess.run([binary, "--reexec", "3"], input=json.dumps(config), capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or json.loads(r.stdout or "null") != config:
        failed += 1
        print(f"FAIL --config - after 3 restarts: rc {r.returncode}, {r.stdout!r} {r.stderr!r}")
    if failed:
        print(f"{failed} failed")
        return 1
    print(f"test_serve_schema: {len(rows)} values agree, --config - survives 3 restarts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
