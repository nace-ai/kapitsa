#!/usr/bin/env python3
"""kapitsa serve's config schema (docs/serve-config.schema.json) and its validator (src/server/service_config.cpp)
give the same answer on the same configs, and `--config -` reads the same JSON after the server execs itself again.

Every field gets the same probe values (numbers, numeric strings like "30", bounds, booleans, null, lists), alone and
beside the fields its combination rules name, plus seeded random configs of several fields. A config the schema takes
and the validator refuses, or the other way round, fails the test, except the gaps listed in SCHEMA_WIDER.

Usage: python3 tests/test_serve_schema.py build/serve_options_test   (needs: jsonschema; no model, no GPU)
"""
import json
import os
import random
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
    ("context", 1, True), ("context", 131072, True), ("context", 131072.0, True), ("port", 8000.0, True), ("port", 8000.5, False), ("context", 2147483647, True), ("context", "131072", False),
    ("context", "128k", True), ("context", "1M", True), ("context", "max", True), ("context", "MAX", True),
    ("context", 0, False), ("context", -1, False), ("context", 2147483648, False), ("context", "0", False),
    ("context", "0k", False), ("context", "30", False), ("context", "128 k", False), ("context", "1g", False), ("context", "", False),
    ("context", 1.5, False),
    ("disk_cache", "auto", True), ("disk_cache", "AUTO", True), ("disk_cache", "0", True), ("disk_cache", "off", True),
    ("disk_cache", "none", True), ("disk_cache", "no", True), ("disk_cache", "16G", True), ("disk_cache", "16GB", True),
    ("disk_cache", "16GiB", True), ("disk_cache", "512m", True), ("disk_cache", "1.5T", True),
    ("disk_cache", ".5g", True), ("disk_cache", "4096B", True), ("disk_cache", "1048576", False), ("disk_cache", "30", False),
    ("disk_cache", " 16G", False), ("disk_cache", "0x10G", False), ("disk_cache", "1e3G", False),
    ("disk_cache", "lots", False), ("disk_cache", "16X", False), ("disk_cache", "-1G", False),
    ("disk_cache", "non", False), ("disk_cache", "", False), ("disk_cache", 16, False),
]
# What a JSON Schema pattern cannot say: the upper bound behind a unit (the server refuses these)
SCHEMA_WIDER = [{"context": "9999999k"}, {"context": "4096m"}, {"disk_cache": "99999999999T"}]

# the same probes for every field
PROBES = ["30", "30.5", "-1", "0", "", "auto", "max", "128k", "16G", "native", "splash", "fast", "both", "per-loop",
          "a:1", "127.0.0.1", "0.0.0.0", 30, 0, 1, 4, 5, 65535, 65536, -1, 0.5, 30.5, 2147483647, 2147483648,
          # integral floats: JSON Schema counts 8000.0 as an integer, so the validator must too
          0.0, 1.0, 4.0, 30.0, 8000.0, 65536.0, 131072.0, 2147483648.0, 1e300,
          True, False, None, [], ["a:1"], [""], [1], {}]
# configs the combination rules read, so each probe also runs where those rules pass and where they refuse
BASES = [{}, {"engine": "splash"}, {"engine": "native"}, {"allow_switch": True, "host": "0.0.0.0"},
         {"allow_switch": True, "host": "0.0.0.0", "api_key": "k"}, {"check": True, "engine": "splash"}]


def generated(fields):
    configs = []
    for base in BASES:
        for key in fields:
            for value in PROBES:
                configs.append({**base, key: value})
    rng = random.Random(1113)
    for _ in range(3000):
        configs.append({key: rng.choice(PROBES) for key in rng.sample(fields, rng.randint(2, 5))})
    return configs


def main():
    binary = sys.argv[1]
    with open(os.path.join(ROOT, "docs", "serve-config.schema.json")) as f:
        schema = jsonschema.Draft202012Validator(json.load(f))
    fields = sorted(schema.schema["properties"])
    rows = [({k: v}, want) for k, v, want in CASES] + [(c, None) for c in SCHEMA_WIDER]
    rows += [(c, "agree") for c in generated(fields)]
    lines = "".join(json.dumps(c) + "\n" for c, _ in rows)
    out = subprocess.run([binary, "--check"], input=lines, capture_output=True, text=True, timeout=60, check=True)
    verdicts = out.stdout.splitlines()
    assert len(verdicts) == len(rows), out.stdout + out.stderr
    failed = 0
    taken = {}
    for (config, want), verdict in zip(rows, verdicts):
        by_schema, by_validator = schema.is_valid(config), verdict == "ok"
        if want is None:
            ok = (by_schema, by_validator) == (True, False)
        elif want == "agree":
            ok = by_schema == by_validator
        else:
            ok = by_schema == by_validator == want
        if not ok:
            failed += 1
            if failed <= 40:
                print(f"FAIL {json.dumps(config)}: schema {by_schema}, validator {verdict}")
        if len(config) == 1 and by_validator:
            taken.setdefault(next(iter(config)), []).append(next(iter(config.values())))
    # a numeric string is never a number: every field that takes a number refuses "30", and only the free-text
    # fields take it at all
    for key in fields:
        prop = json.dumps(schema.schema["properties"][key])
        takes_text = '"type": "string"' in prop and "pattern" not in prop
        if '"integer"' in prop or '"number"' in prop or not takes_text:
            for value in ("30", "30.5"):
                if value in taken.get(key, []):
                    failed += 1
                    print(f"FAIL {key} takes the numeric string {value!r}")
    # --config -: the same read and execv as serve's restart_server(), three times over
    config = {"model": "4b", "port": 8123, "api_key": "k", "peer": ["mac2"]}
    r = subprocess.run([binary, "--reexec", "3"], input=json.dumps(config), capture_output=True, text=True, timeout=60)
    if r.returncode != 0 or json.loads(r.stdout or "null") != config:
        failed += 1
        print(f"FAIL --config - after 3 restarts: rc {r.returncode}, {r.stdout!r} {r.stderr!r}")
    if failed:
        print(f"{failed} failed")
        return 1
    print(f"test_serve_schema: schema and validator agree on {len(rows)} configs, --config - survives 3 restarts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
