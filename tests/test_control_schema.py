#!/usr/bin/env python3
"""The control socket's schema (schema/control/) is valid JSON Schema 2020-12 and its examples agree with it.

CPU only, no build, no network. Needs: jsonschema (4.18 or later, for `referencing`).

What it checks:
- every schema file, and step 1's docs/serve-config.schema.json that they refer to, passes the 2020-12 metaschema;
- every `$ref` in schema/control/ resolves (the instance configuration's fields are step 1's, referred to by `$id`);
- every file in schema/control/examples/ is valid against the message its name starts with, and every file in
  examples/invalid/ is refused by it: each is one change (a JSON merge patch) to a valid example, such as an unknown
  field, a wrong type, an unsupported protocol or an old POC field name, so it fails for that change alone;
- every message has an example, every error code has one, and docs/control-socket.md's error table lists exactly the
  schema's codes.

`--serve-config PATH` reads step 1's schema from another checkout, for runs before it is on this branch.
"""
import argparse
import json
import os
import re
import sys
from urllib.parse import urljoin

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_DIR = os.path.join(ROOT, "schema", "control")
EXAMPLES = os.path.join(SCHEMA_DIR, "examples")
INVALID = os.path.join(EXAMPLES, "invalid")
DOC = os.path.join(ROOT, "docs", "control-socket.md")
SERVE_CONFIG = os.path.join(ROOT, "docs", "serve-config.schema.json")
ROOT_ID = "https://github.com/nace-ai/kapitsa/blob/main/schema/control/control.schema.json"


def load(path):
    with open(path) as f:
        return json.load(f)


def refs(node):
    if isinstance(node, dict):
        if "$ref" in node:
            yield node["$ref"]
        for v in node.values():
            yield from refs(v)
    elif isinstance(node, list):
        for v in node:
            yield from refs(v)


def merge_patch(target, patch):
    """RFC 7386: objects merge key by key, null removes a key, anything else replaces."""
    if not isinstance(patch, dict):
        return patch
    out = dict(target) if isinstance(target, dict) else {}
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = merge_patch(out.get(k), v)
    return out


def example(directory, name):
    """A valid example as written; a must-fail one is {"base": <valid example>, "merge_patch": <the one change>}."""
    doc = load(os.path.join(directory, name))
    if directory == INVALID:
        return merge_patch(load(os.path.join(EXAMPLES, doc["base"])), doc["merge_patch"])
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-config", default=SERVE_CONFIG, help="step 1's docs/serve-config.schema.json")
    args = ap.parse_args()
    failures = []

    if not os.path.exists(args.serve_config):
        sys.exit(f"FAIL {os.path.relpath(args.serve_config, ROOT)} is missing: the control schema refers to step 1's "
                 "serve configuration schema; merge after it, or pass --serve-config")
    schemas = {os.path.join("docs", "serve-config.schema.json"): load(args.serve_config)}
    for name in sorted(os.listdir(SCHEMA_DIR)):
        if name.endswith(".schema.json"):
            schemas[os.path.join("schema", "control", name)] = load(os.path.join(SCHEMA_DIR, name))

    for path, schema in schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
        except Exception as e:  # noqa: BLE001 - report every schema, not the first
            failures.append(f"{path}: not a valid 2020-12 schema: {e.message if hasattr(e, 'message') else e}")
        if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            failures.append(f"{path}: $schema is not 2020-12")
        if path.startswith("schema") and "$id" not in schema:
            failures.append(f"{path}: no $id")

    # Step 1's schema is found where the control schema's relative $refs point, whatever $id it declares itself
    serve = schemas[os.path.join("docs", "serve-config.schema.json")]
    resources = [(urljoin(ROOT_ID, "../../docs/serve-config.schema.json"), serve)]
    resources += [(s["$id"], s) for s in schemas.values() if "$id" in s]
    registry = Registry().with_resources(
        (uri, Resource.from_contents(s, default_specification=DRAFT202012)) for uri, s in resources)
    for path, schema in schemas.items():
        if not path.startswith("schema"):
            continue
        resolver = registry.resolver(base_uri=schema["$id"])
        for ref in sorted(set(refs(schema))):
            if ref.startswith("#"):
                ref = schema["$id"] + ref
            try:
                resolver.lookup(ref)
            except Exception as e:  # noqa: BLE001
                failures.append(f"{path}: $ref {ref} does not resolve ({type(e).__name__})")

    control = schemas[os.path.join("schema", "control", "control.schema.json")]
    messages = control["$defs"]["messages"]
    validators = {m: Draft202012Validator({"$ref": f"{ROOT_ID}#/$defs/messages/{m}"}, registry=registry) for m in messages}

    seen = set()
    for directory, want_valid in ((EXAMPLES, True), (INVALID, False)):
        for name in sorted(os.listdir(directory)):
            if not name.endswith(".json"):
                continue
            message = name.split(".")[0]
            rel = os.path.relpath(os.path.join(directory, name), ROOT)
            if message not in validators:
                failures.append(f"{rel}: no message named {message}")
                continue
            try:
                errors = list(validators[message].iter_errors(example(directory, name)))
            except Exception as e:  # noqa: BLE001 - an unresolved $ref; the $ref check above names it
                failures.append(f"{rel}: cannot validate: {type(e).__name__}")
                continue
            if want_valid:
                seen.add(message)
                for e in errors:
                    failures.append(f"{rel}: {'/'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}")
            elif not errors:
                failures.append(f"{rel}: accepted, but it must be refused")
            print(f"{'ok' if (not errors) == want_valid else 'FAIL'} {rel}")

    for m in messages:
        if m not in seen:
            failures.append(f"message {m} has no example in {os.path.relpath(EXAMPLES, ROOT)}")
    codes = control["$defs"]["error_code"]["enum"]
    for code in codes:
        if not os.path.exists(os.path.join(EXAMPLES, f"error.{code}.json")):
            failures.append(f"error code {code} has no example error.{code}.json")
    with open(DOC) as f:
        documented = re.findall(r"^\| `([a-z_]+)` \| [0-9]{3}", f.read(), re.M)
    if sorted(documented) != sorted(codes):
        failures.append(f"docs/control-socket.md's error table {sorted(documented)} != the schema's codes {sorted(codes)}")

    for f in failures:
        print("FAIL", f)
    print(f"{len(schemas)} schemas, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
