#!/usr/bin/env python3
"""Compare complete Max readout vectors using the preregistered global KL gate."""

import argparse
import array
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import uuid

import logit_compare
from logit_compare import row_metrics

BASELINE_COMMIT = "7f4e709ad9b3195ec72bfafd6667048e8e99fb78"
ROOT = Path(__file__).resolve().parents[1]
IDENTITIES = (
    "measurement_commit", "patch_sha256", "request_sha256", "layout_sha256",
    "config_sha256", "tokenizer_sha256", "head_sha256", "readout_temperature",
    "host", "environment", "build_flags",
)
HASH_KEYS = (
    "binary_sha256", "patch_sha256", "request_sha256", "layout_sha256", "config_sha256",
    "tokenizer_sha256", "head_sha256", "manifest_sha256", "payload_sha256",
)
ROW_KEYS = {"request_index", "branch_index", "question", "option_order", "tokens", "decide", "width", "byte_offset"}
NATIVE_ENVIRONMENT = {"NI_ANE": "0", "NI_NO_MTP": "1", "NI_PREWARM": "0", "NI_PROFILE": "0", "NI_SYS_CACHE": "0"}
POLICY = {"schema_version": 1, "baseline_commit": BASELINE_COMMIT, "metric": "kl",
          "direction": "reference||candidate", "aggregation": "global_max", "threshold_factor": 2}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def numeric_runtime():
    return {"python": sys.version, "backend": "numpy" if logit_compare.np is not None else "stdlib",
            "backend_version": logit_compare.np.__version__ if logit_compare.np is not None else sys.version,
            "logit_compare_sha256": sha256(Path(logit_compare.__file__).read_bytes())}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value, length=64):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def load_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key {key}")
            result[key] = value
        return result

    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON value {x}")))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value, raw


def validate_rows(rows, payload_bytes):
    require(isinstance(rows, list) and rows, "rows must be a nonempty array")
    previous = None
    offset = 0
    for row in rows:
        require(isinstance(row, dict) and set(row) == ROW_KEYS, "invalid row fields")
        for key in ("request_index", "branch_index", "decide", "byte_offset"):
            require(integer(row[key]), f"invalid row {key}")
        key = row["request_index"], row["branch_index"]
        require(key == (0, 0) if previous is None else
                key in ((previous[0], previous[1] + 1), (previous[0] + 1, 0)),
                "missing, duplicate or reordered branch")
        previous = key
        question = row["question"]
        require(isinstance(question, dict) and set(question) == {"id", "type", "keys", "legend"},
                "invalid question metadata")
        require(isinstance(question["id"], str) and question["type"] in ("choice", "noul", "score"),
                "invalid question identity/type")
        options = row["option_order"]
        require(isinstance(options, list) and options and all(isinstance(x, str) for x in options)
                and len(set(options)) == len(options), "invalid option order")
        require(question["keys"] == options, "question keys differ from option order")
        require(isinstance(question["legend"], list) and all(isinstance(x, str) for x in question["legend"]),
                "invalid question legend")
        require(integer(row["width"], 1) and row["width"] == len(options), "width differs from active options")
        tokens = row["tokens"]
        require(isinstance(tokens, list) and tokens and all(integer(x) and x <= 2147483647 for x in tokens),
                "invalid encoded tokens")
        require(row["decide"] < len(tokens), "decide outside branch")
        require(row["byte_offset"] == offset, "payload has gap, overlap or reordered offsets")
        offset += 4 * row["width"]
    require(integer(payload_bytes) and payload_bytes == offset, "payload byte count differs from row inventory")


def validate_provenance(p):
    for key in HASH_KEYS:
        require(digest(p.get(key)), f"invalid provenance {key}")
    for key in ("production_commit", "measurement_commit"):
        require(digest(p.get(key), 40), f"invalid provenance {key}")
    require(finite(p.get("readout_temperature")), "invalid readout temperature")
    require(isinstance(p.get("host"), dict) and set(p["host"]) == {"machine", "chip", "os", "build"}
            and all(isinstance(x, str) and x for x in p["host"].values()), "invalid host identity")
    require(isinstance(p.get("environment"), dict)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in p["environment"].items()),
            "invalid environment")
    require(isinstance(p.get("argv"), list) and p["argv"] and all(isinstance(x, str) for x in p["argv"]), "invalid argv")
    require(isinstance(p.get("build_flags"), str), "invalid build flags")
    require(integer(p.get("process_pid"), 1), "invalid process PID")
    require(isinstance(p.get("run_id"), str), "invalid run ID")
    try:
        uuid.UUID(p["run_id"])
        started = datetime.datetime.fromisoformat(p["process_started_at"].replace("Z", "+00:00"))
    except (ValueError, KeyError, AttributeError, TypeError) as exc:
        raise ValueError("invalid run ID or process start timestamp") from exc
    require(started.utcoffset() == datetime.timedelta(0), "process start must be UTC")


def read_capture(path):
    path = Path(path)
    manifest, manifest_raw = load_json(path / "manifest.json")
    require(set(manifest) == {"schema_version", "complete", "payload_bytes", "rows"}, "invalid manifest fields")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["complete"] is True, "unsupported or incomplete capture")
    validate_rows(manifest["rows"], manifest["payload_bytes"])
    payload = (path / "logits.f32").read_bytes()
    require(len(payload) == manifest["payload_bytes"], "truncated or extra payload bytes")
    provenance, provenance_raw = load_json(path / "provenance.json")
    validate_provenance(provenance)
    require(provenance["manifest_sha256"] == sha256(manifest_raw), "manifest hash mismatch")
    require(provenance["payload_sha256"] == sha256(payload), "payload hash mismatch")
    values = []
    for row in manifest["rows"]:
        start = row["byte_offset"]
        raw = payload[start:start + 4 * row["width"]]
        if logit_compare.np is not None:
            decoded = logit_compare.np.frombuffer(raw, dtype="<f4").astype(logit_compare.np.float64)
        else:
            decoded = array.array("f")
            decoded.frombytes(raw)
            if sys.byteorder == "big":
                decoded.byteswap()
        require(all(math.isfinite(x) for x in decoded), "nonfinite logit")
        values.append((raw, decoded))
    return {"rows": manifest["rows"], "values": values, "provenance": provenance,
            "manifest_sha256": sha256(manifest_raw), "payload_sha256": sha256(payload),
            "provenance_sha256": sha256(provenance_raw)}


def compare(reference, candidate):
    require(reference["rows"] == candidate["rows"], "branch inventory, order, options, metadata or tokens differ")
    for key in IDENTITIES:
        require(reference["provenance"][key] == candidate["provenance"][key], f"identity mismatch: {key}")
    results = []
    for row, a, b in zip(reference["rows"], reference["values"], candidate["values"]):
        metrics = row_metrics(a, b, None)
        kl = metrics["kl"]
        require(finite(kl) and all(finite(value) for key, value in metrics.items()
                                  if key != "bitwise" and value is not None), "nonfinite comparison metric")
        metrics.pop("nll_change")
        metrics.update(request_index=row["request_index"], branch_index=row["branch_index"], gap=max(0.0, kl))
        results.append(metrics)
    return {"metric": "kl", "direction": "reference||candidate", "aggregation": "global_max",
            "numeric_runtime": numeric_runtime(), "global_gap": max(row["gap"] for row in results), "rows": results,
            "bitwise_rows": sum(row["bitwise"] for row in results),
            "top1_agreement": sum(row["top_ref"] == row["top_cand"] for row in results) / len(results),
            "maxdiff": max(row["maxdiff"] for row in results)}


def identity(capture):
    return {key: capture["provenance"][key] for key in IDENTITIES}


def capture_hashes(capture):
    return {key: capture[key] for key in ("manifest_sha256", "payload_sha256", "provenance_sha256")}


def read_protocol(path, reference):
    protocol, raw = load_json(path)
    for key, expected in POLICY.items():
        require(type(protocol.get(key)) is type(expected) and protocol[key] == expected, f"invalid protocol {key}")
    require(protocol.get("context") == 8192 and type(protocol["context"]) is int, "invalid protocol context")
    require(isinstance(protocol.get("model_dir"), str) and protocol["model_dir"], "invalid protocol model_dir")
    require(protocol.get("comparator_sha256") == numeric_runtime()["logit_compare_sha256"], "protocol comparator hash mismatch")
    for file_key, hash_key, provenance_key in (("request_file", "corpus_sha256", "request_sha256"),
                                               ("layout_file", "layout_sha256", "layout_sha256")):
        require(digest(protocol.get(hash_key)), f"invalid protocol {hash_key}")
        require(protocol[hash_key] == reference["provenance"][provenance_key], f"protocol {hash_key} mismatch")
        require(isinstance(protocol.get(file_key), str) and protocol[file_key], f"missing protocol {file_key}")
        file_path = Path(protocol[file_key])
        if not file_path.is_absolute():
            file_path = ROOT / file_path
        require(sha256(file_path.read_bytes()) == protocol[hash_key], f"frozen {file_key} hash mismatch")
        if file_key == "layout_file":
            layout, _ = load_json(file_path)
            require(layout.get("rows") == reference["rows"], "capture differs from frozen encoded layout")
    return protocol, sha256(raw)


def make_floor(reference, repeat, protocol_path):
    protocol, protocol_hash = read_protocol(protocol_path, reference)
    a, b = reference["provenance"], repeat["provenance"]
    require(all(a["environment"].get(key) == value for key, value in NATIVE_ENVIRONMENT.items()),
            "baseline needs the preregistered native environment")
    require(a["production_commit"] == b["production_commit"] == BASELINE_COMMIT, "floor needs the expected baseline commit")
    require(a["binary_sha256"] == b["binary_sha256"], "baseline binary differs")
    for key in ("run_id", "process_pid", "process_started_at"):
        require(a[key] != b[key], f"floor needs independent fresh processes: repeated {key}")
    report = compare(reference, repeat)
    return {"schema_version": 1, "kind": "max_readout_floor", "protocol": protocol,
            "protocol_sha256": protocol_hash, "comparator_sha256": sha256(Path(__file__).read_bytes()),
            "numeric_runtime": numeric_runtime(), "identities": identity(reference), "reference": capture_hashes(reference),
            "repeat": capture_hashes(repeat), "baseline_provenance": [a, b],
            "global_floor": report["global_gap"], "baseline_report": report}


def validate_floor(floor, reference):
    require(type(floor.get("schema_version")) is int and floor["schema_version"] == 1
            and floor.get("kind") == "max_readout_floor", "invalid floor format")
    require(floor.get("comparator_sha256") == sha256(Path(__file__).read_bytes()), "floor comparator hash mismatch")
    require(floor.get("numeric_runtime") == numeric_runtime(), "floor numeric backend or Python version differs")
    require(digest(floor.get("protocol_sha256")), "invalid floor protocol hash")
    protocol = floor.get("protocol")
    require(isinstance(protocol, dict), "missing floor protocol")
    for key, expected in POLICY.items():
        require(type(protocol.get(key)) is type(expected) and protocol[key] == expected, f"invalid floor protocol {key}")
    require(protocol.get("comparator_sha256") == numeric_runtime()["logit_compare_sha256"], "floor protocol comparator mismatch")
    require(floor.get("reference") == capture_hashes(reference), "reference is not the frozen baseline capture")
    require(floor.get("identities") == identity(reference), "floor input identity mismatch")
    require(all(reference["provenance"]["environment"].get(key) == value for key, value in NATIVE_ENVIRONMENT.items()),
            "floor needs the preregistered native environment")
    require(finite(floor.get("global_floor")) and floor["global_floor"] >= 0, "invalid global floor")
    report = floor.get("baseline_report")
    require(isinstance(report, dict) and isinstance(report.get("rows"), list)
            and len(report["rows"]) == len(reference["rows"]), "invalid floor row inventory")
    for row, expected in zip(report["rows"], reference["rows"]):
        require(isinstance(row, dict) and finite(row.get("kl")) and finite(row.get("gap"))
                and row["gap"] == max(0.0, row["kl"])
                and row.get("request_index") == expected["request_index"]
                and row.get("branch_index") == expected["branch_index"], "invalid floor row metric")
    require(report.get("global_gap") == floor["global_floor"] == max(row["gap"] for row in report["rows"]),
            "floor maximum differs from baseline metrics")
    bases = floor.get("baseline_provenance")
    require(isinstance(bases, list) and len(bases) == 2, "missing baseline provenance")
    for base in bases:
        require(isinstance(base, dict), "invalid baseline provenance")
        validate_provenance(base)
        require(base["production_commit"] == BASELINE_COMMIT
                and all(base[key] == floor["identities"][key] for key in IDENTITIES), "floor baseline identity mismatch")
    require(bases[0] == reference["provenance"], "floor baseline provenance differs from reference")
    require(bases[0]["binary_sha256"] == bases[1]["binary_sha256"], "floor baseline binary differs")
    for key in ("run_id", "process_pid", "process_started_at"):
        require(bases[0][key] != bases[1][key], "floor processes are not independent")
    require(protocol.get("corpus_sha256") == reference["provenance"]["request_sha256"]
            and protocol.get("layout_sha256") == reference["provenance"]["layout_sha256"], "floor frozen input hash mismatch")


def check(reference, candidate, floor):
    validate_floor(floor, reference)
    require(candidate["provenance"]["production_commit"] in floor["protocol"].get("candidate_commits", []),
            "candidate production commit is not preregistered")
    require(candidate["provenance"]["binary_sha256"] != reference["provenance"]["binary_sha256"],
            "candidate binary is the baseline binary")
    report = compare(reference, candidate)
    threshold = 2 * floor["global_floor"]
    require(finite(threshold), "nonfinite threshold")
    report.update(schema_version=1, kind="max_readout_check", global_floor=floor["global_floor"], threshold=threshold,
                  passed=all(row["gap"] <= threshold for row in report["rows"]),
                  reference=capture_hashes(reference), candidate=capture_hashes(candidate),
                  protocol_sha256=floor["protocol_sha256"])
    for row in report["rows"]:
        row["passed"] = row["gap"] <= threshold
    return report


def write_exclusive(path, report):
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as out:
        out.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("floor", "check"):
        command = commands.add_parser(name)
        command.add_argument("reference", type=Path)
        command.add_argument("candidate", type=Path)
        command.add_argument("--out", required=True, type=Path)
        command.add_argument("--protocol" if name == "floor" else "--floor", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        require(not args.out.exists(), "output already exists")
        reference = read_capture(args.reference)
        if args.command == "floor":
            report = make_floor(reference, read_capture(args.candidate), args.protocol)
        else:
            floor, floor_raw = load_json(args.floor)
            validate_floor(floor, reference)
            report = check(reference, read_capture(args.candidate), floor)
            report["floor_sha256"] = sha256(floor_raw)
        write_exclusive(args.out, report)
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"invalid Max comparison: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: report[key] for key in ("global_floor",) if key in report}
                     | {key: report[key] for key in ("global_gap", "threshold", "passed") if key in report}))
    return 0 if report.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
