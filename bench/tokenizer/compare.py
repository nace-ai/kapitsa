"""Interleave two CPU encode drivers, comparing their complete token arrays."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = {name: digest(getattr(args, name)) for name in ("reference", "candidate", "tokenizer", "cases")}
    reference = None
    runs = []
    for label, arm in (("A1", "reference"), ("B1", "candidate"), ("A2", "reference"), ("B2", "candidate")):
        result = subprocess.run([str(getattr(args, arm).resolve()), str(args.tokenizer), str(args.cases)],
                                capture_output=True, text=True, check=False)
        (args.output / f"{label}.jsonl").write_text(result.stdout)
        (args.output / f"{label}.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        outputs = [(x["name"], x["ids"]) for x in rows]
        if reference is None:
            reference = outputs
        equal = outputs == reference
        runs.append({"label": label, "all_token_arrays_equal": equal,
                     "rows": [{k: v for k, v in x.items() if k != "ids"} | {"tokens": len(x["ids"])} for x in rows]})
        summary = {"binary_and_input_sha256": identity, "runs": runs,
                   "scope": "CPU encode plus complete-array equality check; no inference or GPU measurement"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if not equal:
            raise RuntimeError(f"{label}: token arrays differ from A1")
        for name in identity:
            if digest(getattr(args, name)) != identity[name]:
                raise RuntimeError(f"{name} changed during {label}")
        print(f"{label}: {len(rows)} complete token arrays match", flush=True)


if __name__ == "__main__":
    main()
