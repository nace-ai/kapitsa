#!/usr/bin/env python3
"""Compare actual sampled token IDs from two ni_run builds (model + GPU required).

Build ni_run in both checkouts with the same Release settings, then, from the
candidate checkout:
  python3 tests/test_sampled_tokens.py --reference /old/build/ni_run \
    --candidate build/ni_run --model /models/qwen3.5-4b-4bit --out /tmp/sample-4b
Repeat with the 35B model; add --spec 1 for a model with an MTP head.

The frozen Qwen prompt must decode exactly with the model's tokenizer. Defaults:
512 tokens, temperature .7, seed 42; ni_run fixes top-k=20 and top-p=.95. DFlash
is disabled explicitly. Both arms must emit the requested count, no EOS, nonempty
text and identical IDs. This is a numerical check, not a speed measurement, and
does not cover K32 or DFlash autoregressive sampling. Logs, IDs and identities are
saved in a new output directory. It takes the shared GPU lock, or respects the
caller's NI_LOCK_HELD=1 / NACE_GPU_LOCK_HELD=1 marker (e.g. vk gpu run).

CPU-only checks of this script: python3 tests/test_sampled_tokens_harness.py
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench/harness"))
from gpu_hogs import GpuLock, held_by_caller


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class Tokenizer:
    """Only the ByteLevel BPE decoder used by the frozen Qwen fixture."""
    def __init__(self, path):
        data = json.loads(path.read_text())
        if data["model"]["type"] != "BPE" or data["decoder"]["type"] != "ByteLevel":
            raise ValueError("fixture requires a ByteLevel BPE tokenizer")
        self.vocab = {v: k for k, v in data["model"]["vocab"].items()}
        self.added = {t["id"]: t["content"] for t in data.get("added_tokens", [])}
        self.vocab.update(self.added)
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        missing = [b for b in range(256) if b not in bs]
        self.bytes = dict(zip(map(chr, bs + list(range(256, 256 + len(missing)))), bs + missing))
        # Both Qwen end markers stop a continuation, even when config lists only one.
        self.eos = {i for i, s in self.added.items() if s in ("<|endoftext|>", "<|im_end|>")}
        if len(self.eos) != 2:
            raise ValueError("missing Qwen EOS markers")

    def decode(self, ids):
        if any(type(i) is not int or i < 0 or i not in self.vocab for i in ids):
            raise ValueError("unknown or invalid token ID")
        return b"".join(self.added[i].encode() if i in self.added else
                        bytes(self.bytes[c] for c in self.vocab[i]) for i in ids).decode(errors="replace")


def parse_ids(log):
    lines = re.findall(r"^ids:([^\r\n]*)", log, re.M)
    if len(lines) != 1 or not re.fullmatch(r"(?:\s+[0-9]+)+\s*", lines[0]):
        raise ValueError("expected exactly one nonempty, valid ids: line")
    return list(map(int, lines[0].split()))


def compare(reference, candidate):
    if reference != candidate:
        at = next((i for i, (a, b) in enumerate(zip(reference, candidate)) if a != b),
                  min(len(reference), len(candidate)))
        a = reference[at] if at < len(reference) else "<end>"
        b = candidate[at] if at < len(candidate) else "<end>"
        raise ValueError(f"sampled token mismatch at index {at}: reference={a}, candidate={b}")


def run_arm(name, binary, args, out, env, tokenizer, record):
    command = [str(binary), str(args.model), "@" + str(out / "prompt.ids"),
               str(args.tokens), str(args.temperature), "--spec", str(args.spec), "--dflash", "none"]
    record.update(command=command, binary_sha256=sha(binary))
    print(f"START {name}", flush=True)
    with (out / f"{name}.log").open("w") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            record["returncode"] = child.wait(timeout=args.timeout)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                record["returncode"] = child.returncode
    if record["returncode"] != 0:
        raise ValueError(f"{name}: ni_run exited {record['returncode']}; see {name}.log")
    log = (out / f"{name}.log").read_text(errors="replace")
    ids = parse_ids(log)
    save(out / f"{name}.ids.json", ids)
    text = tokenizer.decode(ids)
    (out / f"{name}.txt").write_text(text, encoding="utf-8")
    eos = [j for j, i in enumerate(ids) if i in tokenizer.eos]
    record.update(token_count=len(ids), eos_count=len(eos), tokens_before_eos=eos[0] if eos else len(ids),
                  nonempty_text=bool(text.strip()), ids_sha256=sha(out / f"{name}.ids.json"))
    if len(ids) != args.tokens or eos or not text.strip():
        raise ValueError(f"{name}: expected {args.tokens} tokens, no EOS and nonempty text; got {record}")
    if args.spec:
        loaded = [line for line in log.splitlines() if "MTP head loaded" in line]
        stats = re.search(r"spec decode D=1:.*? (\d+) cycles, .*?accepted/drafted ([\d.]+)", log)
        if not loaded or not stats or int(stats[1]) <= 0 or float(stats[2]) <= 0:
            raise ValueError(f"{name}: MTP head / accepted speculative tokens not observed")
        record.update(mtp_head=loaded, cycles=int(stats[1]), accepted_per_cycle=float(stats[2]))
    if sha(binary) != record["binary_sha256"]:
        raise ValueError(f"{name}: binary changed during run")
    print(f"{name}: {len(ids)} tokens, EOS=0, nonempty text", flush=True)
    return ids


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", type=Path, required=True, help="pre-change ni_run executable")
    p.add_argument("--candidate", type=Path, required=True, help="candidate ni_run executable")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new directory; never overwrites evidence")
    p.add_argument("--prompt", type=Path, default=ROOT / "tests/fixtures/sampling/qwen35.json")
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spec", type=int, choices=(0, 1), default=0)
    p.add_argument("--timeout", type=float, default=1800, help="seconds per arm")
    args = p.parse_args(argv)
    if (args.tokens < 1 or not math.isfinite(args.temperature) or args.temperature <= 0
            or not 0 <= args.seed <= 0xffffffff or not math.isfinite(args.timeout) or args.timeout <= 0):
        p.error("require positive tokens, temperature and timeout; seed must fit uint32")
    for key in ("reference", "candidate", "model", "prompt", "out"):
        setattr(args, key, getattr(args, key).resolve())
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        p.exit(1, f"FAIL: {exc}\n")
    report = {"passed": False, "purpose": "sampled token equality; not a performance measurement",
              "machine": platform.platform(), "sampling": {"tokens": args.tokens, "temperature": args.temperature,
              "top_k": 20, "top_p": .95, "seed": args.seed, "spec": args.spec, "dflash": "none"}, "arms": {}}
    try:
        unexpected = [k for k in os.environ if k.startswith("NI_") and k != "NI_LOCK_HELD"]
        if unexpected:
            raise ValueError(f"unset NI_* overrides before comparing: {', '.join(sorted(unexpected))}")
        if args.reference == args.candidate or sha(args.reference) == sha(args.candidate):
            raise ValueError("reference and candidate must be different builds")
        for binary in (args.reference, args.candidate):
            if not os.access(binary, os.X_OK):
                raise ValueError(f"not executable: {binary}")
        tracked = {"config": args.model / "config.json", "tokenizer": args.model / "tokenizer.json",
                   "prompt_fixture": args.prompt}
        identities = {k: {"path": str(v), "sha256": sha(v)} for k, v in tracked.items()}
        tokenizer = Tokenizer(tracked["tokenizer"])
        prompt = json.loads(args.prompt.read_text())
        ids, text = prompt["ids"], prompt["text"]
        if not ids or tokenizer.decode(ids) != text or not text.endswith("<|im_start|>assistant\n"):
            raise ValueError("prompt must decode exactly and end with the assistant chat prefix")
        (args.out / "prompt.ids").write_text(",".join(map(str, ids)), encoding="utf-8")
        report.update(inputs=identities, prompt_tokens=len(ids), prompt_ids_sha256=sha(args.out / "prompt.ids"))
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        report["runner_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        env = dict(os.environ, NI_SEED=str(args.seed), NI_LOCK_HELD="1")
        report["environment"] = {k: v for k, v in env.items() if k.startswith(("NI_", "MTL_", "NACE_GPU_", "BENCH_GPU_"))}
        lock = nullcontext() if held_by_caller() else GpuLock(
            path=str(Path(os.environ.get("BENCH_GPU_LOCK") or "/tmp/kapitsa-gpu.lock").expanduser().resolve()))
        with lock:
            outputs = []
            for name in ("reference", "candidate"):
                report["arms"][name] = {}
                outputs.append(run_arm(name, getattr(args, name), args, args.out, env, tokenizer, report["arms"][name]))
                for key, path in tracked.items():
                    if sha(path) != identities[key]["sha256"]:
                        raise ValueError(f"{key} changed during run")
            compare(*outputs)
        report["passed"] = True
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        report["error"] = str(exc) or type(exc).__name__
        print(f"FAIL: {report['error']}", file=sys.stderr)
    finally:
        save(args.out / "report.json", report)
    if report["passed"]:
        print(f"PASS: {args.tokens} sampled token IDs identical; report: {args.out / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
