#!/usr/bin/env python3
"""Interleaved factorization time, process peak RSS and A/P byte equality; CPU only.

Usage: python3 reproduce_lowrank.py /path/to/kapitsa REFERENCE_SHA CANDIDATE_SHA
Uses the exact low-rank implementation from each commit and the same deterministic packed Q4 input.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo", type=Path)
parser.add_argument("reference")
parser.add_argument("candidate")
args = parser.parse_args()
repo = args.repo.resolve()
here = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="kapitsa-lowrank-check-") as directory:
    temp = Path(directory)
    for name, revision in (("reference", args.reference), ("candidate", args.candidate)):
        path = temp / (name + ".cpp")
        path.write_bytes(subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:src/model/dflash2_lowrank.cpp"]))
        subprocess.run(["clang++", "-std=c++20", "-O3", "-I" + str(repo / "src"), str(path),
                        str(here / "lowrank-screen.cpp"), "-framework", "Accelerate", "-o", str(temp / name)], check=True)
    for n, k, rank in ((128, 64, 16), (5000, 128, 32), (8192, 2560, 768), (4096, 4096, 768)):
        expected = None
        for pair in range(4 if k > 128 else 1):
            for name in ("reference", "candidate"):
                out = temp / "factors.bin"
                print(f"{name} pair {pair}", flush=True)
                subprocess.run([str(temp / name), str(n), str(k), str(rank), str(out)], check=True)
                digest = hashlib.sha256(out.read_bytes()).hexdigest()
                print(f"A/P SHA256 {digest}", flush=True)
                if expected is None:
                    expected = digest
                if digest != expected:
                    raise RuntimeError(f"factor mismatch for {name}, N={n}, K={k}, rank={rank}")
