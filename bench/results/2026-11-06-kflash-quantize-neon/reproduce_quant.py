#!/usr/bin/env python3
"""Compile both revisions' real CPU quantizers, then check and time identical inputs.

Usage: python3 reproduce_quant.py /path/to/kapitsa CHECKOUT_REFERENCE_SHA CHECKOUT_CANDIDATE_SHA [DRAFTER_SAFETENSORS]
The checkout must contain both revisions. Runs on Apple silicon; no model or GPU is loaded.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo", type=Path)
parser.add_argument("reference")
parser.add_argument("candidate")
parser.add_argument("weights", type=Path, nargs="?")
args = parser.parse_args()
repo = args.repo.resolve()
here = Path(__file__).resolve().parent
includes = """#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <arm_neon.h>
#include "util/bf16.h"
#include "util/load_threads.h"
namespace ni {
"""
with tempfile.TemporaryDirectory(prefix="kapitsa-quant-check-") as directory:
    temp = Path(directory)
    sources = []
    for name, revision in (("reference", args.reference), ("candidate", args.candidate)):
        source = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{revision}:src/model/dflash2.cpp"], text=True)
        begin = source.index("void dfl_quantize_cpu(")
        end = source.index("// bf16 rows", begin)
        path = temp / (name + ".cpp")
        path.write_text(includes + source[begin:end].replace("dfl_quantize_cpu(", name + "(") + "\n}\n")
        sources.append(path)
    for name in ("screen", "real"):
        if name == "real" and args.weights is None:
            continue
        binary = temp / name
        command = ["clang++", "-std=c++20", "-O3", "-I" + str(repo / "src"),
                   "-I" + str(repo / "third_party"), *map(str, sources),
                   str(here / ("quant-" + name + ".cpp")), "-o", str(binary)]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)] + ([str(args.weights.resolve())] if name == "real" else []), check=True)
