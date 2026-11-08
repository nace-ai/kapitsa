"""Extract the recorded source functions for CPU-only row-packing comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--reference", required=True)
parser.add_argument("--candidate", required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir()
header = """#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <cstring>
namespace ni {
enum class NhFmt : uint8_t { F32, Q8, FP4, IQ2, BF16, FP8 };
"""
records = []
for arm, commit in (("reference", args.reference), ("candidate", args.candidate)):
    source = subprocess.check_output(
        ["git", "show", commit + ":src/model/nemotron_h.cpp"], text=True
    )
    start = source.index("void nh_row_bytes(")
    end = source.index("\nvoid nh_quant_q8(", start)
    block = source[start:end]
    if arm == "reference":
        for name in ("nh_repack_rows", "nh_src_row_bytes", "nh_row_bytes"):
            block = block.replace(name, "ref_" + name)
    target = args.out / (arm + ".cpp")
    target.write_text(header + block + "\n}\n")
    records.append({
        "arm": arm,
        "commit": commit,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "translation_unit_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    })
(args.out / "extraction.json").write_text(json.dumps(records, indent=2) + "\n")
