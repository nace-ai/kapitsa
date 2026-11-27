#!/usr/bin/env python3
"""Compile the affine bodies from two exact commits and run the CPU-only screen."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=False)
revisions = {"original": "01b4c6d69fbe69e862c00358f88c94a363129592", "vector": "8c0a7ec3"}
header = """#include <arm_neon.h>
#include <algorithm>
#include <bit>
#include <cstdint>
static float bf16_to_f32(uint16_t x) {return std::bit_cast<float>(uint32_t(x)<<16);}
"""
identity = {}
for label, rev in revisions.items():
    source = subprocess.check_output(["git", "-C", str(repo), "show", rev + ":src/model/ngram.cpp"], text=True)
    start = source.index("      const uint32_t mask = (1u << bits_) - 1;", source.index("void NgramTable::row"))
    end = source.index("      break;", start)
    body = source[start:end]
    code = header + 'extern "C" __attribute__((noinline)) void ' + label + '(const uint32_t* q,const uint16_t* sc,const uint16_t* bi,float* out,int bits_,int gs_,int dim_){\n' + body + '}\n'
    (out / (label + ".cpp")).write_text(code)
    identity[label] = {"commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", rev], text=True).strip(), "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "body_sha256": hashlib.sha256(body.encode()).hexdigest()}
cmd = ["clang++", "-O3", "-std=c++20", str(out / "original.cpp"), str(out / "vector.cpp"), str(HERE / "screen.cpp"), "-o", str(out / "screen")]
subprocess.run(cmd, check=True)
identity["compile_command"] = cmd
identity["compiler"] = subprocess.check_output(["clang++", "--version"], text=True)
(out / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
with (out / "screen.log").open("w") as f:
    subprocess.run([str(out / "screen")], stdout=f, stderr=subprocess.STDOUT, check=True)
print((out / "screen.log").read_text())
