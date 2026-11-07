#!/usr/bin/env python3
"""CPU admission comparison using actual sources and a caller-supplied support library."""
import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--support-build", type=Path, required=True)
    p.add_argument("--support-revision", required=True)
    p.add_argument("--out", type=Path, required=True, help="new directory, never overwrite frozen evidence")
    p.add_argument("--reference", default="934a6d72")
    p.add_argument("--candidate", default="877e7b4d")
    a = p.parse_args()
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    lib = a.support_build.resolve() / "libnaceinfer.a"
    flags = ["-std=c++20", "-O3", "-mmacosx-version-min=14.0", "-Isrc", "-Ithird_party"]
    frameworks = []
    for name in ["Metal", "Foundation", "CoreFoundation", "IOSurface", "IOKit", "ImageIO",
                 "CoreGraphics", "Security", "SystemConfiguration", "Accelerate"]:
        frameworks += ["-framework", name]
    record = {"platform": platform.platform(), "compiler": subprocess.check_output(
        ["clang++", "--version"], text=True), "flags": flags,
        "support_revision": a.support_revision, "support_library_sha256": hashlib.sha256(lib.read_bytes()).hexdigest(),
        "scope": "Selected actual source files linked against one common support library; not a full candidate build or GPU run.",
        "commands": [], "sources": {}}

    def run(cmd, log):
        record["commands"].append(cmd)
        with (out / log).open("w") as f:
            subprocess.run(cmd, cwd=root, stdout=f, stderr=subprocess.STDOUT, check=True)

    with tempfile.TemporaryDirectory(prefix="ngram-bounds-") as scratch:
        tmp = Path(scratch)
        for label, ref in [("reference", a.reference), ("candidate", a.candidate)]:
            src = subprocess.check_output(["git", "show", ref + ":src/model/ngram.cpp"], cwd=root)
            source = tmp / (label + ".cpp")
            source.write_bytes(src)
            record["sources"][label] = {"revision": subprocess.check_output(
                ["git", "rev-parse", ref], cwd=root, text=True).strip(), "sha256": hashlib.sha256(src).hexdigest()}
            exe = tmp / label
            run(["clang++", *flags, str(Path(__file__).with_name("probe.cpp").resolve()),
                 str(source), str(lib), *frameworks, "-o", str(exe)], label + "-compile.log")
            run([str(exe), str(tmp / (label + ".bin"))], label + ".log")
            expected = "ADMITTED truncated table" if label == "reference" else "REJECTED n-gram table:"
            if expected not in (out / (label + ".log")).read_text():
                raise RuntimeError(label + " did not reproduce expected admission behavior")
        test = tmp / "fn_store_test.cpp"
        expert = tmp / "expert_store.cpp"
        test.write_bytes(subprocess.check_output(["git", "show", a.candidate + ":tools/fn_store_test.cpp"], cwd=root))
        expert.write_bytes(subprocess.check_output(["git", "show", a.candidate + ":src/model/expert_store.cpp"], cwd=root))
        exe = tmp / "fn_store_test"
        run(["clang++", *flags, "-DNI_EXPERT_CACHE_TEST", str(test), str(tmp / "candidate.cpp"),
             str(expert), str(lib), *frameworks, "-o", str(exe)], "fn-store-compile.log")
        run([str(exe)], "fn-store.log")
    record["result"] = "PASS: reference admits truncated payload; candidate rejects; complete fn_store_test passed"
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(record["result"])


if __name__ == "__main__":
    main()
