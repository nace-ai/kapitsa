#!/usr/bin/env python3
"""`logit_dump`'s output directory on the CPU: no model, no GPU, the GPU lock never reached.

usage: test_logit_dump.py [BUILD_DIR]   (default: build; needs BUILD_DIR/logit_dump)

logit_dump makes <out_dir> and any missing parents before it reads the corpus, takes the GPU lock or loads a model, so
each case below stops at the corpus (a file that is not there) with the directory already made, or at the directory
itself when it cannot be made:
1. a nested path none of whose parents exist (dumps/x/ref): made, then the corpus error
2. a path that already exists: kept, then the corpus error
3. a path under a regular file: "cannot make", and the corpus is never read
"""
import os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
BUILD = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "build"))
LOGIT_DUMP = os.path.join(BUILD, "logit_dump")
FAILS = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok or not detail else f"  ({detail})"), flush=True)
    if not ok:
        FAILS.append(name)


def main():
    if not os.access(LOGIT_DUMP, os.X_OK):
        sys.exit(f"no {LOGIT_DUMP}: build it first (cmake --build build --target logit_dump)")
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "no-model")  # never read: the corpus fails first
        corpus = os.path.join(tmp, "missing-corpus.txt")
        # the lock is a test file, should a regression ever reach it
        env = {k: v for k, v in os.environ.items() if not k.startswith(("NI_", "NACE_"))}
        env["BENCH_GPU_LOCK"] = os.path.join(tmp, "gpu.lock")

        def dump(out):
            return subprocess.run([LOGIT_DUMP, model, out, "--corpus", corpus], capture_output=True, text=True,
                                  timeout=60, env=env)

        nested = os.path.join(tmp, "dumps", "x", "ref")
        r = dump(nested)
        check(r.returncode == 1 and f"cannot read {corpus}" in r.stderr, "nested path: stops at the corpus",
              f"rc {r.returncode}: {r.stderr.strip()}")
        check(os.path.isdir(nested), "nested path: dumps/x/ref made with its parents")

        r = dump(nested)
        check(r.returncode == 1 and f"cannot read {corpus}" in r.stderr, "existing path: kept, stops at the corpus",
              f"rc {r.returncode}: {r.stderr.strip()}")

        blocker = os.path.join(tmp, "a-file")
        open(blocker, "w").close()
        r = dump(os.path.join(blocker, "ref"))
        check(r.returncode == 1 and "cannot make" in r.stderr and "cannot read" not in r.stderr,
              "path under a file: cannot make, before the corpus", f"rc {r.returncode}: {r.stderr.strip()}")
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
