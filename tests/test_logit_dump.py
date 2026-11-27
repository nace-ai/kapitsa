#!/usr/bin/env python3
"""`logit_dump`'s output directory on the CPU: no model, no GPU, the GPU lock never reached.

usage: test_logit_dump.py [BUILD_DIR]   (default: build; needs BUILD_DIR/logit_dump)

logit_dump makes <out_dir> and any missing parents before it reads the corpus, takes the GPU lock or loads a model, so
each case below stops at the corpus (a file that is not there) with the directory already made, or at the directory
itself when it cannot be made:
1. a nested path none of whose parents exist (dumps/x/ref): made, then the corpus error
2. a path that already exists: kept, then the corpus error
3. a path under a regular file: "cannot make", and the corpus is never read
4. --verify-rows 12 (a wide verify runs whole 8-row tiles of 16 or 32): refused before the directory is made
5. --list-paths: the paths each model class has, from a config.json alone (tools/logit_floor.py's), no directory made
"""
import json, os, subprocess, sys, tempfile

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

        def dump(out, *extra):
            return subprocess.run([LOGIT_DUMP, model, out, "--corpus", corpus, *extra], capture_output=True, text=True,
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

        rows12 = os.path.join(tmp, "rows12")
        r = dump(rows12, "--verify-rows", "12")
        check(r.returncode == 1 and "--verify-rows: 8, 16 or 32" in r.stderr and not os.path.exists(rows12),
              "--verify-rows 12: refused before the directory", f"rc {r.returncode}: {r.stderr.strip()}")

        # --list-paths: a config.json each, no weights; the out_dir is never made
        cases = [
            ("qwen3_5", {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}, [],
             "step,verify,batch"),
            ("qwen3_5_moe", {"model_type": "qwen3_5_moe"}, [], "step,verify,batch"),
            ("flash-next", {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}, [],
             "step,verify"),
            ("flash-next text_config", {"text_config": {"model_type": "qwen4_exp_text"}}, [], "step,verify"),
            ("flint", {"model_type": "flint"}, [], "step,verify"),
            ("dory", {"model_type": "dory"}, [], "step"),
            ("dory per-loop", {"model_type": "dory"}, ["--loop-kv", "per-loop"], "step,verify"),
            ("nemotron-h", {"model_type": "nemotron_h"}, [], "step,verify,batch"),
        ]
        for name, cfg, extra, want in cases:
            mdir = os.path.join(tmp, "cfg-" + name.replace(" ", "-"))
            os.makedirs(mdir)
            with open(os.path.join(mdir, "config.json"), "w") as f:
                json.dump(cfg, f)
            out = os.path.join(tmp, "lp-" + name.replace(" ", "-"))
            r = subprocess.run([LOGIT_DUMP, mdir, out, "--list-paths", *extra], capture_output=True, text=True,
                               timeout=60, env=env)
            check(r.returncode == 0 and r.stdout.strip() == want and not os.path.exists(out),
                  f"--list-paths {name}: {want}", f"rc {r.returncode}: {r.stdout.strip()} {r.stderr.strip()}")
        r = subprocess.run([LOGIT_DUMP, os.path.join(tmp, "no-model"), os.path.join(tmp, "lp-none"), "--list-paths"],
                           capture_output=True, text=True, timeout=60, env=env)
        check(r.returncode == 1 and "no config.json" in r.stderr, "--list-paths without a config.json: refused",
              f"rc {r.returncode}: {r.stderr.strip()}")
    print(f"{len(FAILS)} failed" if FAILS else "all passed")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
