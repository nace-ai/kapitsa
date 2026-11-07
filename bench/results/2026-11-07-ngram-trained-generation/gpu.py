#!/usr/bin/env python3
"""Remote trained-weight validation; sparse-table scope is recorded explicitly."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"
MODEL = ROOT / "model-sushi"
ENV = {**os.environ, "NI_LOCK_HELD": "1", "KAPITSA_NO_AUTO_UPDATE": "1"}
COMMANDS = []


def call(args, label, cwd=None, check=True):
    path = E / (label + ".log")
    assert not path.exists(), f"refuse to overwrite {path}"
    start = time.time()
    with path.open("w") as f:
        rc = subprocess.run([str(x) for x in args], cwd=cwd or ROOT, env=ENV,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    COMMANDS.append({"label": label, "argv": [str(x) for x in args], "cwd": str(cwd or ROOT),
                     "exit": rc, "start": start, "elapsed_seconds": time.time() - start})
    (E / "gpu-commands.json").write_text(json.dumps(COMMANDS, indent=2) + "\n")
    if check:
        assert rc == 0, (label, rc)
    return rc


def fingerprint(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def check_floats(path):
    arr = np.memmap(path, dtype="<f4", mode="r")
    assert arr.size > 0
    nonzero = 0
    for a in range(0, arr.size, 1 << 20):
        v = arr[a:a + (1 << 20)]
        assert np.isfinite(v).all(), path
        nonzero += int(np.count_nonzero(v))
    assert nonzero > 0
    return {"bytes": path.stat().st_size, "sha256": fingerprint(path), "finite": True, "nonzero_values": nonzero}


def main():
    (ROOT / "gpu.pid").write_text(str(os.getpid()) + "\n")
    while not all((ROOT / x).exists() for x in ["base-built", "candidate-built"]):
        failures = [x for x in ["download-failed.json", "build-cpu.failed", "prepare-failed.json"] if (ROOT / x).exists()]
        if failures:
            raise RuntimeError(f"dependency failed: {failures}")
        time.sleep(15)
    # A private reference driver pauses before each plain decode input. The controller
    # downloads that token's true n-gram rows, then acknowledges and invalidates
    # the speculative gather made before the rows existed. No timing from this
    # instrumented run is performance evidence. Both libraries remain unchanged.
    build = ROOT / "base/build"
    source = (ROOT / "base/tools/ni_run.cpp").read_text()
    needle = "    m.encode_step(*e, s, pos, sp, uint32_t(step), true);"
    assert source.count(needle) == 1
    hook = '    fprintf(stderr, "ROW_REQUEST %d %d\\n", pos, s.tokens()[pos]);\n    fflush(stderr);\n    if (getchar() != \'\\n\') fatal("row controller disconnected");\n    if (auto* fn = dynamic_cast<Qwen4Exp*>(&m)) fn->ple_forget();\n'
    seeded = source.replace(needle, hook + needle)
    (ROOT / "seed-driver.cpp").write_text(seeded)
    flags_text = (build / "CMakeFiles/ni_run.dir/flags.make").read_text()
    flags = []
    for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
        flags += shlex.split(re.search(r"^" + key + r" = (.*)$", flags_text, re.M).group(1))
    obj = ROOT / "seed-driver.o"
    call(["/usr/bin/clang++", *flags, "-c", ROOT / "seed-driver.cpp", "-o", obj], "compile-seed-driver", build)
    link = shlex.split((build / "CMakeFiles/ni_run.dir/link.txt").read_text())
    link = [str(obj) if x.endswith("tools/ni_run.cpp.o") else x for x in link]
    link[link.index("-o") + 1] = str(ROOT / "seed-driver")
    call(link, "link-seed-driver", build)
    (E / "seed-driver-identity.json").write_text(json.dumps({"source_sha256": fingerprint(ROOT / "seed-driver.cpp"),
        "binary_sha256": fingerprint(ROOT / "seed-driver"), "base_library_sha256": fingerprint(build / "libnaceinfer.a"),
        "change": "Driver-only stdin pause before plain encode_step, then ple_forget() to invalidate the speculative gather performed before the input rows were fetched. Libraries unchanged; no timing claim."}, indent=2) + "\n")
    print("Seed driver compiled; waiting for weights, CPU tests and row coverage", flush=True)
    while not all((ROOT / x).exists() for x in ["weights-passed", "cpu-passed", "rows-passed"]):
        failures = [x for x in ["download-failed.json", "build-cpu.failed", "prepare-failed.json"] if (ROOT / x).exists()]
        if failures:
            raise RuntimeError(f"dependency failed: {failures}")
        time.sleep(15)
    from table_rows import TableRows
    table = TableRows()
    corpus = json.loads((E / "corpus-base.json").read_text())
    table.check(corpus["rows"])
    lock = open("/tmp/kapitsa-gpu.lock", "a")
    print("Waiting for the remote GPU lock", flush=True)
    fcntl.flock(lock, fcntl.LOCK_EX)
    (ROOT / "gpu-lock-acquired").write_text(str(time.time()) + "\n")
    try:
        call(["/usr/bin/sw_vers"], "host-os")
        call(["/usr/sbin/sysctl", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], "host-hardware")
        call(["/bin/df", "-h", str(ROOT)], "disk-before-gpu")
        (E / "gpu-environment.json").write_text(json.dumps({k: v for k, v in ENV.items()
            if k.startswith(("NI_", "NACE_", "KAPITSA_", "MTL_", "BENCH_"))}, indent=2) + "\n")
        # Every input of the default eight teacher-forced windows is already covered.
        for variant in ("base", "candidate"):
            table.check(corpus["rows"])
            args = [ROOT / variant / "build/logit_dump", MODEL, ROOT / ("logits-" + variant), "--paths", "step,verify"]
            if variant == "candidate":
                args += ["--from", ROOT / "logits-base"]
            call(args, "logits-" + variant, ROOT / variant)
        comparisons = []
        for name in ("prefill", "step", "verify"):
            a, b = ROOT / "logits-base" / name, ROOT / "logits-candidate" / name
            assert (a / "sequences.txt").read_bytes() == (b / "sequences.txt").read_bytes()
            assert (a / "records.txt").read_bytes() == (b / "records.txt").read_bytes()
            seqs = [[int(x) for x in re.split(r"[,\s]+", line.strip()) if x] for line in (a / "sequences.txt").read_text().splitlines() if line.strip()]
            assert seqs == corpus["sequences"], name
            fa, fb = check_floats(a / "logits.f32"), check_floats(b / "logits.f32")
            assert fa == fb, name
            comparisons.append({"path": name, **fa, "byte_equal": True,
                                "records": json.loads((a / "meta.json").read_text())["records"]})
        rc = call([sys.executable, ROOT / "candidate/tools/logit_compare.py", ROOT / "logits-base", ROOT / "logits-candidate"],
                  "logit-compare", check=False)
        (E / "logit-byte-comparison.json").write_text(json.dumps({"paths": comparisons, "comparator_exit": rc,
            "coverage": table.check(corpus["rows"]), "scope": "Pinned actual Sushi weights; sparse table contains every input row of these exact windows. No full-table or calibrated-floor claim."}, indent=2) + "\n")
        (ROOT / "logits-passed").write_text("All actual-weight full-vocabulary logits byte equal, finite, nonzero.\n")
        # Fill greedy-input rows before the reference computes them, avoiding zero-hole sampling.
        sys.path.insert(0, str(ROOT / "candidate/tools"))
        import qwen4_exp_pack as pack
        cfg = json.loads((MODEL / "config.json").read_text())
        ng = pack.ngram_spec(cfg.get("text_config", cfg))
        prompt = json.loads((E / "chat-prompt.json").read_text())["sequences"][0]
        sequence = list(prompt)
        table.check(json.loads((E / "chat-prompt.json").read_text())["rows"])
        seed_args = [str(ROOT / "seed-driver"), str(MODEL), "@" + str(E / "chat-prompt.ids"), "256", "0",
                     "--dflash", "none", "--dump-all-logits", str(ROOT / "seed-logits.f32")]
        with (E / "seed-stdout.log").open("w") as out, (E / "seed-stderr.log").open("w") as err:
            proc = subprocess.Popen(seed_args, cwd=ROOT, env=ENV, stdin=subprocess.PIPE, stdout=out,
                                    stderr=subprocess.PIPE, text=True, bufsize=1)
            try:
                for line in proc.stderr:
                    err.write(line); err.flush()
                    if line.startswith("ROW_REQUEST "):
                        _, pos, token = line.split()
                        assert int(pos) == len(sequence)
                        sequence.append(int(token))
                        rows = [int(x) for x in pack.ngram_ids(sequence, ng=ng)[-1]]
                        table.ensure(rows, "seed-input-" + pos, stream=False)
                        proc.stdin.write("\n"); proc.stdin.flush()
                assert proc.wait() == 0
            except BaseException:
                proc.terminate(); proc.wait()
                raise
        text = (E / "seed-stdout.log").read_text()
        generated = [int(x) for x in re.search(r"^ids:(.*)$", text, re.M).group(1).split()]
        assert len(generated) == 256 and sequence == prompt + generated[:-1]
        configured_eos = cfg.get("text_config", cfg).get("eos_token_id", cfg.get("eos_token_id", [248044, 248046]))
        end_tokens = {248044, 248046} | set(configured_eos if isinstance(configured_eos, list) else [configured_eos])
        assert not any(x in end_tokens for x in generated), "early EOS; keep evidence, choose a new prompt in a later labeled run"
        (E / "generated.json").write_text(json.dumps({"sequences": [generated]}) + "\n")
        call([ROOT / "probe-candidate", "decode", MODEL, E / "generated.json", E / "generated-text.json"], "decode-generated")
        decoded = json.loads((E / "generated-text.json").read_text())["texts"][0]
        assert decoded.strip()
        # The unmodified runs also speculatively gather the final output token,
        # although its embedding is not consumed. Cover it as well before those runs.
        required = sorted({int(x) for x in pack.ngram_ids(prompt + generated, ng=ng).reshape(-1)})
        coverage = table.ensure(required, "greedy-final-coverage", stream=False)
        seed_fp = check_floats(ROOT / "seed-logits.f32")
        results = []
        for i, variant in enumerate(("base", "candidate", "base", "candidate"), 1):
            label = f"greedy-{i:02d}-{variant}"
            table.check(required)
            call(["/usr/bin/pmset", "-g", "therm"], label + "-thermal", check=False)
            dump = ROOT / (label + ".f32")
            call([ROOT / variant / "build/ni_run", MODEL, "@" + str(E / "chat-prompt.ids"), "256", "0",
                  "--dflash", "none", "--dump-all-logits", dump], label)
            ids = [int(x) for x in re.search(r"^ids:(.*)$", (E / (label + ".log")).read_text(), re.M).group(1).split()]
            fp = check_floats(dump)
            assert ids == generated and fp == seed_fp
            results.append({"label": label, "tokens": len(ids), "eos_count": 0, "text_nonempty": True,
                            "all_ids_equal": True, "all_logits_byte_equal": True, **fp})
            dump.unlink()
        (E / "greedy-results.json").write_text(json.dumps({"runs": results, "coverage": coverage,
            "seed_driver_instrumented": True, "measurement_drivers_unmodified": True,
            "seed_prefetch_caveat": "Before filling each newly sampled input, an unused speculative gather may read sparse holes. The seed driver invalidates it with ple_forget() before computing that input; all consumed rows are verified. Unmodified A/B runs start with every greedy input and final speculative row present.",
            "full_checkpoint_file": False, "http_and_native_bench": "Not run; arbitrary generation rows are not covered by a sparse table."}, indent=2) + "\n")
        (ROOT / "seed-logits.f32").unlink()
        (ROOT / "gpu-passed").write_text("Actual-weight covered-window logits and four covered 256-token greedy runs pass.\n")
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        (ROOT / "gpu-lock-released").write_text(str(time.time()) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        (ROOT / "gpu-failed.json").write_text(json.dumps({"error": str(e), "time": time.time()}) + "\n")
        raise
