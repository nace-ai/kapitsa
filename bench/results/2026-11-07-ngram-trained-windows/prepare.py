#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
E = ROOT / "evidence"
E.mkdir(exist_ok=True)
MODEL = ROOT / "model-sushi"


def run(args, label, cwd=None):
    with (E / (label + ".log")).open("w") as f:
        subprocess.run([str(x) for x in args], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True)
    commands.append({"label": label, "argv": [str(x) for x in args], "cwd": str(cwd or ROOT)})
    (E / "prepare-commands.json").write_text(json.dumps(commands, indent=2) + "\n")


commands = []


def main():
    (ROOT / "prepare.pid").write_text(str(os.getpid()) + "\n")
    while not all((ROOT / x).exists() for x in ["base-built", "candidate-built", "verified-files/tokenizer.json.json", "verified-files/config.json.json"]):
        if (ROOT / "build-cpu.failed").exists() or (ROOT / "download-failed.json").exists():
            raise RuntimeError("preparation dependency failed; inspect retained logs")
        time.sleep(10)
    identities = {}
    for variant in ("base", "candidate"):
        tree = ROOT / variant
        build = tree / "build"
        flags = (build / "CMakeFiles/ni_run.dir/flags.make").read_text()
        args = []
        for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            args += shlex.split(re.search(r"^" + key + r" = (.*)$", flags, re.M).group(1))
        obj = ROOT / ("probe-" + variant + ".o")
        run(["/usr/bin/clang++", *args, "-c", ROOT / "probe.cpp", "-o", obj], "compile-probe-" + variant, build)
        link = shlex.split((build / "CMakeFiles/ni_run.dir/link.txt").read_text())
        link = [str(obj) if x.endswith("tools/ni_run.cpp.o") else x for x in link]
        link[link.index("-o") + 1] = str(ROOT / ("probe-" + variant))
        run(link, "link-probe-" + variant, build)
        identities[variant] = {"source": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tree, text=True).strip(),
                               "library_sha256": hashlib.sha256((build / "libnaceinfer.a").read_bytes()).hexdigest(),
                               "probe_sha256": hashlib.sha256((ROOT / ("probe-" + variant)).read_bytes()).hexdigest()}
        run([ROOT / ("probe-" + variant), "corpus", MODEL, tree / "bench/corpus.txt", E / ("corpus-" + variant + ".json")],
            "corpus-" + variant, ROOT)
    a = json.loads((E / "corpus-base.json").read_text())
    b = json.loads((E / "corpus-candidate.json").read_text())
    assert a == b
    sys.path.insert(0, str(ROOT / "candidate/tools"))
    import qwen4_exp_pack as pack
    config = json.loads((MODEL / "config.json").read_text())
    ng = pack.ngram_spec(config.get("text_config", config))
    expected = sorted({int(r) for seq in a["sequences"] for r in pack.ngram_ids(seq, ng=ng).reshape(-1)})
    assert expected == a["rows"]
    identities["row_hash_crosscheck"] = {"reference_candidate_python_match": True, "rows": len(expected)}
    identities["probe_source_sha256"] = hashlib.sha256((ROOT / "probe.cpp").read_bytes()).hexdigest()
    (E / "probe-identity.json").write_text(json.dumps(identities, indent=2) + "\n")
    from table_rows import TableRows
    table = TableRows()
    table.ensure(expected, "corpus", stream=True)
    # Both complete libraries decode the actual downloaded rows, with destination guards.
    output_hashes = []
    for variant in ("base", "candidate"):
        out = ROOT / (variant + "-real-rows.f32")
        run([ROOT / ("probe-" + variant), "rows", MODEL, E / "corpus-base.json", out], "real-row-decode-" + variant, ROOT)
        output_hashes.append(hashlib.sha256(out.read_bytes()).hexdigest())
    assert len(set(output_hashes)) == 1
    (E / "real-row-decode.json").write_text(json.dumps({"rows": len(expected), "sha256": output_hashes,
        "complete_bytes_equal": True, "bytes": (ROOT / "base-real-rows.f32").stat().st_size}, indent=2) + "\n")
    for variant in ("base", "candidate"):
        (ROOT / (variant + "-real-rows.f32")).unlink()
    # A fixed chat prefix for later greedy coverage checks, without ignoring end tokens.
    prompt = "<|im_start|>user\nWrite a detailed numbered list explaining the integers from 1 through 100, one sentence per integer. Begin with 1 and continue without stopping early.\n<|im_end|>\n<|im_start|>assistant\n"
    (E / "chat-prompt.txt").write_text(prompt)
    run([ROOT / "probe-candidate", "text", MODEL, E / "chat-prompt.txt", E / "chat-prompt.json"], "chat-prompt", ROOT)
    chat = json.loads((E / "chat-prompt.json").read_text())
    table.ensure(chat["rows"], "chat-prompt", stream=False)
    (E / "chat-prompt.ids").write_text(",".join(map(str, chat["sequences"][0])) + "\n")
    (ROOT / "rows-passed").write_text("Corpus/chat row coverage and full-library real row comparison passed.\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        (ROOT / "prepare-failed.json").write_text(json.dumps({"error": str(e), "time": time.time()}) + "\n")
        raise
