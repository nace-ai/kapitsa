#!/usr/bin/env python3
"""hy4_ref.py score's file checks on tiny synthetic logits (no model, no GPU; needs numpy).

  - --cand with a reference whose prompt is longer than the candidate's ids (ref_for_ids takes a reference that starts
    with them): scores the candidate's rows;
  - --vs with the same prompt and as many rows: scores;
  - --vs with another row count, with other ids, or without --vs-ids: refused with a message, before any scoring.

  python3 tools/hy4_ref_score_test.py   (exit status 1 if a case fails)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

V = 120832  # hy4_ref.py's row width
SCRIPT = Path(__file__).with_name("hy4_ref.py")


def write_rows(path, rows):
    rows.astype("<f4").tofile(path)


def write_ids(path, ids):
    Path(path).write_text(" ".join(map(str, ids)) + "\n")


def run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), "score", *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main():
    rng = np.random.default_rng(1339)
    fails = 0

    def check(name, ok, out):
        nonlocal fails
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            fails += 1
            print("    " + out.strip().replace("\n", "\n    "))

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        full = [int(x) for x in rng.integers(0, V, 12)]
        ref = rng.standard_normal((10, V)).astype(np.float32)
        write_rows(d / "p-native.f32", ref)
        (d / "p-native.f32.json").write_text(json.dumps({"input_ids": full}))
        cand = ref[:4] + 0.01 * rng.standard_normal((4, V)).astype(np.float32)
        write_rows(d / "cand.f32", cand)
        write_ids(d / "ids4.txt", full[:4])
        write_ids(d / "ids4b.txt", full[:3] + [(full[3] + 1) % V])
        write_rows(d / "other4.f32", ref[:4])
        write_rows(d / "other3.f32", ref[:3])

        rc, out = run("--cand", d / "cand.f32", "--ids", d / "ids4.txt", "--ref", d / "p-native.f32")
        check("a 10-row reference of a longer prompt scores a 4-row candidate", rc == 0 and "top-1" in out, out)

        rc, out = run("--cand", d / "cand.f32", "--ids", d / "ids4.txt", "--vs", d / "other4.f32",
                      "--vs-ids", d / "ids4.txt")
        check("--vs with the same prompt and as many rows scores", rc == 0 and "top-1" in out, out)

        rc, out = run("--cand", d / "cand.f32", "--ids", d / "ids4.txt", "--vs", d / "other3.f32",
                      "--vs-ids", d / "ids4.txt")
        check("--vs with 3 rows against 4 is refused", rc != 0 and "3 rows, the candidate 4" in out, out)

        rc, out = run("--cand", d / "cand.f32", "--ids", d / "ids4.txt", "--vs", d / "other4.f32",
                      "--vs-ids", d / "ids4b.txt")
        check("--vs with other ids is refused", rc != 0 and "is not the candidate's" in out, out)

        rc, out = run("--cand", d / "cand.f32", "--ids", d / "ids4.txt", "--vs", d / "other4.f32")
        check("--vs without --vs-ids is refused", rc != 0 and "needs --vs-ids" in out, out)

        # The shorter prompt is also a prefix match for p. Both sequences have their own records.
        write_rows(d / "short-native.f32", ref[:4])
        (d / "short-native.f32.json").write_text(json.dumps({"input_ids": full[:4]}))
        for order in ("short,p", "p,short"):
            dump = d / "dump"
            subprocess.run([sys.executable, str(SCRIPT), "fromdir", str(dump), "--ref", str(d),
                            "--prompts", order], check=True, capture_output=True)
            report = d / "score.json"
            rc, out = run(dump, "--ref", d, "--prompts", "short,p", "--json", report)
            counts = {p: s["rows"] for p, s in json.loads(report.read_text())["prompts"].items()} if rc == 0 else {}
            check(f"shared prefix ({order}): each prompt gets its own rows", counts == {"short": 2, "p": 9}, out)

        # A dump can list a sequence without recording any logits for it.
        dump = d / "short-only"
        subprocess.run([sys.executable, str(SCRIPT), "fromdir", str(dump), "--ref", str(d),
                        "--prompts", "short"], check=True, capture_output=True)
        with (dump / "sequences.txt").open("a") as f:
            f.write(" ".join(map(str, full)) + "\n")
        rc, out = run(dump, "--ref", d, "--prompts", "p", "--json", report)
        rows = json.loads(report.read_text())["prompts"]["p"]["rows"] if rc == 0 else 0
        check("an unrecorded full sequence still scores the available prefix", rows == 2, out)

    print("all checks passed" if not fails else f"{fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
