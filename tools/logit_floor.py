#!/usr/bin/env python3
"""Measures a model's noise floor on this Mac: how far its logits move between conditions the engine legitimately
runs a request under, and the gate tools/logit_compare.py holds a change to (twice that).

  logit_floor.py <model_dir> [--bin build/logit_dump] [--work DIR] [--out FILE] [--keep] [--reuse] [--rescore]

Dumps (build/logit_dump, its default windows: 192 rows from bench/corpus.txt; each run takes the GPU lock unless
NI_LOCK_HELD=1) and compares, on the single-stream verify (Dory: plain decoding; Flint: no batch path):
  repeat  the same build and command twice, every path: expected bitwise equal
  batch   alone against 4 requests decoding together (Engine::batch_rows' rows; the 4B batches only with NI_BATCH=1)
  step    plain decoding (--spec 0) against the verify: the near-tie differences the engine already accepts
  chain   DFlash2's chain verify (NI_KFLASH=0; the batched pass's rows are chains too) against KFlash's tree verify
  chunk   the prefill in chunks of 512 rows against the engine's 2048 (a conversation's cached turns cut it elsewhere)
  ane     the Neural Engine split against the GPU alone (NI_ANE=0), where the model runs the split (dense Qwen, Bolt),
          and chunk_gpu: chunk with the GPU alone (where rows do not move between the Neural Engine and the GPU)
The floor is the worst of these sources per measure; the decode floor the worst of repeat, batch, step and chain (what
moves a row when the prefill stays bitwise the same). Each gate (logit_compare.gate_from): min_top1 = 1 - 2 max(flips,
1) / records, mean and p99 KL at most twice the floor's, no flip at a margin of its tie_margin or more (the largest
top-2 margin the floor moved by half of it: twice the floor's differences flip margins up to there). Writes
tools/logit_floors/<model>.json (--out FILE elsewhere): every source's numbers, the floors, the gates, the conditions.
--rescore derives the floors and gates of such a file again from its sources, without the GPU.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import logit_compare as lc  # noqa: E402

KEEP = ("records", "bitwise", "top1", "flips", "kl_mean", "kl_median", "kl_p99", "kl_max", "margin_change_max",
        "tie_margin", "nll_change_mean", "max_abs_diff")
FLOOR_SOURCES = ("repeat", "batch", "step", "chain", "chunk", "ane", "chunk_gpu")
DECODE_SOURCES = ("repeat", "batch", "step", "chain")  # what moves a row when the prefill stays bitwise the same
WHAT = {
    "repeat": "the same build and command run twice",
    "batch": "alone (the single-stream verify) against 4 requests decoding together (encode_verify_multi)",
    "chunk": "prefill chunks of 512 rows against 2048",
    "ane": "the Neural Engine prefill split against the GPU alone (NI_ANE=0)",
    "chunk_gpu": "prefill chunks of 512 rows against 2048 with the GPU alone (NI_ANE=0: no rows moving to the ANE)",
    "step": "plain decoding (--spec 0) against the 8-row verify",
    "chain": "DFlash2's chain verify (NI_KFLASH=0) against KFlash's tree verify",
}


def summary(s):
    out = {k: s[k] for k in KEEP}
    if s["flip_list"]:
        out["largest_flip_margin"] = s["flip_list"][0]["margin"]
    return out


def dumps(obj):
    """JSON with the top two levels on lines of their own and anything deeper on one line (a short file to review)."""
    def block(d, ind):
        items = [f'{" " * (ind + 2)}{json.dumps(k)}: '
                 + (block(v, ind + 2) if isinstance(v, dict) and ind < 2 else json.dumps(v)) for k, v in d.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * ind + "}"
    return block(obj, 0) + "\n"


def floor_and_gate(sources, key="in_floor"):
    """The floor (worst per measure over the sources marked `key`; the prefill path's 8 rows, a check that the prefill
    is bitwise the same, not a measure) and the gate twice it."""
    rows = [(name, s) for name, src in sources.items() if src.get(key)
            for p, s in (src["paths"].items() if "paths" in src else [(None, src)]) if p != "prefill"]
    if not rows:
        raise ValueError("no source in the floor")
    floor = lc.worst([s for _, s in rows])
    floor["top1"] = 1 - floor["flips"] / floor["records"]
    floor["set_by"] = {k: max(rows, key=lambda r: r[1][k])[0] for k in lc.MEASURES if floor[k] > 0}
    return floor, lc.gate_from(floor)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "build", "logit_dump"))
    ap.add_argument("--work", help="where the dumps go, ~1 GB per path at once (default build/logit_floor/<model>)")
    ap.add_argument("--out", help="default tools/logit_floors/<model>.json")
    ap.add_argument("--keep", action="store_true", help="keep the dumps")
    ap.add_argument("--reuse", action="store_true", help="compare the dumps --work already holds, not run them again")
    ap.add_argument("--rescore", action="store_true",
                    help="no dumps: the floors and gates of the model's JSON again from its sources (no GPU)")
    a = ap.parse_args()
    model = os.path.basename(os.path.normpath(a.model))
    if a.rescore:
        path = a.out or os.path.join(lc.FLOORS, model + ".json")
        with open(path) as f:
            result = json.load(f)
        result["sources"] = {k: result["sources"][k] for k in WHAT if k in result["sources"]}
        for k, v in result["sources"].items():
            v["what"], v["in_floor"], v["decode"] = WHAT[k], k in FLOOR_SOURCES, k in DECODE_SOURCES
        result["floor"], result["gate"] = floor_and_gate(result["sources"])
        result["decode_floor"], result["decode_gate"] = floor_and_gate(result["sources"], "decode")
        with open(path, "w") as f:
            f.write(dumps(result))
        print("->", path)
        return
    out = a.out or os.path.join(lc.FLOORS, model + ".json")
    work = a.work or os.path.join(HERE, "..", "build", "logit_floor", model)
    with open(os.path.join(a.model, "config.json")) as f:
        mtype = json.load(f).get("model_type", "")
    paths = ["step"] if mtype == "dory" else ["step", "verify"] if mtype == "flint" else ["step", "verify", "batch"]
    ref = "verify" if "verify" in paths else "step"

    def dump(name, run_paths, *extra, env=None):
        d = os.path.join(work, name)
        if a.reuse and all(os.path.exists(os.path.join(d, p, "meta.json")) for p in ["prefill", *run_paths]):
            return d
        cmd = [a.bin, a.model, d, "--paths", ",".join(run_paths), *extra]
        print("$", " ".join(([f"{k}={v}" for k, v in (env or {}).items()]) + cmd), flush=True)
        r = subprocess.run(cmd, env=dict(os.environ, **(env or {})))
        if r.returncode:
            sys.exit(f"logit_dump failed ({r.returncode})")
        return d

    def cmp(d1, d2, path):
        return lc.compare(lc.Dump(os.path.join(d1, path)), lc.Dump(os.path.join(d2, path)))

    os.makedirs(work, exist_ok=True)
    da = dump("a", paths)
    db = dump("b", paths)
    with open(os.path.join(da, ref, "meta.json")) as f:
        meta = json.load(f)
    sources = {"repeat": {"paths": {p: summary(cmp(da, db, p)) for p in ["prefill"] + paths}}}
    if not a.keep:
        shutil.rmtree(db)  # (disk: a dump is ~190 MB per path on the Qwen models)
    if "batch" in paths:
        s = lc.compare(lc.Dump(os.path.join(da, "verify")), lc.Dump(os.path.join(da, "batch")))
        sources["batch"] = summary(s)
    sources["chunk"] = summary(cmp(da, dump("chunk", [ref], "--chunk", "512"), ref))
    if meta["ane"]["evals"] > 0:
        dane = dump("ane", [ref], env={"NI_ANE": "0"})
        sources["ane"] = summary(cmp(da, dane, ref))
        sources["chunk_gpu"] = summary(cmp(dane, dump("chunk_gpu", [ref], "--chunk", "512", env={"NI_ANE": "0"}), ref))
    if "verify" in paths and "step" in paths:
        sources["step"] = summary(lc.compare(lc.Dump(os.path.join(da, "verify")), lc.Dump(os.path.join(da, "step"))))
    if "tree" in meta.get("verify", ""):
        sources["chain"] = summary(cmp(da, dump("chain", ["verify"], env={"NI_KFLASH": "0"}), "verify"))
    for k, v in sources.items():
        v["what"] = WHAT[k]
        v["in_floor"] = k in FLOOR_SOURCES
        v["decode"] = k in DECODE_SOURCES
    floor, gate = floor_and_gate(sources)
    dfloor, dgate = floor_and_gate(sources, "decode")
    result = {
        "model": model,
        "measured": {k: meta[k] for k in ("build", "mac", "macos", "date", "records", "source", "env")},
        "reference_path": ref,
        "sources": sources,
        "floor": floor,
        "gate": gate,
        "decode_floor": dfloor,
        "decode_gate": dgate,
    }
    result["measured"]["command"] = "tools/logit_floor.py " + a.model.replace(os.path.expanduser("~"), "~")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(dumps(result))
    print(json.dumps({"floor": floor, "gate": gate, "decode_floor": dfloor, "decode_gate": dgate}, indent=2))
    print("->", out)
    if not a.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
