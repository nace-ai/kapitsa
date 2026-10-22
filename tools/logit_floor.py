#!/usr/bin/env python3
"""Measures a model's noise floor on this Mac: how far its logits move between conditions the engine legitimately
runs a request under, and the gate tools/logit_compare.py holds a change to (twice that).

  logit_floor.py <model_dir> [--bin build/logit_dump] [--work DIR] [--out FILE] [--rows W] [--also ENV=VAL ...]
                 [--reorder] [--keep] [--reuse] [--rescore]

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
tools/logit_floors/<model>.json, or <model>.m<N>.json on an M5 or later (logit_compare.floor_name: its tensor_ops
paths sum in an order no earlier GPU runs), --out FILE elsewhere: every source's numbers, the floors, the gates, the
conditions.
--rows W: W rows a window (logit_dump --rows; default 24, 192 rows): Flint's floor is a few rows whose routing flips,
and 768 rows (--rows 96) move it 6x between builds where 192 move it 23x.
--also ENV=VAL: a path the engine also ships on this Mac (a kill switch, e.g. NI_QMT=0): every source again with it set,
named "<source> ENV=VAL"; the floor is the worst of all of them, so both paths pass the gate.
--reorder: the --also paths differ from the default only in the order their K reductions sum (the author's claim, as
logit_compare --reorder's): each is compared with the default as built (fast math) and with NI_MATH_SAFE=1 on both
sides, and twice the worst of their decode paths is the file's "reorder_gate", which logit_compare --reorder takes for
this model instead of its REORDER_GATE (reorder_floor_and_gate). It bounds a reorder's drift only: a change gated with
it still shows the per-GEMM check and the temperature-0 continuations of AGENTS.md.
--rescore derives the floors and gates of such a file again from its sources, without the GPU (default file: this
Mac's, logit_compare.floor_name).
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


def base(name):
    """A source's kind: "step NI_QMT=0" is a step source (measured with --also NI_QMT=0)."""
    return name.split(" ")[0]


def describe(name):
    k, _, also = name.partition(" ")
    return WHAT[k] + (f", with {also}" if also else "")


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
    # the flips and records of the source flipping the largest share of its rows (sources of more rows than others:
    # 6 flips in 768 rows is a share of 6 / 768, not of the 192 rows of a bitwise source beside it)
    worst_rate = max(rows, key=lambda r: (max(r[1]["flips"], 1) / r[1]["records"], r[1]["flips"]))[1]
    floor["flips"], floor["records"] = worst_rate["flips"], worst_rate["records"]
    floor["top1"] = 1 - floor["flips"] / floor["records"]
    floor["set_by"] = {k: max(rows, key=lambda r: r[1][k])[0] for k in lc.MEASURES if floor[k] > 0}
    return floor, lc.gate_from(floor)


def reorder_floor_and_gate(sources):
    """The reorder floor: the worst per measure of the reorder sources' decode paths, and their worst median KL; its
    gate is twice it (logit_compare.gate_from), max_median_kl twice the worst median (a reorder moves every row of a
    reordered prefill a little; Flint's router moves a few far, which set the mean and p99), max_nll_change twice the
    worst mean NLL change of any path, the prefill rows included."""
    sources = [src for src in sources.values() if src.get("in_reorder_gate", True)]
    dec = [s for src in sources for p, s in src["paths"].items() if p != "prefill"]
    every = [s for src in sources for s in src["paths"].values()]
    floor = lc.worst(dec)
    floor["kl_median"] = max(s["kl_median"] for s in dec)
    floor["top1"] = 1 - floor["flips"] / floor["records"]
    gate = lc.gate_from(floor)
    gate["max_median_kl"] = lc.up3(2 * floor["kl_median"])
    gate["max_nll_change"] = lc.up3(2 * max(abs(s["nll_change_mean"]) for s in every))
    return floor, gate


def this_mac():
    """This Mac's chip as meta.json's "mac" would name it (enough for logit_compare.chip_gen)."""
    try:
        return {"mac": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                      text=True).stdout.strip()}
    except OSError:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--bin", default=os.path.join(HERE, "..", "build", "logit_dump"))
    ap.add_argument("--work", help="where the dumps go, ~1 GB per path at once (default build/logit_floor/<model>)")
    ap.add_argument("--out", help="default tools/logit_floors/<model>.json (<model>.m<N>.json on an M5 or later)")
    ap.add_argument("--rows", type=int, help="rows a window (logit_dump --rows, default 24: 192 rows)")
    ap.add_argument("--also", action="append", default=[], metavar="ENV=VAL",
                    help="every source again with this set (a path the engine also ships), in the floor too")
    ap.add_argument("--reorder", action="store_true",
                    help="the --also paths only reorder the default's sums: their drift is the reorder_gate")
    ap.add_argument("--keep", action="store_true", help="keep the dumps")
    ap.add_argument("--reuse", action="store_true", help="compare the dumps --work already holds, not run them again")
    ap.add_argument("--rescore", action="store_true",
                    help="no dumps: the floors and gates of the model's JSON again from its sources (no GPU)")
    a = ap.parse_args()
    if a.reorder and not a.also:
        ap.error("--reorder compares the --also paths with the default: give one")
    model = os.path.basename(os.path.normpath(a.model))
    if a.rescore:
        path = a.out or lc.floor_name(model, this_mac())
        with open(path) as f:
            result = json.load(f)
        result["sources"] = {k: v for k, v in result["sources"].items() if base(k) in WHAT}
        for k, v in result["sources"].items():
            v["what"], v["in_floor"], v["decode"] = describe(k), base(k) in FLOOR_SOURCES, base(k) in DECODE_SOURCES
        result["floor"], result["gate"] = floor_and_gate(result["sources"])
        result["decode_floor"], result["decode_gate"] = floor_and_gate(result["sources"], "decode")
        if result.get("reorder_sources"):
            result["reorder_floor"], result["reorder_gate"] = reorder_floor_and_gate(result["reorder_sources"])
        with open(path, "w") as f:
            f.write(dumps(result))
        print("->", path)
        return
    work = a.work or os.path.join(HERE, "..", "build", "logit_floor", model)
    with open(os.path.join(a.model, "config.json")) as f:
        mtype = json.load(f).get("model_type", "")
    paths = ["step"] if mtype == "dory" else ["step", "verify"] if mtype == "flint" else ["step", "verify", "batch"]
    ref = "verify" if "verify" in paths else "step"

    rows = ["--rows", str(a.rows)] if a.rows else []

    def dump(name, run_paths, *extra, env=None):
        d = os.path.join(work, name)
        extra = (*extra, *rows)
        if a.reuse and all(os.path.exists(os.path.join(d, p, "meta.json")) for p in ["prefill", *run_paths]):
            return d
        cmd = [a.bin, a.model, d, "--paths", ",".join(run_paths), *extra]
        env = {**cond, **(env or {})}
        print("$", " ".join(([f"{k}={v}" for k, v in env.items()]) + cmd), flush=True)
        r = subprocess.run(cmd, env=dict(os.environ, **env))
        if r.returncode:
            sys.exit(f"logit_dump failed ({r.returncode})")
        return d

    def cmp(d1, d2, path):
        return lc.compare(lc.Dump(os.path.join(d1, path)), lc.Dump(os.path.join(d2, path)))

    def once(d1, d2, path):
        """cmp, then d2 deleted unless --keep (disk: 768 rows a path are ~0.4 GB a dump on Flint)."""
        s = cmp(d1, d2, path)
        if not a.keep:
            shutil.rmtree(d2)
        return s

    def measure():
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
        sources["chunk"] = summary(once(da, dump("chunk", [ref], "--chunk", "512"), ref))
        if meta["ane"]["evals"] > 0:
            dane = dump("ane", [ref], env={"NI_ANE": "0"})
            sources["ane"] = summary(cmp(da, dane, ref))
            sources["chunk_gpu"] = summary(once(dane, dump("chunk_gpu", [ref], "--chunk", "512", env={"NI_ANE": "0"}),
                                                ref))
        if "verify" in paths and "step" in paths:
            sources["step"] = summary(lc.compare(lc.Dump(os.path.join(da, "verify")), lc.Dump(os.path.join(da, "step"))))
        if "tree" in meta.get("verify", ""):
            sources["chain"] = summary(once(da, dump("chain", ["verify"], env={"NI_KFLASH": "0"}), "verify"))
        return sources, meta

    os.makedirs(work, exist_ok=True)
    top, cond = work, {}
    sources, meta = measure()
    for also in a.also:
        k, _, v = also.partition("=")
        work, cond = os.path.join(top, also), {k: v}
        more, _ = measure()
        sources.update({f"{name} {also}": src for name, src in more.items()})
    work, cond = top, {}
    reorder = {}
    if a.reorder:
        safe = dump("safe", paths, env={"NI_MATH_SAFE": "1"})
        for also in a.also:
            k, _, v = also.partition("=")
            safe_also = dump(f"safe {also}", paths, env={"NI_MATH_SAFE": "1", k: v})
            for name, d1, d2, what in [
                    (also, os.path.join(top, "a"), os.path.join(top, also, "a"), f"the default against {also}"),
                    (f"{also} NI_MATH_SAFE=1", safe, safe_also,
                     f"the default against {also}, both with NI_MATH_SAFE=1 (no fast math)")]:
                reorder[name] = {"what": what, "paths": {p: summary(cmp(d1, d2, p)) for p in ["prefill"] + paths}}
    out = a.out or lc.floor_name(model, meta)
    for k, v in sources.items():
        v["what"] = describe(k)
        v["in_floor"] = base(k) in FLOOR_SOURCES
        v["decode"] = base(k) in DECODE_SOURCES
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
    if reorder:
        result["reorder_sources"] = reorder
        result["reorder_floor"], result["reorder_gate"] = reorder_floor_and_gate(reorder)
    result["measured"]["command"] = " ".join(["tools/logit_floor.py", a.model.replace(os.path.expanduser("~"), "~"),
                                              *rows, *[f"--also {x}" for x in a.also],
                                              *(["--reorder"] if a.reorder else [])])
    if a.also:
        result["measured"]["also"] = a.also
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(dumps(result))
    print(json.dumps({"floor": floor, "gate": gate, "decode_floor": dfloor, "decode_gate": dgate,
                      **({"reorder_floor": result["reorder_floor"], "reorder_gate": result["reorder_gate"]}
                         if reorder else {})}, indent=2))
    print("->", out)
    if not a.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
