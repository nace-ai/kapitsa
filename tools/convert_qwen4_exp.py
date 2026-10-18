#!/usr/bin/env python3
"""Converts Qwen/Qwen3.8-Flash-Next (qwen4_exp, 180B MoE, 360 GB bf16) into a Kapitsa / mlx-vlm / oMLX pack without
ever holding more than a few GB of the checkpoint: everything streams from Hugging Face by HTTP range reads.

  1. bench/quality/eval_qwen4_exp.py stream ... --stage S --dense D   (the quality evaluation's streaming pass also
     writes each layer's experts 4-bit g64 to S and every other tensor bf16 to D, plus the MTP head and vision)
  2. convert_qwen4_exp.py table --out PACK [--bits 4 --group 32]       (the PLE n-gram table, 128 shards, into
     PACK/ple.safetensors; resumable)
  3. convert_qwen4_exp.py finalize --stage S --dense D --out PACK --recipe SPEC   (dense tensors quantized per the
     recipe, the pack's files, index, config, PLE manifest, checks, sha256s in conversion.json)

Layout (names are mlx-vlm qwen4_exp's after its sanitize(), so mlx-vlm and oMLX load the folder as is):
  model-000NN-of-000MM.safetensors  00001 = embed_tokens + lm_head + hyper_connection_mixer, then one file per decoder
                                    layer, then the vision tower (bf16)
  ple.safetensors                   the table's 128 row shards (4-bit group 32: 100 bytes a row), for an mmap; in the
                                    index (oMLX) and in ple_manifest.json (mlx-vlm's mmap reader, text_config.ple_storage)
  mtp.safetensors                   the MTP head (mtp.*), in the index (oMLX's embedded head; mlx-vlm drops it)
  config.json                       Hugging Face's + MLX's quantization block (per-module overrides) + ple_storage
  conversion.json                   source revision, recipe, every file's size and sha256
RMSNorm weights stay zero-centred (applied as 1 + w by mlx-vlm and this model), except linear_attn.norm (plain w).
Quantization: MLX affine (bf16 scale and bias per group) with Kapitsa's re-quantizer (src/gguf/quant.cpp), not RTN.
Recipe SPEC as in tools/qwen4_exp_pack.py (e.g. '4,router=8,indexer=8,ple=8,embed=8'); the experts are always the
staged ones (their bits are checked against the recipe).
"""
import argparse
import json
import os
import shutil
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_exp_pack as P  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next"
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TABLE_HF = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}.weight"
TABLE_MLX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{}"
NSHARD = 128
COPY = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "LICENSE"]
log = P.log


# ---------------------------------------------------------------- table

def table_specs(bits, gs, rows=P.NGRAM["shard_rows"], dims=160):
    specs = []
    for k in range(NSHARD):
        n = TABLE_MLX.format(k)
        specs += [(n + ".weight", "U32", [rows, dims * bits // 32]), (n + ".scales", "BF16", [rows, dims // gs]),
                  (n + ".biases", "BF16", [rows, dims // gs])]
    return specs


def cmd_table(a):
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "ple.safetensors")
    done_path = os.path.join(a.out, ".ple.done.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    w = P.SafetensorsWriter(path, table_specs(a.bits, a.group))
    src = P.Remote(REPO, REV)
    todo = [k for k in range(NSHARD) if k not in done]
    log(f"table: {len(todo)} of {NSHARD} shards to do -> {path} ({a.bits}-bit group {a.group})")
    pool = ThreadPoolExecutor(a.conns)
    lock = threading.Condition()
    ready = {}

    def fetcher():
        for k in todo:
            arr = src.fetch([TABLE_HF.format(k)], pool)[TABLE_HF.format(k)]
            with lock:
                while len(ready) >= 2:
                    lock.wait()
                ready[k] = arr
                lock.notify_all()
    threading.Thread(target=fetcher, daemon=True).start()
    t0 = time.time()
    rng = np.random.default_rng(0)
    for j, k in enumerate(todo):
        with lock:
            while k not in ready:
                lock.wait(5)
            x = ready.pop(k)
            lock.notify_all()
        if x.shape != (P.NGRAM["shard_rows"], 160):
            raise SystemExit(f"shard {k}: shape {x.shape}")
        wq, s, b = P.quantize(x, a.bits, a.group)
        # check: a sample of rows dequantizes close to the bf16 rows (relative RMS error of a 4-bit grid ~ 0.1)
        sel = rng.choice(len(x), 256, replace=False)
        ref = P.bf16_to_f32(x[sel])
        err = P.dequantize_np(wq[sel], s[sel], b[sel], a.bits, a.group) - ref
        rel = float(np.sqrt((err ** 2).mean() / max((ref ** 2).mean(), 1e-30)))
        if not np.isfinite(rel) or rel > {3: 0.3, 4: 0.15, 5: 0.08, 6: 0.04, 8: 0.01}[a.bits]:
            raise SystemExit(f"shard {k}: relative error {rel:.3f}")
        n = TABLE_MLX.format(k)
        w.write(n + ".weight", wq)
        w.write(n + ".scales", s)
        w.write(n + ".biases", b)
        done.add(k)
        json.dump(sorted(done), open(done_path, "w"))
        el = time.time() - t0
        log(f"shard {k}: rel err {rel:.4f}; {j + 1}/{len(todo)}, {el / 60:.1f} min, "
            f"eta {el / (j + 1) * (len(todo) - j - 1) / 60:.0f} min")
    log("table done")


# ---------------------------------------------------------------- finalize

def quant_entry(q):
    return {"group_size": q["group_size"], "bits": q["bits"], "mode": "affine"}


def cmd_finalize(a):
    recipe = P.Recipe(a.recipe)
    out = a.out
    os.makedirs(out, exist_ok=True)
    stage, dense = P.Local(a.stage), P.Local(a.dense)
    hfdir = os.path.join(P.WORK, "flashnext-tmp", "hf")
    cfg = json.load(open(os.path.join(hfdir, "config.json")))
    tcfg = cfg["text_config"]
    L = tcfg["num_hidden_layers"]
    expected = json.load(open(a.expected)) if a.expected else None  # mlx-vlm parameter tree (name -> [shape, dtype])

    # experts' format from the stage files: words per row vs scales per row
    def staged_q(mod):
        w, s = stage.index[mod + ".weight"], stage.index[mod + ".scales"]
        cols = s[4][-1] * 64
        return {"bits": w[4][-1] * 32 // cols, "group_size": 64, "mode": "affine"}

    qblock = {"group_size": recipe.default["group_size"] if recipe.default else 64,
              "bits": recipe.default["bits"] if recipe.default else 16, "mode": "affine"}
    files = []  # (file name, [names])
    tensor_file = {}

    def write_file(fname, make_items, consume=()):
        """make_items() -> [(name, np array, dtype)]. A finished file (its marker records its tensors and their
        quantization entries) is not redone (resume); `consume`: stage files deleted once it is written."""
        path = os.path.join(out, fname)
        marker = os.path.join(out, ".done-" + fname + ".json")
        if os.path.exists(marker) and os.path.exists(path):
            m = json.load(open(marker))
            qblock.update(m["qblock"])
            names = m["tensors"]
        else:
            before = dict(qblock)
            items = make_items()
            w = P.SafetensorsWriter(path, [(n, d, list(arr.shape)) for n, arr, d in items], metadata={"format": "mlx"})
            for n, arr, d in items:
                if list(arr.shape) != w_shape(w, n) or arr.dtype != {"U32": np.uint32, "BF16": np.uint16,
                                                                      "I64": np.int64}[d]:
                    raise SystemExit(f"{n}: {arr.dtype} {arr.shape} does not match its header entry")
                w.write(n, arr)
            names = [n for n, _, _ in items]
            json.dump({"tensors": names, "qblock": {k: v for k, v in qblock.items() if before.get(k) != v}},
                      open(marker, "w"))
        for f in consume:
            for x in (f, f + ".done"):
                if os.path.exists(os.path.join(a.stage, x)):
                    os.remove(os.path.join(a.stage, x))
        for n in names:
            tensor_file[n] = fname
        files.append(fname)

    def w_shape(w, n):
        off, nb = w.off[n]
        return json.loads(w.header.decode())[n]["shape"]

    def dense_items(prefixes, src=dense, mtp=False):
        items = []
        for k in sorted(src.index):
            if not any(k.startswith(p) for p in prefixes):
                continue
            arr = src.array(k)
            mod = k[:-7] if k.endswith(".weight") else None
            role = P.role_of(k)
            q = recipe.q(role) if mod else None
            if q is not None and q["mode"] == "affine" and arr.dtype == np.uint16 and arr.ndim >= 2 \
                    and arr.shape[-1] % q["group_size"] == 0:
                wq, s, b = P.quantize(np.asarray(arr), q["bits"], q["group_size"])
                items += [(mod + ".weight", wq, "U32"), (mod + ".scales", s, "BF16"), (mod + ".biases", b, "BF16")]
                if (q["bits"], q["group_size"]) != (qblock["bits"], qblock["group_size"]):
                    qblock[mod] = quant_entry(q)
            else:
                items.append((k, np.asarray(arr), "BF16" if arr.dtype == np.uint16 else "I64"))
                if mod and role in ("inject",) and recipe.default is not None:
                    qblock[mod] = False
        return items

    def expert_items(prefix):
        items = []
        for k in sorted(stage.index):
            if not k.startswith(prefix) or not k.endswith(".weight") or ".switch_mlp." not in k:
                continue
            mod = k[:-7]
            q = staged_q(mod)
            want = recipe.q(P.role_of(k))
            if want is None or want["bits"] != q["bits"] or want["group_size"] != q["group_size"]:
                raise SystemExit(f"{mod}: staged {q['bits']}-bit g{q['group_size']}, recipe asks {want}")
            items += [(mod + ".weight", stage.array(mod + ".weight"), "U32"),
                      (mod + ".scales", stage.array(mod + ".scales"), "BF16"),
                      (mod + ".biases", stage.array(mod + ".biases"), "BF16")]
            if (q["bits"], q["group_size"]) != (qblock["bits"], qblock["group_size"]):
                qblock[mod] = quant_entry(q)
        return items

    nfiles = 1 + L + 1
    name = lambda j: f"model-{j:05d}-of-{nfiles:05d}.safetensors"  # noqa: E731
    t0 = time.time()
    write_file(name(1), lambda: dense_items(["language_model.model.embed_tokens.", "language_model.lm_head.",
                                             "language_model.model.hyper_connection_mixer."]))
    for i in range(L):
        pre = f"language_model.model.layers.{i}."
        write_file(name(2 + i), lambda pre=pre: [x for x in dense_items([pre]) if ".ngram_embedding." not in x[0]]
                   + expert_items(pre), consume=[f"experts-{i:02d}.safetensors"] if a.consume else [])
        log(f"layer {i}: {name(2 + i)} ({(time.time() - t0) / 60:.1f} min)")
    write_file(name(nfiles), lambda: dense_items(["vision_tower."]))
    write_file("mtp.safetensors", lambda: dense_items(["mtp."]) + expert_items("mtp."),
               consume=["experts-mtp.safetensors"] if a.consume else [])

    # the table (written by `table`), its manifest
    tpath = os.path.join(out, "ple.safetensors")
    if not os.path.exists(tpath):
        raise SystemExit("ple.safetensors missing: run `table` first")
    tdone = json.load(open(os.path.join(out, ".ple.done.json")))
    if len(tdone) != NSHARD:
        raise SystemExit(f"ple.safetensors: {len(tdone)} of {NSHARD} shards written")
    tl = P.Local(out)
    tq = None
    manifest_shards = []
    for k in range(NSHARD):
        n = TABLE_MLX.format(k)
        wi, si, bi = tl.index[n + ".weight"], tl.index[n + ".scales"], tl.index[n + ".biases"]
        rows = wi[4][0]
        q = {"bits": wi[4][1] * 32 // 160, "group_size": 160 // si[4][1], "mode": "affine"}
        tq = tq or q
        qblock[n] = quant_entry(q)
        for suf in (".weight", ".scales", ".biases"):
            tensor_file[n + suf] = "ple.safetensors"
        manifest_shards.append({"row_start": k * rows, "row_count": rows, **{
            nm: {"file": "ple.safetensors", "offset": ix[1], "dtype": ix[3], "shape": ix[4]}
            for nm, ix in (("weight", wi), ("scales", si), ("biases", bi))}})
    manifest = {"version": 2, "source_root": ".", "layout": "safetensors_ranges", "row_width": 160,
                "row_count": NSHARD * P.NGRAM["shard_rows"], "quantization": tq, "cache_rows": 0,
                "shards": manifest_shards}
    json.dump(manifest, open(os.path.join(out, "ple_manifest.json"), "w"), indent=1)

    # checks: every tensor's shape and dtype against the mlx-vlm parameter tree (quantized: packed shapes)
    allt = {}
    for f in set(tensor_file.values()):
        with open(os.path.join(out, f), "rb") as fh:
            nh = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(nh))
        for k, v in h.items():
            if k != "__metadata__":
                allt[k] = (v["dtype"], v["shape"])
    problems = []
    if expected:
        for k, (shape, dt) in expected.items():
            mod = k[:-7] if k.endswith(".weight") else None
            if k in allt and allt[k][0] in ("BF16", "I64"):
                if list(allt[k][1]) != list(shape):
                    problems.append(f"{k}: {allt[k][1]} != {shape}")
            elif mod and mod + ".scales" in allt:
                d, s = allt[k]
                q = qblock.get(mod) or qblock
                gs, bits = q["group_size"], q["bits"]
                want_w = list(shape[:-1]) + [shape[-1] * bits // 32]
                want_s = list(shape[:-1]) + [shape[-1] // gs]
                if d != "U32" or list(s) != want_w or list(allt[mod + ".scales"][1]) != want_s \
                        or list(allt[mod + ".biases"][1]) != want_s:
                    problems.append(f"{k}: quantized {s} / {allt[mod + '.scales'][1]} != {want_w} / {want_s}")
            else:
                problems.append(f"{k}: missing")
        extra = [k for k in allt if k not in expected and not (
            (k.endswith(".scales") or k.endswith(".biases")) and k.rsplit(".", 1)[0] + ".weight" in expected)]
        problems += [f"{k}: not a parameter of mlx-vlm's model" for k in extra]
    if problems:
        for p in problems[:30]:
            log("PROBLEM", p)
        raise SystemExit(f"{len(problems)} problems")
    log(f"checked {len(allt)} tensors" + (" against mlx-vlm's parameter tree" if expected else ""))

    # index, config, small files
    total = sum(os.path.getsize(os.path.join(out, f)) for f in set(tensor_file.values()))
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(tensor_file.items()))},
              open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=1)
    cfg["quantization"] = qblock
    cfg["quantization_config"] = qblock
    cfg["text_config"]["ple_storage"] = {"manifest": "ple_manifest.json", "cache_rows": 0}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for f in COPY:
        shutil.copyfile(os.path.join(hfdir, f), os.path.join(out, f))
    # checksums
    allfiles = sorted(set(tensor_file.values())) + ["model.safetensors.index.json", "config.json",
                                                     "ple_manifest.json"] + COPY
    sums = {}
    with ThreadPoolExecutor(8) as ex:
        for f, h in zip(allfiles, ex.map(lambda f: P.sha256_file(os.path.join(out, f)), allfiles)):
            sums[f] = {"size": os.path.getsize(os.path.join(out, f)), "sha256": h}
    json.dump({"source": {"repo": REPO, "revision": REV}, "recipe": a.recipe, "table": tq,
               "experts": "staged by bench/quality/eval_qwen4_exp.py stream (Kapitsa re-quantizer)",
               "converter": "tools/convert_qwen4_exp.py", "files": sums},
              open(os.path.join(out, "conversion.json"), "w"), indent=1)
    log(f"pack complete: {out}, {sum(v['size'] for v in sums.values()) / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["table", "finalize"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--stage")
    ap.add_argument("--dense")
    ap.add_argument("--recipe")
    ap.add_argument("--consume", action="store_true", help="delete each staged experts file once its pack file is "
                    "written (the pack then holds the only copy; saves 70 GB of disk)")
    ap.add_argument("--expected", help="parameter tree json (name -> [shape, dtype]) of mlx-vlm's qwen4_exp model and its "
                    "MTP drafter (mtp.*) to check every tensor against")
    a = ap.parse_args()
    if a.cmd == "table":
        cmd_table(a)
    else:
        cmd_finalize(a)


if __name__ == "__main__":
    main()
