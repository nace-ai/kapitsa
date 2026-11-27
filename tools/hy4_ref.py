#!/usr/bin/env python3
"""Hy4-flash (tencent/Hy4-flash) against its reference: logits scoring and the Engram CPU lookup's check.

  hy4_ref.py score --cand F.f32 --ids FILE [--ref ID|FILE|DIR] [--rows A:B] [--per-row] [--json OUT]
      tools/hy4_dump's output: float32 [n, 120832], rows 0..n-1 of the prompt in FILE (comma- or space-separated
      ids; row p is the distribution of token p + 1). The reference is the file whose prompt starts with those ids
      (--ref: a prompt id, a <id>-native.f32 or a folder; default $HY4_REF_DIR).
  hy4_ref.py score CAND [--vs OTHER] [--rows A:B] [--prompts ID,..] [--ref DIR] [--per-row] [--json OUT]
      Candidate logits against the HF reference (transformers bf16 on CUDA, the 5 gate prompts). CAND is a folder of
      <id>.f32 files (float32 LE [n, 120832], rows 0..n-1 of the prompt, n <= 512) or a logit_dump path folder
      (sequences.txt, records.txt, logits.f32: each sequence is matched to a prompt by its tokens, record (s, L) is
      row L - 1). Row i of a prompt is the logits after its token i (they predict token i + 1).
  Both: per prompt (and pooled), top-1 agreement, KL(ref || cand) mean / p50 / p95 / p99 / max (float64
  log-softmax), the first row whose argmax differs (the first divergent greedy row), rows with non-finite values
  (left out of the KL, counted as top-1 misses). --per-row prints every row; --json OUT (or -) writes all of it.
  --vs OTHER: OTHER (same formats; with --cand, a .f32 of the same prompt, its ids in --vs-ids FILE, as many rows)
  is the reference instead (Kapitsa vs MLX).
  hy4_ref.py fromdir OUT [--rows A:B] [--prompts ID,..]
      The reference in logit_dump's --from layout (one folder, one sequence per prompt; records need L >= 2 and the
      next token, so rows 1..510).
  hy4_ref.py multipliers [--layers 2,12,22,32] [--pack DIR]
      engram_hash_multipliers / engram_head_offsets as the reference draws them (NumPy's default_rng(seed + 10007 *
      layer_id), bounded by the lookup table's max + 1), the tables read from the pack or by HTTP range reads.
  hy4_ref.py engram-test [--bin build/hy4_engram_check] [--work /tmp/hy4-engram] [--keep]
      src/model/hy4_engram.cpp against the reference's own functions (needs torch): a sparse pack in --work with
      the real layer-2 lookup table and head norms and a random 4-bit table, then two 64-token sequences. The
      table file (9.9 GB logical, several GB on disk) is deleted afterwards unless --keep.

Stdlib + NumPy (torch for engram-test). Paths from the environment: HY4_REF_DIR, the reference logits' folder
(<id>-native.f32 with their .f32.json sidecars); HY4_SRC_DIR, a folder with tencent/Hy4-flash's config.json and
model.safetensors.index.json (multipliers, engram-test); HY4_MODELING, transformers' modeling_hy_v4_flash.py at
c2ec8505 (engram-test). The HF range reads take HF_TOKEN, else the token `hf auth login` saved.
"""

import argparse
import ast
import json
import os
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

V = 120832
REF_DIR = os.environ.get("HY4_REF_DIR", "")
PROMPTS = ["final-000-prose", "final-003-code", "final-006-math", "final-010-multilingual", "final-013-instruction"]
SRC_DIR = Path(os.environ.get("HY4_SRC_DIR", "."))
MODELING = os.environ.get("HY4_MODELING", "modeling_hy_v4_flash.py")
REPO, REV = "tencent/Hy4-flash", "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"


# ---------------------------------------------------------------------------------------------------------- scoring
def parse_rows(spec, n):
    if not spec:
        return np.arange(n)
    a, _, b = spec.partition(":")
    lo = int(a) if a else 0
    hi = int(b) if b else n
    return np.arange(max(0, lo), min(n, hi))


def ref_ids(pid, ref_dir=REF_DIR):
    return json.load(open(f"{ref_dir}/{pid}-native.f32.json"))["input_ids"]


class Source:
    """Rows of each prompt from a folder of <id>.f32 files, the reference folder, or a logit_dump path folder."""

    def __init__(self, path, prompts, ref_dir=REF_DIR):
        self.path = Path(path)
        self.rows = {}  # prompt -> {row: memmap row}
        if (self.path / "records.txt").exists():
            seqs = [list(map(int, ln.split())) for ln in open(self.path / "sequences.txt") if ln.strip()]
            recs = [tuple(map(int, ln.split())) for ln in open(self.path / "records.txt") if ln.strip()]
            lg = np.memmap(self.path / "logits.f32", dtype="<f4", mode="r").reshape(-1, V)
            recorded = {si for si, _ in recs}
            for p in prompts:
                ids = ref_ids(p, ref_dir)
                match = [s for s, t in enumerate(seqs) if t == ids[: len(t)]]
                if match:
                    # Prefer the most specific sequence when one prompt is a prefix of another.
                    s = max(match, key=lambda i: (i in recorded, len(seqs[i])))
                    self.rows[p] = {L - 1: lg[k] for k, (si, L) in enumerate(recs) if si == s}
            return
        for p in prompts:
            for f in (self.path / f"{p}.f32", self.path / f"{p}-native.f32"):
                if f.exists():
                    if f.stat().st_size % (V * 4):
                        sys.exit(f"{f}: not a whole number of {V}-wide fp32 rows")
                    m = np.memmap(f, dtype="<f4", mode="r").reshape(-1, V)
                    self.rows[p] = {i: m[i] for i in range(m.shape[0])}
                    break


def log_softmax64(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def compare(ref_rows, cand_rows, rows):
    """Per row: KL(ref || cand) (nan where either row has a non-finite value), top-1 agreement, both argmaxes."""
    kl = np.full(len(rows), np.nan)
    top1 = np.zeros(len(rows), dtype=bool)
    am = np.zeros((len(rows), 2), dtype=np.int64)
    for c0 in range(0, len(rows), 32):
        rr = rows[c0 : c0 + 32]
        R = np.stack([ref_rows[r] for r in rr]).astype(np.float64)
        C = np.stack([cand_rows[r] for r in rr]).astype(np.float64)
        fin = np.isfinite(C).all(axis=1) & np.isfinite(R).all(axis=1)
        am[c0 : c0 + len(rr), 0] = R.argmax(1)
        am[c0 : c0 + len(rr), 1] = np.where(np.isfinite(C), C, -np.inf).argmax(1)
        top1[c0 : c0 + len(rr)] = (am[c0 : c0 + len(rr), 0] == am[c0 : c0 + len(rr), 1]) & fin
        if fin.any():
            lp, lq = log_softmax64(R[fin]), log_softmax64(C[fin])
            kl[c0 : c0 + len(rr)][fin] = (np.exp(lp) * (lp - lq)).sum(axis=1)
    return kl, top1, am


def stats(rows, kl, top1):
    ok = np.isfinite(kl)
    k = kl[ok]
    div = [r for r, t in zip(rows, top1) if not t]
    pct = lambda q: float(np.percentile(k, q)) if k.size else None  # noqa: E731
    return {
        "rows": len(rows),
        "row_range": [int(rows[0]), int(rows[-1])] if rows else None,
        "top1": float(np.mean(top1)) if len(rows) else None,
        "kl_mean": float(k.mean()) if k.size else None,
        "kl_p50": pct(50),
        "kl_p95": pct(95),
        "kl_p99": pct(99),
        "kl_max": float(k.max()) if k.size else None,
        "kl_max_row": int(np.asarray(rows)[ok][k.argmax()]) if k.size else None,
        "first_divergent_row": int(div[0]) if div else None,
        "nonfinite_rows": int((~ok).sum()),
    }


def fmt(name, s):
    if not s["rows"]:
        return f"{name:24s} no rows"
    if s["kl_mean"] is None:
        return f"{name:24s} n={s['rows']} all rows non-finite"
    rr = f" rows {s['row_range'][0]}..{s['row_range'][1]}" if s.get("row_range") else ""
    fd = s["first_divergent_row"]
    where = (f" (row {s['kl_max_row']})  first top-1 miss {'none' if fd is None else 'row ' + str(fd)}"
             if s.get("row_range") else "")
    return (
        f"{name:24s} n={s['rows']:4d}{rr}  top1 {s['top1']:.4f}  KL mean {s['kl_mean']:.5f} p50 {s['kl_p50']:.5f} "
        f"p95 {s['kl_p95']:.5f} p99 {s['kl_p99']:.5f} max {s['kl_max']:.4f}{where}  nonfinite {s['nonfinite_rows']}"
    )


def read_ids_file(path):
    return [int(x) for x in Path(path).read_text().replace(",", " ").replace("[", " ").replace("]", " ").split()]


def ref_for_ids(ref, ids):
    """The reference file whose prompt starts with `ids`: --ref a prompt id, a <id>-native.f32 file or a folder."""
    if Path(ref).is_file():
        cands = [Path(ref)]
    elif Path(f"{REF_DIR}/{ref}-native.f32").exists():
        cands = [Path(f"{REF_DIR}/{ref}-native.f32")]
    else:
        cands = sorted(Path(ref).glob("*-native.f32"))
    for f in cands:
        full = json.load(open(str(f) + ".json"))["input_ids"]
        if full[: len(ids)] == ids:
            return f, full
    sys.exit(f"no reference in {ref} starts with the {len(ids)} ids given")


def cmd_score(a):
    summary = sys.stderr if a.json == "-" else sys.stdout
    out = {"prompts": {}}
    all_kl, all_top1 = [], []
    per_row = {}
    if a.cand_file:  # hy4_dump's layout: one prompt, float32 [n, vocab] for rows 0..n-1
        if not a.ids:
            sys.exit("--cand needs --ids (the prompt the rows belong to)")
        ids = read_ids_file(a.ids)
        rf = Path(a.vs) if a.vs else ref_for_ids(a.ref, ids)[0]
        m = np.memmap(a.cand_file, dtype="<f4", mode="r")
        if m.size % V:
            sys.exit(f"{a.cand_file}: not a whole number of {V}-wide fp32 rows")
        m = m.reshape(-1, V)
        if m.shape[0] > len(ids):
            sys.exit(f"{a.cand_file}: {m.shape[0]} rows for {len(ids)} ids")
        r = np.memmap(rf, dtype="<f4", mode="r")
        if r.size % V:
            sys.exit(f"{rf}: not a whole number of {V}-wide fp32 rows")
        r = r.reshape(-1, V)
        if a.vs:  # (a reference from --ref may be longer: its prompt starts with the ids, ref_for_ids)
            if not a.vs_ids:
                sys.exit("--vs with --cand needs --vs-ids FILE, the prompt OTHER's rows belong to")
            if read_ids_file(a.vs_ids) != ids:
                sys.exit(f"{a.vs}: its prompt ({a.vs_ids}) is not the candidate's ({a.ids})")
            if r.shape[0] != m.shape[0]:
                sys.exit(f"{rf}: {r.shape[0]} rows, the candidate {m.shape[0]}: --vs compares the same prompt's rows")
        pid = Path(a.ids).stem if a.vs else rf.name.removesuffix("-native.f32")
        jobs = [(pid, {i: r[i] for i in range(r.shape[0])}, {i: m[i] for i in range(m.shape[0])})]
        out.update(reference=str(rf), candidate=a.cand_file)
    else:
        if not a.cand:
            sys.exit("score: a candidate folder, or --cand F.f32 --ids FILE")
        prompts = a.prompts.split(",") if a.prompts else PROMPTS
        ref = Source(a.vs or a.ref, prompts, a.ref)
        cand = Source(a.cand, prompts, a.ref)
        out.update(reference=str(ref.path), candidate=str(cand.path))
        jobs = []
        for p in prompts:
            if p not in cand.rows or p not in ref.rows:
                print(f"{p}: missing in {'candidate' if p not in cand.rows else 'reference'}; skipped", file=summary)
                continue
            jobs.append((p, ref.rows[p], cand.rows[p]))
        if not jobs:
            sys.exit("no requested prompt is present in both the reference and the candidate")
    sel = set(parse_rows(a.rows, 1 << 30).tolist()) if a.rows else None
    for p, rrows, crows in jobs:
        if set(rrows) != set(crows):
            print(f"{p}: the reference has {len(rrows)} rows, the candidate {len(crows)}: scoring the "
                  f"{len(set(rrows) & set(crows))} both have", file=summary)
        rows = [i for i in sorted(set(rrows) & set(crows)) if sel is None or i in sel]
        if not rows:
            sys.exit(f"{p}: no row in both the reference and the candidate")
        kl, top1, am = compare(rrows, crows, rows)
        s = stats(rows, kl, top1)
        out["prompts"][p] = s
        per_row[p] = [
            {"row": int(i), "kl": None if np.isnan(k) else float(k), "top1": bool(t), "ref": int(x), "cand": int(y)}
            for i, k, t, (x, y) in zip(rows, kl, top1, am)
        ]
        all_kl.append(kl)
        all_top1.append(top1)
        print(fmt(p, s), file=summary)
        if a.per_row:
            for e in per_row[p]:
                kls = "nonfinite" if e["kl"] is None else f"{e['kl']:.5f}"
                print(f"  row {e['row']:4d}  KL {kls:>10s}  argmax ref {e['ref']:6d} cand {e['cand']:6d}"
                      f"{'' if e['top1'] else '  MISS'}", file=summary)
    if len(jobs) > 1:
        kl, top1 = np.concatenate(all_kl), np.concatenate(all_top1)
        s = stats(list(range(len(kl))), kl, top1)
        s["row_range"] = s["kl_max_row"] = s["first_divergent_row"] = None  # pooled: no row numbers
        out["overall"] = s
        print(fmt("overall", s), file=summary)
    if a.json:
        out["per_row"] = per_row
        txt = json.dumps(out, indent=2) + "\n"
        sys.stdout.write(txt) if a.json == "-" else Path(a.json).write_text(txt)


def cmd_fromdir(a):
    prompts = a.prompts.split(",") if a.prompts else PROMPTS
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs, recs = [], []
    with open(out / "logits.f32", "wb") as f:
        for s, p in enumerate(prompts):
            ids = ref_ids(p, a.ref)
            seqs.append(" ".join(map(str, ids)))
            m = np.memmap(f"{a.ref}/{p}-native.f32", dtype="<f4", mode="r").reshape(-1, V)
            for r in parse_rows(a.rows, m.shape[0]):
                if 1 <= r <= len(ids) - 2:  # record L = r + 1 >= 2 with token L in the sequence
                    recs.append(f"{s} {r + 1}")
                    f.write(np.ascontiguousarray(m[r]).tobytes())
    (out / "sequences.txt").write_text("\n".join(seqs) + "\n")
    (out / "records.txt").write_text("\n".join(recs) + "\n")
    print(f"{out}: {len(prompts)} sequences, {len(recs)} records")


# ---------------------------------------------------------------------------------------------- the HF checkpoint
_headers = {}


def hf_range(fname, a, b):
    # curl rather than urllib: urllib's reads through the redirect to the CDN hang on this Mac (19 Nov)
    tok = os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").read_text().strip()
    r = subprocess.run(
        ["curl", "-sSfL", "--retry", "3", "-m", "300", "-H", "Authorization: Bearer " + tok, "-H",
         f"Range: bytes={a}-{b}", f"https://huggingface.co/{REPO}/resolve/{REV}/{fname}"],
        capture_output=True, check=True,
    )
    if len(r.stdout) != b - a + 1:
        sys.exit(f"{fname}: {len(r.stdout)} bytes for the range {a}-{b}")
    return r.stdout


def hf_tensor(name):
    """A small tensor of the source checkpoint by HTTP range reads: (dtype, shape, bytes)."""
    fname = json.load(open(SRC_DIR / "model.safetensors.index.json"))["weight_map"][name]
    if fname not in _headers:
        n = struct.unpack("<Q", hf_range(fname, 0, 7))[0]
        _headers[fname] = (n, json.loads(hf_range(fname, 8, 8 + n - 1)))
    n, h = _headers[fname]
    e = h[name]
    b0, b1 = e["data_offsets"]
    return e["dtype"], e["shape"], hf_range(fname, 8 + n + b0, 8 + n + b1 - 1)


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def lookup_table(layer_id, pack=None, cache=None):
    name = f"model.layers.{layer_id - 1}.engram_layer.lookup_table"
    if cache and (cache / f"lookup-{layer_id}.npy").exists():
        return np.load(cache / f"lookup-{layer_id}.npy")
    if pack:
        wm = json.load(open(Path(pack) / "model.safetensors.index.json"))["weight_map"]
        f = Path(pack) / wm[name]
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            e = json.loads(fh.read(n))[name]
            fh.seek(8 + n + e["data_offsets"][0])
            t = np.frombuffer(fh.read(e["data_offsets"][1] - e["data_offsets"][0]), dtype="<i4").copy()
    else:
        dt, shape, raw = hf_tensor(name)
        assert dt == "I32", dt
        t = np.frombuffer(raw, dtype="<i4").copy()
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / f"lookup-{layer_id}.npy", t)
    return t


def text_config():
    return json.load(open(SRC_DIR / "config.json"))["text_config"]


def engram_constants(tc, layer_id, table):
    """As convert_hy4_mlx.engram_extras / HYV4FlashEngramLayer.draw_hash_multipliers."""
    vocab = int(table.max()) + 1
    half = max(1, np.iinfo(np.int64).max // vocab // 2)
    g = np.random.default_rng(tc.get("engram_seed", 0) + 10007 * layer_id)
    drawn = g.integers(0, half, size=tc["engram_max_ngram_size"], dtype=np.int64)
    sizes = [s for order in tc["engram_layer_head_vocab_sizes"][str(layer_id)] for s in order]
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return [int(x) for x in drawn * 2 + 1], [int(x) for x in offs]


def cmd_multipliers(a):
    tc = text_config()
    mults, offs = {}, {}
    for lid in map(int, a.layers.split(",")):
        mults[str(lid)], offs[str(lid)] = engram_constants(tc, lid, lookup_table(lid, a.pack))
    print(json.dumps({"engram_hash_multipliers": mults, "engram_head_offsets": offs}))


# ------------------------------------------------------------------------------------------------- the Engram test
def reference_functions():
    """The reference's own Engram functions, cut out of modeling_hy_v4_flash.py (the module's imports need the
    transformers tree it was written for)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    src = Path(MODELING).read_text()
    tree = ast.parse(src)
    want = {
        ("", "_fp8_round_trip"),
        ("", "HYV4FlashEngramEmbedding"),
        ("HYV4FlashEngramLayer", "hash_ngrams"),
        ("HYV4FlashEngramLayer", "draw_hash_multipliers"),
        ("HYV4FlashTextModel", "build_ngram_tokens"),
        ("HYV4FlashModel", "build_engram_inputs"),
    }
    ns = {"torch": torch, "nn": nn, "F": F, "np": np}
    for node in tree.body:
        if ("", getattr(node, "name", None)) in want:
            exec(ast.get_source_segment(src, node), ns)
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if (node.name, getattr(sub, "name", None)) in want:
                    code = textwrap.dedent(ast.get_source_segment(src, sub, padded=True))
                    exec(code, ns)
                    ns[f"{node.name}.{sub.name}"] = ns.pop(sub.name)
    return ns


class NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def mlx_quantize(w):
    """mx.quantize(w, group_size=32, bits=4) for [n, 96] float32 rows: codes, scales, biases (bf16 bit patterns)."""
    g = w.reshape(len(w), 3, 32)
    lo, hi = g.min(-1), g.max(-1)
    scale = np.maximum((hi - lo) / 15.0, 1e-7).astype(np.float32)
    s16 = (scale.view(np.uint32) + 0x7FFF + ((scale.view(np.uint32) >> 16) & 1)) >> 16
    b16 = (lo.view(np.uint32) + 0x7FFF + ((lo.view(np.uint32) >> 16) & 1)) >> 16
    sf = (s16.astype(np.uint32) << 16).view(np.float32)
    bf = (b16.astype(np.uint32) << 16).view(np.float32)
    codes = np.clip(np.rint((g - bf[..., None]) / sf[..., None]), 0, 15).astype(np.uint32).reshape(len(w), 96)
    words = np.zeros((len(w), 12), dtype=np.uint32)
    for k in range(8):
        words |= codes[:, k::8] << np.uint32(4 * k)
    return codes, words, s16.astype("<u2"), b16.astype("<u2")


def write_sparse_pack(work, tc, layer_id, rows, mults, offs, lookup, norms, rng):
    """The pack folder: the table's full logical size with only `rows` written (a sparse file), the lookup table,
    the head norms, an index and a config. Returns {row: dequantized bf16 values as float32}."""
    L = layer_id - 1
    pre = f"model.layers.{L}.engram_layer."
    t = pre + "multi_head_embedding.embedding"
    sizes = [s for order in tc["engram_layer_head_vocab_sizes"][str(layer_id)] for s in order]
    nrows = sum(sizes)
    tens = [
        (t + ".biases", "BF16", [nrows, 3], nrows * 6),
        (t + ".scales", "BF16", [nrows, 3], nrows * 6),
        (t + ".weight", "U32", [nrows, 12], nrows * 48),
        (pre + "lookup_table", "I32", [len(lookup)], len(lookup) * 4),
    ]
    hdr, off = {"__metadata__": {"format": "mlx"}}, 0
    for n, dt, shape, nb in tens:
        hdr[n] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    hj = json.dumps(hdr).encode()
    hj += b" " * (-len(hj) % 8)
    base = 8 + len(hj)
    rows = np.array(sorted(rows), dtype=np.int64)
    # random rows, mostly normal, a few with a long tail (fp8 subnormals) and a few near zero
    w = rng.standard_normal((len(rows), 96)).astype(np.float32) * 0.02
    w[::7] *= rng.standard_exponential((len(w[::7]), 96)).astype(np.float32) ** 3
    w[::31] *= 1e-3
    codes, words, s16, b16 = mlx_quantize(w)
    sf = (s16.astype(np.uint32) << 16).view(np.float32)
    bf = (b16.astype(np.uint32) << 16).view(np.float32)
    deq = codes.reshape(-1, 3, 32).astype(np.float32) * sf[..., None] + bf[..., None]  # exact product, one rounding
    f = work / f"model-engram-L{L:02d}.safetensors"
    with open(f, "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)) + hj)
        fh.truncate(base + off)
        o = {n: base + hdr[n]["data_offsets"][0] for n, *_ in tens}
        for i, r in enumerate(rows.tolist()):
            fh.seek(o[t + ".weight"] + r * 48)
            fh.write(words[i].astype("<u4").tobytes())
            fh.seek(o[t + ".scales"] + r * 6)
            fh.write(s16[i].tobytes())
            fh.seek(o[t + ".biases"] + r * 6)
            fh.write(b16[i].tobytes())
        fh.seek(o[pre + "lookup_table"])
        fh.write(lookup.astype("<i4").tobytes())
    nh = {"__metadata__": {"format": "mlx"}}
    blob, off = b"", 0
    for h, g in enumerate(norms):
        nh[pre + f"lookup_per_head_norms.{h}.weight"] = {"dtype": "BF16", "shape": [96], "data_offsets": [off, off + 192]}
        blob += g
        off += 192
    nj = json.dumps(nh).encode()
    nj += b" " * (-len(nj) % 8)
    (work / "model-norms.safetensors").write_bytes(struct.pack("<Q", len(nj)) + nj + blob)
    wm = {n: f.name for n, *_ in tens}
    wm.update({k: "model-norms.safetensors" for k in nh if k != "__metadata__"})
    (work / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}, indent=1))
    cfg_tc = dict(tc)
    cfg_tc["engram_hash_multipliers"] = {str(layer_id): mults}
    cfg_tc["engram_head_offsets"] = {str(layer_id): offs}
    (work / "config.json").write_text(json.dumps({"model_type": "hy_v4_flash", "text_config": cfg_tc}, indent=1))
    return rows, deq.reshape(-1, 96)


def cuda_like(emb_bf16, gammas):
    """_fp8_round_trip and the head norms as torch 2.15 on CUDA runs them: elementwise ops in fp32 rounded to bf16,
    `amax / 448.0` as a product with the fp32 reciprocal, RMSNorm with eps = FLT_EPSILON in fp32 and one rounding."""
    import torch

    bf = torch.bfloat16
    x = emb_bf16
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = (amax.float() * torch.tensor(1.0 / 448.0, dtype=torch.float32)).to(bf)
    y = (x.float() / scale.float()).to(bf).clamp(-448, 448)
    q = (y.to(torch.float8_e4m3fn).float() * scale.float()).to(bf)
    r = (x.float() + (q.float() - x.float()).to(bf).float()).to(bf)
    xf = r.float()
    ms = (xf.double() ** 2).mean(-1, keepdim=True).float()
    rstd = 1.0 / torch.sqrt(ms + torch.finfo(torch.float32).eps)
    return (gammas.float() * (rstd * xf)).to(bf)


def cmd_engram_test(a):
    import torch

    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    tc = text_config()
    lid = 2
    ref = reference_functions()
    lookup = lookup_table(lid, cache=work / "src")
    normf = work / "src" / "norms.bin"
    if not normf.exists():
        normf.write_bytes(
            b"".join(hf_tensor(f"model.layers.{lid - 1}.engram_layer.lookup_per_head_norms.{h}.weight")[2] for h in range(16))
        )
    norms = [normf.read_bytes()[192 * h : 192 * (h + 1)] for h in range(16)]
    gam = torch.from_numpy(np.stack([bf16_to_f32(g) for g in norms])).to(torch.bfloat16)  # [16, 96]

    # the multipliers, by the reference's draw and by this script's (the converter's) code
    stub = NS(seed=tc["engram_seed"], layer_id=lid, max_ngram_size=tc["engram_max_ngram_size"])
    m_ref = ref["HYV4FlashEngramLayer.draw_hash_multipliers"](stub, int(lookup.max()) + 1).tolist()
    mults, offs = engram_constants(tc, lid, lookup)
    assert m_ref == mults, (m_ref, mults)
    print(f"layer {lid}: lookup table {len(lookup)} ids (max {lookup.max()}), multipliers {mults}")

    base = ref_ids("final-000-prose")[:64]
    seqB = list(base)
    seqB[0] = 120002  # pad at the start
    seqB[10] = 120120  # image
    seqB[20] = 120830  # past the lookup table: reads as the pad id
    seqB[21] = 120002  # pad
    seqB[33] = 120124  # video
    seqs = {"A": base, "B": seqB}

    sizes = [s for order in tc["engram_layer_head_vocab_sizes"][str(lid)] for s in order]
    layer = NS(
        lookup_table=torch.from_numpy(lookup.astype(np.int32)),
        pad_id=tc["engram_pad_id"],
        hash_multipliers=torch.tensor(mults, dtype=torch.int64),
        max_ngram_size=tc["engram_max_ngram_size"],
        n_head_per_ngram=tc["engram_n_head_per_ngram"],
        head_vocab_sizes=torch.tensor(sizes),
    )
    cfg = NS(engram_pad_id=tc["engram_pad_id"], engram_max_ngram_size=tc["engram_max_ngram_size"])
    model = NS(
        config=NS(image_token_id=120120, video_token_id=120124),
        language_model=NS(config=NS(engram_layer_ids=tc["engram_layer_ids"], engram_pad_id=tc["engram_pad_id"])),
    )
    model.language_model.build_ngram_tokens = lambda ids, pos: ref["HYV4FlashTextModel.build_ngram_tokens"](NS(config=cfg), ids, pos)
    hashes, masks = {}, {}
    for k, s in seqs.items():
        ng, mk = ref["HYV4FlashModel.build_engram_inputs"](model, torch.tensor([s]), None)
        hashes[k] = ref["HYV4FlashEngramLayer.hash_ngrams"](layer, ng)[0]  # [64, 16]
        masks[k] = mk[0]
    emb_mod = None
    with torch.device("meta"):
        emb_mod = ref["HYV4FlashEngramEmbedding"](sizes, 96)
    emb_mod.offsets = emb_mod.head_offsets()
    assert emb_mod.offsets.tolist() == offs
    touched = set()
    for k in seqs:
        touched |= set((hashes[k] + emb_mod.offsets).flatten().tolist())
    rng = np.random.default_rng(646)
    rows, deq = write_sparse_pack(work, tc, lid, touched, mults, offs, lookup, norms, rng)
    pos = {r: i for i, r in enumerate(rows.tolist())}
    small = torch.from_numpy(deq).to(torch.bfloat16)  # the table as the pack decodes it, in bf16

    class Gather(torch.nn.Module):
        def forward(self, idx):
            return small[torch.tensor([pos[int(i)] for i in idx.flatten()]).reshape(idx.shape)]

    emb_mod.embedding = Gather()
    du = sum(os.stat(p).st_blocks * 512 for p in work.glob("*.safetensors"))
    lg = sum(os.stat(p).st_size for p in work.glob("*.safetensors"))
    print(f"pack {work}: {len(rows)} table rows written, {lg / 1e9:.1f} GB logical, {du / 1e6:.1f} MB on disk")

    # the reference: literally (CPU torch's bf16 ops) and as CUDA computes it
    want_lit, want_cuda = {}, {}
    for k in seqs:
        e = emb_mod(hashes[k]).to(torch.bfloat16)  # [64, 16, 96]
        e8 = ref["_fp8_round_trip"](e)
        lit = []
        for h in range(16):
            n = torch.nn.RMSNorm(96).to(torch.bfloat16)
            n.weight.data.copy_(gam[h])
            lit.append(n(e8[:, h]))
        want_lit[k] = torch.stack(lit, 1).float().reshape(64, -1).detach()
        want_cuda[k] = cuda_like(e, gam[None]).float().reshape(64, -1)

    binp = Path(a.bin)
    cases = [("A", 0, 64), ("A", 37, 1), ("A", 63, 1), ("B", 0, 64), ("B", 21, 1), ("B", 22, 1), ("B", 40, 8)]
    worst = {"cuda": 0.0, "lit": 0.0}
    ok = True
    for k, p0, R in cases:
        tf = work / f"toks-{k}.txt"
        tf.write_text(" ".join(map(str, seqs[k][: p0 + R])))
        of = work / f"out-{k}-{p0}-{R}.bin"
        r = subprocess.run([str(binp), str(work), str(lid), str(tf), str(p0), str(R), str(of)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{binp} failed:\n{r.stdout}{r.stderr}")
        raw = of.read_bytes()
        got = np.frombuffer(raw[: R * 1536 * 4], dtype="<f4").reshape(R, 1536)
        gm = np.frombuffer(raw[R * 1536 * 4 : R * 1536 * 4 + R], dtype=np.uint8).astype(bool)
        gid = np.frombuffer(raw[R * 1536 * 4 + R :], dtype="<i8").reshape(R, 16)
        wm = masks[k][p0 : p0 + R].numpy()
        wid = (hashes[k] + emb_mod.offsets)[p0 : p0 + R].numpy()
        mask_ok = bool((gm == wm).all())
        ids_ok = bool((gid == wid).all())
        zero_ok = bool((got[~wm] == 0).all())
        d_c = np.abs(got[wm] - want_cuda[k][p0 : p0 + R].numpy()[wm])
        d_l = np.abs(got[wm] - want_lit[k][p0 : p0 + R].numpy()[wm])
        mc = float(d_c.max()) if d_c.size else 0.0
        ml = float(d_l.max()) if d_l.size else 0.0
        worst["cuda"] = max(worst["cuda"], mc)
        worst["lit"] = max(worst["lit"], ml)
        neq = int((d_c > 0).sum())
        ok = ok and mask_ok and ids_ok and zero_ok and mc == 0.0
        print(
            f"seq {k} pos {p0}..{p0 + R - 1}: mask {'agrees' if mask_ok else 'DIFFERS'} ({int(wm.sum())}/{R} rows own an "
            f"n-gram), ids {'agree' if ids_ok else 'DIFFER'}, masked rows zero {zero_ok}; max |diff| vs CUDA-order "
            f"{mc:.3g} ({neq} of {d_c.size} values differ), vs CPU-torch literal {ml:.3g}  [{r.stdout.strip()}]"
        )
    print(f"worst: vs CUDA-order reference {worst['cuda']:.3g}, vs CPU-torch literal {worst['lit']:.3g} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not a.keep:  # APFS fills much of the table file's holes (4.6 GB of 9.9 on 19 Nov): not left behind
        for f in work.glob("model-engram-*.safetensors"):
            f.unlink()
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("cand", nargs="?")
    s.add_argument("--cand", dest="cand_file")
    s.add_argument("--ids")
    s.add_argument("--per-row", action="store_true")
    s.add_argument("--vs")
    s.add_argument("--vs-ids")
    s.add_argument("--rows")
    s.add_argument("--prompts")
    s.add_argument("--ref", default=REF_DIR)
    s.add_argument("--json")
    f = sub.add_parser("fromdir")
    f.add_argument("out")
    f.add_argument("--rows")
    f.add_argument("--prompts")
    f.add_argument("--ref", default=REF_DIR)
    m = sub.add_parser("multipliers")
    m.add_argument("--layers", default="2,12,22,32")
    m.add_argument("--pack")
    e = sub.add_parser("engram-test")
    e.add_argument("--bin", default="build/hy4_engram_check")
    e.add_argument("--work", default="/tmp/hy4-engram")
    e.add_argument("--keep", action="store_true", help="keep the test pack's table file")
    a = ap.parse_args()
    {"score": cmd_score, "fromdir": cmd_fromdir, "multipliers": cmd_multipliers, "engram-test": cmd_engram_test}[a.cmd](a)


if __name__ == "__main__":
    main()
