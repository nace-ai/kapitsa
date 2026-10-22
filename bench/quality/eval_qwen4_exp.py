#!/usr/bin/env python3
"""Quantization quality of Qwen3.8-Flash-Next (qwen4_exp, 180B MoE) against its bf16 checkpoint, which fits neither
this Mac's memory (360 GB) nor its disk: the checkpoint streams from Hugging Face one decoder layer at a time (HTTP
range reads), and every candidate runs through the same layer beside the bf16 model, each on its own hidden states,
with mlx-vlm's own qwen4_exp modules (fp32: 3e-6 relative to transformers' fp32 forward over the first 4 layers,
PLE and QSA's sparse selection included). The PLE n-gram table (102 GB) is never streamed: only the rows the
sequences read are fetched (`rows`).

  eval_qwen4_exp.py rows   --ids @f... [--long] --out rows.npz
  eval_qwen4_exp.py stream --ids @f... --rows rows.npz --work DIR --cand NAME=SPEC ... [--long]
                           [--heads 4,6,8] [--stage DIR --dense DIR] [--dump-ref F --topk K] [--out results.json]
  eval_qwen4_exp.py local  --ids @f... --rows rows.npz --work DIR --pack DIR --cand NAME=SPEC ... [--long]
                           [--dense DIR] [--dump-ref F --topk K] [--dump-hidden F] [--out results.json]

stream: the bf16 reference and the candidates, layer by layer; exact KL(bf16 || candidate) over the whole vocabulary
at the head, top-1 agreement, perplexity. With --stage/--dense it is also the conversion's streaming pass: each
layer's experts are written 4-bit (the codes of the candidates' 4-bit experts) to DIR/experts-NN.safetensors and every
other tensor bf16 (mlx-vlm names) to the dense DIR, plus the MTP head and the vision tower, for
tools/convert_qwen4_exp.py finalize. The reference's final hidden states (after hyper_connection_mixer, fp32) are
saved in --work so that `local` scores later recipes exactly without streaming again. Resumable: the chains' hidden
states are checkpointed in --work after every layer.
local: candidates whose experts are the pack's 4-bit experts and whose other tensors come from the pack or are
re-quantized from the dense bf16 folder; KL against the saved bf16 reference. --dump-hidden writes every layer's
output (4-stream state) for the first sequence's first 64 tokens (layer-by-layer debugging of an implementation).

SPEC: '<default>[,role=<q>...]', q = 4 / 3 / 8 / 4:32 / bf16 (table also fp8), roles of tools/qwen4_exp_pack.py
(experts, shared, router, gdn, attn, indexer, hc, mixer, ple, embed, head, table, inject). A non-bf16 default
implies router 8-bit and table 4-bit group 32. 'rtn:<spec>' quantizes with MLX's round-to-nearest instead of
Kapitsa's re-quantizer; '<spec>@bf16act' runs that chain with bf16 activations ('bf16@bf16act': the noise floor
of bf16 inference). 'pack' (local only): exactly the pack's tensors. Per-position KL goes to <out>_kl.npz.
--long adds one 4096-token sequence (the --ids files concatenated): positions past 2051 exercise QSA's sparse
selection (with 1024-token sequences attention is exactly dense), reported separately; --long-only scores just that.
Every sequence runs alone (batch 1): mlx-vlm's batched QSA prefill is wrong for B > 1.
Activations, norms and dequantized weights are fp32 on both sides (bf16 activations add noise of the size of an
8-bit quantization's error). GPU work holds /tmp/kapitsa-gpu.lock (released while waiting for the network).
"""
import argparse
import json
import math
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# fp32 means fp32: MLX runs fp32 GEMMs as TF32 on GPUs with neural accelerators (M5: 7.7e-4 relative error per
# GEMM, transformers' fp32 forward differed by 3e-4 per block) unless this is set before mlx loads
os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"))
import qwen4_exp_pack as P  # noqa: E402

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

REPO = "Qwen/Qwen3.8-Flash-Next"
REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
log = P.log
TABLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"


def read_ids(f):
    t = open(f[1:] if f.startswith("@") else f).read()
    return [int(x) for x in t.replace("\n", ",").split(",") if x.strip()]


# ---------------------------------------------------------------- units of the checkpoint

def unit_of(name):
    """'embed', ('layer', i), 'head', 'mtp', 'vision', or 'table' for a Hugging Face tensor name."""
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual."):
        return "vision"
    if ".ngram_embedding." in name:
        return "table"
    if ".layers." in name:
        return ("layer", int(name.split(".layers.")[1].split(".")[0]))
    if "embed_tokens" in name:
        return "embed"
    return "head"  # lm_head, hyper_connection_mixer


class Prefetch:
    """Downloads units in order, at most `budget` bytes ahead of the consumer (a background thread)."""

    def __init__(self, src, units, budget, conns):
        self.src, self.units, self.budget = src, units, budget
        self.by_unit = {}
        for k in src.index:
            self.by_unit.setdefault(unit_of(k), []).append(k)
        self.pool = ThreadPoolExecutor(conns)
        self.cv = threading.Condition()
        self.ready, self.used, self.err, self.bytes, self.t0 = {}, 0, None, 0, time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def size(self, u):
        return sum(self.src.index[k][2] for k in self.by_unit[u])

    def _run(self):
        try:
            for u in self.units:
                sz = self.size(u)
                with self.cv:
                    while self.used > 0 and self.used + sz > self.budget:
                        self.cv.wait()
                    self.used += sz
                arrs = self.src.fetch(self.by_unit[u], self.pool)
                with self.cv:
                    self.ready[u] = arrs
                    self.bytes += sz
                    self.cv.notify_all()
        except Exception as e:
            with self.cv:
                self.err = e
                self.cv.notify_all()

    def get(self, u):
        with self.cv:
            if u not in self.ready and self.err is None:
                P.GpuLock.release(mx.synchronize)
            while u not in self.ready and self.err is None:
                self.cv.wait()
            if self.err is not None:
                raise self.err
            return self.ready[u]

    def done(self, u):
        with self.cv:
            self.ready.pop(u)
            self.used -= self.size(u)
            self.cv.notify_all()

    def rate(self):
        return self.bytes / 1e6 / max(time.time() - self.t0, 1e-3)


# ---------------------------------------------------------------- weights per chain

class LayerTensors:
    """One unit's tensors under mlx-vlm names, bf16 (numpy uint16) or int64, plus quantized encodings made on demand
    (cached: every chain asking for the same tensor in the same format shares one encoding)."""

    def __init__(self, hf_arrays=None, mlx_arrays=None, quantized=None):
        self.bf16 = {}  # mlx name -> np array (uint16 bf16 or int64)
        if hf_arrays:
            for k, a in hf_arrays.items():
                for name, how in P.mlx_names(k, a.shape):
                    self.bf16[name] = P.transform(a, how)
        if mlx_arrays:
            self.bf16.update(mlx_arrays)
        self.q = dict(quantized or {})  # (module name, bits, gs) -> (w, s, b) numpy
        self.mx_cache = {}

    def names(self):
        mods = {k[:-7] for k in self.bf16 if k.endswith(".weight")}
        mods |= {k[0] for k in self.q}
        return mods

    def encoding(self, mod, q, method="kq"):
        """Kapitsa's re-quantizer ('kq'), or MLX's round-to-nearest ('rtn', mx.quantize as mlx_lm.convert)."""
        key = (mod, q["bits"], q["group_size"]) + (() if method == "kq" else (method,))
        if key not in self.q:
            if method == "rtn":
                w = mx.array(self.bf16[mod + ".weight"]).view(mx.bfloat16)
                wq, sc, bi = mx.quantize(w, q["group_size"], q["bits"])
                mx.eval(wq, sc, bi)
                self.q[key] = (np.array(wq), np.array(sc.view(mx.uint16)), np.array(bi.view(mx.uint16)))
            else:
                self.q[key] = P.quantize(self.bf16[mod + ".weight"], q["bits"], q["group_size"])
        return self.q[key]

    def mx_float(self, name, f32):
        key = ("f", name, f32)
        if key not in self.mx_cache:
            a = self.bf16[name]
            if a.dtype == np.uint16:
                m = mx.array(a).view(mx.bfloat16)
                m = m.astype(mx.float32) if f32 else m
            else:
                m = mx.array(a)
            self.mx_cache[key] = m
        return self.mx_cache[key]

    def mx_quant(self, mod, q, f32, method="kq"):
        key = ("q", mod, q["bits"], q["group_size"], f32, method)
        if key not in self.mx_cache:
            w, s, b = self.encoding(mod, q, method)
            dt = mx.float32 if f32 else mx.bfloat16
            self.mx_cache[key] = (mx.array(w), mx.array(s).view(mx.bfloat16).astype(dt),
                                  mx.array(b).view(mx.bfloat16).astype(dt))
        return self.mx_cache[key]

    def weights_for(self, recipe, prefix, f32, method="kq"):
        """(weights dict relative to `prefix`, {module: q}) for a chain's recipe. A recipe that names a tensor's
        role at bf16 keeps it bf16; 'pack' takes the encodings as given (local mode)."""
        out, qmods = {}, {}
        done = set()
        for (mod, bits, gs) in [k for k in list(self.q) if len(k) == 3]:
            if recipe == "pack" and mod.startswith(prefix):
                w, s, b = self.mx_quant(mod, {"bits": bits, "group_size": gs}, f32)
                r = mod[len(prefix):]
                out[r + ".weight"], out[r + ".scales"], out[r + ".biases"] = w, s, b
                qmods[r] = {"bits": bits, "group_size": gs, "mode": "affine"}
                done.add(mod)
        for name in self.bf16:
            if not name.startswith(prefix):
                continue
            mod = name[:-7] if name.endswith(".weight") else None
            r = name[len(prefix):]
            if mod in done:
                continue
            q = None
            if mod is not None and recipe != "pack":
                q = recipe.q(P.role_of(name))
            if q is not None and q["mode"] == "affine":
                w, s, b = self.mx_quant(mod, q, f32, method)
                rm = mod[len(prefix):]
                out[rm + ".weight"], out[rm + ".scales"], out[rm + ".biases"] = w, s, b
                qmods[rm] = q
            else:
                out[r] = self.mx_float(name, f32)
        # experts given only quantized (local mode) under a recipe: must match
        for (mod, bits, gs) in [k for k in list(self.q) if len(k) == 3]:
            if recipe == "pack" or not mod.startswith(prefix) or mod in done or (mod + ".weight") in self.bf16:
                continue
            q = recipe.q(P.role_of(mod + ".weight"))
            if q is None or q["bits"] != bits or q["group_size"] != gs:
                raise SystemExit(f"{mod}: only {bits}-bit g{gs} available here, recipe asks {q}")
            w, s, b = self.mx_quant(mod, q, f32)
            rm = mod[len(prefix):]
            out[rm + ".weight"], out[rm + ".scales"], out[rm + ".biases"] = w, s, b
            qmods[rm] = q
        return out, qmods


def build(module, weights, qmods):
    """Quantize the modules named in qmods, then load the weights strictly."""
    def pred(path, m):
        if path in qmods and hasattr(m, "to_quantized"):
            q = qmods[path]
            return {"group_size": q["group_size"], "bits": q["bits"], "mode": "affine"}
        return False
    if qmods:
        nn.quantize(module, class_predicate=pred)
    module.load_weights(list(weights.items()), strict=True)
    return module


class RowLookup:
    """The PLE table as the rows the sequences read (fetched by `rows`), in one chain's table format. Not an
    nn.Module: no parameters to load."""

    def __init__(self, ids, rows_f32, dtype):
        self.ids = np.asarray(ids, np.int64)
        self.rows = mx.array(rows_f32).astype(dtype)

    def __call__(self, row_ids):
        r = np.array(row_ids).astype(np.int64)
        pos = np.searchsorted(self.ids, r)
        pos = np.minimum(pos, len(self.ids) - 1)
        if not (self.ids[pos] == r).all():
            raise SystemExit("a PLE row the sequences read was not fetched: rerun `rows` with these --ids")
        return self.rows[mx.array(pos.astype(np.int32))]


def table_rows(rows_bf16, q, method="kq"):
    """fp32 rows as a table format would give them."""
    if q is None:
        return P.bf16_to_f32(rows_bf16)
    if q["mode"] == "fp8":  # the official FP8 checkpoint: E4M3 with one per-tensor scale
        x = P.bf16_to_f32(rows_bf16)
        scale = FP8_TABLE_SCALE
        return P.fp8_e4m3_round(x / scale) * scale
    if method == "rtn":
        wq, sc, bi = mx.quantize(mx.array(rows_bf16).view(mx.bfloat16), q["group_size"], q["bits"])
        return np.array(mx.dequantize(wq, sc, bi, q["group_size"], q["bits"], dtype=mx.float32))
    w, s, b = P.quantize(rows_bf16, q["bits"], q["group_size"])
    return P.dequantize_np(w, s, b, q["bits"], q["group_size"])


FP8_TABLE_SCALE = None


# ---------------------------------------------------------------- the model pieces (mlx-vlm)

def text_config(cfg):
    from mlx_vlm.models.qwen4_exp.config import TextConfig
    tc = dict(cfg["text_config"])
    tc.pop("ple_storage", None)  # the table rows come from RowLookup here
    return TextConfig.from_dict(tc)


def new_layer(tc, i):
    from mlx_vlm.models.qwen4_exp import language as L
    if i == 1:  # keep mlx-vlm from building the 128 table shards as parameters
        orig = L.ShardedEmbedding
        L.ShardedEmbedding = lambda *a, **k: None
        try:
            layer = L.Qwen4ExpDecoderLayer(tc, i)
        finally:
            L.ShardedEmbedding = orig
        layer.ple.ple_embedding.ngram_embedding = None
        return layer
    return L.Qwen4ExpDecoderLayer(tc, i)


class EmbedWrap(nn.Module):
    def __init__(self, tc):
        super().__init__()
        self.embed_tokens = nn.Embedding(tc.vocab_size, tc.hidden_size)


def new_mixer(tc):
    from mlx_vlm.models.qwen4_exp import language as L
    return L.Qwen4ExpGatedResidual(tc, use_combine=False)


def forward(layer, h, ids):
    """One decoder layer over whole sequences from an empty cache (a prefill, as mlx-vlm runs it)."""
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.qwen4_exp import language as L
    from mlx_vlm.models.qwen3_5.language import _create_qwen3_5_ssm_mask
    if layer.is_linear:
        cache = ArraysCache(size=4 if "ple" in layer else 2)
        mask = _create_qwen3_5_ssm_mask(h, cache)
    else:
        cache = L.QSAKVCache()
        mask = L._create_qwen4_exp_attention_mask(h, cache)
    return layer(h, ids, mask=mask, cache=cache, position_ids=None)


class Chain:
    def __init__(self, name, spec):
        self.name, self.spec = name, spec
        # 'rtn:<spec>': MLX round-to-nearest instead of Kapitsa's re-quantizer; '<spec>@bf16act': bf16 activations
        # for this chain (the noise floor of bf16 inference: 'bf16@bf16act')
        self.method = "rtn" if spec.startswith("rtn:") else "kq"
        spec = spec[4:] if spec.startswith("rtn:") else spec
        self.f32 = not spec.endswith("@bf16act")
        spec = spec[: -len("@bf16act")] if spec.endswith("@bf16act") else spec
        self.recipe = "pack" if spec == "pack" else P.Recipe(spec)
        self.h = None  # [B, N, 10240] (batch of the --ids sequences)
        self.hl = None  # [1, 4096, 10240] (--long)
        self.u = self.ul = None  # final mixer outputs

    def q(self, role):
        return None if self.recipe == "pack" else self.recipe.q(role)


# ---------------------------------------------------------------- scoring

class Scorer:
    """Exact KL(ref || cand) over the vocabulary, top-1 agreement, perplexity; optional top-K dump (ni_eval format)."""

    def __init__(self, seqs, names, chunk=256):
        self.seqs, self.names, self.chunk = seqs, names, chunk

    def run(self, u_ref, head_ref, cands, dump=None, topk=64, positions_from=0):
        """u_ref: [B, N, H] fp32; head_ref(): fp32 [V, H]; cands: name -> (u [B, N, H], head()). Returns stats."""
        B = len(self.seqs)
        res = {n: {"kl": [], "agree": 0, "nll": 0.0, "n": 0} for n in ["ref"] + list(cands)}
        if dump:
            fh = open(dump, "wb")
            fh.write(b"NIEVREF2" + struct.pack("<3I", B, head_ref().shape[0], topk))
        for b in range(B):
            seq = self.seqs[b]
            n = len(seq)
            tgt = mx.array(seq[1:])
            rows = []
            for c0 in range(0, n - 1, self.chunk):
                c1 = min(n - 1, c0 + self.chunk)
                lr = u_ref[b, c0:c1].astype(mx.float32) @ head_ref().T
                lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
                pr = mx.exp(lr)
                ar = mx.argmax(lr, -1)
                t = tgt[c0:c1]
                sel = slice(max(0, positions_from - c0), None)
                if dump:
                    ti = mx.argsort(-lr, axis=-1)[:, :topk]
                    tv = mx.take_along_axis(lr, ti, -1)
                    tl = mx.take_along_axis(lr, t[:, None], -1)[:, 0]
                    mx.eval(ti, tv, tl)
                    rows.append((np.array(tl, np.float32), np.array(ti).astype(np.uint32), np.array(tv, np.float32)))
                if c1 <= positions_from:
                    continue
                r = res["ref"]
                r["nll"] -= float(mx.sum(mx.take_along_axis(lr, t[:, None], -1)[sel]).item())
                r["n"] += (c1 - c0) - sel.start
                for name, (u, head) in cands.items():
                    lq = u[b, c0:c1].astype(mx.float32) @ head().T
                    lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
                    kl = mx.maximum(mx.sum(pr * (lr - lq), -1), 0)[sel]
                    r = res[name]
                    r["kl"] += np.array(kl).tolist()
                    r["agree"] += int(mx.sum((mx.argmax(lq, -1) == ar)[sel]).item())
                    r["nll"] -= float(mx.sum(mx.take_along_axis(lq, t[:, None], -1)[sel]).item())
                    r["n"] += (c1 - c0) - sel.start
            if dump:
                fh.write(struct.pack("<I", n) + np.array(seq, np.int32).tobytes())
                for tl, ti, tv in rows:
                    for p in range(len(tl)):
                        fh.write(tl[p:p + 1].tobytes() + np.stack([ti[p], tv[p].view(np.uint32)], 1).tobytes())
        if dump:
            fh.close()
        self.last_kl = {name: np.array(r["kl"], np.float32) for name, r in res.items() if name != "ref"}
        out = {}
        for name, r in res.items():
            e = {"ppl": math.exp(r["nll"] / max(r["n"], 1)), "positions": r["n"]}
            if name != "ref":
                k = np.array(r["kl"]) if r["kl"] else np.zeros(1)
                e.update({"kl_mean": float(k.mean()), "kl_median": float(np.median(k)),
                          "kl_p95": float(np.percentile(k, 95)), "kl_p99": float(np.percentile(k, 99)),
                          "kl_max": float(k.max()), "top1_agree": r["agree"] / max(r["n"], 1)})
            out[name] = e
        return out


# ---------------------------------------------------------------- rows

def cmd_rows(a):
    seqs = [read_ids(f) for f in a.ids]
    if a.long:
        seqs.append(sum(seqs, []))
    ids = np.unique(np.concatenate([P.ngram_ids(s).ravel() for s in seqs]))
    src = P.Remote(REPO, REV)
    log(f"{len(ids)} table rows")
    rows = np.zeros((len(ids), 160), np.uint16)

    def one(i):
        k, loc = divmod(int(ids[i]), P.NGRAM["shard_rows"])
        f, off, nb, dt, shape = src.index[f"{TABLE_PREFIX}{k}.weight"]
        rows[i] = np.frombuffer(src.read_small(f, off + loc * 320, off + loc * 320 + 320), np.uint16)
    t = time.time()
    with ThreadPoolExecutor(a.conns) as ex:
        for j, _ in enumerate(ex.map(one, range(len(ids)))):
            if j and j % 10000 == 0:
                log(f"{j} rows, {j / (time.time() - t):.0f}/s")
    np.savez(a.out, ids=ids, rows=rows)
    log(f"wrote {a.out}")


# ---------------------------------------------------------------- stream / local

def fp8_scale():
    """The per-tensor scale of the official FP8 checkpoint's n-gram table (one bf16 value)."""
    fp8 = P.Remote(REPO + "-FP8", "main")
    k = next(k for k in fp8.index if k.endswith("ngram_embedding.weight_scale") or
             (".ngram_embedding." in k and k.endswith("weight_scale")))
    f, off, nb, dt, shape = fp8.index[k]
    raw = np.frombuffer(fp8.read(f, off, off + nb), np.uint16)
    return float(P.bf16_to_f32(raw).ravel()[0])


def run(a, mode):
    global FP8_TABLE_SCALE
    if a.cpu:
        mx.set_default_device(mx.cpu)
        os.environ["NI_LOCK_HELD"] = "1"
    os.makedirs(a.work, exist_ok=True)
    seqs = [read_ids(f) for f in a.ids]
    names = [os.path.basename(f.lstrip("@")).rsplit(".", 1)[0] for f in a.ids]
    if a.max_tokens:
        seqs = [s[: a.max_tokens] for s in seqs]
    n = len(seqs[0])
    assert all(len(s) == n for s in seqs), "sequences must have one length"
    long_seq = sum(seqs, []) if a.long else None
    if a.long_only:  # the concatenation alone, as the batch
        seqs, names, long_seq = [sum(seqs, [])], ["long"], None
    rz = np.load(a.rows)
    row_ids, rows_bf16 = rz["ids"], rz["rows"]
    chains = [Chain(*c.split("=", 1)) for c in a.cand]
    if mode == "stream":
        chains.insert(0, Chain("bf16", "bf16"))
    if any(c.q("table") is not None and c.q("table")["mode"] == "fp8" for c in chains):
        FP8_TABLE_SCALE = fp8_scale()
        log(f"FP8 table scale {FP8_TABLE_SCALE}")
    dt = mx.float32 if a.f32 else mx.bfloat16


    if mode == "stream":
        src = P.Remote(REPO, REV)
        cfg = json.loads(src.read("config.json"))
    else:
        cfg = json.load(open(os.path.join(a.pack, "config.json")))
    tc = text_config(cfg)
    L = tc.num_hidden_layers if not a.layers else a.layers
    state_path = os.path.join(a.work, f"state-{mode}.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) and not a.restart else {}
    if state and state.get("cands") != a.cand:
        raise SystemExit(f"{state_path}: a run with other candidates; --restart to start over")
    start = state.get("next_layer", 0)

    # sources of tensors per unit
    if mode == "stream":
        units = (["embed"] if start == 0 else []) + [("layer", i) for i in range(start, L)] + ["head"]
        if a.stage and not a.layers:
            units += ["mtp", "vision"]
        pf = Prefetch(src, units, a.budget_gb * 1e9, a.conns)
        get_unit = lambda u: LayerTensors(hf_arrays=pf.get(u))  # noqa: E731
        done_unit = pf.done
    else:
        pack = P.Local(a.pack)
        dense = P.Local(a.dense) if a.dense else None

        def get_unit(u):
            pre = {"embed": ["language_model.model.embed_tokens."],
                   "head": ["language_model.lm_head.", "language_model.model.hyper_connection_mixer."]}.get(u)
            if pre is None:
                pre = [f"language_model.model.layers.{u[1]}."]
            bf, qd = {}, {}
            for k in pack.index:
                if any(k.startswith(p) for p in pre) and ".ngram_embedding." not in k:
                    if k.endswith(".scales") or k.endswith(".biases"):
                        continue
                    mod = k[:-7] if k.endswith(".weight") else None
                    if mod and mod + ".scales" in pack.index:
                        qc = cfg.get("quantization", {})
                        e = qc.get(mod)
                        e = e if isinstance(e, dict) else {}
                        bits, gs = e.get("bits", qc.get("bits")), e.get("group_size", qc.get("group_size"))
                        qd[(mod, bits, gs)] = (np.array(pack.array(k)), np.array(pack.array(mod + ".scales")),
                                               np.array(pack.array(mod + ".biases")))
                    else:
                        bf[k] = np.array(pack.array(k))
            if dense is not None:  # bf16 originals of everything but the experts
                for k in dense.index:
                    if any(k.startswith(p) for p in pre) and ".switch_mlp." not in k:
                        bf[k] = np.array(dense.array(k))
            return LayerTensors(mlx_arrays=bf, quantized=qd)
        done_unit = lambda u: None  # noqa: E731

    ids = mx.array(seqs)
    # ---- embeddings
    P.GpuLock.take()
    if start == 0:
        ut = get_unit("embed")
        P.GpuLock.take()
        for c in chains:
            w, qm = ut.weights_for(c.recipe, "language_model.model.", a.f32 and c.f32, c.method)
            e = build(EmbedWrap(tc), w, qm).embed_tokens
            cdt = dt if c.f32 else mx.bfloat16
            c.h = mx.tile(e(ids).astype(cdt), (1, 1, tc.hc_count))
            if long_seq:
                c.hl = mx.tile(e(mx.array([long_seq])).astype(cdt), (1, 1, tc.hc_count))
            mx.eval(c.h, *([c.hl] if c.hl is not None else []))
        if mode == "stream" and a.stage:
            write_dense(a.dense, "top", ut)
        del ut
        done_unit("embed")
    else:
        for c in chains:
            cdt = dt if c.f32 else mx.bfloat16
            c.h = mx.array(np.load(os.path.join(a.work, f"h-{mode}-{c.name}.npy"))).astype(cdt)
            p = os.path.join(a.work, f"hl-{mode}-{c.name}.npy")
            c.hl = mx.array(np.load(p)).astype(cdt) if long_seq else None
        log(f"resumed at layer {start}")
    rows_cache = {}
    hidden_dump = [] if getattr(a, "dump_hidden", None) else None
    t0 = time.time()
    for i in range(start, L):
        ut = get_unit(("layer", i))
        P.GpuLock.take()
        tl = time.time()
        prefix = f"language_model.model.layers.{i}."
        for c in chains:
            w, qm = ut.weights_for(c.recipe, prefix, a.f32 and c.f32, c.method)
            layer = build(new_layer(tc, i), w, qm)
            if i == 1:
                tq = c.q("table") if c.recipe != "pack" else PACK_TABLE_Q
                key = json.dumps(tq) + c.method
                if key not in rows_cache:
                    rows_cache[key] = table_rows(rows_bf16, tq, c.method)
                layer.ple.ple_embedding.ngram_embedding = RowLookup(row_ids, rows_cache[key],
                                                                    dt if c.f32 else mx.bfloat16)
            # one sequence at a time: mlx-vlm's batched QSA prefill (B > 1) is wrong (7-9% relative error after the
            # first attention layer in a 4 x 1024 batch, against the same sequences alone)
            rows = []
            for b in range(c.h.shape[0]):
                rows.append(forward(layer, c.h[b:b + 1], ids[b:b + 1]))
                mx.eval(rows[-1])
            c.h = mx.concatenate(rows, axis=0)
            mx.eval(c.h)
            if long_seq:
                c.hl = forward(layer, c.hl, mx.array([long_seq]))
                mx.eval(c.hl)
            if hidden_dump is not None and c is chains[0]:
                hidden_dump.append(np.array(c.h[0, :64].astype(mx.float32)))
            del layer
        if mode == "stream" and a.stage:
            write_layer(a, i, ut)
        del ut
        done_unit(("layer", i))
        mx.clear_cache()
        # checkpoint
        if not a.no_checkpoint and (i + 1 < L):
            # every GPU op (here the fp32 copies) before the lock goes; only file writes after it
            saves = [(f"h-{mode}-{c.name}.npy", np.array(c.h.astype(mx.float32))) for c in chains]
            if long_seq:
                saves += [(f"hl-{mode}-{c.name}.npy", np.array(c.hl.astype(mx.float32))) for c in chains]
            P.GpuLock.release(mx.synchronize)
            for f, arr in saves:
                np.save(os.path.join(a.work, f), arr)
            del saves
            json.dump({"cands": a.cand, "next_layer": i + 1}, open(state_path, "w"))
        el = time.time() - t0
        rate = f", {pf.rate():.0f} MB/s" if mode == "stream" else ""
        log(f"layer {i + 1}/{L} ({'GDN' if (i + 1) % 4 else 'attention'}) compute {time.time() - tl:.1f} s{rate}, "
            f"{el / 60:.1f} min")

    # ---- head
    ut = get_unit("head")
    P.GpuLock.take()
    heads = {}
    for c in chains:
        w, qm = ut.weights_for(c.recipe, "language_model.model.hyper_connection_mixer.", a.f32 and c.f32, c.method)
        mixer = build(new_mixer(tc), w, qm)
        c.u = mixer(c.h).astype(mx.float32)
        c.ul = mixer(c.hl).astype(mx.float32) if long_seq else None
        mx.eval(c.u, *([c.ul] if c.ul is not None else []))
        hw, hq = ut.weights_for(c.recipe, "language_model.", a.f32 and c.f32, c.method)
        if "lm_head" in hq:
            q = hq["lm_head"]
            heads[c.name] = (lambda W=hw["lm_head.weight"], S=hw["lm_head.scales"], B=hw["lm_head.biases"], q=q:
                             mx.dequantize(W, S, B, q["group_size"], q["bits"], dtype=mx.float32))
        else:
            heads[c.name] = (lambda W=hw["lm_head.weight"]: W.astype(mx.float32))
    if hidden_dump is not None:
        hidden_dump.append(np.array(chains[0].u[0, :64]))
    extra = {}
    if mode == "stream":
        for hb in (a.heads.split(",") if a.heads else []):
            q = P.parse_q(hb)
            wq = ut.mx_quant("language_model.lm_head", q, a.f32)
            extra[f"bf16+head{hb}"] = (chains[0].u, chains[0].ul,
                                       (lambda wq=wq, q=q: mx.dequantize(wq[0], wq[1], wq[2], q["group_size"],
                                                                         q["bits"], dtype=mx.float32)))
        np.save(os.path.join(a.work, "ul-ref.npy" if a.long_only else "u-ref.npy"), np.array(chains[0].u))
        if long_seq:
            np.save(os.path.join(a.work, "ul-ref.npy"), np.array(chains[0].ul))
        ref_u, ref_ul, ref_head = chains[0].u, chains[0].ul, heads["bf16"]
        cands = chains[1:]
    else:
        ref_u = mx.array(np.load(os.path.join(a.work, "ul-ref.npy" if a.long_only else "u-ref.npy")))
        ref_ul = mx.array(np.load(os.path.join(a.work, "ul-ref.npy"))) if long_seq else None
        hb = ut.bf16.get("language_model.lm_head.weight")
        if hb is None:
            raise SystemExit("local mode needs --dense (the bf16 lm_head) for the reference")
        ref_head = (lambda W=mx.array(hb).view(mx.bfloat16): W.astype(mx.float32))
        cands = chains
    if mode == "stream" and a.stage:
        write_dense(a.dense, "head", ut)

    sc = Scorer(seqs, names)
    out = {"repo": f"{REPO}@{REV}", "mode": mode, "layers": L, "sequences": names, "f32": a.f32, "candidates": {}}
    cd = {c.name: (c.u, heads[c.name]) for c in cands}
    cd.update({k: (v[0], v[2]) for k, v in extra.items()})
    dump_target = a.dump_ref
    if mode == "local" and dump_target:
        # the dump is of the FIRST candidate (e.g. the pack): its own top-K, scored as the reference
        c = cands[0]
        Scorer(seqs, names).run(c.u, heads[c.name], {}, dump=dump_target, topk=a.topk)
        log(f"wrote {dump_target} ({c.name})")
        dump_target = None
    out["candidates"] = sc.run(ref_u, ref_head, cd, dump=dump_target, topk=a.topk)
    kl_arrays = {f"seqs/{k}": v for k, v in sc.last_kl.items()}
    if a.long_only:
        out["long_sparse"] = sc.run(ref_u, ref_head, cd, positions_from=2051)
    if long_seq:
        scl = Scorer([long_seq], ["long"])
        cdl = {c.name: (c.ul, heads[c.name]) for c in cands}
        cdl.update({k: (v[1], v[2]) for k, v in extra.items()})
        out["long_sparse"] = scl.run(ref_ul, ref_head, cdl, positions_from=2051)
        out["long_dense"] = scl.run(ref_ul, ref_head, cdl, positions_from=0)
        kl_arrays.update({f"long/{k}": v for k, v in scl.last_kl.items()})
    for c in cands:
        out["candidates"][c.name]["spec"] = c.spec
    P.GpuLock.release(mx.synchronize)
    if hidden_dump is not None:
        np.savez(a.dump_hidden, ids=np.array(seqs[0][:64], np.int32),
                 **{f"layer{j:02d}": h for j, h in enumerate(hidden_dump[:-1])}, final_mixer=hidden_dump[-1])
        log(f"wrote {a.dump_hidden}")
    out["elapsed_min"] = round((time.time() - t0) / 60, 1)
    for k, v in out["candidates"].items():
        log(k, json.dumps(v))
    if long_seq:
        for k, v in out["long_sparse"].items():
            log("long>2051", k, json.dumps(v))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        np.savez(a.out.rsplit(".", 1)[0] + "_kl.npz", **kl_arrays)  # per-position KL (segments, tails)
    if mode == "stream":
        json.dump({"cands": a.cand, "next_layer": L, "complete": True}, open(state_path, "w"))
    if mode == "stream" and a.stage and not a.layers:  # the rest of the conversion's streaming pass
        for u in ("mtp", "vision"):
            ut = get_unit(u)
            if u == "mtp":
                write_layer(a, "mtp", ut)
            else:
                write_dense(a.dense, "vision", ut)
            del ut
            done_unit(u)
            log(f"staged {u}")


PACK_TABLE_Q = {"mode": "affine", "bits": 4, "group_size": 32}


# ---------------------------------------------------------------- conversion outputs of the streaming pass

def write_dense(dense_dir, unit, ut):
    """Every non-expert tensor of a unit, bf16 / int64 as stored, under mlx-vlm names."""
    os.makedirs(dense_dir, exist_ok=True)
    path = os.path.join(dense_dir, f"{unit}.safetensors")
    if os.path.exists(path + ".done"):
        return
    items = [(k, v) for k, v in sorted(ut.bf16.items()) if ".switch_mlp." not in k]
    dt = lambda v: "BF16" if v.dtype == np.uint16 else "I64"  # noqa: E731
    w = P.SafetensorsWriter(path, [(k, dt(v), v.shape) for k, v in items])
    for k, v in items:
        w.write(k, v)
    open(path + ".done", "w").close()


def write_layer(a, i, ut):
    """A layer's (or the MTP head's) experts at the conversion's bits (4-bit g64 by default) + its dense tensors."""
    tag = f"{i:02d}" if isinstance(i, int) else i
    os.makedirs(a.stage, exist_ok=True)
    path = os.path.join(a.stage, f"experts-{tag}.safetensors")
    if not os.path.exists(path + ".done"):
        q = {"bits": a.expert_bits, "group_size": 64}
        mods = sorted(k[:-7] for k in ut.bf16 if ".switch_mlp." in k)
        specs, data = [], []
        for m in mods:
            wq, s, b = ut.encoding(m, q)
            for suf, arr, d in ((".weight", wq, "U32"), (".scales", s, "BF16"), (".biases", b, "BF16")):
                specs.append((m + suf, d, arr.shape))
                data.append((m + suf, arr))
        w = P.SafetensorsWriter(path, specs)
        for k, arr in data:
            w.write(k, arr)
        open(path + ".done", "w").close()
    write_dense(a.dense, "mtp" if i == "mtp" else f"layer-{tag}", ut)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["rows", "stream", "local"])
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--rows")
    ap.add_argument("--out")
    ap.add_argument("--work")
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--long-only", action="store_true", help="only the 4096-token concatenation (as the batch)")
    ap.add_argument("--heads", default="")
    ap.add_argument("--stage")
    ap.add_argument("--dense")
    ap.add_argument("--pack")
    ap.add_argument("--expert-bits", type=int, default=4)
    ap.add_argument("--dump-ref")
    ap.add_argument("--dump-hidden")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--budget-gb", type=float, default=12.0)
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--bf16-act", dest="f32", action="store_false", help="bf16 activations (default fp32)")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    a = ap.parse_args()
    if a.cmd == "rows":
        return cmd_rows(a)
    if a.stage and not a.dense:
        raise SystemExit("--stage needs --dense")
    run(a, a.cmd)


if __name__ == "__main__":
    main()
