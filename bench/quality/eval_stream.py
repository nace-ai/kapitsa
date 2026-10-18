#!/usr/bin/env python3
"""Quantization quality of a Qwen3.5 / 3.6 / 3.8 model too large for this Mac in BF16, against the true BF16 model.

The BF16 checkpoint (an MLX conversion, e.g. mlx-community/Qwen3.6-35B-A3B-bf16) never touches the disk: its
tensors stream from Hugging Face into RAM one decoder layer at a time (HTTP range requests), and the held-out
sequences run through each layer in BF16 with mlx-lm's
own modules. Every candidate quantization runs through the same layer beside it, on its own hidden states:
  rtn:<spec>   round-to-nearest from the BF16 layer with mx.quantize (what mlx_lm.convert does), recipe below
  repo:<id>    the candidate's own published tensors, streamed the same way (DWQ, mixed recipes, ...)
At the end the reference's final norm and LM head give fp32 log-probs; each candidate's (its own head, dequantized
to fp32) are compared position by position: exact KL(BF16 || candidate) over the whole vocabulary, top-1 agreement,
perplexity. The BF16 run's top-K log-probs can be written as a reference file for Kapitsa's tools/ni_eval --ref.

  eval_stream.py --base <repo>[@rev] --cand NAME=SPEC ... --ids @f1 @f2 ... [--out results.json]
                 [--dump-ref ref.bin] [--topk 256] [--budget-gb 6] [--conns 16] [--layers N] [--local DIR]

rtn spec: rtn:<bits>[:<group>] or rtn:<mode> (mxfp4, nvfp4, mxfp8), then optional ",role=<bits>[:<group>]" overrides
for roles embed, head, attn, gdn, router, experts, shared, mlp. MLX's own default for MoE models (8-bit router and
shared-expert gate) applies unless overridden. Example: rtn:4,attn=8,gdn=8,embed=8,head=8 (8-bit except the experts).
--local DIR reads the base tensors from a local MLX directory instead (tests). --layers N stops after N layers
(debug: the head then runs on partial hidden states). GPU work holds ~/.cache/nace-gpu.lock, released while waiting
for downloads.
"""
import argparse
import fcntl
import json
import math
import os
import struct
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import qwen3_5

UA = {"User-Agent": "kapitsa-quant-eval"}
LOCK = os.path.expanduser("~/.cache/nace-gpu.lock")
NP_DT = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "U32": np.uint32, "I32": np.int32, "U8": np.uint8}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class GpuLock:
    held = None

    @staticmethod
    def take():
        if os.environ.get("NI_LOCK_HELD"):  # the caller holds the lock for the whole run
            return
        if GpuLock.held is None:
            f = os.fdopen(os.open(LOCK, os.O_RDONLY | os.O_CREAT, 0o644))
            t = time.time()
            fcntl.flock(f, fcntl.LOCK_EX)
            GpuLock.held = f
            w = time.time() - t
            if w > 5:
                log(f"GPU lock after {w:.0f} s")

    @staticmethod
    def release():
        if GpuLock.held is not None:
            mx.synchronize()
            fcntl.flock(GpuLock.held, fcntl.LOCK_UN)
            GpuLock.held.close()
            GpuLock.held = None


# ---------------------------------------------------------------- remote / local safetensors

class Source:
    """A model's safetensors: tensor name -> (file, absolute offset, nbytes, dtype, shape)."""

    def __init__(self, repo, local=None, conns=16):
        self.local = local
        self.conns = conns
        if local:
            self.repo, self.rev = local, None
            files = [f for f in os.listdir(local) if f.endswith(".safetensors")]
            self.config = json.load(open(os.path.join(local, "config.json")))
        else:
            self.repo, _, self.rev = repo.partition("@")
            self.rev = self.rev or "main"
            meta = json.loads(self._get_url(f"https://huggingface.co/api/models/{self.repo}/revision/{self.rev}"))
            self.rev = meta.get("sha", self.rev)
            files = [s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors")
                     and "/" not in s["rfilename"]]
            self.config = json.loads(self.read("config.json"))
        self.index = {}
        for f in sorted(files):
            n = struct.unpack("<Q", self.read(f, 0, 8))[0]
            hdr = json.loads(self.read(f, 8, 8 + n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                a, b = v["data_offsets"]
                self.index[k] = (f, 8 + n + a, b - a, v["dtype"], v["shape"])

    def _get_url(self, url, rng=None, tries=10):
        h = dict(UA)
        if rng:
            h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
        for t in range(tries):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
                    d = r.read()
                if rng and len(d) != rng[1] - rng[0]:
                    raise IOError("short read")
                return d
            except Exception:
                if t == tries - 1:
                    raise
                time.sleep(2 * (t + 1))

    def read(self, f, a=None, b=None):
        if self.local:
            with open(os.path.join(self.local, f), "rb") as fh:
                if a is None:
                    return fh.read()
                fh.seek(a)
                return fh.read(b - a)
        url = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{f}"
        return self._get_url(url, (a, b) if a is not None else None)

    def fetch(self, names, pool):
        """name -> mx.array (parallel ranged reads, 64 MB pieces)."""
        jobs = []
        bufs = {}
        for k in names:
            f, off, nb, dt, shape = self.index[k]
            bufs[k] = bytearray(nb)
            for p in range(0, nb, 64 << 20):
                jobs.append((k, f, off + p, off + min(nb, p + (64 << 20)), p))

        def work(j):
            k, f, a, b, p = j
            bufs[k][p:p + (b - a)] = self.read(f, a, b)

        list(pool.map(work, jobs))
        # numpy only here: MLX ops recorded on this (prefetch) thread's stream cannot run on the main thread's
        out = {}
        for k in names:
            f, off, nb, dt, shape = self.index[k]
            out[k] = (np.frombuffer(bufs[k], dtype=NP_DT[dt]).reshape(shape), dt)
        return out


def unit_of(name):
    """'embed', ('layer', i), 'head' or None for a tensor name (MLX conversions, any prefix)."""
    if "vision_tower" in name or "visual" in name or name.startswith("mtp."):
        return None
    if ".layers." in name:
        return ("layer", int(name.split(".layers.")[1].split(".")[0]))
    if "embed_tokens" in name:
        return "embed"
    if name.endswith("model.norm.weight") or "lm_head" in name:
        return "head"
    return None


class Prefetch:
    """Downloads the units of several sources in order, at most `budget` bytes ahead of the consumer."""

    def __init__(self, sources, units, budget, conns):
        self.sources, self.units, self.budget = sources, units, budget
        self.pool = ThreadPoolExecutor(conns)
        self.cv = threading.Condition()
        self.ready = {}
        self.used = 0
        self.err = None
        self.bytes = 0
        self.t0 = time.time()
        threading.Thread(target=self._run, daemon=True).start()

    def _size(self, u):
        return sum(s.index[k][2] for s in self.sources for k in s.index if unit_of(k) == u)

    def _run(self):
        try:
            for u in self.units:
                sz = self._size(u)
                with self.cv:
                    while self.used > 0 and self.used + sz > self.budget:
                        self.cv.wait()
                    self.used += sz
                parts = []
                for s in self.sources:
                    names = [k for k in s.index if unit_of(k) == u]
                    parts.append(s.fetch(names, self.pool))
                with self.cv:
                    self.ready[u] = (parts, sz)
                    self.bytes += sz
                    self.cv.notify_all()
        except Exception as e:  # surfaced to the consumer
            with self.cv:
                self.err = e
                self.cv.notify_all()

    def get(self, u):
        with self.cv:
            if u not in self.ready and self.err is None:
                GpuLock.release()  # waiting for the network: let other GPU jobs run
            while u not in self.ready and self.err is None:
                self.cv.wait()
            if self.err is not None:
                raise self.err
            parts, sz = self.ready[u]
        # mx arrays made on the calling (main) thread
        conv = [{k: (mx.array(a).view(mx.bfloat16) if dt == "BF16" else mx.array(a)) for k, (a, dt) in p.items()}
                for p in parts]
        return conv, sz

    def done(self, u):
        with self.cv:
            parts, sz = self.ready.pop(u)
            self.used -= sz
            self.cv.notify_all()

    def rate(self):
        return self.bytes / 1e6 / max(time.time() - self.t0, 1e-3)


# ---------------------------------------------------------------- recipes

ROLES = ("embed", "head", "attn", "gdn", "router", "experts", "shared", "mlp")


def role_of(path):
    """Role of a module path relative to a decoder layer (or 'embed' / 'head')."""
    if path in ("embed", "head"):
        return path
    if path.startswith("linear_attn."):
        return "gdn"
    if path.startswith("self_attn."):
        return "attn"
    if path in ("mlp.gate", "mlp.shared_expert_gate"):
        return "router"
    if path.startswith("mlp.switch_mlp."):
        return "experts"
    if path.startswith("mlp.shared_expert."):
        return "shared"
    if path.startswith("mlp."):
        return "mlp"
    return None


def parse_q(s):
    """'4' / '4:32' / 'nvfp4' -> dict(bits, group_size, mode)"""
    if s in ("mxfp4", "nvfp4", "mxfp8"):
        return {"mode": s, "bits": 8 if s == "mxfp8" else 4, "group_size": 16 if s == "nvfp4" else 32}
    if s in ("bf16", "none"):
        return None
    b, _, g = s.partition(":")
    return {"mode": "affine", "bits": int(b), "group_size": int(g or 64)}


class Recipe:
    def __init__(self, spec, moe):
        parts = spec.split(",")
        self.default = parse_q(parts[0])
        self.over = {}
        if moe:  # mlx-lm's quant_predicate for qwen3_5 MoE: 8-bit router and shared-expert gate
            self.over["router"] = {"mode": "affine", "bits": 8, "group_size": 64}
        for p in parts[1:]:
            r, _, q = p.partition("=")
            if r not in ROLES:
                raise SystemExit(f"unknown role {r} in {spec}")
            self.over[r] = parse_q(q)

    def q(self, path):
        r = role_of(path)
        return self.over.get(r, self.default) if r in self.over else self.default


# ---------------------------------------------------------------- model pieces

def quantize_module(mod, prefix, qfor):
    """nn.quantize over a module tree with per-path settings qfor(path) -> dict or None (keep)."""
    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        q = qfor(path if not prefix else prefix)
        if q is None:
            return False
        return {"group_size": q["group_size"], "bits": q["bits"], "mode": q["mode"]}
    nn.quantize(mod, class_predicate=pred)


def repo_qcfg(cfg, full_prefix):
    """qfor(path) from an MLX repo's config quantization block (per-module entries keyed by full path)."""
    qc = cfg.get("quantization") or cfg.get("quantization_config") or {}
    base = {"bits": qc.get("bits", 4), "group_size": qc.get("group_size", 64), "mode": qc.get("mode", "affine")}

    def qfor(path):
        e = qc.get(full_prefix + path)
        if e is False:
            return None
        if isinstance(e, dict):
            return {"bits": e.get("bits", base["bits"]), "group_size": e.get("group_size", base["group_size"]),
                    "mode": e.get("mode", base["mode"])}
        return base
    return qfor


def strip(tensors, prefix):
    return {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}


def load_layer(targs, i, weights, qfor=None):
    """DecoderLayer i with `weights` (relative names). qfor: the module's quantization (quantized tensors given)."""
    layer = qwen3_5.DecoderLayer(targs, i)
    if qfor is not None:
        names = set(weights)

        def pred(path, m):
            if not hasattr(m, "to_quantized") or f"{path}.scales" not in names:
                return False
            q = qfor(path)
            return {"group_size": q["group_size"], "bits": q["bits"], "mode": q["mode"]}
        nn.quantize(layer, class_predicate=pred)
    layer.load_weights(list(weights.items()), strict=True)
    return layer


def rtn_layer(targs, i, bf16_weights, recipe):
    layer = qwen3_5.DecoderLayer(targs, i)
    layer.load_weights(list(bf16_weights.items()), strict=True)

    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        q = recipe.q(path)
        return False if q is None else {"group_size": q["group_size"], "bits": q["bits"], "mode": q["mode"]}
    nn.quantize(layer, class_predicate=pred)
    return layer


def dequant_f32(m):
    """fp32 weight of an Embedding / Linear, quantized or not."""
    if hasattr(m, "scales"):
        return mx.dequantize(m.weight, m.scales, getattr(m, "biases", None), m.group_size, m.bits,
                             mode=getattr(m, "mode", "affine"), dtype=mx.float32)
    return m.weight.astype(mx.float32)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--local")
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--out")
    ap.add_argument("--dump-ref")
    ap.add_argument("--topk", type=int, default=256)
    ap.add_argument("--budget-gb", type=float, default=6.0)
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=256, help="positions per head chunk")
    ap.add_argument("--max-tokens", type=int, default=0, help="truncate the sequences (tests)")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU (tests; no GPU lock)")
    ap.add_argument("--f32", action="store_true",
                    help="activations, norms and dequantized weights in fp32 (no bf16 rounding noise in either model)")
    a = ap.parse_args()
    if a.cpu:
        mx.set_default_device(mx.cpu)
        os.environ["NI_LOCK_HELD"] = "1"

    seqs = []
    for f in a.ids:
        t = open(f[1:] if f.startswith("@") else f).read()
        seqs.append([int(x) for x in t.replace("\n", ",").split(",") if x.strip()])
    if a.max_tokens:
        seqs = [q[: a.max_tokens] for q in seqs]
    n = len(seqs[0])
    assert all(len(s) == n for s in seqs), "sequences must have one length"
    names = [os.path.basename(f.lstrip("@")).rsplit(".", 1)[0] for f in a.ids]

    t0 = time.time()
    base = Source(a.base, a.local, a.conns)
    cfg = base.config
    tc = dict(cfg.get("text_config", cfg))
    tc.setdefault("model_type", cfg.get("model_type", "qwen3_5"))
    targs = qwen3_5.TextModelArgs.from_dict(tc)
    moe = targs.num_experts > 0
    L = targs.num_hidden_layers if not a.layers else min(a.layers, targs.num_hidden_layers)
    log(f"base {base.repo}@{(base.rev or 'local')[:12]}: {len(base.index)} tensors, {L} layers, "
        f"{'MoE' if moe else 'dense'}, hidden {targs.hidden_size}")

    cands = []  # (name, kind, recipe | Source)
    for c in a.cand:
        name, _, spec = c.partition("=")
        if spec.startswith("rtn:"):
            cands.append((name, "rtn", Recipe(spec[4:], moe)))
        elif spec.startswith("repo:"):
            s = Source(spec[5:], None, a.conns)
            cands.append((name, "repo", s))
            log(f"candidate {name}: {s.repo}@{s.rev[:12]}, {len(s.index)} tensors")
        else:
            raise SystemExit(f"bad candidate {c}")
    sources = [base] + [c[2] for c in cands if c[1] == "repo"]
    units = ["embed"] + [("layer", i) for i in range(L)] + ["head"]
    pf = Prefetch(sources, units, a.budget_gb * 1e9, a.conns)

    def prefix_of(src, u):
        k = next(k for k in src.index if unit_of(k) == u)
        if u == "embed":
            return k[:k.index("embed_tokens")]
        if u == "head":
            return k[:k.index("model.norm")] if "model.norm" in k else k[:k.index("lm_head")]
        return k[:k.index(".layers.") + len(".layers.")] + f"{u[1]}."

    ids = mx.array(seqs)
    hidden = OrderedDict()  # model name -> [B, N, H] bf16 (fp32 with --f32)

    def f32(m):
        """--f32: a module's floating-point parameters (bf16 weights, norms, quantization scales and biases) in fp32;
        quantized codes stay as they are, so a quantized matmul dequantizes exactly (s * q + b in fp32)."""
        if a.f32:
            from mlx.utils import tree_map
            m.update(tree_map(lambda x: x.astype(mx.float32) if x.dtype in (mx.bfloat16, mx.float16) else x,
                              m.parameters()))
        return m

    def act(h):
        return h.astype(mx.float32) if a.f32 else h

    # embeddings
    parts, _ = pf.get("embed")
    GpuLock.take()
    bt = parts[0]
    emb_name = next(k for k in bt if k.endswith("embed_tokens.weight"))
    E = bt[emb_name]
    hidden["bf16"] = act(E[ids])
    embeds = {"bf16": E}
    ri = 1
    for name, kind, obj in cands:
        if kind == "rtn":
            q = obj.q("embed")
            m = nn.Embedding(*E.shape)
            m.weight = E
            if q is not None:
                m = m.to_quantized(group_size=q["group_size"], bits=q["bits"], mode=q["mode"])
        else:
            t = parts[ri]
            ri += 1
            pre = prefix_of(obj, "embed")
            ew = strip(t, pre + "embed_tokens.")
            qfor = repo_qcfg(obj.config, pre)
            m = nn.Embedding(*E.shape)
            if "scales" in ew:
                q = qfor("embed_tokens")
                m = m.to_quantized(group_size=q["group_size"], bits=q["bits"], mode=q["mode"])
            m.load_weights(list(ew.items()))
        embeds[name] = m
        hidden[name] = act(m(ids))
    mx.eval(list(hidden.values()))
    pf.done("embed")
    log(f"embeddings done ({pf.rate():.1f} MB/s)")

    for i in range(L):
        u = ("layer", i)
        parts, sz = pf.get(u)
        GpuLock.take()
        tl = time.time()
        pre = prefix_of(base, u)
        bw = strip(parts[0], pre)
        layer = f32(load_layer(targs, i, bw))
        hidden["bf16"] = layer(hidden["bf16"], mask="causal" if not layer.is_linear else None)
        mx.eval(hidden["bf16"])
        del layer
        ri = 1
        for name, kind, obj in cands:
            if kind == "rtn":
                lyr = f32(rtn_layer(targs, i, bw, obj))
            else:
                t = parts[ri]
                ri += 1
                cp = prefix_of(obj, u)
                lyr = f32(load_layer(targs, i, strip(t, cp), repo_qcfg(obj.config, cp)))
            hidden[name] = lyr(hidden[name], mask="causal" if not lyr.is_linear else None)
            mx.eval(hidden[name])
            del lyr
        del bw, parts
        pf.done(u)
        mx.clear_cache()
        el = time.time() - t0
        log(f"layer {i + 1}/{L} ({'GDN' if (i + 1) % targs.full_attention_interval else 'attention'}) "
            f"compute {time.time() - tl:.1f} s, {pf.rate():.1f} MB/s, {el / 60:.1f} min")

    # head: final norm + LM head per model, fp32 log-probs, exact KL against bf16
    parts, _ = pf.get("head")
    GpuLock.take()
    pre = prefix_of(base, "head")
    bt = parts[0]
    norm_w = bt[pre + "model.norm.weight"]
    tied = targs.tie_word_embeddings or (pre + "lm_head.weight") not in bt
    # each model's fp32 head as a function: rebuilt lazily per chunk, so no dequantized copy stays resident (the 27B's
    # is 5 GB in fp32)
    heads = {}
    if tied:
        heads["bf16"] = (lambda E=embeds["bf16"]: E.astype(mx.float32))
    else:
        heads["bf16"] = (lambda W=bt[pre + "lm_head.weight"]: W.astype(mx.float32))
    norms = {"bf16": norm_w}
    ri = 1
    for name, kind, obj in cands:
        if kind == "rtn":
            if tied:
                heads[name] = (lambda m=embeds[name]: dequant_f32(m))
            else:
                q = obj.q("head")
                hw = bt[pre + "lm_head.weight"]
                if q is None:
                    heads[name] = (lambda W=hw: W.astype(mx.float32))
                else:
                    qq = mx.quantize(hw, q["group_size"], q["bits"], mode=q["mode"])
                    wq, sc, bi = (tuple(qq) + (None,))[:3]
                    mx.eval(wq, sc, *( [bi] if bi is not None else []))
                    heads[name] = (lambda wq=wq, sc=sc, bi=bi, q=q: mx.dequantize(
                        wq, sc, bi, q["group_size"], q["bits"], mode=q["mode"], dtype=mx.float32))
            norms[name] = norm_w
        else:
            t = parts[ri]
            ri += 1
            cp = prefix_of(obj, "head")
            norms[name] = t[cp + "model.norm.weight"]
            if tied:
                heads[name] = (lambda m=embeds[name]: dequant_f32(m))
            else:
                hw = strip(t, cp + "lm_head.")
                if "scales" in hw:
                    q = repo_qcfg(obj.config, cp)("lm_head")
                    heads[name] = (lambda hw=hw, q=q: mx.dequantize(hw["weight"], hw["scales"], hw.get("biases"),
                                                                     q["group_size"], q["bits"], mode=q["mode"],
                                                                     dtype=mx.float32))
                else:
                    heads[name] = (lambda W=hw["weight"]: W.astype(mx.float32))
    B = len(seqs)
    res = {name: {"kl": [], "agree": 0, "nll": 0.0, "n": 0} for name in hidden}
    dump = open(a.dump_ref, "wb") if a.dump_ref else None
    if dump:
        dump.write(b"NIEVREF2" + struct.pack("<3I", B, targs.vocab_size, a.topk))
    eps = targs.rms_norm_eps
    for b in range(B):
        seq = seqs[b]
        tgt = mx.array(seq[1:])
        hn = {name: mx.fast.rms_norm(h[b, : n - 1].astype(mx.float32) if a.f32 else h[b, : n - 1],
                                     norms[name].astype(mx.float32) if a.f32 else norms[name], eps).astype(mx.float32)
              for name, h in hidden.items()}
        top_rows = []
        for c0 in range(0, n - 1, a.chunk):
            c1 = min(n - 1, c0 + a.chunk)
            lr = hn["bf16"][c0:c1] @ heads["bf16"]().T
            lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
            pr = mx.exp(lr)
            ar = mx.argmax(lr, -1)
            t = tgt[c0:c1]
            res["bf16"]["nll"] -= float(mx.sum(mx.take_along_axis(lr, t[:, None], -1)).item())
            res["bf16"]["n"] += c1 - c0
            if dump:
                ti = mx.argsort(-lr, axis=-1)[:, : a.topk]
                tv = mx.take_along_axis(lr, ti, -1)
                tl = mx.take_along_axis(lr, t[:, None], -1)[:, 0]
                mx.eval(ti, tv, tl)
                top_rows.append((np.array(tl, dtype=np.float32), np.array(ti).astype(np.uint32), np.array(tv, dtype=np.float32)))
            for name in hidden:
                if name == "bf16":
                    continue
                lq = hn[name][c0:c1] @ heads[name]().T
                lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
                kl = mx.maximum(mx.sum(pr * (lr - lq), -1), 0)
                r = res[name]
                r["kl"] += np.array(kl).tolist()
                r["agree"] += int(mx.sum(mx.argmax(lq, -1) == ar).item())
                r["nll"] -= float(mx.sum(mx.take_along_axis(lq, t[:, None], -1)).item())
                r["n"] += c1 - c0
        if dump:
            dump.write(struct.pack("<I", n) + np.array(seq, dtype=np.int32).tobytes())
            for tl, ti, tv in top_rows:
                for p in range(len(tl)):
                    dump.write(tl[p:p + 1].tobytes() + np.stack([ti[p], tv[p].view(np.uint32)], 1).astype(np.uint32).tobytes())
        log(f"scored {names[b]}")
    GpuLock.release()
    if dump:
        dump.close()
    out = {"base": f"{base.repo}@{base.rev}", "layers": L, "sequences": names, "positions": res["bf16"]["n"],
           "elapsed_min": round((time.time() - t0) / 60, 1), "candidates": {}}
    for name, r in res.items():
        k = np.array(r["kl"]) if r["kl"] else np.zeros(1)
        e = {"ppl": math.exp(r["nll"] / max(r["n"], 1))}
        if name != "bf16":
            e.update({"kl_mean": float(k.mean()), "kl_median": float(np.median(k)), "kl_p95": float(np.percentile(k, 95)),
                      "kl_p99": float(np.percentile(k, 99)), "kl_max": float(k.max()), "top1_agree": r["agree"] / r["n"]})
            spec = next(c for c in a.cand if c.startswith(name + "="))
            e["spec"] = spec.partition("=")[2]
        out["candidates"][name] = e
        log(name, json.dumps(e))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
