#!/usr/bin/env python3
"""Quantization quality of a Qwen3.5 / 3.6 / 3.8 model too large for this Mac in BF16, against the true BF16 model.

The BF16 checkpoint (an MLX conversion, e.g. mlx-community/Qwen3.6-35B-A3B-bf16) never touches the disk: its
tensors stream from Hugging Face into RAM one decoder layer at a time (HTTP range requests), and the held-out
sequences run through each layer in BF16 with mlx-lm's
own modules. Every candidate quantization runs through the same layer beside it, on its own hidden states:
  rtn:<spec>   round-to-nearest from the BF16 layer with mx.quantize (what mlx_lm.convert does), recipe below
  repo:<id>    the candidate's own published tensors, streamed the same way (DWQ, mixed recipes, ...); a local
               folder works too (repo:/path/to/pack)
At the end the reference's final norm and LM head give fp32 log-probs; each candidate's (its own head) are compared
position by position: exact KL(BF16 || candidate) over the whole vocabulary, top-1 agreement, perplexity. The BF16
run's top-K log-probs can be written as a reference file for Kapitsa's tools/ni_eval --ref.

  eval_stream.py --base <repo>[@rev] --cand NAME=SPEC ... [--cands FILE] --ids @f1 @f2 ... [--out results.json]
                 [--aside NAME,...] [--positions out.npz] [--dump-ref ref.bin] [--dump-ref-aside ref.bin]
                 [--topk 256] [--budget-gb 6] [--conns 16] [--layers N] [--local DIR]

rtn spec: <default>[,<sel>=<q>]...: the default is <bits>[:<group>] (bits 1-6 or 8, group 32, 64 or 128), an MLX
mode (mxfp4, nvfp4, mxfp8) or bf16 (unquantized), then overrides, applied left to right (a later match wins). <sel>
is a role, L<a>[-<b>] (every linear of decoder layers a..b) or L<a>[-<b>].<role>. Roles: embed, head, attn, gdn,
mix (attn or gdn: the layer's token mixer), mlp (dense MLP), down (down_proj), gu (gate_proj and up_proj), router,
experts, shared. An override that selects nothing in the model is an error. MLX's own default for quantized MoE
models (8-bit router and shared-expert gate) applies unless overridden. Examples:
rtn:4,attn=8,gdn=8,embed=8,head=8 (8-bit except the experts); rtn:4,L56-63=8 (layers 56-63 8-bit);
rtn:4,L5.mlp=8 (the KL that layer 5's MLP at 8-bit removes from the 4-bit model).

Candidates that agree on the embedding and on every layer so far share one hidden state, and a layer is quantized
once per distinct recipe: forks such as rtn:4,L<i>.mix=8 for every i cost one pass from layer i on, not a model
each. --cands reads NAME=SPEC lines. --aside names sequences kept out of the headline numbers (reported per
sequence only); --dump-ref writes the headline sequences' reference, --dump-ref-aside the others'. --positions
writes every candidate's per-position KL, top-1 agreement and NLL. Each candidate's decode bytes are counted as
Kapitsa reads them: every linear of the layers and the LM head (the embedding only as a tied head), plus the
DFlash2 draft head (98,304 rows of the LM head, an 8-bit copy unless the head is 4- or 8-bit group 64).
--local DIR reads the base tensors from a local MLX directory instead (tests). --layers N stops after N layers
(debug: the head then runs on partial hidden states). GPU work holds /tmp/kapitsa-gpu.lock, released while waiting
for downloads (NI_LOCK_HELD=1: the caller holds it for the whole run). --f32 sets MLX_ENABLE_TF32=0 (MLX otherwise
runs fp32 GEMMs as TF32 on the M5).
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
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor

if "--f32" in sys.argv:  # before MLX starts: fp32 GEMMs in fp32, not TF32
    os.environ.setdefault("MLX_ENABLE_TF32", "0")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import qwen3_5

UA = {"User-Agent": "kapitsa-quant-eval"}
LOCK = os.path.expanduser("/tmp/kapitsa-gpu.lock")
NP_DT = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "U32": np.uint32, "I32": np.int32, "U8": np.uint8}
DRAFT_HEAD_ROWS = 98304  # DFlash2::kHeadRowsIdOrder (src/model/dflash2.h): the draft head without a ranked id list


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
                if k in self.index:
                    raise SystemExit(f"{self.repo}: tensor {k} is in both {self.index[k][0]} and {f}")
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

ROLES = ("embed", "head", "attn", "gdn", "mix", "mlp", "down", "gu", "router", "experts", "shared", "all")
AFFINE_BITS = (1, 2, 3, 4, 5, 6, 8)  # what mx.quantize takes (MLX 0.32)
GROUPS = (32, 64, 128)
Q8 = {"mode": "affine", "bits": 8, "group_size": 64}


def roles_of(path):
    """Roles of a module path relative to a decoder layer (or 'embed' / 'head')."""
    if path in ("embed", "head"):
        return {path}
    r = set()
    if path.startswith("linear_attn."):
        r |= {"gdn", "mix"}
    elif path.startswith("self_attn."):
        r |= {"attn", "mix"}
    elif path in ("mlp.gate", "mlp.shared_expert_gate"):
        r.add("router")
    elif path.startswith("mlp.switch_mlp."):
        r.add("experts")
    elif path.startswith("mlp.shared_expert."):
        r.add("shared")
    elif path.startswith("mlp."):
        r.add("mlp")
    if path.endswith("down_proj"):
        r.add("down")
    elif path.endswith(("gate_proj", "up_proj")):
        r.add("gu")
    return r


def parse_q(s):
    """'4' / '4:32' / 'nvfp4' -> dict(bits, group_size, mode); 'bf16' -> None"""
    if s in ("mxfp4", "nvfp4", "mxfp8"):
        return {"mode": s, "bits": 8 if s == "mxfp8" else 4, "group_size": 16 if s == "nvfp4" else 32}
    if s in ("bf16", "none"):
        return None
    b, _, g = s.partition(":")
    try:
        bits, gs = int(b), int(g or 64)
    except ValueError:
        raise SystemExit(f"bad quantization {s!r}: <bits>[:<group>], mxfp4, nvfp4, mxfp8 or bf16")
    if bits not in AFFINE_BITS or gs not in GROUPS:
        raise SystemExit(f"bad quantization {s!r}: bits 1-6 or 8, group 32, 64 or 128")
    return {"mode": "affine", "bits": bits, "group_size": gs}


def qkey(q):
    return None if q is None else (q["bits"], q["group_size"], q["mode"])


class Recipe:
    def __init__(self, spec, moe):
        parts = spec.split(",")
        self.spec = spec
        self.default = parse_q(parts[0])
        self.rules = []  # (layer range or None, role, q, text): applied in order, a later match wins
        if moe and self.default is not None:  # mlx-lm's quant_predicate for qwen3_5 MoE: 8-bit router / shared gate
            self.rules.append((None, "router", Q8, None))
        for p in parts[1:]:
            sel, eq, q = p.partition("=")
            layers, role = None, sel
            if sel[:1] == "L" and sel[1:2].isdigit():
                rng, _, role = sel[1:].partition(".")
                a, _, b = rng.partition("-")
                if not a.isdigit() or (b and not b.isdigit()):
                    raise SystemExit(f"bad layer range in {p} ({spec})")
                layers = (int(a), int(b or a))
                if layers[0] > layers[1]:
                    raise SystemExit(f"bad layer range in {p} ({spec}): {layers[0]} > {layers[1]}")
                role = role or "all"
            if not eq or role not in ROLES or (role == "all" and layers is None) or (layers and role in ("embed", "head")):
                raise SystemExit(f"bad override {p} in {spec}")
            self.rules.append((layers, role, parse_q(q), p))

    def q(self, path, layer=None):
        """quantization of 'embed', 'head' or a module path of decoder layer `layer` (None: bf16)"""
        rs = roles_of(path)
        out = self.default
        for layers, role, q, _ in self.rules:
            if layers is not None and (layer is None or not layers[0] <= layer <= layers[1]):
                continue
            if role in rs or (role == "all" and layer is not None):
                out = q
        return out

    def check(self, n_layers, paths_of, tied):
        """Every override must select something in this model: a layer range inside it, a role it has there."""
        for layers, role, q, text in self.rules:
            if text is None:
                continue
            if role == "head" and tied:
                raise SystemExit(f"{text} in {self.spec}: this model's head is its embedding (set embed)")
            if role in ("embed", "head"):
                continue
            lo, hi = layers if layers is not None else (0, n_layers - 1)
            if hi >= n_layers:
                raise SystemExit(f"{text} in {self.spec}: the model has {n_layers} layers")
            if not any(role == "all" or role in roles_of(p) for i in range(lo, hi + 1) for p in paths_of(i)):
                raise SystemExit(f"{text} in {self.spec} selects nothing in this model")


# ---------------------------------------------------------------- model pieces

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
    layer.eval()  # inference mode: mlx-lm's DeltaNet takes its Metal kernel only outside training
    return layer


def rtn_layer(targs, i, bf16_weights, plan):
    """DecoderLayer i from its BF16 weights, each linear quantized as `plan` ({path: qkey or None}) says."""
    layer = qwen3_5.DecoderLayer(targs, i)
    layer.load_weights(list(bf16_weights.items()), strict=True)

    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        k = plan.get(path)
        return False if k is None else {"bits": k[0], "group_size": k[1], "mode": k[2]}
    nn.quantize(layer, class_predicate=pred)
    layer.eval()  # inference mode: mlx-lm's DeltaNet takes its Metal kernel only outside training
    return layer


def linear_paths(targs, i):
    """Paths of the quantizable modules of decoder layer i (built lazily: nothing is allocated)."""
    return sorted(p for p, m in qwen3_5.DecoderLayer(targs, i).named_modules() if hasattr(m, "to_quantized"))


def qbytes(params, q):
    """Bytes of `params` weights stored as q (None: bf16)."""
    if q is None:
        return params * 2
    if q["mode"] == "affine":
        return params * q["bits"] / 8 + params / q["group_size"] * 4  # bf16 scale and bias per group
    return params * q["bits"] / 8 + params / q["group_size"]  # one 8-bit scale per group


class Head:
    """An LM head (or a tied embedding) as fp32 logits of fp32 rows: x @ W.T, quantized or not (no dequantized copy
    of a quantized head is kept; affine scales and biases in fp32, so s * q + b is exact)."""

    def __init__(self, w, scales=None, biases=None, q=None):
        if scales is None:
            self.w = w.astype(mx.float32)
            mx.eval(self.w)
            self.q = None
        else:
            affine = q["mode"] == "affine"
            self.w, self.q = w, q
            self.s = scales.astype(mx.float32) if affine else scales
            self.b = biases.astype(mx.float32) if affine and biases is not None else biases
            mx.eval(self.w, self.s, *([self.b] if self.b is not None else []))

    def __call__(self, x):
        if self.q is None:
            return x @ self.w.T
        return mx.quantized_matmul(x, self.w, self.s, self.b, transpose=True, group_size=self.q["group_size"],
                                   bits=self.q["bits"], mode=self.q["mode"])


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--local")
    ap.add_argument("--cand", action="append", default=[])
    ap.add_argument("--cands", help="file of NAME=SPEC lines (# comments)")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--aside", default="", help="sequences left out of the headline numbers (comma list)")
    ap.add_argument("--out")
    ap.add_argument("--positions", help="npz of per-position KL / top-1 / NLL per candidate")
    ap.add_argument("--dump-ref", help="reference file (ni_eval --ref) of the headline sequences")
    ap.add_argument("--dump-ref-aside", help="reference file of the --aside sequences")
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
    cand_specs = list(a.cand)
    if a.cands:
        cand_specs += [l.strip() for l in open(a.cands) if l.strip() and not l.lstrip().startswith("#")]

    seqs = []
    for f in a.ids:
        t = open(f[1:] if f.startswith("@") else f).read()
        seqs.append([int(x) for x in t.replace("\n", ",").split(",") if x.strip()])
    if a.max_tokens:
        seqs = [q[: a.max_tokens] for q in seqs]
    n = len(seqs[0])
    assert all(len(s) == n for s in seqs), "sequences must have one length"
    names = [os.path.basename(f.lstrip("@")).rsplit(".", 1)[0] for f in a.ids]
    aside = set(x for x in a.aside.split(",") if x)
    if aside - set(names):
        raise SystemExit(f"--aside {','.join(sorted(aside - set(names)))}: not among the sequences {names}")
    main_rows = [b for b, nm in enumerate(names) if nm not in aside]
    aside_rows = [b for b, nm in enumerate(names) if nm in aside]
    if a.dump_ref_aside and not aside_rows:
        raise SystemExit("--dump-ref-aside without --aside sequences")

    t0 = time.time()
    base = Source(a.base, a.local, a.conns)
    cfg = base.config
    tc = dict(cfg.get("text_config", cfg))
    tc.setdefault("model_type", cfg.get("model_type", "qwen3_5"))
    targs = qwen3_5.TextModelArgs.from_dict(tc)
    moe = targs.num_experts > 0
    L = targs.num_hidden_layers if not a.layers else min(a.layers, targs.num_hidden_layers)
    tied = targs.tie_word_embeddings or not any("lm_head" in k for k in base.index)
    log(f"base {base.repo}@{(base.rev or 'local')[:12]}: {len(base.index)} tensors, {L} layers, "
        f"{'MoE' if moe else 'dense'}, hidden {targs.hidden_size}{', tied head' if tied else ''}")
    paths = {}  # layer kind (is_linear) -> quantizable module paths

    def paths_of(i):
        lin = (i + 1) % targs.full_attention_interval != 0
        if lin not in paths:
            paths[lin] = linear_paths(targs, i)
        return paths[lin]

    # candidates: (name, kind, recipe | Source, spec); the reference is the BF16 recipe
    cands = [("bf16", "rtn", Recipe("bf16", False), "bf16")]
    for c in cand_specs:
        name, _, spec = c.partition("=")
        if name == "bf16" or any(name == x[0] for x in cands):
            raise SystemExit(f"duplicate candidate name {name}")
        if spec.startswith("rtn:"):
            r = Recipe(spec[4:], moe)
            r.check(targs.num_hidden_layers, paths_of, tied)
            cands.append((name, "rtn", r, spec))
        elif spec.startswith("repo:"):
            loc = os.path.expanduser(spec[5:])
            s = Source(None, loc, a.conns) if os.path.isdir(loc) else Source(spec[5:], None, a.conns)
            cands.append((name, "repo", s, spec))
            log(f"candidate {name}: {s.repo}@{(s.rev or 'local')[:12]}, {len(s.index)} tensors")
        else:
            raise SystemExit(f"bad candidate {c}")
    repos = [c for c in cands if c[1] == "repo"]
    sources = [base] + [c[2] for c in repos]
    part_of = {c[0]: 1 + j for j, c in enumerate(repos)}  # index of a repo candidate's tensors in a unit's parts
    units = ["embed"] + [("layer", i) for i in range(L)] + ["head"]

    # decode bytes as Kapitsa reads them per forward pass, and the DFlash2 draft head
    K = targs.hidden_size
    lin_params = {}  # (layer, path) -> params, from the base's bf16 shapes
    for k, (f, off, nb, dt, shape) in base.index.items():
        u = unit_of(k)
        if isinstance(u, tuple) and u[1] < L and k.endswith(".weight") and len(shape) >= 2:
            path = k.split(".layers.")[1].split(".", 1)[1][: -len(".weight")]
            if path in paths_of(u[1]):
                lin_params[(u[1], path)] = int(np.prod(shape))
    head_params = targs.vocab_size * K

    def draft_head_bytes(q):
        kq = q if q is not None and q["mode"] == "affine" and q["group_size"] == 64 and q["bits"] in (4, 8) else Q8
        return qbytes(min(DRAFT_HEAD_ROWS, targs.vocab_size) * K, kq)

    def decode_bytes(name, kind, obj):
        if kind == "rtn":
            layers = sum(qbytes(p, obj.q(path, i)) for (i, path), p in lin_params.items())
            hq = obj.q("embed" if tied else "head")
            head = qbytes(head_params, hq)
            emb = qbytes(head_params, obj.q("embed"))
        else:
            qfor = repo_qcfg(obj.config, "")
            layers = head = emb = 0
            for k, (f, off, nb, dt, shape) in obj.index.items():
                u = unit_of(k)
                if isinstance(u, tuple) and u[1] < L and k.rsplit(".", 1)[1] in ("weight", "scales", "biases"):
                    path = k.split(".layers.")[1].split(".", 1)[1].rsplit(".", 1)[0]
                    if path in paths_of(u[1]):
                        layers += nb
                elif u == "embed":
                    emb += nb
                elif u == "head" and "lm_head" in k:
                    head += nb
            if tied:
                head = emb
            hname = next((k[: -len(".scales")] for k in obj.index if k.endswith(".scales") and
                          (("embed_tokens" in k) if tied else ("lm_head" in k))), None)
            hq = qfor(hname) if hname else None
        dh = draft_head_bytes(hq)
        return {"decode": layers + head, "draft_head": dh, "dflash2_cycle": layers + head + dh,
                "weights": layers + head + (0 if tied else emb)}

    pf = Prefetch(sources, units, a.budget_gb * 1e9, a.conns)

    def prefix_of(src, u):
        k = next(k for k in src.index if unit_of(k) == u)
        if u == "embed":
            return k[:k.index("embed_tokens")]
        if u == "head":
            return k[:k.index("model.norm")] if "model.norm" in k else k[:k.index("lm_head")]
        return k[:k.index(".layers.") + len(".layers.")] + f"{u[1]}."

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

    # Shared hidden states: a stream is named by the recipe that produced it (embedding, then each layer's plan);
    # candidates on the same stream share it. cur[name] = the candidate's stream id.
    sig_ids = {}

    def sig(key):
        return sig_ids.setdefault(key, len(sig_ids))

    hidden = {}
    cur = {}
    ids = mx.array(seqs)

    # embeddings
    parts, _ = pf.get("embed")
    GpuLock.take()
    bt = parts[0]
    emb_name = next(k for k in bt if k.endswith("embed_tokens.weight"))
    E = bt[emb_name]
    embeds = {}  # qkey or ('repo', name) -> Embedding module (kept only as the head of tied models)
    for name, kind, obj, _ in cands:
        if kind == "rtn":
            key = qkey(obj.q("embed"))
            if key not in embeds:
                m = nn.Embedding(*E.shape)
                m.weight = E
                if key is not None:
                    m = m.to_quantized(group_size=key[1], bits=key[0], mode=key[2])
                embeds[key] = m
        else:
            key = ("repo", name)
            pre = prefix_of(obj, "embed")
            ew = strip(parts[part_of[name]], pre + "embed_tokens.")
            m = nn.Embedding(*E.shape)
            if "scales" in ew:
                q = repo_qcfg(obj.config, pre)("embed_tokens")
                m = m.to_quantized(group_size=q["group_size"], bits=q["bits"], mode=q["mode"])
            m.load_weights(list(ew.items()))
            embeds[key] = m
        s = sig(("embed", key))
        cur[name] = s
        if s not in hidden:
            m = embeds[key]
            if a.f32 and hasattr(m, "scales") and getattr(m, "mode", "affine") == "affine":
                # the rows dequantized in fp32: MLX computes s * q + b in the scales' dtype, so bf16 scales round it
                bi = getattr(m, "biases", None)
                hidden[s] = mx.dequantize(m.weight[ids], m.scales[ids].astype(mx.float32),
                                          bi[ids].astype(mx.float32) if bi is not None else None,
                                          m.group_size, m.bits, mode="affine", dtype=mx.float32)
            else:
                hidden[s] = act(m(ids))
            mx.eval(hidden[s])
    pf.done("embed")
    if not tied:  # the embeddings are not needed again
        embeds.clear()
    del E, bt, parts
    mx.clear_cache()
    log(f"embeddings done: {len(hidden)} streams ({pf.rate():.1f} MB/s)")

    for i in range(L):
        u = ("layer", i)
        parts, sz = pf.get(u)
        GpuLock.take()
        tl = time.time()
        pre = prefix_of(base, u)
        bw = strip(parts[0], pre)
        lin = (i + 1) % targs.full_attention_interval != 0
        # each candidate's plan for this layer; streams that need it
        plans = OrderedDict()  # plan key -> {old stream: new stream}
        nxt = {}
        for name, kind, obj, _ in cands:
            if kind == "rtn":
                pk = ("rtn",) + tuple((p, qkey(obj.q(p, i))) for p in paths_of(i))
            else:
                pk = ("repo", name)
            s = sig((cur[name], pk))
            plans.setdefault(pk, {})[cur[name]] = s
            nxt[name] = s
        uses = Counter(old for m in plans.values() for old in m)
        for pk, m in plans.items():
            if pk[0] == "rtn":
                lyr = f32(rtn_layer(targs, i, bw, dict(pk[1:])))
            else:
                name = pk[1]
                obj = next(c[2] for c in cands if c[0] == name)
                cp = prefix_of(obj, u)
                lyr = f32(load_layer(targs, i, strip(parts[part_of[name]], cp), repo_qcfg(obj.config, cp)))
            for old, new in m.items():
                hidden[new] = lyr(hidden[old], mask="causal" if not lyr.is_linear else None)
                mx.eval(hidden[new])
                uses[old] -= 1
                if uses[old] == 0:
                    del hidden[old]
            del lyr
            mx.clear_cache()
        cur = nxt
        del bw, parts
        pf.done(u)
        mx.clear_cache()
        el = time.time() - t0
        log(f"layer {i + 1}/{L} ({'GDN' if lin else 'attention'}): {len(plans)} recipes, {len(hidden)} streams, "
            f"compute {time.time() - tl:.1f} s, {pf.rate():.1f} MB/s, active {mx.get_active_memory() / 1e9:.1f} GB "
            f"(peak {mx.get_peak_memory() / 1e9:.1f}), {el / 60:.1f} min")

    # head: final norm + LM head per model, fp32 log-probs, exact KL against bf16. One head besides the reference's
    # is resident at a time (head-major order); a stream is freed after its last head.
    def tied_maker(m):
        return lambda: Head(m.weight, getattr(m, "scales", None), getattr(m, "biases", None),
                            {"bits": m.bits, "group_size": m.group_size, "mode": m.mode} if hasattr(m, "scales") else None)

    def rtn_maker(hw, hk):
        if hk is None:
            return lambda: Head(hw)

        def mk():
            q = {"bits": hk[0], "group_size": hk[1], "mode": hk[2]}
            qq = tuple(mx.quantize(hw, q["group_size"], q["bits"], mode=q["mode"])) + (None,)
            return Head(qq[0], qq[1], qq[2], q)
        return mk

    def head_makers():
        """head key -> () -> Head, head key -> final norm, candidate -> (stream, head key). The makers hold the source
        tensors they need; nothing else does once this returns and pf.done("head") has run."""
        parts, _ = pf.get("head")
        pre = prefix_of(base, "head")
        bt = parts[0]
        norm_w = bt[pre + "model.norm.weight"]
        hw = None if tied else bt[pre + "lm_head.weight"]
        makers, norms, pair_of = {}, {}, {}
        for name, kind, obj, _ in cands:
            if kind == "rtn":
                hk = ("tied", qkey(obj.q("embed"))) if tied else qkey(obj.q("head"))
                if hk not in makers:
                    makers[hk] = tied_maker(embeds[hk[1]]) if tied else rtn_maker(hw, hk)
                    norms[hk] = norm_w
            else:
                hk = ("repo", name)
                rt = parts[part_of[name]]
                cp = prefix_of(obj, "head")
                norms[hk] = rt[cp + "model.norm.weight"]
                if tied:
                    makers[hk] = tied_maker(embeds[hk])
                else:
                    hwr = strip(rt, cp + "lm_head.")
                    if "scales" in hwr:
                        q = repo_qcfg(obj.config, cp)("lm_head")
                        makers[hk] = (lambda hwr=hwr, q=q: Head(hwr["weight"], hwr["scales"], hwr.get("biases"), q))
                    else:
                        makers[hk] = (lambda hwr=hwr: Head(hwr["weight"]))
            pair_of[name] = (cur[name], hk)
        return makers, norms, pair_of

    makers, norms, pair_of = head_makers()
    pf.done("head")
    GpuLock.take()
    ref_pair = pair_of["bf16"]
    pairs = list(OrderedDict.fromkeys(pair_of.values()))
    groups = OrderedDict([(ref_pair[1], [])])
    for p in pairs:
        groups.setdefault(p[1], []).append(p)
    left = Counter(p[0] for p in pairs)  # stream -> heads still to score it
    B = len(seqs)
    P = n - 1
    res = {p: {"kl": np.zeros((B, P), np.float32), "agree": np.zeros((B, P), bool), "nll": np.zeros((B, P), np.float32)}
           for p in pairs}
    eps = targs.rms_norm_eps
    log(f"head: {len(pairs)} distinct (stream, head) pairs, {len(groups)} heads")

    def normed(p, b, c0, c1):
        h = hidden[p[0]][b, c0:c1]
        w = norms[p[1]]
        if a.f32:
            return mx.fast.rms_norm(h.astype(mx.float32), w.astype(mx.float32), eps)
        return mx.fast.rms_norm(h, w, eps).astype(mx.float32)

    dumps = {}  # row -> file
    for path, rows in ((a.dump_ref, main_rows), (a.dump_ref_aside, aside_rows)):
        if path:
            fh = open(path, "wb")
            fh.write(b"NIEVREF2" + struct.pack("<3I", len(rows), targs.vocab_size, a.topk))
            for b in rows:
                dumps[b] = fh
    th = time.time()
    ref_head = makers[ref_pair[1]]()
    for gi, (hk, gpairs) in enumerate(groups.items()):
        head = ref_head if hk == ref_pair[1] else makers[hk]()
        for b in range(B):
            seq = seqs[b]
            tgt = mx.array(seq[1:])
            top_rows = []
            for c0 in range(0, P, a.chunk):
                c1 = min(P, c0 + a.chunk)
                lr = ref_head(normed(ref_pair, b, c0, c1))
                lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
                pr = mx.exp(lr)
                ar = mx.argmax(lr, -1)
                t = tgt[c0:c1]
                if gi == 0:  # the reference itself
                    r = res[ref_pair]
                    r["nll"][b, c0:c1] = -np.array(mx.take_along_axis(lr, t[:, None], -1)[:, 0])
                    r["agree"][b, c0:c1] = True
                    if b in dumps:
                        ti = mx.argsort(-lr, axis=-1)[:, : a.topk]
                        tv = mx.take_along_axis(lr, ti, -1)
                        tl = mx.take_along_axis(lr, t[:, None], -1)[:, 0]
                        mx.eval(ti, tv, tl)
                        top_rows.append((np.array(tl, dtype=np.float32), np.array(ti).astype(np.uint32),
                                         np.array(tv, dtype=np.float32)))
                for p in gpairs:
                    if p == ref_pair:
                        continue
                    lq = head(normed(p, b, c0, c1))
                    lq = lq - mx.logsumexp(lq, axis=-1, keepdims=True)
                    kl = mx.maximum(mx.sum(pr * (lr - lq), -1), 0)
                    ag = mx.argmax(lq, -1) == ar
                    nl = -mx.take_along_axis(lq, t[:, None], -1)[:, 0]
                    mx.eval(kl, ag, nl)
                    r = res[p]
                    r["kl"][b, c0:c1] = np.array(kl)
                    r["agree"][b, c0:c1] = np.array(ag)
                    r["nll"][b, c0:c1] = np.array(nl)
            if top_rows:
                fh = dumps[b]
                fh.write(struct.pack("<I", n) + np.array(seq, dtype=np.int32).tobytes())
                for tl, ti, tv in top_rows:
                    for q in range(len(tl)):
                        fh.write(tl[q:q + 1].tobytes() + np.stack([ti[q], tv[q].view(np.uint32)], 1).astype(np.uint32).tobytes())
        for p in gpairs:
            left[p[0]] -= 1
            if left[p[0]] == 0 and p[0] != ref_pair[0]:
                del hidden[p[0]]
        if head is not ref_head:
            del head
        makers[hk] = None
        mx.clear_cache()
        log(f"head {gi + 1}/{len(groups)} {hk}: {len(gpairs)} pairs, {len(hidden)} streams left ({time.time() - th:.0f} s)")
    GpuLock.release()
    for fh in set(dumps.values()):
        fh.close()

    def stats(r, rows, ref):
        if not rows:
            return None
        e = {"ppl": math.exp(float(r["nll"][rows].mean()))}
        if not ref:
            k = r["kl"][rows].ravel()
            e.update({"kl_mean": float(k.mean()), "kl_median": float(np.median(k)),
                      "kl_p95": float(np.percentile(k, 95)), "kl_p99": float(np.percentile(k, 99)),
                      "kl_max": float(k.max()), "top1_agree": float(r["agree"][rows].mean())})
        return e

    out = {"base": f"{base.repo}@{base.rev}", "layers": L, "sequences": [names[b] for b in main_rows],
           "aside": [names[b] for b in aside_rows], "positions": len(main_rows) * P,
           "elapsed_min": round((time.time() - t0) / 60, 1), "candidates": {}}
    for name, kind, obj, spec in cands:
        r = res[pair_of[name]]
        e = stats(r, main_rows, name == "bf16") or {}
        e["spec"] = spec
        e["bytes"] = decode_bytes(name, kind, obj)
        e["per_sequence"] = {names[b]: stats(r, [b], name == "bf16") for b in range(B)}
        out["candidates"][name] = e
        log(name, json.dumps({k: v for k, v in e.items() if k != "per_sequence"}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    if a.positions:
        arr = {}
        for name, *_ in cands:
            r = res[pair_of[name]]
            arr[f"{name}/kl"], arr[f"{name}/agree"], arr[f"{name}/nll"] = r["kl"], r["agree"], r["nll"]
        np.savez_compressed(a.positions, sequences=np.array(names), **arr)


if __name__ == "__main__":
    main()
