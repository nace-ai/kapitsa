#!/usr/bin/env python3
"""Converts DiffuSpace's Clover-35B-A3B-Preview (Hugging Face, bf16, 69 GB) into a Kapitsa pack (src/model/clover.h).

  convert_clover.py SRC OUT [--bits 4] [--head-bits 4] [--threads 0]

SRC is a local copy of https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview (config.json, the model-*.safetensors
shards, bd_gdn_conv_right_weight.safetensors, bd_gdn_softmax_scalars.safetensors, the tokenizer files). OUT gets the
MLX layout Kapitsa reads for the Qwen3.6-35B-A3B packs, under the same names (language_model.model.*):
  - every linear MLX affine `bits`-bit group 64 (the MoE router and the shared-expert gate 8-bit, as the 35B packs;
    the LM head --head-bits), with Kapitsa's re-quantizer (src/gguf/quant.cpp through tools/mlx_quant.cpp), not RTN;
  - the fused experts.gate_up_proj [E, 2I, H] split into switch_mlp.gate_proj and up_proj [E, I, H] (gate first, as
    the reference's chunk(2, -1)), experts.down_proj into switch_mlp.down_proj;
  - everything else as stored (bf16): norms (zero-centred: conv1d stays [C, 1, 4], which tells the loader so), A_log,
    dt_bias, conv1d, and Clover's own conv_right_weight [C, 1, 3] and bd_softmax_* [Hv] of each DeltaNet layer.
config.json is the source's (model_type clover_moe) with MLX's quantization block; the tokenizer, chat template,
generation_config.json (the generation controls) and the licence files are copied. Shards of about 5 GB, an index,
and conversion.json (the source's revision if SRC has one, the recipe, every file's size and sha256).
"""
import argparse
import ctypes
import hashlib
import json
import mmap
import os
import shutil
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
PREFIX = "language_model."
COPY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "LICENSE",
        "ATTRIBUTIONS.md", "README.md"]
DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1}
LINEARS = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj",
           "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj", "mlp.gate",
           "shared_expert_gate", "embed_tokens", "lm_head")
SHARD = 5 << 30


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lib():
    """tools/mlx_quant.cpp built once into the work folder (as tools/qwen4_exp_pack.py does)."""
    out = os.path.join(os.path.expanduser(os.environ.get("KAPITSA_HOME", "~/.cache/kapitsa")), "work", "libmlxq.dylib")
    srcs = [os.path.join(HERE, "mlx_quant.cpp"), os.path.join(REPO_ROOT, "src/gguf/quant.cpp"),
            os.path.join(REPO_ROOT, "src/gguf/gguf.cpp")]
    if not os.path.exists(out) or os.path.getmtime(out) < max(os.path.getmtime(s) for s in srcs):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.check_call(["c++", "-O3", "-std=c++20", "-shared", "-fPIC", "-I" + os.path.join(REPO_ROOT, "src"),
                               "-I" + os.path.join(REPO_ROOT, "third_party")] + srcs + ["-o", out])
    q = ctypes.CDLL(out)
    q.mlxq_quantize_bf16.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    q.mlxq_quantize_bf16.restype = ctypes.c_int
    return q


class Source:
    """The safetensors files of SRC, mapped (copy-on-write, so ctypes can point into them; nothing is written)."""

    def __init__(self, path):
        self.t, self.maps = {}, {}
        for f in sorted(os.listdir(path)):
            if not f.endswith(".safetensors"):
                continue
            fp = os.path.join(path, f)
            with open(fp, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
                self.maps[f] = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_COPY)
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                if k in self.t:
                    sys.exit(f"{k} is in two files of {path}")
                a, b = v["data_offsets"]
                self.t[k] = (f, v["dtype"], v["shape"], 8 + n + a, b - a)

    def ptr(self, name, off=0):
        f, _, _, a, _ = self.t[name]
        return ctypes.addressof(ctypes.c_char.from_buffer(self.maps[f], a + off))

    def bytes(self, name):
        f, _, _, a, nb = self.t[name]
        return self.maps[f][a:a + nb]


def plan(src, bits, head_bits):
    """[(out name, how, source name, args, dtype, shape)] in file order, and the quantization overrides."""
    out, overrides = [], {}

    def quant(dst, name, b, rows=None, part=None):
        _, dt, shape, _, _ = src.t[name]
        if dt != "BF16":
            sys.exit(f"{name}: {dt}, the converter quantizes bf16")
        shape = list(shape) if rows is None else rows
        k = shape[-1]
        if k % 64:
            sys.exit(f"{name}: {k} inputs, not a multiple of 64")
        lead = shape[:-1]
        out.append((dst + ".weight", "q", name, (b, part), "U32", lead + [k * b // 32]))
        out.append((dst + ".scales", "s", name, (b, part), "BF16", lead + [k // 64]))
        out.append((dst + ".biases", "b", name, (b, part), "BF16", lead + [k // 64]))
        if b != bits:
            overrides[dst] = {"bits": b, "group_size": 64}

    def layer_of(n):
        return int(n.split(".")[2]) if n.startswith("model.layers.") else -1

    names = sorted(src.t, key=lambda n: (layer_of(n), n))
    for n in names:
        _, dt, shape, _, _ = src.t[n]
        dst = PREFIX + n
        if n.endswith("mlp.experts.gate_up_proj"):
            e, two_i, h = shape
            base = PREFIX + n[: -len("experts.gate_up_proj")] + "switch_mlp."
            quant(base + "gate_proj", n, bits, [e, two_i // 2, h], "gate")
            quant(base + "up_proj", n, bits, [e, two_i // 2, h], "up")
        elif n.endswith("mlp.experts.down_proj"):
            quant(PREFIX + n[: -len("experts.down_proj")] + "switch_mlp.down_proj", n, bits)
        elif n.endswith(".weight") and len(shape) == 2 and any(n.endswith(x + ".weight") for x in LINEARS):
            mod = dst[: -len(".weight")]
            b = 8 if n.endswith("mlp.gate.weight") or n.endswith("shared_expert_gate.weight") else bits
            if n == "lm_head.weight":
                b = head_bits
            quant(mod, n, b)
        else:
            out.append((dst, "copy", n, None, dt, list(shape)))
    return out, overrides


def nbytes(dt, shape):
    n = DT_SIZE[dt]
    for s in shape:
        n *= s
    return n


def produce(q, src, entry, threads, cache):
    """The bytes of one output tensor. Quantized ones are computed once for their three outputs (cache)."""
    dst, how, name, args, dt, shape = entry
    if how == "copy":
        return src.bytes(name)
    b, part = args
    key = (name, part)
    if key not in cache:
        cache.clear()
        _, _, sshape, _, _ = src.t[name]
        k = sshape[-1]
        if part is None:
            rows = 1
            for s in sshape[:-1]:
                rows *= s
            blocks = [(0, rows)]
            out_rows = rows
        else:  # half of each expert's [2I, H] rows
            e, two_i, _ = sshape
            i = two_i // 2
            blocks = [(x * two_i + (0 if part == "gate" else i), i) for x in range(e)]
            out_rows = e * i
        w = bytearray(out_rows * k * b // 8)
        s = bytearray(out_rows * (k // 64) * 2)
        bb = bytearray(len(s))
        r0 = 0
        for row, n in blocks:
            P = ctypes.c_char * 1
            wp = ctypes.addressof(P.from_buffer(w, r0 * k * b // 8))
            sp = ctypes.addressof(P.from_buffer(s, r0 * (k // 64) * 2))
            bp = ctypes.addressof(P.from_buffer(bb, r0 * (k // 64) * 2))
            if q.mlxq_quantize_bf16(src.ptr(name, row * k * 2), n, k, b, 64, wp, sp, bp, threads):
                sys.exit(f"{name}: cannot quantize to {b}-bit group 64")
            r0 += n
        cache[key] = {"q": w, "s": s, "b": bb}
    return cache[key][how]


def write_shard(path, entries, q, src, threads, cache):
    hdr, off = {}, 0
    for dst, _, _, _, dt, shape in entries:
        nb = nbytes(dt, shape)
        hdr[dst] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    hdr["__metadata__"] = {"format": "mlx"}
    h = json.dumps(hdr, separators=(",", ":")).encode()
    h += b" " * ((-(8 + len(h))) % 16384)
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for e in entries:
            data = produce(q, src, e, threads, cache)
            if len(data) != nbytes(e[4], e[5]):
                sys.exit(f"{e[0]}: {len(data)} bytes, expected {nbytes(e[4], e[5])}")
            f.write(data)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 5, 6, 8])
    ap.add_argument("--head-bits", type=int, default=None, choices=[4, 5, 6, 8], help="the LM head (default: --bits)")
    ap.add_argument("--threads", type=int, default=0, help="quantizer threads (0: every core)")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    if cfg.get("model_type") != "clover_moe":
        sys.exit(f"{a.src}: model_type {cfg.get('model_type')!r}, not clover_moe")
    src = Source(a.src)
    for n in ("model.layers.0.linear_attn.conv_right_weight", "model.layers.0.linear_attn.bd_softmax_logit_scale"):
        if n not in src.t:
            sys.exit(f"{a.src}: {n} missing (bd_gdn_*.safetensors not downloaded?)")
    q = lib()
    entries, overrides = plan(src, a.bits, a.head_bits or a.bits)
    os.makedirs(a.out, exist_ok=True)
    # Kapitsa reads every *.safetensors in a folder: shards of an earlier run (other bits, an interrupted one) would mix in
    old = [f for f in os.listdir(a.out) if f.endswith(".safetensors") or f.endswith(".part")]
    if old:
        sys.exit(f"{a.out} already holds {', '.join(sorted(old)[:3])}{' ...' if len(old) > 3 else ''}: "
                 "convert into an empty folder")
    shards, cur, size = [], [], 0
    for e in entries:
        nb = nbytes(e[4], e[5])
        if cur and size + nb > SHARD and e[1] in ("copy", "q"):
            shards.append(cur)
            cur, size = [], 0
        cur.append(e)
        size += nb
    shards.append(cur)
    total = len(shards)
    weight_map, cache, t0 = {}, {}, time.time()
    for i, sh in enumerate(shards):
        name = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        write_shard(os.path.join(a.out, name), sh, q, src, a.threads, cache)
        for e in sh:
            weight_map[e[0]] = name
        log(f"{name}: {len(sh)} tensors ({time.time() - t0:.0f} s)")
    json.dump({"metadata": {"total_size": sum(nbytes(e[4], e[5]) for e in entries)}, "weight_map": weight_map},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    cfg.pop("auto_map", None)
    qb = {"group_size": 64, "bits": a.bits, "mode": "affine", **overrides}
    cfg["quantization"] = qb
    cfg["quantization_config"] = qb
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in COPY:
        if os.path.exists(os.path.join(a.src, f)):
            shutil.copyfile(os.path.join(a.src, f), os.path.join(a.out, f))
    files = sorted(set(weight_map.values()) | {"model.safetensors.index.json", "config.json"} |
                   {f for f in COPY if os.path.exists(os.path.join(a.out, f))})
    conv = {"source": "Diffuspace/Clover-35B-A3B-Preview",
            "recipe": {"bits": a.bits, "group_size": 64, "head_bits": a.head_bits or a.bits, "router": 8,
                       "shared_expert_gate": 8, "quantizer": "src/gguf/quant.cpp (tools/mlx_quant.cpp)"},
            "files": {f: {"size": os.path.getsize(os.path.join(a.out, f)), "sha256": sha256(os.path.join(a.out, f))}
                      for f in files}}
    rev = os.path.join(a.src, "revision")
    if os.path.exists(rev):
        conv["revision"] = open(rev).read().strip()
    json.dump(conv, open(os.path.join(a.out, "conversion.json"), "w"), indent=2)
    log(f"{a.out}: {total} shards, {sum(os.path.getsize(os.path.join(a.out, f)) for f in files) / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
