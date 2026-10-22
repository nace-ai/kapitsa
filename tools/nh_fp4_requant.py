#!/usr/bin/env python3
"""A ModelOpt NVFP4 Nemotron-H checkpoint with the linears NVIDIA left BF16 quantized to NVFP4 too, except the LM
head: for Nemotron Nano 9B v2 NVFP4, the first and last two layers' Mamba / MLP projections and the four attention
layers' q / k / v / o (hf_quant_config.json's exclude_modules, less the Mamba conv1d weights, which stay BF16 as the
embedding and the norms do, and less lm_head). 0.80 B parameters: a decoded token then reads 4.97 GB of weights in
place of 6.11 (Kapitsa loads a BF16 head of a dense NVFP4 model as Q8_0 rows), the files take 6.69 GB in place of
7.84; every other tensor is copied byte for byte. Measured on Nano 9B v2 against the BF16 release (nh_eval --ref, 4092
positions): mean KL 0.0169 vs the stock checkpoint's 0.0116, top-1 93.7% vs 95.1%; decode 1.20x stock on an M3 Pro.

  python3 tools/nh_fp4_requant.py <NVFP4 checkpoint dir> <out dir> [--head] [--keep REGEX]... [--dry-run]

--head: the LM head to NVFP4 as well (4.67 GB a token, decode 1.27x stock), which costs far more: mean KL 0.0454,
top-1 88.7% on the same sequences. For measurements, not for use.
--keep REGEX: modules matching it stay BF16 (repeatable; e.g. --keep lm_head, --keep 'q_proj|k_proj|v_proj|o_proj'),
for measuring which part costs the quality. The output folder holds the rewritten shards, their index,
hf_quant_config.json without the quantized modules, every other file of the input folder (shards with nothing to
quantize as they are: APFS clones on macOS, which take no space), and requant.json (the
modules quantized, the input's own exclude list and this script's sha256). The result is the same to the byte on
every run.

The quantization is ModelOpt's NVFP4 post-training recipe (NVFP4QTensor, group 16): weight_scale_2 = amax(|W|) /
(6 * 448) in fp32; each block of 16's scale = amax(|block|) / (6 * weight_scale_2) cast to FP8 E4M3 (round to nearest
even; an all-zero block's scale is 1.0, and a block whose scale rounds to FP8 zero gets codes 0, where ModelOpt would
divide by zero: the block reads as zero either way); each value W / (scale * weight_scale_2) to the nearest E2M1 value (ties to
the even code, as ModelOpt's _cast_fp4 rounds), sign bit set for any negative value; two codes per byte, the even
element in the low nibble. On a layer NVIDIA quantized (Nano 9B v2's layer 3 up_proj, from the BF16 repo's weights)
it gives NVIDIA's weight_scale_2 exactly, 99.98% of its block scales and 99.34% of its codes: NVIDIA's checkpoint
was trained further after its quantization (quantization-aware distillation), so its weights differ from the BF16
release by about one rounding step in places, and the layers it kept BF16 get no such training here. No input_scale
is written: Kapitsa runs NVFP4 as W4A16 and does not read it (an engine with 4-bit activations needs a calibration).
Needs numpy. CPU only (15 M parameters a second on an M5 Pro core) and under 2 GB of memory.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys

import numpy as np

GROUP = 16
# positive FP8 E4M3 values by code (0..126; 127 is NaN) and E2M1 values by code (0..7)
E4M3 = np.array([(m * 2.0**-9) if e == 0 else (1 + m / 8) * 2.0 ** (e - 7) for e in range(16) for m in range(8)][:127],
                np.float64)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)
# ModelOpt's _cast_fp4: the code is the number of bounds below the value, plus one when the value equals a bound of an
# odd index (0.75, 1.75, 3.5: the tie goes up to the even code; 0.25, 1.25, 2.5, 5 go down to it)
E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)
E2M1_UP = np.array([0.75, 1.75, 3.5], np.float32)
DTYPES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "I16": 2, "I32": 4, "I64": 8,
          "U16": 2, "U32": 4, "U64": 8, "BOOL": 1}
LINEARS = ("in_proj", "out_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head")


def e4m3_codes(x):
    """FP8 E4M3 codes of non-negative float values (round to nearest, ties to the even code; at most 448)."""
    x = np.minimum(np.asarray(x, np.float64), 448.0)
    i = np.clip(np.searchsorted(E4M3, x), 1, len(E4M3) - 1)
    lo, hi = E4M3[i - 1], E4M3[i]
    d = (x - lo) - (hi - x)
    up = (d > 0) | ((d == 0) & (i % 2 == 0))
    return np.where(up, i, i - 1).astype(np.uint8)


def e2m1_codes(x):
    """E2M1 codes (sign in bit 3) of float32 values already divided by their scales, as ModelOpt's _cast_fp4."""
    a = np.abs(x)
    code = np.searchsorted(E2M1_BOUNDS, a, side="left").astype(np.uint8)
    code += np.isin(a, E2M1_UP).astype(np.uint8)
    return code | ((x < 0).astype(np.uint8) << 3)


def quantize(w, rows=2048):
    """NVFP4 of a float32 matrix [N][K]: packed codes uint8 [N][K/2], scale codes uint8 [N][K/16], weight_scale_2."""
    N, K = w.shape
    if K % GROUP:
        raise ValueError(f"K = {K} is not a multiple of {GROUP}")
    amax = np.float32(0)
    for r in range(0, N, rows):
        amax = max(amax, np.float32(np.abs(w[r:r + rows]).max()))
    ws2 = np.float32(amax / np.float32(6.0 * 448.0))
    packed = np.empty((N, K // 2), np.uint8)
    scales = np.empty((N, K // GROUP), np.uint8)
    for r in range(0, N, rows):
        b = np.asarray(w[r:r + rows], np.float32).reshape(-1, K // GROUP, GROUP)
        sc = (np.abs(b).max(-1) / (np.float32(6.0) * ws2)).astype(np.float32)
        sc[sc == 0] = 1.0
        code = e4m3_codes(sc)
        den = (E4M3[code].astype(np.float32) * ws2)[..., None]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(den > 0, b / den, np.float32(0)).astype(np.float32)
        q = e2m1_codes(x).reshape(b.shape[0], K)
        packed[r:r + rows] = q[:, 0::2] | (q[:, 1::2] << 4)
        scales[r:r + rows] = code
    return packed, scales, ws2


def dequantize(packed, scales, ws2):
    """The float32 matrix an NVFP4 tensor stands for (for checks)."""
    N = packed.shape[0]
    q = np.stack([packed & 15, packed >> 4], -1).reshape(N, -1)
    v = E2M1[q & 7] * np.where(q & 8, np.float32(-1), np.float32(1))
    s = np.repeat(E4M3[scales].astype(np.float32), GROUP, axis=1)
    return v * s * np.float32(ws2)


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return h, 8 + n


def bf16_matrix(path, base, info):
    a, b = info["data_offsets"]
    u = np.memmap(path, np.uint16, "r", base + a, (b - a) // 2).reshape(info["shape"])
    return u, lambda r0, r1: (np.asarray(u[r0:r1], np.uint32) << 16).view(np.float32)


class Rows:
    """A BF16 matrix's rows as float32, read in slices (the whole matrix is never converted at once)."""

    def __init__(self, path, base, info):
        self.u, self.get = bf16_matrix(path, base, info)
        self.shape = tuple(info["shape"])

    def __getitem__(self, s):
        return self.get(s.start, min(s.stop, self.shape[0]))


def write_shard(path, tensors, metadata):
    """tensors: [(name, dtype, shape, bytes-like or callable writing to a file)] in file order."""
    header, off = {}, 0
    if metadata:
        header["__metadata__"] = metadata
    for name, dt, shape, data, size in tensors:
        header[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + size]}
        off += size
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * (-len(hj) % 8)
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for chunk in (struct.pack("<Q", len(hj)), hj):
            f.write(chunk)
            h.update(chunk)
        for name, dt, shape, data, size in tensors:
            for chunk in data():
                f.write(chunk)
                h.update(chunk)
    return off, h.hexdigest()


def copier(path, start, size, step=64 << 20):
    def gen():
        with open(path, "rb") as f:
            f.seek(start)
            left = size
            while left:
                c = f.read(min(step, left))
                if not c:
                    raise OSError(f"{path}: short read")
                left -= len(c)
                yield c
    return gen


def file_sha256(path, step=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while c := f.read(step):
            h.update(c)
    return h.hexdigest()


def clone_or_copy(src, dst):
    """An APFS clone where the filesystem makes one (no space used until either file changes), else a copy."""
    if os.path.exists(dst):
        os.remove(dst)
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", src, dst], capture_output=True).returncode == 0:
        return
    shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--head", action="store_true", help="the LM head to NVFP4 too (measurements only: 3.9x the KL)")
    ap.add_argument("--keep", action="append", default=[], help="modules matching this regex stay BF16")
    ap.add_argument("--dry-run", action="store_true", help="list what would be quantized, write nothing")
    a = ap.parse_args()

    qc_path = os.path.join(a.src, "hf_quant_config.json")
    qc = json.load(open(qc_path))
    q = qc.get("quantization", {})
    if q.get("quant_algo") not in ("NVFP4", "W4A16_NVFP4") or q.get("group_size") != GROUP:
        sys.exit(f"{qc_path}: not a ModelOpt NVFP4 checkpoint of group {GROUP}")
    idx_path = os.path.join(a.src, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        index = json.load(open(idx_path))
        shards = sorted(set(index["weight_map"].values()))
    else:
        index, shards = None, ["model.safetensors"]
    excluded = q.get("exclude_modules", [])
    keep = [re.compile(k) for k in a.keep] + ([] if a.head else [re.compile(r"(^|\.)lm_head$")])
    # the modules to quantize: excluded linears with a BF16 2-D weight whose K is a multiple of 16
    headers = {s: read_header(os.path.join(a.src, s)) for s in shards}
    where = {n: s for s, (h, _) in headers.items() for n in h if n != "__metadata__"}
    targets = []
    for m in excluded:
        if m.rsplit(".", 1)[-1] not in LINEARS or any(k.search(m) for k in keep):
            continue
        info = headers[where[m + ".weight"]][0][m + ".weight"] if m + ".weight" in where else None
        if not info or info["dtype"] != "BF16" or len(info["shape"]) != 2 or info["shape"][1] % GROUP:
            sys.exit(f"{m}: not a BF16 matrix with K % {GROUP} == 0 ({info})")
        if any(m + s in where for s in (".weight_scale", ".weight_scale_2", ".input_scale")):
            sys.exit(f"{m}: already has NVFP4 scales")
        targets.append(m)
    before = sum(headers[where[m + ".weight"]][0][m + ".weight"]["data_offsets"][1] -
                 headers[where[m + ".weight"]][0][m + ".weight"]["data_offsets"][0] for m in targets)
    print(f"{len(targets)} modules to NVFP4 ({before / 1e9:.3f} GB of BF16 -> ~{before * 9 / 32 / 1e9:.3f} GB)")
    for m in targets:
        print("  " + m)
    if a.dry_run:
        return
    if os.path.abspath(a.out) == os.path.abspath(a.src):
        sys.exit("the output folder must differ from the input")
    os.makedirs(a.out, exist_ok=True)

    tset = set(targets)
    weight_map, shard_sha, total, rewritten = {}, {}, 0, set()
    for s in shards:
        h, base = headers[s]
        src = os.path.join(a.src, s)
        if not any(n.endswith(".weight") and n[:-7] in tset for n in h):
            # no module of this shard changes: the same file (a clone)
            clone_or_copy(src, os.path.join(a.out, s))
            total += sum(i["data_offsets"][1] - i["data_offsets"][0] for n, i in h.items() if n != "__metadata__")
            shard_sha[s] = file_sha256(os.path.join(a.out, s))
            weight_map.update({n: s for n in h if n != "__metadata__"})
            print(f"{s}: unchanged (cloned)", flush=True)
            continue
        names = sorted((n for n in h if n != "__metadata__"), key=lambda n: h[n]["data_offsets"][0])
        out = []
        for n in names:
            info = h[n]
            m = n[:-len(".weight")] if n.endswith(".weight") else None
            if m in tset:
                print(f"{s}: {m} {info['shape']}", flush=True)
                packed, scales, ws2 = quantize(Rows(src, base, info))
                N, K = info["shape"]
                out.append((m + ".weight", "U8", (N, K // 2), (lambda p=packed: iter([p.tobytes()])), packed.nbytes))
                out.append((m + ".weight_scale", "F8_E4M3", (N, K // GROUP), (lambda c=scales: iter([c.tobytes()])),
                            scales.nbytes))
                out.append((m + ".weight_scale_2", "F32", (), (lambda v=ws2: iter([np.float32(v).tobytes()])), 4))
            else:
                a0, a1 = info["data_offsets"]
                out.append((n, info["dtype"], info["shape"], copier(src, base + a0, a1 - a0), a1 - a0))
        size, sha = write_shard(os.path.join(a.out, s), out, h.get("__metadata__"))
        total += size
        shard_sha[s] = sha
        for t in out:
            weight_map[t[0]] = s
        side = s[:-len(".safetensors")] + ".json"
        if os.path.isfile(os.path.join(a.src, side)):
            # ModelOpt's per-shard index (NVIDIA's NVFP4 repos have one per shard): this shard's tensors now
            json.dump({"metadata": {"total_size": size}, "weight_map": {n: s for n in sorted(t[0] for t in out)}},
                      open(os.path.join(a.out, side), "w"), indent=4)
            rewritten.add(side)
        print(f"{s}: {size / 1e9:.3f} GB, sha256 {sha}", flush=True)
    if index is not None:
        index = dict(index)
        index["metadata"] = dict(index.get("metadata", {}), total_size=total)
        index["weight_map"] = dict(sorted(weight_map.items()))
        json.dump(index, open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    qc["quantization"] = dict(q, exclude_modules=[m for m in excluded if m not in tset])
    json.dump(qc, open(os.path.join(a.out, "hf_quant_config.json"), "w"), indent=4)
    skip = set(shards) | rewritten | {"model.safetensors.index.json", "hf_quant_config.json", "requant.json", ".gitattributes"}
    for f in sorted(os.listdir(a.src)):
        p = os.path.join(a.src, f)
        if f not in skip and os.path.isfile(p):
            shutil.copyfile(p, os.path.join(a.out, f))
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    json.dump({"tool": "tools/nh_fp4_requant.py", "tool_sha256": me, "recipe": "ModelOpt NVFP4 PTQ, group 16, W4A16",
               "quantized": targets, "head": a.head, "kept_bf16_by_flag": a.keep, "source_exclude_modules": excluded,
               "shards_sha256": shard_sha},
              open(os.path.join(a.out, "requant.json"), "w"), indent=2)
    print(f"wrote {a.out}: {total / 1e9:.3f} GB of weights")


if __name__ == "__main__":
    main()
