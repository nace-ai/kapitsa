#!/usr/bin/env python3
"""NgramTable's MLX affine rows (src/model/ngram.cpp, read by Hy4-flash's EngramCpu) against mx.dequantize.

  ngram_affine_test.py [--bin build/ngram_rows] [--work /tmp/ngram-affine] [--no-mlx]

For each format (bits, group size, row width): random rows quantized as mx.quantize does, written as a safetensors
file (codes U32 [rows, dim * bits / 32], scales / biases BF16 [rows, dim / gs]), read back through build/ngram_rows
(bits and group size from the shapes, as EngramCpu opens the Engram tables), every value compared with the
dequantized table. mx.quantize makes the table and mx.dequantize is the reference; without mlx the test stops (exit 2)
unless --no-mlx asks for a NumPy copy of both (the codes of a row as one little-endian bit stream in uint32 words: MLX's packing, for 3 bits eight values
in three bytes). Kapitsa's value is fp32 scale * code + bias (exact product, one rounding): bitwise against that and
against mx.dequantize given the scales and biases in fp32; mx.dequantize in bf16 (product and sum each rounded to bf16)
must be within its two roundings of it. Needs NumPy.
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

try:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)  # no GPU: this test takes no lock
except ImportError:
    mx = None

# (bits, group, dim): Hy4's Engram rows are 96 wide (group 32; 64 does not divide 96), the rest the reader's range
FORMATS = [(3, 32, 96), (4, 32, 96), (3, 64, 192), (3, 128, 256), (3, 32, 160), (4, 64, 192), (2, 32, 96), (6, 32, 96),
           (8, 32, 96)]


def bf16_bits(x):
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_float(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def np_quantize(w, gs, bits):
    """mx.quantize (affine): per group scale = (max - min) / (2^bits - 1) with the edge of larger magnitude exact,
    codes rounded and clamped, scale and bias rounded to bf16; codes packed LSB first across uint32 words."""
    n, dim = w.shape
    g = w.reshape(n, dim // gs, gs)
    mxv, mnv = g.max(-1), g.min(-1)
    bins = float((1 << bits) - 1)
    mask = np.abs(mnv) > np.abs(mxv)
    scale = np.maximum((mxv - mnv) / bins, 1e-7).astype(np.float32)
    scale = np.where(mask, scale, -scale)
    edge = np.where(mask, mnv, mxv)
    q0 = np.rint(edge / scale)
    scale = np.where(q0 != 0, edge / np.where(q0 != 0, q0, 1), scale).astype(np.float32)
    bias = np.where(q0 == 0, 0.0, edge).astype(np.float32)
    s16, b16 = bf16_bits(scale), bf16_bits(bias)
    codes = np.clip(np.rint((g - bias[..., None]) / scale[..., None]), 0, bins).astype(np.uint64).reshape(n, dim)
    words = np.zeros((n, dim * bits // 32), dtype=np.uint32)
    for i in range(dim):
        at = i * bits
        words[:, at // 32] |= ((codes[:, i] << np.uint64(at % 32)) & 0xFFFFFFFF).astype(np.uint32)
        if at % 32 + bits > 32:
            words[:, at // 32 + 1] |= (codes[:, i] >> np.uint64(32 - at % 32)).astype(np.uint32)
    return words, s16, b16


def np_codes(words, dim, bits):
    """mx.dequantize's unpacking: value i at bits [bits i, bits i + bits) of the row's little-endian bit stream."""
    w64 = words.astype(np.uint64)
    out = np.zeros((words.shape[0], dim), dtype=np.uint64)
    for i in range(dim):
        at = i * bits
        v = w64[:, at // 32] >> np.uint64(at % 32)
        if at % 32 + bits > 32:
            v |= w64[:, at // 32 + 1] << np.uint64(32 - at % 32)
        out[:, i] = v & np.uint64((1 << bits) - 1)
    return out


def write_st(path, tensors):
    hdr, off, blobs = {}, 0, []
    for name, dt, arr in tensors:
        b = np.ascontiguousarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="build/ngram_rows")
    ap.add_argument("--work", default=None)
    ap.add_argument("--no-mlx", action="store_true", help="compare with the NumPy copy of mx.quantize / mx.dequantize")
    a = ap.parse_args()
    if mx is None and not a.no_mlx:
        print("SKIP: mlx is not installed, so there is no mx.dequantize to compare with (--no-mlx compares with the "
              "NumPy copy instead)")
        return 2
    work = a.work or tempfile.mkdtemp(prefix="ngram-affine.")
    os.makedirs(work, exist_ok=True)
    rng = np.random.default_rng(646)
    fails = 0
    print(f"reference: {'mlx ' + mx.__version__ if mx else 'NumPy copy of mx.quantize / mx.dequantize'}")
    for bits, gs, dim in FORMATS:
        rows = 513
        w = (rng.standard_normal((rows, dim)) * rng.uniform(0.01, 3.0, (rows, 1)) + rng.standard_normal((rows, 1))).astype(
            np.float32)
        if mx is not None:
            q, s, b = mx.quantize(mx.array(w).astype(mx.bfloat16), group_size=gs, bits=bits)
            # mx.dequantize with the bf16 scales and biases widened to fp32: its own unpacking, fp32 arithmetic
            ref_mx = np.array(mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32), group_size=gs, bits=bits))
            ref_bf = np.array(mx.dequantize(q, s, b, group_size=gs, bits=bits).astype(mx.float32))
            words = np.array(q).astype(np.uint32)
            s16 = np.array(s.astype(mx.float32)).view(np.uint32) >> 16
            b16 = np.array(b.astype(mx.float32)).view(np.uint32) >> 16
            s16, b16 = s16.astype(np.uint16), b16.astype(np.uint16)
        else:
            words, s16, b16 = np_quantize(w, gs, bits)
            ref_bf = ref_mx = None
        codes = np_codes(words, dim, bits).astype(np.float32)
        sc = np.repeat(bf16_float(s16), gs, axis=1)
        bi = np.repeat(bf16_float(b16), gs, axis=1)
        ref32 = (sc * codes + bi).astype(np.float32)  # the product is exact: one rounding, as fma
        if ref_mx is None:
            ref_mx = ref32
        if ref_bf is None:
            ref_bf = bf16_float(bf16_bits(ref32))
        path = os.path.join(work, f"t_{bits}_{gs}_{dim}.safetensors")
        write_st(path, [("w", "U32", words), ("s", "BF16", s16), ("b", "BF16", b16)])
        out = path + ".f32"
        subprocess.run([a.bin, path, "w", "s", "b", str(dim), out], check=True, stdout=subprocess.DEVNULL)
        got = np.fromfile(out, dtype=np.float32).reshape(rows, dim)
        os.remove(out)
        os.remove(path)
        bad32 = int((got.view(np.uint32) != ref32.view(np.uint32)).sum())
        badmx = int((got.view(np.uint32) != ref_mx.astype(np.float32).view(np.uint32)).sum())
        # mx.dequantize in bf16 rounds the product and then the sum to bf16 (2^-8 relative each: 8 significant bits):
        # its value is within (|scale * code| + |value|) * 2^-8 of the fp32 one (the ratio printed, <= 1)
        bound = (np.abs(sc * codes) + np.abs(got)) * 2.0**-8 * (1 + 2.0**-7) + 1e-38
        bfmax = float((np.abs(ref_bf - got) / bound).max())
        qerr = float(np.abs(ref32 - w).max() / np.abs(w).max())
        ok = bad32 == 0 and badmx == 0 and bfmax <= 1.0
        fails += not ok
        ref = "mx.dequantize" if mx else "the NumPy copy of mx.dequantize"
        print(f"{bits}-bit group {gs:3d}, rows of {dim}: {rows * dim} values, {bad32} differ from fp32 scale * code + bias, "
              f"{badmx} from {ref} (fp32 scales), bf16 {ref} at {bfmax:.2f} of its rounding bound; "
              f"quantization error {qerr:.3f} of max |w|: {'ok' if ok else 'FAILED'}")
    try:
        os.rmdir(work)
    except OSError:
        pass
    print("all checks passed" if not fails else f"FAILED: {fails} formats")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
