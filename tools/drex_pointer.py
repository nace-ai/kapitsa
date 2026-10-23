#!/usr/bin/env python3
"""Extract Drex v2.0's pointer head into Kapitsa's pointer.bin.

Reads either a llama.cpp GGUF (pointer.* are F32 there) or a model folder of safetensors (an HF or a Kapitsa import
pack, any of F32 / F16 / BF16), finds the five head tensors and writes pointer.bin next to the model or anywhere else:

  tools/drex_pointer.py drex-v2.0-Q4_K_M.gguf  MODEL_DIR/pointer.bin
  tools/drex_pointer.py MODEL_DIR              MODEL_DIR/pointer.bin

pointer.bin layout (little endian):
  magic  b"KPS1"
  u32    version = 1
  u32    hidden  H
  u32    pointer dim dp
  f32    q_w[dp][H]  (row d: H values, the tensor's own row-major layout)
  f32    q_b[dp]
  f32    k_w[dp][H]
  f32    k_b[dp]
  f32    temperature[1]
"""
import json
import math
import os
import struct
import sys

GGUF_MAGIC = 0x46554747  # "GGUF"

FMT = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 9: ("<Q", 8), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}

WANT = ("pointer.q.weight", "pointer.q.bias", "pointer.k.weight", "pointer.k.bias",
        "pointer.temperature.weight")


def rd(f, fmt):
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise EOFError("short read")
    return struct.unpack(fmt, data)[0]


def rd_str(f):
    n = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def skip_value(f, t):
    if t == 8:
        rd_str(f)
    elif t == 9:
        et = rd(f, "<I")
        n = rd(f, "<Q")
        for _ in range(n):
            skip_value(f, et)
    else:
        rd(f, FMT[t][0])


def read_value(f, t):
    if t == 8:
        return rd_str(f)
    if t == 9:
        et = rd(f, "<I")
        n = rd(f, "<Q")
        return [read_value(f, et) for _ in range(n)]
    return rd(f, FMT[t][0])


def to_f32(raw, dtype, count):
    """Decodes `count` elements of `dtype` into a list of Python floats."""
    if dtype == "F32":
        return list(struct.unpack("<%df" % count, raw))
    if dtype == "F16":
        return list(struct.unpack("<%de" % count, raw))
    if dtype == "BF16":
        halves = struct.unpack("<%dH" % count, raw)
        return list(struct.unpack("<%df" % count, struct.pack("<%dI" % count, *[h << 16 for h in halves])))
    raise SystemExit("unsupported tensor dtype %s (F32, F16 and BF16 only)" % dtype)


def from_gguf(path):
    with open(path, "rb") as f:
        magic = rd(f, "<I")
        if magic != GGUF_MAGIC:
            raise SystemExit("not a GGUF file")
        version = rd(f, "<I")
        n_tensors = rd(f, "<Q")
        n_kv = rd(f, "<Q")
        print("GGUF v%d: %d tensors, %d metadata KVs" % (version, n_tensors, n_kv))
        meta = {}
        for _ in range(n_kv):
            key = rd_str(f)
            meta[key] = read_value(f, rd(f, "<I"))
        print("arch:", meta.get("general.architecture"), "name:", meta.get("general.name"))
        infos = []
        for _ in range(n_tensors):
            name = rd_str(f)
            dims = [rd(f, "<Q") for _ in range(rd(f, "<I"))]
            infos.append((name, dims, rd(f, "<I"), rd(f, "<Q")))
        align = meta.get("general.alignment", 32)
        header_end = f.tell()
        data_start = (header_end + align - 1) // align * align
        got, shapes = {}, {}
        for name, dims, ttype, off in infos:
            if name not in WANT:
                continue
            if ttype != 0:
                raise SystemExit("%s is GGUF type %d, expected F32 (0)" % (name, ttype))
            count = 1
            for d in dims:
                count *= d
            f.seek(data_start + off)
            raw = f.read(count * 4)
            if len(raw) != count * 4:
                raise EOFError(name)
            got[name] = to_f32(raw, "F32", count)
            shapes[name] = list(reversed(dims))
        return got, shapes


def from_safetensors(model_dir):
    files = sorted(n for n in os.listdir(model_dir) if n.endswith(".safetensors"))
    if not files:
        raise SystemExit("no *.safetensors in %s" % model_dir)
    got = {}
    for name in files:
        path = os.path.join(model_dir, name)
        with open(path, "rb") as f:
            n = rd(f, "<Q")
            header = json.loads(f.read(n))
            base = 8 + n
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                short = key[len("model.language_model."):] if key.startswith("model.language_model.") else key
                short = short[len("model."):] if short.startswith("model.") else short
                if short not in WANT or short in [k for k in got]:
                    continue
                start, end = info["data_offsets"]
                count = 1
                for d in info["shape"]:
                    count *= d
                f.seek(base + start)
                raw = f.read(end - start)
                if len(raw) != end - start:
                    raise EOFError(key)
                got[short] = (to_f32(raw, info["dtype"], count), list(info["shape"]))
        print("read %s (%d tensors)" % (name, len([k for k in header if k != "__metadata__"])))
    shapes = {}
    flat = {}
    for key, (vals, shape) in got.items():
        flat[key] = vals
        shapes[key] = shape
    return flat, shapes


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: drex_pointer.py <GGUF file | model folder> <pointer.bin out>")
    src, out = sys.argv[1], sys.argv[2]
    if os.path.isdir(src):
        got, shapes = from_safetensors(src)
    else:
        got, shapes = from_gguf(src)
    print("shapes:", {k: shapes[k] for k in sorted(shapes)})
    missing = [w for w in WANT if w not in got]
    if missing:
        raise SystemExit("missing pointer tensors: %s (found %s)" % (missing, sorted(got)))

    qw, qb, kw, kb, temp = (got["pointer.q.weight"], got["pointer.q.bias"],
                            got["pointer.k.weight"], got["pointer.k.bias"],
                            got["pointer.temperature.weight"])
    qw_dims = shapes["pointer.q.weight"][-2:]
    dp, H = qw_dims[0], qw_dims[1]
    if H * dp != len(qw):
        raise SystemExit("pointer.q.weight holds %d values, expected %d x %d" % (len(qw), dp, H))
    if len(kw) != len(qw):
        raise SystemExit("pointer.k.weight holds %d values, expected %d" % (len(kw), len(qw)))
    for name, vals, want in (("pointer.q.bias", qb, dp), ("pointer.k.bias", kb, dp),
                             ("pointer.temperature.weight", temp, 1)):
        if len(vals) != want:
            raise SystemExit("%s holds %d values, expected %d" % (name, len(vals), want))
    t = temp[0]
    if not math.isfinite(t) or t <= 0:
        raise SystemExit("temperature %r is not a positive finite number" % t)

    with open(out, "wb") as g:
        g.write(b"KPS1")
        g.write(struct.pack("<III", 1, H, dp))
        g.write(struct.pack("<%df" % len(qw), *qw))
        g.write(struct.pack("<%df" % dp, *qb))
        g.write(struct.pack("<%df" % len(kw), *kw))
        g.write(struct.pack("<%df" % dp, *kb))
        g.write(struct.pack("<f", t))
    print("hidden=%d dp=%d temperature=%g -> %s (%.1f MB)" % (H, dp, t, out, os.path.getsize(out) / 1e6))
    print("q_w[0][:4]", [round(v, 5) for v in qw[:4]], "q_b[:4]", [round(v, 5) for v in qb[:4]])


if __name__ == "__main__":
    main()
