#!/usr/bin/env python3
"""Extract a Drex projected or Max readout head into Kapitsa's pointer.bin.

Reads only head data from a GGUF with F32 pointer tensors or safetensors with F32, F16 or BF16 pointer tensors.
Detects a five-tensor projected head or a two-tensor readout head and writes pointer.bin:

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

Version 2 replaces the projected weights and biases with f32 readout_w[dim][H], then temperature[1].
Its dim is the number of option-code outputs, at most 255. Readout temperature defaults to 1 when absent or zero;
otherwise any finite value is used. The projected head still requires a positive temperature.
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

PROJECTED = ("pointer.q.weight", "pointer.q.bias", "pointer.k.weight", "pointer.k.bias",
             "pointer.temperature.weight")
READOUT = ("pointer.readout.weight", "pointer.temperature.weight")
WANT = PROJECTED + ("pointer.readout.weight",)


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
            count = math.prod(dims)
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


def write_head(got, shapes, out):
    readout = "pointer.readout.weight" in got
    if readout and "pointer.temperature.weight" not in got:
        got = dict(got, **{"pointer.temperature.weight": [1.]})
        shapes = dict(shapes, **{"pointer.temperature.weight": [1]})
    names = READOUT if readout else PROJECTED
    if readout and any(name in got for name in PROJECTED[:-1]):
        raise SystemExit("mixed projected and readout pointer tensors")
    missing = [name for name in names if name not in got]
    if missing:
        raise SystemExit("missing pointer tensors: %s (found %s)" % (missing, sorted(got)))
    weight = names[0]
    shape = shapes[weight]
    if len(shape) != 2 or any(not isinstance(d, int) or d <= 0 for d in shape):
        raise SystemExit("%s needs a positive two-dimensional shape" % weight)
    dim, hidden = shape
    if hidden > 0x7fffffff or dim > (255 if readout else 4096):
        raise SystemExit("pointer head has a bad shape")
    expected = {weight: [dim, hidden], "pointer.temperature.weight": [1]}
    if not readout:
        expected.update({"pointer.q.bias": [dim], "pointer.k.weight": [dim, hidden], "pointer.k.bias": [dim]})
    for name in names:
        shape = shapes[name]
        if shape != expected[name]:
            raise SystemExit("%s has shape %s, expected %s" % (name, shape, expected[name]))
        if len(got[name]) != math.prod(shape):
            raise SystemExit("%s value count does not match its shape" % name)
        if any(not math.isfinite(value) or abs(value) > 3.4028234663852886e38 for value in got[name]):
            raise SystemExit("%s has non-finite or out-of-F32-range values" % name)
    temperature = got["pointer.temperature.weight"][0]
    if not readout and temperature <= 0:
        raise SystemExit("temperature %r is not a positive finite number" % temperature)
    if readout and temperature == 0:
        temperature = 1.
        got = dict(got, **{"pointer.temperature.weight": [temperature]})
    version = 2 if readout else 1
    with open(out, "wb") as f:
        f.write(b"KPS1")
        f.write(struct.pack("<III", version, hidden, dim))
        for name in names:
            values = got[name]
            f.write(struct.pack("<%df" % len(values), *values))
    return version, hidden, dim, temperature


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: drex_pointer.py <GGUF file | model folder> <pointer.bin out>")
    src, out = sys.argv[1], sys.argv[2]
    got, shapes = from_safetensors(src) if os.path.isdir(src) else from_gguf(src)
    print("shapes:", {k: shapes[k] for k in sorted(shapes)})
    version, hidden, dim, temperature = write_head(got, shapes, out)
    print("version=%d hidden=%d dim=%d temperature=%g -> %s (%.1f MB)" %
          (version, hidden, dim, temperature, out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
