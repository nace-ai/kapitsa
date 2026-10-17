#!/usr/bin/env python3
"""Builds a Kapitsa MTP sidecar (mtp.safetensors) from an official Qwen3.5 checkpoint's MTP head.

  mtp_sidecar.py <hf repo | local dir of HF safetensors> <out mtp.safetensors> [--hidden post_norm|pre_norm]

MLX conversions of the small Qwen3.5 models (e.g. lmstudio-community/Qwen3.5-4B-MLX-4bit) drop the
model's multi-token-prediction head. The official checkpoint (Qwen/Qwen3.5-4B, Apache-2.0) keeps it as
15 bf16 tensors named mtp.* (241 MB for the 4B). From a Hugging Face repo id only those tensors are
fetched, with HTTP range requests against each shard's header and the tensors' byte ranges, not the
9 GB checkpoint. RMSNorm weights are stored zero-centred by Hugging Face (applied as 1 + w) and in
MLX's convention (w + 1) by every MLX conversion Kapitsa loads, so they get + 1 here; linears stay bf16
(Kapitsa quantizes them to 8 bits at load). The header records which trunk hidden state the head reads
(base_hidden_variant; Qwen3.5-4B: post_norm, 0.76 vs 0.74 drafts accepted at depth 1 on a code prompt).

Put the output next to the model's files (or in a directory that links them): Kapitsa loads
<model dir>/mtp.safetensors and decodes speculatively by default. Stdlib only.
"""
import json, os, struct, sys, urllib.request


def bf16_plus_one(raw):
    out = bytearray(len(raw))
    for i in range(0, len(raw), 2):
        h = raw[i] | raw[i + 1] << 8
        f = struct.unpack("<f", struct.pack("<I", h << 16))[0] + 1.0
        u = struct.unpack("<I", struct.pack("<f", f))[0]
        u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16  # round to nearest even
        out[i], out[i + 1] = u & 0xFF, u >> 8
    return bytes(out)


class Source:
    def __init__(self, src):
        self.local = os.path.isdir(src)
        self.src = src

    def read(self, fname, a, b):  # bytes [a, b) of a file of the checkpoint
        if self.local:
            with open(os.path.join(self.src, fname), "rb") as f:
                f.seek(a)
                return f.read(b - a)
        url = f"https://huggingface.co/{self.src}/resolve/main/{fname}"
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", "User-Agent": "kapitsa-mtp-sidecar"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != b - a:
                    raise IOError(f"{fname}: {len(data)} bytes for a {b - a}-byte range")
                return data
            except Exception:
                if attempt == 3:
                    raise

    def index(self):
        if self.local:
            p = os.path.join(self.src, "model.safetensors.index.json")
            if os.path.exists(p):
                return json.load(open(p))["weight_map"]
            files = [f for f in os.listdir(self.src) if f.endswith(".safetensors")]
            wm = {}
            for f in files:
                for k in self.header(f)[0]:
                    wm[k] = f
            return wm
        url = f"https://huggingface.co/{self.src}/resolve/main/model.safetensors.index.json"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "kapitsa-mtp-sidecar"}),
                                    timeout=120) as r:
            return json.load(r)["weight_map"]

    def header(self, fname):
        n = struct.unpack("<Q", self.read(fname, 0, 8))[0]
        return json.loads(self.read(fname, 8, 8 + n)), 8 + n


def main():
    args = sys.argv[1:]
    hidden = "post_norm"
    if "--hidden" in args:
        i = args.index("--hidden")
        hidden = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2 or hidden not in ("post_norm", "pre_norm"):
        sys.exit(__doc__)
    src, out = Source(args[0]), args[1]
    shards = {}
    for k, f in src.index().items():
        if k.startswith("mtp."):
            shards.setdefault(f, []).append(k)
    if not shards:
        sys.exit(f"{args[0]} has no mtp.* tensors")
    tensors = {}
    for fname, keys in sorted(shards.items()):
        hdr, data0 = src.header(fname)
        for k in sorted(keys):
            t = hdr[k]
            if t["dtype"] != "BF16":
                sys.exit(f"{k}: expected bf16, found {t['dtype']}")
            a, b = t["data_offsets"]
            raw = src.read(fname, data0 + a, data0 + b)
            if k.endswith("norm.weight") or k.endswith("norm_embedding.weight") or k.endswith("norm_hidden.weight"):
                raw = bf16_plus_one(raw)
            tensors[k] = (t["dtype"], t["shape"], raw)
            print(f"{k} {t['shape']} {len(raw)} bytes", flush=True)
    meta = {"__metadata__": {"format": "mlx", "source": args[0], "norms": "mlx (+1)", "base_hidden_variant": hidden}}
    off = 0
    for k in sorted(tensors):
        meta[k] = {"dtype": tensors[k][0], "shape": tensors[k][1], "data_offsets": [off, off + len(tensors[k][2])]}
        off += len(tensors[k][2])
    h = json.dumps(meta, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    tmp = out + ".part"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for k in sorted(tensors):
            f.write(tensors[k][2])
    os.replace(tmp, out)
    print(f"wrote {out}: {len(tensors)} tensors, {off / 1e6:.1f} MB, base_hidden_variant {hidden}")


if __name__ == "__main__":
    main()
