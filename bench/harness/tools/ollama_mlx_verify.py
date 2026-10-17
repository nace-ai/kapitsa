#!/usr/bin/env python3
"""Verify a zero-copy Ollama import against its source checkpoint, tensor by tensor.

For every source tensor (index + mtp.safetensors) it finds the Ollama-named tensor in the manifest's tensor
blobs and compares the bytes (norms: checks bf16(stored + 1) == source, i.e. what Ollama's loader computes).
Also checks every blob digest, that no tensor is listed twice, and prints the quant metadata seen by Ollama's
per-blob reader (global quant_type/group_size + per-tensor overrides).
usage: ollama_mlx_verify.py <src dir> <OLLAMA_MODELS> <manifest path> [--fast]
"""
import hashlib, json, os, struct, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ollama_mlx_import import SHIFT_EXACT, SHIFT_SUFFIXES, bf16_to_f32, f32_to_bf16  # noqa: E402


def hdr(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return n, h


def main():
    src, models, manp = sys.argv[1:4]
    fast = "--fast" in sys.argv
    man = json.load(open(manp))
    blobs = os.path.join(models, "blobs")
    # Ollama view: name -> (blob path, base, entry, blob meta)
    view = {}
    metas = {}
    for L in man["layers"]:
        if L["mediaType"] != "application/vnd.ollama.image.tensor":
            continue
        p = os.path.join(blobs, L["digest"].replace(":", "-"))
        assert os.path.getsize(p) == L["size"], p
        if not fast:
            h = hashlib.sha256()
            with open(p, "rb", buffering=0) as f:
                for b in iter(lambda: f.read(64 << 20), b""):
                    h.update(b)
            assert "sha256:" + h.hexdigest() == L["digest"], ("digest mismatch", p)
        n, h = hdr(p)
        meta = h.pop("__metadata__", {}) or {}
        metas[L.get("name")] = (meta, len(h))
        for k, v in h.items():
            assert k not in view, ("duplicate tensor", k)
            view[k] = (p, 8 + n, v, meta)
    # source
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    files = sorted(set(idx.values()))
    if os.path.exists(os.path.join(src, "mtp.safetensors")):
        files.append("mtp.safetensors")
    model_norm = next(k for k in idx if k.endswith("model.norm.weight") and "mtp" not in k)
    checked = shifted = 0
    seen = set()
    nbytes = 0
    for fn in files:
        sp = os.path.join(src, fn)
        n, h = hdr(sp)
        h.pop("__metadata__", None)
        with open(sp, "rb") as sf:
            for k, v in h.items():
                if fn != "mtp.safetensors" and idx.get(k) != fn:
                    continue
                if k.endswith(".scales"):
                    nk = k[:-len(".scales")] + ".weight.scale"
                elif k.endswith(".biases"):
                    nk = k[:-len(".biases")] + ".weight.bias"
                else:
                    nk = k
                assert nk in view, ("missing in ollama view", nk)
                seen.add(nk)
                bp, bbase, bv, meta = view[nk]
                assert bv["dtype"] == v["dtype"] and bv["shape"] == v["shape"], (nk, bv, v)
                a, b = v["data_offsets"]
                sf.seek(8 + n + a)
                sdata = sf.read(b - a)
                with open(bp, "rb") as bf:
                    bf.seek(bbase + bv["data_offsets"][0])
                    bdata = bf.read(bv["data_offsets"][1] - bv["data_offsets"][0])
                nbytes += len(sdata)
                is_norm = ("vision" not in k and "visual" not in k) and (k.endswith(SHIFT_SUFFIXES) or k in SHIFT_EXACT or k == model_norm)
                if is_norm:
                    u = np.frombuffer(sdata, dtype=np.uint16)
                    st = np.frombuffer(bdata, dtype=np.uint16)
                    back = f32_to_bf16(bf16_to_f32(st) + np.float32(1.0))
                    bad = int((back != u).sum())
                    if bad:
                        print(f"  norm {k}: {bad}/{u.size} elements differ after Ollama's +1")
                    shifted += 1
                else:
                    assert sdata == bdata, ("data mismatch", k)
                checked += 1
    extra = set(view) - seen
    print(f"checked {checked} source tensors ({nbytes/1e9:.2f} GB), {shifted} un-shifted norms; "
          f"ollama tensors {len(view)}, extra {sorted(extra)[:5]}")
    for name, (meta, cnt) in metas.items():
        if meta:
            per = {k: v for k, v in meta.items() if k not in ("quant_type", "group_size")}
            print(f"  blob {name}: {cnt} tensors, quant_type={meta.get('quant_type')} group_size={meta.get('group_size')}"
                  + (f", {len(per)} per-tensor keys" if per else ""))
    assert not extra
    print("OK")


if __name__ == "__main__":
    main()
