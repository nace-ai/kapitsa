#!/usr/bin/env python3
"""Zero-copy oMLX view ("symlink farm") of an MLX pack whose MTP head is an MTPLX-style side-car.

Why: oMLX's Lightning MTP only sees an MTP head that the checkpoint index declares and that sits in a
``model*.safetensors`` shard. The packs here ship the head as a side-car ``mtp.safetensors`` that the index does not
list, so oMLX logs "config declares mtp heads but checkpoint ships no mtp.* weights" and runs without MTP. oMLX's own
fix (admin "import MTPLX side-car") rewrites the pack in place and writes a remapped copy of the head. This script
does the same thing as a view instead, without touching or copying the weights:

  <farm>/<every pack file>        -> symlink to the original
  <farm>/model-mtp.safetensors    -> symlink to <pack>/mtp.safetensors
  <farm>/model.safetensors.index.json   real file: original map + mtp.* keys
  <farm>/config.json              real file: original + per-module quant entries for side-car modules whose group
                                  size differs from the body (35B experts: g32), and mtp_num_hidden_layers >= 1

The side-car itself is NOT linked under its own name, so nothing loads it twice. Keys stay ``mtp.*``; oMLX's patched
mlx-vlm sanitize maps them to ``language_model.mtp.*`` (and stacks numbered experts into switch_mlp).
The benchmark's 4B config (configs/omlx.json, q35-4b-omlx-best) reads such a farm of qwen3.5-4b-4bit-mtplx.

usage: omlx_mtp_farm.py <pack_dir> <farm_dir>      (idempotent; rebuilds the farm). Stdlib only.
"""
import json, os, struct, sys


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main(src, dst):
    src, dst = os.path.abspath(os.path.expanduser(src)), os.path.abspath(os.path.expanduser(dst))
    sidecar = os.path.join(src, "mtp.safetensors")
    if not os.path.isfile(sidecar):
        sys.exit(f"no mtp.safetensors in {src}")
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(dst):  # rebuild: only ever remove our own links / generated files
        p = os.path.join(dst, name)
        if os.path.islink(p) or name in ("config.json", "model.safetensors.index.json", "FARM.json"):
            os.remove(p)
        else:
            sys.exit(f"unexpected non-farm entry {p}; refusing to continue")
    skip = {"mtp.safetensors", "config.json", "model.safetensors.index.json"}
    for name in sorted(os.listdir(src)):
        if name.startswith(".") or name in skip:
            continue
        os.symlink(os.path.join(src, name), os.path.join(dst, name))
    os.symlink(sidecar, os.path.join(dst, "model-mtp.safetensors"))

    h = header(sidecar)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = idx.setdefault("weight_map", {})
    clash = [k for k in h if k in wm]
    if clash:
        sys.exit(f"index already lists side-car keys, e.g. {clash[:3]}")
    for k in h:
        wm[k] = "model-mtp.safetensors"
    meta = idx.setdefault("metadata", {})
    meta["total_size"] = int(meta.get("total_size", 0) or 0) + os.path.getsize(sidecar)
    json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=1)

    cfg = json.load(open(os.path.join(src, "config.json")))
    q = cfg.get("quantization") or {}
    gbits, ggs = int(q.get("bits", 4)), int(q.get("group_size", 64))
    side_q = cfg.get("mtplx_mtp_quantization") or {}
    added = {}
    for k, v in h.items():
        if not k.endswith(".scales"):
            continue
        base = k[: -len(".scales")]
        w = h.get(base + ".weight")
        if not w or w["dtype"] != "U32":
            continue
        bits = int(side_q.get("bits", gbits))
        in_dim = w["shape"][-1] * 32 // bits
        gs = in_dim // v["shape"][-1]
        if (bits, gs) == (gbits, ggs):
            continue
        mod = "language_model." + base
        parts = mod.split(".")
        if "experts" in parts:  # numbered experts are stacked into switch_mlp by sanitize
            i = parts.index("experts")
            mod = ".".join(parts[:i] + ["switch_mlp"] + parts[i + 2:])
        spec = {"group_size": gs, "bits": bits, "mode": q.get("mode", "affine")}
        if added.get(mod, spec) != spec:
            sys.exit(f"inconsistent quantization inside {mod}")
        added[mod] = spec
    for section in ("quantization", "quantization_config"):
        if isinstance(cfg.get(section), dict):
            cfg[section].update(added)
    tc = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    tc["mtp_num_hidden_layers"] = max(int(tc.get("mtp_num_hidden_layers") or 0), 1)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    json.dump({"source": src, "sidecar_tensors": len(h), "quant_entries_added": added},
              open(os.path.join(dst, "FARM.json"), "w"), indent=2)
    print(f"farm {dst}: {len(os.listdir(dst))} entries, {len(h)} mtp tensors indexed, "
          f"{len(added)} quant entries added {sorted(set(json.dumps(s) for s in added.values()))}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
