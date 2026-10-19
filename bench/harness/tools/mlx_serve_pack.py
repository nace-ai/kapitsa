#!/usr/bin/env python3
"""Builds the model folders mlx-serve reads in the cross-engine benchmark (configs/mlx-serve.json), from the
weights the other engines read plus only the vendor files that differ. Stdlib only; APFS clones (`cp -c`), so
the shared shards take no extra disk space.

  mlx_serve_pack.py qwen3.6-35b-a3b [--models DIR] [--shard FILE]
  mlx_serve_pack.py qwen3.8-27b     [--models DIR] [--shard FILE]
  mlx_serve_pack.py qwen3.5-4b      [--models DIR] [--head-src FILE]
  mlx_serve_pack.py qwen3.5-9b      [--models DIR] [--head-src FILE]

qwen3.6-35b-a3b -> <models>/mlx-serve/Qwen3.6-35B-A3B-MLX-Serve-4bit, which is ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit:
  its trunk is mlx-community/Qwen3.6-35B-A3B-4bit, byte-identical (sha256) to <models>/qwen3.6-35b-a3b-4bit's
  shards, which are cloned; config.json (the same model config without MTPLX's keys) and the MTP sidecar
  mtp/weights.safetensors (475 MB, mlx-community's 4-bit MoE head) are downloaded.
qwen3.8-27b -> <models>/mlx-serve/qwen3.8-27b-4bit-mlx-serve: the lmstudio-community trunk of
  <models>/qwen3.8-27b-4bit-mtp (cloned), the drafter/ that ddalcu/Qwen3.8-27B-MLX-Serve-4bit ships (the same
  file as <models>/drafters/Qwen3.8-27B-DFlash2, cloned), and that pack's MTP head (its 29 language_model.mtp.*
  tensors, 314 MB, byte for byte) as the sidecar mtp/weights.safetensors. The head is read with HTTP range
  requests from the pack's last shard, or from a local copy of that shard (--shard, checked against its sha256).
  ddalcu's own trunk is a different conversion (bf16 embedding, some tensors quantized differently, 18.2 GB);
  every engine in the benchmark reads the lmstudio-community trunk instead.
qwen3.5-4b -> <models>/mlx-serve/qwen3.5-4b-4bit-mlx-serve: the vendor publishes no pack for this model. The
  lmstudio-community trunk of <models>/qwen3.5-4b-4bit-mtp (cloned) and, as the sidecar mtp/weights.safetensors, the
  4-bit MTP head mlx-community/Qwen3.5-4B-MTP-4bit with its tensors renamed from the bare fc.weight layout to
  mtp.fc.weight (the bytes unchanged), as the vendor's 35B pack does with mlx-community's 35B head. --head-src reads
  the head from a local file instead of downloading it (its tensor names, dtypes and shapes are checked against the
  pinned repo's header).
qwen3.5-9b -> <models>/mlx-serve/qwen3.5-9b-4bit-mlx-serve: as the 4B, from <models>/qwen3.5-9b-4bit
  (lmstudio-community/Qwen3.5-9B-MLX-4bit, two shards) and mlx-community/Qwen3.5-9B-MTP-4bit's head (137 MB).
"""
import argparse, hashlib, json, os, shutil, struct, subprocess, sys, urllib.request

PACKS = {
    "qwen3.6-35b-a3b": {
        "dir": "Qwen3.6-35B-A3B-MLX-Serve-4bit",
        "trunk": "qwen3.6-35b-a3b-4bit",
        "repo": "ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit", "rev": "6122e2b20a1d2e6c811b02b7912a91f1e8548de8",
        "clone": ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
                  "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
                  "model.safetensors.index.json", "chat_template.jinja", "configuration.json", "generation_config.json",
                  "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                  "video_preprocessor_config.json", "vocab.json"],
        "download": ["config.json", "mtp/weights.safetensors", "README.md"],
    },
    "qwen3.8-27b": {
        "dir": "qwen3.8-27b-4bit-mlx-serve",
        "trunk": "qwen3.8-27b-4bit-mtp",
        "drafter": "drafters/Qwen3.8-27B-DFlash2",
        "repo": "ddalcu/Qwen3.8-27B-MLX-Serve-4bit", "rev": "901aa73a1ff5456752c02e55582609e0040ca470",
        "clone": ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
                  "model-00003-of-00003.safetensors", "model.safetensors.index.json", "config.json",
                  "chat_template.jinja", "generation_config.json", "preprocessor_config.json", "processor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"],
        "head_shard": "model-00004.safetensors",
        # the sidecar this script writes from that shard (deterministic): checked, so a range read cannot go wrong quietly
        "head_sha256": "b4aee9bfb627f8453cbc4224eab59aee634de8607ed1ca4e0e44aa5e001dc58c",
    },
    "qwen3.5-4b": {
        "dir": "qwen3.5-4b-4bit-mlx-serve",
        "trunk": "qwen3.5-4b-4bit-mtp",
        "clone": ["model.safetensors", "model.safetensors.index.json", "config.json", "chat_template.jinja",
                  "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                  "video_preprocessor_config.json", "vocab.json"],
        "head_repo": "mlx-community/Qwen3.5-4B-MTP-4bit", "head_rev": "ab6f59bc6627196c611ab8851638651078170485",
        "head_file": "model.safetensors",
    },
    "qwen3.5-9b": {
        "dir": "qwen3.5-9b-4bit-mlx-serve",
        "trunk": "qwen3.5-9b-4bit",
        "clone": ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json",
                  "config.json", "chat_template.jinja", "preprocessor_config.json", "processor_config.json",
                  "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json"],
        "head_repo": "mlx-community/Qwen3.5-9B-MTP-4bit", "head_rev": "222dfd2c23fc9518d7b817e4f8e0cb0571787489",
        "head_file": "model.safetensors",
    },
}


def url(repo, rev, path):
    return f"https://huggingface.co/{repo}/resolve/{rev}/{path}"


def fetch(u, a=None, b=None):
    """Bytes of u, or of its range [a, b)."""
    headers = {"User-Agent": "kapitsa-bench-harness"}
    if a is not None:
        headers["Range"] = f"bytes={a}-{b - 1}"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=headers), timeout=300) as r:
                data = r.read()
            if a is not None and len(data) != b - a:
                raise IOError(f"{len(data)} bytes for a {b - a}-byte range")
            return data
        except Exception:
            if attempt == 4:
                raise


def tree(repo, rev):
    """{path: (size, lfs sha256 or None)} of every file of the repo at rev."""
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true", timeout=60) as r:
        return {e["path"]: (e["size"], (e.get("lfs") or {}).get("oid")) for e in json.load(r) if e["type"] == "file"}


def lfs_sha256(repo, rev, path):
    return tree(repo, rev).get(path, (None, None))[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clone(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    subprocess.run(["cp", "-c", os.path.realpath(src), dst], check=True)  # APFS clone: no data copied


def write_head(p, out, shard):
    """The pack's language_model.mtp.* tensors, byte for byte, as a single-file sidecar."""
    if shard:
        want = lfs_sha256(p["repo"], p["rev"], p["head_shard"])
        if sha256(shard) != want:
            sys.exit(f"{shard}: sha256 is not {p['head_shard']}'s ({want})")
        read = lambda a, b: _read_local(shard, a, b)
    else:
        u = url(p["repo"], p["rev"], p["head_shard"])
        read = lambda a, b: fetch(u, a, b)
    n = struct.unpack("<Q", read(0, 8))[0]
    header = json.loads(read(8, 8 + n))
    keys = sorted((k for k in header if k != "__metadata__" and ".mtp." in "." + k),
                  key=lambda k: header[k]["data_offsets"][0])
    if not any(k.endswith("mtp.fc.weight") for k in keys):
        sys.exit(f"no MTP head in {p['head_shard']}")
    new, off = {}, 0
    for k in keys:
        a, b = header[k]["data_offsets"]
        new[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + b - a]}
        off += b - a
    raw = json.dumps(new, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for k in keys:
            a, b = header[k]["data_offsets"]
            f.write(read(8 + n + a, 8 + n + b))
    got = sha256(out + ".part")
    if got != p["head_sha256"]:
        sys.exit(f"{out}.part: sha256 {got}, expected {p['head_sha256']}")
    os.replace(out + ".part", out)
    print(f"  wrote mtp/weights.safetensors: {len(keys)} tensors, {off / 1e6:.0f} MB (sha256 checked)")


def write_renamed_head(p, out, src):
    """The head file of p["head_repo"] (or a local copy, src) with every tensor renamed to mtp.<name>, bytes unchanged."""
    u = url(p["head_repo"], p["head_rev"], p["head_file"])
    read_remote = lambda a, b: fetch(u, a, b)
    n = struct.unpack("<Q", read_remote(0, 8))[0]
    ref = json.loads(read_remote(8, 8 + n))
    ref.pop("__metadata__", None)
    read = (lambda a, b: _read_local(src, a, b)) if src else read_remote
    n = struct.unpack("<Q", read(0, 8))[0]
    header = json.loads(read(8, 8 + n))
    meta = header.pop("__metadata__", None)
    shape = lambda h: {k: (v["dtype"], v["shape"]) for k, v in h.items()}
    if shape(header) != shape(ref):
        sys.exit(f"{src or u}: tensors differ from {p['head_repo']}@{p['head_rev'][:7]}")
    keys = sorted(header, key=lambda k: header[k]["data_offsets"][0])
    new, off = ({"__metadata__": meta} if meta else {}), 0
    for k in keys:
        a, b = header[k]["data_offsets"]
        new["mtp." + k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + b - a]}
        off += b - a
    raw = json.dumps(new, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
        for k in keys:
            a, b = header[k]["data_offsets"]
            f.write(read(8 + n + a, 8 + n + b))
    os.replace(out + ".part", out)
    print(f"  wrote mtp/weights.safetensors: {len(keys)} tensors of {p['head_repo']}, {off / 1e6:.0f} MB, renamed to mtp.*")


def _read_local(path, a, b):
    with open(path, "rb") as f:
        f.seek(a)
        return f.read(b - a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=sorted(PACKS))
    ap.add_argument("--models", default=os.environ.get("BENCH_MODELS") or "~/models",
                    help="the harness weights folder (BENCH_MODELS, default ~/models)")
    ap.add_argument("--shard", help="27B: a local copy of the pack's model-00004.safetensors to read the head from")
    ap.add_argument("--head-src", help="4B: a local copy of the head file to read instead of downloading it")
    a = ap.parse_args()
    p, models = PACKS[a.model], os.path.expanduser(a.models)
    out = os.path.join(models, "mlx-serve", p["dir"])
    os.makedirs(out, exist_ok=True)
    print(f"{out}:")
    files = tree(p["repo"], p["rev"]) if "repo" in p else {}
    cloned = [(os.path.join(models, p["trunk"], f), f) for f in p["clone"]]
    cloned += [(os.path.join(models, p["drafter"], f), "drafter/" + f) for f in ("config.json", "model.safetensors")
               if "drafter" in p]
    for src, f in cloned:
        # the 35B pack's files and the 27B pack's drafter/ are the vendor's own: same size or the wrong file
        if (a.model == "qwen3.6-35b-a3b" or f.startswith("drafter/")) and os.path.getsize(src) != files[f][0]:
            sys.exit(f"{src}: {os.path.getsize(src)} bytes, {p['repo']} has {files[f][0]}: not the same file")
        clone(src, os.path.join(out, f))
    print(f"  cloned {len(cloned)} files from {p['trunk']}/" + (f" and {p['drafter']}/" if "drafter" in p else ""))
    for f in p.get("download", []):
        dst = os.path.join(out, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with urllib.request.urlopen(url(p["repo"], p["rev"], f), timeout=300) as r, open(dst + ".part", "wb") as w:
            shutil.copyfileobj(r, w, 16 << 20)
        os.replace(dst + ".part", dst)
        want = lfs_sha256(p["repo"], p["rev"], f)
        if want and sha256(dst) != want:
            sys.exit(f"{dst}: sha256 mismatch (want {want})")
        print(f"  downloaded {f}" + (" (sha256 checked)" if want else ""))
    if "head_shard" in p:
        write_head(p, os.path.join(out, "mtp", "weights.safetensors"), a.shard)
    if "head_repo" in p:
        write_renamed_head(p, os.path.join(out, "mtp", "weights.safetensors"), a.head_src)


if __name__ == "__main__":
    main()
