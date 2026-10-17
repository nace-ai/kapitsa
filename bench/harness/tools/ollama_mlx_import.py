#!/usr/bin/env python3
"""Zero-copy import of a local MLX affine-quantized Qwen3.5-family checkpoint (+ MTP sidecar) into an
Ollama 0.40.5 model store, so Ollama's MLX runner runs the exact same 4-bit weights as the other engines.
Written for the cross-engine benchmark (bench/harness/README.md, "Ollama"). Needs numpy, APFS (the source
weights and the store on the same volume) and a dedicated `ollama serve` on --host with OLLAMA_MODELS=<store>,
which `ollama create` talks to:

  OLLAMA_HOST=127.0.0.1:8193 OLLAMA_MODELS=<store> OLLAMA_NOPRUNE=1 ollama serve &
  python3 ollama_mlx_import.py --src <MLX dir> --name nace/qwen3.6:35b-a3b-mlx-4bit-mtp --models <store> \
      --work <empty scratch dir> --mtp-quant 4:32 --renderer qwen3.5 --parser qwen3.5 --requires 0.19.0 \
      --params '{"min_p": 0, "presence_penalty": 1.5, "repeat_penalty": 1, "temperature": 1, "top_k": 20, "top_p": 0.95}'

Why not plain `ollama create FROM <dir>`: Ollama's safetensors importer rewrites every tensor into its own
content-addressed blob (create/prequant.go + create/writer.go), i.e. a full second copy of the weights
(~20 GB for the 35B, ~16 GB for the 27B), and it labels every prequantized blob with the config's global
quant type (int4/g64), which is wrong for mlx-community's 8-bit router gates and for the g32 MTP experts.

What this does instead (all inside OLLAMA_MODELS, nothing outside is modified):
  * big blobs: each source .safetensors file is APFS-cloned (`cp -c`, copy-on-write, no data copied) and its
    JSON header is rewritten IN PLACE in the clone (same header length, space padded), listing the tensors with
    Ollama's names (`X.scales` -> `X.weight.scale`, `X.biases` -> `X.weight.bias`) and quant metadata
    (`quant_type`/`group_size`, per-tensor overrides where needed). Data offsets are unchanged, so the tensor
    bytes are the source bytes. Only the rewritten header blocks are new on disk.
  * norms: Ollama's qwen3_5 loader adds +1 to RMSNorm weights whenever `mtp.*` tensors exist (HF zero-centred
    convention). mlx-lm/MTPLX checkpoints already store the shifted (+1) values, so the shifted norms are
    stored un-shifted (bf16(w-1)); Ollama's bf16 AddScalar restores them (checked: bit-exact for every element
    except a few of one MTP norm in the 35B sidecar; the count is printed, and ollama_mlx_verify.py rechecks).
  * small blobs: the un-shifted norms go through the real `ollama create` (float import of a tiny
    model.safetensors next to symlinks of the config/tokenizer JSONs), which also writes the ConfigV2 blob
    (renderer/parser/capabilities), JSON config layers and params. Quantized groups whose quant differs from
    their file's default (8-bit gates) and any tensors that must leave a big header so it fits are written as
    small packed blobs in Ollama's own layout.
  * manifest: the manifest written by `ollama create` gets the extra tensor layers and is re-stored as a
    content-addressed manifest blob + manifests-v2 symlink, exactly like manifest.WriteManifestData.

Every blob digest is the real sha256 of the blob file. Clones share blocks with the source files: deleting
either side alone frees (almost) nothing; `du` counts both.
"""
import argparse, hashlib, json, os, re, shutil, struct, subprocess, sys, tempfile, time

import numpy as np

OLLAMA = os.environ.get("OLLAMA_BIN") or shutil.which("ollama") or "/Applications/Ollama.app/Contents/Resources/ollama"
SHIFT_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                  ".self_attn.q_norm.weight", ".self_attn.k_norm.weight")
SHIFT_EXACT = ("mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight")
CONFIG_JSONS = ("config.json", "configuration.json", "generation_config.json", "preprocessor_config.json",
                "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json")
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1, "I8": 1, "U16": 2, "I64": 8, "F64": 8, "BOOL": 1}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        raw = f.read(n)
    h = json.loads(raw)
    meta = h.pop("__metadata__", None)
    return n, h, meta


def is_shifted_norm(name, model_norm):
    if "vision" in name or "visual" in name:
        return False
    return name.endswith(SHIFT_SUFFIXES) or name in SHIFT_EXACT or name == model_norm


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


def compact(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path, bufsize=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Src:
    """One source safetensors file: header, kept/moved tensors."""

    def __init__(self, path, names):
        self.path = path
        self.n, self.h, self.meta = read_header(path)
        if names is not None:  # restrict to indexed tensors
            self.h = {k: v for k, v in self.h.items() if k in names}
        self.base = 8 + self.n

    def nbytes(self, k):
        a, b = self.h[k]["data_offsets"]
        return b - a

    def read(self, k):
        a, b = self.h[k]["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.base + a)
            return f.read(b - a)


def plan(args):
    src_dir = args.src
    idx = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))["weight_map"]
    files = {}
    for k, fn in idx.items():
        files.setdefault(fn, set()).add(k)
    srcs = [Src(os.path.join(src_dir, fn), names) for fn, names in sorted(files.items())]
    mtp_path = os.path.join(src_dir, "mtp.safetensors")
    mtp = None
    if os.path.exists(mtp_path) and not args.no_mtp:
        mtp = Src(mtp_path, None)
        srcs.append(mtp)
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    gbits, ggroup = int(q["bits"]), int(q["group_size"])
    overrides = {k: (int(v["bits"]), int(v["group_size"])) for k, v in q.items() if isinstance(v, dict)}
    mb, mg = (int(x) for x in args.mtp_quant.split(":"))

    # model.norm name
    model_norm = next(k for k in idx if k.endswith("model.norm.weight") and "mtp" not in k)

    blobs = []        # big blob plans
    norms = []        # (src, name)
    small = {}        # (bits, group) or None -> list of (src, base or name, is_group)
    # Quantized groups whose members sit in different shards (35B: one) are copied whole into a small blob,
    # so every blob carries a weight together with its scale (Ollama reads per-tensor quant info per blob).
    where = {}
    for s in srcs:
        for k in s.h:
            where[k] = s
    split = {}
    for k, s in where.items():
        if k.endswith(".scales"):
            b = k[: -len(".scales")]
            mem = [b + x for x in (".weight", ".scales", ".biases") if b + x in where]
            if len({id(where[t]) for t in mem}) > 1:
                split[b] = mem
    for b, mem in split.items():
        w, sc = where[b + ".weight"].h[b + ".weight"], where[b + ".scales"].h[b + ".scales"]
        bits, group = overrides.get(b, (gbits, ggroup))
        assert bits * group == w["shape"][-1] * 32 // sc["shape"][-1], b
        small.setdefault((bits, group), []).append(("split", b, mem))
    split_members = {t for mem in split.values() for t in mem}
    for s in srcs:
        is_mtp = s is mtp
        names = set(s.h) - split_members
        groups = {}
        for k in names:
            if k.endswith(".scales"):
                b = k[: -len(".scales")]
                assert b + ".weight" in names, k
                groups[b] = [b + ".weight", k] + ([b + ".biases"] if b + ".biases" in names else [])
        in_group = {t for g in groups.values() for t in g}
        for k in names:
            if k not in in_group and (k.endswith(".scale") or k.endswith(".biases") or k.endswith(".scales")):
                raise SystemExit(f"unexpected companion-like tensor {k}")
        quant = {}
        for b, members in groups.items():
            w, sc = s.h[members[0]], s.h[members[1]]
            assert w["dtype"] == "U32", (b, w["dtype"])
            prod = w["shape"][-1] * 32 // sc["shape"][-1]
            if is_mtp:
                bits, group = mb, mg
            else:
                bits, group = overrides.get(b, (gbits, ggroup))
            if bits * group != prod:
                raise SystemExit(f"{b}: quant {bits}/{group} inconsistent with shapes (bits*group={prod})")
            quant[b] = (bits, group)
        default = (mb, mg) if is_mtp else (gbits, ggroup)
        moved = set()
        for k in names:
            if k not in in_group and is_shifted_norm(k, model_norm):
                assert s.h[k]["dtype"] == "BF16", k
                norms.append((s, k))
                moved.add(k)
        for b, bq in quant.items():
            if bq != default:
                small.setdefault(bq, []).append((s, b, True))
                moved.update(groups[b])

        def header_for(excl):
            hdr = {"__metadata__": {"quant_type": "int%d" % default[0], "group_size": str(default[1])}}
            for k in sorted(names - excl):
                v = s.h[k]
                if k in in_group:
                    b = k.rsplit(".", 1)[0]
                    suf = k.rsplit(".", 1)[1]
                    nk = b + {"weight": ".weight", "scales": ".weight.scale", "biases": ".weight.bias"}[suf]
                else:
                    nk = k
                hdr[nk] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": v["data_offsets"]}
            return compact(hdr)

        hdr = header_for(moved)
        # Make the header fit in place: move the cheapest (smallest) units out until it does.
        if len(hdr) > s.n:
            units = []
            for b, members in groups.items():
                if not (set(members) & moved):
                    units.append((sum(s.nbytes(t) for t in members), b, True))
            for k in names - in_group - moved:
                units.append((s.nbytes(k), k, False))
            units.sort()
            for size, key, is_group in units:
                if len(hdr) <= s.n:
                    break
                bq = quant[key] if is_group else None
                small.setdefault(bq, []).append((s, key, is_group))
                moved.update(groups[key] if is_group else [key])
                # cheap estimate first, exact header every 16 moves
                hdr = header_for(moved)
        assert len(hdr) <= s.n
        blobs.append({"src": s, "header": hdr, "moved": moved | (set(s.h) & split_members), "default": default,
                      "tensors": len(names - moved), "name": "nace-zc/" + os.path.basename(s.path)})
    # resolve "split" entries to (src, key, False) per member with group renames
    for bq, ents in small.items():
        out = []
        for e in ents:
            if e[0] == "split":
                out.append(("split", e[1], [(where[t], t) for t in e[2]]))
            else:
                out.append(e)
        small[bq] = out
    return {"srcs": srcs, "blobs": blobs, "norms": norms, "small": small, "cfg": cfg, "model_norm": model_norm,
            "has_mtp": mtp is not None}


def entry_tensors(e):
    """(new name, src, source key) for a small-blob entry."""
    if e[0] == "split":
        b = e[1]
        ren = {".weight": ".weight", ".scales": ".weight.scale", ".biases": ".weight.bias"}
        return [(b + ren[t[len(b):]], s, t) for s, t in e[2]]
    s, key, is_group = e
    if is_group:
        return [(key + ns, s, key + suf) for suf, ns in ((".weight", ".weight"), (".scales", ".weight.scale"),
                                                         (".biases", ".weight.bias")) if key + suf in s.h]
    return [(key, s, key)]


def small_blob_bytes(entries, metadata, rename=True):
    """entries: list of (src, key, is_group) -> (header bytes, list of (newname, bytes-getter))."""
    tensors = [t for e in entries for t in entry_tensors(e)]
    hdr = {"__metadata__": metadata} if metadata else {}
    off = 0
    for nk, s, k in tensors:
        v = s.h[k]
        n = s.nbytes(k)
        hdr[nk] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + n]}
        off += n
    hb = compact(hdr)
    hb += b" " * ((8 - (len(hb) + 8) % 8) % 8)
    return hb, tensors


def write_small_blob(blobs_dir, entries, metadata):
    hb, tensors = small_blob_bytes(entries, metadata)
    fd, tmp = tempfile.mkstemp(dir=blobs_dir, prefix="sha256-")
    h = hashlib.sha256()
    size = 0
    with os.fdopen(fd, "wb") as f:
        for chunk in (struct.pack("<Q", len(hb)), hb):
            f.write(chunk); h.update(chunk); size += len(chunk)
        for nk, s, k in tensors:
            b = s.read(k)
            f.write(b); h.update(b); size += len(b)
    dig = h.hexdigest()
    dst = os.path.join(blobs_dir, "sha256-" + dig)
    os.chmod(tmp, 0o644)
    os.replace(tmp, dst)
    return {"mediaType": "application/vnd.ollama.image.tensor", "digest": "sha256:" + dig, "size": size}


def write_big_blob(blobs_dir, bp):
    s = bp["src"]
    tmp = os.path.join(blobs_dir, ".zc-" + os.path.basename(s.path))
    if os.path.exists(tmp):
        os.remove(tmp)
    subprocess.run(["/bin/cp", "-c", s.path, tmp], check=True)
    hb = bp["header"] + b" " * (s.n - len(bp["header"]))
    assert len(hb) == s.n
    fd = os.open(tmp, os.O_WRONLY)
    try:
        os.pwrite(fd, struct.pack("<Q", s.n) + hb, 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    dig = sha256_file(tmp)
    dst = os.path.join(blobs_dir, "sha256-" + dig)
    os.chmod(tmp, 0o644)
    os.replace(tmp, dst)
    return {"mediaType": "application/vnd.ollama.image.tensor", "digest": "sha256:" + dig,
            "size": os.path.getsize(dst), "name": bp["name"]}


def norm_source(work, norms, unshift=True):
    """model.safetensors of the (un-shifted) norms (original names) for the real `ollama create`.
    Without mtp.* tensors Ollama does not add +1, so the norms are then stored as they are."""
    hdr, off, datas = {}, 0, []
    exact_bad = 0
    for s, k in norms:
        u = np.frombuffer(s.read(k), dtype=np.uint16)
        un = f32_to_bf16(bf16_to_f32(u) - np.float32(1.0)) if unshift else u
        back = f32_to_bf16(bf16_to_f32(un) + np.float32(1.0)) if unshift else u
        exact_bad += int((back != u).sum())
        b = un.tobytes()
        hdr[k] = {"dtype": "BF16", "shape": s.h[k]["shape"], "data_offsets": [off, off + len(b)]}
        off += len(b)
        datas.append(b)
    hb = compact(hdr)
    hb += b" " * ((8 - (len(hb) + 8) % 8) % 8)
    with open(os.path.join(work, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)) + hb)
        for b in datas:
            f.write(b)
    return exact_bad


def split_name(name):
    m = re.fullmatch(r"(?:([^/]+)/)?([^:/]+)(?::(.+))?", name)
    ns, model, tag = m.group(1) or "library", m.group(2), m.group(3) or "latest"
    return ns, model, tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True, help="e.g. nace/qwen3.6:35b-a3b-mlx-4bit-mtp")
    ap.add_argument("--models", required=True, help="OLLAMA_MODELS dir to write into")
    ap.add_argument("--work", required=True)
    ap.add_argument("--mtp-quant", default="4:64", help="bits:group of the MTP sidecar's quantized tensors")
    ap.add_argument("--no-mtp", action="store_true")
    ap.add_argument("--renderer", required=True)
    ap.add_argument("--parser", required=True)
    ap.add_argument("--requires", default=None)
    ap.add_argument("--params", default="{}", help="JSON of PARAMETERs (mirror the official tag)")
    ap.add_argument("--host", default="127.0.0.1:8193", help="the dedicated `ollama serve` (OLLAMA_MODELS=--models) to create the model on")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    P = plan(args)
    print(f"plan for {args.src}:")
    for bp in P["blobs"]:
        s = bp["src"]
        moved_bytes = sum(s.nbytes(k) for k in bp["moved"])
        print(f"  {os.path.basename(s.path)}: header {len(bp['header'])}/{s.n} B, {bp['tensors']} tensors kept, "
              f"{len(bp['moved'])} moved out ({moved_bytes/1e6:.1f} MB), default quant int{bp['default'][0]}/g{bp['default'][1]}")
    print(f"  norms un-shifted: {len(P['norms'])} tensors ({sum(s.nbytes(k) for s, k in P['norms'])/1e6:.2f} MB)")
    for bq, ents in P["small"].items():
        nb = sum(s.nbytes(k) for e in ents for _, s, k in entry_tensors(e))
        print(f"  small blob {bq}: {len(ents)} units, {nb/1e6:.1f} MB copied")
    if args.dry_run:
        return

    models = os.path.abspath(args.models)
    blobs_dir = os.path.join(models, "blobs")
    os.makedirs(blobs_dir, exist_ok=True)
    work = os.path.abspath(args.work)
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    # 1) real `ollama create` for the un-shifted norms + config JSONs + params
    bad = norm_source(work, P["norms"], unshift=P["has_mtp"])
    print(f"norm un-shift round trip mismatches (elements): {bad}")
    for fn in CONFIG_JSONS + ("chat_template.jinja",):
        p = os.path.join(args.src, fn)
        if os.path.exists(p):
            os.symlink(os.path.realpath(p), os.path.join(work, fn))
    mf = [f"FROM {work}", f"RENDERER {args.renderer}", f"PARSER {args.parser}"]
    if args.requires:
        mf.append(f"REQUIRES {args.requires}")
    for k, v in json.loads(args.params).items():
        mf.append(f"PARAMETER {k} {v}")
    open(os.path.join(work, "Modelfile"), "w").write("\n".join(mf) + "\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMA_")}
    env.update(OLLAMA_MODELS=models, OLLAMA_HOST=args.host, OLLAMA_NOPRUNE="1")
    t0 = time.time()
    r = subprocess.run([OLLAMA, "create", args.name, "-f", os.path.join(work, "Modelfile")], env=env,
                       capture_output=True, text=True)
    print("ollama create:", r.returncode, (r.stdout + r.stderr).strip().splitlines()[-3:], f"{time.time()-t0:.1f}s")
    if r.returncode != 0:
        raise SystemExit(r.stdout + r.stderr)
    ns, model, tag = split_name(args.name)
    # manifest.V2PathForName canonicalises the registry host (registry.ollama.ai -> ollama.com)
    cands = [os.path.join(models, "manifests-v2", h, ns, model, tag) for h in os.listdir(os.path.join(models, "manifests-v2"))]
    cands = [c for c in cands if os.path.lexists(c)]
    assert len(cands) == 1, cands
    v2 = cands[0]
    man_text = open(v2).read()
    man = json.loads(man_text)
    # 2) big blobs (clones) and 3) small blobs
    new_layers = []
    for bp in P["blobs"]:
        t0 = time.time()
        L = write_big_blob(blobs_dir, bp)
        print(f"  big blob {bp['name']}: {L['digest'][:19]}… {L['size']/1e9:.2f} GB ({time.time()-t0:.1f}s)")
        new_layers.append(L)
    for bq, ents in P["small"].items():
        meta = {"quant_type": "int%d" % bq[0], "group_size": str(bq[1])} if bq else None
        L = write_small_blob(blobs_dir, ents, meta)
        L["name"] = "nace-zc/small-" + ("int%d-g%d" % bq if bq else "plain")
        print(f"  small blob {L['name']}: {L['digest'][:19]}… {L['size']/1e6:.1f} MB")
        new_layers.append(L)
    # MTP big blob last among the tensor layers: Root.Open takes the model-wide quant type from the first
    # tensor layer that has blob metadata, which must be a main-body shard.
    new_layers.sort(key=lambda L: ("mtp" in L.get("name", ""), "small" in L.get("name", "")))
    man["layers"] = new_layers + man["layers"]
    data = (json.dumps(man, separators=(",", ":")) + "\n").encode()
    dig = hashlib.sha256(data).hexdigest()
    mblob = os.path.join(blobs_dir, "sha256-" + dig)
    with open(mblob, "wb") as f:
        f.write(data)
    os.chmod(mblob, 0o644)
    tmp_link = v2 + ".zc-tmp"
    if os.path.lexists(tmp_link):
        os.remove(tmp_link)
    os.symlink(os.path.relpath(mblob, os.path.dirname(v2)), tmp_link)
    os.replace(tmp_link, v2)
    json.dump({"name": args.name, "src": args.src, "manifest_digest": "sha256:" + dig,
               "layers_added": new_layers, "norm_roundtrip_mismatches": bad,
               "built": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(models, f"zc-{model}-{tag}.json"), "w"), indent=1)
    print(f"manifest sha256:{dig} -> {v2}")


if __name__ == "__main__":
    main()
