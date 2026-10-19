#!/usr/bin/env python3
"""A mixed-precision pack from a local MLX pack plus some modules of another precision's published pack.

  mixed_pack.py <base pack dir> <out dir> --from <repo>[@rev] --modules <spec> [--from ... --modules ...]
                [--conns 16] [--dry-run]

Every file of the base pack is an APFS clone in <out dir> (no disk space), the selected modules' tensors (weight,
scales, biases) are read from the other pack by HTTP range requests (no full download) into one more shard,
model-zz-mixed.safetensors, and config.json gets MLX per-module quantization entries for them, so Kapitsa, mlx-lm and
eval_stream.py's repo: candidates read the mix. The base shards keep their own copies of those modules: Kapitsa's
loader reads the files in name order and the later shard's tensors replace them (the shadowed copies only cost
memory: fine for measurements, not a pack to publish).

<spec>: comma list of embed, head, or L<a>[-<b>][.<role>] with role mix (attention / DeltaNet projections), mlp,
down (down_proj), gu (gate_proj and up_proj); e.g. head,L0-1,L62-63.mix. Stdlib only.
"""
import json, os, re, shutil, struct, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "kapitsa-quant-eval"}


def get(url, rng=None, tries=8):
    h = dict(UA)
    if rng:
        h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=300) as r:
                d = r.read()
            if rng and len(d) != rng[1] - rng[0]:
                raise IOError("short read")
            return d
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))


class Remote:
    def __init__(self, ref):
        self.repo, _, rev = ref.partition("@")
        meta = json.loads(get(f"https://huggingface.co/api/models/{self.repo}/revision/{rev or 'main'}"))
        self.rev = meta["sha"]
        self.config = json.loads(get(self.url("config.json")))
        self.index = {}
        for f in sorted(s["rfilename"] for s in meta["siblings"]):
            if not f.endswith(".safetensors") or "/" in f:
                continue
            n = struct.unpack("<Q", get(self.url(f), (0, 8)))[0]
            hdr = json.loads(get(self.url(f), (8, 8 + n)))
            for k, v in hdr.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.index[k] = (f, 8 + n + a, b - a, v["dtype"], v["shape"])

    def url(self, f):
        return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{f}"


def module_of(name):
    """'language_model.model.layers.3.mlp.down_proj.weight' -> ('language_model.model.layers.3.mlp.down_proj', 3, 'mlp.down_proj')"""
    base = name.rsplit(".", 1)[0]
    m = re.search(r"\.layers\.(\d+)\.(.*)$", base)
    if m:
        return base, int(m.group(1)), m.group(2)
    return base, None, base


def selected(spec, layer, path):
    for s in spec.split(","):
        if s in ("embed", "head"):
            if layer is None and (("embed_tokens" in path) if s == "embed" else ("lm_head" in path)):
                return True
            continue
        m = re.fullmatch(r"L(\d+)(?:-(\d+))?(?:\.(\w+))?", s)
        if not m:
            sys.exit(f"bad module spec {s}")
        a, b, role = int(m.group(1)), int(m.group(2) or m.group(1)), m.group(3) or "all"
        if layer is None or not a <= layer <= b:
            continue
        if role == "all" or (role == "mix" and path.startswith(("self_attn.", "linear_attn."))) \
                or (role == "mlp" and path.startswith("mlp.")) or (role == "down" and path.endswith("down_proj")) \
                or (role == "gu" and path.endswith(("gate_proj", "up_proj"))):
            return True
    return False


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    src, out = args[0], args[1]
    rest = args[2:]
    pairs, conns, dry = [], 16, False
    i = 0
    while i < len(rest):
        if rest[i] == "--from":
            pairs.append([rest[i + 1], None])
            i += 2
        elif rest[i] == "--modules":
            pairs[-1][1] = rest[i + 1]
            i += 2
        elif rest[i] == "--conns":
            conns = int(rest[i + 1])
            i += 2
        elif rest[i] == "--dry-run":
            dry = True
            i += 1
        else:
            sys.exit(f"unknown argument {rest[i]}")
    cfg = json.load(open(os.path.join(src, "config.json")))
    qbase = cfg.get("quantization") or {}
    jobs = []  # (name, remote, bytes)
    entries = {}
    for ref, spec in pairs:
        r = Remote(ref)
        rq = r.config.get("quantization") or {}
        bits, gs = rq.get("bits"), rq.get("group_size", 64)
        mods = {}
        for k in r.index:
            if "vision_tower" in k or k.startswith("mtp."):
                continue
            mod, layer, path = module_of(k)
            if k.endswith((".weight", ".scales", ".biases")) and selected(spec, layer, path):
                mods.setdefault(mod, []).append(k)
        for mod, names in sorted(mods.items()):
            if not any(n.endswith(".scales") for n in names):
                continue  # a norm or other float tensor: not a quantized module
            e = rq.get(mod) if isinstance(rq.get(mod), dict) else {"bits": bits, "group_size": gs}
            entries[mod] = {"group_size": e.get("group_size", gs), "bits": e.get("bits", bits)}
            for n in names:
                jobs.append((n, r, r.index[n][2]))
        print(f"{r.repo}@{r.rev[:8]} ({bits}-bit): {len(mods)} modules from {spec}")
    total = sum(j[2] for j in jobs)
    print(f"{len(entries)} modules, {len(jobs)} tensors, {total / 1e9:.2f} GB to fetch")
    if dry:
        for m, e in sorted(entries.items())[:12]:
            print(" ", m, e)
        return
    free = shutil.disk_usage(os.path.dirname(os.path.abspath(out)) or ".").free
    if free - total < 6e9:
        sys.exit(f"would leave {(free - total) / 1e9:.1f} GB free on the disk")
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(src)):  # APFS clones of the base pack (cp -c), config.json rewritten below
        if f in (".kapitsa", "config.json"):
            continue
        subprocess.run(["cp", "-cR", os.path.join(src, f), os.path.join(out, f)], check=True)
    # the override shard: tensors in name order, 256-byte aligned offsets are not needed by the format
    hdr, off = {}, 0
    for n, r, nb in sorted(jobs, key=lambda j: j[0]):
        f, o, nb, dt, shape = r.index[n]
        hdr[n] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    hdr["__metadata__"] = {"format": "mlx"}
    hj = json.dumps(hdr, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    path = os.path.join(out, "model-zz-mixed.safetensors")
    t0 = time.time()
    with open(path + ".part", "wb") as fh:
        fh.write(struct.pack("<Q", len(hj)) + hj)
        base = fh.tell()
        fh.truncate(base + off)
    pieces = []
    for n, r, nb in jobs:
        f, o, nb, dt, shape = r.index[n]
        for p in range(0, nb, 64 << 20):
            pieces.append((r, f, o + p, o + min(nb, p + (64 << 20)), base + hdr[n]["data_offsets"][0] + p))

    def work(pc):
        r, f, a, b, dst = pc
        d = get(r.url(f), (a, b))
        with open(path + ".part", "r+b") as fh:
            fh.seek(dst)
            fh.write(d)

    with ThreadPoolExecutor(conns) as ex:
        list(ex.map(work, pieces))
    os.replace(path + ".part", path)
    print(f"fetched {total / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    qc = dict(qbase)
    qc.update(entries)
    cfg["quantization"] = qc
    if "quantization_config" in cfg:
        cfg["quantization_config"] = dict(qc)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    idx = os.path.join(out, "model.safetensors.index.json")
    if os.path.exists(idx):  # mlx-lm reads the index: point the promoted tensors at the new shard
        os.remove(idx)  # a clone: removing it leaves the base pack's file alone
        j = json.load(open(os.path.join(src, "model.safetensors.index.json")))
        for n, _, _ in jobs:
            j["weight_map"][n] = "model-zz-mixed.safetensors"
        json.dump(j, open(idx, "w"), indent=2)
    print(f"wrote {out}: {len(entries)} modules from the other packs")


if __name__ == "__main__":
    main()
