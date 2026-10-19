#!/usr/bin/env python3
"""A mixed-precision pack from a local MLX pack plus some modules of other precisions' packs, without copying weights.

  mixed_pack.py <base pack dir> <out dir> --from <pack dir | repo[@rev]> --modules <spec> [--from ... --modules ...]
                [--conns 16] [--dry-run]

The base pack's files become APFS clones in <out dir> (cp -c: no disk space), and each cloned shard's header drops
the tensors of the selected modules. A header keeps its length (padded with spaces, as the safetensors format
allows), so every other tensor keeps its offset; the dropped tensors' bytes stay in the file, unreferenced. The
selected modules come from the other packs: a local pack's shards are cloned the same way with headers that keep
only the selected tensors (no copy); from a Hugging Face repo they are fetched by range requests into one new shard.
config.json gets MLX per-module quantization entries (bits, group size, mode). Every tensor is then listed exactly
once, so no loader depends on the order of the files. Kapitsa, MLX and eval_stream.py read such shards; strict
safetensors readers (the data must be contiguous) do not: a pack to measure, not to publish.

<spec>: comma list of embed, head, or L<a>[-<b>][.<role>] with role mix (attention / DeltaNet projections), mlp,
down (down_proj), gu (gate_proj and up_proj); e.g. head,L0-1,L62-63.mix. Two --from groups may not select the same
module, and every selector must select something. Stdlib only.
"""
import argparse, json, os, re, shutil, struct, subprocess, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "kapitsa-quant-eval"}
ROLES = ("all", "mix", "mlp", "down", "gu")


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


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return n, json.loads(fh.read(n))


def shards(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))


class Pack:
    """A pack's tensors: name -> (file, header length, header entry); local folder or Hugging Face repo."""

    def __init__(self, ref):
        self.local = os.path.isdir(ref)
        self.index = {}
        if self.local:
            self.dir = os.path.realpath(ref)
            self.name = self.dir
            self.config = json.load(open(os.path.join(self.dir, "config.json")))
            for f in shards(self.dir):
                n, hdr = read_header(os.path.join(self.dir, f))
                self._add(f, n, hdr)
        else:
            self.repo, _, rev = ref.partition("@")
            meta = json.loads(get(f"https://huggingface.co/api/models/{self.repo}/revision/{rev or 'main'}"))
            self.rev = meta["sha"]
            self.name = f"{self.repo}@{self.rev[:8]}"
            self.config = json.loads(get(self.url("config.json")))
            for f in sorted(s["rfilename"] for s in meta["siblings"]):
                if f.endswith(".safetensors") and "/" not in f:
                    n = struct.unpack("<Q", get(self.url(f), (0, 8)))[0]
                    self._add(f, n, json.loads(get(self.url(f), (8, 8 + n))))

    def _add(self, f, n, hdr):
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            if k in self.index:
                sys.exit(f"{self.name}: tensor {k} is in both {self.index[k][0]} and {f}")
            self.index[k] = (f, n, v)

    def url(self, f):
        return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{f}"

    def quant(self, module):
        qc = self.config.get("quantization") or {}
        e = qc.get(module) if isinstance(qc.get(module), dict) else {}
        return {"group_size": e.get("group_size", qc.get("group_size", 64)), "bits": e.get("bits", qc.get("bits")),
                "mode": e.get("mode", qc.get("mode", "affine"))}


def parse_spec(spec, n_layers):
    out = []
    for s in spec.split(","):
        if s in ("embed", "head"):
            out.append((s, None, None))
            continue
        m = re.fullmatch(r"L(\d+)(?:-(\d+))?(?:\.(\w+))?", s)
        if not m:
            sys.exit(f"bad module selector {s!r}: embed, head or L<a>[-<b>][.mix|.mlp|.down|.gu]")
        a, b, role = int(m.group(1)), int(m.group(2) or m.group(1)), m.group(3) or "all"
        if a > b or b >= n_layers or role not in ROLES:
            sys.exit(f"bad module selector {s!r}: layers 0-{n_layers - 1}, a <= b, roles {', '.join(ROLES)}")
        out.append((s, (a, b), role))
    return out


def selects(sel, module):
    """Does a selector take this module (full name without .weight)?"""
    s, layers, role = sel
    m = re.search(r"\.layers\.(\d+)\.(.*)$", module)
    if layers is None:
        return m is None and (("embed_tokens" in module) if s == "embed" else ("lm_head" in module))
    if not m or not layers[0] <= int(m.group(1)) <= layers[1]:
        return False
    path = m.group(2)
    return role == "all" or (role == "mix" and path.startswith(("self_attn.", "linear_attn."))) or \
        (role == "mlp" and path.startswith("mlp.")) or (role == "down" and path.endswith("down_proj")) or \
        (role == "gu" and path.endswith(("gate_proj", "up_proj")))


def quantized_modules(index):
    """module -> its tensor names, for every quantized linear / embedding (a .scales tensor) outside vision and MTP"""
    mods = {}
    for k in index:
        if "vision_tower" in k or k.startswith("mtp.") or not k.endswith(".scales"):
            continue
        m = k[: -len(".scales")]
        mods[m] = [m + s for s in (".weight", ".scales", ".biases") if m + s in index]
    return mods


def clone(src, dst):
    """APFS clone of src's real file at dst (refuses a copy across volumes, which would take the full size)."""
    real = os.path.realpath(src)
    if os.stat(real).st_dev != os.stat(os.path.dirname(dst)).st_dev:
        sys.exit(f"{real} is on another volume than {dst}: cp -c would copy it")
    subprocess.run(["cp", "-c", real, dst], check=True)
    if os.stat(dst).st_ino == os.stat(real).st_ino:
        sys.exit(f"{dst} is not a separate file")


def rewrite_header(path, n, keep):
    """Rewrite a cloned shard's header in place keeping only `keep` (name -> entry): same length, space-padded."""
    hj = json.dumps(keep, separators=(",", ":")).encode()
    if len(hj) > n:
        sys.exit(f"{path}: the new header is longer than the old one")
    with open(path, "r+b") as fh:
        fh.seek(8)
        fh.write(hj + b" " * (n - len(hj)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--from", dest="srcs", action="append", required=True, help="local pack dir or repo[@rev]")
    ap.add_argument("--modules", action="append", required=True, help="selector list for the preceding --from")
    ap.add_argument("--conns", type=int, default=16)
    ap.add_argument("--min-free-gb", type=float, default=6.0, help="refuse a fetch that leaves less free disk")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if len(a.srcs) != len(a.modules):
        sys.exit("give one --modules per --from")
    base_dir = os.path.realpath(a.base)
    cfg = json.load(open(os.path.join(base_dir, "config.json")))
    n_layers = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    base = Pack(base_dir)
    base_mods = quantized_modules(base.index)

    taken = {}  # module -> (pack, its tensor names, quantization)
    for ref, spec in zip(a.srcs, a.modules):
        p = Pack(ref)
        mods = quantized_modules(p.index)
        for sel in parse_spec(spec, n_layers):
            hit = [m for m in base_mods if selects(sel, m)]
            if not hit:
                sys.exit(f"{sel[0]} selects no quantized module of {a.base}")
            for m in hit:
                if m not in mods:
                    sys.exit(f"{p.name} has no quantized {m} (bf16 or missing): only quantized modules can be taken")
                if m in taken and taken[m][0] is not p:
                    sys.exit(f"{m} is selected from both {taken[m][0].name} and {p.name}")
                taken[m] = (p, mods[m], p.quant(m))
        print(f"{p.name}: {sum(1 for t in taken.values() if t[0] is p)} modules from {spec}")
    nbytes = sum(t[0].index[k][2]["data_offsets"][1] - t[0].index[k][2]["data_offsets"][0]
                 for t in taken.values() for k in t[1])
    remote = sum(1 for t in taken.values() if not t[0].local)
    print(f"{len(taken)} modules, {nbytes / 1e9:.2f} GB of tensors ({'some fetched' if remote else 'all cloned'})")
    if a.dry_run:
        for m, (p, names, q) in sorted(taken.items())[:12]:
            print(" ", m, q)
        return

    os.makedirs(a.out, exist_ok=True)
    out = os.path.realpath(a.out)
    if os.listdir(out):
        sys.exit(f"{out} is not empty")
    drop = {k for m in taken for k in base_mods[m]}  # the base's own tensors of the taken modules
    # the base pack: every file cloned; shards without the taken modules' tensors
    for f in sorted(os.listdir(base_dir)):
        if f in (".kapitsa", "config.json", "model.safetensors.index.json"):
            continue
        src, dst = os.path.join(base_dir, f), os.path.join(out, f)
        if os.path.isdir(src):  # e.g. the DFlash2 drafter: cloned, links followed, never copied across volumes
            if os.stat(os.path.realpath(src)).st_dev != os.stat(out).st_dev:
                sys.exit(f"{os.path.realpath(src)} is on another volume than {out}: cp -c would copy it")
            subprocess.run(["cp", "-cRL", os.path.realpath(src), dst], check=True)
            continue
        clone(src, dst)
        if f.endswith(".safetensors"):
            n, hdr = read_header(dst)
            if drop & set(hdr):
                rewrite_header(dst, n, {k: v for k, v in hdr.items() if k not in drop})
    # the taken modules: local packs' shards cloned with only those tensors; remote ones fetched into one shard
    for gi, p in enumerate(dict.fromkeys(t[0] for t in taken.values())):
        names = sorted(k for t in taken.values() if t[0] is p for k in t[1])
        if p.local:
            for f in sorted({p.index[k][0] for k in names}):
                dst = os.path.join(out, f"model-mix{gi}-{f}")
                clone(os.path.join(p.dir, f), dst)
                n, hdr = read_header(dst)
                keep = {k: v for k, v in hdr.items() if k in names}
                if "__metadata__" in hdr:
                    keep["__metadata__"] = hdr["__metadata__"]
                rewrite_header(dst, n, keep)
            continue
        size = sum(p.index[k][2]["data_offsets"][1] - p.index[k][2]["data_offsets"][0] for k in names)
        if shutil.disk_usage(out).free - size < a.min_free_gb * 1e9:
            sys.exit(f"fetching {size / 1e9:.2f} GB would leave less than {a.min_free_gb:g} GB free")
        hdr, off = {}, 0
        for k in names:
            e = p.index[k][2]
            nb = e["data_offsets"][1] - e["data_offsets"][0]
            hdr[k] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [off, off + nb]}
            off += nb
        hdr["__metadata__"] = {"format": "mlx"}
        hj = json.dumps(hdr, separators=(",", ":")).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        path = os.path.join(out, f"model-mix{gi}-fetched.safetensors")
        with open(path + ".part", "wb") as fh:
            fh.write(struct.pack("<Q", len(hj)) + hj)
            fh.truncate(8 + len(hj) + off)
        base_off = 8 + len(hj)
        pieces = []
        for k in names:
            f, n, e = p.index[k]
            a0, a1 = 8 + n + e["data_offsets"][0], 8 + n + e["data_offsets"][1]
            for q in range(a0, a1, 64 << 20):
                pieces.append((f, q, min(a1, q + (64 << 20)), base_off + hdr[k]["data_offsets"][0] + q - a0))

        def work(pc, path=path, p=p):
            f, s, e, dst = pc
            d = get(p.url(f), (s, e))
            with open(path + ".part", "r+b") as fh:
                fh.seek(dst)
                fh.write(d)

        t0 = time.time()
        with ThreadPoolExecutor(a.conns) as ex:
            list(ex.map(work, pieces))
        os.replace(path + ".part", path)
        print(f"fetched {size / 1e9:.2f} GB from {p.name} in {time.time() - t0:.0f} s")
    # config: per-module quantization entries; index: every tensor's shard
    qc = dict(cfg.get("quantization") or {})
    for m, (p, names, q) in taken.items():
        qc[m] = dict(q)
    cfg["quantization"] = qc
    if "quantization_config" in cfg:
        cfg["quantization_config"] = dict(qc)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    wm, seen = {}, {}
    for f in shards(out):
        n, hdr = read_header(os.path.join(out, f))
        for k in hdr:
            if k != "__metadata__":
                if k in seen:
                    sys.exit(f"{k} is in both {seen[k]} and {f}")
                seen[k] = wm[k] = f
    for m in taken:
        if taken[m][1][0] not in wm:
            sys.exit(f"{m}: lost")
    if os.path.exists(os.path.join(base_dir, "model.safetensors.index.json")):
        j = json.load(open(os.path.join(base_dir, "model.safetensors.index.json")))
        j["weight_map"] = {k: v for k, v in sorted(wm.items()) if not k.startswith("mtp.")}
        json.dump(j, open(os.path.join(out, "model.safetensors.index.json"), "w"), indent=2)
    print(f"wrote {out}: {len(taken)} modules from {', '.join(dict.fromkeys(t[0].name for t in taken.values()))}")


if __name__ == "__main__":
    main()
