#!/usr/bin/env python3
"""A Qwen3.8-Flash-Next (qwen4_exp) pack with Q2_0 routed experts for Kapitsa (src/model/moe_q20.h): the experts of
ISTA-DASLab's GSQ-RCO Q2_0 GGUF, every other tensor, the MTP head and the PLE n-gram table from Nace AI's MLX 4-bit pack.
Either source can be a local file / folder or Hugging Face (HTTP range reads): only the GGUF's expert tensors and the
pack's non-expert tensors are fetched, straight into the output files.

  qwen4_exp_q20.py convert --out DIR [--pack DIR | --pack-repo REPO@REV] [--gguf FILE | --gguf-repo REPO@REV:PATH]
                           [--table full|link|sparse] [--rows-ids @f ...] [--stream] [--verify N] [--conns 16]
  qwen4_exp_q20.py check DIR    the output's experts against its GGUF again (bytes) and its config
  qwen4_exp_q20.py rows DIR @f ... [--stream]   adds the token sequences' rows to DIR's sparse table (e.g. a
                                     generated continuation, so a speed run reads the rows the model would)

Q2_0 (ggml type 42 in ISTA-DASLab's / Strata's ggml): blocks of 64 weights in 18 bytes, an fp16 d and 16 bytes of 2-bit
codes, LSB first, weight = (code - 1) d. GGUF expert tensors are [E][N][K] row-major (ne = K, N, E), as MLX's
switch_mlp.{gate,up,down}_proj. Each becomes, under the pack's name:
  .weight  U32 [E, N, K/16]   the blocks' code bytes in order (4 little-endian words a block: MLX's 2-bit order)
  .scales  F16 [E, N, K/64]   the blocks' d
with no .biases, and config.json's quantization (and quantization_config) gets {"bits": 2, "group_size": 64,
"mode": "q2_0"} for the module. Only Kapitsa reads such a pack: MLX, mlx-vlm and oMLX have no q2_0 mode.
Everything else keeps the source pack's files and names; model-000NN files hold the same tensors as the source's.
--verify N: N expert rows of every converted tensor dequantized from both sources must correlate (> 0.6; ~0.9 for two
quantizations of the same weights, ~0 for another expert, a gate/up swap or a transposition).
--table: full (ple.safetensors copied: 32 GB), link (an APFS clone or hard link of a local pack's), sparse (the file at
full size, holes but for the rows of the --rows-ids token sequences: teacher-forced evaluation of those sequences is
exact, other rows read as zero).
Resumable: finished files are listed in DIR/.q20.done.json.
"""
import argparse
import bisect
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_exp_pack as P  # noqa: E402

PACK_REPO = "nace-ai/Qwen3.8-Flash-Next-MLX-4bit@5d423f9f563c1f1b7ca7945b268e65c0b92b0eae"
GGUF_REPO = ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f92082b1e93c0e96d60a8b11aab089b52f09:"
             "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf")
Q2_0, Q2_0_BYTES = 42, 18
EXPERT = "language_model.model.layers.{}.mlp.switch_mlp.{}_proj"
GGUF_EXPERT = {"gate": "blk.{}.ffn_gate_exps.weight", "up": "blk.{}.ffn_up_exps.weight",
               "down": "blk.{}.ffn_down_exps.weight"}
SMALL = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
         "generation_config.json", "preprocessor_config.json", "video_preprocessor_config.json", "ple_manifest.json",
         "LICENSE"]
TABLE = "ple.safetensors"
log = P.log


# ---------------------------------------------------------------- GGUF

class _Short(Exception):
    pass


class _Reader:
    def __init__(self, b):
        self.b, self.p = b, 0

    def raw(self, n):
        if self.p + n > len(self.b):
            raise _Short()
        v = self.b[self.p:self.p + n]
        self.p += n
        return v

    def fmt(self, f):
        return struct.unpack("<" + f, self.raw(struct.calcsize("<" + f)))[0]

    def string(self):
        return self.raw(self.fmt("Q")).decode("utf-8", "replace")

    def value(self, t, keep=True):
        scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in scalar:
            return self.fmt(scalar[t])
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.fmt("I"), self.fmt("Q")
            vals = [self.value(et, keep and n <= 64) for _ in range(n)]
            return vals if keep and n <= 64 else f"<array of {n}>"
        raise ValueError(f"GGUF value type {t}")


class Gguf:
    """A GGUF file's metadata and tensor table; read(a, b) returns its bytes [a, b)."""

    def __init__(self, read, name):
        self.read, self.name = read, name
        need = 8 << 20
        while True:
            try:
                self._parse(read(0, need))
                return
            except _Short:
                need *= 2
                if need > 1 << 30:
                    raise ValueError(f"{name}: no GGUF header in the first GB")

    def _parse(self, b):
        r = _Reader(b)
        if r.raw(4) != b"GGUF":
            raise ValueError(f"{self.name} is not a GGUF file")
        self.version = r.fmt("I")
        n_tensors, n_kv = r.fmt("Q"), r.fmt("Q")
        self.kv = {}
        for _ in range(n_kv):
            k = r.string()
            self.kv[k] = r.value(r.fmt("I"))
        self.tensors = {}
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.fmt("Q") for _ in range(r.fmt("I"))]
            typ, off = r.fmt("I"), r.fmt("Q")
            self.tensors[name] = (dims, typ, off)
        align = int(self.kv.get("general.alignment", 32))
        self.data0 = (r.p + align - 1) // align * align

    def expert(self, name):
        """(E, N, K, absolute offset, bytes) of a Q2_0 expert tensor."""
        if name not in self.tensors:
            raise KeyError(f"{self.name}: no tensor {name}")
        (K, N, E), typ, off = self.tensors[name]
        if typ != Q2_0:
            raise ValueError(f"{name}: ggml type {typ}, not Q2_0 ({Q2_0})")
        if K % 64:
            raise ValueError(f"{name}: rows of {K} are not whole 64-blocks")
        return E, N, K, self.data0 + off, E * N * (K // 64) * Q2_0_BYTES


def q20_planes(raw, E, N, K):
    """GGUF Q2_0 bytes of an [E, N, K] tensor -> (codes uint32 [E, N, K/16], d float16 [E, N, K/64])."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(E, N, K // 64, Q2_0_BYTES)
    d = np.ascontiguousarray(b[..., :2]).view("<f2").reshape(E, N, K // 64)
    codes = np.ascontiguousarray(b[..., 2:]).reshape(E, N, K // 4).view("<u4")
    return codes, d


def q20_dequant_blocks(raw, E, N, K):
    """The GGUF's own reading of Q2_0 (the reference): float32 [E, N, K], code j of a block at bits 2(j % 4) of byte
    j / 4 of its 16 code bytes."""
    b = np.frombuffer(raw, dtype=np.uint8).reshape(E, N, K // 64, Q2_0_BYTES)
    d = b[..., :2].copy().view("<f2").astype(np.float32)  # [E, N, K/64, 1]
    q = b[..., 2:]
    codes = np.stack([(q >> (2 * j)) & 3 for j in range(4)], axis=-1).reshape(E, N, K // 64, 64)
    return ((codes.astype(np.float32) - 1.0) * d).reshape(E, N, K)


def q20_dequant_planes(codes, d):
    """Kapitsa's reading of the planes (zq2moe.metal): code j of a row at bits 2(j % 16) of word j / 16, scale d and
    bias -d per 64 -> float32 [..., K]."""
    w = np.asarray(codes, dtype=np.uint32)
    s = np.asarray(d).astype(np.float32)
    q = np.stack([(w >> (2 * j)) & 3 for j in range(16)], axis=-1).reshape(w.shape[:-1] + (w.shape[-1] * 16,))
    return (q.astype(np.float32) - 1.0) * np.repeat(s, 64, axis=-1)


# ---------------------------------------------------------------- sources

def parse_repo(spec, default):
    spec = spec or default
    repo, _, rest = spec.partition("@")
    rev, _, path = rest.partition(":")
    return repo, rev or "main", path


class PackSource:
    """Nace's MLX pack: a local folder (P.Local) or a Hugging Face repo (P.Remote)."""

    def __init__(self, a, pool):
        self.pool = pool
        if a.pack:
            self.local, self.src = True, P.Local(a.pack)
            self.where = os.path.abspath(a.pack)
        else:
            repo, rev, _ = parse_repo(a.pack_repo, PACK_REPO)
            self.local, self.src = False, P.Remote(repo, rev)
            self.where = f"{repo}@{self.src.rev}"
        self.index = self.src.index

    def small(self, f):
        if self.local:
            p = os.path.join(self.src.path, f)
            return open(p, "rb").read() if os.path.exists(p) else None
        return self.src.read(f) if f in self.src.siblings else None

    def tensor_bytes(self, names):
        if self.local:
            return {k: np.array(self.src.array(k)).tobytes() for k in names}
        out = self.src.fetch(names, self.pool)
        return {k: v.tobytes() for k, v in out.items()}

    def rows(self, f, ranges):
        """[(a, b)] byte ranges of file f -> [bytes] (small ranges: row gathers)."""
        if self.local:
            fd = os.open(os.path.join(self.src.path, f), os.O_RDONLY)
            try:
                return [os.pread(fd, b - a, a) for a, b in ranges]
            finally:
                os.close(fd)
        return list(self.pool.map(lambda ab: self.src.read_small(f, ab[0], ab[1]), ranges))

    def copy_file(self, f, dst, piece=64 << 20):
        """A whole file (mtp.safetensors, the table): an APFS clone of a local one, else parallel ranged reads."""
        if self.local:
            src = os.path.join(self.src.path, f)
            if subprocess.call(["cp", "-c", src, dst], stderr=subprocess.DEVNULL) != 0:
                shutil.copyfile(src, dst)
            return
        hdr = self.src.read(f, 0, 8)
        n = struct.unpack("<Q", hdr)[0]
        header = json.loads(self.src.read(f, 8, 8 + n))
        size = 8 + n + max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        fd = os.open(dst, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, size)
            jobs = [(a, min(size, a + piece)) for a in range(0, size, piece)]

            def work(ab):
                os.pwrite(fd, self.src.read(f, ab[0], ab[1]), ab[0])
            list(self.pool.map(work, jobs))
        finally:
            os.close(fd)


class GgufSource:
    """The Q2_0 GGUF: a local file or a file of a Hugging Face repo (ranged reads)."""

    def __init__(self, a, pool):
        self.pool = pool
        if a.gguf:
            path = a.gguf
            fd = os.open(path, os.O_RDONLY)
            self.read = lambda x, y: os.pread(fd, y - x, x)
            self.where = os.path.abspath(path)
        else:
            repo, rev, path = parse_repo(a.gguf_repo, GGUF_REPO)
            r = P.Remote(repo, rev)
            self.read = lambda x, y: r.read(path, x, y)
            self.where = f"{repo}@{r.rev}:{path}"
        self.g = Gguf(self.read, self.where)

    def fetch(self, off, nbytes, piece=32 << 20):
        buf = bytearray(nbytes)
        jobs = [(p, min(nbytes, p + piece)) for p in range(0, nbytes, piece)]

        def work(ab):
            buf[ab[0]:ab[1]] = self.read(off + ab[0], off + ab[1])
        list(self.pool.map(work, jobs))
        return bytes(buf)


# ---------------------------------------------------------------- convert

def expert_of(name):
    """(layer, 'gate'|'up'|'down', 'weight'|'scales'|'biases') of a trunk expert tensor, else None."""
    pre = "language_model.model.layers."
    if not name.startswith(pre) or ".mlp.switch_mlp." not in name:
        return None
    layer = int(name[len(pre):].split(".")[0])
    rest = name.split(".mlp.switch_mlp.")[1]
    mat, _, part = rest.partition("_proj.")
    return (layer, mat, part) if mat in GGUF_EXPERT and part in ("weight", "scales", "biases") else None


def verify_rows(pack, layer, mat, codes, d, n, rng):
    """Correlation of n random rows of the converted matrix with the same rows of the 4-bit pack."""
    base = EXPERT.format(layer, mat)
    w_i, s_i, b_i = pack.index[base + ".weight"], pack.index[base + ".scales"], pack.index[base + ".biases"]
    E, N = codes.shape[0], codes.shape[1]
    wpr, gpr = w_i[4][-1], s_i[4][-1]
    bits = wpr * 32 // (gpr * 64)

    def row_of(t, row, width):  # one row of a stacked [E, N, width] tensor, from its own file
        return pack.rows(t[0], [(t[1] + row * width, t[1] + (row + 1) * width)])[0]
    worst = 1.0
    for _ in range(n):
        e, r = int(rng.integers(E)), int(rng.integers(N))
        row = e * N + r
        ref = P.dequantize_np(np.frombuffer(row_of(w_i, row, wpr * 4), "<u4")[None],
                              np.frombuffer(row_of(s_i, row, gpr * 2), "<u2")[None],
                              np.frombuffer(row_of(b_i, row, gpr * 2), "<u2")[None], bits, 64)[0]
        q2 = q20_dequant_planes(codes[e, r][None], d[e, r][None])[0]
        c = float(np.corrcoef(ref, q2)[0, 1])
        worst = min(worst, c)
    return worst


def sparse_rows(id_files):
    """The table rows (global) the token sequences read (P.ngram_ids)."""
    rows = set()
    for f in id_files:
        for line in open(f[1:] if f.startswith("@") else f):
            line = line.strip()
            if line:
                seq = [int(x) for x in line.replace(",", " ").split()]
                rows.update(int(x) for x in P.ngram_ids(seq).reshape(-1))
    return sorted(rows)


def write_sparse_table(pack, out_path, rows, add=False, stream=False, piece=16 << 20):
    """The pack's table file at full size with only `rows` written (APFS keeps the rest as holes). add: into the
    existing file, skipping rows already there (nonzero scales). stream: the rows are cut from whole reads of their
    shards' tensors (`piece`-byte ranges) instead of three small reads a row, for many rows from a remote pack (its CDN
    answers a few dozen small reads a second; a 4K prompt reads ~50K rows). Returns (file size, table rows, rows
    written)."""
    f = TABLE
    if pack.local:
        hb = open(os.path.join(pack.src.path, f), "rb").read(8)
        n = struct.unpack("<Q", hb)[0]
        head = open(os.path.join(pack.src.path, f), "rb").read(8 + n)
    else:
        n = struct.unpack("<Q", pack.src.read(f, 0, 8))[0]
        head = pack.src.read(f, 0, 8 + n)
    h = json.loads(head[8:])
    size = 8 + n + max(v["data_offsets"][1] for k, v in h.items() if k != "__metadata__")
    shards = sorted({k.rsplit(".", 1)[0] for k in h if k != "__metadata__"},
                    key=lambda s: int(s.rsplit(".", 1)[1]))
    first, layout = 0, []
    for s in shards:
        w, sc, bi = h[s + ".weight"], h[s + ".scales"], h[s + ".biases"]
        nrows = w["shape"][0]
        layout.append((first, nrows, 8 + n + w["data_offsets"][0], w["shape"][1] * 4,
                       8 + n + sc["data_offsets"][0], sc["shape"][1] * 2, 8 + n + bi["data_offsets"][0]))
        first += nrows
    fd = os.open(out_path, os.O_RDWR if add else os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        if add:
            if os.fstat(fd).st_size != size or os.pread(fd, len(head), 0) != head:
                raise ValueError(f"{out_path} is not the table of {pack.where}")
        else:
            os.pwrite(fd, head, 0)
            os.ftruncate(fd, size)
        starts = [L[0] for L in layout]
        want = {}  # shard -> its rows (local indices) to write
        for r in sorted(set(rows)):
            k = bisect.bisect_right(starts, r) - 1
            f0, nr, wo, wb, so, gb, bo = layout[k]
            if not 0 <= r - f0 < nr:
                raise ValueError(f"table row {r} out of range")
            j = r - f0
            if add and any(os.pread(fd, gb, so + j * gb)):
                continue
            want.setdefault(k, []).append(j)
        jobs = []  # (a, b, tensor offset, row bytes, rows in [a, b))
        for k, js in sorted(want.items()):
            f0, nr, wo, wb, so, gb, bo = layout[k]
            for base, width in ((wo, wb), (so, gb), (bo, gb)):
                if not stream:
                    jobs += [(base + j * width, base + (j + 1) * width, base, width, [j]) for j in js]
                    continue
                per = max(1, piece // width)  # rows a piece
                groups = {}
                for j in js:
                    groups.setdefault(j // per, []).append(j)
                jobs += [(base + p * per * width, base + min(nr, (p + 1) * per) * width, base, width, g)
                         for p, g in sorted(groups.items())]
        batch = 32 if stream else 3000
        for i in range(0, len(jobs), batch):
            part = jobs[i:i + batch]
            for (a, b, base, width, g), data in zip(part, pack.rows(f, [(x[0], x[1]) for x in part])):
                for j in g:
                    o = base + j * width
                    os.pwrite(fd, data[o - a:o - a + width], o)
    finally:
        os.close(fd)
    return size, first, sum(len(js) for js in want.values())


def cmd_convert(a):
    os.makedirs(a.out, exist_ok=True)
    pool = ThreadPoolExecutor(a.conns)
    pack, gg = PackSource(a, pool), GgufSource(a, pool)
    log(f"pack {pack.where}: {len(pack.index)} tensors; GGUF {gg.where}: {len(gg.g.tensors)} tensors")
    done_path = os.path.join(a.out, ".q20.done.json")
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()

    def finish(name):
        done.add(name)
        json.dump(sorted(done), open(done_path, "w"))

    # the trunk's expert modules: their GGUF tensors must exist and match the pack's shapes
    files = {}
    for k, (f, off, nb, dt, shape) in pack.index.items():
        files.setdefault(f, []).append((off, k))
    layers = sorted({expert_of(k)[0] for k in pack.index if expert_of(k)})
    modules = {}
    for L in layers:
        for mat, gname in GGUF_EXPERT.items():
            E, N, K, off, nb = gg.g.expert(gname.format(L))
            ws = pack.index[EXPERT.format(L, mat) + ".weight"][4]
            if ws[0] != E or ws[1] != N:
                raise ValueError(f"layer {L} {mat}: GGUF [{E}, {N}, {K}] vs pack {ws}")
            modules[(L, mat)] = (E, N, K, off, nb)
    log(f"{len(layers)} layers x 3 expert tensors in Q2_0, e.g. layer 0 gate [{modules[(layers[0], 'gate')][:3]}]")

    rng = np.random.default_rng(1)
    worst = {}
    total = 0
    model_files = sorted(f for f in files if f.startswith("model-") and f.endswith(".safetensors"))
    for f in model_files:
        names = [k for _, k in sorted(files[f])]
        specs, experts, plain = [], {}, []
        for k in names:
            ex = expert_of(k)
            shape = pack.index[k][4]
            if ex is None:
                specs.append((k, pack.index[k][3], shape))
                plain.append(k)
                continue
            L, mat, part = ex
            E, N, K, _, _ = modules[(L, mat)]
            if part == "weight":
                specs.append((k, "U32", [E, N, K // 16]))
                experts[(L, mat)] = k[:-len(".weight")]
            elif part == "scales":
                specs.append((k, "F16", [E, N, K // 64]))
        path = os.path.join(a.out, f)
        w = P.SafetensorsWriter(path, specs)
        total += w.size
        if f in done:
            continue
        t0 = time.time()
        for k, v in pack.tensor_bytes(plain).items():
            w.write(k, np.frombuffer(v, np.uint8))
        for (L, mat), base in sorted(experts.items()):
            E, N, K, off, nb = modules[(L, mat)]
            raw = gg.fetch(off, nb)
            codes, d = q20_planes(raw, E, N, K)
            w.write(base + ".weight", codes)
            w.write(base + ".scales", d)
            if a.verify:
                c = verify_rows(pack, L, mat, codes, d, a.verify, rng)
                worst[mat] = min(worst.get(mat, 1.0), c)
                if c < 0.6:
                    raise ValueError(f"layer {L} {mat}: rows correlate {c:.3f} with the 4-bit pack's")
        finish(f)
        log(f"{f}: {len(plain)} tensors copied, {len(experts)} expert tensors in Q2_0 ({w.size / 1e9:.2f} GB, "
            f"{time.time() - t0:.0f} s)" + (f"; worst row correlation {min(worst.values()):.3f}" if worst else ""))

    # the MTP head (its experts stay 4-bit: Kapitsa's MTP path reads 4-bit experts)
    for f in sorted(files):
        if f.startswith("model-") or f == TABLE or f in done:
            continue
        pack.copy_file(f, os.path.join(a.out, f))
        finish(f)
        log(f"{f}: copied")
    # the n-gram table
    tpath = os.path.join(a.out, TABLE)
    if TABLE in files and TABLE not in done:
        if a.table == "sparse":
            rows = sparse_rows(a.rows_ids or [])
            size, nrows, _ = write_sparse_table(pack, tpath, rows, stream=a.stream)
            log(f"{TABLE}: sparse, {len(rows)} of {nrows} rows written ({size / 1e9:.1f} GB apparent)")
        elif a.table == "link":
            if not pack.local:
                raise ValueError("--table link needs a local --pack")
            src = os.path.join(pack.src.path, TABLE)
            if subprocess.call(["cp", "-c", src, tpath], stderr=subprocess.DEVNULL) != 0:
                os.link(src, tpath)
            log(f"{TABLE}: cloned from {src}")
        else:
            pack.copy_file(TABLE, tpath)
            log(f"{TABLE}: copied")
        finish(TABLE)

    # config, index, small files, conversion record
    for f in SMALL:
        b = pack.small(f)
        if b is None:
            continue
        if f == "config.json":
            cfg = json.loads(b)
            for key in ("quantization", "quantization_config"):
                if key in cfg:
                    for L in layers:
                        for mat in GGUF_EXPERT:
                            cfg[key][EXPERT.format(L, mat)] = {"group_size": 64, "bits": 2, "mode": "q2_0"}
            b = (json.dumps(cfg, indent=2) + "\n").encode()
        open(os.path.join(a.out, f), "wb").write(b)
    weight_map, size = {}, 0
    for f in sorted(os.listdir(a.out)):
        if f.endswith(".safetensors"):
            with open(os.path.join(a.out, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            for k, v in h.items():
                if k != "__metadata__":
                    weight_map[k] = f
                    size += v["data_offsets"][1] - v["data_offsets"][0]
    json.dump({"metadata": {"total_size": size}, "weight_map": dict(sorted(weight_map.items()))},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    rec = {"tool": "tools/qwen4_exp_q20.py", "date": time.strftime("%Y-%m-%d %H:%M:%S"), "pack": pack.where,
           "gguf": gg.where, "experts": "Q2_0 (ggml 42): .weight U32 codes [E, N, K/16], .scales F16 d [E, N, K/64]",
           "table": a.table, "rows_ids": a.rows_ids, "verify_rows": a.verify,
           "worst_row_correlation": worst, "files": {f: os.path.getsize(os.path.join(a.out, f))
                                                     for f in sorted(os.listdir(a.out)) if not f.startswith(".")}}
    json.dump(rec, open(os.path.join(a.out, "conversion_q20.json"), "w"), indent=2)
    log(f"done: {a.out} ({total / 1e9:.1f} GB of model files)")


def cmd_check(a):
    """The output's expert planes against the GGUF named in its conversion record, and its config."""
    rec = json.load(open(os.path.join(a.dir, "conversion_q20.json")))
    cfg = json.load(open(os.path.join(a.dir, "config.json")))
    out = P.Local(a.dir)
    src = rec["gguf"]
    if os.path.exists(src):
        fd = os.open(src, os.O_RDONLY)
        read = lambda x, y: os.pread(fd, y - x, x)  # noqa: E731
    else:
        repo, rev, path = parse_repo(src, None)
        r = P.Remote(repo, rev)
        read = lambda x, y: r.read(path, x, y)  # noqa: E731
    g = Gguf(read, src)
    rng = np.random.default_rng(2)
    bad = 0
    layers = sorted({expert_of(k)[0] for k in out.index if expert_of(k)})
    for L in layers:
        for mat, gname in GGUF_EXPERT.items():
            E, N, K, off, nb = g.expert(gname.format(L))
            base = EXPERT.format(L, mat)
            if cfg["quantization"].get(base, {}).get("mode") != "q2_0":
                bad += 1
                log(f"{base}: config has no q2_0 entry")
            if base + ".biases" in out.index:
                bad += 1
            codes, d = out.array(base + ".weight"), out.array(base + ".scales")
            for e in rng.integers(E, size=2):
                blk = N * (K // 64) * Q2_0_BYTES
                raw = read(off + int(e) * blk, off + (int(e) + 1) * blk)
                c2, d2 = q20_planes(raw, 1, N, K)
                if not (np.array_equal(c2[0], codes[e]) and np.array_equal(d2[0].view("<u2"), d[e].view("<u2"))):
                    bad += 1
                    log(f"{base} expert {e}: bytes differ from the GGUF")
    log(f"check: {len(layers)} layers, {'all equal' if not bad else f'{bad} problems'}")
    return 1 if bad else 0


def cmd_rows(a):
    """More token sequences' rows into the output's sparse table, from the pack named in its conversion record."""
    rec = json.load(open(os.path.join(a.dir, "conversion_q20.json")))
    a.pack, a.pack_repo = (rec["pack"], None) if os.path.isdir(rec["pack"]) else (None, rec["pack"])
    with ThreadPoolExecutor(a.conns) as pool:
        pack = PackSource(a, pool)
        rows = sparse_rows(a.ids)
        _, _, wrote = write_sparse_table(pack, os.path.join(a.dir, TABLE), rows, add=True, stream=a.stream)
    log(f"{TABLE}: {len(rows)} rows read by the sequences, {wrote} of them added")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--out", required=True)
    c.add_argument("--pack", help="local MLX pack folder (default: --pack-repo)")
    c.add_argument("--pack-repo", default=PACK_REPO)
    c.add_argument("--gguf", help="local Q2_0 GGUF file (default: --gguf-repo)")
    c.add_argument("--gguf-repo", default=GGUF_REPO)
    c.add_argument("--table", choices=["full", "link", "sparse"], default="full")
    c.add_argument("--rows-ids", nargs="*", help="token id files (@path) whose table rows --table sparse writes")
    c.add_argument("--stream", action="store_true", help="--table sparse: rows cut from whole reads of the table's "
                   "tensors (16 MB pieces) instead of three small reads a row: for many rows from a remote pack")
    c.add_argument("--verify", type=int, default=4, help="rows per expert tensor compared with the 4-bit pack")
    c.add_argument("--conns", type=int, default=16)
    k = sub.add_parser("check")
    k.add_argument("dir")
    r = sub.add_parser("rows")
    r.add_argument("dir")
    r.add_argument("ids", nargs="+", help="token id files (@path)")
    r.add_argument("--stream", action="store_true", help="rows cut from whole reads of the table's tensors (16 MB "
                   "pieces) instead of three small reads a row")
    r.add_argument("--conns", type=int, default=16)
    a = ap.parse_args()
    sys.exit({"convert": cmd_convert, "check": cmd_check, "rows": cmd_rows}[a.cmd](a))


if __name__ == "__main__":
    main()
