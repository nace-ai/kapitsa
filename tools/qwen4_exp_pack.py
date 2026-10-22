#!/usr/bin/env python3
"""Shared pieces of the Qwen3.8-Flash-Next (qwen4_exp) converter and its quality evaluation.

Used by tools/convert_qwen4_exp.py and bench/quality/eval_qwen4_exp.py; the pack layout they write is described in
those tools' docstrings. Here:
  Remote / Local    a safetensors checkpoint by tensor name: Hugging Face (HTTP range reads, many connections,
                    headers cached) or a local folder
  mlx_names()       Hugging Face name -> the names mlx-vlm's qwen4_exp uses after its sanitize() (+ layout change)
  role_of()         the precision role of a tensor (experts, shared, router, gdn, attn, indexer, hc, ple, embed, ...)
  quantize()        MLX affine quantization with Kapitsa's re-quantizer (tools/mlx_quant.cpp -> src/gguf/quant.cpp)
  ngram_ids()       the 16 PLE table rows each token reads (hash of the token ids; SPEC 2.7)
  SafetensorsWriter a safetensors file written tensor by tensor at offsets fixed by its header (resumable)
"""
import ctypes
import fcntl
import hashlib
import http.client
import json
import os
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import numpy as np

UA = {"User-Agent": "kapitsa-convert"}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
WORK = os.path.expanduser(os.environ.get("KAPITSA_WORK", "~/.cache/kapitsa/work"))
GPU_LOCK = os.path.expanduser("/tmp/kapitsa-gpu.lock")

NP_DT = {"BF16": np.uint16, "F32": np.float32, "F16": np.float16, "U32": np.uint32, "I32": np.int32,
         "U8": np.uint8, "I64": np.int64, "F8_E4M3": np.uint8}
DT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8, "U8": 1, "F8_E4M3": 1}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = os.path.expanduser("~/.cache/huggingface/token")
    return open(p).read().strip() if os.path.exists(p) else None


# ---------------------------------------------------------------- checkpoints

class Remote:
    """A Hugging Face repo's safetensors by tensor name: name -> (file, absolute offset, nbytes, dtype, shape)."""

    def __init__(self, repo, rev="main", cache_dir=None):
        self.repo, self.rev = repo, rev
        self.tls = threading.local()
        tok = hf_token()
        self.auth = {"Authorization": f"Bearer {tok}"} if tok else {}
        meta = json.loads(self._get(f"https://huggingface.co/api/models/{repo}/revision/{rev}"))
        self.rev = meta["sha"]
        self.siblings = [s["rfilename"] for s in meta["siblings"]]
        cache = os.path.join(cache_dir or WORK, f"headers-{repo.replace('/', '--')}-{self.rev[:12]}.json")
        if os.path.exists(cache):
            self.index = {k: tuple(v) for k, v in json.load(open(cache)).items()}
        else:
            self.index = {}
            files = sorted(f for f in self.siblings if f.endswith(".safetensors") and "/" not in f)
            from concurrent.futures import ThreadPoolExecutor

            def hdr(f):
                n = struct.unpack("<Q", self.read(f, 0, 8))[0]
                return f, n, json.loads(self.read(f, 8, 8 + n))
            with ThreadPoolExecutor(16) as ex:
                for f, n, h in ex.map(hdr, files):
                    for k, v in h.items():
                        if k != "__metadata__":
                            a, b = v["data_offsets"]
                            self.index[k] = (f, 8 + n + a, b - a, v["dtype"], v["shape"])
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            json.dump(self.index, open(cache, "w"))

    def url(self, f):
        return f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{urllib.parse.quote(f)}"

    def _get(self, url, rng=None, tries=12):
        h = dict(UA, **self.auth)
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
                time.sleep(min(60, 2 * (t + 1)))

    def read(self, f, a=None, b=None):
        return self._get(self.url(f), (a, b) if a is not None else None)

    def _cdn(self, f):
        """The file's CDN URL (the Hub answers resolve/ with a redirect), cached per thread."""
        c = getattr(self.tls, "cdn", None)
        if c is None:
            c = self.tls.cdn = {}
        if f not in c:
            req = urllib.request.Request(self.url(f), headers=dict(UA, **self.auth, Range="bytes=0-0"))

            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *a, **k):
                    return None
            op = urllib.request.build_opener(NoRedirect)
            try:
                op.open(req, timeout=60)
                c[f] = self.url(f)
            except urllib.error.HTTPError as e:
                c[f] = e.headers["Location"] if e.code in (301, 302, 303, 307, 308) else self.url(f)
        return c[f]

    def read_small(self, f, a, b, tries=12):
        """A small range on a kept-alive connection to the CDN (row gathers: thousands of requests)."""
        for t in range(tries):
            try:
                u = urllib.parse.urlparse(self._cdn(f))
                conns = getattr(self.tls, "conns", None)
                if conns is None:
                    conns = self.tls.conns = {}
                conn = conns.get(u.netloc)
                if conn is None:
                    conn = conns[u.netloc] = http.client.HTTPSConnection(u.netloc, timeout=120)
                path = u.path + ("?" + u.query if u.query else "")
                conn.request("GET", path, headers=dict(UA, Range=f"bytes={a}-{b - 1}"))
                r = conn.getresponse()
                d = r.read()
                if r.status not in (200, 206) or len(d) != b - a:
                    raise IOError(f"status {r.status}, {len(d)} bytes")
                return d
            except Exception:
                self.tls.conns = {}
                if t >= 2:
                    getattr(self.tls, "cdn", {}).pop(f, None)
                if t == tries - 1:
                    raise
                time.sleep(min(30, 1 + t))

    def fetch(self, names, pool, piece=64 << 20):
        """name -> numpy array (uint16 for bf16), parallel ranged reads in `piece`-byte pieces."""
        jobs, bufs = [], {}
        for k in names:
            f, off, nb, dt, shape = self.index[k]
            bufs[k] = bytearray(nb)
            for p in range(0, nb, piece):
                jobs.append((k, f, off + p, off + min(nb, p + piece), p))

        def work(j):
            k, f, a, b, p = j
            bufs[k][p:p + (b - a)] = self.read(f, a, b)
        list(pool.map(work, jobs))
        return {k: np.frombuffer(bufs[k], dtype=NP_DT[self.index[k][3]]).reshape(self.index[k][4]) for k in names}


class Local:
    """A local folder of safetensors (an MLX pack, or HF files), same interface as Remote.fetch."""

    def __init__(self, path):
        self.path = path
        self.index = {}
        for f in sorted(os.listdir(path)):
            if not f.endswith(".safetensors"):
                continue
            with open(os.path.join(path, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            for k, v in h.items():
                if k != "__metadata__":
                    a, b = v["data_offsets"]
                    self.index[k] = (f, 8 + n + a, b - a, v["dtype"], v["shape"])
        self.maps = {}

    def array(self, k):
        f, off, nb, dt, shape = self.index[k]
        if f not in self.maps:
            self.maps[f] = np.memmap(os.path.join(self.path, f), dtype=np.uint8, mode="r")
        return self.maps[f][off:off + nb].view(NP_DT[dt]).reshape(shape)

    def fetch(self, names, pool=None):
        return {k: np.array(self.array(k)) for k in names}


# ---------------------------------------------------------------- names, roles

HF_TEXT = "model.language_model."
MLX_TEXT = "language_model.model."


def mlx_names(name, shape):
    """[(mlx name, transform)] for one Hugging Face tensor; transform in (None, 'conv', 'gate_up0', 'gate_up1',
    'patch'). The fused expert tensor gives two outputs. Names are mlx-vlm qwen4_exp's after its sanitize()."""
    if name.startswith("mtp."):
        base = name
    elif name.startswith(HF_TEXT):
        base = MLX_TEXT + name[len(HF_TEXT):]
    elif name.startswith("model.visual."):
        base = "vision_tower." + name[len("model.visual."):]
    elif name.startswith("lm_head."):
        base = "language_model." + name
    else:
        raise ValueError(f"unexpected tensor {name}")
    base = base.replace(".ngram_embedding.shard_", ".ngram_embedding.shards.")
    if base.endswith(".mlp.experts.gate_up_proj"):
        p = base[: -len("experts.gate_up_proj")]
        return [(p + "switch_mlp.gate_proj.weight", "gate_up0"), (p + "switch_mlp.up_proj.weight", "gate_up1")]
    if base.endswith(".mlp.experts.down_proj"):
        return [(base[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight", None)]
    if base.endswith("conv1d.weight") and len(shape) == 3 and shape[-1] != 1:
        return [(base, "conv")]
    if base.endswith("patch_embed.proj.weight") and len(shape) == 5 and shape[1] == 3:
        return [(base, "patch")]
    return [(base, None)]


def transform(x, how):
    """Apply a mlx_names transform to a numpy array (any dtype)."""
    if how is None:
        return x
    if how == "conv":
        return np.ascontiguousarray(np.moveaxis(x, 2, 1))
    if how in ("gate_up0", "gate_up1"):
        h = x.shape[1] // 2
        return np.ascontiguousarray(x[:, :h] if how == "gate_up0" else x[:, h:])
    if how == "patch":
        return np.ascontiguousarray(x.transpose(0, 2, 3, 4, 1))
    raise ValueError(how)


def mlx_shape(shape, how):
    s = list(shape)
    if how == "conv":
        return [s[0], s[2], s[1]]
    if how in ("gate_up0", "gate_up1"):
        return [s[0], s[1] // 2, s[2]]
    if how == "patch":
        return [s[0], s[2], s[3], s[4], s[1]]
    return s


ROLES = ("experts", "shared", "router", "gdn", "attn", "indexer", "hc", "mixer", "ple", "embed", "head", "table",
         "mtp", "mtp_experts", "inject", "vision")


def role_of(name):
    """Precision role of an MLX tensor/module name; None for tensors never quantized (norms, conv, A_log, ...)."""
    n = name[:-7] if name.endswith(".weight") else name
    if name.startswith("vision_tower."):
        return "vision"
    if n.endswith("block_inject_weight"):
        return "inject"
    if name.startswith("mtp."):
        if ".switch_mlp." in n:
            return "mtp_experts"
        if n.endswith("mlp.gate") or n.endswith("shared_expert_gate"):
            return "router"
        if n.endswith("_proj") or n.endswith("input_mix_weight_down") or n.endswith("input_mix_weight_up") \
                or n.endswith("fc_embedding") or n.endswith("fc_hidden"):
            return "mtp"
        return None
    if ".ngram_embedding.shards." in n:
        return "table"
    if n.endswith("embed_tokens"):
        return "embed"
    if n.endswith("lm_head"):
        return "head"
    if ".switch_mlp." in n:
        return "experts"
    if n.endswith("mlp.gate") or n.endswith("shared_expert_gate"):
        return "router"
    if ".shared_expert." in n:
        return "shared"
    if n.endswith("input_mix_weight_down") or n.endswith("input_mix_weight_up"):
        # the final collapse (13 MB in bf16) has its own role, by default hc's precision
        return "mixer" if ".hyper_connection_mixer." in n else "hc"
    if ".indexer.index_qk_proj" in n:
        return "indexer"
    if ".linear_attn." in n and (n.endswith("_proj") or n.endswith("in_proj_qkv") or n.endswith("in_proj_a")
                                 or n.endswith("in_proj_b") or n.endswith("in_proj_z")):
        return "gdn"
    if ".self_attn." in n and n.split(".")[-1] in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return "attn"
    if ".ple." in n and (n.endswith("key_proj") or n.endswith("value_proj")):
        return "ple"
    return None


def parse_q(s):
    """'4' / '4:32' -> {'bits', 'group_size', 'mode'}; 'bf16' -> None; 'fp8' (table only) -> {'mode': 'fp8'}."""
    if s in ("bf16", "none"):
        return None
    if s == "fp8":
        return {"mode": "fp8", "bits": 8, "group_size": 0}
    b, _, g = s.partition(":")
    return {"mode": "affine", "bits": int(b), "group_size": int(g or 64)}


class Recipe:
    """A precision per role. Spec: '<default>[,role=<q>...]' e.g. '4,router=8,embed=8,indexer=8,ple=8'.
    Always (unless overridden): inject and vision bf16, mtp_experts = experts; with a quantized default also
    router 8-bit, table 4-bit group 32, mtp 8-bit. A bf16 default means everything bf16 but what is named."""

    def __init__(self, spec):
        parts = spec.split(",")
        self.spec = spec
        self.default = parse_q(parts[0])
        self.over = {"inject": None, "vision": None}
        if self.default is not None:  # a quantized recipe: MLX's 8-bit router, the table 4-bit group 32
            self.over.update({"router": parse_q("8"), "table": parse_q("4:32"), "mtp": parse_q("8")})
        for p in parts[1:]:
            r, _, q = p.partition("=")
            if r not in ROLES:
                raise SystemExit(f"unknown role {r} in {spec}")
            self.over[r] = parse_q(q)
        if "mtp_experts" not in self.over:
            self.over["mtp_experts"] = self.over.get("experts", self.default)

    def q(self, role):
        if role is None:
            return None
        if role == "mixer" and role not in self.over:
            role = "hc"
        return self.over[role] if role in self.over else self.default


# ---------------------------------------------------------------- quantization

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        out = os.path.join(WORK, "flashnext-tmp", "libmlxq.dylib")
        srcs = [os.path.join(HERE, "mlx_quant.cpp"), os.path.join(REPO_ROOT, "src/gguf/quant.cpp"),
                os.path.join(REPO_ROOT, "src/gguf/gguf.cpp")]
        if not os.path.exists(out) or os.path.getmtime(out) < max(os.path.getmtime(s) for s in srcs):
            os.makedirs(os.path.dirname(out), exist_ok=True)
            subprocess.check_call(["c++", "-O3", "-std=c++20", "-shared", "-fPIC", "-I" + os.path.join(REPO_ROOT, "src"),
                                   "-I" + os.path.join(REPO_ROOT, "third_party")] + srcs + ["-o", out])
        _LIB = ctypes.CDLL(out)
    return _LIB


def quantize(x, bits, gs, threads=0):
    """bf16 bits (uint16 array, last axis = row) -> (words uint32, scales uint16 (bf16), biases uint16)."""
    x = np.ascontiguousarray(x, dtype=np.uint16)
    shape = x.shape
    cols = shape[-1]
    rows = x.size // cols
    w = np.empty(shape[:-1] + (cols * bits // 32,), np.uint32)
    s = np.empty(shape[:-1] + (cols // gs,), np.uint16)
    b = np.empty_like(s)
    P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
    r = _lib().mlxq_quantize_bf16(P(x), ctypes.c_int64(rows), ctypes.c_int64(cols), bits, gs, P(w), P(s), P(b),
                                  threads)
    if r:
        raise ValueError(f"cannot quantize {shape} to {bits}-bit group {gs}")
    return w, s, b


def bf16_to_f32(x):
    return (np.asarray(x, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def dequantize_np(w, s, b, bits, gs):
    """MLX affine -> fp32 (numpy; tests and table rows)."""
    w = np.asarray(w, dtype=np.uint32)
    cols = w.shape[-1] * 32 // bits
    flat = w.reshape(-1, w.shape[-1])
    bitpos = np.arange(cols, dtype=np.int64) * bits
    word, sh = bitpos // 32, (bitpos % 32).astype(np.uint64)
    lo = flat[:, word].astype(np.uint64) >> sh
    hi_need = (sh + bits) > 32
    if hi_need.any():
        nxt = np.minimum(word + 1, flat.shape[1] - 1)
        hi = flat[:, nxt].astype(np.uint64) << (np.uint64(32) - sh)
        lo = np.where(hi_need, lo | hi, lo)
    q = (lo & np.uint64((1 << bits) - 1)).astype(np.float32).reshape(w.shape[:-1] + (cols,))
    sc = np.repeat(bf16_to_f32(s), gs, axis=-1)
    bi = np.repeat(bf16_to_f32(b), gs, axis=-1)
    return q * sc + bi


def fp8_e4m3_round(x):
    """fp32 -> nearest E4M3FN value (saturating at 448), as fp32 (round to nearest even)."""
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), 448.0)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -9)))
    e = np.clip(e, -6, 8)
    step = 2.0 ** (e - 3)
    r = np.round(a / step) * step  # numpy rounds half to even
    return (np.sign(x) * np.minimum(r, 448.0)).astype(np.float32)


# ---------------------------------------------------------------- PLE n-gram ids

NGRAM = {
    "multipliers": [23703573157769, 20109073645365, 8052911324071],
    "sizes": [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
              20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171],
    "eos": 248044,
    "rows": 320001536,
    "shard_rows": 2500012,
}
NGRAM["offsets"] = [int(x) for x in np.concatenate([[0], np.cumsum(NGRAM["sizes"])[:-1]])]


def ngram_ids(seq, eos=NGRAM["eos"]):
    """[T, 16] int64 table rows for a token sequence starting a document (no previous context)."""
    t = np.asarray(seq, dtype=np.int64)
    T = len(t)
    pos = np.arange(T)
    last_eos = np.maximum.accumulate(np.where(t == eos, pos, -1))
    prev_eos = np.concatenate([[-1], last_eos[:-1]])  # an EOS strictly before t
    start = prev_eos + 1

    def tok(k):
        src = pos - k
        ok = (src >= 0) & (src >= start)
        return np.where(ok, t[np.maximum(src, 0)], eos)
    m = NGRAM["multipliers"]
    x2 = (tok(0) * m[0]) ^ (tok(1) * m[1])
    x3 = x2 ^ (tok(2) * m[2])
    sizes = np.array(NGRAM["sizes"], dtype=np.int64)
    offs = np.array(NGRAM["offsets"], dtype=np.int64)
    ids = np.empty((T, 16), np.int64)
    ids[:, :8] = x2[:, None] % sizes[None, :8] + offs[None, :8]
    ids[:, 8:] = x3[:, None] % sizes[None, 8:] + offs[None, 8:]
    return ids


# ---------------------------------------------------------------- safetensors

class SafetensorsWriter:
    """A safetensors file whose header (names, dtypes, shapes, order) is fixed up front; tensors are written at their
    offsets in any order (pwrite), so an interrupted writer can resume. align: the data section starts at a multiple
    of it (spaces in the header)."""

    def __init__(self, path, specs, align=16384, metadata=None):
        """specs: [(name, dtype, shape)] in file order."""
        self.path = path
        hdr, off = {}, 0
        if metadata:
            hdr["__metadata__"] = metadata
        self.off = {}
        for name, dt, shape in specs:
            nb = int(np.prod(shape)) * DT_SIZE[dt] if len(shape) else DT_SIZE[dt]
            hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + nb]}
            self.off[name] = (off, nb)
            off += nb
        h = json.dumps(hdr, separators=(",", ":")).encode()
        pad = (-(8 + len(h))) % align
        self.header = h + b" " * pad
        self.data0 = 8 + len(self.header)
        self.size = self.data0 + off
        new = not os.path.exists(path)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if new or os.path.getsize(path) != self.size:
                os.ftruncate(fd, self.size)
            os.pwrite(fd, struct.pack("<Q", len(self.header)) + self.header, 0)
        finally:
            os.close(fd)

    def write(self, name, arr):
        off, nb = self.off[name]
        b = np.ascontiguousarray(arr).tobytes()
        if len(b) != nb:
            raise ValueError(f"{name}: {len(b)} bytes, header says {nb}")
        fd = os.open(self.path, os.O_WRONLY)
        try:
            v, pos = memoryview(b), 0
            while pos < nb:
                pos += os.pwrite(fd, v[pos:pos + (256 << 20)], self.data0 + off + pos)
        finally:
            os.close(fd)

    def abs_offset(self, name):
        return self.data0 + self.off[name][0]


def sha256_file(path, chunk=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class GpuLock:
    """/tmp/kapitsa-gpu.lock, shared with the other GPU jobs on this Mac (NI_LOCK_HELD=1: the caller holds it)."""
    held = None

    @staticmethod
    def take():
        if os.environ.get("NI_LOCK_HELD") or GpuLock.held is not None:
            return
        f = os.fdopen(os.open(GPU_LOCK, os.O_RDONLY | os.O_CREAT, 0o644))
        t = time.time()
        fcntl.flock(f, fcntl.LOCK_EX)
        GpuLock.held = f
        if time.time() - t > 5:
            log(f"GPU lock after {time.time() - t:.0f} s")

    @staticmethod
    def release(sync=None):
        if GpuLock.held is not None:
            if sync:
                sync()
            fcntl.flock(GpuLock.held, fcntl.LOCK_UN)
            GpuLock.held.close()
            GpuLock.held = None
