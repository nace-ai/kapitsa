#!/usr/bin/env python3
"""tools/nh_fp4_requant.py: its FP8 E4M3 and E2M1 rounding against a scalar reference of ModelOpt's NVFP4 recipe
(written apart from the tool's vectorized one: per value, from the formats' definitions), ties and zero blocks, the
dequantized error bound, and a run on a small fake NVFP4 checkpoint (two shards): the modules quantized and those
kept (the head without --head, --keep, conv1d, already NVFP4), every other tensor copied byte for byte, the index, hf_quant_config.json and
requant.json, the same bytes on a second run, --dry-run writing nothing, and the refusal of a non-NVFP4 folder.
CPU only. Needs numpy.

usage: test_nh_fp4_requant.py
"""
import hashlib, json, math, os, random, struct, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "tools", "nh_fp4_requant.py")
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import numpy as np  # noqa: E402
import nh_fp4_requant as rq  # noqa: E402

FAILED = []


def check(ok, name, detail=""):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  ({detail})"))
    if not ok:
        FAILED.append(name)


# ---- scalar reference ----
def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def e4m3_value(code):
    e, m = (code >> 3) & 15, code & 7
    return math.ldexp(m, -9) if e == 0 else math.ldexp(8 + m, e - 10)


def e4m3_ref(x):
    """Nearest E4M3 code of x >= 0 (ties to an even code), x clamped to 448."""
    x = min(x, 448.0)
    best = None
    for c in range(127):
        d = abs(e4m3_value(c) - x)
        if best is None or d < best[0] or (d == best[0] and c % 2 == 0):
            best = (d, c)
    return best[1]


def e2m1_ref(x):
    """Nearest E2M1 code of x (ties to the even code), sign bit for any negative x."""
    vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    a = abs(x)
    best = None
    for c, v in enumerate(vals):
        d = abs(v - a)
        if best is None or d < best[0] or (d == best[0] and c % 2 == 0):
            best = (d, c)
    return best[1] | (8 if x < 0 else 0)


def quantize_ref(w):
    """ModelOpt's NVFP4 of a list of rows of float32 values: (codes per row, scale codes per row, weight_scale_2)."""
    amax = max(abs(v) for row in w for v in row)
    ws2 = f32(amax / 2688.0)
    codes, scales = [], []
    for row in w:
        cr, sr = [], []
        for b in range(0, len(row), 16):
            blk = row[b:b + 16]
            sc = f32(max(abs(v) for v in blk) / f32(6.0 * ws2))
            if sc == 0:
                sc = 1.0
            s = e4m3_ref(sc)
            den = f32(e4m3_value(s) * ws2)
            cr += [e2m1_ref(f32(v / den)) if den else 0 for v in blk]
            sr.append(s)
        codes.append(cr)
        scales.append(sr)
    return codes, scales, ws2


# ---- formats ----
codes = np.arange(127)
check(np.array_equal(rq.e4m3_codes(rq.E4M3[codes]), codes), "E4M3: every code's value encodes back to the code")
check([e4m3_value(c) for c in range(127)] == list(rq.E4M3), "E4M3 table equals the format's definition")
mids = (rq.E4M3[:-1] + rq.E4M3[1:]) / 2  # exact ties: to the even code
check(np.array_equal(rq.e4m3_codes(mids), [c + (c % 2) for c in range(126)]), "E4M3 ties go to the even code")
check(rq.e4m3_codes(np.array([500.0, 448.0, 0.0, 2.0**-11]))[:].tolist() == [126, 126, 0, 0], "E4M3 clamp, zero, tiny")
pts = np.array([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 7, 0.2499, 0.2501, -0.75, -1e-9, -0.0, 1e-9, -6.5], np.float32)
check([int(c) for c in rq.e2m1_codes(pts)] == [e2m1_ref(float(x)) for x in pts], "E2M1 at the bounds, ties and signs",
      f"{rq.e2m1_codes(pts).tolist()} vs {[e2m1_ref(float(x)) for x in pts]}")
rnd = np.random.default_rng(1).standard_normal(20000).astype(np.float32) * 3
check([int(c) for c in rq.e2m1_codes(rnd)] == [e2m1_ref(float(x)) for x in rnd], "E2M1 of 20000 random values")

# ---- quantize vs the scalar reference ----
rng = random.Random(7)
for name, rows, cols in [("random", 24, 64), ("small K", 5, 16), ("wide", 3, 4480)]:
    w = [[f32(rng.gauss(0, 0.02)) for _ in range(cols)] for _ in range(rows)]
    if name == "random":
        w[2][:16] = [0.0] * 16  # an all-zero block: scale 1.0, codes 0
        w[3][:16] = [0.0] * 15 + [f32(-1e-30)]  # a nearly-zero block
        w[4][:16] = [f32(v) for v in [0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.005] * 2]  # ties appear
    p, s, ws2 = rq.quantize(np.array(w, np.float32), rows=7)
    c_ref, s_ref, ws2_ref = quantize_ref(w)
    got = np.stack([p & 15, p >> 4], -1).reshape(rows, cols)
    check(float(ws2) == ws2_ref, f"quantize {name}: weight_scale_2", f"{ws2} vs {ws2_ref}")
    check(s.tolist() == s_ref, f"quantize {name}: block scale codes")
    check(got.tolist() == c_ref, f"quantize {name}: codes (packed low nibble first)",
          f"{int(np.sum(got != np.array(c_ref)))} differ")
    if name == "random":
        check(s[2, 0] == rq.e4m3_codes(np.array([1.0]))[0] and not got[2, :16].any(), "an all-zero block: scale 1.0")
        check(s[3, 0] == 0 and not got[3, :16].any(), "a block whose scale rounds to zero: codes 0")
    deq = rq.dequantize(p, s, ws2)
    wa = np.array(w, np.float32)
    # each value within half the block's largest E2M1 step (2 x scale x ws2) plus the scale's own rounding (1/16)
    bound = np.repeat(rq.E4M3[s].astype(np.float32) * np.float32(ws2), 16, axis=1) * (1.0 + 1 / 8)
    bmax = np.repeat(np.abs(wa).reshape(rows, -1, 16).max(-1), 16, axis=1)
    check(bool(np.all(np.abs(deq - wa) <= np.maximum(bound, bmax / 6 + 1e-30) * 1.01)), f"quantize {name}: error bound")

# ---- a fake NVFP4 checkpoint ----
DT = {"BF16": np.uint16, "U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32}


def bf16(a):
    u = np.asarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def write_st(path, tensors):
    header, off, blobs = {"__metadata__": {"format": "pt"}}, 0, []
    for n, dt, arr in tensors:
        b = np.ascontiguousarray(arr).tobytes()
        header[n] = {"dtype": dt, "shape": list(np.shape(arr)), "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hj = json.dumps(header).encode()
    hj += b" " * (-len(hj) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)) + hj + b"".join(blobs))


def read_st(path):
    h, base = rq.read_header(path)
    raw = open(path, "rb").read()
    return {n: (i["dtype"], i["shape"], raw[base + i["data_offsets"][0]:base + i["data_offsets"][1]])
            for n, i in h.items() if n != "__metadata__"}


g = np.random.default_rng(3)
H = 64
src_w = {
    "lm_head": g.standard_normal((96, H)).astype(np.float32) * 0.05,
    "backbone.layers.0.mixer.in_proj": g.standard_normal((80, H)).astype(np.float32) * 0.03,
    "backbone.layers.0.mixer.out_proj": g.standard_normal((H, 48)).astype(np.float32) * 0.03,
    "backbone.layers.1.mixer.q_proj": g.standard_normal((H, H)).astype(np.float32) * 0.03,
}
shard1 = [("backbone.embeddings.weight", "BF16", bf16(g.standard_normal((96, H)) * 0.1)),
          ("backbone.layers.0.mixer.in_proj.weight", "BF16", bf16(src_w["backbone.layers.0.mixer.in_proj"])),
          ("backbone.layers.0.mixer.conv1d.weight", "BF16", bf16(g.standard_normal((48, 1, 4)))),
          ("backbone.layers.0.mixer.out_proj.weight", "BF16", bf16(src_w["backbone.layers.0.mixer.out_proj"])),
          ("backbone.layers.0.norm.weight", "BF16", bf16(np.ones(H)))]
shard2 = [("backbone.layers.1.mixer.q_proj.weight", "BF16", bf16(src_w["backbone.layers.1.mixer.q_proj"])),
          ("backbone.layers.2.mixer.up_proj.weight", "U8", g.integers(0, 256, (32, H // 2), dtype=np.uint8)),
          ("backbone.layers.2.mixer.up_proj.weight_scale", "F8_E4M3", g.integers(40, 60, (32, H // 16), dtype=np.uint8)),
          ("backbone.layers.2.mixer.up_proj.weight_scale_2", "F32", np.float32(1e-4)),
          ("backbone.layers.2.mixer.up_proj.input_scale", "F32", np.float32(0.5)),
          ("lm_head.weight", "BF16", bf16(src_w["lm_head"]))]
shard3 = [("backbone.norm_f.weight", "BF16", bf16(np.ones(H)))]
excl = ["lm_head", "backbone.layers.0.mixer.in_proj", "backbone.layers.0.mixer.out_proj", "backbone.layers.0.mixer.conv1d",
        "backbone.layers.1.mixer.q_proj"]
with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, "src")
    os.makedirs(src)
    write_st(os.path.join(src, "model-00001-of-00002.safetensors"), shard1)
    side1 = {"metadata": {"total_size": 1}, "weight_map": {n: "model-00001-of-00002.safetensors" for n, _, _ in shard1}}
    json.dump(side1, open(os.path.join(src, "model-00001-of-00002.json"), "w"))  # ModelOpt's per-shard index
    side3 = '{"metadata": {"total_size": 0}, "weight_map": {"backbone.norm_f.weight": "model-00003-of-00003.safetensors"}}'
    open(os.path.join(src, "model-00003-of-00003.json"), "w").write(side3)
    write_st(os.path.join(src, "model-00002-of-00002.safetensors"), shard2)
    write_st(os.path.join(src, "model-00003-of-00003.safetensors"), shard3)  # nothing to quantize: cloned
    wm = {n: "model-00001-of-00002.safetensors" for n, _, _ in shard1}
    wm.update({n: "model-00002-of-00002.safetensors" for n, _, _ in shard2})
    wm.update({n: "model-00003-of-00003.safetensors" for n, _, _ in shard3})
    json.dump({"metadata": {"total_size": 0}, "weight_map": wm}, open(os.path.join(src, "model.safetensors.index.json"), "w"))
    qc = {"producer": {"name": "modelopt"},
          "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None, "group_size": 16, "exclude_modules": excl}}
    json.dump(qc, open(os.path.join(src, "hf_quant_config.json"), "w"))
    json.dump({"model_type": "nemotron_h"}, open(os.path.join(src, "config.json"), "w"))
    open(os.path.join(src, "tokenizer.json"), "w").write("{}")

    def run(out, *extra):
        return subprocess.run([sys.executable, TOOL, src, out, *extra], capture_output=True, text=True)

    r = run(os.path.join(td, "dry"), "--dry-run")
    check(r.returncode == 0 and "3 modules to NVFP4" in r.stdout and "lm_head" not in r.stdout and
          not os.path.exists(os.path.join(td, "dry")), "--dry-run: 3 modules, the head kept, nothing written",
          r.stdout + r.stderr)
    r = run(os.path.join(td, "dry"), "--dry-run", "--head")
    check(r.returncode == 0 and "4 modules to NVFP4" in r.stdout, "--dry-run --head: 4 modules", r.stdout + r.stderr)
    out = os.path.join(td, "out")
    r = run(out, "--head")
    check(r.returncode == 0, "run on the fake checkpoint", r.stderr[-500:])
    o1, o2 = (read_st(os.path.join(out, f"model-0000{i}-of-00002.safetensors")) for i in (1, 2))
    s1, s2 = read_st(os.path.join(src, "model-00001-of-00002.safetensors")), read_st(os.path.join(src, "model-00002-of-00002.safetensors"))
    allo, alls = {**o1, **o2}, {**s1, **s2}
    for m, w in src_w.items():
        p, s, ws2 = rq.quantize((bf16(w).astype(np.uint32) << 16).view(np.float32))
        ok = (allo[m + ".weight"][0] == "U8" and allo[m + ".weight"][2] == p.tobytes() and
              allo[m + ".weight_scale"] == ("F8_E4M3", [w.shape[0], w.shape[1] // 16], s.tobytes()) and
              allo[m + ".weight_scale_2"] == ("F32", [], np.float32(ws2).tobytes()) and m + ".input_scale" not in allo)
        check(ok, f"{m}: NVFP4 as quantize() of its BF16 weight, no input_scale")
    copied = [n for n in alls if n[:-7] not in src_w]
    check(all(allo[n] == alls[n] for n in copied) and len(copied) == 7, "the other 7 tensors copied byte for byte")
    check(set(o1) == {n for n in s1 if n[:-7] not in src_w} | {m + x for m in src_w for x in (".weight", ".weight_scale", ".weight_scale_2") if wm[m + ".weight"].startswith("model-00001")},
          "each module stays in its shard")
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    sizes = sum(os.path.getsize(os.path.join(out, f)) - rq.read_header(os.path.join(out, f))[1] for f in set(idx["weight_map"].values()))
    check(set(idx["weight_map"]) == set(allo) | {"backbone.norm_f.weight"} and all(n in read_st(os.path.join(out, f)) for n, f in idx["weight_map"].items())
          and idx["metadata"]["total_size"] == sizes, "index: every tensor, its shard, total_size")
    qo = json.load(open(os.path.join(out, "hf_quant_config.json")))
    check(qo["quantization"]["exclude_modules"] == ["backbone.layers.0.mixer.conv1d"] and qo["producer"] == qc["producer"],
          "hf_quant_config: only conv1d still excluded")
    rj = json.load(open(os.path.join(out, "requant.json")))
    check(sorted(rj["quantized"]) == sorted(src_w) and rj["source_exclude_modules"] == excl, "requant.json lists the modules")
    check(open(os.path.join(out, "tokenizer.json")).read() == "{}" and os.path.exists(os.path.join(out, "config.json")),
          "other files copied")
    sha = lambda d: {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d))}
    out2 = os.path.join(td, "out2")
    run(out2, "--head")
    check(sha(out) == sha(out2), "a second run writes the same bytes")
    sd = json.load(open(os.path.join(out, "model-00001-of-00002.json")))
    o1h = rq.read_header(os.path.join(out, "model-00001-of-00002.safetensors"))[0]
    check(sorted(sd["weight_map"]) == sorted(n for n in o1h if n != "__metadata__") and
          sd["metadata"]["total_size"] == idx["metadata"]["total_size"] - sum(
              os.path.getsize(os.path.join(out, f)) - rq.read_header(os.path.join(out, f))[1]
              for f in ("model-00002-of-00002.safetensors", "model-00003-of-00003.safetensors")),
          "a rewritten shard's per-shard index: its tensors and size")
    check(open(os.path.join(out, "model-00003-of-00003.json")).read() == side3, "an unchanged shard's index as it was")
    s3 = "model-00003-of-00003.safetensors"
    check(sha(out)[s3] == hashlib.sha256(open(os.path.join(src, s3), "rb").read()).hexdigest() and
          idx["weight_map"]["backbone.norm_f.weight"] == s3, "a shard with nothing to quantize: the same file")
    check(rj["shards_sha256"] == {f: h for f, h in sha(out).items() if f.endswith(".safetensors")}, "requant.json shard sha256")
    out3 = os.path.join(td, "out3")
    r = run(out3, "--keep", "q_proj")  # the head stays BF16 without --head
    o = {**read_st(os.path.join(out3, "model-00001-of-00002.safetensors")), **read_st(os.path.join(out3, "model-00002-of-00002.safetensors"))}
    check(r.returncode == 0 and o["lm_head.weight"] == alls["lm_head.weight"] and
          o["backbone.layers.1.mixer.q_proj.weight"] == alls["backbone.layers.1.mixer.q_proj.weight"] and
          o["backbone.layers.0.mixer.in_proj.weight"][0] == "U8" and
          json.load(open(os.path.join(out3, "hf_quant_config.json")))["quantization"]["exclude_modules"] ==
          ["lm_head", "backbone.layers.0.mixer.conv1d", "backbone.layers.1.mixer.q_proj"], "the head and --keep's modules stay BF16")
    check(json.load(open(os.path.join(out3, "requant.json")))["head"] is False, "requant.json: head false")
    qc["quantization"]["quant_algo"] = "W4A16_NVFP4"  # (ModelOpt's name for the same weights, Dory's)
    json.dump(qc, open(os.path.join(src, "hf_quant_config.json"), "w"))
    check(run(os.path.join(td, "w4a16"), "--dry-run").returncode == 0, "accepts W4A16_NVFP4")
    qc["quantization"]["quant_algo"] = "FP8"
    json.dump(qc, open(os.path.join(src, "hf_quant_config.json"), "w"))
    r = run(os.path.join(td, "fp8"))
    check(r.returncode != 0 and "not a ModelOpt NVFP4" in r.stderr, "refuses a checkpoint that is not NVFP4", r.stderr)
    check(run(src).returncode != 0, "refuses to write over its input")

print("all passed" if not FAILED else f"{len(FAILED)} failed: {FAILED}")
sys.exit(1 if FAILED else 0)
