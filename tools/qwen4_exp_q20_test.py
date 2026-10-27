#!/usr/bin/env python3
"""tools/qwen4_exp_q20.py on a tiny synthetic GGUF and MLX pack (no downloads, no GPU):
  * expert planes: the GGUF's Q2_0 blocks re-laid out byte for byte; Kapitsa's reading of the planes (MLX 2-bit order,
    scale d, bias -d) equals the GGUF's reading of the blocks exactly; no .biases; config.json's q2_0 entries;
  * every other tensor and mtp.safetensors byte for byte the pack's; the index lists every output tensor;
  * --verify passes, and refuses a GGUF whose gate and up are swapped;
  * the check command; a sparse table: the chosen rows equal the pack's, the others read zero, holes on disk; rows
    added later (the rows command) write only the missing ones and only into the pack's table;
  * --table gguf: the GGUF's table part (IQ4_NL) cloned as ple.gguf, no ple.safetensors or ple_manifest.json, its rows
    verified against the pack's (a table shifted by one row is refused), and the rows command refuses the folder.
  python3 tools/qwen4_exp_q20_test.py [--keep DIR]   (--keep: write the synthetic pack and its conversion to DIR,
  e.g. for build/fn_q2_bench pack DIR/out)
"""
import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen4_exp_pack as P  # noqa: E402
import qwen4_exp_q20 as Q  # noqa: E402

D, I, SI, E, LAYERS = 256, 128, 128, 8, 2  # hidden, expert width, shared width, experts, layers (tiny, kernel-legal)
rng = np.random.default_rng(7)
fails = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        fails.append(what)


def q2_0_encode(w):
    """float32 [..., K] -> GGUF Q2_0 bytes (d = max |w| / 2 per 64, codes round(w / d) + 1 in 0..3)."""
    shp = w.shape
    blocks = w.reshape(-1, 64)
    d = (np.abs(blocks).max(axis=1) / 2.0).astype(np.float16)
    dd = np.where(d == 0, 1, d).astype(np.float32)
    c = np.clip(np.rint(blocks / dd[:, None]) + 1, 0, 3).astype(np.uint8)
    qs = c.reshape(-1, 16, 4)
    packed = (qs[..., 0] | (qs[..., 1] << 2) | (qs[..., 2] << 4) | (qs[..., 3] << 6)).astype(np.uint8)
    out = np.concatenate([d.view(np.uint8).reshape(-1, 2), packed], axis=1)
    assert out.shape[1] == 18 and out.shape[0] * 64 == np.prod(shp)
    return out.tobytes()


def rtn4(w, gs=64):
    """float32 [..., K] -> MLX 4-bit affine group gs (codes uint32, scales / biases bf16), round to nearest."""
    g = w.reshape(-1, gs)
    lo, hi = g.min(axis=1), g.max(axis=1)
    s = np.maximum((hi - lo) / 15.0, 1e-8)
    q = np.clip(np.rint((g - lo[:, None]) / s[:, None]), 0, 15).astype(np.uint32).reshape(w.shape[:-1] + (-1,))
    words = np.zeros(w.shape[:-1] + (w.shape[-1] // 8,), np.uint32)
    for j in range(8):
        words |= q[..., j::8] << np.uint32(4 * j)
    sh = w.shape[:-1] + (w.shape[-1] // gs,)
    return words, P.f32_to_bf16(s).reshape(sh), P.f32_to_bf16(lo).reshape(sh)


def write_st(path, tensors):
    """safetensors from {name: numpy array} (dtype from the array; uint16 arrays are written as BF16)."""
    dt = {np.dtype("uint32"): "U32", np.dtype("uint16"): "BF16", np.dtype("float16"): "F16",
          np.dtype("float32"): "F32"}
    specs = [(k, dt[v.dtype], list(v.shape)) for k, v in tensors.items()]
    w = P.SafetensorsWriter(path, specs)
    for k, v in tensors.items():
        w.write(k, v)


def write_gguf(path, tensors, kv=None):
    """A GGUF v3 file: tensors = [(name, dims (ne order), ggml type, raw bytes)], 32-byte alignment."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = kv or {"general.architecture": "qwen4exp"}
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, v in kv.items():
        head += s(k) + struct.pack("<I", 8) + s(v)
    off, infos, datas = 0, b"", []
    for name, dims, typ, raw in tensors:
        infos += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims)
        infos += struct.pack("<IQ", typ, off)
        datas.append((off, raw))
        off += (len(raw) + 31) // 32 * 32
    head += infos
    data0 = (len(head) + 31) // 32 * 32
    with open(path, "wb") as f:
        f.write(head + b"\0" * (data0 - len(head)))
        for o, raw in datas:
            f.seek(data0 + o)
            f.write(raw)


def iq4_nl_encode(x):
    """float32 [rows, dim] -> GGUF IQ4_NL bytes (per 32 values d = the extreme / its level, codes the nearest level)."""
    g = x.reshape(-1, 32)
    ext = g[np.arange(len(g)), np.abs(g).argmax(axis=1)]
    d = (ext / -127.0).astype(np.float16)
    dd = np.where(d == 0, 1, d).astype(np.float32)
    c = np.abs(g[..., None] / dd[:, None, None] - Q.IQ4_LEVELS).argmin(axis=-1).astype(np.uint8)
    q = c[:, :16] | (c[:, 16:] << 4)
    return np.concatenate([d.view(np.uint8).reshape(-1, 2), q], axis=1).tobytes()


def table_values(pack):
    """The synthetic pack's table rows, dequantized: float32 [rows, 160]."""
    t = P.Local(pack)
    sh = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{}."
    return np.concatenate([P.dequantize_np(t.array(sh.format(k) + "weight"), t.array(sh.format(k) + "scales"),
                                           t.array(sh.format(k) + "biases"), 4, 32) for k in range(2)])


def make_sources(root, swap_gate_up=False):
    """The synthetic pack (root/pack) and GGUF (root/q2_0.gguf); the same float weights quantized both ways."""
    pack = os.path.join(root, "pack")
    os.makedirs(pack, exist_ok=True)
    gg = []
    files = {}
    for L in range(LAYERS):
        t = {}
        t[f"language_model.model.layers.{L}.mlp.gate.weight"] = rng.integers(0, 2**32, (E, D // 4), dtype=np.uint32)
        t[f"language_model.model.layers.{L}.mlp.gate.scales"] = P.f32_to_bf16(rng.random((E, D // 64), np.float32))
        t[f"language_model.model.layers.{L}.mlp.gate.biases"] = P.f32_to_bf16(rng.random((E, D // 64), np.float32))
        t[f"language_model.model.layers.{L}.post_attention_layernorm.weight"] = P.f32_to_bf16(rng.random(D, np.float32))
        for mat in ("gate", "up", "down"):
            N, K = (I, D) if mat != "down" else (D, I)
            w = (rng.standard_normal((E, N, K)) * 0.02).astype(np.float32)
            words, sc, bi = rtn4(w)
            base = Q.EXPERT.format(L, mat)
            t[base + ".weight"], t[base + ".scales"], t[base + ".biases"] = words, sc, bi
            gname = Q.GGUF_EXPERT[{"gate": "up", "up": "gate"}.get(mat, mat) if swap_gate_up else mat].format(L)
            gg.append((gname, [K, N, E], Q.Q2_0, q2_0_encode(w)))
        for mat in ("gate", "up", "down"):  # the shared expert, 4-bit
            N, K = (SI, D) if mat != "down" else (D, SI)
            w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
            words, sc, bi = rtn4(w)
            base = f"language_model.model.layers.{L}.mlp.shared_expert.{mat}_proj"
            t[base + ".weight"], t[base + ".scales"], t[base + ".biases"] = words, sc, bi
        files[f"model-{L + 1:05d}-of-{LAYERS:05d}.safetensors"] = t
    for f, t in files.items():
        write_st(os.path.join(pack, f), t)
    write_st(os.path.join(pack, "mtp.safetensors"),
             {"mtp.layers.0.mlp.switch_mlp.gate_proj.weight": rng.integers(0, 2**32, (E, I, D // 8), dtype=np.uint32)})
    shard = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{}"
    table = {}
    for k in range(2):  # 4-bit group 32, 160 values a row (the pack's layout)
        words, sc, bi = rtn4((rng.standard_normal((1000, 160)) * 0.05).astype(np.float32), 32)
        table[shard.format(k) + ".weight"], table[shard.format(k) + ".scales"], table[shard.format(k) + ".biases"] = (
            words, sc, np.where(bi == 0, np.uint16(1), bi))  # (no all-zero row: the sparse checks read zeros as holes)
    write_st(os.path.join(pack, "ple.safetensors"), table)
    q = {"group_size": 64, "bits": 4, "mode": "affine", "language_model.model.layers.0.mlp.gate": {"group_size": 64,
                                                                                                  "bits": 8}}
    json.dump({"model_type": "qwen4_exp", "quantization": q, "quantization_config": dict(q),
               "text_config": {"hidden_size": D, "moe_intermediate_size": I, "num_experts": E}},
              open(os.path.join(pack, "config.json"), "w"))
    for f in ("tokenizer.json", "chat_template.jinja", "LICENSE", "ple_manifest.json"):
        open(os.path.join(pack, f), "w").write(f"synthetic {f}\n")
    path = os.path.join(root, "q2_0_swapped.gguf" if swap_gate_up else "q2_0.gguf")
    write_gguf(path, gg + [("output_norm.weight", [D], 0, rng.random(D, np.float32).tobytes())])
    return pack, path


def args(**k):
    a = dict(out=None, pack=None, pack_repo=None, gguf=None, gguf_repo=None, table="link", rows_ids=None, stream=False,
             verify=4, conns=4)
    a.update(k)
    return types.SimpleNamespace(**a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep")
    root = ap.parse_args().keep
    tmp = None
    if not root:
        tmp = tempfile.TemporaryDirectory()
        root = tmp.name
    os.makedirs(root, exist_ok=True)
    pack, gguf = make_sources(root)
    out = os.path.join(root, "out")
    print(f"synthetic pack {pack}, GGUF {gguf} -> {out}")
    Q.cmd_convert(args(out=out, pack=pack, gguf=gguf))

    src, dst = P.Local(pack), P.Local(out)
    g = Q.Gguf(lambda x, y: open(gguf, "rb").read()[x:y], gguf)
    raw_all = open(gguf, "rb").read()
    print("expert planes")
    for L in range(LAYERS):
        for mat, gname in Q.GGUF_EXPERT.items():
            En, N, K, off, nb = g.expert(gname.format(L))
            raw = raw_all[off:off + nb]
            base = Q.EXPERT.format(L, mat)
            codes, d = dst.array(base + ".weight"), dst.array(base + ".scales")
            c2, d2 = Q.q20_planes(raw, En, N, K)
            check(codes.dtype == np.uint32 and list(codes.shape) == [En, N, K // 16] and d.dtype == np.float16
                  and list(d.shape) == [En, N, K // 64], f"layer {L} {mat}: U32 [E, N, K/16] codes, F16 [E, N, K/64] d")
            check(np.array_equal(codes, c2) and np.array_equal(d.view("<u2"), d2.view("<u2")),
                  f"layer {L} {mat}: planes are the GGUF's bytes")
            check(np.array_equal(Q.q20_dequant_planes(codes, d), Q.q20_dequant_blocks(raw, En, N, K)),
                  f"layer {L} {mat}: Kapitsa's dequantization of the planes = the GGUF's of the blocks, exactly")
            check(base + ".biases" not in dst.index, f"layer {L} {mat}: no .biases")
    print("everything else")
    same = all(np.array_equal(np.asarray(src.array(k)).view(np.uint8), np.asarray(dst.array(k)).view(np.uint8))
               for k in src.index if not Q.expert_of(k) and k in dst.index)
    missing = [k for k in src.index if not Q.expert_of(k) and k not in dst.index]
    check(same and not missing, "every non-expert tensor (shared expert, router, norms, mtp, table) byte for byte")
    cfg = json.load(open(os.path.join(out, "config.json")))
    want = {"group_size": 64, "bits": 2, "mode": "q2_0"}
    check(all(cfg[k][Q.EXPERT.format(L, m)] == want for k in ("quantization", "quantization_config")
              for L in range(LAYERS) for m in Q.GGUF_EXPERT), "config.json: q2_0 entries in both blocks")
    check(cfg["quantization"]["language_model.model.layers.0.mlp.gate"]["bits"] == 8 and
          cfg["quantization"]["bits"] == 4, "config.json: the other entries kept")
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    check(set(idx) == set(dst.index), "the index lists every output tensor")
    check(open(os.path.join(out, "mtp.safetensors"), "rb").read() == open(os.path.join(pack, "mtp.safetensors"),
                                                                          "rb").read(), "mtp.safetensors unchanged")
    check(Q.cmd_check(types.SimpleNamespace(dir=out)) == 0, "the check command")

    print("verification against the 4-bit pack")
    root2 = os.path.join(root, "swapped")
    os.makedirs(root2, exist_ok=True)
    pack2, bad_gguf = make_sources(root2, swap_gate_up=True)  # its own pack: only gate and up are exchanged
    try:
        Q.cmd_convert(args(out=os.path.join(root2, "out"), pack=pack2, gguf=bad_gguf, verify=8))
        check(False, "a GGUF with gate and up swapped is refused")
    except ValueError as e:
        check("correlate" in str(e), f"a GGUF with gate and up swapped is refused ({e})")

    print("sparse table")
    sp = os.path.join(root, "sparse.safetensors")
    pk = Q.PackSource(args(pack=pack), None)
    rows = [0, 5, 999, 1000, 1500, 1999]
    size, nrows, wrote = Q.write_sparse_table(pk, sp, rows)
    check(wrote == len(rows), "sparse table: every chosen row written")
    d2 = os.path.join(root, "sparse")
    os.makedirs(d2, exist_ok=True)
    tp = os.path.join(d2, "ple.safetensors")
    shutil.move(sp, tp)

    def table_rows(want):  # (the rows in want equal the pack's, every other row reads zero)
        t_src, t_dst = P.Local(pack), P.Local(d2)
        ok, zero = True, True
        for r in range(nrows):
            k, j = divmod(r, 1000)
            for part in ("weight", "scales", "biases"):
                n = f"language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{k}.{part}"
                a, b = t_src.array(n)[j], t_dst.array(n)[j]
                if r in want:
                    ok &= np.array_equal(a, b)
                else:
                    zero &= not b.any()
        return ok, zero
    ok, zero = table_rows(set(rows))
    st = os.stat(tp)
    check(ok, "sparse table: the chosen rows equal the pack's")
    check(zero, "sparse table: every other row reads zero")
    check(st.st_size == os.path.getsize(os.path.join(pack, "ple.safetensors")), "sparse table: full apparent size")
    more = [5, 6, 1000, 1001, 1999]
    _, _, wrote = Q.write_sparse_table(pk, tp, more, add=True)
    ok, zero = table_rows(set(rows) | set(more))
    check(wrote == 2 and ok and zero, f"sparse table, rows added: only the 2 missing written ({wrote}), the earlier "
                                      "ones kept, the rest zero")
    try:
        Q.write_sparse_table(pk, os.path.join(out, "mtp.safetensors"), more, add=True)
        check(False, "rows are not added to a file other than the pack's table")
    except ValueError:
        check(True, "rows are not added to a file other than the pack's table")
    # streamed: the same rows cut from whole-tensor reads (pieces of 3 rows of codes, 24 of scales: rows share pieces)
    tp2 = os.path.join(root, "streamed.safetensors")
    _, _, wrote = Q.write_sparse_table(pk, tp2, rows, stream=True, piece=240)
    _, _, wrote2 = Q.write_sparse_table(pk, tp2, more, add=True, stream=True, piece=240)
    check(wrote == len(rows) and wrote2 == 2, f"streamed sparse table: {wrote} rows, then the 2 missing ({wrote2})")
    check(open(tp2, "rb").read() == open(tp, "rb").read(), "streamed sparse table: the same file as row by row")
    print("table from the GGUF")
    vals = table_values(pack)
    with open(gguf, "rb") as fh:
        first = fh.read()
    for f, x in (("split-00001-of-00002.gguf", None), ("split-00002-of-00002.gguf", vals),
                 ("shifted-00001-of-00002.gguf", None), ("shifted-00002-of-00002.gguf", np.roll(vals, 1, axis=0))):
        if x is None:
            open(os.path.join(root, f), "wb").write(first)
        else:
            write_gguf(os.path.join(root, f), [(Q.GGUF_TABLE, [160, len(x)], Q.IQ4_NL, iq4_nl_encode(x))],
                       {"split.no": "1"})
    part1, part2 = os.path.join(root, "split-00001-of-00002.gguf"), os.path.join(root, "split-00002-of-00002.gguf")
    g2 = Q.Gguf(lambda x, y: open(part2, "rb").read()[x:y], part2)
    rows, dim, _, toff, rb = g2.table()
    dec = Q.table_rows(open(part2, "rb").read()[toff:toff + rows * rb]).reshape(rows, dim)
    check(rows == len(vals) and np.corrcoef(dec.reshape(-1), vals.reshape(-1))[0, 1] > 0.95,
          "the IQ4_NL table decodes to the rows it encodes")
    outg = os.path.join(root, "out_gguf")
    Q.cmd_convert(args(out=outg, pack=pack, gguf=part1, table="gguf"))
    shutil.copytree(outg, os.path.join(root, "out_gguf2"))
    rec = json.load(open(os.path.join(outg, "conversion_q20.json")))
    check(open(os.path.join(outg, Q.GTABLE), "rb").read() == open(part2, "rb").read(),
          "ple.gguf is the GGUF's table part byte for byte")
    check(not os.path.exists(os.path.join(outg, Q.TABLE)) and not os.path.exists(os.path.join(outg, "ple_manifest.json"))
          and os.path.exists(os.path.join(out, "ple_manifest.json")), "no ple.safetensors or ple_manifest.json")
    check(rec["table"] == "gguf" and rec["table_gguf"] == part2 and rec["worst_row_correlation"]["table"] > 0.9,
          f"the record names the part and its rows' correlation ({rec['worst_row_correlation'].get('table')})")
    check(Q.cmd_check(types.SimpleNamespace(dir=outg)) == 0, "the check command (table from the GGUF)")
    gb = bytearray(open(os.path.join(outg, Q.GTABLE), "rb").read())
    gb[toff + 3] ^= 1
    os.remove(os.path.join(outg, Q.GTABLE))  # (a new file: ple.gguf may be a hard link of the part)
    open(os.path.join(outg, Q.GTABLE), "wb").write(gb)
    check(Q.cmd_check(types.SimpleNamespace(dir=outg)) == 1, "the check command finds a changed byte of ple.gguf")
    os.remove(os.path.join(outg, ".q20.done.json"))
    Q.cmd_convert(args(out=outg, pack=pack, gguf=gguf, table="link"))
    check(not os.path.exists(os.path.join(outg, Q.GTABLE)) and os.path.exists(os.path.join(outg, Q.TABLE)),
          "a later run with another --table removes ple.gguf")
    try:
        Q.cmd_convert(args(out=os.path.join(root, "out_shifted"), pack=pack,
                           gguf=os.path.join(root, "shifted-00001-of-00002.gguf"), table="gguf"))
        check(False, "a GGUF table shifted by one row is refused")
    except ValueError as e:
        check("correlate" in str(e), f"a GGUF table shifted by one row is refused ({e})")
    try:
        Q.cmd_rows(types.SimpleNamespace(dir=os.path.join(root, "out_gguf2"), ids=["@x"], stream=False, conns=2))
        check(False, "the rows command refuses a folder whose table is the GGUF's")
    except SystemExit as e:
        check("GGUF" in str(e), "the rows command refuses a folder whose table is the GGUF's")
    print(f"\n{'all checks passed' if not fails else f'{len(fails)} FAILED'}")
    if tmp:
        tmp.cleanup()
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
