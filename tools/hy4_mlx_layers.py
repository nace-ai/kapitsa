#!/usr/bin/env python3
"""Hy4-flash layer bisection on the MLX side: the pack's own MLX model truncated to its first N decoder layers.

  hy4_mlx_layers.py --pack DIR --ids FILE --layers N --out F.f32 [--hidden-out H.f32]
      Builds the model from DIR/config.json with the pack's own model file (config "model_file", hy_v4_flash.py) and
      num_hidden_layers forced to N (layer_types, mlp_layer_types and engram_layer_ids cut to the first N layers),
      loads only the weights those layers and the head need, runs the whole prompt as one forward pass without a
      cache and writes float32 [n, vocab] logits, row-major, every prompt row (row p is the distribution of token
      p + 1; no softmax). The head (hc_head, norm, lm_head in fp32) runs after layer N - 1 exactly as after the
      last layer of the full model. --hidden-out writes the residual streams after layer N - 1 (the hc_head input),
      float32 [n, hc_mult, hidden]. FILE: comma- or whitespace-separated token ids.
      Trunk weights load lazily (mx.load reads only the tensors used). The routed experts (2 GB per layer) and the
      Engram table (9.9 GB) are not loaded: their files are memory-mapped and only the experts the router picked and
      the table rows the prompt hashes to are read, then dequantized by the same MLX ops (gather_qmm, dequantize)
      the full modules would run, so the numbers are those of the stock modules. Each MoE layer is evaluated as it
      runs, so memory stays at the trunk plus one layer's picked experts. --full-load uses the stock modules with
      the whole tables instead (needs their size in RAM), to check that claim.
      --rows A:B writes only those rows. --prefill-step K runs the prompt through the model's own KV cache in
      chunks of K tokens (mlx-lm's prefill_step_size) instead of one pass without a cache. --dense makes every
      indexed layer attend densely at any length (a control for the compressed indexer past 2048 keys).
  hy4_mlx_layers.py --compare A.f32 B.f32 [--rows A:B] [--b-offset K] [--per-row]
      Per row KL(A || B) over softmax (float64), top-1 agreement, the max absolute logit difference and how many
      rows are bitwise identical.

Needs mlx and mlx_lm (the pack's model file imports it) and NumPy; run it under the GPU lock:
lockf -k -t 900 /tmp/kapitsa-gpu.lock <python> tools/hy4_mlx_layers.py ...
"""

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

VOCAB = 120832


def read_ids(path):
    s = Path(path).read_text().replace(",", " ").replace("[", " ").replace("]", " ")
    return [int(x) for x in s.split()]


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_memmap(path, keys):
    """name -> np.memmap of a safetensors tensor (bf16 as uint16)."""
    hdr, base = st_header(path)
    dt = {"U32": np.uint32, "BF16": np.uint16, "F32": np.float32, "I32": np.int32, "F16": np.float16}
    out = {}
    for k in keys:
        h = hdr[k]
        out[k] = (np.memmap(path, dtype=dt[h["dtype"]], mode="r", offset=base + h["data_offsets"][0],
                            shape=tuple(h["shape"])), h["dtype"])
    return out


def to_mx(a, dtype_name):
    import mlx.core as mx
    x = mx.array(np.ascontiguousarray(a))
    return x.view(mx.bfloat16) if dtype_name == "BF16" else x


def load_arch(pack, config):
    spec = importlib.util.spec_from_file_location("hy4_pack_model", pack / config["model_file"])
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    return arch


def build(pack, n_layers, full_load=False):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.utils import infer_quant_config

    config = json.loads((pack / "config.json").read_text())
    tc = dict(config["text_config"])
    total = tc["num_hidden_layers"]
    if not 1 <= n_layers <= total:
        sys.exit(f"--layers must be in 1..{total}")
    tc["num_hidden_layers"] = n_layers
    config["full_num_hidden_layers"] = total
    for key in ("layer_types", "mlp_layer_types"):
        if tc.get(key) is not None:
            tc[key] = tc[key][:n_layers]
    if tc.get("engram_layer_ids"):
        tc["engram_layer_ids"] = [i for i in tc["engram_layer_ids"] if i <= n_layers]  # 1-based
    config["text_config"] = tc
    config["dfly_target_kv_layers"] = []  # no drafter (model files that build it from this key)
    arch = load_arch(pack, config)
    model = arch.Model(arch.ModelArgs.from_dict(config))

    wmap = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]

    def wanted(k):
        if k.startswith("model.layers."):
            return int(k.split(".")[2]) < n_layers
        return not (k.startswith("model.dfly_layers.") or k.startswith("vit."))

    keys = [k for k in wmap if wanted(k)]
    mapped = {}  # module path -> {param: (file, key)}: the experts and the Engram table, read by memmap
    lazy_keys = {}
    for k in keys:
        mod, param = k.rsplit(".", 1)
        if not full_load and (".mlp.switch_mlp." in k or ".multi_head_embedding.embedding." in k):
            mapped.setdefault(mod, {})[param] = (wmap[k], k)
        else:
            lazy_keys.setdefault(wmap[k], []).append(k)
    weights = {}
    for fname, ks in lazy_keys.items():
        f = pack / fname
        if not f.exists():
            sys.exit(f"{f} is missing (still downloading?)")
        allw = mx.load(str(f))  # lazy: only the tensors evaluated are read
        weights.update({k: allw[k] for k in ks})
    for params in mapped.values():
        for fname, _ in params.values():
            if not (pack / fname).exists():
                sys.exit(f"{pack / fname} is missing (still downloading?)")
    weights = model.sanitize(weights)

    quant = config["quantization"]
    shapes = {}  # scales/weight shapes of the mapped tensors, for infer_quant_config's fallback
    for mod, params in mapped.items():
        hdr, _ = st_header(pack / params["weight"][0])
        for p in ("weight", "scales"):
            shapes[f"{mod}.{p}"] = hdr[params[p][1]]

    def class_predicate(p, m):
        if p in quant:
            return quant[p]
        if not hasattr(m, "to_quantized"):
            return False
        if f"{p}.scales" in weights:
            return infer_quant_config(p, m, weights)
        if f"{p}.scales" in shapes:
            in_dims = m.weight.shape[-1]
            gs = in_dims // shapes[f"{p}.scales"]["shape"][-1]
            return {"group_size": gs, "bits": shapes[f"{p}.weight"]["shape"][-1] * 32 // in_dims, "mode": "affine"}
        return False

    nn.quantize(model, group_size=quant["group_size"], bits=quant["bits"], mode=quant.get("mode", "affine"),
                class_predicate=class_predicate)
    model.eval()

    MappedEmbedding, MappedSwitch = _make_mapped_classes()
    for i, layer in enumerate(model.model.layers if not full_load else []):
        mlp = layer.mlp
        if hasattr(mlp, "switch_mlp"):
            path = f"model.layers.{i}.mlp.switch_mlp"
            mlp.switch_mlp = MappedSwitch(mlp.switch_mlp, pack, path, mapped)
        if layer.has_engram:
            e = layer.engram_layer.multi_head_embedding
            e.embedding = MappedEmbedding(e.embedding, pack, f"model.layers.{i}.engram_layer.multi_head_embedding"
                                          ".embedding", mapped)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, config


def _maps(pack, prefix, mapped):
    by_file = {}
    for p, (fname, key) in mapped[prefix].items():
        by_file.setdefault(fname, []).append((p, key))
    out = {}
    for fname, items in by_file.items():
        mm = st_memmap(pack / fname, [k for _, k in items])
        out.update({p: mm[k] for p, k in items})
    return out


def _make_mapped_classes():
    import mlx.core as mx
    import mlx.nn as nn

    class MappedEmbedding(nn.Module):
        """nn.QuantizedEmbedding whose table stays on disk: the gathered rows of weight/scales/biases are read from
        the memory-mapped file, then mx.dequantize exactly as QuantizedEmbedding.__call__ does."""

        def __init__(self, qe, pack, prefix, mapped):
            super().__init__()
            object.__setattr__(self, "_mm", _maps(pack, prefix, mapped))
            object.__setattr__(self, "_cfg", (qe.group_size, qe.bits, qe.mode))

        def __call__(self, x):
            idx = np.array(x).astype(np.int64)
            flat = idx.reshape(-1)
            g = {p: to_mx(a[flat], d).reshape(*idx.shape, a.shape[-1]) for p, (a, d) in self._mm.items()}
            gs, bits, mode = self._cfg
            return mx.dequantize(g["weight"], scales=g["scales"], biases=g.get("biases"), group_size=gs, bits=bits,
                                 mode=mode)

    class MappedSwitch(nn.Module):
        """SwitchGLU whose experts stay on disk: per call, the experts the router picked are read from the
        memory-mapped file into a SwitchGLU of that many experts (same quantization), indices remapped in order,
        and the stock SwitchGLU runs on it. The output is evaluated before returning so the slices can go."""

        def __init__(self, sw, pack, prefix, mapped):
            super().__init__()
            object.__setattr__(self, "_sw", sw)
            mm = {proj: _maps(pack, f"{prefix}.{proj}", mapped) for proj in ("gate_proj", "up_proj", "down_proj")}
            object.__setattr__(self, "_mm", mm)

        def __call__(self, x, indices):
            idx = np.array(indices)
            uniq = np.unique(idx)
            remap = np.searchsorted(uniq, idx).astype(idx.dtype)
            sw = self._sw
            for proj, params in self._mm.items():
                lin = getattr(sw, proj)
                for p, (a, d) in params.items():
                    lin[p] = to_mx(a[uniq], d)
            y = sw(x, mx.array(remap))
            mx.eval(y)
            for proj in self._mm:
                lin = getattr(sw, proj)
                for p in self._mm[proj]:
                    lin[p] = mx.zeros((1,))
            return y

    return MappedEmbedding, MappedSwitch


def parse_rows(spec, n):
    if not spec:
        return 0, n
    lo, _, hi = spec.partition(":")
    lo, hi = int(lo or 0), int(hi or n)
    if not 0 <= lo < hi <= n:
        sys.exit(f"--rows {spec} is outside 0:{n}")
    return lo, hi


def run(args):
    import mlx.core as mx

    t0 = time.time()
    pack = Path(args.pack).expanduser()
    ids = read_ids(args.ids)
    n = len(ids)
    lo, hi = parse_rows(args.rows, n)
    model, config = build(pack, args.layers, args.full_load)
    if args.dense:  # control: every indexed layer attends densely whatever the context
        for layer in model.model.layers:
            layer.self_attn.candidate_topk = 1 << 40
    t1 = time.time()

    captured = []
    head = model.model.hc_head
    orig = type(head).__call__

    class Tap(type(head)):
        def __call__(self, x):
            captured.append(x)
            return orig(self, x)

    head.__class__ = Tap
    sel = install_sel_tap(model, lo, hi) if args.sel_out else None
    step = args.prefill_step or n
    cache = model.make_cache() if args.prefill_step else None
    rows, hidden, dtype = [], [], None
    for c0 in range(0, n, step):
        c1 = min(n, c0 + step)
        logits = model(mx.array([ids[c0:c1]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        dtype = logits.dtype
        a, b = max(lo, c0), min(hi, c1)
        if a < b and (args.out or args.sha_out):
            r = np.array(logits[0, a - c0 : b - c0].astype(mx.float32), dtype=np.float32)
            rows.append([hashlib.sha256(x.tobytes()).hexdigest() for x in r] if not args.out else r)
            if args.hidden_out:
                hidden.append(np.array(captured[-1][0, a - c0 : b - c0].astype(mx.float32), dtype=np.float32))
        act = captured[-1].dtype
        captured.clear()
        del logits
        if cache is not None:
            print(f"  prefill {c0}..{c1 - 1}: {time.time() - t1:.1f} s, peak {mx.get_peak_memory() / 2**30:.2f} GiB",
                  flush=True)
    t2 = time.time()

    if sel is not None:
        np.savez(args.sel_out, **{f"{k}_{i}": v for i, d in sel.items() for k, v in d.items()})
        print(f"selections: {args.sel_out} (layers {list(sel)}, rows {lo}..{hi - 1})")
    if args.sha_out:
        sha = [x for chunk in rows for x in chunk] if not args.out else \
            [hashlib.sha256(x.tobytes()).hexdigest() for x in np.concatenate(rows)]
        Path(args.sha_out).write_text("\n".join(sha) + "\n")
        print(f"row hashes: {args.sha_out} (sha256 of each float32 row, rows {lo}..{hi - 1})")
    if not args.out:
        return
    out = np.concatenate(rows)
    assert out.shape == (hi - lo, VOCAB), out.shape
    out.tofile(args.out)
    if args.hidden_out:
        h = np.concatenate(hidden)
        h.tofile(args.hidden_out)
        print(f"hidden: {args.hidden_out} float32 {list(h.shape)} (residual streams after layer {args.layers - 1})")
    lt = model.text_args.layer_types
    eng = [i for i in range(args.layers) if model.model.layers[i].has_engram]
    print(f"model: {args.layers} of {config['full_num_hidden_layers']} layers, attention {lt}, "
          f"mlp {model.text_args.mlp_layer_types}, engram layers (0-based) {eng}")
    print(f"mode: {'dense control' if args.dense else 'indexer on'}, "
          f"{f'cache, prefill step {step}' if cache is not None else 'one pass, no cache'}, {n} tokens")
    print(f"precision: activations {act}, embeddings dequantized to {model.model.embed_tokens.scales.dtype}; "
          f"hc/router/softmax fp32; lm_head fp32 (logits {dtype})")
    print(f"wrote {args.out}: float32 [{hi - lo}, {VOCAB}] = rows {lo}..{hi - 1}")
    for r in sorted({lo, hi - 1}):
        row = out[r - lo]
        print(f"row {r}: argmax {int(row.argmax())} logit {row.max():.4f}")
    print(f"time: load {t1 - t0:.2f} s, forward {t2 - t1:.2f} s, total {time.time() - t0:.2f} s; "
          f"peak MLX memory {mx.get_peak_memory() / 2**30:.2f} GiB")


def install_sel_tap(model, lo, hi):
    """Record each indexed layer's group selection (and its group scores) for positions lo..hi-1: returns
    {layer: {"pos": [rows], "sel": bool [rows, groups], "logits": fp32 [rows, groups]}}, filled as the model runs."""
    import mlx.core as mx

    store = {}
    for i, layer in enumerate(model.model.layers):
        ci = getattr(layer.self_attn, "compressed_indexer", None)
        if ci is None or not hasattr(ci, "group_logits"):
            continue
        orig = ci.recall

        def recall(x, q_resid, group_keys, positions, n_keys, i=i, ci=ci, orig=orig):
            mask = orig(x, q_resid, group_keys, positions, n_keys)
            pos = np.array(positions)
            keep = (pos >= lo) & (pos < hi)
            if keep.any():
                lg = np.array(ci.group_logits(x, q_resid, group_keys, positions)[0].astype(mx.float32))[keep]
                sg = np.array(mask[0, :, :: ci.ratio])[keep]
                d = store.setdefault(i, {"pos": [], "sel": [], "logits": []})
                d["pos"].append(pos[keep])
                d["sel"].append(sg)
                d["logits"].append(lg)
            return mask

        object.__setattr__(ci, "recall", recall)

    return SelView(store)


class SelView:
    """The selections recorded so far, per layer, padded to the widest group count."""

    def __init__(self, store):
        self.store = store

    def items(self):
        for i in sorted(self.store):
            d = self.store[i]
            g = max(a.shape[1] for a in d["sel"])
            sel = np.concatenate([np.pad(a, ((0, 0), (0, g - a.shape[1]))) for a in d["sel"]])
            lg = np.concatenate([np.pad(a, ((0, 0), (0, g - a.shape[1])), constant_values=-np.inf)
                                 for a in d["logits"]])
            yield i, {"pos": np.concatenate(d["pos"]), "sel": sel, "logits": lg}

    def __iter__(self):
        return iter(sorted(self.store))


def read_kapitsa_sel(path, groups=512):
    """NI_HY4_DUMP_SEL's file -> {layer: {pos: sorted group ids, the query's own group excluded}}."""
    raw = np.fromfile(path, dtype=np.int32)
    out, at = {}, 0
    while at < raw.size:
        layer, pos0, R = (int(v) for v in raw[at : at + 3])
        rows = raw[at + 3 : at + 3 + R * groups].reshape(R, groups).view(np.uint32)
        at += 3 + R * groups
        for t in range(R):
            G = (pos0 + t) // 4
            n = min(G, groups - 1) + 1
            ids = rows[t, :n].astype(np.int64)
            assert ids[-1] == G, (layer, pos0 + t, ids[-1], G)
            out.setdefault(layer, {})[pos0 + t] = ids[:-1]
    return out


def read_mlx_sel(path):
    """--sel-out's npz -> {layer: {pos: sorted group ids, the query's own group excluded}}."""
    z = np.load(path)
    out = {}
    for k in z.files:
        if k.startswith("pos_"):
            i = int(k.split("_")[1])
            sel = z[f"sel_{i}"]
            out[i] = {int(p): np.flatnonzero(sel[r, : int(p) // 4]) for r, p in enumerate(z[k])}
    return out


def sel_compare(mlx_npz, kapitsa_bin, rows=None):
    """Per layer: rows whose recalled group set is identical, mean Jaccard, and for the mismatched rows how
    close the swapped groups sit to the selection boundary in A's (MLX's) scores: the score gap and the rank
    distance (r: every swapped group lies within r ranks of the boundary between rank k-1 and k, k = 511).
    B may be Kapitsa's NI_HY4_DUMP_SEL file or another --sel-out npz (MLX against itself: the noise floor)."""
    z = np.load(mlx_npz)
    kap = read_mlx_sel(kapitsa_bin) if kapitsa_bin.endswith(".npz") else read_kapitsa_sel(kapitsa_bin)
    layers = sorted(int(k.split("_")[1]) for k in z.files if k.startswith("pos_"))
    for i in layers:
        pos, sel, lg = z[f"pos_{i}"], z[f"sel_{i}"], z[f"logits_{i}"]
        if rows:
            a, b = (int(v) for v in rows.split(":"))
            keep = (pos >= a) & (pos < b)
            pos, sel, lg = pos[keep], sel[keep], lg[keep]
        if i not in kap:
            print(f"layer {i}: not in {kapitsa_bin}")
            continue
        same = jac = 0
        n = 0
        gaps, rels, nswap, ranks, spacing = [], [], [], [], []
        for r, p in enumerate(pos):
            p = int(p)
            if p not in kap[i]:
                continue
            n += 1
            G = p // 4
            row = lg[r, :G]
            kk = min(G, 511)
            order = np.argsort(-row, kind="stable")
            if kk < G:
                spacing.append(row[order[kk - 1]] - row[order[kk]])
            m = set(np.flatnonzero(sel[r, :G]).tolist())
            k = set(kap[i][p].tolist())
            if m == k:
                same += 1
                jac += 1.0
                continue
            jac += len(m & k) / len(m | k)
            rank = np.empty(G, np.int64)
            rank[order] = np.arange(G)
            ranks.append(max(abs(rank[g] - (kk - 0.5)) for g in m ^ k))
            thr = row[sorted(m, key=lambda g: row[g])[0]]  # MLX's weakest recalled score
            only_k = sorted(k - m)
            only_m = sorted(m - k)
            nswap.append(len(only_k))
            worst = max(abs(row[g] - thr) for g in only_k + only_m)
            scale = np.abs(row[np.isfinite(row)]).max()
            gaps.append(worst)
            rels.append(worst / scale)
        print(f"layer {i}: rows {n}; identical selection {same}/{n} ({100 * same / max(n, 1):.1f}%); "
              f"mean Jaccard {jac / max(n, 1):.5f}")
        if gaps:
            gaps, rels, nswap = np.array(gaps), np.array(rels), np.array(nswap)
            print(f"  mismatched rows {len(gaps)}: groups swapped per row mean {nswap.mean():.2f} max {nswap.max()}; "
                  f"max |score - boundary| of a swapped group (MLX scores): median {np.median(gaps):.3e} "
                  f"max {gaps.max():.3e}; relative to the row's max |score|: median {np.median(rels):.2e} "
                  f"max {rels.max():.2e}")
            ranks = np.array(ranks)
            print(f"  boundary spacing (score k-1 minus score k, all rows): median {np.median(spacing):.3e}; "
                  f"swapped groups within r ranks of the boundary: r <= 1.5 {np.mean(ranks <= 1.5):.0%}, "
                  f"r <= 3.5 {np.mean(ranks <= 3.5):.0%}, r <= 10.5 {np.mean(ranks <= 10.5):.0%}, max {ranks.max()}")


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def compare(a_path, b_path, rows=None, per_row=False, b_offset=0):
    a = np.memmap(a_path, dtype=np.float32, mode="r")
    b = np.memmap(b_path, dtype=np.float32, mode="r")
    if a.size % VOCAB or b.size % VOCAB:
        sys.exit("file size is not a multiple of the vocab")
    a, b = a.reshape(-1, VOCAB), b.reshape(-1, VOCAB)
    n = min(len(a), len(b) - b_offset)
    if len(a) != len(b) and not rows:
        print(f"note: {len(a)} vs {len(b)} rows, comparing the first {n}")
    lo, hi = parse_rows(rows, n)
    kl, ta, tb, mad, finite, same = [], [], [], [], [], []
    for r0 in range(lo, hi, 64):
        r1 = min(hi, r0 + 64)
        x, y = np.asarray(a[r0:r1]), np.asarray(b[r0 + b_offset : r1 + b_offset])
        la, lb = log_softmax(x), log_softmax(y)
        kl.append((np.exp(la) * (la - lb)).sum(axis=-1))
        ta.append(x.argmax(-1))
        tb.append(y.argmax(-1))
        mad.append(np.abs(x.astype(np.float64) - y).max(axis=-1))
        finite.append(np.isfinite(x).all(-1) & np.isfinite(y).all(-1))
        same.append((x.view(np.uint32) == y.view(np.uint32)).all(-1))
    kl, ta, tb, mad, finite, same = map(np.concatenate, (kl, ta, tb, mad, finite, same))
    m = hi - lo
    if per_row or m <= 64:
        print(" row  KL(A||B)     top1 A  top1 B  max|dA-B|")
        for i in range(m):
            print(f"{lo + i:4d}  {kl[i]:.4e}  {ta[i]:7d} {tb[i]:7d}  {mad[i]:.4f}"
                  f"{'' if finite[i] else '  non-finite'}")
    k = kl[finite]
    print(f"rows {lo}..{hi - 1} ({m}): top-1 agreement {int((ta == tb).sum())}/{m}; KL(A||B) mean {k.mean():.4e} "
          f"p50 {np.median(k):.4e} p95 {np.percentile(k, 95):.4e} max {k.max():.4e} "
          f"(row {lo + int(np.argmax(np.where(finite, kl, -1)))}); max|A-B| {mad.max():.4f}; "
          f"bitwise identical rows {int(same.sum())}/{m}; non-finite rows {int((~finite).sum())}")
    return kl, ta == tb, mad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack")
    ap.add_argument("--ids")
    ap.add_argument("--layers", type=int)
    ap.add_argument("--out")
    ap.add_argument("--hidden-out")
    ap.add_argument("--sha-out", help="write the sha256 of every written row (one per line) instead of, or with, --out")
    ap.add_argument("--full-load", action="store_true",
                    help="load the experts and the Engram table whole through mx.load (stock modules), to check the "
                         "memory-mapped path against them")
    ap.add_argument("--rows", help="A:B, the rows to write (run) or to compare (--compare); default all")
    ap.add_argument("--prefill-step", type=int, default=0,
                    help="run through the model's KV cache in chunks of this many tokens (default: one pass, no cache)")
    ap.add_argument("--dense", action="store_true", help="control: indexed layers attend densely at any length")
    ap.add_argument("--sel-out", help="write the indexed layers' group selections and scores of --rows (npz)")
    ap.add_argument("--sel-compare", nargs=2, metavar=("MLX.npz", "B"),
                    help="compare --sel-out against Kapitsa's NI_HY4_DUMP_SEL file (or another --sel-out npz)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--per-row", action="store_true")
    ap.add_argument("--b-offset", type=int, default=0, help="--compare: row r of A against row r + K of B")
    args = ap.parse_args()
    if args.sel_compare:
        sel_compare(*args.sel_compare, rows=args.rows)
        return
    if args.compare:
        compare(*args.compare, rows=args.rows, per_row=args.per_row, b_offset=args.b_offset)
        return
    if not (args.pack and args.ids and args.layers and (args.out or args.sel_out or args.sha_out)):
        ap.error("--pack, --ids, --layers and --out, --sha-out or --sel-out are required (or --compare A B)")
    run(args)


if __name__ == "__main__":
    main()
