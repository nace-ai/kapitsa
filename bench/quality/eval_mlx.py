#!/usr/bin/env python3
"""Teacher-forced next-token distributions of a local MLX model (mlx-lm), the MLX side of Kapitsa's ni_eval.

  eval_mlx.py <model dir> [--ref F] [--dump-ref F] [--ref-model M] [--topk 256] [--bf16-logits] @ids...
  eval_mlx.py <model dir> --greedy N @prompt     greedy ids (and each step's top-2 margin) as JSON
  --f32: the model cast to float32 (activations in fp32), MLX without its bf16 rounding

Same reference file as tools/ni_eval.cpp (NIEVREF2: per position the top-K log-probs and the realized next token's
log-prob). --ref-model: a second MLX model in this process, exact KL over the vocabulary. Logits come from an fp32
LM head over MLX's final-norm hidden state (bf16 activations as mlx-lm runs them); --bf16-logits keeps mlx-lm's own
bf16 logits instead (their rounding alone is a KL of ~1e-4). Holds /tmp/kapitsa-gpu.lock while it computes (NI_LOCK_HELD=1: the caller holds it).
"""
import fcntl, json, os, struct, sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

args = sys.argv[1:]
def opt(name, default=None, flag=False):
    if name in args:
        i = args.index(name)
        if flag:
            del args[i]
            return True
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default
ref_file, dump_file, ref_model = opt("--ref"), opt("--dump-ref"), opt("--ref-model")
topk = int(opt("--topk", "256"))
greedy = opt("--greedy")
bf16_logits = opt("--bf16-logits", flag=True)
f32 = opt("--f32", flag=True)  # cast the model to float32 (activations in fp32): MLX without its bf16 rounding
model_dir, seqs = args[0], args[1:]


def read_ids(a):
    t = open(a[1:]).read() if a.startswith("@") else a
    return [int(x) for x in t.replace("\n", ",").split(",") if x.strip()]


ids = [read_ids(s) for s in seqs]
if not os.environ.get("NI_LOCK_HELD"):  # set by a caller that already holds the lock (lockf ... sh)
    lock = os.fdopen(os.open(os.path.expanduser("/tmp/kapitsa-gpu.lock"), os.O_RDONLY | os.O_CREAT, 0o644))
    fcntl.flock(lock, fcntl.LOCK_EX)


def text_model(m):
    return m.language_model if hasattr(m, "language_model") else m


def logprobs(m, seq):
    """[n, V] float32 log-probs of positions 0..n-1 (teacher forced)."""
    x = mx.array(seq)[None]
    if bf16_logits:
        lg = m(x).astype(mx.float32)[0]
    else:
        tm = text_model(m)
        h = tm.model(x)  # final-norm hidden, model dtype
        m_ = tm.model.embed_tokens if getattr(tm.args, "tie_word_embeddings", False) else tm.lm_head
        if hasattr(m_, "scales"):
            w = mx.dequantize(m_.weight, m_.scales, getattr(m_, "biases", None), m_.group_size, m_.bits,
                              mode=getattr(m_, "mode", "affine"), dtype=mx.float32)
        else:
            w = m_.weight.astype(mx.float32)
        lg = (h.astype(mx.float32) @ w.T)[0]
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    mx.eval(lp)
    return lp


model, tok = load(model_dir)
if f32:
    from mlx.utils import tree_map
    model.update(tree_map(lambda x: x.astype(mx.float32) if x.dtype == mx.bfloat16 else x, model.parameters()))
if greedy:
    seq = list(ids[0])
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    out, margins = [], []
    logits = model(mx.array(seq)[None], cache=cache)[0, -1].astype(mx.float32)
    for _ in range(int(greedy)):
        top2 = mx.topk(logits, 2)
        t = int(mx.argmax(logits).item())
        v = sorted(np.array(top2).tolist(), reverse=True)
        margins.append(round(v[0] - v[1], 4))
        out.append(t)
        logits = model(mx.array([[t]]), cache=cache)[0, -1].astype(mx.float32)
    print(json.dumps({"ids": out, "margins": margins}))
    sys.exit(0)

ref = None
if ref_model:
    ref, _ = load(ref_model)
refpos = None
if ref_file:
    f = open(ref_file, "rb")
    assert f.read(8) == b"NIEVREF2"
    nseq, V, K = struct.unpack("<3I", f.read(12))
    assert nseq == len(ids), "reference has other sequences"
    refpos = []
    for q in range(nseq):
        n = struct.unpack("<I", f.read(4))[0]
        rid = list(struct.unpack(f"<{n}i", f.read(4 * n)))
        assert rid == ids[q], f"sequence {q} differs from the reference"
        rec = np.frombuffer(f.read((n - 1) * (4 + 8 * K)), dtype=np.uint8).reshape(n - 1, 4 + 8 * K)
        lpn = rec[:, :4].copy().view(np.float32)[:, 0]
        pairs = rec[:, 4:].copy().view(np.uint32).reshape(n - 1, K, 2)
        refpos.append((lpn, pairs[:, :, 0].astype(np.int64), pairs[:, :, 1].copy().view(np.float32)))
dump = open(dump_file, "wb") if dump_file else None
if dump:
    dump.write(b"NIEVREF2" + struct.pack("<3I", len(ids), 0, topk))


def stats_line(name, kl, klt, agree, nll, nll_ref, n, exact):
    k = kl if exact else klt
    pct = lambda q: float(np.percentile(k, q)) if len(k) else 0.0
    return json.dumps({"seq": name, "positions": n, "kl_mean": float(np.mean(k)) if len(k) else 0.0,
                       "kl_median": pct(50), "kl_p95": pct(95), "kl_p99": pct(99), "kl_max": pct(100),
                       "kl_topk_mean": float(np.mean(klt)) if len(klt) else 0.0,
                       "top1_agree": agree / max(n, 1), "ppl": float(np.exp(nll / max(n, 1))),
                       "ppl_ref": float(np.exp(nll_ref / max(n, 1))) if nll_ref else 0.0, "exact": exact})


tot = dict(kl=[], klt=[], agree=0, nll=0.0, nll_ref=0.0, n=0)
for q, seq in enumerate(ids):
    n = len(seq)
    lp = logprobs(model, seq)[: n - 1]
    nxt = mx.array(seq[1:])
    tok_lp = mx.take_along_axis(lp, nxt[:, None], axis=-1)[:, 0]
    nll = -float(mx.sum(tok_lp).item())
    kl, klt, agree, nll_ref = [], [], 0, 0.0
    if dump:
        top_i = mx.argsort(-lp, axis=-1)[:, :topk]
        top_v = mx.take_along_axis(lp, top_i, axis=-1)
        mx.eval(top_i, top_v)
        ti, tv, tl = np.array(top_i).astype(np.uint32), np.array(top_v).astype(np.float32), np.array(tok_lp).astype(np.float32)
        dump.write(struct.pack("<I", n) + np.array(seq, dtype=np.int32).tobytes())
        for p in range(n - 1):
            dump.write(tl[p:p + 1].tobytes() + np.stack([ti[p], tv[p].view(np.uint32)], 1).astype(np.uint32).tobytes())
    if ref is not None:
        lr = logprobs(ref, seq)[: n - 1]
        pr = mx.exp(lr)
        klv = mx.sum(pr * (lr - lp), axis=-1)
        agree = int(mx.sum(mx.argmax(lr, -1) == mx.argmax(lp, -1)).item())
        nll_ref = -float(mx.sum(mx.take_along_axis(lr, nxt[:, None], axis=-1)).item())
        kl = np.maximum(np.array(klv), 0)
    if refpos is not None:
        lpn, ti, tv = refpos[q]
        lpa = np.array(lp)
        lq = np.take_along_axis(lpa, ti, axis=-1)
        p = np.exp(tv.astype(np.float64))
        rest_p = np.maximum(1 - p.sum(-1), 1e-12)
        rest_q = np.maximum(1 - np.exp(lq.astype(np.float64)).sum(-1), 1e-12)
        klt = np.maximum((p * (tv - lq)).sum(-1) + rest_p * (np.log(rest_p) - np.log(rest_q)), 0)
        if ref is None:
            agree = int((lpa.argmax(-1) == ti[:, 0]).sum())
            nll_ref = -float(lpn.sum())
    print(stats_line(seqs[q], kl, klt, agree, nll, nll_ref, n - 1, ref is not None), flush=True)
    tot["kl"] += list(kl)
    tot["klt"] += list(klt)
    tot["agree"] += agree
    tot["nll"] += nll
    tot["nll_ref"] += nll_ref
    tot["n"] += n - 1
print(stats_line("total", np.array(tot["kl"]), np.array(tot["klt"]), tot["agree"], tot["nll"], tot["nll_ref"], tot["n"],
                 ref is not None))
