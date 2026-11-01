#!/usr/bin/env python3
"""The qwen4_exp converter's n-gram table from a config.json (no downloads, no GPU):
  * qwen4_exp_pack.ngram_spec of Flash-Next's settings: the multipliers and head sizes its checkpoint carries
    (layer_multipliers, ngram_heads_vocab_sizes), 320,001,536 rows in 128 shards of 2,500,012, rows of 160 values;
  * ngram_ids: Flash-Next's ids as the 2- and 3-gram formula the converter used before; another geometry (5-grams of
    3 heads each) as src/model/ngram.cpp's NgramHash::ids computes them, row by row;
  * convert_qwen4_exp.table_layout: the layer and shards from the config, checked against a checkpoint's shard shapes
    (a wrong shape, a missing or an extra shard is refused); table_specs names the config's layer.
  python3 tools/qwen4_exp_pack_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import convert_qwen4_exp as C  # noqa: E402
import qwen4_exp_pack as P  # noqa: E402

fails = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what)
    if not cond:
        fails.append(what)


def refused(f):
    try:
        f()
    except SystemExit:
        return True
    return False


# Qwen/Qwen3.8-Flash-Next@de4b8e4: model.language_model.layers.1.ple.ple_embedding.{layer_multipliers,
# ngram_heads_vocab_sizes}, and the table's 128 shards of [2500012, 160]
MULT = [23703573157769, 20109073645365, 8052911324071]
SIZES = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
         20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]

ng = P.NGRAM
check(ng["multipliers"] == MULT, "Flash-Next's multipliers from its seed")
check(ng["sizes"] == SIZES, "Flash-Next's 16 head sizes: the primes from 20,000,000")
check(ng["offsets"][0] == 0 and ng["offsets"][-1] == sum(SIZES[:-1]), "the heads' rows end to end")
check((ng["rows"], ng["parts"], ng["shard_rows"], ng["width"]) == (320001536, 128, 2500012, 160),
      "320,001,536 rows in 128 shards of 2,500,012, rows of 160 values")


def flash_next_ids(seq, eos=248044):
    """the converter's formula before ngram_spec (Flash-Next's 2- and 3-grams, 8 heads each)"""
    t = np.asarray(seq, dtype=np.int64)
    pos = np.arange(len(t))
    last_eos = np.maximum.accumulate(np.where(t == eos, pos, -1))
    start = np.concatenate([[-1], last_eos[:-1]]) + 1

    def tok(k):
        src = pos - k
        return np.where((src >= 0) & (src >= start), t[np.maximum(src, 0)], eos)
    x2 = (tok(0) * MULT[0]) ^ (tok(1) * MULT[1])
    x3 = x2 ^ (tok(2) * MULT[2])
    sizes = np.array(SIZES, np.int64)
    offs = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    ids = np.empty((len(t), 16), np.int64)
    ids[:, :8] = x2[:, None] % sizes[None, :8] + offs[None, :8]
    ids[:, 8:] = x3[:, None] % sizes[None, 8:] + offs[None, 8:]
    return ids


def ngram_cpp_ids(seq, ng):
    """src/model/ngram.cpp's NgramHash::ids, one position at a time"""
    out = np.empty((len(seq), ng["heads"]), np.int64)
    for t in range(len(seq)):
        sh, cut = [seq[t]], False
        for k in range(1, ng["ngram"]):
            cut = cut or t - k < 0 or (k > 1 and seq[t - k + 1] == ng["eos"])
            sh.append(ng["eos"] if cut else seq[t - k])
        x = sh[0] * ng["multipliers"][0]
        for m in range(2, ng["ngram"] + 1):
            x ^= sh[m - 1] * ng["multipliers"][m - 1]
            for j in range(ng["per_ngram"]):
                hd = (m - 2) * ng["per_ngram"] + j
                out[t, hd] = x % ng["sizes"][hd] + ng["offsets"][hd]
    return out


rng = np.random.default_rng(3)
seq = rng.integers(0, 248320, 300)
seq[[0, 40, 41, 150, 299]] = 248044  # documents' ends, two in a row among them
check(np.array_equal(P.ngram_ids(seq), flash_next_ids(seq)), "Flash-Next's ids as before")
check(np.array_equal(P.ngram_ids(seq), ngram_cpp_ids([int(x) for x in seq], ng)), "Flash-Next's ids as ngram.cpp's")

other = P.ngram_spec({"vocab_size": 2048, "hidden_size": 480, "ngram_size": 5, "heads_per_ngram": 3,
                      "ngram_vocab_size_base": 1000, "make_ngram_vocab_size_divisible_by": 64,
                      "split_ngram_parts": 4, "eos_token_id": [2047, 2046], "seed": 99})
check((other["heads"], other["width"], other["eos"]) == (12, 40, 2047), "5-grams of 3 heads: 12 heads of 40 values")
check(all(m % 2 == 1 and m * 2048 < 2 ** 63 for m in other["multipliers"]), "odd multipliers below 2^63 / vocab")
check(other["rows"] % 64 == 0 and other["rows"] >= sum(other["sizes"]) > other["rows"] - 64
      and other["shard_rows"] * 4 >= other["rows"], "rows padded to 64, in 4 shards")
s2 = rng.integers(0, 2048, 200)
s2[[5, 6, 7, 100]] = 2047
check(np.array_equal(P.ngram_ids(s2, ng=other), ngram_cpp_ids([int(x) for x in s2], other)),
      "5-grams of 3 heads as ngram.cpp's")

# the converter's table layout: Flash-Next's from its config and shard shapes
cfg = {"text_config": dict(P.FLASH_NEXT_NGRAM, ple_layer_ids=[2])}
index = {C.TABLE_HF.format(layer=1, k=k): ("f", 0, 0, "BF16", [2500012, 160]) for k in range(128)}
layer, lng = C.table_layout(cfg, index)
check(layer == 1 and lng == ng, "Flash-Next's table: layer 1, its spec")
specs = C.table_specs(4, 32, layer, lng)
check(len(specs) == 3 * 128 and specs[0] == (C.TABLE_MLX.format(layer=1, k=0) + ".weight", "U32", [2500012, 20]),
      "table_specs: 128 shards of 4-bit group-32 rows of 160 in layer 1")
bad = dict(index)
bad[C.TABLE_HF.format(layer=1, k=7)] = ("f", 0, 0, "BF16", [2500012, 128])
check(refused(lambda: C.table_layout(cfg, bad)), "a shard of another width is refused")
bad = dict(index)
del bad[C.TABLE_HF.format(layer=1, k=127)]
check(refused(lambda: C.table_layout(cfg, bad)), "a missing shard is refused")
bad = dict(index)
bad[C.TABLE_HF.format(layer=1, k=128)] = ("f", 0, 0, "BF16", [2500012, 160])
check(refused(lambda: C.table_layout(cfg, bad)), "a shard past split_ngram_parts is refused")
moved = {"text_config": dict(P.FLASH_NEXT_NGRAM, ple_layer_ids=[5])}
mindex = {C.TABLE_HF.format(layer=4, k=k): v for k, v in enumerate(index.values())}
check(C.table_layout(moved, mindex)[0] == 4 and refused(lambda: C.table_layout(moved, index)),
      "the table in layer 5 (one-indexed) is read from layer 4's names, not layer 1's")
check(refused(lambda: C.table_layout({"text_config": dict(P.FLASH_NEXT_NGRAM, ple_layer_ids=[2, 30])})),
      "two n-gram layers are refused")
check(refused(lambda: C.table_layout({"text_config": dict(P.FLASH_NEXT_NGRAM, ple_layer_ids=[2], heads_per_ngram=6)})),
      "a hidden size the heads do not divide is refused")

print(f"{'FAIL' if fails else 'PASS'} ({len(fails)} failed)")
sys.exit(1 if fails else 0)
