#!/usr/bin/env python3
"""A small random Qwen3.8-Flash-Next (qwen4_exp) pack with Q2_0 routed experts, for engine tests that need no download:
the real model's structure and formats (4 hyper-connection streams, Gated DeltaNet and QSA attention layers, the PLE
n-gram table at layer 1, 512 routed experts of top-10 in Q2_0 beside a 4-bit shared expert, 8-bit routers, 6-bit
hyper-connection mixes, the MTP head with 4-bit experts) at small widths: hidden 512 (the folded hyper-connection
kernels of decode need a stream of at least 512: a 1024-input chunk spans at most two streams), experts of 64, 6
layers (the 4th full attention), a 2048-token vocabulary. Random weights: the outputs mean nothing, but every kernel of
a Flash-Next decode, verify and prefill runs on them, and the routed experts can be streamed (NI_EXPERT_STORE, src/
model/expert_store.h): 85 MB of experts in 32 KiB slots.
  python3 tools/fn_synth.py <out_dir> [--seed N] [--layers N] [--hidden N] [--tokenizer MODEL_DIR]
About 130 MB. --tokenizer: a Qwen model folder whose tokenizer files (tokenizer.json, tokenizer_config.json,
chat_template.jinja) the pack takes, with that model's vocabulary and end-of-turn token, so that `kapitsa serve` runs it
(engine tests: replies are random tokens, the same ones every time); about 330 MB then. --hidden: another width (a
multiple of 128 from 512; the real model's 2560 gives a split's rows their real size, 51 KB)."""
import argparse
import json
import os
import shutil
import struct

import numpy as np

D, S, RANK = 512, 4, 64            # hidden, residual streams, hyper-connection rank
E, K_USED, I_E, I_S = 512, 10, 64, 64  # experts, top-k, expert width, shared expert width
HK, HV, DK, DV = 1, 3, 128, 128    # Gated DeltaNet heads (Flash-Next: 16 / 48 of 128)
HQ, HKV, HD = 2, 1, 256            # attention heads (Flash-Next: 24 / 2 of 256)
IXH, IXD = 4, 128                  # QSA indexer heads x dim
V = 2048                           # vocabulary (eos = V - 1)
PER_NGRAM, NGRAM, NG_BASE = 4, 3, 1000  # n-gram hash: 8 heads of D / 8 = 64 values, ~8000 rows


def bf16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


class Writer:
    """one safetensors file; the data section starts on a 16 KiB page (as tools/qwen4_exp_pack.py writes them)"""

    def __init__(self):
        self.t = {}

    def add(self, name, dtype, shape, data):
        self.t[name] = (dtype, [int(x) for x in shape], np.ascontiguousarray(data).tobytes())

    def write(self, path):
        header, off = {}, 0
        for name in sorted(self.t):
            dtype, shape, b = self.t[name]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(b)]}
            off += len(b)
        header["__metadata__"] = {"format": "mlx"}
        h = json.dumps(header, separators=(",", ":")).encode()
        h += b" " * ((-(8 + len(h))) % 16384)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(h)))
            f.write(h)
            for name in sorted(self.t):
                f.write(self.t[name][2])


class Pack:
    def __init__(self, rng):
        self.rng = rng
        self.quant = {}  # module -> {"bits", "group_size"[, "mode"]} where not the default 4-bit group 64

    def affine(self, w, name, n, k, bits, out_std=1.0):
        """MLX affine bits-bit group 64 codes of a random [n, k] linear whose outputs have ~out_std for unit inputs"""
        rng = self.rng
        lv = (1 << bits) - 1
        s = out_std * 2.0 * np.sqrt(3.0) / (lv * np.sqrt(k))
        codes = rng.integers(0, lv + 1, size=(n, k), dtype=np.uint64)
        words = np.zeros((n, k * bits // 32), dtype=np.uint64)
        for j in range(k):  # little-endian bit stream along each row
            bit = j * bits
            words[:, bit // 32] |= (codes[:, j] << np.uint64(bit % 32)) & np.uint64(0xFFFFFFFF)
            if bit % 32 + bits > 32:
                words[:, bit // 32 + 1] |= codes[:, j] >> np.uint64(32 - bit % 32)
        g = k // 64
        sc = (s * rng.uniform(0.8, 1.2, size=(n, g))).astype(np.float32)
        w.add(name + ".weight", "U32", (n, k * bits // 32), words.astype(np.uint32))
        w.add(name + ".scales", "BF16", (n, g), bf16(sc))
        w.add(name + ".biases", "BF16", (n, g), bf16(-sc * lv / 2))
        if bits != 4:
            self.quant[name] = {"group_size": 64, "bits": bits}

    def experts4(self, w, name, n, k):
        """stacked [E, n, k] 4-bit group 64"""
        rng = self.rng
        words = rng.integers(0, 1 << 32, size=(E, n, k // 8), dtype=np.uint64).astype(np.uint32)
        s = (2.0 * np.sqrt(3.0) / (15 * np.sqrt(k)) * rng.uniform(0.8, 1.2, size=(E, n, k // 64))).astype(np.float32)
        w.add(name + ".weight", "U32", (E, n, k // 8), words)
        w.add(name + ".scales", "BF16", (E, n, k // 64), bf16(s))
        w.add(name + ".biases", "BF16", (E, n, k // 64), bf16(-7.5 * s))

    def experts_q20(self, w, name, n, k):
        """stacked [E, n, k] Q2_0: codes [E, n, k/16] U32 (MLX 2-bit order), d [E, n, k/64] F16, weight = (code - 1) d"""
        rng = self.rng
        words = rng.integers(0, 1 << 32, size=(E, n, k // 16), dtype=np.uint64).astype(np.uint32)
        d = (rng.uniform(0.8, 1.2, size=(E, n, k // 64)) / np.sqrt(1.5 * k)).astype(np.float16)
        w.add(name + ".weight", "U32", (E, n, k // 16), words)
        w.add(name + ".scales", "F16", (E, n, k // 64), d)
        self.quant[name] = {"group_size": 64, "bits": 2, "mode": "q2_0"}

    def bf16t(self, w, name, a):
        w.add(name, "BF16", a.shape, bf16(a))

    def mix(self, w, p, inject, bits):
        """a gated residual (hyper-connection): per-stream norm, low-rank mix down / up, injection weights"""
        self.bf16t(w, p + ".hc_norm.weight", np.zeros(S * D, np.float32))
        self.affine(w, p + ".input_mix_weight_down", RANK, S * D, bits)
        self.affine(w, p + ".input_mix_weight_up", S * D, RANK, bits)
        if inject:
            self.bf16t(w, p + ".block_inject_weight.weight",
                       self.rng.uniform(0.3, 0.7, size=(S, S * D)).astype(np.float32))
            self.quant[p + ".block_inject_weight"] = False

    def attention(self, w, p, bits):
        rng = self.rng
        self.affine(w, p + "q_proj", 2 * HQ * HD, D, bits)
        self.affine(w, p + "k_proj", HKV * HD, D, bits)
        self.affine(w, p + "v_proj", HKV * HD, D, bits)
        self.affine(w, p + "o_proj", D, HQ * HD, bits, 0.5)
        self.bf16t(w, p + "q_norm.weight", np.zeros(HD, np.float32))
        self.bf16t(w, p + "k_norm.weight", np.zeros(HD, np.float32))
        self.affine(w, p + "indexer.index_qk_proj", (IXH + 1) * IXD, D, 8)
        self.bf16t(w, p + "indexer.q_layernorm.weight", np.zeros(IXD, np.float32))
        self.bf16t(w, p + "indexer.k_layernorm.weight", rng.normal(0, 0.05, IXD).astype(np.float32))

    def moe(self, w, p, routed, shared_bits):
        self.affine(w, p + "gate", E, D, 8)
        self.affine(w, p + "shared_expert_gate", 1, D, 8)
        for m, (n, k) in (("gate", (I_S, D)), ("up", (I_S, D)), ("down", (D, I_S))):
            self.affine(w, p + "shared_expert." + m + "_proj", n, k, shared_bits, 0.5 if m == "down" else 1.0)
        for m, (n, k) in (("gate", (I_E, D)), ("up", (I_E, D)), ("down", (D, I_E))):
            routed(w, p + "switch_mlp." + m + "_proj", n, k)


def primes_after(base, count):
    out, p = [], base - 1
    while len(out) < count:
        p += 1
        if p > 1 and all(p % q for q in range(2, int(p ** 0.5) + 1)):
            out.append(p)
    return out


def main():
    global V, D
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--hidden", type=int, default=D)
    a = ap.parse_args()
    L = a.layers
    if a.hidden < 512 or a.hidden % 128:
        raise SystemExit("--hidden: a multiple of 128, at least 512")
    D = a.hidden
    eos = V - 1
    if a.tokenizer:  # the model's vocabulary and its chat turn's end (<|im_end|>)
        src = json.load(open(os.path.join(a.tokenizer, "config.json")))
        V = src.get("text_config", src)["vocab_size"]
        tok = json.load(open(os.path.join(a.tokenizer, "tokenizer.json")))
        eos = next(t["id"] for t in tok["added_tokens"] if t["content"] == "<|im_end|>")
    if L < 4:
        raise SystemExit("--layers: at least 4 (the 4th is the attention layer)")
    os.makedirs(a.out, exist_ok=True)
    P = Pack(np.random.default_rng(a.seed))
    rng = P.rng
    pre = "language_model.model."
    types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(L)]
    files = []
    w = Writer()
    P.affine(w, pre + "embed_tokens", V, D, 8, 4.0)
    P.affine(w, "language_model.lm_head", V, D, 4)
    P.mix(w, pre + "hyper_connection_mixer", False, 6)
    files.append(w)
    for i in range(L):
        w = Writer()
        p = pre + f"layers.{i}."
        P.mix(w, p + "attn_hyper_connection", True, 6)
        P.mix(w, p + "mlp_hyper_connection", True, 6)
        if types[i] == "linear_attention":
            q = p + "linear_attn."
            C = 2 * HK * DK + HV * DV
            P.affine(w, q + "in_proj_qkv", C, D, 4)
            P.affine(w, q + "in_proj_z", HV * DV, D, 4)
            P.affine(w, q + "in_proj_a", HV, D, 4)
            P.affine(w, q + "in_proj_b", HV, D, 4)
            P.affine(w, q + "out_proj", D, HV * DV, 4, 0.5)
            P.bf16t(w, q + "conv1d.weight", rng.normal(0, 0.3, (C, 4, 1)).astype(np.float32))
            P.bf16t(w, q + "A_log", np.log(rng.uniform(1, 16, HV)).astype(np.float32))
            P.bf16t(w, q + "dt_bias", rng.uniform(0, 1, HV).astype(np.float32))
            P.bf16t(w, q + "norm.weight", np.ones(DV, np.float32))
        else:
            P.attention(w, p + "self_attn.", 6)
        P.moe(w, p + "mlp.", P.experts_q20, 4)
        if i == 1:  # the n-gram embedding (ple_layer_ids [2]: before the second layer)
            q = p + "ple."
            P.affine(w, q + "key_proj", S * D, D, 8)
            P.affine(w, q + "value_proj", D, D, 8)
            for n in ("norm_key", "norm_query", "norm_conv"):
                P.bf16t(w, q + n + ".weight", np.zeros(S * D, np.float32))
            P.bf16t(w, q + "conv1d.weight", rng.normal(0, 0.3, (S * D, 4, 1)).astype(np.float32))
        files.append(w)
    # the n-gram table: 8 heads of primes after NG_BASE rows of D / 8 values, 4-bit group 32
    heads = (NGRAM - 1) * PER_NGRAM
    rows = sum(primes_after(NG_BASE, heads))
    dim = D // heads
    tw = Writer()
    tp = pre + "layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    tw.add(tp + ".weight", "U32", (rows, dim // 8), rng.integers(0, 1 << 32, size=(rows, dim // 8), dtype=np.uint64).astype(np.uint32))
    sc = rng.uniform(0.02, 0.05, size=(rows, dim // 32)).astype(np.float32)
    tw.add(tp + ".scales", "BF16", (rows, dim // 32), bf16(sc))
    tw.add(tp + ".biases", "BF16", (rows, dim // 32), bf16(-7.5 * sc))
    tw.write(os.path.join(a.out, "ple.safetensors"))
    # the MTP head (its experts 4-bit, its other linears 8-bit, as Nace AI's pack)
    w = Writer()
    P.affine(w, "mtp.fc_embedding", D, D, 8)
    P.affine(w, "mtp.fc_hidden", D, D, 8)
    P.bf16t(w, "mtp.pre_fc_norm_embedding.weight", np.zeros(D, np.float32))
    P.bf16t(w, "mtp.pre_fc_norm_hidden.weight", np.zeros(S * D, np.float32))
    P.mix(w, "mtp.hyper_connection_mixer", False, 8)
    P.mix(w, "mtp.layers.0.attn_hyper_connection", True, 8)
    P.mix(w, "mtp.layers.0.mlp_hyper_connection", True, 8)
    P.attention(w, "mtp.layers.0.self_attn.", 8)
    P.moe(w, "mtp.layers.0.mlp.", P.experts4, 8)
    w.write(os.path.join(a.out, "mtp.safetensors"))
    n = len(files)
    for k, f in enumerate(files):
        f.write(os.path.join(a.out, f"model-{k + 1:05d}-of-{n:05d}.safetensors"))
    q = {"group_size": 64, "bits": 4, "mode": "affine"}
    q.update({k: v for k, v in sorted(P.quant.items())})
    text = {
        "model_type": "qwen4_exp_text", "hidden_size": D, "num_hidden_layers": L, "vocab_size": V,
        "layer_types": types, "full_attention_interval": 4,
        "linear_num_key_heads": HK, "linear_num_value_heads": HV, "linear_key_head_dim": DK,
        "linear_value_head_dim": DV, "linear_conv_kernel_dim": 4,
        "num_attention_heads": HQ, "num_key_value_heads": HKV, "head_dim": HD,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "rope_type": "default",
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
        "rms_norm_eps": 1e-6, "num_experts": E, "num_experts_per_tok": K_USED, "moe_intermediate_size": I_E,
        "shared_expert_intermediate_size": I_S, "norm_topk_prob": True,
        "hc_count": S, "hc_lowrank": RANK, "indexer_n_heads": IXH, "indexer_head_dim": IXD, "indexer_budget": 2048,
        "indexer_compress_ratio": 4, "output_gate_type": "sigmoid",
        "ple_layer_ids": [2], "ngram_size": NGRAM, "heads_per_ngram": PER_NGRAM, "seed": 1234,
        "ple_conv_kernel_size": 4, "ngram_vocab_size_base": NG_BASE, "eos_token_id": eos,
        "mtp_num_hidden_layers": 1, "tie_word_embeddings": False,
    }
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "text_config": text,
           "tie_word_embeddings": False, "eos_token_id": eos, "quantization": q, "quantization_config": q}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja") if a.tokenizer else ():
        if os.path.exists(os.path.join(a.tokenizer, f)):
            shutil.copy(os.path.join(a.tokenizer, f), os.path.join(a.out, f))
    size = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"{a.out}: {L} layers, {E} experts of {I_E} (Q2_0), hidden {D}, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
