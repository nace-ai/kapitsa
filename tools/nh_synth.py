#!/usr/bin/env python3
"""A small random Nemotron-H checkpoint for exactness tests that need no download: Nemotron 3.5 Lightning's layer
shapes and formats (FP8 Mamba projections, BF16 attention, F32 router, NVFP4 experts / shared expert / LM head, BF16
embedding) cut to six layers (Mamba, MoE, Mamba, attention, Mamba, MoE), 8 experts and an 8192-token vocabulary, and,
with --dspark, a random DSpark drafter in <out>/dflash (two layers over all six target layers, Lightning's formats).
Random weights: the outputs mean nothing, but every kernel of a Lightning verify runs on them (tools/nh_tree_check).
  python3 tools/nh_synth.py <out_dir> [--dspark] [--small] [--seed N]
About 340 MB (the drafter: 150 MB more); --small: four layers (Mamba, MoE, attention, Mamba), 4 experts, a 4096-token
vocabulary and a one-layer drafter, about 280 MB in all."""
import argparse
import json
import os
import struct

import numpy as np

H, V, MH, MD, NG, DS = 2688, 8192, 64, 64, 8, 128
HQ, HKV, HD = 32, 2, 128
E, K_USED, I_E, I_S = 8, 4, 1856, 3712
LAYERS = ["mamba", "moe", "mamba", "attention", "mamba", "moe"]
DI = MH * MD
CH = DI + 2 * NG * DS
IN_PROJ = 2 * DI + 2 * NG * DS + MH


def bf16(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def e4m3_bytes(rng, shape, lo_exp=5, hi_exp=8):
    """random E4M3 bytes of magnitude 2^(lo_exp - 7) .. 2^(hi_exp - 6) (no NaN, no subnormals)"""
    e = rng.integers(lo_exp, hi_exp + 1, size=shape, dtype=np.uint8)
    m = rng.integers(0, 8, size=shape, dtype=np.uint8)
    s = rng.integers(0, 2, size=shape, dtype=np.uint8)
    return (s << 7) | (e << 3) | m


class Writer:
    def __init__(self):
        self.t = {}

    def add(self, name, dtype, shape, data):
        self.t[name] = (dtype, list(shape), np.ascontiguousarray(data).tobytes())

    def bf16(self, name, a):
        self.add(name, "BF16", a.shape, bf16(a))

    def f32(self, name, a):
        a = np.asarray(a, dtype=np.float32)
        self.add(name, "F32", a.shape, a)

    def fp8(self, rng, name, n, k, std):
        # E4M3 values of mean magnitude ~1.4 times one per-tensor scale
        self.add(name, "F8_E4M3", (n, k), e4m3_bytes(rng, (n, k)))
        self.f32(name + "_scale", np.array(std / 1.4, dtype=np.float32).reshape(()))

    def fp4(self, rng, name, n, k, std):
        # ModelOpt NVFP4: E2M1 codes (two per byte, low nibble first), an E4M3 scale per 16, one F32 scale
        codes = rng.integers(0, 16, size=(n, k // 2), dtype=np.uint8)
        codes = np.where((codes & 7) == 7, codes - 1, codes).astype(np.uint8)  # (no +-6, the largest code)
        codes = np.where(((codes >> 4) & 7) == 7, codes - 16, codes).astype(np.uint8)
        self.add(name, "U8", (n, k // 2), codes)
        scales = e4m3_bytes(np.random.default_rng(rng.integers(1 << 30)), (n, k // 16), 6, 7) & 0x7F
        self.add(name.replace(".weight", ".weight_scale"), "F8_E4M3", (n, k // 16), scales)
        self.f32(name.replace(".weight", ".weight_scale_2"), np.array(std / 2.0, dtype=np.float32).reshape(()))

    def write(self, path):
        header, off, blobs = {}, 0, []
        for name in sorted(self.t):
            dtype, shape, b = self.t[name]
            header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(b)]}
            blobs.append(b)
            off += len(b)
        header["__metadata__"] = {"format": "pt"}
        hj = json.dumps(header).encode()
        hj += b" " * ((8 - len(hj) % 8) % 8)
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hj)))
            f.write(hj)
            for b in blobs:
                f.write(b)


def target(out, rng):
    w = Writer()
    w.bf16("backbone.embeddings.weight", rng.standard_normal((V, H)).astype(np.float32))
    w.fp4(rng, "lm_head.weight", V, H, 2.0 / np.sqrt(H))
    w.bf16("backbone.norm_f.weight", 1.0 + 0.1 * rng.standard_normal(H))
    for i, kind in enumerate(LAYERS):
        b = f"backbone.layers.{i}."
        mx = b + "mixer."
        w.bf16(b + "norm.weight", 1.0 + 0.1 * rng.standard_normal(H))
        if kind == "mamba":
            w.fp8(rng, mx + "in_proj.weight", IN_PROJ, H, 1.0 / np.sqrt(H))
            w.fp8(rng, mx + "out_proj.weight", H, DI, 0.5 / np.sqrt(DI))
            w.bf16(mx + "conv1d.weight", 0.5 * rng.standard_normal((CH, 1, 4)))
            w.bf16(mx + "conv1d.bias", 0.1 * rng.standard_normal(CH))
            w.bf16(mx + "A_log", np.log(rng.uniform(1, 16, MH)))
            dt = rng.uniform(1e-3, 0.1, MH)
            w.bf16(mx + "dt_bias", dt + np.log(-np.expm1(-dt)))  # softplus^-1
            w.bf16(mx + "D", np.ones(MH))
            w.bf16(mx + "norm.weight", 1.0 + 0.1 * rng.standard_normal(DI))
        elif kind == "attention":
            for p, n, k in (("q", HQ * HD, H), ("k", HKV * HD, H), ("v", HKV * HD, H), ("o", H, HQ * HD)):
                w.bf16(mx + p + "_proj.weight", rng.standard_normal((n, k)) / np.sqrt(k))
        else:
            w.f32(mx + "gate.weight", rng.standard_normal((E, H)) / np.sqrt(H))
            w.f32(mx + "gate.e_score_correction_bias", np.zeros(E))
            for e in range(E):
                w.fp4(rng, mx + f"experts.{e}.up_proj.weight", I_E, H, 1.0 / np.sqrt(H))
                w.fp4(rng, mx + f"experts.{e}.down_proj.weight", H, I_E, 0.5 / np.sqrt(I_E))
            w.fp4(rng, mx + "shared_experts.up_proj.weight", I_S, H, 1.0 / np.sqrt(H))
            w.fp4(rng, mx + "shared_experts.down_proj.weight", H, I_S, 0.5 / np.sqrt(I_S))
    w.write(os.path.join(out, "model.safetensors"))
    cfg = {
        "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "hidden_size": H, "vocab_size": V,
        "mamba_num_heads": MH, "mamba_head_dim": MD, "ssm_state_size": DS, "n_groups": NG, "conv_kernel": 4,
        "num_attention_heads": HQ, "num_key_value_heads": HKV, "head_dim": HD, "n_routed_experts": E,
        "num_experts_per_tok": K_USED, "moe_intermediate_size": I_E, "moe_shared_expert_intermediate_size": I_S,
        "n_shared_experts": 1, "moe_latent_size": None, "routed_scaling_factor": 2.5, "norm_topk_prob": True,
        "layer_norm_epsilon": 1e-5, "mamba_hidden_act": "silu", "mlp_hidden_act": "relu2", "use_conv_bias": True,
        "layers_block_type": LAYERS, "bos_token_id": 1, "eos_token_id": 2, "tie_word_embeddings": False,
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    json.dump({"eos_token_id": [2]}, open(os.path.join(out, "generation_config.json"), "w"))


def dspark(out, rng, layers=2, inter=6144, rank=512):
    w = Writer()
    w.bf16("embed_tokens.weight", np.zeros((V, H)))  # (only the mask token's row is read)
    taps = list(range(len(LAYERS)))
    w.fp4(rng, "fc.weight", H, H * len(taps), 1.0 / np.sqrt(H * len(taps)))
    w.bf16("hidden_norm.weight", np.ones(H))
    w.bf16("norm.weight", np.ones(H))
    for l in range(layers):
        b = f"layers.{l}."
        w.bf16(b + "input_layernorm.weight", np.ones(H))
        w.bf16(b + "post_attention_layernorm.weight", np.ones(H))
        for p, n, k in (("q", HQ * HD, H), ("k", HKV * HD, H), ("v", HKV * HD, H), ("o", H, HQ * HD)):
            w.bf16(b + f"self_attn.{p}_proj.weight", rng.standard_normal((n, k)) / np.sqrt(k))
        w.bf16(b + "self_attn.q_norm.weight", np.ones(HD))
        w.bf16(b + "self_attn.k_norm.weight", np.ones(HD))
        w.bf16(b + "self_attn.attention_sink_bias", np.zeros(HQ))
        w.fp4(rng, b + "mlp.gate_proj.weight", inter, H, 1.0 / np.sqrt(H))
        w.fp4(rng, b + "mlp.up_proj.weight", inter, H, 1.0 / np.sqrt(H))
        w.fp4(rng, b + "mlp.down_proj.weight", H, inter, 0.5 / np.sqrt(inter))
    w.bf16("markov_head.markov_w1.weight", rng.standard_normal((V, rank)) / np.sqrt(rank))
    w.fp4(rng, "markov_head.markov_w2.weight", V, rank, 1.0)
    w.write(os.path.join(out, "model.safetensors"))
    cfg = {
        "architectures": ["Qwen3DSparkModel"], "model_type": "qwen3", "hidden_size": H, "intermediate_size": inter,
        "num_hidden_layers": layers, "num_attention_heads": HQ, "num_key_value_heads": HKV, "head_dim": HD,
        "rms_norm_eps": 1e-6, "rope_theta": 10000, "sliding_window": 1024, "vocab_size": V, "block_size": 8,
        "mask_token_id": 990, "target_layer_ids": taps, "dspark_fixup_head_type": "markov", "dspark_markov_rank": rank,
        "markov_rank": rank, "attention_sink_bias": True, "has_lm_head": False, "dspark_bonus_anchor": True,
        "sample_from_anchor": False,
        "dflash_config": {"causal": True, "mask_token_id": 990, "swa_window_size": 1024, "target_layer_ids": taps,
                          "use_swa": True, "attention_sink_bias": True, "sample_from_anchor": False},
    }
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)


def main():
    global V, E, K_USED, LAYERS
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dspark", action="store_true")
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.small:
        V, E, K_USED, LAYERS = 4096, 4, 2, ["mamba", "moe", "attention", "mamba"]
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    target(a.out, rng)
    if a.dspark:
        os.makedirs(os.path.join(a.out, "dflash"), exist_ok=True)
        dspark(os.path.join(a.out, "dflash"), rng, *((1, 2048) if a.small else ()))


if __name__ == "__main__":
    main()
