#!/usr/bin/env python3
"""Reference forward of the Qwen3.5 vision tower on the CPU in float32 (Hugging Face transformers' Qwen3_5VisionModel
with the checkpoint's vision weights), for comparison with Kapitsa's GPU encoder (build/vit_check tower).

usage: ref_tower.py MODEL_DIR IN.rgb W H OUT.f32 [LAYERS]
  IN.rgb: raw 8-bit RGB of W x H (multiples of 32: no resize); OUT.f32: the merger's output [tokens, out_hidden], or
  with LAYERS the hidden state after that many blocks [patches, hidden] (the residual stream, before any norm).
Needs torch and transformers (e.g. research/inference/venvs/rapid-mlx). CPU only.
"""
import json, os, struct, sys

import numpy as np
import torch

model_dir, rgb_path, W, H, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
layers = int(sys.argv[6]) if len(sys.argv) > 6 else None
torch.set_grad_enabled(False)

cfg = json.load(open(os.path.join(model_dir, "config.json")))
vc = dict(cfg["vision_config"])
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

vconf = Qwen3_5VisionConfig(**{k: v for k, v in vc.items() if k != "model_type"})
vconf._attn_implementation = "sdpa"
model = Qwen3_5VisionModel(vconf).float().eval()

# the checkpoint's vision tensors
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
prefix = "vision_tower." if any(k.startswith("vision_tower.") for k in idx) else "model.visual."
state = {}
for fname in sorted({f for k, f in idx.items() if k.startswith(prefix)}):
    path = os.path.join(model_dir, fname)
    hdr, base = read_header(path)
    with open(path, "rb") as f:
        for k, v in hdr.items():
            if not k.startswith(prefix):
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            raw = f.read(b - a)
            assert v["dtype"] == "BF16", (k, v["dtype"])
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(v["shape"]).float()
            name = k[len(prefix):]
            if name == "patch_embed.proj.weight" and t.shape[-1] == 3:  # MLX [O, T, P, P, C] -> PyTorch [O, C, T, P, P]
                t = t.permute(0, 4, 1, 2, 3).contiguous()
            state[name] = t
missing, unexpected = model.load_state_dict(state, strict=False)
missing = [m for m in missing if "rotary" not in m]
assert not missing and not unexpected, (missing, unexpected)

# pixel_values as the processor makes them (verified bit-exact against it by check_cpu.py)
P, T, M = vc["patch_size"], vc["temporal_patch_size"], vc["spatial_merge_size"]
a = np.fromfile(rgb_path, np.uint8).reshape(H, W, 3)
x = np.transpose(a, (2, 0, 1)).astype(np.float32) * np.float32(1 / 255)
x = (x - np.float32(0.5)) / np.float32(0.5)
gh, gw = H // P, W // P
x = np.repeat(x[None], T, axis=0).reshape(T, 3, gh // M, M, P, gw // M, M, P).transpose(2, 5, 3, 6, 1, 0, 4, 7)
pixel_values = torch.from_numpy(np.ascontiguousarray(x.reshape(gh * gw, 3 * T * P * P)))
grid = torch.tensor([[1, gh, gw]])

if layers is None:
    out = model(pixel_values, grid_thw=grid).pooler_output
else:  # the forward's steps up to `layers` blocks
    from transformers.vision_utils import (get_vision_attention_seqlens, get_vision_interpolation_indices_and_weights,
                                           get_vision_position_ids)
    ii, iw = get_vision_interpolation_indices_and_weights(grid, num_grid_per_side=model.num_grid_per_side, mode="bilinear",
                                                         align_corners=True, spatial_merge_size=M)
    pos_ids = get_vision_position_ids(grid, M)
    cu, mx = get_vision_attention_seqlens(grid, model.config)
    h = model.patch_embed(pixel_values)
    h = h + (model.pos_embed(ii) * iw[:, :, None]).sum(1)
    rot = model.rotary_pos_emb(pos_ids).reshape(h.shape[0], -1)
    emb = torch.cat((rot, rot), dim=-1)
    pe = (emb.cos(), emb.sin())
    for blk in model.blocks[:layers]:
        h = blk(h, cu_seqlens=cu, max_seqlen=mx, position_embeddings=pe)
    out = h
out.float().numpy().astype(np.float32).tofile(out_path)
print(f"{tuple(out.shape)} written to {out_path}")
