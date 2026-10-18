#!/usr/bin/env python3
"""mlx_vlm's Qwen3.5 vision tower alone (only the vision tensors loaded, in the checkpoint's bf16, as mlx_vlm runs
it), for comparing its precision with Kapitsa's and the float32 reference (ref_tower.py). GPU: run it under the lock.

usage: mlx_tower.py MODEL_DIR IN.rgb W H OUT.f32     (W, H multiples of 32: no resize)
"""
import json, os, sys

import numpy as np
import mlx.core as mx
from mlx_vlm.models.qwen3_5.config import ModelConfig
from mlx_vlm.models.qwen3_5.vision import VisionModel

model_dir, rgb, W, H, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
cfg = json.load(open(os.path.join(model_dir, "config.json")))
vc = ModelConfig.from_dict(cfg).vision_config
model = VisionModel(vc)
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
weights = {}
for f in sorted({v for k, v in idx.items() if k.startswith("vision_tower.")}):
    for k, v in mx.load(os.path.join(model_dir, f)).items():
        if k.startswith("vision_tower."):
            weights[k[len("vision_tower."):]] = v
model.load_weights(list(model.sanitize(weights).items()))
P, T, M = vc.patch_size, vc.temporal_patch_size, vc.spatial_merge_size
a = np.fromfile(rgb, np.uint8).reshape(H, W, 3)
x = np.transpose(a, (2, 0, 1)).astype(np.float32) * np.float32(1 / 255)
x = (x - np.float32(0.5)) / np.float32(0.5)
gh, gw = H // P, W // P
x = np.repeat(x[None], T, axis=0).reshape(T, 3, gh // M, M, P, gw // M, M, P).transpose(2, 5, 3, 6, 1, 0, 4, 7)
pv = mx.array(np.ascontiguousarray(x.reshape(gh * gw, 3 * T * P * P))).astype(model.patch_embed.proj.weight.dtype)
feats, _ = model(pv, mx.array([[1, gh, gw]]))
np.array(feats.astype(mx.float32)).tofile(out)
print(feats.shape, feats.dtype)
