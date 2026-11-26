#!/usr/bin/env python3
"""Reference of Inkling-2's image encoder: Hugging Face transformers' Inkling2VisionModel and Inkling2ImageProcessor
(transformers 5.24 or later) with the checkpoint's model.visual.* tensors, on the CPU, for comparison with Kapitsa's
GPU encoder (build/vit_check tower; tests/vision/check_ink2.py drives both).

usage: ref_ink2.py MODEL_DIR IMAGE OUT_DIR [--bf16] [--prep]
IMAGE: an image file, opened as the reference opens it (PIL, the EXIF orientation applied, transparency kept for the
processor's checkerboard), any size (the processor pads it to whole 40-pixel tokens). Writes to OUT_DIR, float32:
pv.f32 (the processor's pixel_values [tokens, 40, 40, 3]), h0.f32 (every row after the embeddings' LayerNorm: the class
token, then the patches), h1.f32 and hL.f32 (the same after block 1 and after the last block) and emb.f32 (the merger's
output [tokens, text hidden]); prints "grid_h grid_w tokens". --bf16 runs the tower in bfloat16 as it ships (its error
against float32 is the scale for Kapitsa's); --prep stops after pv.f32. Needs torch, torchvision and transformers. CPU only.
"""
import json, os, struct, sys

import numpy as np
import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model_dir, image_path, out_dir = args[0], args[1], args[2]
dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
os.makedirs(out_dir, exist_ok=True)
torch.set_grad_enabled(False)

from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig
from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel

# the processor on the file as PIL opens it (an RGBA, LA or transparent palette image keeps its alpha)
from PIL import Image, ImageOps

proc = Inkling2ImageProcessor()
feats = proc(images=[ImageOps.exif_transpose(Image.open(image_path))], return_tensors="pt")
pixel_values, grid = feats["pixel_values"], feats["image_grid_thw"]
pixel_values.float().numpy().astype(np.float32).tofile(os.path.join(out_dir, "pv.f32"))
gt, gh, gw = (int(v) for v in grid[0])
result = f"{gh} {gw} {gt * gh * gw}"  # grid_h grid_w tokens
if "--prep" in sys.argv:
    print(result)
    sys.exit(0)

cfg = json.load(open(os.path.join(model_dir, "config.json")))
vc = dict(cfg["vision_config"])
vconf = Inkling2VisionConfig(**{k: v for k, v in vc.items() if k != "model_type"})
vconf._attn_implementation = "eager"
model = Inkling2VisionModel(vconf).to(dtype).eval()

# the checkpoint's vision tensors, renamed as transformers' conversion mapping renames them
RENAMES = [
    ("backbone.embeddings.position_embedding", "embeddings.position_embedding.weight"),
    ("backbone.embeddings.", "embeddings."),
    ("backbone.ln_pre.", "pre_layernorm."),
    ("backbone.layers.", "layers."),
    ("proj_in.", "merger.linear_1."),
    ("proj_out.", "merger.linear_2."),
]
LAYER_RENAMES = [(".ln_1.", ".layernorm_before."), (".ln_2.", ".layernorm_after."), (".mlp.c_fc.", ".mlp.fc1."),
                 (".mlp.c_proj.", ".mlp.fc2."), (".ls_1.gamma", ".lambda_1"), (".ls_2.gamma", ".lambda_2")]


def rename(name):
    for a, b in RENAMES:
        if name.startswith(a):
            name = b + name[len(a):]
            break
    for a, b in LAYER_RENAMES:
        name = name.replace(a, b)
    return name


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


prefix = "model.visual."
index = os.path.join(model_dir, "model.safetensors.index.json")
if os.path.exists(index):
    files = sorted({f for k, f in json.load(open(index))["weight_map"].items() if k.startswith(prefix)})
else:
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
state = {}
for fname in files:
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
            t = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(v["shape"])
            state[rename(k[len(prefix):])] = t.to(dtype)
missing, unexpected = model.load_state_dict(state, strict=False)
missing = [m for m in missing if "inv_freq" not in m]
assert not missing and not unexpected, (missing, unexpected)

# the forward, with the hidden state after the embeddings' LayerNorm, block 1 and the last block
taps = {}
model.pre_layernorm.register_forward_hook(lambda m, i, o: taps.__setitem__("h0", o))
model.layers[0].register_forward_hook(lambda m, i, o: taps.__setitem__("h1", o))
model.layers[-1].register_forward_hook(lambda m, i, o: taps.__setitem__("hL", o))
out = model(pixel_values=pixel_values, image_grid_thw=grid)
for name, x in (("h0", taps["h0"]), ("h1", taps["h1"]), ("hL", taps["hL"]), ("emb", out.pooler_output)):
    x.float().numpy().astype(np.float32).tofile(os.path.join(out_dir, name + ".f32"))
print(result)
