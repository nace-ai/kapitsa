#!/usr/bin/env python3
"""Reference of Nemotron 3.5 Bolt's and Super-VL's image encoder: the model bundle's own code (NVIDIA's modeling_radio.py
RadioModel and the vision projector of modeling_nemotron_h_omni.py, with Super-VL's final LayerNorm, and its image
processor) on the CPU, only the vision tensors loaded (~3.3 GB in float32). Run it while no model runs.

usage: ref_radio.py MODEL_DIR IN.rgb W H OUT_DIR [--bf16]
Writes to OUT_DIR, float32: h1.f32 / h32.f32 (every row's hidden state after block 1 / the last block: cls and register
tokens, then the patches) and emb.f32 (the projector's output, the language model's image rows). --bf16 runs the model
in bfloat16 as it ships (its error against float32 is the scale for Kapitsa's).
"""
import importlib, importlib.util, json, os, sys, types

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model_dir, rgb_path, W, H, out_dir = args[0], args[1], int(args[2]), int(args[3]), args[4]
dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
os.makedirs(out_dir, exist_ok=True)
torch.set_grad_enabled(False)

# the bundle as a package (its files import each other relatively)
spec = importlib.util.spec_from_file_location("bolt", os.path.join(model_dir, "__init__.py"),
                                              submodule_search_locations=[model_dir])
pkg = importlib.util.module_from_spec(spec)
sys.modules["bolt"] = pkg
spec.loader.exec_module(pkg)
radio_cfg = importlib.import_module("bolt.configuration_radio")
radio = importlib.import_module("bolt.modeling_radio")
# the Omni module imports the language model's classes, which need mamba-ssm (CUDA); the projector does not use them
for name, cls in (("bolt.modeling_nemotron_h", "NemotronHForCausalLM"), ("bolt.modeling_nemotron_h_puzzle",
                                                                         "NemotronHPuzzleForCausalLM")):
    stub = types.ModuleType(name)
    setattr(stub, cls, type(cls, (), {}))
    sys.modules[name] = stub
omni = importlib.import_module("bolt.modeling_nemotron_h_omni")
ipn = importlib.import_module("bolt.image_processing_nemotron_h_omni")

cfg = json.load(open(os.path.join(model_dir, "config.json")))
vc = cfg["vision_config"]
fields = set(radio_cfg.RadioConfig.__dataclass_fields__) if hasattr(radio_cfg.RadioConfig, "__dataclass_fields__") else None
vkw = {k: v for k, v in vc.items() if k not in ("args", "model_type", "dtype") and (fields is None or k in fields)}
rcfg = radio_cfg.RadioConfig(**vkw)
model = radio.RadioModel(rcfg)
model.make_preprocessor_external()

# the vision tensors, renamed as the bundle's conversion mapping renames them
idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
pre = "vision_model.radio_model.model."
sd = {}
fln = "vision_projector.vision_final_layernorm."  # Super-VL's LayerNorm on the tower's output
for f in sorted({v for k, v in idx.items() if k.startswith(pre) or k.startswith("mlp1.") or k.startswith(fln)}):
    with safe_open(os.path.join(model_dir, f), "pt") as st:
        for k in st.keys():
            if not (k.startswith(pre) or k.startswith("mlp1.") or k.startswith(fln)):
                continue
            t = st.get_tensor(k).to(torch.float32)
            if k.startswith("mlp1.") or k.startswith(fln):
                sd[k] = t
                continue
            n = k[len(pre):]
            n = n.replace("patch_generator.video_embedder", "embeddings.video_patch_projection")
            n = n.replace("patch_generator.embedder", "embeddings.patch_projection")
            n = n.replace("patch_generator.pos_embed", "embeddings.position_embedding")
            n = n.replace("patch_generator.cls_token.token", "embeddings.cls_register_token")
            n = n.replace("blocks.", "encoder.layer.").replace("attn.proj", "attention.output.dense")
            if "attn.qkv" in n:
                for part, chunk in zip(("query", "key", "value"), t.chunk(3, dim=0)):
                    sd[n.replace("attn.qkv", "attention.attention." + part)] = chunk.clone()
                continue
            sd[n] = t
tower_sd = {k: v for k, v in sd.items() if not (k.startswith("mlp1.") or k.startswith(fln))}
missing, unexpected = model.load_state_dict(tower_sd, strict=False)
missing = [m for m in missing if "layer_scale" not in m and m != "summary_idxs"]  # LayerScale 1.0; idxs from config
if missing or unexpected:
    sys.exit(f"tower weights: missing {missing[:5]}, unexpected {unexpected[:5]}")
model = model.to(dtype).eval()

pcfg = types.SimpleNamespace(downsample_ratio=cfg["downsample_ratio"], video_temporal_patch_size=vc.get("video_temporal_patch_size"),
                             ps_version=cfg.get("ps_version", "v2"), vit_hidden_size=cfg["vit_hidden_size"],
                             projector_hidden_size=cfg["projector_hidden_size"],
                             llm_config=types.SimpleNamespace(
                                 hidden_size=cfg["llm_config"]["hidden_size"],
                                 num_nextn_predict_layers=cfg["llm_config"].get("num_nextn_predict_layers", 0)),
                             vision_final_layernorm=cfg.get("vision_final_layernorm"),  # None: from the MTP layers
                             vision_config=types.SimpleNamespace(layer_norm_eps=vc["layer_norm_eps"]))
proj = omni.NemotronH_Omni_Reasoning_V3VisionProjector(pcfg)
proj.mlp1.norm.weight.copy_(sd["mlp1.0.weight"])
proj.mlp1.linear1.weight.copy_(sd["mlp1.1.weight"])
proj.mlp1.linear2.weight.copy_(sd["mlp1.3.weight"])
if proj.vision_final_layernorm is not None:
    proj.vision_final_layernorm.weight.copy_(sd[fln + "weight"])
    proj.vision_final_layernorm.bias.copy_(sd[fln + "bias"])
proj = proj.to(dtype).eval()

# the image through NVIDIA's processor (a size it keeps: no resize), then the tower and the projector
pp = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
keys = ("norm_mean", "norm_std", "patch_size", "downsample_ratio", "min_num_patches", "max_num_patches", "max_model_len")
proc = ipn.NemotronH_Omni_Reasoning_V3ImageProcessor(**{k: v for k, v in pp.items() if k in keys})
img = Image.fromarray(np.fromfile(rgb_path, np.uint8).reshape(H, W, 3))
pv = proc(images=[img], return_tensors="pt")["pixel_values"]
if tuple(pv.shape[-2:]) != (H, W):
    sys.exit(f"the processor resizes {W}x{H} to {pv.shape[-1]}x{pv.shape[-2]}")
out = model(pv.to(dtype), output_hidden_states=True)
hs = out.hidden_states
hs[1][0].to(torch.float32).numpy().tofile(os.path.join(out_dir, "h1.f32"))
hs[-1][0].to(torch.float32).numpy().tofile(os.path.join(out_dir, "h32.f32"))
emb = proj._project(out.features, H // 16, W // 16)
emb[0].to(torch.float32).numpy().tofile(os.path.join(out_dir, "emb.f32"))
feat = out.features if proj.vision_final_layernorm is None else proj.vision_final_layernorm(out.features)
lin1 = proj.mlp1.linear1(proj.mlp1.norm(
    proj.pixel_shuffle(feat.reshape(1, H // 16, W // 16, -1), scale_factor=0.5).reshape(1, -1, 4 * vc["hidden_size"])))
print(f"{len(hs) - 1} blocks, {hs[-1].shape[1]} rows; final LayerNorm {proj.vision_final_layernorm is not None}; "
      f"embeddings {tuple(emb.shape)}, dtype {dtype}; projector's inner "
      f"max |x| {float(lin1.abs().max()):.1f} (ReLU^2 then {float(lin1.clamp(min=0).square().max()):.1f})")
