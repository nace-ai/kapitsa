#!/usr/bin/env python3
"""Numerical validation of Kapitsa's Ink2ViT (src/vision/ink2_vit.cpp) against
Hugging Face transformers' reference Inkling2ImageProcessor + Inkling2VisionModel.

Usage:
  /Users/audreyt/w/ink2-ref-venv/bin/python tests/vision/test_ink2_vit.py [MODEL_DIR]
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from transformers.models.inkling2 import (
    Inkling2ImageProcessor,
    Inkling2VisionConfig,
    Inkling2VisionModel,
)
from transformers.vision_utils import get_vision_position_ids


def bf16_bytes_to_f32(path: str, shape: tuple[int, ...]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).reshape(shape)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    return float(np.dot(af, bf) / denom) if denom > 0 else 1.0


def load_hf_vision_model(model_dir: str) -> Inkling2VisionModel:
    cfg_json = json.load(open(os.path.join(model_dir, "config.json")))
    vcfg_dict = dict(cfg_json["vision_config"])
    vcfg_dict.pop("model_type", None)
    vcfg = Inkling2VisionConfig(**vcfg_dict)
    vcfg._attn_implementation = "sdpa"
    model = Inkling2VisionModel(vcfg)

    sd = {}
    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(model_dir, fname), framework="pt", device="cpu") as f:
            for k in f.keys():
                if not k.startswith("model.visual."):
                    continue
                t = f.get_tensor(k).float()
                nk = k[len("model.visual."):]
                if nk.startswith("backbone."):
                    nk = nk[len("backbone."):]
                if nk == "embeddings.position_embedding":
                    nk = "embeddings.position_embedding.weight"
                nk = nk.replace(".ln_1.", ".layernorm_before.").replace(".ln_2.", ".layernorm_after.")
                nk = nk.replace(".ls_1.gamma", ".lambda_1").replace(".ls_2.gamma", ".lambda_2")
                nk = nk.replace(".mlp.c_fc.", ".mlp.fc1.").replace(".mlp.c_proj.", ".mlp.fc2.")
                nk = nk.replace("ln_pre.", "pre_layernorm.")
                if nk == "proj_in.weight":
                    nk = "merger.linear_1.weight"
                elif nk == "proj_out.weight":
                    nk = "merger.linear_2.weight"
                sd[nk] = t

    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return model.float().eval()


@torch.no_grad()
def hf_forward_at_layer(model: Inkling2VisionModel, pv: torch.Tensor, grid: torch.Tensor, stop_layers: int) -> np.ndarray:
    merge = model.config.spatial_merge_size
    patch_grid = grid.cpu() * torch.tensor([1, merge, merge])
    ppi = patch_grid.prod(-1)
    cu = torch.nn.functional.pad((ppi + 1).cumsum(0), (1, 0)).to(torch.int32)
    image_ids = torch.arange(1, patch_grid.shape[0] + 1)
    patch_index = torch.arange(pv.shape[0] * merge**2) + torch.repeat_interleave(image_ids, ppi)
    h = model.embeddings(pv.float(), patch_grid, patch_index)
    h = model.pre_layernorm(h)
    pos_ids = get_vision_position_ids(patch_grid, merge)
    pos_ids = pos_ids.new_zeros(h.shape[0], 2).index_copy(0, patch_index, pos_ids + 1)
    pe = model.rotary_emb(h, pos_ids)
    for layer in model.layers[:stop_layers]:
        h = layer(h, cu_seqlens=cu, position_embeddings=pe)
    return h.numpy()


def main() -> int:
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/audreyt/models/Inkling-2-Mini"
    vit_check = os.environ.get("VIT_CHECK", os.path.join(repo, "build", "vit_check"))
    if not os.path.exists(vit_check):
        raise FileNotFoundError(f"vit_check binary not found at {vit_check}")

    fails = 0
    tmp = tempfile.mkdtemp(prefix="ink2_vit_test_")

    def report(label: str, ok: bool, detail: str):
        nonlocal fails
        print(f"{'PASS' if ok else 'FAIL'} {label}: {detail}", flush=True)
        if not ok:
            fails += 1

    # 1. Metal kernel self-checks under shader validation
    env = dict(os.environ, MTL_SHADER_VALIDATION="1", MTL_DEBUG_LAYER="1")
    r = subprocess.run([vit_check, "kernels"], env=env, capture_output=True, text=True)
    report("vit_check kernels (MTL_SHADER_VALIDATION=1)", r.returncode == 0,
           (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else f"rc={r.returncode}")

    # 2. Image processor pixel_values exactness across diverse aspect ratios & unaligned dimensions
    proc = Inkling2ImageProcessor()
    prep_sizes = [(320, 320), (137, 93), (300, 200), (480, 160), (160, 480), (25, 35), (1000, 760)]
    rng = np.random.default_rng(42)
    for w, h in prep_sizes:
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        png_path = os.path.join(tmp, f"img_{w}x{h}.png")
        pv_path = os.path.join(tmp, f"pv_{w}x{h}.f32")
        Image.fromarray(arr, mode="RGB").save(png_path)
        res = subprocess.run([vit_check, "prep", model_dir, png_path, pv_path], capture_output=True, text=True, check=True)
        gh, gw, tokens, ow, oh = (int(v) for v in res.stdout.strip().split())
        ref = proc(images=Image.open(png_path), return_tensors="pt")
        want_pv = ref["pixel_values"].numpy()
        want_grid = ref["image_grid_thw"][0].tolist()
        got_pv = np.fromfile(pv_path, dtype=np.float32).reshape(tokens, 40, 40, 3)
        diff = float(np.abs(got_pv - want_pv).max())
        ok = (ow == w and oh == h and [1, gh, gw] == want_grid and got_pv.shape == want_pv.shape and diff < 1e-5)
        report(f"prep {w}x{h}", ok, f"grid={[1, gh, gw]} tokens={tokens} max_abs={diff:.2e}")

    # 3. Tower layer-by-layer and end-to-end validation against float32 HF Inkling2VisionModel
    model = load_hf_vision_model(model_dir)
    tower_cases = [(137, 93), (320, 320), (480, 160), (160, 480)]
    for idx, (w, h) in enumerate(tower_cases):
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        rgb_path = os.path.join(tmp, f"raw_{w}x{h}.rgb")
        arr.tofile(rgb_path)
        ref = proc(images=Image.fromarray(arr, mode="RGB"), return_tensors="pt")
        pv = ref["pixel_values"]
        grid = ref["image_grid_thw"]
        tokens = int(grid[0, 1] * grid[0, 2])
        np_rows = 1 + tokens * 4

        if idx == 0:
            for layer_stop in (0, 1, 11, 23):
                out_path = os.path.join(tmp, f"hs_{w}x{h}_l{layer_stop}.f32")
                subprocess.run([vit_check, "tower", model_dir, rgb_path, str(w), str(h), out_path, str(layer_stop)],
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                got = np.fromfile(out_path, dtype=np.float32).reshape(np_rows, 1024)
                want = hf_forward_at_layer(model, pv, grid, layer_stop)
                cs = cos_sim(got, want)
                mad = float(np.abs(got - want).max())
                report(f"tower {w}x{h} layer={layer_stop}", cs > 0.9999 and got.shape == want.shape,
                       f"cos_sim={cs:.6f} max_abs={mad:.3e}")

        out_path = os.path.join(tmp, f"emb_{w}x{h}.bf16")
        subprocess.run([vit_check, "tower", model_dir, rgb_path, str(w), str(h), out_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        got_emb = bf16_bytes_to_f32(out_path, (tokens, 2304))
        with torch.no_grad():
            want_emb = model(pixel_values=pv.float(), image_grid_thw=grid).pooler_output.numpy()
        cs = cos_sim(got_emb, want_emb)
        mad = float(np.abs(got_emb - want_emb).max())
        mean_diff = float(np.abs(got_emb - want_emb).mean())
        report(f"tower {w}x{h} final", cs > 0.9999 and got_emb.shape == want_emb.shape,
               f"shape={got_emb.shape} cos_sim={cs:.6f} max_abs={mad:.3e} mean_abs={mean_diff:.3e}")

    # 4. High-res full output test (1000x760 -> 25x19 = 475 image tokens, 1901 sequence rows)
    w, h = 1000, 760
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    rgb_path = os.path.join(tmp, f"raw_{w}x{h}.rgb")
    out_path = os.path.join(tmp, f"emb_{w}x{h}.bf16")
    arr.tofile(rgb_path)
    subprocess.run([vit_check, "tower", model_dir, rgb_path, str(w), str(h), out_path],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ref = proc(images=Image.fromarray(arr, mode="RGB"), return_tensors="pt")
    pv, grid = ref["pixel_values"], ref["image_grid_thw"]
    tokens = int(grid[0, 1] * grid[0, 2])
    got_emb = bf16_bytes_to_f32(out_path, (tokens, 2304))
    with torch.no_grad():
        want_emb = model(pixel_values=pv.float(), image_grid_thw=grid).pooler_output.numpy()
    cs = cos_sim(got_emb, want_emb)
    mad = float(np.abs(got_emb - want_emb).max())
    report(f"tower {w}x{h} final", cs > 0.9999, f"shape={got_emb.shape} cos_sim={cs:.6f} max_abs={mad:.3e}")

    # 5. Performance benchmark summary
    for bw, bh in ((512, 512), (1280, 720)):
        rb = subprocess.run([vit_check, "bench", model_dir, str(bw), str(bh), "3"],
                            capture_output=True, text=True, check=True)
        print(f"BENCH {bw}x{bh}: {rb.stdout.strip()}")

    print(f"\nSUMMARY: {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
