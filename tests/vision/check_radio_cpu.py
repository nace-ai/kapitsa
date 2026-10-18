#!/usr/bin/env python3
"""Nemotron 3.5 Bolt's image processor in Kapitsa (src/vision/radio_vit.cpp) against NVIDIA's own, the model bundle's
image_processing_nemotron_h_omni.py (it needs torch): the patch grid, the antialiased bicubic resize, and the
normalized pixel_values of real images. CPU only.

usage: check_radio_cpu.py MODEL_DIR VIT_CHECK IMAGE.png ...
"""
import importlib.util, json, os, random, subprocess, sys, tempfile

import numpy as np
import torch
from PIL import Image

model_dir, vit_check, images = sys.argv[1], sys.argv[2], sys.argv[3:]
spec = importlib.util.spec_from_file_location("nvidia_ip", os.path.join(model_dir, "image_processing_nemotron_h_omni.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = json.load(open(os.path.join(model_dir, "preprocessor_config.json")))
keys = ("norm_mean", "norm_std", "patch_size", "downsample_ratio", "min_num_patches", "max_num_patches", "max_model_len")
proc = mod.NemotronH_Omni_Reasoning_V3ImageProcessor(**{k: v for k, v in cfg.items() if k in keys})
tmp = tempfile.mkdtemp()
fails = 0


def report(name, ok, detail):
    global fails
    print(("ok   " if ok else "FAIL ") + name + ": " + detail, flush=True)
    fails += not ok


# the patch grid: random sizes and budgets
class Size:
    def __init__(self, w, h):
        self.width, self.height = w, h


rng = random.Random(3)
cases = []
for _ in range(4000):
    w = rng.choice([rng.randint(1, 64), rng.randint(64, 4000), rng.randint(4000, 12000)])
    h = rng.choice([rng.randint(1, 64), rng.randint(64, 4000), rng.randint(4000, 12000)])
    budget = rng.choice([13312, 4687, 2048, 1024, rng.randint(1024, 13312)])
    cases.append((w, h, budget))
cases += [(w, h, 13312) for w in range(496, 530) for h in (16, 480, 496, 800, 1800)]
path = os.path.join(tmp, "sizes.txt")
with open(path, "w") as f:
    for w, h, b in cases:
        f.write(f"{w} {h} {b} {proc.min_num_patches}\n")
ours = subprocess.run([vit_check, "patches", path], capture_output=True, text=True, check=True).stdout.split("\n")
bad = []
for (w, h, b), line in zip(cases, ours):
    want = proc._compute_target_patches(Size(w, h), b)
    got = tuple(int(v) for v in line.split())
    if got != tuple(want):
        bad.append((w, h, b, want, got))
report("patch grid", not bad, f"{len(cases) - len(bad)} of {len(cases)} sizes equal" + (f"; first: {bad[0]}" if bad else ""))

# the antialiased bicubic resize (float32, as torch.nn.functional.interpolate)
worst = 0.0
exact = 0
pairs = [((37, 29), (16, 48)), ((640, 480), (512, 384)), ((1000, 700), (1280, 896)), ((3000, 20), (160, 16)),
         ((2880, 1800), (1376, 864)), ((1441, 901), (1440, 896)), ((300, 2000), (304, 2000))]
for (w, h), (ow, oh) in pairs:
    a = np.random.default_rng(w * h).integers(0, 256, (h, w, 3), dtype=np.uint8)
    a.tofile(os.path.join(tmp, "in.rgb"))
    subprocess.run([vit_check, "aaresize", os.path.join(tmp, "in.rgb"), str(w), str(h), str(ow), str(oh),
                    os.path.join(tmp, "out.f32")], check=True)
    got = np.fromfile(os.path.join(tmp, "out.f32"), np.float32).reshape(3, oh, ow)
    t = torch.from_numpy(a.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    want = torch.nn.functional.interpolate(t, size=(oh, ow), mode="bicubic", align_corners=False, antialias=True)[0].numpy()
    d = np.abs(got - want).max()
    worst = max(worst, d)
    exact += bool((got == want).all())
report("antialiased bicubic resize", worst < 1e-3, f"{exact} of {len(pairs)} sizes bit-exact, max |diff| {worst:.3g} (of 255)")

# pixel_values of real images (PNG: decoded alike)
for img in images:
    out = os.path.join(tmp, "pv.f32")
    r = subprocess.run([vit_check, "prep", model_dir, img, out], capture_output=True, text=True, check=True)
    gh, gw, tokens, W, H = (int(v) for v in r.stdout.split())
    got = np.fromfile(out, np.float32).reshape(3, H, W)
    ref = proc(images=[Image.open(img).convert("RGB")], return_tensors="pt")
    want = ref["pixel_values"][0].numpy()
    same_shape = want.shape == got.shape and int(ref["num_tokens"][0]) == tokens
    d = np.abs(got - want).max() if same_shape else float("inf")
    report(f"pixel_values {os.path.basename(img)}", same_shape and d < 1e-4,
           f"{W}x{H}, {tokens} tokens (reference {tuple(want.shape[1:][::-1])}, {int(ref['num_tokens'][0])}); "
           f"max |diff| {d:.3g}, bit-exact {bool(same_shape and (got == want).all())}")
print(f"\n{fails} failed")
sys.exit(1 if fails else 0)
