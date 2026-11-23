#!/usr/bin/env python3
"""Inkling-2's image input against Hugging Face's reference (Inkling2ImageProcessor and Inkling2VisionModel,
tests/vision/ref_ink2.py).

usage: check_ink2.py MODEL_DIR VIT_CHECK IMAGE.png... [--gpu] [--python PY]
  1. the processor (CPU): `vit_check prep` against Inkling2ImageProcessor on every image (decoded alike): the token
     grid and pixel_values, bit for bit
  2. --gpu (loads the tower: run it under the GPU lock, `lockf -k /tmp/kapitsa-gpu.lock`): `vit_check tower` against
     ref_ink2.py in float32 after the embeddings' LayerNorm, block 1 and the last block (every row: the class token and
     the patches) and at the merger's output (the language model's image rows); the tower in bfloat16 as it ships
     gives the scale. Passes when Kapitsa's relative error is within twice bfloat16's at every stage and no output
     row's cosine to the reference is under 0.999.
Needs torch, torchvision, transformers 5.24 or later, Pillow and numpy (--python PY: run ref_ink2.py with PY, which
has them, while this script needs only Pillow and numpy).
"""
import atexit, json, os, shutil, subprocess, sys, tempfile

import numpy as np
from PIL import Image

args = [a for a in sys.argv[1:] if not a.startswith("--")]
model_dir, vit_check, images = args[0], args[1], args[2:]
GPU = "--gpu" in sys.argv
py = sys.executable
for i, a in enumerate(sys.argv):
    if a == "--python" and i + 1 < len(sys.argv):
        py = sys.argv[i + 1]
if py in images:
    images.remove(py)
if not images:
    sys.exit(__doc__)
ref_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_ink2.py")
tmp = tempfile.mkdtemp(prefix="ink2_vision_")
atexit.register(shutil.rmtree, tmp, True)
fails = 0


def report(name, ok, detail):
    global fails
    print(("ok   " if ok else "FAIL ") + name + ": " + detail, flush=True)
    fails += not ok


def run(*cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: {r.stderr.strip()}")
    return r.stdout.strip()


def decoded(img):
    """the decoded RGB as the reference sees it (vit_check decode: PIL's convert("RGB") values), and its size"""
    rgb = os.path.join(tmp, "in.rgb")
    w, h, _ = run(vit_check, "decode", img, rgb).split()
    return rgb, int(w), int(h)


def stats(got, ref):
    """relative L2 error over every value, the largest |diff|, the smallest row cosine"""
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
    worst = float(np.abs(got - ref).max())
    cos = (got * ref).sum(1) / np.maximum(np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1), 1e-30)
    return rel, worst, float(cos.min())


# ---- 1. the processor ----
proc = None
if py == sys.executable:
    from transformers.models.inkling2.image_processing_inkling2 import Inkling2ImageProcessor
    proc = Inkling2ImageProcessor()
for img in images:
    out = os.path.join(tmp, "pv.f32")
    gh, gw, tokens, w, h = (int(v) for v in run(vit_check, "prep", model_dir, img, out).split())
    got = np.fromfile(out, np.float32)
    if proc is not None:
        ref = proc(images=[Image.open(img).convert("RGB")], return_tensors="pt")
        want = ref["pixel_values"].numpy().reshape(-1)
        want_grid = tuple(int(v) for v in ref["image_grid_thw"][0][1:])
    else:  # the reference interpreter writes the processor's output too
        rgb, rw, rh = decoded(img)
        rgh, rgw, rt = (int(v) for v in run(py, "-I", ref_script, model_dir, rgb, rw, rh, os.path.join(tmp, "ref")).split())
        want = np.fromfile(os.path.join(tmp, "ref", "pv.f32"), np.float32)
        want_grid = (rgh, rgw)
    same = want_grid == (gh, gw) and want.size == got.size
    d = float(np.abs(got - want).max()) if same else float("inf")
    report(f"pixel_values {os.path.basename(img)}", same and d == 0.0,
           f"{w}x{h}: {gh}x{gw} tokens (reference {want_grid[0]}x{want_grid[1]}), max |diff| {d:.3g}")

# ---- 2. the tower (GPU) ----
if GPU:
    depth = None
    for img in images:
        rgb, w, h = decoded(img)
        ref32, ref16 = os.path.join(tmp, "f32"), os.path.join(tmp, "bf16")
        gh, gw, tokens = (int(v) for v in run(py, "-I", ref_script, model_dir, rgb, w, h, ref32).split())
        run(py, "-I", ref_script, model_dir, rgb, w, h, ref16, "--bf16")
        rows = 1 + tokens * 4
        depth = depth or int(json.load(open(os.path.join(model_dir, "config.json")))["vision_config"]["num_hidden_layers"])
        worst_ratio, worst_cos = 0.0, 1.0
        lines = []
        for stage, layers in (("h0", 0), ("h1", 1), ("hL", depth), ("emb", None)):
            out = os.path.join(tmp, "out.bin")
            cmd = [vit_check, "tower", model_dir, rgb, w, h, out] + ([layers] if layers is not None else [])
            r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"vit_check tower: {r.stderr.strip()}")
            ref = np.fromfile(os.path.join(ref32, stage + ".f32"), np.float32)
            bf = np.fromfile(os.path.join(ref16, stage + ".f32"), np.float32)
            if layers is not None:
                got = np.fromfile(out, np.float32).reshape(rows, -1)
            else:  # bf16 embeddings
                raw = np.fromfile(out, np.uint16).astype(np.uint32) << 16
                got = raw.view(np.float32).reshape(tokens, -1)
            ref, bf = ref.reshape(got.shape), bf.reshape(got.shape)
            krel, kmax, kcos = stats(got, ref)
            brel, bmax, bcos = stats(bf, ref)
            ratio = krel / max(brel, 1e-9)
            worst_ratio = max(worst_ratio, ratio)
            if stage == "emb":
                worst_cos = kcos
            lines.append(f"{stage}: Kapitsa rel {krel:.2e} max {kmax:.2e} cos>={kcos:.5f} | bf16 rel {brel:.2e} "
                         f"max {bmax:.2e} cos>={bcos:.5f} | ratio {ratio:.2f}")
        ok = worst_ratio <= 2.0 and worst_cos >= 0.999
        report(f"tower {os.path.basename(img)}", ok,
               f"{w}x{h}, {tokens} tokens ({gh}x{gw}), {rows} rows; worst ratio {worst_ratio:.2f}, output cos>={worst_cos:.5f}\n  "
               + "\n  ".join(lines))

print(f"\n{fails} failed")
sys.exit(1 if fails else 0)
