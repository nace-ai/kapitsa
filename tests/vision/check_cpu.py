#!/usr/bin/env python3
"""CPU checks of Kapitsa's image input against the reference implementations (no GPU work).

usage: check_cpu.py <path to build/vit_check> <model dir> [--quick]

Run it in a venv with Pillow and numpy; with torch, torchvision and transformers (e.g. research/inference/venvs/
rapid-mlx) it also compares against Hugging Face's Qwen2-VL image processors and Qwen3.5's get_rope_index.
  1. decoding: vit_check decode vs PIL (ImageOps.exif_transpose(Image.open(f)).convert("RGB")) over PNG (RGB, RGBA,
     gray, gray + alpha, palette, palette + transparency, 16-bit), JPEG (+ EXIF orientation), WebP, GIF, BMP, TIFF
  2. resizing: Pillow's bicubic resize of random images to random sizes, byte for byte
  3. smart_resize over many sizes
  4. preprocessing: pixel_values vs the reference's float32 arithmetic on Pillow's pixels (mlx_vlm's processor), and
     vs Hugging Face's PIL and torchvision processors
  5. M-RoPE positions of prompts with several images of different sizes vs get_rope_index
"""
import atexit, io, json, math, os, random, shutil, subprocess, sys, tempfile

import numpy as np
from PIL import Image, ImageOps

VIT, MODEL = sys.argv[1], sys.argv[2]
QUICK = "--quick" in sys.argv
rng = random.Random(7)
fails = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + (": " + detail if detail else ""))
    if not ok:
        fails.append(name)


def run(*a):
    r = subprocess.run([VIT, *map(str, a)], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"vit_check {' '.join(map(str, a))}: {r.stderr.strip()}")
    return r.stdout.strip()


def noise(w, h, smooth=False):
    a = np.random.default_rng(w * 1000 + h).integers(0, 256, (h, w, 3), dtype=np.uint8)
    if smooth:  # gradients + noise: resampling sees edges and flat areas
        y, x = np.mgrid[0:h, 0:w]
        a = ((a.astype(np.int32) // 4 + (x * 255 // max(w - 1, 1))[..., None] // 2 + (y * 255 // max(h - 1, 1))[..., None] // 4)
             % 256).astype(np.uint8)
    return a


tmp = tempfile.mkdtemp(prefix="vit_cpu_")
atexit.register(shutil.rmtree, tmp, True)  # small files, deleted as the checks go and at the end

# ---- 1. decoding ----
base = Image.fromarray(noise(97, 61, smooth=True))
alpha = Image.fromarray(np.random.default_rng(1).integers(0, 256, (61, 97), dtype=np.uint8))
cases = {
    "rgb.png": (base, {}),
    "rgba.png": (Image.merge("RGBA", (*base.split(), alpha)), {}),
    "gray.png": (base.convert("L"), {}),
    "graya.png": (Image.merge("LA", (base.convert("L"), alpha)), {}),
    "pal.png": (base.convert("P", palette=Image.Palette.ADAPTIVE, colors=200), {}),
    "pal_t.png": (base.convert("P", palette=Image.Palette.ADAPTIVE, colors=200), {"transparency": 3}),
    "q90.jpg": (base, {"quality": 90}),
    "lossless.webp": (base, {"lossless": True}),
    "lossy.webp": (base, {"quality": 80}),
    "anim.gif": (base.convert("P", palette=Image.Palette.ADAPTIVE, colors=128), {}),
    "rgb.bmp": (base, {}),
    "rgb.tiff": (base, {}),
}
for name, (im, kw) in cases.items():
    im.save(os.path.join(tmp, name), **kw)
# 16-bit RGB PNG (PIL reads it as 8-bit RGB by the high bytes)
a16 = (np.random.default_rng(3).integers(0, 65536, (33, 45, 3))).astype(">u2")
import zlib, struct


def png16(path, arr):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)) +
                chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


png16(os.path.join(tmp, "rgb16.png"), a16)
# EXIF orientations 2-8 on a JPEG (re-encoded from one decode so the pixels agree)
for o in range(2, 9):
    ex = Image.Exif()
    ex[0x0112] = o
    base.save(os.path.join(tmp, f"orient{o}.jpg"), quality=95, exif=ex.tobytes())

for name in sorted(os.listdir(tmp)):
    path = os.path.join(tmp, name)
    ref = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB"))
    out = run("decode", path, path + ".rgb").split()
    w, h = int(out[0]), int(out[1])
    ours = np.fromfile(path + ".rgb", np.uint8)
    if (h, w) != ref.shape[:2] or ours.size != ref.size:
        check(f"decode {name}", False, f"size {w}x{h} vs {ref.shape[1]}x{ref.shape[0]}")
        continue
    os.remove(path + ".rgb")
    d = np.abs(ours.reshape(ref.shape).astype(int) - ref.astype(int))
    lossy = name.endswith((".jpg", "lossy.webp"))
    exact = d.max() == 0
    check(f"decode {name}", exact or lossy,
          ("exact" if exact else f"max diff {d.max()}, mean {d.mean():.3f}, {100 * (d > 0).mean():.1f}% of values differ"
           + (" (lossy codec: ImageIO's decoder is not libjpeg/libwebp)" if lossy else "")))

# ---- 2. resizing ----
n_ok = n = 0
worst = ""
sizes = [(97, 61, 32, 32), (61, 97, 256, 256), (1920, 1080, 1920, 1088), (1080, 1920, 1088, 1920), (500, 500, 512, 480),
         (2880, 1800, 2880, 1792), (17, 3000, 32, 2976), (640, 480, 640, 480), (1000, 1, 1024, 32), (3, 5, 256, 192),
         # more than 100 times taller than wide: Pillow runs the vertical pass first
         (3, 368, 372, 196), (4, 3317, 305, 646), (10, 1001, 32, 992), (10, 1000, 32, 992), (2, 201, 64, 64)]
for _ in range(20 if QUICK else 120):
    w, h = rng.randint(1, 900), rng.randint(1, 900)
    sizes.append((w, h, rng.randint(1, 1200), rng.randint(1, 1200)))
for w, h, ow, oh in sizes:
    a = noise(w, h, smooth=n % 2 == 0)
    p = os.path.join(tmp, "r.rgb")
    a.tofile(p)
    run("resize", p, w, h, ow, oh, p + ".out")
    ours = np.fromfile(p + ".out", np.uint8).reshape(oh, ow, 3)
    ref = np.asarray(Image.fromarray(a).resize((ow, oh), Image.BICUBIC))
    n += 1
    if np.array_equal(ours, ref):
        n_ok += 1
    elif not worst:
        worst = f"{w}x{h} -> {ow}x{oh}: max diff {np.abs(ours.astype(int) - ref).max()}"
check("Pillow bicubic resize", n_ok == n, f"{n_ok}/{n} sizes byte-exact" + (f"; first mismatch {worst}" if worst else ""))


# ---- 3. smart_resize ----
def smart_resize(height, width, factor, min_pixels, max_pixels):  # Hugging Face's qwen2_vl smart_resize
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


bad = []
probes = [(1080, 1920), (1800, 2880), (16, 16), (48, 48), (80, 3), (3, 500), (4000, 3000), (8000, 6000), (16, 3200)]
probes += [(rng.randint(1, 9000), rng.randint(1, 9000)) for _ in range(300 if QUICK else 3000)]
probes += [(32 * k + 16, 32 * j + 16) for k in range(1, 40, 7) for j in range(1, 40, 5)]  # Python's round half to even
for h, w in probes:
    if max(h, w) / min(h, w) > 200:
        continue
    for lim in [(65536, 16777216), (3136, 1003520), (65536, 1048576)]:
        ref = smart_resize(h, w, 32, *lim)
        ours = tuple(map(int, run("smart", h, w, 32, *lim).split()))
        if ref != ours:
            bad.append(f"{h}x{w} {lim}: {ours} vs {ref}")
check("smart_resize", not bad, f"{len(probes) * 3} cases" + (f"; {bad[:3]}" if bad else ""))

# ---- 4. preprocessing ----
pre = json.load(open(os.path.join(MODEL, "preprocessor_config.json")))
proc = json.load(open(os.path.join(MODEL, "processor_config.json"))).get("image_processor", {}) if os.path.exists(
    os.path.join(MODEL, "processor_config.json")) else {}
pre.update(proc)
min_px, max_px = pre["size"]["shortest_edge"], pre["size"]["longest_edge"]
min_px, max_px = pre.get("min_pixels", min_px), pre.get("max_pixels", max_px)
mean = np.array(pre.get("image_mean", [0.5] * 3), np.float32)[:, None, None]
std = np.array(pre.get("image_std", [0.5] * 3), np.float32)[:, None, None]
rescale = pre.get("rescale_factor", 1 / 255)
P, T, M = pre.get("patch_size", 16), pre.get("temporal_patch_size", 2), pre.get("merge_size", 2)


def reference_pixel_values(img):  # the reference processor's arithmetic (mlx_vlm's numpy port: float32 steps)
    a = np.asarray(img.convert("RGB"))
    h, w = smart_resize(a.shape[0], a.shape[1], P * M, min_px, max_px)
    if (h, w) != a.shape[:2]:
        a = np.asarray(Image.fromarray(a).resize((w, h), Image.BICUBIC))
    x = np.transpose(a, (2, 0, 1)).astype(np.float32) * rescale
    x = (x - mean) / std
    gh, gw = h // P, w // P
    x = np.repeat(x[None], T, axis=0)  # (T, C, H, W)
    x = x.reshape(T, 3, gh // M, M, P, gw // M, M, P).transpose(2, 5, 3, 6, 1, 0, 4, 7)
    return x.reshape(gh * gw, 3 * T * P * P), (1, gh, gw)


try:
    import torch  # noqa: F401
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    kw = dict(min_pixels=min_px, max_pixels=max_px, patch_size=P, temporal_patch_size=T, merge_size=M,
              image_mean=list(mean.ravel()), image_std=list(std.ravel()))
    hf = {"HF torchvision": Qwen2VLImageProcessor(**kw), "HF PIL": Qwen2VLImageProcessorPil(**kw)}
except Exception as e:  # noqa: BLE001
    print("(no Hugging Face processors here:", type(e).__name__, str(e)[:80], ")")
    hf = {}

prep_images = []
for (w, h) in [(640, 480), (1280, 800), (1920, 1080), (333, 777), (50, 2000), (256, 256), (1000, 999)]:
    prep_images.append((f"noise {w}x{h}", Image.fromarray(noise(w, h, smooth=True))))
photo = "/System/Library/Wallpapers/.default/DefaultAerial.jpg"  # a real photo (4096 x 2160), at 1280 x 675
if os.path.exists(photo):
    prep_images.append(("DefaultAerial.jpg at 1280x675", Image.open(photo).convert("RGB").resize((1280, 675), Image.LANCZOS)))
for name, img in prep_images:
    path = os.path.join(tmp, "prep.png")
    img.convert("RGB").save(path)  # PNG: our decode equals PIL's
    out = run("prep", MODEL, path, path + ".f32").split()
    ours = np.fromfile(path + ".f32", np.float32)
    os.remove(path + ".f32")
    ref, grid = reference_pixel_values(Image.open(path))
    ok_grid = (int(out[0]) * M, int(out[1]) * M) == grid[1:]
    exact = ours.size == ref.size and np.array_equal(ours.reshape(ref.shape), ref)
    detail = f"grid {grid}, {ref.shape[0]} patches: " + ("bit-exact" if exact else
              f"max diff {np.abs(ours.reshape(ref.shape) - ref).max() if ours.size == ref.size else 'shape'}")
    for hn, hp in hf.items():
        r = hp(images=[Image.open(path)], return_tensors="np")
        pv, g = np.asarray(r["pixel_values"], np.float32), tuple(int(v) for v in r["image_grid_thw"][0])
        if g != grid or pv.shape != ref.shape:
            detail += f"; {hn}: grid {g} differs"
            continue
        d = np.abs(ours.reshape(ref.shape) - pv)
        detail += f"; {hn}: max diff {d.max():.2e}" + (f" ({(d > 1e-6).mean() * 100:.2f}% > 1e-6)" if d.max() > 1e-6 else "")
    check(f"pixel_values {name}", ok_grid and exact, detail)

# ---- 5. M-RoPE positions ----
try:
    import torch
    from types import SimpleNamespace
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    IMG = 248056
    bad = []
    for trial in range(40 if QUICK else 200):
        ids, spans, grids, pos = [], [], [], 0
        for _ in range(rng.randint(1, 4)):
            text = rng.randint(0, 40)
            ids += [rng.randint(10, 1000) for _ in range(text)] + [248053]
            gh, gw = rng.randint(1, 12), rng.randint(1, 12)
            if rng.random() < 0.2:
                gh = gw = rng.randint(1, 12)
            spans.append(f"{len(ids)}:{gh}:{gw}")
            grids.append([1, gh * 2, gw * 2])
            ids += [IMG] * (gh * gw) + [248054]
        ids += [rng.randint(10, 1000) for _ in range(rng.randint(0, 30))]
        n = len(ids)
        input_ids = torch.tensor([ids])
        mm = (input_ids == IMG).int()
        fake = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)),
                               get_vision_position_ids=lambda *a, **k: Qwen3_5Model.get_vision_position_ids(None, *a, **k))
        ref, delta = Qwen3_5Model.get_rope_index(fake, input_ids, mm, image_grid_thw=torch.tensor(grids))
        out = os.path.join(tmp, "rope.i32")
        d_ours = int(run("rope", f"{n} {','.join(spans)}", out))
        ours = np.fromfile(out, np.int32).reshape(n, 3).T
        if not np.array_equal(ours, ref[:, 0].numpy()) or d_ours != int(delta[0, 0]):
            bad.append(f"trial {trial}: delta {d_ours} vs {int(delta[0, 0])}")
    check("M-RoPE positions vs get_rope_index", not bad, f"{40 if QUICK else 200} prompts, 1-4 images each" +
          (f"; {bad[:2]}" if bad else ""))
except ImportError as e:
    print("(no torch / transformers here for get_rope_index:", e, ")")

print(f"\n{len(fails)} failed" + (": " + ", ".join(fails) if fails else ""))
sys.exit(1 if fails else 0)
