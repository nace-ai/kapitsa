#!/usr/bin/env python3
"""Inkling-2's image input against Hugging Face's reference (Inkling2ImageProcessor and Inkling2VisionModel,
tests/vision/ref_ink2.py).

usage: check_ink2.py MODEL_DIR VIT_CHECK IMAGE... [--gpu] [--python PY]
  1. the processor (CPU): `vit_check prep` against Inkling2ImageProcessor on every image file as PIL opens it
     (transparency kept: the reference composites it onto its checkerboard): the token grid and pixel_values, bit for bit
  2. --gpu: `vit_check tower` against ref_ink2.py in float32 after the embeddings' LayerNorm, block 1 and the last block
     (every row: the class token and the patches) and at the merger's output (the language model's image rows); the
     tower in bfloat16 as it ships gives the scale. Passes when Kapitsa's relative error is within twice bfloat16's at
     every stage and no output row's cosine to the reference is under 0.999. This leg holds the GPU lock
     ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) throughout, unless the caller says it holds it (NI_LOCK_HELD=1 or
     NACE_GPU_LOCK_HELD=1).
Needs torch, torchvision, transformers 5.24 or later, Pillow and numpy (--python PY: run ref_ink2.py with PY, which
has them, while this script needs only Pillow and numpy). tests/fixtures/vision/ink2_*.png are transparent fixtures.
"""
import atexit, json, os, shutil, subprocess, sys, tempfile

try:
    import numpy as np
except ImportError:  # the lock's checks run this script with stand-ins and no numpy: every comparison then fails
    np = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench", "harness"))
import gpu_hogs  # noqa: E402

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
child_env = dict(os.environ)


def report(name, ok, detail):
    global fails
    print(("ok   " if ok else "FAIL ") + name + ": " + detail, flush=True)
    fails += not ok


def run(*cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=child_env)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd))}: {r.stderr.strip()}")
    return r.stdout.strip()


def stats(got, ref):
    """relative L2 error over every value, the largest |diff|, the smallest row cosine"""
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-30))
    worst = float(np.abs(got - ref).max())
    cos = (got * ref).sum(1) / np.maximum(np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1), 1e-30)
    return rel, worst, float(cos.min())


# ---- 1. the processor ----
for img in images:
    name = os.path.basename(img)
    try:
        out, ref = os.path.join(tmp, "pv.f32"), os.path.join(tmp, "ref")
        gh, gw, tokens, w, h = (int(v) for v in run(vit_check, "prep", model_dir, img, out).split())
        got = np.fromfile(out, np.float32)
        rgh, rgw, _ = (int(v) for v in run(py, "-I", ref_script, model_dir, img, ref, "--prep").split())
        want = np.fromfile(os.path.join(ref, "pv.f32"), np.float32)
        same = (rgh, rgw) == (gh, gw) and want.size == got.size
        d = float(np.abs(got - want).max()) if same else float("inf")
        report(f"pixel_values {name}", same and d == 0.0,
               f"{w}x{h}: {gh}x{gw} tokens (reference {rgh}x{rgw}), max |diff| {d:.3g}")
    except Exception as e:  # noqa: BLE001 (one image's failure is reported, the others still run)
        report(f"pixel_values {name}", False, str(e)[:300])

# ---- 2. the tower (GPU): under the GPU lock from the first tower run to the last ----
if GPU:
    lock = None
    if not gpu_hogs.held_by_caller():
        gpu_hogs.configure()
        lock = gpu_hogs.GpuLock().acquire()
    child_env.update(NI_LOCK_HELD="1")  # the children run under our lock: none takes it again
    try:
        depth = int(json.load(open(os.path.join(model_dir, "config.json")))["vision_config"]["num_hidden_layers"])
    except (OSError, KeyError, ValueError):
        depth = None
    try:
        for img in images:
            name = os.path.basename(img)
            try:
                outs = {}
                for stage, layers in (("h0", 0), ("h1", 1), ("hL", depth), ("emb", None)):
                    outs[stage] = os.path.join(tmp, stage + ".bin")
                    run(vit_check, "tower", model_dir, img, outs[stage], *([layers] if layers is not None else []))
                ref32, ref16 = os.path.join(tmp, "f32"), os.path.join(tmp, "bf16")
                gh, gw, tokens = (int(v) for v in run(py, "-I", ref_script, model_dir, img, ref32).split())
                run(py, "-I", ref_script, model_dir, img, ref16, "--bf16")
                rows = 1 + tokens * 4
                worst_ratio, worst_cos, lines = 0.0, 1.0, []
                for stage in ("h0", "h1", "hL", "emb"):
                    ref = np.fromfile(os.path.join(ref32, stage + ".f32"), np.float32)
                    bf = np.fromfile(os.path.join(ref16, stage + ".f32"), np.float32)
                    if stage != "emb":
                        got = np.fromfile(outs[stage], np.float32).reshape(rows, -1)
                    else:  # bf16 embeddings
                        raw = np.fromfile(outs[stage], np.uint16).astype(np.uint32) << 16
                        got = raw.view(np.float32).reshape(tokens, -1)
                    ref, bf = ref.reshape(got.shape), bf.reshape(got.shape)
                    krel, kmax, kcos = stats(got, ref)
                    brel, bmax, bcos = stats(bf, ref)
                    ratio = krel / max(brel, 1e-9)
                    worst_ratio = max(worst_ratio, ratio)
                    worst_cos = kcos if stage == "emb" else worst_cos
                    lines.append(f"{stage}: Kapitsa rel {krel:.2e} max {kmax:.2e} cos>={kcos:.5f} | bf16 rel {brel:.2e} "
                                 f"max {bmax:.2e} cos>={bcos:.5f} | ratio {ratio:.2f}")
                report(f"tower {name}", worst_ratio <= 2.0 and worst_cos >= 0.999,
                       f"{tokens} tokens ({gh}x{gw}), {rows} rows; worst ratio {worst_ratio:.2f}, output cos>={worst_cos:.5f}"
                       "\n  " + "\n  ".join(lines))
            except Exception as e:  # noqa: BLE001
                report(f"tower {name}", False, str(e)[:300])
    finally:
        if lock:
            lock.release()

print(f"\n{fails} failed")
sys.exit(1 if fails else 0)
