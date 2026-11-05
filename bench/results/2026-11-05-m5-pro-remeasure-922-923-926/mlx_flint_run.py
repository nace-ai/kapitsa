#!/usr/bin/env python3
"""#343: mlx-vlm 0.7.11 on Flint 4-bit. The pack stores packed weights as safetensors U8; mlx-vlm expects uint32, so
mx.load's result is reinterpreted (same bytes: .view(uint32)) for every 2-D+ '*.weight' that is uint8. Nothing on disk changes.
usage: mlx_flint_run.py MODEL_DIR PROMPT_FILE MAX_TOKENS [--validate]   (--validate: lazy load on the CPU, shapes only)"""
import sys, time
import mlx.core as mx
_load = mx.load
def load(f, *a, **k):
    w = _load(f, *a, **k)
    if isinstance(w, dict):
        for key, v in list(w.items()):
            if key.endswith(".weight") and v.dtype == mx.uint8 and v.ndim >= 2:
                w[key] = v.view(mx.uint32)
    return w
mx.load = load
import mlx_vlm.utils as U
U.mx.load = load
model_dir, prompt_file, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
if "--validate" in sys.argv:
    mx.set_default_device(mx.cpu)
    model, proc = U.load(model_dir, lazy=True)
    print("VALIDATED: weights loaded lazily on the CPU, shapes match"); sys.exit(0)
from mlx_vlm import generate
model, proc = U.load(model_dir)
prompt = open(prompt_file).read()
t0 = time.time()
r = generate(model, proc, prompt, max_tokens=max_tokens, temperature=0.0, verbose=False)
print("RESULT prompt_tokens=%s prompt_tps=%.1f gen_tokens=%s gen_tps=%.2f peak_gb=%.2f wall_s=%.1f" % (
    r.prompt_tokens, r.prompt_tps, r.generation_tokens, r.generation_tps, r.peak_memory, time.time() - t0))
