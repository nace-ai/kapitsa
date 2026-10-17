#!/usr/bin/env python3
"""Is a published MLX quantization exactly round-to-nearest (mx.quantize) of the BF16 weights? Compares one decoder
layer's tensors (CPU only: no GPU lock).

  check_rtn.py <bf16 repo> <quantized repo> [layer=0]
"""
import json, sys

import mlx.core as mx

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from eval_stream import Source, unit_of, repo_qcfg, strip  # noqa: E402
from concurrent.futures import ThreadPoolExecutor

mx.set_default_device(mx.cpu)
base, cand = Source(sys.argv[1]), Source(sys.argv[2])
li = int(sys.argv[3]) if len(sys.argv) > 3 else 0
u = ("layer", li)
pool = ThreadPoolExecutor(16)
bt = base.fetch([k for k in base.index if unit_of(k) == u], pool)
ct = cand.fetch([k for k in cand.index if unit_of(k) == u], pool)
bp = next(k for k in bt)[: next(k for k in bt).index(".layers.") + len(".layers.")] + f"{li}."
cp = next(k for k in ct)[: next(k for k in ct).index(".layers.") + len(".layers.")] + f"{li}."
bw, cw = strip(bt, bp), strip(ct, cp)
qfor = repo_qcfg(cand.config, cp)
same = diff = 0
for k in sorted(cw):
    if not k.endswith(".scales"):
        continue
    m = k[: -len(".scales")]
    q = qfor(m)
    w = bw[m + ".weight"]
    out = mx.quantize(w, q["group_size"], q["bits"], mode=q["mode"])
    names = ["weight", "scales", "biases"][: len(out)]
    eq = all(mx.array_equal(o, cw[f"{m}.{n}"]).item() for o, n in zip(out, names))
    if not eq:
        wd = mx.dequantize(*out, q["group_size"], q["bits"], mode=q["mode"], dtype=mx.float32) if len(out) == 3 else None
        cd = mx.dequantize(cw[m + ".weight"], cw[m + ".scales"], cw.get(m + ".biases"), q["group_size"], q["bits"],
                           mode=q["mode"], dtype=mx.float32)
        codes = mx.mean((out[0] != cw[m + ".weight"]).astype(mx.float32)).item()
        print(f"{m}: DIFFERS ({q['bits']}-bit g{q['group_size']}): words differing {codes:.4f}, "
              f"max|dw| {mx.max(mx.abs(wd - cd)).item() if wd is not None else -1:.3g}")
        diff += 1
    else:
        same += 1
print(json.dumps({"layer": li, "identical": same, "different": diff}))
