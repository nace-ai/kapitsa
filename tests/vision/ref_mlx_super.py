#!/usr/bin/env python3
"""Reference next-token distributions of Nemotron 3.5 Super-VL with images, from mlx_vlm (git main: its
nemotron_h_nano_omni model on NVIDIA's NVFP4 checkpoint as stored), for tools/nh_eval --image --ref. By default in
float32: every floating parameter cast to fp32 and the FP8 matrices dequantized in fp32 (run it with MLX_ENABLE_TF32=0);
--bf16 runs mlx_vlm as it loads by default. GPU and ~90 GB: run it under the GPU lock, alone. Both engines get the
same tokens and pixels: the prompt from Kapitsa's nh_tok (the chat template), each image's pixel_values from vit_check
prep (bit-exact with NVIDIA's processor).

usage: ref_mlx_super.py MODEL_DIR BUILD_DIR OUT_DIR CASES.json [--topk K] [--bf16]
CASES.json: [{"name", "question", "answer", "images": [paths]}]: one user message (the images, then the question;
thinking off), then the answer, scored teacher-forced. Writes OUT_DIR/NAME.ids (nh_eval's @file input), NAME.ref
(NIEVREF2: the realized token's log-prob and the top-K, answer positions; the prompt's zero), NAME.json (skip) and
NAME.emb.f32 (mlx_vlm's image embeddings, the projector's output, float32 [tokens, hidden] per image in order).
Then: nh_eval MODEL_DIR --ref NAME.ref --skip SKIP --topk K --image IMAGE ... @NAME.ids
"""
import json, os, struct, subprocess, sys, tempfile, time

import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

args = sys.argv[1:]
topk = 4096
if "--topk" in args:
    i = args.index("--topk")
    topk = int(args[i + 1])
    del args[i:i + 2]
bf16 = "--bf16" in args
if bf16:
    args.remove("--bf16")
model_dir, build, out_dir, cases = args[0], args[1], args[2], json.load(open(args[3]))
os.makedirs(out_dir, exist_ok=True)
tmp = tempfile.mkdtemp()
cfg = json.load(open(os.path.join(model_dir, "config.json")))
IMG, IMG_START, IMG_END = cfg.get("img_context_token_id", 18), 19, 20

if not bf16:
    # mlx_vlm dequantizes FP8 matrices with an F32 scale to bf16, which would round the weights: fp32 instead
    from mlx_vlm.quantization import checkpoint as ck

    def deq32(weight, scale, block_structure=None):
        assert block_structure is None, "per-tensor / per-channel FP8 only"
        decoded = ck._E4M3_DECODE_LUT[weight.astype(mx.uint32)]
        s = scale.astype(mx.float32)
        if s.ndim == 1:
            s = s[:, None]
        return decoded * s

    ck._dequantize_compressed_tensors_fp8_weight = deq32

from mlx_vlm import load


def tok(*a):
    return [int(v) for v in subprocess.run([os.path.join(build, "nh_tok"), model_dir, *a], capture_output=True,
                                           text=True, check=True).stdout.strip().split(",") if v]


t0 = time.time()
model, _ = load(model_dir)
if not bf16:
    model.update(tree_unflatten([(k, v.astype(mx.float32)) for k, v in tree_flatten(model.parameters())
                                 if v.dtype in (mx.bfloat16, mx.float16)]))
mx.eval(model.parameters())
if model.vision_final_layernorm is None:
    sys.exit("mlx_vlm built Super-VL's projector without its final LayerNorm")
print(f"loaded {model_dir} in {time.time() - t0:.0f} s ({'bf16' if bf16 else 'float32'})", flush=True)
for case in cases:
    name = case["name"]
    pvs, ntok = [], []
    for k, img in enumerate(case["images"]):
        f = os.path.join(tmp, f"{name}{k}.f32")
        r = subprocess.run([os.path.join(build, "vit_check"), "prep", model_dir, img, f], capture_output=True,
                           text=True, check=True)
        _, _, tokens, W, H = (int(v) for v in r.stdout.split())
        pvs.append(np.fromfile(f, np.float32).reshape(1, 3, H, W))
        ntok.append(tokens)
    req = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}}
                                                     for _ in case["images"]] + [{"type": "text", "text": case["question"]}]}],
           "chat_template_kwargs": {"enable_thinking": False}}
    rq = os.path.join(tmp, "req.json")
    json.dump(req, open(rq, "w"))
    prompt, k = [], 0
    for t in tok("--render", rq, "--ids"):
        if t == IMG:
            prompt += [IMG_START] + [IMG] * ntok[k] + [IMG_END]
            k += 1
        else:
            prompt.append(t)
    if k != len(pvs):
        sys.exit(f"{name}: the template rendered {k} of {len(pvs)} images")
    ids = prompt + tok(case["answer"])
    pixel = None if not pvs else (mx.array(pvs[0]) if len(pvs) == 1 else [mx.array(p) for p in pvs])
    t0 = time.time()
    if pvs:
        emb = mx.concatenate([model.extract_feature(mx.array(p)).reshape(-1, cfg["llm_config"]["hidden_size"])
                              for p in pvs]).astype(mx.float32)
        mx.eval(emb)
        np.array(emb).tofile(os.path.join(out_dir, name + ".emb.f32"))
    out = model(mx.array([ids]), pixel_values=pixel)
    logits = (out.logits if hasattr(out, "logits") else out)[0].astype(mx.float32)
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(logp)
    logp = np.array(logp)
    skip = len(prompt) - 1
    with open(os.path.join(out_dir, name + ".ref"), "wb") as f:
        f.write(b"NIEVREF2")
        f.write(struct.pack("<III", 1, logp.shape[1], topk))
        f.write(struct.pack("<I", len(ids)))
        f.write(np.array(ids, np.int32).tobytes())
        for p in range(len(ids) - 1):
            if p < skip:
                f.write(b"\0" * (4 + 8 * topk))
                continue
            row = logp[p]
            top = np.argpartition(-row, topk)[:topk]
            top = top[np.argsort(-row[top])]
            f.write(struct.pack("<f", float(row[ids[p + 1]])))
            f.write(np.stack([top.astype(np.uint32).view(np.float32), row[top].astype(np.float32)], 1).tobytes())
    open(os.path.join(out_dir, name + ".ids"), "w").write(",".join(map(str, ids)))
    json.dump({"skip": skip, "prompt_tokens": len(prompt), "answer_tokens": len(ids) - len(prompt), "image_tokens": ntok,
               "images": case["images"]}, open(os.path.join(out_dir, name + ".json"), "w"))
    print(f"{name}: {len(prompt)} prompt + {len(ids) - len(prompt)} answer tokens, images {ntok}, forward "
          f"{time.time() - t0:.1f} s", flush=True)
