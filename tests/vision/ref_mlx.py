#!/usr/bin/env python3
"""Reference next-token distributions of image prompts from mlx_vlm, for build/ni_eval --image (GPU: run it under the
GPU lock, alone).

usage: ref_mlx.py MODEL_DIR OUT_DIR CASES.json [--max-tokens N] [--topk K]
CASES.json: [{"name": ..., "question": ..., "images": [paths]}, ...]. For each case, one user message (the images, then
the question; thinking off): mlx_vlm greedily writes an answer, then the prompt + answer is scored teacher-forced in one
forward. Writes to OUT_DIR:
  NAME.ids        the token ids (comma separated: ni_eval's @file input)
  NAME.ref        ni_eval's reference file (NIEVREF2): per scored position the realized token's log-prob and the
                  top-K log-probs (positions from the last prompt token on; the others zero)
  NAME.json       the answer text, prompt length, grids, timings
  NAME.pv.npy     mlx_vlm's pixel_values, NAME.vision.npy its vision tower output (float32)
Then: ni_eval MODEL --ref NAME.ref --skip <prompt_len - 1> --topk K --image IMAGE ... @NAME.ids
"""
import json, os, struct, sys, time

import numpy as np

args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return type(default)(v)
    return default
max_tokens = opt("--max-tokens", 96)
topk = opt("--topk", 4096)
model_dir, out_dir, cases = args[0], args[1], json.load(open(args[2]))

import mlx.core as mx
from mlx_vlm import load, prepare_inputs, stream_generate

t0 = time.time()
model, processor = load(model_dir)
tok = processor.tokenizer
t_load = time.time() - t0

for case in cases:
    name, question, images = case["name"], case["question"], case["images"]
    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": question}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)

    # the answer: greedy
    answer_ids = []
    t0 = time.time()
    for r in stream_generate(model, processor, prompt, image=list(images), max_tokens=max_tokens, temperature=0.0):
        if r.token is not None:
            answer_ids.append(int(r.token))
    t_gen = time.time() - t0
    answer = tok.decode(answer_ids)

    inputs = prepare_inputs(processor, images=list(images), prompts=prompt)
    prompt_ids = np.array(inputs["input_ids"]).reshape(-1).tolist()
    pixel_values = inputs.get("pixel_values")  # None: a text-only case (the baseline)
    grid = inputs.get("image_grid_thw")
    full = prompt_ids + answer_ids
    n = len(full)
    skip = len(prompt_ids) - 1

    if pixel_values is not None:  # the vision tower alone (as get_input_embeddings runs it)
        pv = pixel_values.astype(model.vision_tower.patch_embed.proj.weight.dtype)
        feats, _ = model.vision_tower(pv, grid)
        np.save(os.path.join(out_dir, name + ".vision.npy"), np.array(feats.astype(mx.float32)))
        np.save(os.path.join(out_dir, name + ".pv.npy"), np.array(pixel_values.astype(mx.float32)))

    # teacher-forced forward of prompt + answer
    t0 = time.time()
    out = (model(mx.array([full]), pixel_values=pixel_values, image_grid_thw=grid) if pixel_values is not None
           else model(mx.array([full])))
    logits = out.logits if hasattr(out, "logits") else out
    lg = logits[0, skip:n - 1].astype(mx.float32)
    lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
    mx.eval(lp)
    lp = np.array(lp)
    t_fwd = time.time() - t0
    V = lp.shape[1]
    K = min(topk, V)
    with open(os.path.join(out_dir, name + ".ref"), "wb") as f:
        f.write(b"NIEVREF2" + struct.pack("<III", 1, V, K))
        f.write(struct.pack("<I", n) + np.array(full, np.int32).tobytes())
        zero = np.zeros(1 + 2 * K, np.float32).tobytes()
        for p in range(n - 1):
            if p < skip:
                f.write(zero)
                continue
            row = lp[p - skip]
            top = np.argpartition(-row, K - 1)[:K]
            top = top[np.argsort(-row[top])]
            rec = np.empty((K, 2), np.float32)
            rec[:, 0] = top.astype(np.uint32).view(np.float32)
            rec[:, 1] = row[top]
            f.write(np.float32(row[full[p + 1]]).tobytes() + rec.tobytes())
    with open(os.path.join(out_dir, name + ".ids"), "w") as f:
        f.write(",".join(map(str, full)))
    json.dump({"answer": answer, "prompt_tokens": len(prompt_ids), "answer_tokens": len(answer_ids), "skip": skip,
               "grids": np.array(grid).tolist() if grid is not None else [], "load_s": t_load, "generate_s": t_gen, "forward_s": t_fwd,
               "images": images, "question": question},
              open(os.path.join(out_dir, name + ".json"), "w"), indent=1)
    print(json.dumps({"name": name, "answer": answer, "prompt_tokens": len(prompt_ids),
                      "grids": np.array(grid).tolist() if grid is not None else []}))
