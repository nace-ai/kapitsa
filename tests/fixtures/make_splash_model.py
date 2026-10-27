#!/usr/bin/env python3
"""Writes tests/fixtures/splash_model/: a small stand-in for Splash's resolved model folder, for tests/test_splash_engine.py
where no Splash is installed (CI). It keeps what the engine reads from the real folder: the structure of tokenizer.json (its
normalizer, pre-tokenizer, decoder and added tokens), tokenizer_config.json's eos token and the chat template as it is; the
vocabulary is the 256 byte pieces plus the few words the test replies use, with their merges.
usage: python3 tests/fixtures/make_splash_model.py [<Splash resolved model folder>]   (default: the first one under ~/Library)"""
import glob, json, os, shutil, sys

here = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.expanduser("~/Library/Application Support/Splash/models/.resolved/*")))[0]
out = os.path.join(here, "splash_model")
os.makedirs(out + "/tokenizer", exist_ok=True)
t = json.load(open(src + "/tokenizer/tokenizer.json"))
bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))  # the byte-level alphabet of GPT-2 style tokenizers
cs, n = bs[:], 0
for b in range(256):
    if b not in bs:
        bs.append(b); cs.append(256 + n); n += 1
vocab = {chr(c): i for i, c in enumerate(sorted(cs))}
merges = ["o k", "H i", "\u010a \u010a", "\u0120 t", "\u0120t h", "\u0120th e", "\u0120the r", "\u0120ther e"]
for m in merges:
    a, b = m.split(" ")
    vocab.setdefault(a + b, len(vocab))
t["model"]["vocab"], t["model"]["merges"] = vocab, merges
base = 1000
for i, a in enumerate(t["added_tokens"]):
    a["id"] = base + i
t["added_tokens"] = [a for a in t["added_tokens"] if a["content"] in (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<think>", "</think>")]
json.dump(t, open(out + "/tokenizer/tokenizer.json", "w"), ensure_ascii=False)
json.dump({"eos_token": "<|im_end|>"}, open(out + "/tokenizer/tokenizer_config.json", "w"))
shutil.copy(src + "/tokenizer/chat_template.jinja", out + "/tokenizer/chat_template.jinja")
json.dump({"model_type": "fixture"}, open(out + "/config.json", "w"))
