#!/usr/bin/env python3
"""Compares Kapitsa's tokenizer with HF tokenizers on a mixed corpus.
usage: tok_compare.py <ni_tok binary> <tokenizer.json> [files...]"""
import subprocess, sys, random
from tokenizers import Tokenizer
binary, tj, files = sys.argv[1], sys.argv[2], sys.argv[3:]
hf = Tokenizer.from_file(tj)
texts = [
    "Hello world", "  leading spaces", "trailing   ", "tabs\tand\nnewlines\r\n\r\n  x",
    "I'm can't we'll they've she'd HE'S", "numbers 1234567 3.14159 1e-9 0x1F",
    "unicode: café naïve résumé Zürich — “quotes” ‘single’", "CJK: 你好，世界！こんにちは 안녕하세요",
    "emoji 😀👍🏽👨‍👩‍👧 ✅", "math ∑∫√∞ ≠ ≤ ≥", "combining: e\u0301 a\u0308 n\u0303", "arabic مرحبا بالعالم",
    "hindi नमस्ते दुनिया", "thai สวัสดีชาวโลก", "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "def f(x):\n    return x**2  # comment\n\n\n\nclass A:\n\tpass\n", "    \n\n   \n", "a\u00a0b\u2003c", "!!!???...,,,;;;",
    "' '' ''' 's", "<tool_call>\n<function=read_file>\n<parameter=path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>",
]
for f in files:
    data = open(f, encoding="utf-8", errors="ignore").read()
    rnd = random.Random(0)
    for _ in range(40):
        a = rnd.randrange(0, max(1, len(data) - 2000)); texts.append(data[a:a + rnd.randrange(50, 2000)])
inp = "\n\x1e\n".join(texts)
out = subprocess.run([binary, tj], input=inp.encode(), capture_output=True).stdout.decode().splitlines()
bad = 0
for t, line in zip(texts, out):
    ours = [int(x) for x in line.split(",")] if line else []
    ref = hf.encode(t, add_special_tokens=False).ids  # (a post-processor's BOS, as Llama's, is not text)
    if ours != ref:
        bad += 1
        if bad <= 5:
            i = next((k for k in range(min(len(ours), len(ref))) if ours[k] != ref[k]), min(len(ours), len(ref)))
            print("MISMATCH", repr(t[:80]), "at", i, "ours", ours[i:i+5], [hf.decode([x]) for x in ours[i:i+5]], "ref", ref[i:i+5], [hf.decode([x]) for x in ref[i:i+5]])
print(f"{len(texts) - bad}/{len(texts)} identical")
