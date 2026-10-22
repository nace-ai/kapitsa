#!/usr/bin/env python3
"""Rebuilds bench_corpus.txt from its source files and checks it byte for byte (NOTICE.md).

The corpus is excerpts of Hugging Face Transformers' modeling files at tag v5.17.0 (Apache License 2.0): each file's
first 20,000 characters under a "### FILE: <name>" heading in a code fence, concatenated in FILES order and cut to
650,000 characters. The first eight files, cut to 144,500 characters, are the original corpus (sha256 V1_SHA256);
the extension only appends, so every prompt that fits in the first 144,500 characters is unchanged.

usage: python3 make_corpus.py [--src DIR] [--out FILE]
  --src DIR   the modeling_*.py files, downloaded there if missing (default: a temp dir)
  --out FILE  write the corpus there (default: only check it against bench_corpus.txt and the sha256 below)
Stdlib only; downloads from raw.githubusercontent.com.
"""
import argparse, hashlib, os, sys, tempfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "v5.17.0"
URL = "https://raw.githubusercontent.com/huggingface/transformers/{tag}/src/transformers/models/{m}/modeling_{m}.py"
FILES = [
    # the original corpus (V1)
    "granite_speech", "modernvbert", "altclip", "xcodec", "t5", "ctrl", "emu3", "videoprism",
    # the extension (alphabetical; files whose license header is not in their first lines were left out)
    "bart", "bert", "bloom", "clip", "deberta_v2", "detr", "electra", "falcon", "gemma", "gemma2", "gpt2", "led",
    "llama", "longformer", "mamba", "mbart", "mixtral", "opt", "pegasus", "phi3", "qwen2_moe", "qwen3", "roberta",
    "swin", "vit",
]
EXCERPT = 20000
CHARS = 650000
V1_CHARS = 144500
V1_SHA256 = "1a9bb103b3f6d39ee5b9f5bf7207e304c5ce92788c273685f597508c7d2bde28"
SHA256 = "58a7756fd94fc48a7b7306d330b2c44663bdf22faa943f82430966c2878081b2"  # NOTICE.md


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build(src):
    parts = []
    for m in FILES:
        p = os.path.join(src, f"modeling_{m}.py")
        if not os.path.exists(p):
            urllib.request.urlretrieve(URL.format(tag=TAG, m=m), p)
        with open(p, encoding="utf-8", newline="") as fh:
            text = fh.read()
        if "Apache License, Version 2.0" not in text[:3000]:
            raise SystemExit(f"{p}: no Apache-2.0 header in its first 3,000 characters")
        parts.append(f"\n\n### FILE: modeling_{m}.py\n```python\n{text[:EXCERPT]}\n```")
    c = "".join(parts)
    if len(c) < CHARS:
        raise SystemExit(f"the excerpts have {len(c)} characters, fewer than {CHARS}")
    return c[:CHARS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    src = a.src or tempfile.mkdtemp(prefix="bench-corpus-")
    os.makedirs(src, exist_ok=True)
    c = build(src)
    ok = True
    for what, got, want in (("the first 144,500 characters (V1)", sha(c[:V1_CHARS]), V1_SHA256),
                            ("the corpus", sha(c), SHA256)):
        print(f"{what}: sha256 {got} {'ok' if got == want else 'DIFFERS from ' + want}")
        ok &= got == want
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="") as fh:
            fh.write(c)
        print(f"wrote {a.out}: {len(c):,} characters, {len(c.encode('utf-8')):,} bytes")
    else:
        with open(os.path.join(HERE, "bench_corpus.txt"), encoding="utf-8", newline="") as fh:
            same = fh.read() == c
        print("bench_corpus.txt:", "identical" if same else "DIFFERS")
        ok &= same
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
