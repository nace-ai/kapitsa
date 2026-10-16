#!/usr/bin/env python3
"""Regenerates the token-id prompt files used by ni_run A/B runs (comma-separated ids).

  p700 / p4000 / p16000  first N ids of a fixed corpus (the first 40 mlx_lm model sources, sorted)
  code_35b / code_27b    the harness's decode prompt (bench.py t_decode, fixed run tag), chat
                         template applied with enable_thinking=False and the generation prompt

usage: make_prompts.py <python site-packages with mlx_lm> <35B model dir> <27B model dir> [out dir]
Needs `tokenizers` and `jinja2` (both in the mlx-lm venv).
"""
import glob, json, os, sys

from jinja2 import Environment
from tokenizers import Tokenizer

site, m35, m27 = sys.argv[1:4]
out = sys.argv[4] if len(sys.argv) > 4 else os.path.dirname(os.path.abspath(__file__))

MSGS = [{"role": "system", "content": "You are a helpful coding assistant. Answer with code only."},
        {"role": "user", "content": "[run 0.123456] Write a complete, well-commented Python module that implements an LRU "
                                    "cache with TTL expiry, a thread-safe wrapper, and a pytest test suite. Write at least "
                                    "600 lines. Do not stop early."}]


def chat_ids(model_dir):
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    tpl_path = os.path.join(model_dir, "chat_template.jinja")
    tpl = open(tpl_path).read() if os.path.exists(tpl_path) else json.load(
        open(os.path.join(model_dir, "tokenizer_config.json")))["chat_template"]
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    text = env.from_string(tpl).render(messages=MSGS, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False).ids


def main():
    tok = Tokenizer.from_file(os.path.join(m35, "tokenizer.json"))
    src = "".join(open(f).read() for f in sorted(glob.glob(os.path.join(site, "mlx_lm/models/*.py")))[:40])
    ids = tok.encode(src).ids
    assert len(ids) >= 16000, len(ids)
    for n in (700, 4000, 16000):
        open(os.path.join(out, f"p{n}.txt"), "w").write(",".join(map(str, ids[:n])))
    for name, d in (("code_35b", m35), ("code_27b", m27)):
        c = chat_ids(d)
        open(os.path.join(out, f"{name}.txt"), "w").write(",".join(map(str, c)))
        print(name, len(c))
    print("corpus ids", len(ids))


if __name__ == "__main__":
    main()
