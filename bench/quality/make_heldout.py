#!/usr/bin/env python3
"""Builds the held-out token sequences for quality measurements (tools/ni_eval, eval_mlx.py, eval_stream.py).

  make_heldout.py <model dir with tokenizer.json (+ chat template)> [out dir]

Four sequences of 1024 tokens from fixed texts in src/ (Kapitsa's own code and docs, and a coding-agent chat
written for this set): code_cpp (C++), code_py (Python), prose (English technical prose), chat (a multi-turn
agent conversation with tool calls, rendered with the model's chat template). The Qwen3.5 / 3.6 / 3.8 models share
one tokenizer; the script prints a hash of the ids so runs can be matched. Needs `tokenizers` and `jinja2`.
"""
import hashlib, json, os, sys

from jinja2 import Environment
from tokenizers import Tokenizer

N = 1024
model, out = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)))
here = os.path.dirname(os.path.abspath(__file__))
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))


def render_chat(msgs):
    p = os.path.join(model, "chat_template.jinja")
    tpl = open(p).read() if os.path.exists(p) else json.load(open(os.path.join(model, "tokenizer_config.json")))["chat_template"]
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    return env.from_string(tpl).render(messages=msgs, add_generation_prompt=False, enable_thinking=False)


texts = {
    "code_cpp": open(os.path.join(here, "src/code_cpp.txt")).read(),
    "code_py": open(os.path.join(here, "src/code_py.txt")).read(),
    "prose": open(os.path.join(here, "src/prose.txt")).read(),
    "chat": render_chat(json.load(open(os.path.join(here, "src/chat.json")))["messages"]),
}
h = hashlib.sha256()
for name, text in texts.items():
    ids = tok.encode(text, add_special_tokens=False).ids
    if len(ids) < N:
        sys.exit(f"{name}: only {len(ids)} tokens")
    ids = ids[:N]
    h.update(",".join(map(str, ids)).encode())
    open(os.path.join(out, f"{name}.txt"), "w").write(",".join(map(str, ids)))
    print(f"{name}: {len(ids)} tokens")
print("ids sha256", h.hexdigest()[:16])
