#!/usr/bin/env python3
# ruff: noqa: F821  (MODEL, Server, REPLY, post come from the exec of the header of tests/test_splash_engine.py below)
"""Kapitsa's tokenizer, loaded from Splash's resolved tokenizer.json inside `kapitsa serve --engine splash`, against the
`tokenizers` library on the same file: the prompt ids Kapitsa sends (logged by tests/fake_splash.py) must re-encode to
themselves from their decoded text, and each user text's ids (stripped: the template trims) must sit inside the prompt.
usage: python3 tools/splash_tokenizer_check.py   (from the repo root, build/kapitsa built; needs `pip install tokenizers`)"""
import json, os, sys, glob
sys.argv=[sys.argv[0],"build/kapitsa"]
src=open("tests/test_splash_engine.py").read().split("MSG = [")[0]
__file__=os.path.abspath("tests/test_splash_engine.py")
exec(src)
from tokenizers import Tokenizer
ref=Tokenizer.from_file(MODEL+"/tokenizer/tokenizer.json")
texts=["hello","Привет, мир! 你好 🙂 naïve café","def f(x):\n    return x**2  # tab\there\n\n\n  indented","  leading spaces and trailing  \n","12345 3.14159 0xFF https://a.b/c?d=e&f=g","<tool_call> literal <think> in user text","a"*500+"\r\n"+"b\u00a0c\u2003d"]
s=Server("--reply",str(REPLY[0]))
bad=0
try:
    for t in texts:
        post(s.base,{"messages":[{"role":"system","content":"You are terse."},{"role":"user","content":t}],"max_tokens":1})
    prompts=s.prompts()
    assert len(prompts)==len(texts),"%d prompts logged for %d texts"%(len(prompts),len(texts))
    for t,p in zip(texts,prompts):
        text=ref.decode(p["prompt"],skip_special_tokens=False)
        again=ref.encode(text,add_special_tokens=False).ids
        ok= again==p["prompt"]
        # also: reference encodes the user text part identically
        part=ref.encode(t.strip(),add_special_tokens=False).ids
        inside=any(p["prompt"][i:i+len(part)]==part for i in range(len(p["prompt"])-len(part)+1))
        print(ok,inside,len(p["prompt"]),repr(t[:30]))
        bad+= (not ok) or (not inside)
finally: s.close()
print("mismatches",bad)
sys.exit(1 if bad else 0)
