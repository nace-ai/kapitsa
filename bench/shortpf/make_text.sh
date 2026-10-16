#!/bin/bash
# Decodes bench/prompts/p4000.txt and p16000.txt into the plain texts ttft.py sends (needs the
# `tokenizers` package; default: the mlx-lm research venv).
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
TOK="${TOK:-$HOME/.cache/nace-infer-models/qwen3.8-27b-4bit-mtp/tokenizer.json}"
for n in 4000 16000; do
  "$PY" -I -c "
import sys
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1])
ids = [int(x) for x in open(sys.argv[2]).read().strip().split(',')]
open(sys.argv[3], 'w').write(t.decode(ids))
" "$TOK" "$HERE/../prompts/p$n.txt" "$HERE/p${n}_text.txt"
done
