#!/bin/zsh
# the output check on 83ad33a's code (the checkout at 4aca316: results and docs on top), from <repo>, under the GPU lock
# (run_after.sh's first step): logit_dump with NI_MOE4_DX=0 (old) and the default (new), step and verify paths
NI_MOE4_DX=0 build/logit_dump ~/models/qwen3.8-flash-next-4bit <scratch>/lc/ocdx/old --paths step,verify > dump-old.out 2>&1
build/logit_dump ~/models/qwen3.8-flash-next-4bit <scratch>/lc/ocdx/new --paths step,verify > dump-new.out 2>&1
for p in prefill step verify; do python3 tools/logit_compare.py <scratch>/lc/ocdx/old/$p <scratch>/lc/ocdx/new/$p; done > compare.out 2>&1
