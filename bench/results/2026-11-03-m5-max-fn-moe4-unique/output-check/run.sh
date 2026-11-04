#!/bin/zsh
# the output check: logit_dump of main d93297d (old, <ref>) and of this branch (new, <repo>), step and verify paths
<ref>/build/logit_dump ~/models/qwen3.8-flash-next-4bit <scratch>/muab/oc/old --paths step,verify > dump-old.out 2>&1
<repo>/build/logit_dump ~/models/qwen3.8-flash-next-4bit <scratch>/muab/oc/new --paths step,verify > dump-new.out 2>&1
for p in prefill step verify; do python3 tools/logit_compare.py <scratch>/muab/oc/old/$p <scratch>/muab/oc/new/$p; done > compare.out 2>&1
