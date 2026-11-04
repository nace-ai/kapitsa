#!/bin/zsh
# the output check: logit_dump of this build with NI_HC_WIDE=0 (old) and the default (new), step and verify paths
NI_HC_WIDE=0 build/logit_dump ~/models/qwen3.8-flash-next-4bit <scratch>/fnab/oc/old --paths step,verify > dump-old.out 2>&1
build/logit_dump ~/models/qwen3.8-flash-next-4bit <scratch>/fnab/oc/new --paths step,verify > dump-new.out 2>&1
for p in prefill step verify; do python3 tools/logit_compare.py <scratch>/fnab/oc/old/$p <scratch>/fnab/oc/new/$p; done > compare.out 2>&1
