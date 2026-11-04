#!/bin/zsh
# the A/B: one build (fn-hc-split), NI_HC_WIDE=0 against the default, 3 rounds, under the GPU lock
cd <scratch>/fnab
AB_HARNESS=<repo>/bench/harness python3 ab_fn.py <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt run1 3
