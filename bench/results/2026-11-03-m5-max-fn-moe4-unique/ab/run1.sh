#!/bin/zsh
# the A/B: main d93297d (old) against this branch (new), 3 rounds, under the GPU lock
cd <scratch>/muab
AB_HARNESS=<repo>/bench/harness python3 ab_mu.py <ref>/build/ni_run <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt run1 3
