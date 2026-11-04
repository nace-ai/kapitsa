#!/bin/zsh
# the A/B again with this session idle, each run after a quiet GPU (no foreign process above 5% for 15 s, waiting up
# to 900 s): two batches, run under caffeinate -is lockf -k -t 3600 /tmp/kapitsa-gpu.lock (their folders were named
# hc and hc2, here batch1 and batch2). The build: run1's, of fn-hc-split before its squash merge (#921, e87d0f8)
cd <scratch>/idle
AB_HARNESS=<repo>/bench/harness python3 ab_fn.py <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt batch1 3 > batch1.out 2>&1
AB_HARNESS=<repo>/bench/harness python3 ab_fn.py <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt batch2 2 > batch2.out 2>&1
