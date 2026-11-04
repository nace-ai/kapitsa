#!/bin/zsh
# the A/B again with this session idle, each run after a quiet GPU (no foreign process above 5% for 15 s, waiting up
# to 900 s): two batches, run under caffeinate -is lockf -k -t 3600 /tmp/kapitsa-gpu.lock (their folders were named
# mu and mu2, here batch1 and batch2). The builds: main d93297d (<ref>) and this branch (<repo>), as in run1
cd <scratch>/idle
AB_HARNESS=<repo>/bench/harness python3 ab_mu.py <ref>/build/ni_run <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt batch1 3 > batch1.out 2>&1
AB_HARNESS=<repo>/bench/harness python3 ab_mu.py <ref>/build/ni_run <repo>/build/ni_run ~/models/qwen3.8-flash-next-4bit prompt-ids.txt batch2 1 > batch2.out 2>&1
