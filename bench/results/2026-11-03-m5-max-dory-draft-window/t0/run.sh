#!/bin/zsh
caffeinate -is lockf -k -t 14400 /tmp/kapitsa-gpu.lock python3 <run-root>/ab_dw.py <run-root>/bin/kapitsa-dw1 ~/models/nemotron-labs-dory1.0-4b-nvfp4 ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/run1 '[{}, {"NI_DORY_DRAFT_WIN":"4096"}, {"NI_DORY_DRAFT_WIN":"1024"}, {}, {"NI_DORY_DRAFT_WIN":"4096"}, {"NI_DORY_DRAFT_WIN":"1024"}]'
