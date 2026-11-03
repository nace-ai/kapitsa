#!/bin/zsh
# #916 again, with the GPU load sampled: the 27B's 8-row tree verify on the tree path (default) against the zone path
# (NI_V8_TREE_G6=0), a 2-slot server, a warm-up arm, then old, new, old, new; each request after a quiet GPU
export AB_SW=NI_V8_TREE_G6 AB_SLOTS=2 AB_WARM=1 AB_HARNESS=<repo>/bench/harness
python3 <run-root>/ab_q.py <run-root>/bin/kapitsa-t6 ~/models/qwen3.8-27b-4bit-mtp ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/ab27b '[{}, {"NI_V8_TREE_G6":"0"}, {}, {"NI_V8_TREE_G6":"0"}, {}]'
date
