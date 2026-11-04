#!/bin/zsh
# the 27B's 8-row tree verify on the tree path (default) against the zone path (NI_V8_TREE_G6=0), a 2-slot server (its
# default 8-row drafting block): a warm-up arm, then old, new, old, new
export AB_SW=NI_V8_TREE_G6 AB_SLOTS=2 AB_WARM=1
python3 <run-root>/ab_q.py <run-root>/bin/kapitsa-t6 ~/models/qwen3.8-27b-4bit-mtp ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/ab27 '[{}, {"NI_V8_TREE_G6":"0"}, {}, {"NI_V8_TREE_G6":"0"}, {}]'
date
