#!/bin/zsh
# the 35B's decode A/B again: a first arm only fills the SSD cache with both documents' states, then old / new / old / new
# all restore them (the same prefix state), every arm's text kept
M=~/models/qwen3.6-35b-a3b-4bit
AB_WARM=1 python3 <run-root>/ab_q35.py <run-root>/bin/kapitsa-qs1 $M ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/ab2 '[{}, {"NI_V8T_STAGE":"0"}, {}, {"NI_V8T_STAGE":"0"}, {}]'
date
