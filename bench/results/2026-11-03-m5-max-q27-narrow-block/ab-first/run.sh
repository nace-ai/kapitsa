#!/bin/zsh
# the 27B, one slot (its 16-row drafting block): 8-row blocks from 16K-token prompts (default) against 16 rows always
# (NI_DFLASH_NARROW=0), interleaved after a warm-up arm; a last arm loads the drafter with 8 rows (NI_DFLASH_BLOCK=8), whose
# text on the long prompts the default's must match
export AB_SW=NI_DFLASH_NARROW AB_SLOTS=1 AB_WARM=1 AB_DOCS='{"8K": 34000, "32K": 135000, "131K": 548000}'
python3 <run-root>/ab_q.py <run-root>/bin/kapitsa-nb1 ~/models/qwen3.8-27b-4bit-mtp ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/abnb '[{}, {"NI_DFLASH_NARROW":"0"}, {}, {"NI_DFLASH_NARROW":"0"}, {}, {"NI_DFLASH_BLOCK":"8"}]'
date
