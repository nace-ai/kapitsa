#!/bin/zsh
# the 27B's drafting block, 8 rows (tree path, #916) against 16 rows (wide verify), at ~8K / 32K / 64K tokens: ni_run, 256
# tokens at temperature 0, NI_CYCLE_TRACE=1
M=~/models/qwen3.8-27b-4bit-mtp
for n in 34000 135000 270000; do
  for b in 16 8; do
    echo "== $n chars, block $b" && date
    NI_LOCK_HELD=1 NI_DFLASH_BLOCK=$b NI_CYCLE_TRACE=1 <run-root>/bin/ni_run-t6 $M @<run-root>/ids27_$n.txt 256 0 --dflash auto --ignore-eos 2>&1 | grep -E "^prompt|cycle trace" | cut -c1-260
  done
done
date
