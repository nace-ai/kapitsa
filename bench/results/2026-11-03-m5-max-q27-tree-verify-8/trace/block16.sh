#!/bin/zsh
# the 27B at ~131K tokens with the server's 16-row drafting block (wide 16-row tree verifies), cycle trace
M=~/models/qwen3.8-27b-4bit-mtp
NI_LOCK_HELD=1 NI_DFLASH_BLOCK=16 NI_CYCLE_TRACE=1 <run-root>/bin/ni_run-qs1 $M @<run-root>/ids27_133k.txt 256 0 --dflash auto --ignore-eos 2>&1 | grep -E "prompt|KFlash: [0-9]+ cycles|cycle trace|drafting pass|16-row" | cut -c1-400
date
