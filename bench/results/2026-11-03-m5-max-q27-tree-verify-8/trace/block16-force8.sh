#!/bin/zsh
# the 27B at ~131K: the 16-row drafting block with every cycle verifying 8 rows (NI_KFLASH_FORCE=8), on the zone_g6 build
M=~/models/qwen3.8-27b-4bit-mtp
NI_LOCK_HELD=1 NI_DFLASH_BLOCK=16 NI_KFLASH_FORCE=8 NI_CYCLE_TRACE=1 <run-root>/bin/ni_run-z6 $M @<run-root>/ids27_133k.txt 256 0 --dflash auto --ignore-eos 2>&1 | grep -E "prompt|KFlash: [0-9]+ cycles|cycle trace|drafting pass" | cut -c1-400
date
