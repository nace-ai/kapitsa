#!/bin/zsh
# the 27B's decode cycle at ~131K tokens: NI_CYCLE_TRACE (where a cycle's wall time goes), then NI_PROFILE (each kernel's
# GPU time, isolated dispatches) over a few cycles
M=~/models/qwen3.8-27b-4bit-mtp
echo "== cycle trace" && date
NI_LOCK_HELD=1 NI_CYCLE_TRACE=1 <run-root>/bin/ni_run-qs1 $M @<run-root>/ids27_133k.txt 256 0 --dflash auto --ignore-eos 2>&1 | grep -E "prompt|KFlash: [0-9]+ cycles|cycle trace|drafting pass" | cut -c1-400
echo "== profile" && date
NI_LOCK_HELD=1 NI_PROFILE=1 <run-root>/bin/ni_run-qs1 $M @<run-root>/ids27_133k.txt 48 0 --dflash auto --ignore-eos 2>&1 | grep -v "^ids:" | tail -45
date
