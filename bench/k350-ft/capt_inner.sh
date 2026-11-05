#!/bin/bash
# #350 GPU job 2 (caller holds /tmp/kapitsa-gpu.lock): the drafter's context rows of every answered conversation
# (k350_capt, 4-bit drafter as deployed), from <data>/capt.real.list written by render.py.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
W=~/Documents/nace/kapitsa/.worktrees/k350-ft; D=/tmp/k350d; M=~/Documents/nace/models/qwen3.6-35b-a3b-4bit
[ -s $D/data/capt.real.list ] && [ -d "/Volumes/Extreme Pro/downloads" ] || { echo PREFLIGHT FAIL; exit 9; }
mkdir -p $D/feat
echo "capt start $(date +%T) $(pmset -g batt | head -1) $(wc -l < $D/data/capt.real.list) sequences"
timeout 3000 $W/build/k350_capt $M $D/data/capt.real.list 2>&1 | grep -v "^\[kapitsa\]" | tail -8
rc=${PIPESTATUS[0]}; echo "capt rc=$rc $(date +%T) $(pmset -g batt | head -1)"; du -sh $D/feat; exit $rc
