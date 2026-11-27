#!/bin/bash
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
W=~/Documents/nace/kapitsa-wt325
M=~/Documents/nace/kapitsa-m5pro/models/flash-next-q20
P=$W/bench/results/2026-10-24-m5-pro-flash-next-ssd/prompts
NI_EXPERT_STORE=24 NI_PROFILE=2 $W/build/ni_run $M @/Users/alisheramantay/Documents/nace/k325/c128k.ids 96 0 --spec 2 --ignore-eos > ~/Documents/nace/k325/prof128.out 2> ~/Documents/nace/k325/prof128.err
echo rc=$?
