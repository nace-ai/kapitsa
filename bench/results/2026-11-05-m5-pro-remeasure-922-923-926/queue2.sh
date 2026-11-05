#!/bin/bash
# #343 one lockf job (orchestrator, 5 Nov 01:50): #923 down ROUNDS=2, then #926 EXL3, then MLX Flint; quiet waits capped, rows labelled
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
K=~/Documents/nace/k343; cd $K
ROUNDS=2 bash inner.sh 923-down-R3 923-down-R4
bash exl3_inner.sh
bash mlx_flint.sh
echo QUEUE2DONE $(date +%T)
