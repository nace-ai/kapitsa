#!/bin/bash
# #343: Kapitsa plain greedy decode on Flint 4-bit for the MLX comparison: prompt 893 tokens (the MLX leg's prompt length), 128 steps, 3 reps.
# Merge build of #922 (f9e85aa7). The prompt is corpus text of the same length, not the same text. Caller holds the GPU lock.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
until pmset -g batt | head -1 | grep -q "AC Power"; do echo "waiting for AC"; sleep 30; done
mkdir -p ~/Documents/nace/k343/out/kapitsa-flint; cd ~/Documents/nace/k343/wt-f9e85aa7
for r in 1 2; do
  echo "=== kapitsa flint run $r $(date +%T) load $(sysctl -n vm.loadavg)"
  build/flint_bench ~/Documents/nace/kapitsa-m5pro/models/flint-4bit --prompt 893 --steps 128 --reps 3 > ~/Documents/nace/k343/out/kapitsa-flint/run$r.out 2>&1; echo rc=$?
  rg "^rep" ~/Documents/nace/k343/out/kapitsa-flint/run$r.out | cut -c1-200
done
echo KFLINTDONE
