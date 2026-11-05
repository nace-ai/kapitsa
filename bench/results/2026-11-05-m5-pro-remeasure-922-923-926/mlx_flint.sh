#!/bin/bash
# #343: mlx-vlm 0.7.11 greedy decode tok/s on Flint 4-bit (patched config in an APFS clone + U8->uint32 view in the wrapper), 3 runs. Caller holds the GPU lock.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
until pmset -g batt | head -1 | grep -q "AC Power"; do echo "waiting for AC"; sleep 30; done
mkdir -p ~/Documents/nace/k343/out/mlx; cd ~/Documents/nace/k343/out/mlx
for r in 1 2 3; do
  echo "=== mlx-vlm flint run $r $(date +%T) load $(sysctl -n vm.loadavg) ac=$(pmset -g batt | head -1 | cut -c1-30)"
  ~/Documents/nace/k325/mlxenv/bin/python ~/Documents/nace/k343/mlx_flint_run.py ~/Documents/nace/k343/flint-mlx ~/Documents/nace/k343/prompt4k.txt 128 > run$r.out 2>&1; rc=$?
  echo rc=$rc; rg "RESULT|Error" run$r.out | cut -c1-200
done
echo MLXDONE
