#!/bin/bash
# #325: native Flash-Next on the M5 Pro, old = 71826a86 (before #887/#890/#892), new = a2f42108; ABABAB per context.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 NI_EXPERT_STORE=24
N=~/Documents/nace/kapitsa-wt325/build/ni_run
O=~/Documents/nace/kapitsa-wt325-old/build/ni_run
M=~/Documents/nace/kapitsa-m5pro/models/flash-next-q20
P=~/Documents/nace/kapitsa-wt325/bench/results/2026-10-24-m5-pro-flash-next-ssd/prompts
K=~/Documents/nace/k325
OUT=$K/ab
for ctx in 4k 32k 128k; do
  case $ctx in 4k) ids=$P/pf4_seed11.ids;; 32k) ids=$P/c32k.ids;; 128k) ids=$K/c128k.ids;; esac
  for r in 1 2 3; do
    for arm in old new; do
      bin=$N; [ $arm = old ] && bin=$O
      f=$OUT/$ctx-$arm-$r
      echo "=== $ctx $arm $r $(date -u +%T)"
      $bin $M @$ids 192 0 --spec 2 --ignore-eos > $f.out 2> $f.err
      echo "rc=$? $(grep -E '^prompt' $f.err | cut -c1-200)"
      sleep 20
    done
  done
done
echo ALLDONE $(date -u +%T)
