#!/bin/bash
# k363: TATTN (qattn_t_sparse_g12d2, 4,096 B static threadgroup memory) under Metal shader validation, 10,240-token
# prompt (chunks 5 of 5 pass 8,192 keys, so the sparse path runs), fast profile, 8 tokens of decode. Untimed.
# Run 1: MTL_SHADER_VALIDATION=1 alone (out-of-bounds and nil-resource checks). Run 2: + MTL_DEBUG_LAYER=1, expected to
# stop on a dense prefill kernel over 16,384 B (AGENTS.md), which is recorded, not retried.
# It ran under `lockf -k /tmp/kapitsa-gpu.lock` (k363 resume8.sh). Run directly, it takes the lock itself first.
[ -n "$NI_LOCK_HELD" ] || exec lockf -k "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" env NI_LOCK_HELD=1 bash "$0" "$@"
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 NI_EXPERT_STORE=24 KAPITSA_PROFILE=fast
B=$HOME/Documents/nace/kapitsa/.worktrees/k363-rebase/build/ni_run; M=$HOME/Documents/nace/kapitsa-m5pro/models/flash-next-q20
O=${K363_OUT:?}; K=$HOME/Documents/nace/k363
guarded(){ local out=$1; shift; "$@" > $out 2>&1 & lp=$!; t=0
  while kill -0 $lp 2>/dev/null; do
    sw=$(sysctl -n vm.swapusage | awk '{v=$6; sub("M","",v); print int(v)}'); dk=$(df -g / | awk 'NR==2{print $4}')
    if [ "$sw" -gt 8192 ] || [ "$dk" -lt 40 ] || [ $t -ge 1800 ]; then echo "GUARD swap ${sw}M disk ${dk}G t ${t}s" >> $out; kill $lp; fi
    sleep 5; t=$((t+5)); done; wait $lp; }
echo "$(date +%T) run1 shader validation"; guarded $O/sv.out env MTL_SHADER_VALIDATION=1 $B $M @$K/fn-10K.ids 8 0; echo "rc=$?"
grep -m3 -iE "tattn|sparse|validation|error|fault" $O/sv.out
echo "$(date +%T) run2 shader validation + debug layer"; guarded $O/svdl.out env MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 $B $M @$K/fn-10K.ids 8 0; echo "rc=$?"
grep -m3 -iE "threadgroupMemoryLength|tattn|error|fault" $O/svdl.out
