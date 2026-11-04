#!/bin/zsh
# #923's A/B again (NI_MOE4_DX=0 against the default, the build of 83ad33a run_lc.sh made), 4 rounds: dx_fn/'s waits all
# ran out with the Claude app's helper near 3% of the GPU. Under caffeinate -is lockf -k /tmp/kapitsa-gpu.lock, idle.
set -e
export NI_LOCK_HELD=1
S=${S:?}
cd $S/lc
AB_HARNESS=$S/kapitsa-md/bench/harness python3 ab_lc.py dx_fn.json dx_fn2 4 > dx_fn2.out 2>&1
echo done
