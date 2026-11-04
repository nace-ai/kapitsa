#!/bin/zsh
# #923's A/B again, 6 more rounds: dx_fn2/ left one round clean on every check (quiet, 0.0% foreign, load, AC).
# The build of 83ad33a run_lc.sh made. Under caffeinate -is lockf -k /tmp/kapitsa-gpu.lock, the session idle.
set -e
export NI_LOCK_HELD=1
S=${S:?}
cd $S/lc
AB_HARNESS=$S/kapitsa-md/bench/harness python3 ab_lc.py dx_fn.json dx_fn3 6 > dx_fn3.out 2>&1
echo done
