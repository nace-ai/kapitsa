#!/bin/zsh
# #922's Flash-Next A/B (main 8715427 against moe4-unique-par ea5308d), four more rounds: the first four (mu_fn/) left
# one round whose legs both ended below load 4. The builds run_lc.sh made, unchanged. Under caffeinate -is lockf -k
# /tmp/kapitsa-gpu.lock, the session idle.
set -e
export NI_LOCK_HELD=1
S=${S:?}
cd $S/lc
AB_HARNESS=$S/kapitsa-mu/bench/harness python3 ab_lc.py mu_fn.json mu_fn2 4 > mu_fn2.out 2>&1
echo done
