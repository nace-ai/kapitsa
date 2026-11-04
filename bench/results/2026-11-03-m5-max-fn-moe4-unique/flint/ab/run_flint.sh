#!/bin/zsh
# #922 on Flint, again: run_lc.sh started flint_bench in this folder, where bench/corpus.txt is missing, and its prompt
# loop never ended (killed after 2 h). Each arm now runs from its checkout (mu_flint.json's "cwd"), as does each dump.
# The builds run_lc.sh made, unchanged. Under caffeinate -is lockf -k /tmp/kapitsa-gpu.lock, the session idle.
set -e
export NI_LOCK_HELD=1
S=${S:?}
cd $S/lc
AB_HARNESS=$S/kapitsa-mu/bench/harness python3 ab_lc.py mu_flint.json mu_flint 3 > mu_flint.out 2>&1
mkdir -p oc
(cd $S/kapitsa-ref && build/logit_dump ~/models/flint-4bit $S/lc/oc/old --paths step,verify) > oc/dump-old.out 2>&1
(cd $S/kapitsa-mu && build/logit_dump ~/models/flint-4bit $S/lc/oc/new --paths step,verify) > oc/dump-new.out 2>&1
for p in prefill step verify; do python3 $S/kapitsa-mu/tools/logit_compare.py oc/old/$p oc/new/$p; done > oc/compare.out 2>&1
echo done
