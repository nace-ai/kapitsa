#!/bin/zsh
# #922 and #923's A/Bs again with the load average and the power source logged on every leg (ab_lc.py), and #922 on
# Flint: its verify pass of 8 rows (flint_bench) and the output check (logit_dump step and verify, both builds). Run
# under caffeinate -is lockf -k -t 14400 /tmp/kapitsa-gpu.lock, the session idle. S: the scratch folder.
set -e
export NI_LOCK_HELD=1
S=${S:?}
H=$S/kapitsa-mu/bench/harness
cd $S/lc
git -C $S/kapitsa-ref checkout -q --detach 8715427
for w in kapitsa-ref kapitsa-mu; do cmake --build $S/$w/build -j --target ni_run logit_dump flint_bench > /dev/null; done
cmake --build $S/kapitsa-md/build -j --target ni_run > /dev/null
for w in kapitsa-ref kapitsa-mu kapitsa-md; do
  echo "$w $(git -C $S/$w rev-parse --short HEAD) $(git -C $S/$w status --short --untracked-files=no | wc -l | tr -d ' ') modified"
done > builds.txt
AB_HARNESS=$H python3 ab_lc.py mu_fn.json mu_fn 4 > mu_fn.out 2>&1
AB_HARNESS=$H python3 ab_lc.py dx_fn.json dx_fn 4 > dx_fn.out 2>&1
AB_HARNESS=$H python3 ab_lc.py mu_flint.json mu_flint 3 > mu_flint.out 2>&1
mkdir -p oc
$S/kapitsa-ref/build/logit_dump ~/models/flint-4bit $S/lc/oc/old --paths step,verify > oc/dump-old.out 2>&1
$S/kapitsa-mu/build/logit_dump ~/models/flint-4bit $S/lc/oc/new --paths step,verify > oc/dump-new.out 2>&1
for p in prefill step verify; do python3 $S/kapitsa-mu/tools/logit_compare.py oc/old/$p oc/new/$p; done > oc/compare.out 2>&1
echo done
