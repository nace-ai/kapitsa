#!/bin/zsh
# The output check that exercises the change: logit_dump with 48-row prefill chunks (every prefill pass then runs
# 48-row GEMMs: the short-prefill tiles on main, the 64-row tensor tile on the branch), main d9d479d -> ref, the
# branch's final form --from ref, logit_compare.py with the 27B's M5 floor. Then one confirmation round of the final
# form on 33 and 4,129 ids (45 s idle before each). Under caffeinate -is, on AC power, the session idle.
S=${S:?}; R=$S/kapitsa-m3; N=$S/kapitsa-sq; D=$N/bench/results/2026-11-05-m5-max-qmt-short
m=qwen3.8-27b-4bit-mtp
(cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/qs-oc48/$m-ref --chunk 48 > $D/output-check-48/$m-ref.out 2>&1)
(cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/qs-oc48/$m-new --from $S/qs-oc48/$m-ref --chunk 48 > $D/output-check-48/$m-new.out 2>&1)
python3 $N/tools/logit_compare.py $S/qs-oc48/$m-ref $S/qs-oc48/$m-new > $D/output-check-48/$m-compare.out 2>&1
for lab in p33:prompt-ids-33.txt:4 4k:prompt-ids-4129.txt:2; do
  l=${lab%%:*}; r=${lab##*:}; f=${${lab#*:}%:*}; sleep 45
  lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $N/build/ni_run ~/models/$m @$D/$f 4 0 --pf-reps $r --dflash none \
    > $D/final/$l-new.out 2>&1
done
