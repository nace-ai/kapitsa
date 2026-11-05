#!/bin/zsh
# The 27B's Neural Engine share retuned for the 128-row tile (default_units: 16 -> 14 of 34 units at 40 cores) against
# the NI_QMT128 branch (16): ni_run's 4K prefill (bench/quality/ids_qwen38's four files, 4,096 ids), 3 reps, interleaved
# old new / new old per round with 45 s idle before each run, each arm's own build (old: the qmt-m128-stage branch in
# <base>/build, new: this branch in <branch>/build). Then the output check: logit_dump old -> ref, new --from ref,
# logit_compare.py. Under caffeinate -is, on AC power, the session idle.
S=${S:?}; R=<base>; N=<branch>
OLD=$R/build/ni_run; NEW=$N/build/ni_run
m=qwen3.8-27b-4bit-mtp
i=0
for rd in 1 2 3 4; do
  if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
  for a in $arms; do
    i=$((i+1)); sleep 45
    B=$OLD; [ $a = new ] && B=$NEW
    lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $B ~/models/$m @$S/pf35/ids4k.txt 8 0 --pf-reps 3 --dflash none \
      --dump-logits $S/ane/ab/$m-$i-$a.bin > $S/ane/ab/$m-$i-$a.out 2>&1
    echo "$m round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") units $(grep -o '[0-9]*/34 units' $S/ane/ab/$m-$i-$a.out) reps $(grep -E 'prefill rep [012]:' $S/ane/ab/$m-$i-$a.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ')"
  done
done
(cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/ane/oc/$m-ref > $S/ane/oc/$m-ref.out 2>&1)
(cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/ane/oc/$m-new --from $S/ane/oc/$m-ref > $S/ane/oc/$m-new.out 2>&1)
echo "== output check $m"
python3 $N/tools/logit_compare.py $S/ane/oc/$m-ref $S/ane/oc/$m-new > $S/ane/oc/$m-compare.out 2>&1; echo "compare exit $?"
tail -12 $S/ane/oc/$m-compare.out
echo done
