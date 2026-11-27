#!/bin/zsh
# The 128-row tile's whole-simdgroup outputs through the cooperative tensors' store, against main (the 2026-11-05 cells'
# build 446165e): ni_run's 4K prefill (bench/quality/ids_qwen38's four files, 4,096 ids), 3 reps, interleaved
# old new / new old per round with 45 s idle before each run, each arm's own build (old: main code, the cells build 446165e in
# <base>/build, new: this branch in <branch>/build). Then the output check: logit_dump old -> ref, new --from ref,
# logit_compare.py. Under caffeinate -is, on AC power, the session idle.
S=${S:?}; R=<base>; N=<branch>
OLD=$R/build/ni_run; NEW=$N/build/ni_run
for m in qwen3.6-35b-a3b-4bit qwen3.8-27b-4bit-mtp; do
i=0
for rd in 1 2 3 4; do
  if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
  for a in $arms; do
    i=$((i+1)); sleep 45
    B=$OLD; [ $a = new ] && B=$NEW
    lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $B ~/models/$m @$S/pf35/ids4k.txt 8 0 --pf-reps 3 --dflash none \
      --dump-logits $S/st/ab/$m-$i-$a.bin > $S/st/ab/$m-$i-$a.out 2>&1
    echo "$m round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") units $(grep -o '[0-9]*/34 units' $S/st/ab/$m-$i-$a.out) reps $(grep -E 'prefill rep [012]:' $S/st/ab/$m-$i-$a.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ')"
  done
done
done
for m in qwen3.6-35b-a3b-4bit qwen3.8-27b-4bit-mtp; do
(cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/st/oc/$m-ref > $S/st/oc/$m-ref.out 2>&1)
(cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/st/oc/$m-new --from $S/st/oc/$m-ref > $S/st/oc/$m-new.out 2>&1)
echo "== output check $m"
python3 $N/tools/logit_compare.py $S/st/oc/$m-ref $S/st/oc/$m-new > $S/st/oc/$m-compare.out 2>&1; echo "compare exit $?"
tail -12 $S/st/oc/$m-compare.out
done
echo done
