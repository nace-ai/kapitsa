#!/bin/zsh
# The 128-row qmt tile (NI_QMT128 default) against main: ni_run's 4K prefill (bench/quality/ids_qwen38's four files,
# 4,096 ids), 3 reps (rep 0 dropped), interleaved old new / new old per round with 45 s idle before each run, each
# arm's own build (old: main fa0bc00 in <repo>/build, new: the branch in <branch>/build). Then the output check:
# logit_dump old -> ref, new --from ref, logit_compare.py. Under caffeinate -is, on AC power, the session idle.
S=${S:?}; R=<repo>; N=<branch>
OLD=$R/build/ni_run; NEW=$N/build/ni_run
ab() {  # model rounds
  local m=$1 rounds=$2 i=0
  for rd in $(seq 1 $rounds); do
    if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
    for a in $arms; do
      i=$((i+1)); sleep 45
      B=$OLD; [ $a = new ] && B=$NEW
      lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $B ~/models/$m @$S/pf35/ids4k.txt 8 0 --pf-reps 3 --dflash none \
        --dump-logits $S/q128/ab/$m-$i-$a.bin > $S/q128/ab/$m-$i-$a.out 2>&1
      echo "$m round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") load $(sysctl -n vm.loadavg | awk '{print $2}') reps $(grep -E 'prefill rep [12]:' $S/q128/ab/$m-$i-$a.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ')"
    done
  done
}
ab qwen3.6-35b-a3b-4bit 6
ab qwen3.8-27b-4bit-mtp 4
ab clover-35b-a3b-4bit 2
oc() {  # model paths
  local m=$1 paths=$2
  (cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/q128/oc/$m-ref ${=paths} > $S/q128/oc/$m-ref.out 2>&1)
  (cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/q128/oc/$m-new --from $S/q128/oc/$m-ref ${=paths} > $S/q128/oc/$m-new.out 2>&1)
  echo "== output check $m"
  python3 $N/tools/logit_compare.py $S/q128/oc/$m-ref $S/q128/oc/$m-new > $S/q128/oc/$m-compare.out 2>&1; echo "compare exit $?"
  tail -5 $S/q128/oc/$m-compare.out
}
oc qwen3.6-35b-a3b-4bit ""
oc qwen3.8-27b-4bit-mtp ""
oc clover-35b-a3b-4bit ""
echo done
