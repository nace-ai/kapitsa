#!/bin/zsh
# The 27B's 4096-row prefill chunks (the wavefront's halves at 2048 rows) against main d9d479d (2048-row chunks):
# ni_run's 4K and 8K prefill, 2 reps (rep 0 cold), interleaved old new / new old per round with 45 s idle before each run,
# each arm's own build (old: main in <base>/build, NI_CHUNK unset; new: the branch in <branch>/build, NI_CHUNK=4096 as
# the engine now picks for the single-slot 27B on the 40-core M5). Then the output check: logit_dump old -> ref (its
# default 2048-row chunks), new --chunk 4096 --from ref, logit_compare.py. Under caffeinate -is, on AC power, idle.
S=${S:?}; R=<base>; N=<branch>
m=qwen3.8-27b-4bit-mtp
run() {  # label prompt rounds
  local lab=$1 ids=$2 rounds=$3 i=0
  for rd in $(seq 1 $rounds); do
    if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
    for a in $arms; do
      i=$((i+1)); sleep 45
      if [ $a = new ]; then B=$N/build/ni_run; E=(NI_CHUNK=4096); else B=$R/build/ni_run; E=(); fi
      lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $E $B ~/models/$m @$ids 8 0 --pf-reps 2 --dflash none \
        > $S/wr/ab/$lab-$i-$a.out 2>&1
      echo "$lab round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") reps $(grep -E 'prefill rep [01]:' $S/wr/ab/$lab-$i-$a.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ') $(grep -o '[0-9]* split chunks' $S/wr/ab/$lab-$i-$a.out | head -1)"
    done
  done
}
run 4k $S/pf35/ids4k.txt 4
run 8k $S/pf27/ids27_8k.txt 2
(cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/wr/oc/$m-ref > $S/wr/oc/$m-ref.out 2>&1)
(cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/wr/oc/$m-new --from $S/wr/oc/$m-ref --chunk 4096 > $S/wr/oc/$m-new.out 2>&1)
echo "== output check $m"
python3 $N/tools/logit_compare.py $S/wr/oc/$m-ref $S/wr/oc/$m-new > $S/wr/oc/$m-compare.out 2>&1; echo "compare exit $?"
grep -E "^[a-z]+:|KL|PASS|FAIL|bitwise" $S/wr/oc/$m-compare.out
echo done
