#!/bin/zsh
# The dense 4-bit GEMMs of 17-63 rows on the tensor tile (NI_QMT_MIN 17 on the M5 Max) against main d9d479d (64):
# the 27B's prefill of 33 and 48 ids (one short chunk) and of 4,129 ids (the harness's 4K prompt length: two
# 2048-row chunks and a 33-row tail), interleaved old new / new old per round, each arm's own Release build, idle
# before each run (20 s for the short prompts, 45 s for 4K). Then the output check: logit_dump old -> ref, new
# --from ref, logit_compare.py, on the 27B and the 35B-A3B. Under caffeinate -is, on AC power, the session idle.
S=${S:?}; R=$S/kapitsa-m3; N=$S/kapitsa-sq; D=$N/bench/results/2026-11-05-m5-max-qmt-short
m=qwen3.8-27b-4bit-mtp
run() {  # label prompt rounds reps idle
  local lab=$1 ids=$2 rounds=$3 reps=$4 idle=$5 i=0
  for rd in $(seq 1 $rounds); do
    if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
    for a in $arms; do
      i=$((i+1)); sleep $idle
      if [ $a = new ]; then B=$N/build/ni_run; else B=$R/build/ni_run; fi
      lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $B ~/models/$m @$ids 4 0 --pf-reps $reps --dflash none \
        > $D/ab/$lab-$i-$a.out 2>&1
      echo "$lab round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") reps $(grep -E 'prefill rep [0-9]:' $D/ab/$lab-$i-$a.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ')"
    done
  done
}
run p33 $D/prompt-ids-33.txt 3 4 20
run p48 $D/prompt-ids-48.txt 3 4 20
run 4k $D/prompt-ids-4129.txt 4 2 45
for mm in $m qwen3.6-35b-a3b-4bit; do
  (cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$mm $S/qs-oc/$mm-ref > $D/output-check-default-chunks/$mm-ref.out 2>&1)
  (cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$mm $S/qs-oc/$mm-new --from $S/qs-oc/$mm-ref > $D/output-check-default-chunks/$mm-new.out 2>&1)
  echo "== output check $mm"
  python3 $N/tools/logit_compare.py $S/qs-oc/$mm-ref $S/qs-oc/$mm-new > $D/output-check-default-chunks/$mm-compare.out 2>&1; echo "compare exit $?"
  grep -E "^[a-z]+:|KL|PASS|FAIL|bitwise" $D/output-check-default-chunks/$mm-compare.out
done
echo done
