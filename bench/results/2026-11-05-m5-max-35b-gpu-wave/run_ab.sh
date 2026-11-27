#!/bin/zsh
# The 35B-A3B's GPU wavefront (NI_PF_WAVE) with 8,192-row chunks against main (4,096-row chunks, no wavefront):
# ni_run's prefill of 4,129 ids (the harness's 4K length) and 16,288 ids, each arm in the chunks its server picks
# (old: 4,129 ids in one 4,224-row chunk, 16,288 in 4,096-row chunks; new: 8,192-row chunks), --pf-reps 2 (rep 0 cold),
# rounds alternating old new / new old, 45 s idle before each run, each arm's own Release build. Then the output check:
# logit_dump old -> ref (its default chunks), new --chunk 8192 --from ref, logit_compare.py. Under caffeinate -is, on
# AC power, the session idle.
S=${S:?}; R=$S/kapitsa-cells3; N=$S/kapitsa-wv; D=$N/bench/results/2026-11-05-m5-max-35b-gpu-wave
m=qwen3.6-35b-a3b-4bit
run() {  # label prompt rounds old-chunk
  local lab=$1 ids=$2 rounds=$3 oc=$4 i=0
  for rd in $(seq 1 $rounds); do
    if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
    for a in $arms; do
      i=$((i+1)); sleep 45
      if [ $a = new ]; then B=$N/build/ni_run; c=8192; else B=$R/build/ni_run; c=$oc; fi
      lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NI_CHUNK=$c $B ~/models/$m @$ids 8 0 --pf-reps 2 --dflash none \
        > $D/ab/$lab-$i-$a.out 2>&1
      echo "$lab round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") reps $(grep -E 'prefill rep [01]:' $D/ab/$lab-$i-$a.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ') $(grep '^ids' $D/ab/$lab-$i-$a.out)"
    done
  done
}
run 4k $D/prompt-ids-4129.txt 3 4224
run 16k $D/prompt-ids-16288.txt 3 4096
(cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/wv-oc/$m-ref > $D/output-check/$m-ref.out 2>&1)
(cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/wv-oc/$m-new --from $S/wv-oc/$m-ref --chunk 8192 > $D/output-check/$m-new.out 2>&1)
echo "== output check $m"
python3 $N/tools/logit_compare.py $S/wv-oc/$m-ref $S/wv-oc/$m-new > $D/output-check/$m-compare.out 2>&1; echo "compare exit $?"
grep -E "^[a-z]+:|KL|PASS|FAIL|bitwise|reference:|candidate:" $D/output-check/$m-compare.out
echo done
