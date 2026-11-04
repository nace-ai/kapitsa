#!/bin/zsh
# DFlash2's context commit on the tensor tile (qmt_m64_n64) against main: ni_run's 4K prompt with the model's DFlash2
# drafter (the "prompt N tok in X ms" line: the prefill and the drafter's commit), then 64 greedy tokens; rounds
# interleave old new / new old, 45 s idle before every run, each arm's own build (old: main d9e5430 in <repo>/build,
# new: the branch in <branch>/build). Then logit_dump old -> ref, new --from ref, logit_compare.py.
S=${S:?}; R=<repo>; N=$S/kapitsa-dq
ab() {  # label model rounds dflash
  local lab=$1 m=$2 rounds=$3 df=$4 i=0
  for rd in $(seq 1 $rounds); do
    if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
    for a in $arms; do
      i=$((i+1)); sleep 45
      B=$R/build/ni_run; [ $a = new ] && B=$N/build/ni_run
      lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $B ~/models/$m @$S/pf35/ids4k.txt 64 0 --dflash $df --ignore-eos \
        > $S/dq/ab/$lab-$i-$a.out 2>&1
      echo "$lab round $rd $a $(pmset -g batt | head -1 | grep -o "'.*'") load $(sysctl -n vm.loadavg | awk '{print $2}') | $(grep -o '^prompt [0-9]* tok in [0-9]* ms' $S/dq/ab/$lab-$i-$a.out) | $(grep -o '[0-9.]* tok/cycle' $S/dq/ab/$lab-$i-$a.out | head -1) | ids $(grep '^ids' $S/dq/ab/$lab-$i-$a.out | md5)"
    done
  done
}
ab q35 qwen3.6-35b-a3b-4bit 6 auto
ab clover clover-35b-a3b-4bit 2 ~/models/qwen3.6-35b-a3b-4bit/dflash2
ab q4b qwen3.5-4b-4bit-mtp 2 auto
ab q9b qwen3.5-9b-4bit 2 auto
ab q27 qwen3.8-27b-4bit-mtp 3 auto
oc() {  # model
  local m=$1
  (cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/dq/oc/$m-ref > $S/dq/oc/$m-ref.out 2>&1)
  (cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/dq/oc/$m-new --from $S/dq/oc/$m-ref > $S/dq/oc/$m-new.out 2>&1)
  python3 $N/tools/logit_compare.py $S/dq/oc/$m-ref $S/dq/oc/$m-new > $S/dq/oc/$m-compare.out 2>&1
  echo "== output check $m: exit $? $(grep -c 'every row bitwise identical' $S/dq/oc/$m-compare.out) paths identical"
}
oc qwen3.6-35b-a3b-4bit
oc qwen3.8-27b-4bit-mtp
echo done
