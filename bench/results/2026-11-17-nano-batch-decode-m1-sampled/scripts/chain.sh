#!/bin/zsh
# chain.sh MODEL TAG WHAT: WHAT = ab (3 interleaved pairs, each under one GPU lock, then exact.sh and compare.sh,
# each under the lock) or sampled (sampled.sh under the lock). Waits for both builds first.
set -u
MODEL=$1; TAG=$2; WHAT=$3
D=${0:A:h}
LOCK=(lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1)
until [ -x $D/bin-main/kapitsa -a -x $D/bin-pr/kapitsa -a -x $D/bin-pr/logit_dump ]; do sleep 20; done
echo "$(date) start $WHAT"
if [ $WHAT = ab ]; then
  for i in 1 2 3; do
    if (( i % 2 )); then o=(main pr); else o=(pr main); fi
    $LOCK zsh -c "$D/arm.sh $D/bin-${o[1]}/kapitsa $MODEL $D/out/$TAG/pair$i-${o[1]}; $D/arm.sh $D/bin-${o[2]}/kapitsa $MODEL $D/out/$TAG/pair$i-${o[2]}"
    echo "$(date) pair $i done"
  done
  $LOCK $D/exact.sh $MODEL $TAG
  echo "$(date) exact done"
  $LOCK $D/compare.sh $TAG
else
  $LOCK $D/sampled.sh $MODEL $TAG
fi
echo "$(date) CHAINDONE"
