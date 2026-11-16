#!/bin/zsh
# pairs.sh MODEL TAG N: N interleaved pairs (main, pr / pr, main alternating), each pair under one GPU lock
set -u
MODEL=$1; TAG=$2; N=$3
D=${0:A:h}
for i in $(seq 1 $N); do
  if (( i % 2 )); then o=(main pr); else o=(pr main); fi
  lockf -k /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 zsh -c "
    $D/arm.sh $D/bin-${o[1]}/kapitsa $MODEL $D/out/$TAG/pair$i-${o[1]};
    $D/arm.sh $D/bin-${o[2]}/kapitsa $MODEL $D/out/$TAG/pair$i-${o[2]}"
done
echo ALLDONE
