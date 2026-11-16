#!/bin/zsh
# rest.sh MODEL TAG FIRST LAST: (run under one GPU lock) pairs FIRST..LAST of the interleaved A/B (odd: main first),
# then exact.sh, then (PROBE=1) probe9.sh
set -u
MODEL=$1; TAG=$2; F=$3; L=$4
D=${0:A:h}
for i in $(seq $F $L); do
  if (( i % 2 )); then o=(main ${PR:-pr}); else o=(${PR:-pr} main); fi
  $D/arm.sh $D/bin-${o[1]}/kapitsa $MODEL $D/out/$TAG/pair$i-${o[1]}
  $D/arm.sh $D/bin-${o[2]}/kapitsa $MODEL $D/out/$TAG/pair$i-${o[2]}
done
echo ALLDONE
$D/exact.sh $MODEL $TAG
[ "${PROBE:-0}" = 1 ] && $D/probe9.sh
echo CHAINDONE
