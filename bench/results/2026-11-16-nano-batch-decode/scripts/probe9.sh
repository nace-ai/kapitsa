#!/bin/zsh
# probe9.sh: (one GPU lock) the Nano 9B v2 with --spec 4 (prompt lookup, so the branch batches it): batched (default)
# against NI_BATCH=0 (taking turns), streams of 256 tokens at 1, 2, 4; order batched, turns, turns, batched
set -u
D=${0:A:h}; M=~/kapitsa-run/home/models/nemotron-nano-9b-v2-nvfp4
export SERVE_ARGS="--spec 4" TOKENS=256
$D/arm.sh $D/bin-pr/kapitsa $M $D/out/nano9b-probe/r1-batched
NI_BATCH=0 $D/arm.sh $D/bin-pr/kapitsa $M $D/out/nano9b-probe/r1-turns
NI_BATCH=0 $D/arm.sh $D/bin-pr/kapitsa $M $D/out/nano9b-probe/r2-turns
$D/arm.sh $D/bin-pr/kapitsa $M $D/out/nano9b-probe/r2-batched
echo PROBEDONE
