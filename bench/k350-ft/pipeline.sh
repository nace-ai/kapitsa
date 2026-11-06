#!/bin/bash
# #350 after generation (no lock itself): waits for job k350-ft-j1 to end, then capture, training chunks and the timed
# eval, each phase under its own `lockf -k /tmp/kapitsa-gpu.lock` so other tasks' legs interleave between them.
S=~/Documents/nace/kapitsa/.worktrees/k350-ft/bench/k350-ft; D=/tmp/k350d; DT=~/.agents/skills/session-orchestration/scripts/detach.py
L="lockf -k /tmp/kapitsa-gpu.lock"
while python3 $DT status k350-ft-j1 | head -1 | grep -q "k350-ft-j1 running"; do sleep 30; done
echo "j1 ended $(date +%T): $(python3 $DT status k350-ft-j1 | head -1)"
[ -s $D/data/capt.real.list ] || { echo "no capt.real.list"; exit 2; }
# orchestrator queue order (6 Nov): these jobs take the GPU first; wait until each reports "not running: finished"
GATE=${GATE:-"k276-ab k363-q1 swift8b ab362 k361-rerun"}
while :; do
  left=""; for n in $GATE; do python3 $DT status $n 2>/dev/null | head -1 | grep -q "not running: finished" || left="$left $n"; done
  [ -z "$left" ] && break
  echo "$(date +%T) waiting for:$left"; sleep 120
done
echo "gate clear $(date +%T)"
for c in $(seq 1 12); do  # capture in ~200-sequence chunks, each its own lockf process
  [ -f $D/capt.done ] && break
  $L bash $S/capt_inner.sh 2>&1 | tee $D/capt.last.log; rc=${PIPESTATUS[0]}
  [ $rc = 0 ] || { echo "capture failed rc=$rc"; exit 3; }
  grep -q "capture complete" $D/capt.last.log && touch $D/capt.done
done
[ -f $D/capt.done ] || { echo "capture not complete"; exit 3; }
for c in 1 2 3 4; do
  [ -f $D/ckpt/DONE ] && break
  $L bash $S/train_chunk.sh || { echo "train chunk failed"; exit 4; }
done
[ -f $D/ckpt/DONE ] || { echo "training not done"; exit 4; }
CK=$D/ckpt/best; [ -r $CK/model.safetensors ] || CK=$D/ckpt/last
echo "drafter for eval: $CK"; bash $S/mkmodels.sh $CK || exit 5
$L bash $S/eval_inner.sh; rc=$?
python3 $S/verdict.py $D/eval; echo "PIPELINE DONE rc=$rc $(date +%T)"
