#!/bin/bash
# #350 after generation (no lock itself): waits for job k350-ft-j1 to end, then capture, training chunks and the timed
# eval, each phase under its own `lockf -k /tmp/kapitsa-gpu.lock` so other tasks' legs interleave between them.
S=~/Documents/nace/kapitsa/.worktrees/k350-ft/bench/k350-ft; D=/tmp/k350d; DT=~/.agents/skills/session-orchestration/scripts/detach.py
L="lockf -k /tmp/kapitsa-gpu.lock"
while python3 $DT status k350-ft-j1 | grep -q running; do sleep 30; done
echo "j1 ended $(date +%T): $(python3 $DT status k350-ft-j1 | head -1)"
[ -s $D/data/capt.real.list ] || { echo "no capt.real.list"; exit 2; }
if [ ! -f $D/capt.done ]; then $L bash $S/capt_inner.sh && touch $D/capt.done || { echo "capture failed"; exit 3; }; fi
for c in 1 2 3 4; do
  [ -f $D/ckpt/DONE ] && break
  $L bash $S/train_chunk.sh || { echo "train chunk failed"; exit 4; }
done
[ -f $D/ckpt/DONE ] || { echo "training not done"; exit 4; }
CK=$D/ckpt/best; [ -r $CK/model.safetensors ] || CK=$D/ckpt/last
echo "drafter for eval: $CK"; bash $S/mkmodels.sh $CK || exit 5
$L bash $S/eval_inner.sh; rc=$?
python3 $S/verdict.py $D/eval; echo "PIPELINE DONE rc=$rc $(date +%T)"
