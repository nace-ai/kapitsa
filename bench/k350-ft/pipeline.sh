#!/bin/bash
# #350 after generation (no lock itself): waits for job k350-ft-j1 to end, then capture, training chunks and the timed
# eval, each phase under its own `lockf -k /tmp/kapitsa-gpu.lock` so other tasks' legs interleave between them.
S=~/Documents/nace/kapitsa/.worktrees/k350-ft/bench/k350-ft; D=/tmp/k350d; DT=~/.agents/skills/session-orchestration/scripts/detach.py
L="lockf -k /tmp/kapitsa-gpu.lock"
wait_ac() { until pmset -g batt | grep -q "AC Power"; do echo "waiting for AC $(date +%T) $(pmset -g batt | grep -o "[0-9]*%")"; sleep 60; done; }  # outside the lock
# orchestrator 6 Nov: before training or eval takes the lock, free swap >= 3 GB and memory pressure normal (level 1)
# orchestrator 7 Nov: used swap stays high for hours after pressure ends (#373's pressure test), so gate on what is free now:
# pressure normal (level 1), free RAM >= 12 GB (vm_stat free+inactive+speculative+purgeable) and no other process over 20 GB RSS
# the data, features and checkpoints live on the Extreme Pro (/tmp/k350d links there); wait outside the lock until it is
# mounted with >= 26 GB free (orchestrator 7 Nov: leave 20 GB; a state save writes a 5.5 GB .tmp before the rename)
drive_ok() { [ -d $D/data ] && dfree=$(df -g "$D/" | awk 'NR==2{print $4}') && [ "$dfree" -ge "${1:-26}" ]; }  # eval saves no state: 20 GB
wait_drive() { until drive_ok "$1"; do echo "waiting for the Extreme Pro $(date +%T): ${dfree:-not mounted} GB free"; dfree=; sleep 120; done; }
# orchestrator 7 Nov: don't train between #373's legs (its pressure test pushes swap past 8 GB within 25 steps);
# wait outside the lock until no detach job whose name contains 373 is running
wait_373() { while python3 $DT list 2>/dev/null | grep -E "^RUNNING +[^ ]*373" >/dev/null; do echo "waiting for #373 $(date +%T): $(python3 $DT list | awk '/^RUNNING +[^ ]*373/{printf "%s ", $2}')"; sleep 120; done; }
mem_ok() { lv=$(sysctl -n kern.memorystatus_vm_pressure_level)
  fr=$(vm_stat | awk '/page size of/{ps=$8} /Pages (free|inactive|speculative|purgeable):/{gsub("\\.","",$NF); n+=$NF} END{printf "%.1f", n*ps/2^30}')
  big=$(ps -axo rss=,pid=,comm= | awk '$1>20*2^20{printf "%s(%.0fGB) ", $3, $1/2^20}')
  [ "$lv" = 1 ] && awk -v f="$fr" 'BEGIN{exit !(f>=12)}' && [ -z "$big" ]; }
wait_mem() { until mem_ok; do echo "waiting for memory $(date +%T) pressure level $lv, free ${fr} GB, big: ${big:-none}"; sleep 60; done; }
while python3 $DT status k350-ft-j1 | head -1 | grep -q "k350-ft-j1 running"; do sleep 30; done
echo "j1 ended $(date +%T): $(python3 $DT status k350-ft-j1 | head -1)"
wait_drive 20; [ -s $D/data/capt.real.list ] || { echo "no capt.real.list"; exit 2; }
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
for c in $(seq 1 40); do  # a chunk ends at a 30 GB footprint after ~225 steps (~4 min of the 80): ~20 chunks
  [ -f $D/ckpt/DONE ] && break
  wait_drive; wait_ac; wait_373; wait_mem; $L bash $S/train_chunk.sh; rc=$?
  [ $rc = 75 ] && { echo "chunk stopped on swap > 8 GB, state saved; gate again"; continue; }
  [ $rc = 0 ] || { echo "train chunk failed"; exit 4; }
done
[ -f $D/ckpt/DONE ] || { echo "training not done"; exit 4; }
CK=$D/ckpt/best; [ -r $CK/model.safetensors ] || CK=$D/ckpt/last
echo "drafter for eval: $CK"; bash $S/mkmodels.sh $CK || exit 5
wait_drive 20; wait_ac; wait_373; wait_mem; $L bash $S/eval_inner.sh; rc=$?
python3 $S/verdict.py $D/eval; echo "PIPELINE DONE rc=$rc $(date +%T)"
