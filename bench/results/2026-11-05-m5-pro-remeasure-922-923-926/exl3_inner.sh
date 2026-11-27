#!/bin/bash
# #343: #926 on this M5 Pro, the PR's own real-weight A/B (original vs register kernels in one executable), 2 timed runs.
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
W=~/Documents/nace/k343/wt-af7d7958; E=$W/bench/results/2026-11-04-m4pro-exl3-registers; X="/Volumes/Extreme Pro/downloads/k343-exl3-L0"
until pmset -g batt | head -1 | grep -q "AC Power"; do echo "waiting for AC"; sleep 30; done
bash ~/.agents/skills/session-orchestration/scripts/hide_dsh.sh on
trap 'bash ~/.agents/skills/session-orchestration/scripts/hide_dsh.sh off' EXIT
cd $W
for r in 1 2; do
  echo "=== realbench $r $(date +%T)"
  python3 $E/run343.py realbench --reference ~/Documents/nace/k343/exl3_reference.metal --weights "$X" --out ~/Documents/nace/k343/out/exl3-realbench-$r; echo rc=$?
done
echo EXL3DONE
