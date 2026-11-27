#!/bin/bash
# #343 timed legs. Run as: lockf -k /tmp/kapitsa-gpu.lock bash inner.sh (via detach.py)
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
K=~/Documents/nace/k343; H=$K/wt-f9e85aa7/bench/harness; export AB_HARNESS=$H
bash ~/.agents/skills/session-orchestration/scripts/hide_dsh.sh on
trap 'bash ~/.agents/skills/session-orchestration/scripts/hide_dsh.sh off' EXIT
waitac() { until pmset -g batt | head -1 | grep -q "AC Power"; do echo "waiting for AC $(date +%T)"; sleep 30; done; }
cd $K; mkdir -p out
for c in "$@"; do
  waitac
  echo "=== $c $(date +%T)"
  python3 ab_lc.py cfg-$c.json out/$c ${ROUNDS:-4} > out/$c.log 2>&1; echo "rc=$? $c"; tail -5 out/$c.log | cut -c1-260
done
echo ALLDONE $(date +%T)
