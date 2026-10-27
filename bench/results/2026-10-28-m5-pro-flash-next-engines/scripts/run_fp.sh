#!/bin/bash
# Footprint pass: K (ff6151c8, group K2), S (group S2), N, one leg each per round, two rounds, 2 reps per leg.
# Same harness as run_all.sh/run_k2.sh plus peak phys_footprint of the process tree sampled about every 1 s.
D=$(cd "$(dirname "$0")/.." && pwd); OUT=$D/legs; B="python3 $D/scripts/flash_next_bench.py"
KB=${KB:-$HOME/Documents/nace/kapitsa-splash-ff/build/kapitsa}
leg() { # idx arm [extra args]
  idx=$1; arm=$2; shift 2
  [ -f "$OUT/leg$idx-$arm.json" ] && return
  echo "=== leg $idx $arm $(date -u +%FT%TZ)"
  for try in $(seq 10); do
    nice -n 5 lockf -k -t 120 "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" $B leg --arm $arm --idx $idx --out "$OUT" --reps 2 "$@" > "$OUT/leg$idx-$arm.out" 2>&1
    grep -q 'already running, not starting' "$OUT/leg$idx-$arm.out" || break; echo "  busy (try $try)"; sleep 30
  done
  echo "leg $idx $arm done $(date -u +%FT%TZ)"; tail -1 "$OUT/leg$idx-$arm.out"
}
leg 20 K --group K2 --kbin "$KB"; leg 21 S --group S2; leg 22 N
leg 23 S --group S2; leg 24 K --group K2 --kbin "$KB"; leg 25 N
python3 $D/scripts/flash_next_bench.py summarize "$OUT"
echo "=== all done $(date -u +%FT%TZ)"
