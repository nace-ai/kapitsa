#!/bin/bash
# Driver (2026-10-30 copy of 2026-10-28-m5-pro-flash-next-engines/scripts/run_all.sh; only the legs changed): K and S
# alternate, K S K S, 5 legs per arm, 2 repetitions per leg. Each leg is one server start under the GPU lock. Legs that
# already have a JSON are skipped.
D=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$D/legs}
B="python3 $D/scripts/flash_next_bench.py"
mkdir -p "$OUT"
leg() { # idx arm reps [extra]
  [ -f "$OUT/leg$1-$2.json" ] && return
  echo "=== leg $1 $2 $(date -u +%FT%TZ)"
  for try in 1 2 3 4 5 6 7 8 9 10; do  # another worker's short model start makes the leg refuse to start: wait, retry
    nice -n 5 lockf -k -t 120 "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" $B leg --arm $2 --idx $1 --out "$OUT" --reps $3 $4 > "$OUT/leg$1-$2.out" 2>&1
    rc=$?; grep -q 'already running, not starting' "$OUT/leg$1-$2.out" || break; echo "  busy (try $try), waiting"; sleep 30
  done
  echo "leg $1 $2 rc=$? $(date -u +%FT%TZ)"; tail -2 "$OUT/leg$1-$2.out"
}
leg 1 K 2; leg 2 S 2; leg 3 K 2; leg 4 S 2; leg 5 K 2
leg 6 S 2; leg 7 K 2; leg 8 S 2; leg 9 K 2; leg 10 S 2
$B summarize "$OUT"
echo "=== all done $(date -u +%FT%TZ)"
