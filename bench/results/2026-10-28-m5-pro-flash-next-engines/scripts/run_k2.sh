#!/bin/bash
# Second Kapitsa build (ff6151c8, splash-6-plan) interleaved with plain Splash: K2 S2 K2 S2 K2 S2, 2 reps each (n=6).
D=$(cd "$(dirname "$0")/.." && pwd); OUT=$D/legs; B="python3 $D/scripts/flash_next_bench.py"
KB=${KB:-$HOME/Documents/nace/kapitsa-splash-ff/build/kapitsa}
leg() { # idx arm group
  [ -f "$OUT/leg$1-$2.json" ] && return
  echo "=== leg $1 $2/$3 $(date -u +%FT%TZ)"
  for try in $(seq 10); do
    nice -n 5 lockf -k -t 120 "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" $B leg --arm $2 --group $3 --kbin "$KB" --idx $1 --out "$OUT" --reps 2 > "$OUT/leg$1-$2.out" 2>&1
    grep -q 'already running, not starting' "$OUT/leg$1-$2.out" || break; echo "  busy (try $try)"; sleep 30
  done
  echo "leg $1 $2 rc=$? $(date -u +%FT%TZ)"; tail -2 "$OUT/leg$1-$2.out"
}
leg 14 K K2; leg 15 S S2; leg 16 K K2; leg 17 S S2; leg 18 K K2; leg 19 S S2
python3 $D/scripts/flash_next_bench.py summarize "$OUT"
echo "=== all done $(date -u +%FT%TZ)"
