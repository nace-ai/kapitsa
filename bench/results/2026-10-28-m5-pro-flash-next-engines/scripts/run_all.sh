#!/bin/bash
# Driver. Each leg is one server start under the GPU lock. Legs 0-3 ran first (L smoke, K, S, N; 2 reps each); the
# first L leg (leg 4) was killed by hand after 1K/4K prefill measured 40 tok/s (llama.cpp streams the 66 GB Q2_0 files
# from the SSD), so L runs with one repetition per leg from leg 7 on. Legs that already have a JSON are skipped.
D=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$D/legs}
B="python3 $D/scripts/flash_next_bench.py"
mkdir -p "$OUT"
leg() { # idx arm reps [extra]
  [ -f "$OUT/leg$1-$2.json" ] && return
  echo "=== leg $1 $2 $(date -u +%FT%TZ)"
  nice -n 5 lockf -k -t 120 /tmp/kapitsa-gpu.lock $B leg --arm $2 --idx $1 --out "$OUT" --reps $3 $4 > "$OUT/leg$1-$2.out" 2>&1
  echo "leg $1 $2 rc=$? $(date -u +%FT%TZ)"; tail -2 "$OUT/leg$1-$2.out"
}
leg 0 L 1 --smoke
leg 1 K 2; leg 2 S 2; leg 3 N 2
leg 4 K 2; leg 5 S 2; leg 6 N 2; leg 7 L 1
leg 8 K 2; leg 9 S 2; leg 10 N 2; leg 11 L 1
$B summarize "$OUT"
echo "=== all done $(date -u +%FT%TZ)"
