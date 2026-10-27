#!/bin/bash
# Driver: L smoke, then three interleaved passes K S N L (ABAB-style), 2 repetitions per cell per leg (n = 6), each leg under the GPU lock.
D=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$D/legs}
B="python3 $D/scripts/flash_next_bench.py"
mkdir -p "$OUT"
leg() { # idx arm [extra]
  echo "=== leg $1 $2 $(date -u +%FT%TZ)"
  nice -n 5 lockf -k -t 120 /tmp/kapitsa-gpu.lock $B leg --arm $2 --idx $1 --out "$OUT" --reps 2 $3 > "$OUT/leg$1-$2.out" 2>&1
  echo "leg $1 $2 rc=$? $(date -u +%FT%TZ)"; tail -3 "$OUT/leg$1-$2.out"
}
leg 0 L --smoke
i=1
for pass in 1 2 3; do for arm in K S N L; do
  [ $arm = L ] && [ -n "$NO_L" ] && continue
  leg $i $arm; i=$((i+1))
done; done
$B summarize "$OUT"
echo "=== all done $(date -u +%FT%TZ)"
