#!/bin/bash
# One GPU leg of coding10: for each model, start kapitsa serve, run the given arms, stop the server.
# Usage:  REPS=3 bash run_gpu.sh OUTDIR "4b 9b" arms/a.json ...   (takes the GPU lock itself)
set -u
# take the GPU lock unless the caller holds it, then tell kapitsa it is held
[ -n "${NI_LOCK_HELD:-}" ] || exec lockf -k "${BENCH_GPU_LOCK:-/tmp/kapitsa-gpu.lock}" env NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 bash "$0" "$@"
export NACE_GPU_LOCK_HELD=1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; MODELS=$2; shift 2
mkdir -p "$OUT"
KAP=${KAPITSA:-/opt/homebrew/bin/kapitsa}
PORT=8350
SP=
trap '[ -n "$SP" ] && kill $SP 2>/dev/null' EXIT
for M in $MODELS; do
  echo "== $M $(date) load=$(sysctl -n vm.loadavg) $(pmset -g batt | head -1)"
  SARGV="$KAP serve --model $M --port $PORT --private --context 32k"
  $SARGV > "$OUT/serve-$M.log" 2>&1 & SP=$!
  if curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null; then echo "port $PORT already serves something; stop it first" >&2; exit 1; fi
  ready=0
  for i in $(seq 1 300); do
    kill -0 $SP 2>/dev/null || { echo "kapitsa serve exited, see $OUT/serve-$M.log" >&2; exit 1; }
    curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && { ready=1; break; }; sleep 2
  done
  [ $ready = 1 ] || { echo "kapitsa serve not ready after 600 s" >&2; exit 1; }
  CODING10_SERVER_ARGV="$SARGV" CODING10_KAPITSA_VERSION="$($KAP --version)" \
    timeout 3600 python3 "$HERE/coding10.py" --url http://127.0.0.1:$PORT --label "$M" --out "$OUT/$M.json" --reps "${REPS:-1}" --require-ac --arms "$@"
  rc=$?
  echo "== $M harness rc=$rc"
  kill $SP; wait $SP 2>/dev/null; sleep 3
done
