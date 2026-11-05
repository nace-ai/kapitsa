#!/bin/bash
# One GPU leg of coding10: for each model, start kapitsa serve, run the given arms, stop the server.
# Run under the lock:  detach.py run --name k350-X -- lockf -k /tmp/kapitsa-gpu.lock bash run_gpu.sh OUTDIR "4b 9b" arms/a.json ...
set -u
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; MODELS=$2; shift 2
mkdir -p "$OUT"
KAP=${KAPITSA:-/opt/homebrew/bin/kapitsa}
PORT=8350
bash ~/.agents/skills/session-orchestration/scripts/hide_dsh.sh on || true
trap 'kill $SP 2>/dev/null; bash ~/.agents/skills/session-orchestration/scripts/hide_dsh.sh off || true' EXIT
for M in $MODELS; do
  echo "== $M $(date) load=$(sysctl -n vm.loadavg) $(pmset -g batt | head -1)"
  SARGV="$KAP serve --model $M --port $PORT --private --context 32k"
  $SARGV > "$OUT/serve-$M.log" 2>&1 & SP=$!
  for i in $(seq 1 300); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 2; done
  CODING10_SERVER_ARGV="$SARGV" CODING10_KAPITSA_VERSION="$($KAP --version)" \
    timeout 3600 python3 "$HERE/coding10.py" --url http://127.0.0.1:$PORT --label "$M" --out "$OUT/$M.json" --reps "${REPS:-1}" --require-ac --arms "$@"
  rc=$?
  echo "== $M harness rc=$rc"
  kill $SP; wait $SP 2>/dev/null; sleep 3
done
