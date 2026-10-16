#!/bin/bash
# usage: serve_ttft.sh <server_bin> <model_dir> <log_file> <client args...>
# Starts the server as the benchmark harness does (--ctx 32768 --slots 1, port $PORT or 8097),
# waits for /health, runs cached_ttft.py with the given arguments, stops the server and prints the
# engine's per-request prefill lines (NI_LOG=1). Run it under the GPU lock. Environment (NI_*) is
# passed to the server.
set -u
BIN="$1"; MODEL="$2"; LOG="$3"; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-8097}
NI_LOG=${NI_LOG:-1} "$BIN" --model "$MODEL" --model-id x --port $PORT --ctx 32768 --slots 1 >"$LOG" 2>&1 &
PID=$!
CPID=
cleanup() { [ -n "$CPID" ] && kill $CPID 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null; }
trap 'cleanup; exit 124' ALRM TERM INT
for i in $(seq 1 600); do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  sleep 0.25
done
python3 "$HERE/cached_ttft.py" --base "http://127.0.0.1:$PORT/v1" "$@" &
CPID=$!
wait $CPID
RC=$?
kill $PID 2>/dev/null
wait $PID 2>/dev/null
grep -E "chat:|fatal|NI_" "$LOG"
exit $RC
