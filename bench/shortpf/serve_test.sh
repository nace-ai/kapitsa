#!/bin/bash
# usage: serve_test.sh <server_bin> <model_dir> <log_file>
# Starts the server with its default slots (2, the side-slot check needs a second one) on $PORT or
# 8097, runs tests/test_server.py against it and stops the server. Run it under the GPU lock.
set -u
BIN="$1"; MODEL="$2"; LOG="$3"
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-8097}
NI_LOG=${NI_LOG:-1} "$BIN" --model "$MODEL" --model-id x --port $PORT --ctx 32768 >"$LOG" 2>&1 &
PID=$!
CPID=
cleanup() { [ -n "$CPID" ] && kill $CPID 2>/dev/null; kill $PID 2>/dev/null; wait $PID 2>/dev/null; }
trap 'cleanup; exit 124' ALRM TERM INT
for i in $(seq 1 600); do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  sleep 0.25
done
python3 "$HERE/../../tests/test_server.py" "http://127.0.0.1:$PORT" &
CPID=$!
wait $CPID
RC=$?
kill $PID 2>/dev/null
wait $PID 2>/dev/null
grep -E "chat:|fatal" "$LOG"
exit $RC
