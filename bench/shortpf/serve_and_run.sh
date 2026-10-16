#!/bin/bash
# usage: serve_and_run.sh <server_bin> <model_dir> <log_file> <client args...>
# Starts the server (1 slot, ctx 32768, port 8097), waits for /health, runs ttft.py, stops the server.
# Run under the GPU lock. Extra env (NI_SHORTPF etc.) is inherited by the server.
set -u
BIN="$1"; MODEL="$2"; LOG="$3"; shift 3
HERE="$(cd "$(dirname "$0")" && pwd)"
PORT=${PORT:-8097}
NI_LOG=${NI_LOG:-1} "$BIN" --model "$MODEL" --model-id x --port $PORT --ctx 32768 --slots 1 >"$LOG" 2>&1 &
PID=$!
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then break; fi
  if ! kill -0 $PID 2>/dev/null; then echo "server died"; cat "$LOG"; exit 1; fi
  sleep 0.5
done
python3 "$HERE/ttft.py" "http://127.0.0.1:$PORT" "$@"
RC=$?
kill $PID 2>/dev/null
wait $PID 2>/dev/null
grep "chat:" "$LOG"
exit $RC
