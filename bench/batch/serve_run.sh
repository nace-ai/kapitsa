#!/bin/bash
# usage: serve_run.sh <kapitsa binary> <model dir> <log file> -- <client command> [-- <client command> ...]
# Starts `kapitsa serve` on $PORT (default 8131) with --ctx ${CTX:-32768} --slots ${SLOTS:-2} (extra server flags in
# $SERVE_ARGS, NI_* from the environment), waits until it is ready, runs each client command (the string BASE becomes
# the server's /v1 URL), stops the server and prints the engine's request lines. Run it under the GPU lock:
#   lockf -k -t 3000 /tmp/kapitsa-gpu.lock perl -e 'alarm 1200; exec @ARGV' bench/batch/serve_run.sh ...
set -u
BIN="$1"; MODEL="$2"; LOG="$3"; shift 3
PORT=${PORT:-8131}
"$BIN" serve --model "$MODEL" --model-id x --port $PORT --ctx ${CTX:-32768} --slots ${SLOTS:-2} ${SERVE_ARGS:-} >"$LOG" 2>&1 &
PID=$!
cleanup() { kill $PID 2>/dev/null; wait $PID 2>/dev/null; }
trap 'cleanup; exit 124' ALRM TERM INT
for i in $(seq 1 1200); do
  grep -q "^Ready" "$LOG" && break
  if ! kill -0 $PID 2>/dev/null; then echo "server died"; tail -20 "$LOG"; exit 1; fi
  sleep 0.25
done
grep -E "engine ready|memory:|DFlash2|LiLiCorr" "$LOG" | cut -c1-220
# the Neural Engine prefill split compiles in the background after start (the 27B and Bolt): its program must be
# ready before the clients run, or prefills before and after it differ (GPU alone vs split numerics)
if grep -q "compiling the Neural Engine program in the background" "$LOG"; then
  for i in $(seq 1 600); do
    grep -qE "Neural Engine program ready|NI_ANE: .*(failed|off)" "$LOG" && break
    sleep 0.5
  done
  grep -E "Neural Engine program ready|NI_ANE: .*(failed|off)" "$LOG" | cut -c1-160
fi
RC=0
cmd=()
run() {
  [ ${#cmd[@]} -eq 0 ] && return
  echo "== ${cmd[*]}"
  "${cmd[@]//BASE/http://127.0.0.1:$PORT/v1}" || RC=$?
  cmd=()
}
for a in "$@"; do
  if [ "$a" = "--" ]; then run; else cmd+=("$a"); fi
done
run
cleanup
grep -E "chat:|memory:|fatal|GPU error" "$LOG" | cut -c1-400 | tail -${TAIL:-40}
exit $RC
