#!/bin/bash
# rv_llama.sh <gguf> <ids file> <ntok> <port> <mode>: one llama.cpp request on the same token ids as Kapitsa's ni_run.
#   mode: gpu (-ngl 99) | cpumoe (-ngl 99 --cpu-moe: routed experts on the CPU, the usual small-Mac setup)
# llama-server is this script's direct child (PID recorded); the trap TERMs it (10 s grace, then KILL) on any exit.
# Prints the server's timings JSON line ("TIMINGS {...}") and the generated token ids ("TOKENS ...").
set -u
G=$1; IDS=$2; NT=$3; PORT=$4; MODE=$5
SRV=${RV_LLAMA_SERVER:-/tmp/llama-main/build/bin/llama-server}
X=""; [ "$MODE" = cpumoe ] && X="--cpu-moe"
$SRV -m "$G" -ngl 99 $X -c ${RV_LLAMA_CTX:-8192} -fa on --port $PORT --host 127.0.0.1 -np 1 --no-webui --metrics 2>&1 & S=$!
echo "llama-server pid $S lstart $(ps -o lstart= -p $S | xargs) mode $MODE"
stop() { kill -0 $S 2>/dev/null || return 0; kill -TERM $S; for i in $(seq 1 20); do kill -0 $S 2>/dev/null || return 0; sleep 0.5; done; kill -9 $S; }
trap stop EXIT
trap 'stop; exit 143' INT TERM
t0=$(date +%s)
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do
  kill -0 $S 2>/dev/null || { echo "llama-server exited during load"; wait $S; exit 3; }
  sleep 1
done
echo "LOAD $(( $(date +%s) - t0 )) s"
BODY=$(python3 -c "
import json,re,sys
ids=[int(x) for x in re.findall(r'\d+', open('$IDS').read())]
print(json.dumps({'prompt': ids, 'n_predict': $NT, 'temperature': 0, 'top_k': 1, 'ignore_eos': True,
                  'cache_prompt': False, 'return_tokens': True, 'stream': False}))")
R=$(curl -s -X POST http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d "$BODY")
python3 -c "
import json,sys
r=json.loads(sys.argv[1]); t=r.get('timings',{})
print('TIMINGS', json.dumps(t))
print('TOKENS', ' '.join(map(str, r.get('tokens', []))))
" "$R" || { echo "bad response: ${R:0:300}"; exit 4; }
