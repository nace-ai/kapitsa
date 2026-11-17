#!/bin/zsh
# arm.sh BIN MODEL OUT: one arm of the interleaved A/B (run under the GPU lock), as 2026-11-16-nano-batch-decode's
# arm.sh, with the context and slots from CTX and SLOTS (default 16384 and 4): kapitsa serve, then concurrency.py
# streams at 1, 2 and 4 requests at once (512-token greedy code replies) and single (the same 3 prompts one at a time,
# decode tok/s); vm_stat swaps and the busiest other process before and after.
set -u
BIN=$1; MODEL=$2; OUT=$3
D=${0:A:h}
PORT=${PORT:-8141}
mkdir -p $OUT
snap() { vm_stat | egrep "Swapins|Swapouts" | tr -s ' ' | tr '\n' ' '; echo; ps -Ao pcpu,comm -r | sed -n 2,3p | tr -s ' '; }
echo "before: $(snap)" > $OUT/env.txt
$BIN --version >> $OUT/env.txt
$BIN serve --model $MODEL --model-id x --port $PORT --ctx ${CTX:-16384} --slots ${SLOTS:-4} --disk-cache 0 ${=SERVE_ARGS:-} > $OUT/server.log 2>&1 &
PID=$!
for i in {1..2400}; do grep -q "^Ready" $OUT/server.log && break; kill -0 $PID 2>/dev/null || { echo "server died"; tail $OUT/server.log; exit 1; }; sleep 0.25; done
B=http://127.0.0.1:$PORT/v1
for n in ${=NS:-1 2 4}; do
  python3 $D/bench/batch/concurrency.py streams --base $B --n $n --tokens ${TOKENS:-512} > $OUT/streams$n.json 2>> $OUT/client.err
done
python3 $D/bench/batch/concurrency.py single --base $B --reps 3 --tokens ${TOKENS:-512} --sizes "" --prefill-reps 0 > $OUT/single.json 2>> $OUT/client.err
kill $PID; wait $PID 2>/dev/null
echo "after: $(snap)" >> $OUT/env.txt
cat $OUT/streams*.json $OUT/single.json | cut -c1-300
