#!/bin/zsh
# exact.sh MODEL TAG: (under one GPU lock) logit_dump main and pr; greedy texts alone on main; on pr alone and
# together (pipelined plain rows), then with NI_TRACE_TOKENS alone and together (margins)
set -u
MODEL=$1; TAG=$2
D=${0:A:h}; O=$D/out/$TAG/exact; mkdir -p $O
PORT=8142; B=http://127.0.0.1:$PORT/v1
serve() {  # BIN LOG [env...]
  local bin=$1 log=$2; shift 2
  env "$@" $bin serve --model $MODEL --model-id x --port $PORT --ctx 16384 --slots 4 --disk-cache 0 > $log 2>&1 &
  SP=$!
  for i in {1..2400}; do grep -q "^Ready" $log && break; sleep 0.25; done
}
$D/bin-main/logit_dump $MODEL $O/ld-main --paths step,verify,batch > $O/ld-main.log 2>&1
$D/bin-${PR:-pr}/logit_dump $MODEL $O/ld-pr --paths step,verify,batch --from $O/ld-main > $O/ld-pr.log 2>&1
$D/bin-${PR:-pr}/logit_dump $MODEL $O/ld-pr-b2 --paths batch --batch 2 --batch-rows 1 --from $O/ld-main > $O/ld-pr-b2.log 2>&1
serve $D/bin-main/kapitsa $O/main-alone.log
python3 $D/gtexts.py $B alone $O/main-alone.json 8 300 > $O/main-alone.out 2>&1
kill $SP; wait $SP 2>/dev/null
serve $D/bin-${PR:-pr}/kapitsa $O/pr.log
python3 $D/gtexts.py $B alone $O/pr-alone.json 8 300 > $O/pr-alone.out 2>&1
python3 $D/gtexts.py $B together $O/pr-together.json 8 300 > $O/pr-together.out 2>&1
python3 $D/gtexts.py $B together $O/pr-together2.json 2 300 > $O/pr-together2.out 2>&1
kill $SP; wait $SP 2>/dev/null
rm -f $O/trace-alone.jsonl $O/trace-together.jsonl
serve $D/bin-${PR:-pr}/kapitsa $O/pr-trace-alone.log NI_TRACE_TOKENS=$O/trace-alone.jsonl
python3 $D/bench/batch/concurrency.py greedy --base $B --mode alone --n 8 --tokens 300 > $O/trace-alone.out 2>&1
kill $SP; wait $SP 2>/dev/null
serve $D/bin-${PR:-pr}/kapitsa $O/pr-trace-together.log NI_TRACE_TOKENS=$O/trace-together.jsonl
python3 $D/bench/batch/concurrency.py greedy --base $B --mode together --n 8 --tokens 300 > $O/trace-together.out 2>&1
kill $SP; wait $SP 2>/dev/null
echo EXACTDONE
