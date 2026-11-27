#!/bin/zsh
# The 35B-A3B's cell, main (87d7dd5's tree, built as 7afb51d) against this branch, interleaved old new / new old for two
# rounds, 60 s idle before each run, the session idle; same command as the cells.
S=${S:?}; D=$S/kapitsa-wv/bench/results/2026-11-05-m5-max-35b-gpu-wave/cell-ab
cd $S/kapitsa-wv/bench/harness
for rd in 1 2; do
  if (( rd % 2 == 1 )); then arms=(old new); else arms=(new old); fi
  for a in $arms; do
    sleep 60
    if [ $a = new ]; then B=$S/kapitsa-wv/build/kapitsa; else B=$S/kapitsa-cells3/build/kapitsa; fi
    echo "=== round $rd $a $(date +%H:%M:%S) $(pmset -g batt | head -1)" >> $D/session.log
    lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$B \
      python3 compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain \
      --quiet-timeout 60 --out-dir $D/runs-$a >> $D/session.log 2>&1
  done
done
echo done >> $D/session.log
