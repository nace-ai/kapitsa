#!/bin/zsh
# The 35B-A3B's M5 Max cell (q36-35b-kapitsa) on this branch: compare_engines.py configs/kapitsa.json --only
# q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain, three runs, built Release, the session idle; each run takes
# the GPU lock itself. Main's cell on the same tree without the change is bench/results/2026-11-05-m5-max-qwen-cells-87d7dd5.
S=${S:?}; W=$S/kapitsa-wv; D=$W/bench/results/2026-11-05-m5-max-35b-gpu-wave/cell
cd $W/bench/harness
for i in 1 2 3; do
  echo "=== run $i $(date +%H:%M:%S) $(pmset -g | grep -o 'powermode *[0-9]') $(pmset -g batt | head -1)" >> $D/session.log
  lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
    python3 compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain \
    --quiet-timeout 60 --out-dir $D/runs >> $D/session.log 2>&1
done
echo done >> $D/session.log
