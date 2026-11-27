#!/bin/zsh
# The 35B-A3B's M5 Max cell (q36-35b-kapitsa) on main after #975 (NI_PF_WAVE, 8192-row chunks): compare_engines.py
# configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain, three runs, built Release,
# 120 s idle before each run, after 5 minutes idle, the session idle; each run takes the GPU lock itself. S: the scratch folder.
set -e
S=${S:?}; W=$S/kapitsa-cells4
cd $W && git rev-parse --short HEAD > $S/cells4/commit.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=$(git rev-parse --short HEAD) > $S/cells4/cmake.out 2>&1
cmake --build build -j --target kapitsa > $S/cells4/build.out 2>&1
cd $W/bench/harness
for i in 1 2 3; do
  sleep 120
  echo "=== q36-35b-kapitsa run $i $(date +%H:%M:%S) $(pmset -g | grep -o 'powermode *[0-9]') $(pmset -g batt | head -1)" >> $S/cells4/session.log
  lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
    python3 compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain \
    --quiet-timeout 60 --out-dir $S/cells4/runs >> $S/cells4/session.log 2>&1
done
echo done >> $S/cells4/session.log
