#!/bin/zsh
# The 35B-A3B's M5 Max cell (q36-35b-kapitsa) on main after #941 and #946 (Kapitsa 0.2.28 plus #951, results only):
# compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain, three runs,
# from a clean checkout of main built Release. Under caffeinate -is, on AC power, the session idle; each run takes the
# GPU lock itself (lockf ... env NACE_GPU_LOCK_HELD=1). S: the scratch folder.
set -e
S=${S:?}
W=$S/kapitsa-cell35
cd $W && git rev-parse --short HEAD > $S/cell35/commit.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=$(git rev-parse --short HEAD) > $S/cell35/cmake.out 2>&1
cmake --build build -j --target kapitsa > $S/cell35/build.out 2>&1
cd $W/bench/harness
for i in 1 2 3; do
  echo "=== run $i $(date +%H:%M:%S) $(pmset -g | grep -o 'powermode *[0-9]') $(pmset -g batt | head -1)" >> $S/cell35/session.log
  lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
    python3 compare_engines.py configs/kapitsa.json --only q36-35b-kapitsa --agentic --skip tools,agent,cache,sustain \
    --quiet-timeout 60 --out-dir $S/cell35/runs >> $S/cell35/session.log 2>&1
done
echo done
