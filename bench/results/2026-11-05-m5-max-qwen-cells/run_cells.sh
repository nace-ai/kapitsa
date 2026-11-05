#!/bin/zsh
# The 27B's and 35B-A3B's M5 Max cells (q38-27b-kapitsa, q36-35b-kapitsa) on main after #959 and #961:
# compare_engines.py configs/kapitsa.json --only <label> --agentic --skip tools,agent,cache,sustain, three runs each, from
# a clean checkout of main built Release. Under caffeinate -is, on AC power, the session idle; each run takes the GPU
# lock itself (lockf ... env NACE_GPU_LOCK_HELD=1). S: the scratch folder.
set -e
S=${S:?}
W=$S/kapitsa-cells2
cd $W && git rev-parse --short HEAD > $S/cells/commit.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=$(git rev-parse --short HEAD) > $S/cells/cmake.out 2>&1
cmake --build build -j --target kapitsa > $S/cells/build.out 2>&1
cd $W/bench/harness
for label in q38-27b-kapitsa q36-35b-kapitsa; do
  for i in 1 2 3; do
    echo "=== $label run $i $(date +%H:%M:%S) $(pmset -g | grep -o 'powermode *[0-9]') $(pmset -g batt | head -1)" >> $S/cells/session.log
    lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
      python3 compare_engines.py configs/kapitsa.json --only $label --agentic --skip tools,agent,cache,sustain \
      --quiet-timeout 60 --out-dir $S/cells/runs >> $S/cells/session.log 2>&1
  done
done
echo done >> $S/cells/session.log
# The 27B's three runs were then repeated (the first three are in first-27b-runs/, their prefill and decode phases
# contention-flagged at a mean of 6-17% foreign GPU load), the same command with --out-dir <scratch>/cells/runs27b,
# the session left idle:
#   for i in 1 2 3; do lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
#     python3 compare_engines.py configs/kapitsa.json --only q38-27b-kapitsa --agentic \
#     --skip tools,agent,cache,sustain --quiet-timeout 60 --out-dir $S/cells/runs27b; done
