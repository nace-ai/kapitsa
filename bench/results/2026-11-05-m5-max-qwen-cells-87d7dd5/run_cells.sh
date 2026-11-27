#!/bin/zsh
# The 27B's and 35B-A3B's M5 Max cells (q38-27b-kapitsa, q36-35b-kapitsa) on 7afb51d, main 49304c9 (#964, #969) with #970 merged in (the tree of main 87d7dd5)
# in: compare_engines.py configs/kapitsa.json --only <label> --agentic --skip tools,agent,cache,sustain, three runs
# each, built Release. Under caffeinate -is, on AC power, the session idle; each run takes the GPU lock itself
# (lockf ... env NACE_GPU_LOCK_HELD=1). S: the scratch folder.
set -e
S=${S:?}
W=$S/kapitsa-cells3
cd $W && git rev-parse --short HEAD > $S/cells3/commit.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=$(git rev-parse --short HEAD) > $S/cells3/cmake.out 2>&1
cmake --build build -j --target kapitsa > $S/cells3/build.out 2>&1
cd $W/bench/harness
for label in q38-27b-kapitsa q36-35b-kapitsa; do
  for i in 1 2 3; do
    echo "=== $label run $i $(date +%H:%M:%S) $(pmset -g | grep -o 'powermode *[0-9]') $(pmset -g batt | head -1)" >> $S/cells3/session.log
    lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
      python3 compare_engines.py configs/kapitsa.json --only $label --agentic --skip tools,agent,cache,sustain \
      --quiet-timeout 60 --out-dir $S/cells3/runs >> $S/cells3/session.log 2>&1
  done
done
echo done >> $S/cells3/session.log
# Then the 27B's 4K prompt with no decode phase before it (prefill-alone/), twice, the same build:
#   lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
#     python3 compare_engines.py configs/kapitsa.json --only q38-27b-kapitsa --agentic \
#     --skip decode,greedy,tools,agent,cache,sustain,sweep --sizes 4000 --quiet-timeout 60 --out-dir $S/pfonly/runs
