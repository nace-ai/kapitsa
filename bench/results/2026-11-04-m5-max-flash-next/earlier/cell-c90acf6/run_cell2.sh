#!/bin/zsh
# Flash-Next's M5 Max cell (flash-kapitsa) after #921, #922 and #923 (#923's head c90acf6: main f9e85aa plus #923):
# (compare_engines.py configs/flash-next.json --only flash-kapitsa --agentic --skip tools,agent,cache,sustain), three
# runs (4-6: runs 1-3 spread 98.8-110.1 at T 0.6), from a clean worktree of main at $1 built Release. Under caffeinate -is, on AC power, the session idle; each run
# takes the GPU lock itself (lockf ... env NACE_GPU_LOCK_HELD=1). S: the scratch folder.
set -e
S=${S:?}
W=$S/kapitsa-cell
cd $W && git rev-parse --short HEAD > $S/cell/commit.txt && git status --short --untracked-files=no | wc -l >> $S/cell/commit.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > $S/cell/cmake.out 2>&1
cmake --build build -j --target kapitsa > $S/cell/build.out 2>&1
cd $W/bench/harness
for i in 4 5 6; do
  echo "=== run $i $(date +%H:%M:%S) $(pmset -g | grep -o 'powermode *[0-9]') $(pmset -g batt | head -1)" >> $S/cell/session.log
  lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
    python3 compare_engines.py configs/flash-next.json --only flash-kapitsa --agentic --skip tools,agent,cache,sustain \
    --quiet-timeout 60 --out-dir $S/cell/runs >> $S/cell/session.log 2>&1
done
echo done
