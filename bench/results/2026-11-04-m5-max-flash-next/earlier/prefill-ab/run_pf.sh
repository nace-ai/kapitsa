#!/bin/zsh
# Flash-Next prefill, old cell's commit 7f4e709 against #923's head c90acf6 (the new cell's build), interleaved
# (old new, new old, ...), 3 rounds: compare_engines.py configs/flash-next.json --only flash-kapitsa
# --skip decode,greedy,tools,agent (the 4K and 16K cold prefills), each arm's own build. Under caffeinate -is, idle.
set -e
S=${S:?}
cd $S/kapitsa-old && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > $S/cell/old-cmake.out 2>&1 && cmake --build build -j --target kapitsa > $S/cell/old-build.out 2>&1
for rd in 0 1 2; do
  if (( rd % 2 == 0 )); then arms=(old new); else arms=(new old); fi
  for a in $arms; do
    if [[ $a == old ]]; then W=$S/kapitsa-old; else W=$S/kapitsa-cell; fi
    echo "=== round $rd $a $(date +%H:%M:%S) $(git -C $W rev-parse --short HEAD) $(pmset -g batt | head -1)" >> $S/cell/pf.log
    (cd $W/bench/harness && lockf -k -t 3000 /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 KAPITSA_BIN=$W/build/kapitsa \
      python3 compare_engines.py configs/flash-next.json --only flash-kapitsa --skip decode,greedy,tools,agent \
      --quiet-timeout 60 --out-dir $S/cell/pf-$a) >> $S/cell/pf.log 2>&1
  done
done
echo done
