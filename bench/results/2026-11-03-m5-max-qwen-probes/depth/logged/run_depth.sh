#!/bin/zsh
# Flash-Next MTP depth 2 (old) against 3 (new), main 8715427's unpatched build, every leg's conditions logged
# (ab_lc.py: quiet GPU, load average and AC before and after); under caffeinate -is lockf -k /tmp/kapitsa-gpu.lock.
set -e
export NI_LOCK_HELD=1
S=${S:?}
cd $S/lc
AB_HARNESS=$S/kapitsa-pr/bench/harness python3 ab_lc.py depth.json depth 3 > depth.out 2>&1
echo done
