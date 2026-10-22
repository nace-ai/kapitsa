#!/bin/bash
# After the clean sweep: Flint protocol (Kapitsa only), interleaved A B C A B C:
#   A = 3da3798 (#306 battery build), B = cff1b1e (step-1 main), C = a97e596 (main now, after #324 touched Flint)
. ~/kenv.sh
until grep -q DL-DONE ~/kq/dl.log 2>/dev/null; do sleep 30; done
L=~/kq/flint/run4.log
cd ~/kapitsa-flint && git worktree add -q --detach ~/kq/head a97e596 && cd ~/kq/head && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build -j 10 --target kapitsa logit_dump 2>&1 | tail -1 >> $L
echo "$(date +%T) $(~/kq/head/build/kapitsa --version)" >> $L
sleep 60
cd ~/kq/main/bench/harness
step() {
  pmset displaysleepnow; sleep 5
  echo "$(date +%T) power $(pmset -g batt | head -1 | sed "s/Now drawing from //") $(pmset -g batt | grep -o "[0-9]*%") free $(memory_pressure | tail -1 | grep -o "[0-9]*%") :: $1 $2" >> $L
  BENCH_OUT=~/kq/flint/ab KAPITSA_BIN=$2 python3 compare_engines.py configs/flint.json --skip agent --only flint-kapitsa >> $L 2>&1; echo "$(date +%T) rc=$? :: $1" >> $L
}
for i in 1 2; do
  step A ~/kapitsa-flint/build/kapitsa
  step B ~/kq/main/build/kapitsa
  step C ~/kq/head/build/kapitsa
done
echo "$(date +%T) AB-DONE" >> $L
