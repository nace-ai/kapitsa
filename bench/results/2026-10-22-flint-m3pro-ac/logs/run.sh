#!/bin/bash
# Flint on AC power, main build: protocol K L K L (+ rec once to record its refusal), then the context sweep.
. ~/kenv.sh; cd ~/kq/main/bench/harness
L=~/kq/flint/run.log
pw() { pmset -g batt | head -1 | sed "s/Now drawing from //"; }
step() {
  echo "$(date +%T) power $(pw) $(pmset -g batt | grep -o "[0-9]*%") free $(memory_pressure | tail -1 | grep -o "[0-9]*%") :: $*" >> $L
  python3 compare_engines.py configs/flint.json --skip agent "$@" >> $L 2>&1; echo "$(date +%T) rc=$? :: $*" >> $L
}
step --only flint-kapitsa
step --only flint-llamacpp
step --only flint-kapitsa
step --only flint-llamacpp
step --only flint-rec
echo "$(date +%T) PROTOCOL-DONE" >> $L
cd ~/kq/flint/sweep; S=~/kq/flint/sweep/sweep.log
echo "$(date +%T) power $(pw) :: kapitsa" >> $S
~/bin/lockf -k /tmp/kapitsa-gpu.lock zsh ./sweep_engine.sh kapitsa 1024,2048,4096,8192,16384,32768,65536,131072 >> $S 2>&1; echo "$(date +%T) rc=$? kapitsa" >> $S
echo "$(date +%T) power $(pw) :: llamacpp -c 70000" >> $S
LLAMA_CTX=70000 ~/bin/lockf -k /tmp/kapitsa-gpu.lock zsh ./sweep_engine.sh llamacpp 1024,2048,4096,8192,16384,32768,65536 >> $S 2>&1; echo "$(date +%T) rc=$? llamacpp" >> $S
echo "$(date +%T) SWEEP-DONE power $(pw)" >> $S; echo "$(date +%T) ALL-DONE" >> $L
