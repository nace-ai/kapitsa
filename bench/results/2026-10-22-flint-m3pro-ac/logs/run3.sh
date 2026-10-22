#!/bin/bash
# Clean context-sweep rerun: the first sweep (21:20-21:41) overlapped the niced model downloads (pull.sh, 21:19-21:40).
. ~/kenv.sh
until grep -q PROTOCOL2-DONE ~/kq/flint/run2.log 2>/dev/null; do sleep 30; done
pw() { pmset -g batt | head -1 | sed "s/Now drawing from //"; }
cd ~/kq/flint/sweep2; S=~/kq/flint/sweep2/sweep.log
pmset displaysleepnow; sleep 5
echo "$(date +%T) power $(pw) free $(memory_pressure | tail -1 | grep -o "[0-9]*%") :: kapitsa" >> $S
~/bin/lockf -k /tmp/kapitsa-gpu.lock zsh ./sweep_engine.sh kapitsa 1024,2048,4096,8192,16384,32768,65536,131072 >> $S 2>&1; echo "$(date +%T) rc=$? kapitsa" >> $S
pmset displaysleepnow; sleep 5
echo "$(date +%T) power $(pw) free $(memory_pressure | tail -1 | grep -o "[0-9]*%") :: llamacpp -c 70000" >> $S
LLAMA_CTX=70000 ~/bin/lockf -k /tmp/kapitsa-gpu.lock zsh ./sweep_engine.sh llamacpp 1024,2048,4096,8192,16384,32768,65536 >> $S 2>&1; echo "$(date +%T) rc=$? llamacpp" >> $S
echo "$(date +%T) SWEEP2-DONE power $(pw)" >> $S
