#!/bin/zsh
# llama.cpp 128K alone, the memory check before the load as for Kapitsa/rec; swap watchdog kills the server > 9 GB
SP=<scratch>
f=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}'); echo "$(date +%T) disk free $f GiB"; [[ $f -lt 15 ]] && exit 4
m=$(memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'); echo "$(date +%T) memory free $m% before load, $(sysctl -n vm.swapusage)"
[[ $m -lt 40 ]] && { echo "skip: memory"; exit 3; }
( while sleep 5; do s=$(sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1)
    [[ $s -gt 9216 ]] && { echo "$(date +%T) swap $s MB > 9216: killing llama-server"; pkill -f "llama-server -m .*flint-int4"; exit; }; done ) &
W=$!
TAG=-128k $SP/sweep_engine.sh llamacpp 131072
kill $W 2>/dev/null
echo "$(date +%T) end, $(sysctl -n vm.swapusage)"
