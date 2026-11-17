#!/bin/zsh
# run3.sh MODEL: ab_nohit.py (main against the branch, 3 pairs), under the GPU lock taken once by the caller
# (lockf ... env NI_LOCK_HELD=1 ...)
cd ~/kapitsa-ssdcache/v3
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
M=$1
MD=$KAPITSA_HOME/models/$M
OUT=out/$M-$(hostname -s)
mkdir -p $OUT
sysinfo() { echo "== $(date '+%F %T') $1"; vm_stat | egrep 'Swapins|Swapouts'; ps -Ao %cpu,command -r | sed -n 2,3p | cut -c1-120; }
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; ./kapitsa --version 2>&1 | head -2; ./kapitsa-main --version 2>&1 | head -2; } > $OUT/machine.txt
sysinfo nohit >> $OUT/swap.log
python3 ab_nohit.py ./kapitsa-main ./kapitsa $MD bench/harness/corpus/bench_corpus.txt --log $OUT/ab_nohit-serve.log > $OUT/ab_nohit.log 2>&1
sysinfo nohit-end >> $OUT/swap.log
