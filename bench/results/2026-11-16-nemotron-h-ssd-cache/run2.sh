#!/bin/zsh
# run2.sh MODEL: the head's runs, under the GPU lock taken once by the caller (lockf ... env NI_LOCK_HELD=1 ...):
# func (tests/test_disk_cache.py), ttft (2K/8K/16K, 3 rounds), ab (ab.py: main against the head, 3 pairs)
cd ~/kapitsa-ssdcache/head
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
M=$1
MD=$KAPITSA_HOME/models/$M
OUT=out/head-$M-$(hostname -s)
mkdir -p $OUT
sysinfo() { echo "== $(date '+%F %T') $1"; vm_stat | egrep 'Swapins|Swapouts'; ps -Ao %cpu,command -r | sed -n 2,3p | cut -c1-120; }
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; ./kapitsa --version 2>&1 | head -2; ./kapitsa-main --version 2>&1 | head -2; } > $OUT/machine.txt
sysinfo func >> $OUT/swap.log
python3 tests/test_disk_cache.py ./kapitsa $MD --keep > $OUT/func.log 2>&1
sysinfo func-end >> $OUT/swap.log
K=$(sed -n 's/^kept //p' $OUT/func.log); [[ -f $K/serve.log ]] && cp $K/serve.log $OUT/func-serve.log && rm -rf $K
for n in 2000 8000 16000; do
  sysinfo ttft-$n >> $OUT/swap.log
  python3 tests/test_disk_cache.py ./kapitsa $MD --ttft --tokens $n --rounds 3 > $OUT/ttft-$n.log 2>&1
  sysinfo ttft-$n-end >> $OUT/swap.log
done
sysinfo ab >> $OUT/swap.log
python3 ab.py ./kapitsa-main ./kapitsa $MD bench/harness/corpus/bench_corpus.txt --log $OUT/ab-serve.log > $OUT/ab.log 2>&1
sysinfo ab-end >> $OUT/swap.log
echo done > $OUT/DONE
