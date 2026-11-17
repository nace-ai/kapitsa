#!/bin/zsh
# runSSD.sh MODEL ROUNDS OUT: the SSD head's (b4e33647) runs under one hold of the GPU lock: ab_nohit2.py (main 4a9cce41
# against it, three pairs, vm_stat sampled), test_disk_cache.py --ttft --tokens 16000 --rounds ROUNDS (0: none)
cd ~/kapitsa-ssdcache/v3
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
M=$1; R=$2; OUT=out/$3; mkdir -p $OUT
B=~/kapitsa-build/b4e33647/build/kapitsa
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; $B --version | head -1; ./kapitsa-main --version | head -1; } > $OUT/machine.txt
python3 vmlog.py $OUT/vm.log & V=$!
python3 ab_nohit2.py ./kapitsa-main $B $KAPITSA_HOME/models/$M bench/harness/corpus/bench_corpus.txt --log $OUT/ab_nohit-serve.log --server-args "--host 127.0.0.1 --private" > $OUT/ab_nohit.log 2>&1
kill $V
[[ $R -gt 0 ]] && python3 tests/test_disk_cache-ssd.py $B $KAPITSA_HOME/models/$M --ttft --tokens 16000 --rounds $R --server-args "--host 127.0.0.1 --private" > $OUT/ttft-16000.log 2>&1
echo done > $OUT/DONE
