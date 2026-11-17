#!/bin/zsh
# runAB.sh BRANCH_BINARY MODEL TOKENS OUT: ab_nohit2.py, main 4a9cce41 (kapitsa-main21: built on the rented M1 with the
# same toolchain as the branch's binaries, Apple clang 21) against BRANCH_BINARY, three pairs, request 1 of TOKENS
# tokens, vm_stat sampled; under one hold of the GPU lock
cd ~/kapitsa-ssdcache/v3
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
B=$1; M=$2; T=$3; OUT=out/$4; mkdir -p $OUT
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.memsize; $B --version | head -1; ./kapitsa-main21 --version | head -1; } > $OUT/machine.txt
python3 vmlog.py $OUT/vm.log & V=$!
python3 ab_nohit2.py ./kapitsa-main21 $B $KAPITSA_HOME/models/$M bench/harness/corpus/bench_corpus.txt --tokens $T --log $OUT/ab_nohit-serve.log --server-args "--host 127.0.0.1 --private" > $OUT/ab_nohit.log 2>&1
kill $V
echo done > $OUT/DONE
