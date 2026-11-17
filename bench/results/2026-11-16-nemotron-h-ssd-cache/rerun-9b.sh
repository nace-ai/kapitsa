#!/bin/zsh
# rerun-9b.sh: ab_nohit.py again on the M2 (the binaries of nohit-nano-9b-m2/), vm_stat sampled by vmlog.py; a loop
# beside it logged kern.memorystatus_vm_pressure_level and kern.memorystatus_level each second (pressure.log).
# Under one hold of the GPU lock (lockf ... env NI_LOCK_HELD=1 ...)
cd ~/kapitsa-ssdcache/v3
O=out/rerun-9b; mkdir -p $O
python3 vmlog.py $O/vm.log & V=$!
python3 ab_nohit.py ./kapitsa-main ./kapitsa ~/kapitsa-run/home/models/nemotron-nano-9b-v2-nvfp4 bench/harness/corpus/bench_corpus.txt --log $O/ab_nohit-serve.log > $O/ab_nohit.log 2>&1
kill $V
echo done > $O/DONE
