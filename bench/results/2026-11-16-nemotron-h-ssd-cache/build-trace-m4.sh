#!/bin/zsh
# under the GPU lock: build the instrumented commit and the head, the head CPU tests, then the 4B trace
CM=$(ls ~/Library/Python/*/bin/cmake | head -1)
for c in b681cde3 49b176d8; do
  cd ~/kapitsa-build/$c
  t=kapitsa; [[ $c == 49b176d8 ]] && t="kapitsa prefix_cache_test api_test kflash_cache_test"
  $CM -S src -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT=$c > cfg.log 2>&1 && $CM --build build -j --target ${=t} > build.log 2>&1; echo "build $c rc $?"
done
cd ~/kapitsa-build/49b176d8/src
for t in prefix_cache_test api_test kflash_cache_test; do ../build/$t > ../$t.log 2>&1; echo "$t rc $? $(tail -1 ../$t.log)"; done
cd ~/kapitsa-ssdcache/v3
cp ~/kapitsa-build/b681cde3/build/kapitsa ./kapitsa-trace
export KAPITSA_HOME=~/kapitsa-run/home KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
OUT=out/trace-4b; mkdir -p $OUT
./kapitsa-trace --version | head -1 > $OUT/machine.txt; sw_vers >> $OUT/machine.txt
python3 trace_ttft.py $OUT/trace ./kapitsa-trace $KAPITSA_HOME/models/nemotron-3-nano-4b-fp8 --ttft --tokens 16000 --rounds 1 > $OUT/trace.log 2>&1; echo "trace rc $?"
echo ALLDONE
