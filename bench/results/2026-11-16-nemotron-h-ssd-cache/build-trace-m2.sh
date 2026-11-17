#!/bin/zsh
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
OUT=out/trace-9b; mkdir -p $OUT
{ ./kapitsa-trace --version | head -1; sw_vers; } > $OUT/machine.txt
python3 trace_ttft.py $OUT/trace ./kapitsa-trace $KAPITSA_HOME/models/nemotron-nano-9b-v2-nvfp4 --ttft --tokens 16000 --rounds 1 > $OUT/trace.log 2>&1; echo "trace rc $?"
OUT=out/cli; mkdir -p $OUT
{ ~/kapitsa-build/49b176d8/build/kapitsa --version | head -1; sw_vers; } > $OUT/machine.txt
(cd ~/kapitsa-build/49b176d8/src && env -u KAPITSA_HOME python3 tests/test_server.py --cli ../build/kapitsa) > $OUT/test_server_cli.log 2>&1; echo "cli rc $?"
echo ALLDONE
