#!/bin/bash
# #343: niced builds (another job holds the GPU lock: -j2)
b() { d=$1; shift; cd ~/Documents/nace/k343/wt-$d || exit 1
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null || { echo "cmake fail $d"; return; }
  nice -n 19 taskpolicy -b cmake --build build -j2 --target "$@" > build.log 2>&1; echo "$d targets=$* rc=$? head=$(git rev-parse --short HEAD)"; }
b f9e85aa7 flint_bench
b b210e9ab flint_bench
b af7d7958 exl3_register_check exl3_check
b 153dff95 fn_check
echo BUILDS_DONE
