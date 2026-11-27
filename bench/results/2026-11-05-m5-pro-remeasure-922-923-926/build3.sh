#!/bin/bash
b() { d=$1; shift; cd ~/Documents/nace/k343/wt-$d || exit 1
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null || { echo "cmake fail $d"; return; }
  nice -n 19 taskpolicy -b cmake --build build -j2 --target "$@" > build3.log 2>&1; echo "$d targets=$* rc=$? head=$(git rev-parse --short HEAD) sha=$(shasum -a 256 build/$1 | cut -c1-16)"; }
b b210e9ab moe4_time
b f9e85aa7 moe4_time
b 153dff95 moe4_time
echo BUILD3_DONE
