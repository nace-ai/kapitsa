#!/bin/zsh
# Every probe in this folder, run under the GPU lock (caffeinate -is lockf -k -t 7200 /tmp/kapitsa-gpu.lock ./run.sh):
# each patch applied to main 8715427 in turn, built, and its default and variant run interleaved (A B A B); then
# Flash-Next's MTP depth 2 against 3, the load-only layout bench, and the 35B and 27B decode profiles on the unpatched
# build. REPO: a checkout of 8715427 configured as build/ (Release); OUT: this folder.
# usage: REPO=<repo> OUT=<this folder> ./run.sh
set -e
export NI_LOCK_HELD=1
FN=~/models/qwen3.8-flash-next-4bit
IDS="$(cat $OUT/prompt-ids.txt)"
cd $REPO
build() { cmake --build build -j --target ni_run fn_check > /dev/null; }
prof() { NI_PROFILE=1 build/ni_run $FN "$IDS" 96 0 --spec 2 --dflash none > $OUT/$1 2>&1; }
git apply $OUT/qmvp3.diff && build
for i in 1 2; do prof qmvp3/base-$i.out; NI_QMVP3=1 prof qmvp3/var-$i.out; done
git checkout -q -- . && git apply $OUT/qmvs-rows.diff && build
FN_ROWS=1 FN_VERBOSE=1 build/fn_check kernels > $OUT/qmvs-rows/fn-check.out 2>&1 || true
for i in 1 2; do prof qmvs-rows/base-$i.out; NI_QMVS_ROWS=1 prof qmvs-rows/var-$i.out; done
git checkout -q -- . && git apply $OUT/qmvs-shape.diff && build
for i in 1 2; do for v in base w2 w8 t4 t8; do NI_QMVS_VAR=$v prof qmvs-shape/$v-$i.out; done; done
git checkout -q -- . && build
for i in 1 2; do for d in 2 3; do
  build/ni_run $FN "$IDS" 512 0 --spec $d --dflash none --ignore-eos > $OUT/depth/d$d-$i.out 2>&1
done; done
clang++ -std=c++20 -O2 -Iinclude -Isrc -Ithird_party $OUT/bench_layout.cpp build/libnaceinfer.a -framework Metal \
  -framework Foundation -framework CoreFoundation -framework IOSurface -framework IOKit -framework ImageIO \
  -framework CoreGraphics -framework Security -framework SystemConfiguration -lc++ -o build/bench_layout
build/bench_layout > $OUT/layout/bench_layout.out 2>&1
NI_PROFILE=1 build/ni_run ~/models/qwen3.6-35b-a3b-4bit "$IDS" 128 0 --dflash auto > $OUT/profiles/q36-35b.out 2>&1
NI_PROFILE=1 build/ni_run ~/models/qwen3.8-27b-4bit-mtp "$IDS" 128 0 --dflash auto > $OUT/profiles/q38-27b.out 2>&1
echo done
