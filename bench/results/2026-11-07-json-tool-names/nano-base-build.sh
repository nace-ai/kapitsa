#!/bin/bash
set -euo pipefail
export PATH=/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
R=/Users/bytedance/kapitsa-json-tools-20261107
mkdir -p "$R/evidence"
cd /Users/bytedance/kapitsa-stop-sequence-20261107/candidate
git fetch "$R/nano-base.bundle" refs/remotes/upstream/main:refs/remotes/nano-followup/main
git worktree add --detach "$R/base" ef1053e0dc172477e70f1685fe7b39c1863c7437
cd "$R/base"
cp "$R/nh_chat_check.cpp" tools/nh_chat_check.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$R/evidence/base-configure.log" 2>&1
cmake --build build -j 6 --target nh_chat_check > "$R/evidence/base-build.log" 2>&1
set +e
build/nh_chat_check "$R/nano9b-tokenizer" > "$R/evidence/base-nano.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$R/evidence/base-nano.rc"
git rev-parse HEAD > "$R/evidence/base-head.txt"
shasum -a 256 tools/nh_chat_check.cpp src/engine/output_parser.h build/nh_chat_check > "$R/evidence/base-hashes.txt"
touch "$R/evidence/base-complete"
