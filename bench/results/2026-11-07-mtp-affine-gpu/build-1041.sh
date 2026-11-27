#!/bin/bash
set -euo pipefail
P=/Users/bytedance/kapitsa-mtp-affine-20261107
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
for arm in base candidate; do
  cd "$P/$arm"
  git rev-parse HEAD > "$P/evidence-gpu-1041/$arm-source.txt"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$P/evidence-gpu-1041/$arm-configure.log" 2>&1
  cmake --build build -j 12 --target kapitsa logit_dump ni_run affine_pack_test load_error_test > "$P/evidence-gpu-1041/$arm-build.log" 2>&1
  build/kapitsa --version > "$P/evidence-gpu-1041/$arm-version.txt"
  build/affine_pack_test > "$P/evidence-gpu-1041/$arm-affine-pack.log" 2>&1
  touch "$P/$arm-built-1041"
done
