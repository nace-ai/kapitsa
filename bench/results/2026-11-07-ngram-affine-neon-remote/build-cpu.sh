#!/bin/bash
set -euo pipefail
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 MAKEFLAGS=-l12
P=/Users/bytedance/kapitsa-ngram-affine-20261107
mkdir -p "$P/evidence"
echo $$ > "$P/build-cpu.pid"
trap 'rc=$?; if [ "$rc" != 0 ]; then echo "$rc" > "$P/build-cpu.failed"; fi' EXIT
cd "$P"
if [ ! -d source.git ]; then tar -xzf source-git.tar.gz; fi
if [ ! -d candidate ]; then git clone source.git candidate; fi
if [ ! -d base ]; then git -C candidate worktree add --detach ../base 01b4c6d6; fi
export VK_EVIDENCE="$P/evidence/vk"
for variant in base candidate; do
  cd "$P/$variant"
  actual=$(git rev-parse HEAD)
  if [ "$variant" = base ]; then expected=01b4c6d69fbe69e862c00358f88c94a363129592; else expected=8c0a7ec3bf72623f6c7f92524ef229f3643a9e67; fi
  test "$actual" = "$expected"
  git rev-parse HEAD HEAD: > "$P/evidence/$variant-source.txt"
  git status --porcelain > "$P/evidence/$variant-status-before.txt"
  test ! -s "$P/evidence/$variant-status-before.txt"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT="${actual:0:8}" > "$P/evidence/$variant-configure.log" 2>&1
  .dsh/skills/verify-kapitsa/scripts/vk build kapitsa ni_run logit_dump fn_store_test systemone > "$P/evidence/$variant-build.log" 2>&1
  build/kapitsa --version > "$P/evidence/$variant-version.txt"
  build/fn_store_test > "$P/evidence/$variant-fn-store.log" 2>&1
  touch "$P/$variant-built"
done
cd "$P/candidate"
.dsh/skills/verify-kapitsa/scripts/vk cpu-tests > "$P/evidence/candidate-cpu.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build execution_session_test http_stream_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone systemone_test systemone_codes_probe_test systemone_codes_driver splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test parser_test dflash_accept_test > "$P/evidence/candidate-extra-build.log" 2>&1
for target in execution_session_test http_stream_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test; do
  "build/$target" > "$P/evidence/$target.log" 2>&1
done
build/dflash_accept_test --cpu > "$P/evidence/dflash-accept-cpu.log" 2>&1
python3 tests/test_drex_max.py --cli build > "$P/evidence/drex-max.log" 2>&1
python3 tests/test_logit_dump.py build > "$P/evidence/logit-dump-cli.log" 2>&1
python3 tools/qwen4_exp_pack_test.py > "$P/evidence/exp-pack.log" 2>&1
python3 tests/test_splash_patches.py --apply > "$P/evidence/splash-patches.log" 2>&1
prek run --all-files > "$P/evidence/prek.log" 2>&1
git diff --check
git status --porcelain > "$P/evidence/candidate-status-after.txt"
test ! -s "$P/evidence/candidate-status-after.txt"
touch "$P/cpu-passed"
