#!/bin/bash
set -euo pipefail
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
export PYTHONUNBUFFERED=1 MAKEFLAGS=-l12
P=/Users/bytedance/kapitsa-affine-rebase-20261107
mkdir -p "$P/evidence"
echo $$ > "$P/build.pid"
trap 'rc=$?; echo "$rc" > "$P/build.exit"' EXIT
repo=/Users/bytedance/kapitsa-ngram-affine-20261107/candidate
git -C "$repo" fetch /Users/bytedance/kapitsa-affine-cost-20261107/rebase.bundle refs/remotes/upstream/main
for variant in base candidate; do
  if [ "$variant" = base ]; then ref=4d1d7445; else ref=6bd1dd1c; fi
  git -C "$repo" worktree add --detach "$P/$variant" "$ref"
  cd "$P/$variant"
  actual=$(git rev-parse HEAD)
  git rev-parse HEAD HEAD: > "$P/evidence/$variant-source.txt"
  export VK_EVIDENCE="$P/evidence/vk"
  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DKAPITSA_GIT="${actual:0:8}" > "$P/evidence/$variant-configure.log" 2>&1
  .dsh/skills/verify-kapitsa/scripts/vk build kapitsa ni_run logit_dump fn_store_test > "$P/evidence/$variant-build.log" 2>&1
  build/kapitsa --version > "$P/evidence/$variant-version.txt"
  build/fn_store_test > "$P/evidence/$variant-fn-store.log" 2>&1
  touch "$P/$variant-built"
done
cd "$P/candidate"
.dsh/skills/verify-kapitsa/scripts/vk cpu-tests > "$P/evidence/candidate-cpu.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build execution_session_test http_stream_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone systemone_test systemone_codes_probe_test systemone_codes_driver splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test device_profile_test > "$P/evidence/candidate-extra-build.log" 2>&1
for target in execution_session_test http_stream_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test device_profile_test; do
  "build/$target" > "$P/evidence/$target.log" 2>&1
done
python3 tools/qwen4_exp_pack_test.py > "$P/evidence/exp-pack.log" 2>&1
python3 tests/test_profile_gates.py > "$P/evidence/profile-gates.log" 2>&1
python3 tests/test_comment_gates.py > "$P/evidence/comment-gates.log" 2>&1
prek run --all-files > "$P/evidence/prek.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk cleanup > "$P/evidence/cleanup.log" 2>&1
touch "$P/cpu-passed"
