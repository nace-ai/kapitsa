#!/bin/bash
set -euo pipefail
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
export MAKEFLAGS=-l12 PYTHONUNBUFFERED=1
P=/Users/bytedance/kapitsa-nameless-tool-20261107
mkdir -p "$P/evidence"
echo $$ > "$P/build.pid"
trap 'echo $? > "$P/build.exit"' EXIT
cd "$P/candidate"
git rev-parse HEAD HEAD: > "$P/evidence/source.txt"
git status --porcelain > "$P/evidence/status-before.txt"
test ! -s "$P/evidence/status-before.txt"
export VK_EVIDENCE="$P/evidence/vk"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$P/evidence/configure.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build > "$P/evidence/build.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build parser_test execution_session_test http_stream_test drafter_attach_test exl3_test device_profile_test qwen4_config_test json_file_test affine_pack_test systemone systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test logit_dump dflash_accept_test dflash_lowrank_cache_test dflash_plan_test > "$P/evidence/extra-build.log" 2>&1
build/kapitsa --version > "$P/evidence/version.txt"
.dsh/skills/verify-kapitsa/scripts/vk cpu-tests > "$P/evidence/cpu.log" 2>&1
for t in execution_session_test http_stream_test drafter_attach_test exl3_test device_profile_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test dflash_lowrank_cache_test dflash_plan_test; do
  build/$t > "$P/evidence/$t.log" 2>&1
done
build/dflash_accept_test --cpu > "$P/evidence/dflash-accept.log" 2>&1
python3 tests/test_drex_max.py --cli build > "$P/evidence/drex.log" 2>&1
python3 tests/test_logit_dump.py build > "$P/evidence/logit-dump.log" 2>&1
python3 tests/test_profile_gates.py > "$P/evidence/profile-gates.log" 2>&1
python3 tests/test_comment_gates.py > "$P/evidence/comment-gates.log" 2>&1
python3 tests/test_idle_unload.py --no-gpu > "$P/evidence/idle-unload.log" 2>&1
python3 tools/qwen4_exp_pack_test.py > "$P/evidence/pack.log" 2>&1
python3 tests/test_splash_patches.py --apply > "$P/evidence/patches.log" 2>&1
prek run --all-files > "$P/evidence/prek.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk cleanup > "$P/evidence/cleanup.log" 2>&1
git status --porcelain > "$P/evidence/status-after.txt"
test ! -s "$P/evidence/status-after.txt"
