#!/bin/bash
set -euo pipefail
P=/Users/bytedance/kapitsa-stop-sequence-20261107
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
export MAKEFLAGS=-l12 PYTHONUNBUFFERED=1 VK_EVIDENCE="$P/evidence/vk"
echo $$ > "$P/finish-cpu.pid"
trap 'rc=$?; if [ "$rc" != 0 ]; then echo "$rc" > "$P/finish-cpu.failed"; fi' EXIT
cd "$P/candidate"
.dsh/skills/verify-kapitsa/scripts/vk build device_profile_test execution_session_test http_stream_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test systemone_codes_driver splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test parser_test dflash_accept_test > "$P/evidence/extra-build.log" 2>&1
for target in device_profile_test execution_session_test http_stream_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test; do
  "build/$target" > "$P/evidence/$target.log" 2>&1
done
build/grammar_test > "$P/evidence/grammar.log" 2>&1
build/dflash_accept_test --cpu > "$P/evidence/dflash-accept-cpu.log" 2>&1
python3 tests/test_drex_max.py --cli build > "$P/evidence/drex-max.log" 2>&1
python3 tests/test_logit_dump.py build > "$P/evidence/logit-dump-cli.log" 2>&1
python3 tools/qwen4_exp_pack_test.py > "$P/evidence/exp-pack.log" 2>&1
python3 tests/test_splash_patches.py --apply > "$P/evidence/splash-patches.log" 2>&1
python3 tests/test_profile_gates.py > "$P/evidence/profile-gates.log" 2>&1
python3 tests/test_comment_gates.py > "$P/evidence/comment-gates.log" 2>&1
model=/Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit
build/parser_test "$model" > "$P/evidence/parser.log" 2>&1
python3 tests/test_structured.py --cli build "$model" > "$P/evidence/structured.log" 2>&1
prek run --all-files > "$P/evidence/prek.log" 2>&1
git diff --check
# Completion marker only; test_update has two preserved failures, not a CPU pass.
touch "$P/cpu-checked"
touch "$P/build-complete"
