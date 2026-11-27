#!/bin/bash
set -euo pipefail
R=/Users/bytedance/kapitsa-json-tools-20261107
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
export MAKEFLAGS=-l6 PYTHONUNBUFFERED=1 VK_EVIDENCE="$R/evidence/vk"
printf '%s\n' $$ > "$R/candidate.pid"
cd /Users/bytedance/kapitsa-stop-sequence-20261107/candidate
git fetch "$R/nano-candidate.bundle" fix/json-tool-names:refs/remotes/nano-followup/candidate
git worktree add --detach "$R/candidate" refs/remotes/nano-followup/candidate
cd "$R/candidate"
run() {
  label=$1; shift
  printf '%q ' "$@" > "$R/evidence/$label.cmd"
  printf '\n' >> "$R/evidence/$label.cmd"
  set +e
  "$@" > "$R/evidence/$label.log" 2>&1
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$R/evidence/$label.rc"
  printf '%s %s\n' "$label" "$rc" >> "$R/evidence/results.txt"
  return 0
}
git rev-parse HEAD > "$R/evidence/candidate-head.txt"
run candidate-configure cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
run candidate-build .dsh/skills/verify-kapitsa/scripts/vk build kapitsa nh_chat_check parser_test logit_dump systemone
[[ $(cat "$R/evidence/candidate-build.rc") == 0 ]]
build/kapitsa --version > "$R/evidence/candidate-version.txt"
run candidate-nano build/nh_chat_check "$R/nano9b-tokenizer"
run cpu .dsh/skills/verify-kapitsa/scripts/vk cpu-tests
run extra-build .dsh/skills/verify-kapitsa/scripts/vk build device_profile_test execution_session_test http_stream_test ws_upgrade_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test systemone_codes_driver splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test dflash_accept_test
for target in device_profile_test execution_session_test http_stream_test ws_upgrade_test drafter_attach_test exl3_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test; do run "$target" "build/$target"; done
run dflash-accept-cpu build/dflash_accept_test --cpu
run drex-max python3 tests/test_drex_max.py --cli build
run logit-dump-cli python3 tests/test_logit_dump.py build
run exp-pack python3 tools/qwen4_exp_pack_test.py
run splash-patches python3 tests/test_splash_patches.py --apply
run profile-gates python3 tests/test_profile_gates.py
run comment-gates python3 tests/test_comment_gates.py
run pr-size python3 tests/test_pr_size.py
run grammar build/grammar_test
model4=/Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit
model35=/Users/bytedance/Models/Qwen3.6-35B-A3B-4bit
run parser-4b build/parser_test "$model4"
run structured-4b python3 tests/test_structured.py --cli build "$model4"
run grammar-4b build/grammar_test "$model4/tokenizer.json"
run prek prek run --all-files
run cleanup .dsh/skills/verify-kapitsa/scripts/vk cleanup
shasum -a 256 src/engine/output_parser.h tools/nh_chat_check.cpp build/kapitsa build/nh_chat_check "$R"/nano9b-tokenizer/*.json > "$R/evidence/candidate-hashes.txt"
sw_vers > "$R/evidence/environment.txt"
sysctl hw.memsize machdep.cpu.brand_string >> "$R/evidence/environment.txt"
clang --version >> "$R/evidence/environment.txt"
touch "$R/evidence/candidate-complete"
