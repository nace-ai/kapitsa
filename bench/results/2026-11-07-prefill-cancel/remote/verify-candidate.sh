#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
cd /Users/bytedance/kapitsa-cancel-checkpoint-20261107/candidate
export VK_RUN=cancel-checkpoint-926bce8b VK_EVIDENCE=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/evidence
trap 'echo $? > ../verify-candidate.rc' EXIT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > ../configure-926bce8b.log 2>&1
V=.dsh/skills/verify-kapitsa/scripts/vk
$V build kapitsa api_test execution_session_test http_stream_test bench_test update_test prefix_cache_test grammar_test structured_check fn_store_test fn_dflash_test drafter_attach_test tokenizer_test iq_layout_test exl3_test remote_test cluster_test split_place_test gemm_slices_test device_profile_test qwen4_config_test json_file_test affine_pack_test systemone systemone_test systemone_codes_probe_test systemone_codes_driver splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test ws_upgrade_test logit_dump
$V cpu-tests
mkdir -p ../extra-cpu-926bce8b
for target in execution_session_test http_stream_test drafter_attach_test exl3_test device_profile_test qwen4_config_test json_file_test affine_pack_test systemone_test systemone_codes_probe_test splash_protocol_test splash_runtime_test splash_discover_test splash_plan_test; do
  build/$target > ../extra-cpu-926bce8b/$target.log 2>&1
  echo "PASS extra $target"
done
for script in tests/test_profile_gates.py tests/test_comment_gates.py tests/test_drex_max.py tests/test_logit_dump.py tools/qwen4_exp_pack_test.py; do
  name=$(basename "$script" .py)
  case "$name" in
    test_drex_max) python3 "$script" --cli build > ../extra-cpu-926bce8b/$name.log 2>&1 ;;
    test_logit_dump) python3 "$script" build > ../extra-cpu-926bce8b/$name.log 2>&1 ;;
    *) python3 "$script" > ../extra-cpu-926bce8b/$name.log 2>&1 ;;
  esac
  echo "PASS extra $name"
done
$V checks 2c41b56cc6c5f0895d4359d5ff33d1459b22bc39
build/kapitsa --version
$V gpu run --eta 4 --label 'prefill cancellation regression 4B' -- python3 tests/test_prefill_cancel.py --cli build/kapitsa --reference ../reference-kapitsa --model /Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit --output ../regression-4b-926bce8b --port 8298
