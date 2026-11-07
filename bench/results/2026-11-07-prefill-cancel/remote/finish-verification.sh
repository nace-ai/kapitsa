#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
cd /Users/bytedance/kapitsa-cancel-checkpoint-20261107/candidate
export VK_RUN=cancel-checkpoint-926bce8b VK_EVIDENCE=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/evidence
trap 'echo $? > ../finish-verification.rc' EXIT
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
.dsh/skills/verify-kapitsa/scripts/vk checks 2c41b56cc6c5f0895d4359d5ff33d1459b22bc39
build/kapitsa --version
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 4 --label 'prefill cancellation regression 4B' -- python3 tests/test_prefill_cancel.py --cli build/kapitsa --reference ../reference-kapitsa --model /Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit --output ../regression-4b-926bce8b --port 8298
