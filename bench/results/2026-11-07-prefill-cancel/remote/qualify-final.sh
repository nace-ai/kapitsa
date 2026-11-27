#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
cd /Users/bytedance/kapitsa-cancel-checkpoint-20261107/candidate
export VK_RUN=cancel-checkpoint-final VK_EVIDENCE=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/evidence
trap 'echo $? > ../qualify-final.rc' EXIT
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >../configure-final.log 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build kapitsa
shasum -a 256 build/CMakeFiles/naceinfer.dir/src/engine/engine.cpp.o >../engine-object-after.txt
if cmp -s ../engine-object-before.txt ../engine-object-after.txt; then
  echo 'Engine object SHA256 unchanged after comment-only source edit'
else
  echo 'Engine object differs; inspect source diff and retained hashes'
fi
mkdir -p ../extra-cpu-final
python3 tests/test_comment_gates.py >../extra-cpu-final/test_comment_gates.log 2>&1
python3 tests/test_drex_max.py --cli build >../extra-cpu-final/test_drex_max.log 2>&1
python3 tests/test_logit_dump.py build >../extra-cpu-final/test_logit_dump.log 2>&1
python3 tools/qwen4_exp_pack_test.py >../extra-cpu-final/qwen4_exp_pack_test.log 2>&1
echo 'PASS remaining 4 supplementary CPU checks'
.dsh/skills/verify-kapitsa/scripts/vk checks 2c41b56cc6c5f0895d4359d5ff33d1459b22bc39
build/kapitsa --version
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 4 --label 'prefill cancellation regression 4B final' -- python3 tests/test_prefill_cancel.py --cli build/kapitsa --reference ../reference-kapitsa --model /Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit --output ../regression-4b-final --port 8298
