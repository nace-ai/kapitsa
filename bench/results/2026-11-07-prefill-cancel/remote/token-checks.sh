#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
cd /Users/bytedance/kapitsa-cancel-checkpoint-20261107/candidate
export VK_RUN=cancel-checkpoint-final VK_EVIDENCE=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/evidence
trap 'echo $? > ../token-checks.rc' EXIT
mkdir ../token-checks
.dsh/skills/verify-kapitsa/scripts/vk build parser_test structured_check grammar_test
for label in 4b 35b; do
  if [ "$label" = 4b ]; then model=/Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit; else model=/Users/bytedance/models/Qwen3.6-35B-A3B-4bit; fi
  build/parser_test "$model" > ../token-checks/parser-$label.log 2>&1
  python3 tests/test_structured.py --cli build "$model" > ../token-checks/structured-$label.log 2>&1
  echo "PASS parser and structured CLI $label"
done
.dsh/skills/verify-kapitsa/scripts/vk cleanup
