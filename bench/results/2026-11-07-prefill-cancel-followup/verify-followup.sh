#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
ROOT=/Users/bytedance/kapitsa-cancel-checkpoint-20261107
OUT=$ROOT/followup-133912
trap 'echo $? > "$OUT/driver.rc"' EXIT
export VK_RUN=cancel-checkpoint-133912 VK_EVIDENCE=$OUT/evidence
printf '%s\n' 'base build' > "$OUT/stage"
cd "$ROOT/reference"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$OUT/configure-base.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build kapitsa logit_dump ni_run update_test
cd "$ROOT/candidate"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > "$OUT/configure-candidate.log" 2>&1
.dsh/skills/verify-kapitsa/scripts/vk build kapitsa logit_dump ni_run
build/kapitsa --version > "$OUT/candidate-version.txt"
../reference/build/kapitsa --version > "$OUT/base-version.txt"
touch "$OUT/build-ready"
printf '%s\n' '35B cancellation regression' > "$OUT/stage"
set +e
.dsh/skills/verify-kapitsa/scripts/vk gpu run --eta 5 --label '#1051 35B cancellation regression' -- python3 tests/test_prefill_cancel.py --cli build/kapitsa --reference ../reference/build/kapitsa --model /Users/bytedance/models/Qwen3.6-35B-A3B-4bit --output "$OUT/cancel-35b" --port 8298 > "$OUT/cancel-35b.log" 2>&1
rc=$?
echo "$rc" > "$OUT/cancel-35b.rc"
set -e
printf '%s\n' 'source-specific updater diagnostics' > "$OUT/stage"
for tag in reference candidate; do
  set +e
  python3 "$OUT/update-full-diagnostic.py" "$ROOT/$tag/tests/test_update.py" "$ROOT/$tag/build" > "$OUT/update-$tag.log" 2>&1
  echo $? > "$OUT/update-$tag.rc"
  set -e
done
printf '%s\n' 'full GPU gates' > "$OUT/stage"
python3 "$OUT/run_checks.py" "$OUT/gpu-config.json" > "$OUT/gpu-driver.log" 2>&1
printf '%s\n' 'completed' > "$OUT/stage"
cd "$ROOT/candidate"
.dsh/skills/verify-kapitsa/scripts/vk cleanup
