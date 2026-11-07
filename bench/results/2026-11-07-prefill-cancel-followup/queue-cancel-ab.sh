#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
OUT=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/followup-133912
trap 'echo $? > "$OUT/cancel-ab.rc"' EXIT
while [ ! -f "$OUT/driver.rc" ]; do sleep 5; done
test "$(cat "$OUT/driver.rc")" = 0
python3 "$OUT/cancellation_ab.py" "$OUT/cancel-ab-config.json"
