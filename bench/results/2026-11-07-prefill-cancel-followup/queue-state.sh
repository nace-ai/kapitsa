#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
OUT=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/followup-133912
trap 'echo $? > "$OUT/state-cases.rc"' EXIT
while [ ! -f "$OUT/cancel-ab.rc" ]; do sleep 5; done
python3 "$OUT/state_cases.py" "$OUT/state-config.json" > "$OUT/state-cases.log" 2>&1
