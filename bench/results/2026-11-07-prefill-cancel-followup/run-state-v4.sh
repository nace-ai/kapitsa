#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
OUT=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/followup-133912
trap 'echo $? > "$OUT/state-cases-v4.rc"' EXIT
python3 "$OUT/state_cases_v4.py" "$OUT/state-config-v4.json" > "$OUT/state-cases-v4.log" 2>&1
