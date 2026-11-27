#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
OUT=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/followup-133912
trap 'echo $? > "$OUT/state-cases-v2.rc"' EXIT
python3 "$OUT/state_cases_v2.py" "$OUT/state-config-v2.json" > "$OUT/state-cases-v2.log" 2>&1
