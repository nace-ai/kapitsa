#!/usr/bin/env bash
set -euo pipefail
export PATH=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/usr/local/bin:$PATH
OUT=/Users/bytedance/kapitsa-cancel-checkpoint-20261107/followup-133912
trap 'echo $? > "$OUT/state-cases-v3.rc"' EXIT
python3 "$OUT/state_cases_v3.py" "$OUT/state-config-v3.json" > "$OUT/state-cases-v3.log" 2>&1
