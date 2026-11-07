#!/bin/bash
set -euo pipefail
R=/Users/bytedance/kapitsa-json-tools-20261107
cd "$R/candidate"
while kill -0 10634 2>/dev/null && ! test -f "$R/evidence/controls-complete"; do sleep 5; done
test -f "$R/evidence/controls-complete"
label=grammar-4b-correct
model=/Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit
test ! -e "$R/evidence/$label.rc"
printf '%q ' build/grammar_test "$model" > "$R/evidence/$label.cmd"
printf '\n' >> "$R/evidence/$label.cmd"
set +e
build/grammar_test "$model" > "$R/evidence/$label.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$R/evidence/$label.rc"
printf '%s %s\n' "$label" "$rc" >> "$R/evidence/results.txt"
touch "$R/evidence/grammar-correct-complete"
