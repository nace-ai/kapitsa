#!/bin/bash
set -euo pipefail
R=/Users/bytedance/kapitsa-json-tools-20261107
export PATH="/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin:/opt/homebrew/bin:/Users/bytedance/.local/bin:$PATH"
cd "$R/candidate"
while kill -0 8868 2>/dev/null && ! test -f "$R/evidence/candidate-complete"; do sleep 5; done
test -f "$R/evidence/candidate-complete"
run() {
 label=$1; shift
 printf '%q ' "$@" > "$R/evidence/$label.cmd"; printf '\n' >> "$R/evidence/$label.cmd"
 set +e
 "$@" > "$R/evidence/$label.log" 2>&1
 rc=$?
 set -e
 printf '%s\n' "$rc" > "$R/evidence/$label.rc"
 printf '%s %s\n' "$label" "$rc" >> "$R/evidence/results.txt"
}
model=/Users/bytedance/models/Qwen3.6-35B-A3B-4bit
run parser-35b build/parser_test "$model"
run structured-35b python3 tests/test_structured.py --cli build "$model"
shasum -a 256 "$model/tokenizer.json" /Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit/tokenizer.json > "$R/evidence/control-tokenizers.txt"
touch "$R/evidence/controls-complete"
