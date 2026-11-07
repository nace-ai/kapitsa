#!/bin/bash
set -u
root=/Users/bytedance/kapitsa-stop-sequence-20261107
py=/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin/python
for arm in base candidate; do
  cd "$root/$arm" || exit 1
  "$py" -u "$root/update-full-diagnostic.py" tests/test_update.py build > "$root/evidence/update-full-diagnostic-0815-$arm-pair.log" 2>&1
  result=$?
  echo "$arm $result" >> "$root/evidence/update-full-diagnostic-0815-pair-results.txt"
done
