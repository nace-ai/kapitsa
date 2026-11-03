#!/bin/zsh
# the staged tree-verify attention for the 35B: the output check (logit_dump with NI_V8T_STAGE=0 and with the default) and
# the interleaved decode A/B
cd <repo>
M=~/models/qwen3.6-35b-a3b-4bit
echo "== output check" && date
NI_LOCK_HELD=1 NI_V8T_STAGE=0 <run-root>/bin/logit_dump-qs1 $M <run-root>/oc/old --paths step,verify 2>&1 | tail -4
NI_LOCK_HELD=1 <run-root>/bin/logit_dump-qs1 $M <run-root>/oc/new --paths step,verify 2>&1 | tail -4
python3 tools/logit_compare.py <run-root>/oc/old <run-root>/oc/new --json <run-root>/oc/compare.json
echo "== A/B" && date
python3 <run-root>/ab_q35.py <run-root>/bin/kapitsa-qs1 $M ~/kapitsa-bench-runs/2026-11-02-m5-max-long/bench_corpus_long.txt <run-root>/ab1 '[{"NI_V8T_STAGE":"0"}, {}, {"NI_V8T_STAGE":"0"}, {}]'
date
