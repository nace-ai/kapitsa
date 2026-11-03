#!/bin/zsh
# the 27B's 8-row tree verify on attn8t_partial_tree_g6 (the tree path, one row tile) instead of the zone path
# (attn8m_partial_zone): the cycle at ~131K (ni_run's 8-row block), the output check against NI_V8_TREE_G6=0, and the
# verify under the validation layer (the prefill on attn_prefill11r there: qattn_t_g6d2s8's 24 KB read as 48 KB)
cd <repo>
M=~/models/qwen3.8-27b-4bit-mtp
echo "== 8-row cycle trace, tree path" && date
NI_LOCK_HELD=1 NI_CYCLE_TRACE=1 <run-root>/bin/ni_run-t6 $M @<run-root>/ids27_133k.txt 256 0 --dflash auto --ignore-eos 2>&1 | grep -E "prompt|KFlash: [0-9]+ cycles|cycle trace|drafting pass" | cut -c1-400
echo "== output check" && date
NI_LOCK_HELD=1 NI_V8_TREE_G6=0 <run-root>/bin/logit_dump-t6 $M <run-root>/oc27t/old --paths step,verify 2>&1 | tail -3
NI_LOCK_HELD=1 <run-root>/bin/logit_dump-t6 $M <run-root>/oc27t/new --paths step,verify 2>&1 | tail -3
python3 tools/logit_compare.py <run-root>/oc27t/old <run-root>/oc27t/new --floor tools/logit_floors/qwen3.8-27b-4bit-mtp.m5.json --json <run-root>/oc27t/compare.json
echo "== validation layer" && date
NI_LOCK_HELD=1 NI_QATTN_G6=0 NI_ANE=0 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 <run-root>/bin/logit_dump-t6 $M <run-root>/oc27t/val --paths verify --windows 2 2>&1 | tail -4
echo "exit $?"
date
