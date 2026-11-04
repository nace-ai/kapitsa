#!/bin/zsh
# the 27B's 8-row tree verify on attn8t_partial_zone_g6: its cycle at ~131K (ni_run's 8-row block), the output check
# against NI_V8_ATTN_T=0 (attn8m_partial_zone, main's path), and logit_dump's verify under the validation layer
cd <repo>
M=~/models/qwen3.8-27b-4bit-mtp
echo "== 8-row cycle trace, zone_g6" && date
NI_LOCK_HELD=1 NI_CYCLE_TRACE=1 <run-root>/bin/ni_run-z6 $M @<run-root>/ids27_133k.txt 256 0 --dflash auto --ignore-eos 2>&1 | grep -E "prompt|KFlash: [0-9]+ cycles|cycle trace|drafting pass" | cut -c1-400
echo "== output check" && date
NI_LOCK_HELD=1 NI_V8_ATTN_T=0 <run-root>/bin/logit_dump-z6 $M <run-root>/oc27/old --paths step,verify 2>&1 | tail -4
NI_LOCK_HELD=1 <run-root>/bin/logit_dump-z6 $M <run-root>/oc27/new --paths step,verify 2>&1 | tail -4
python3 tools/logit_compare.py <run-root>/oc27/old <run-root>/oc27/new --floor tools/logit_floors/qwen3.8-27b-4bit-mtp.m5.json --json <run-root>/oc27/compare.json
echo "== validation layer" && date
NI_LOCK_HELD=1 MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 <run-root>/bin/logit_dump-z6 $M <run-root>/oc27/val --paths verify --windows 2 2>&1 | tail -6
date
