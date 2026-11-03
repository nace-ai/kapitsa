#!/bin/bash
# the 27B's wide tree verify, attn8m_partial_tree (NI_V8_ATTN_T=0) against attn8t_partial_tree_g6 (default), one binary
set -e
WT=$1; D=$2; M=$HOME/models/qwen3.8-27b-4bit-mtp
export NI_LOCK_HELD=1
for R in 16 32; do
  NI_V8_ATTN_T=0 $WT/build/logit_dump $M $D/ref$R --paths verify --verify-rows $R --rows 32 2>&1 | tail -3
  $WT/build/logit_dump $M $D/new$R --paths verify --verify-rows $R --rows 32 2>&1 | tail -3
done
echo "=== done"
