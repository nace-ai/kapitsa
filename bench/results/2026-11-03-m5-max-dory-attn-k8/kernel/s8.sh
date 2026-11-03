#!/bin/bash
for L in 131072 262144; do
  ./dd_check dory_s8.metal 1 $L 40 dory_attn_partial,dory_attn_partial_s8,dory_attn_partial_k8
  ./dd_check dory_s8.metal 3 $L 40 dory_attn_partial_rows,dory_attn_partial_rows_s8,dory_attn_partial_rows_k8
  ./dd_check dory_s8.metal 4 $L 40 dory_attn_partial_rows,dory_attn_partial_rows_s8,dory_attn_partial_rows_k8
done
