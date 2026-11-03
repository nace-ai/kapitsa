#!/bin/bash
# 32K again, warm: twice through
for i in 1 2; do for R in 1 3 4; do
  [ $R = 1 ] && K=dory_attn_partial,dory_attn_partial_k8 || K=dory_attn_partial_rows,dory_attn_partial_rows_k8
  ./dd_check dory_k8.metal $R 32768 40 $K
done; done
./dd_check dory_k8.metal 3 131072 40 dory_attn_partial_rows,dory_attn_partial_rows_k8
./dd_check dory_k8.metal 1 131072 40 dory_attn_partial,dory_attn_partial_k8
