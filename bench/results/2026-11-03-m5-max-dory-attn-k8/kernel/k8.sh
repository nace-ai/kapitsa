#!/bin/bash
# the branch's kernels against the shipped ones: one row and a 3- and 4-row verify, 32K and 262K positions
for L in 32768 262144; do
  ./dd_check dory_k8.metal 1 $L 40 dory_attn_partial,dory_attn_partial_k8
  ./dd_check dory_k8.metal 3 $L 40 dory_attn_partial_rows,dory_attn_partial_rows_k8
  ./dd_check dory_k8.metal 4 $L 40 dory_attn_partial_rows,dory_attn_partial_rows_k8
done
