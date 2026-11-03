#!/bin/bash
for L in 32768 262144; do
  ./fd_check flint_main.metal 1 $L 40 flint_attn_partial
  for R in 2 4 8; do ./fd_check flint_main.metal $R $L 40 flint_attn_partial_rows,flint_attn_partial; done
done
