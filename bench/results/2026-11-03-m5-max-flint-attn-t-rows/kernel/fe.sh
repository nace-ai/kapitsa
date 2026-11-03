#!/bin/bash
# the kernels against a double-precision softmax (rows x heads 0, 9, 18, 31)
for L in 1000 32768; do for R in 2 8; do
  ./fd_check flint_main.metal tensor_fl.metal $R $L 40 flint_attn_partial_rows,flint_attn_t_rows,flint_attn_t_rows_q1
done; done
