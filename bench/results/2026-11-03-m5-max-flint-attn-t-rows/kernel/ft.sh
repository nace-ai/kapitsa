#!/bin/bash
# flint_attn_t_rows against flint_attn_partial_rows: one full-attention layer, the engine's 40 chunks
for L in 32768 131072 262144; do for R in 2 4 8; do
  ./fd_check flint_main.metal tensor_fl.metal $R $L 40 flint_attn_partial_rows,flint_attn_t_rows,flint_attn_t_rows_q1,flint_attn_t_rows_p
done; done
