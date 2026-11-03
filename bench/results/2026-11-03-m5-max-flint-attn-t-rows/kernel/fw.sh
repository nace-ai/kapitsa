#!/bin/bash
# full layers (ragged R, short and long prefixes) and sliding ones (window 513 over a ring of 528 slots, and over a full cache)
K=flint_attn_partial_rows,flint_attn_t_rows
for R in 2 3 5 7 8; do ./fd_check flint_main.metal tensor_fl.metal $R 37 40 $K; ./fd_check flint_main.metal tensor_fl.metal $R 5000 40 $K; done
for R in 2 5 8; do for L in 100 513 600 4000 100000; do WINDOW=513 RING=528 ./fd_check flint_main.metal tensor_fl.metal $R $L 40 $K; done; done
WINDOW=513 ./fd_check flint_main.metal tensor_fl.metal 8 3000 40 $K
