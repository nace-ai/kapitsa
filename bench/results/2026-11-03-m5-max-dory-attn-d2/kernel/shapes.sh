#!/bin/bash
# dory_attn_td2_g4s8 against dory_attn_t_g4s8k16: ragged T, full layers, sliding windows over a ring and over a full cache
K=dory_attn_t_g4s8k16,dory_attn_td2_g4s8
echo "-- full layer, ragged T"
for T in 37 64 1000 2047 4096; do ./da_check tensor_dd.metal $K $T 0,1,333,8192; done
echo "-- sliding window 1024, a ring of 1024 slots"
for T in 64 1000 2048; do WINDOW=1024 RING=1024 ./da_check tensor_dd.metal $K $T 0,500,1024,5000,100000; done
echo "-- window 1024, no ring"
WINDOW=1024 ./da_check tensor_dd.metal $K 2048 0,3000,60000
