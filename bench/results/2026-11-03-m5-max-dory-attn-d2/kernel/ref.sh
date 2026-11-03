#!/bin/bash
K=dory_attn_t_g4s8k16,dory_attn_td2_g4s8
./da_check tensor_dd.metal $K 2047 1,333,8192,32768
WINDOW=1024 RING=1024 ./da_check tensor_dd.metal $K 1000 0,5000
