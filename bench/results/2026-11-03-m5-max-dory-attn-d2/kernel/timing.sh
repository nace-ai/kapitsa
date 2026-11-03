#!/bin/bash
# one full-attention layer of Dory's heads, 2,048 rows after 14K / 63K / 129K / 254K positions, best of 3
./da_check tensor_dd.metal dory_attn_t_g4s8k16,dory_attn_td2_g4s8 2048 14336,63488,129024,253952
