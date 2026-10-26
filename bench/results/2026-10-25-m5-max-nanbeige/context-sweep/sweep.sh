#!/bin/bash
# Prefill and decode from 1K to 128K tokens: Kapitsa (flint_bench), mlx-lm (mlx_lm.benchmark), llama.cpp (llama-bench)
# on the same 4-bit weights (MLX pack; llama.cpp's Q4_K_M), interleaved per length, 20 s of rest between runs
# Run from the repository root: KBUILD = a Kapitsa build folder (flint_bench), PY = a Python with mlx-lm 0.32.2, LB = llama.cpp
# b11871's binaries. TAG=8bit PACKDIR=~/models/nanbeige4.3-33b-8bit GGUF=Q8_0 LENS="1024 4096 16384 32768" ran the 8-bit table.
B=${KBUILD:-build}; M=${PACKDIR:-$HOME/models/nanbeige4.3-33b-4bit}; G=$HOME/models/nanbeige4.3-33b-gguf/Nanbeige4.3-33B-${GGUF:-Q4_K_M}.gguf
PY=${PY:-python3}; LB=${LB:-$HOME/engines/llama-b11871}; O=${OUT:-sweep}/${TAG:-4bit}
mkdir -p $O
H=bench/harness
# 20 s of rest, then until no other process holds more than 5% of the GPU for 15 s (the harness's gpu_hogs check)
rest() { t=0; while [ $t -lt 20 ]; do sleep 1; t=$((t+1)); done; (cd $H && python3 gpu_hogs.py wait 5 15 1800) | tail -1; }
rest
for n in ${LENS:-1024 4096 16384 32768 65536 131072}; do
  echo "== $n $(date +%H:%M)"
  $B/flint_bench $M --prompt $n --steps 64 --reps 1 > $O/kapitsa-$n.txt 2>&1; grep "^rep" $O/kapitsa-$n.txt; rest
  HF_HUB_OFFLINE=1 $PY -m mlx_lm.benchmark --model $M -p $n -g 64 -n 1 > $O/mlxlm-$n.txt 2>&1; grep "Averages" $O/mlxlm-$n.txt; rest
  $LB/llama-bench -m $G -ngl 99 -fa 1 -b 2048 -ub 2048 -p $n -n 0 -r 1 -o md > $O/llamacpp-pp-$n.txt 2>&1; grep "pp$n" $O/llamacpp-pp-$n.txt; rest
  $LB/llama-bench -m $G -ngl 99 -fa 1 -b 2048 -ub 2048 -p 0 -n 64 -d $n -r 1 -o md > $O/llamacpp-tg-$n.txt 2>&1; grep "tg64" $O/llamacpp-tg-$n.txt; rest
done
echo "sweep done $(date +%H:%M)"
