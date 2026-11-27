#!/bin/bash
# d27 strict rerun: legs 1-6 of runs2 were all voided by foreign kapitsa processes (CPU test suites, 12 Nov 19:54-20:11).
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 PATH=/Users/alisheramantay/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
K=/Users/alisheramantay/Documents/nace/k495
for _ in $(seq 1 40); do m=$(memory_pressure | tail -1 | grep -o '[0-9]*%' | tr -d %); [ "${m:-0}" -ge 40 ] && break; sleep 30; done
BIN=$K/kapitsa-0.2.35/kapitsa ORDER="A B B A" OUT=$K/d27/runs3 MODEL_A=$K/d27/models/qwen3.8-27b-4bit MODEL_B=$K/d27/models/qwen3.8-27b-kquant bash $K/legs2.sh
