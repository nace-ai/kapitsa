#!/bin/zsh
# Dory's decode attention 8 keys at a time: the output check, the continuations, the interleaved decode A/B
D=<run-root>/ab11-output-check
mkdir -p $D
M=$HOME/models/nemotron-labs-dory1.0-4b-nvfp4
cd <repo>
for arm in ref new; do
  [ $arm = ref ] && export NI_DORY_ATTN_K8=0 || unset NI_DORY_ATTN_K8
  caffeinate -is lockf -k -t 14400 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump $M $D/$arm --paths step,verify --loop-kv per-loop 2>&1 | tail -4
done
unset NI_DORY_ATTN_K8
python3 tools/logit_compare.py $D/ref $D/new > $D/compare.txt 2>&1; cat $D/compare.txt
python3 tools/logit_compare.py --reorder $D/ref $D/new > $D/compare-reorder.txt 2>&1; tail -3 $D/compare-reorder.txt
caffeinate -is lockf -k -t 14400 /tmp/kapitsa-gpu.lock python3 <scratch>/dory/greedy8/greedy_check.py <repo>/build/kapitsa $M <run-root>/bench_corpus_long.txt <scratch>/dory/greedy8/greedy.json 2>&1
<run-root>/run_ab11.sh dory-k8-old dory-k8-new dory-k8-old dory-k8-new dory-k8-new-256 dory-k8-old-256
