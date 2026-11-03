#!/bin/zsh
# the output check (one binary, NI_DORY_ATTN_D2=0 against the default), then the interleaved prefill A/B
D=<run-root>/ab10-output-check
mkdir -p $D
M=$HOME/models/nemotron-labs-dory1.0-4b-nvfp4
cd <repo>
for arm in ref new; do
  [ $arm = ref ] && export NI_DORY_ATTN_D2=0 || unset NI_DORY_ATTN_D2
  caffeinate -is lockf -k -t 14400 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump $M $D/$arm --paths step,verify --loop-kv per-loop 2>&1 | tail -4
done
unset NI_DORY_ATTN_D2
python3 tools/logit_compare.py $D/ref $D/new > $D/compare.txt 2>&1; cat $D/compare.txt
python3 tools/logit_compare.py --reorder $D/ref $D/new > $D/compare-reorder.txt 2>&1; tail -3 $D/compare-reorder.txt
<run-root>/run_ab10.sh dory-d2-old dory-d2-new dory-d2-old dory-d2-new dory-d2-new-256 dory-d2-old-256
