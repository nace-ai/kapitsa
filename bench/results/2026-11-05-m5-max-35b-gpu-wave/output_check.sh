#!/bin/zsh
# The output check (run_ab.sh's last step, rerun after building logit_dump in the old arm's tree): logit_dump old ->
# ref at its default chunks, new --chunk 8192 --from ref, logit_compare.py with the 35B-A3B's M5 floor.
S=${S:?}; R=$S/kapitsa-cells3; N=$S/kapitsa-wv; D=$N/bench/results/2026-11-05-m5-max-35b-gpu-wave
m=qwen3.6-35b-a3b-4bit
rm -rf $S/wv-oc/$m-ref $S/wv-oc/$m-new
(cd $R && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/wv-oc/$m-ref > $D/output-check/$m-ref.out 2>&1)
(cd $N && lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 build/logit_dump ~/models/$m $S/wv-oc/$m-new --from $S/wv-oc/$m-ref --chunk 8192 > $D/output-check/$m-new.out 2>&1)
python3 $N/tools/logit_compare.py $S/wv-oc/$m-ref $S/wv-oc/$m-new > $D/output-check/$m-compare.out 2>&1; echo "compare exit $?"
