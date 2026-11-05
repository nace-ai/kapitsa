#!/bin/zsh
# 27B 4K prefill on the NI_QMT128 build, NI_ANE_UNITS swept (0 = NI_ANE=0, GPU alone), 45 s idle before each run.
S=${S:?}; B=<base>/build/ni_run
for pass in 1 2; do
  for u in 16 12 8 20 0 14; do
    sleep 45
    if [ $u = 0 ]; then E=(NI_ANE=0); else E=(NI_ANE_UNITS=$u); fi
    lockf -k -t 900 /tmp/kapitsa-gpu.lock env NI_LOCK_HELD=1 $E $B ~/models/qwen3.8-27b-4bit-mtp @$S/pf35/ids4k.txt 8 0 --pf-reps 3 --dflash none > $S/ane/u$u-$pass.out 2>&1
    echo "units $u pass $pass reps $(grep -E 'prefill rep [012]:' $S/ane/u$u-$pass.out | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | tr '\n' ' ') $(grep -o 'thermal [a-z]*' $S/ane/u$u-$pass.out | head -1)"
  done
done
