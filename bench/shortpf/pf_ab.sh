#!/bin/bash
# usage: pf_ab.sh <ni_run> <model_dir> <spec 0|D> <rounds> [N...]
# Interleaved A/B of prefill-only time: A = default, B = NI_SHORTPF=1; prints the median of reps 1..5
# (rep 0 includes pipeline creation) for each run.
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$1"; MODEL="$2"; SPEC="$3"; ROUNDS="$4"; shift 4
NS="${@:-8 16 32 64}"
A=""; [ "$SPEC" != "0" ] && A="--spec $SPEC"
med() { sort -n | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
for R in $(seq 1 $ROUNDS); do
  for N in $NS; do
    for MODE in default shortpf; do
      if [ $MODE = shortpf ]; then E="NI_SHORTPF=1"; else E="NI_SHORTPF="; fi
      M=$(env $E "$RUN" "$MODEL" @"$HERE/p700_$N.txt" 1 0 --pf-reps 6 $A 2>&1 | grep "prefill rep" | grep -v "rep 0" | sed -E 's/.* in ([0-9.]+) ms.*/\1/' | med)
      echo "round=$R N=$N spec=$SPEC $MODE ${M} ms"
    done
  done
done
