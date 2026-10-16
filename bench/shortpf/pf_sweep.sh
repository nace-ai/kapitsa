#!/bin/bash
# usage: pf_sweep.sh <ni_run> <model_dir> <spec 0|D> [N...]  -- prefill-only timings of the first N ids of p700
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$1"; MODEL="$2"; SPEC="$3"; shift 3
NS="${@:-8 16 32 64}"
for N in $NS; do
  if [ "$SPEC" != "0" ]; then A="--spec $SPEC"; else A=""; fi
  "$RUN" "$MODEL" @"$HERE/p700_$N.txt" 1 0 --pf-reps 6 $A 2>&1 | grep "prefill rep" | sed "s/^/N=$N spec=$SPEC /"
done
