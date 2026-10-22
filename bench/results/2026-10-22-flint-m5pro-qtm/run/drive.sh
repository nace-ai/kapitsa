#!/bin/bash
R=<scratch>
for leg in 1 2; do
  while :; do
    lockf -k -t 10800 /tmp/kapitsa-gpu.lock $R/inner.sh $leg; rc=$?
    [ $rc -eq 75 ] || break; sleep 60
  done
  echo "$(date +%T) leg $leg done rc=$rc"
done
echo "$(date +%T) all legs done"
