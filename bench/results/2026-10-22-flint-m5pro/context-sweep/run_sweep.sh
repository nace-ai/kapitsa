#!/bin/zsh
# run_sweep.sh ENGINE STAGE: run under `lockf -k /tmp/kapitsa-gpu.lock`; waits for >= 40% free memory, then sweep_engine.sh
SP=<scratch>
E=$1; S=$2; SIZES=${3:-1024,2048,4096,8192,16384,32768,65536,131072}
st() { python3 $SP/stage.py "$@"; }
memfree() { memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'; }
for i in {1..120}; do
  m=$(memfree); [[ $m -ge 40 ]] && break
  [[ $i == 1 ]] && st set $S waiting "lock held; waiting for >= 40% free memory (now $m%)"
  sleep 15
done
if [[ $m -lt 40 ]]; then st set $S failed "memory never reached 40% free (last $m%)"; exit 3; fi
st set $S running "sweep $SIZES, memory free $m% at start, $(sysctl -n vm.swapusage | awk '{print "swap used " $6}')"
$SP/sweep_engine.sh $E $SIZES
rc=$?
n=$(python3 -c "import json;print(len(json.load(open('$SP/sweep/$E$TAG.json'))))" 2>/dev/null)
st set $S $([[ $rc == 0 && -n $n ]] && echo done || echo failed) "exit $rc; $n points; $(sysctl -n vm.swapusage | awk '{print "swap used " $6}')"
