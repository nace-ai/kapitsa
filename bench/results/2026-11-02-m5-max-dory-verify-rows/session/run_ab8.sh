#!/bin/zsh
# run.sh LABELS...: the long-context sweep, one run per label, under the harness's GPU lock
R=<run-root>
cd <main>/bench/harness
C=(configs/kapitsa.json configs/clover.json configs/flint.json configs/lightning.json configs/bolt.json configs/flash-next.json configs/super-vl.json configs/dory.json $R/long.json $R/ab8.json)
for l in "$@"; do
  while pmset -g batt | grep -q "Battery Power"; do sleep 30; done   # AC only: wait for the charger
  echo "=== $l $(date +%H:%M:%S) powermode=$(pmset -g | awk '/powermode/{print $2}') $(pmset -g batt | head -1) load=$(sysctl -n vm.loadavg) top=$(ps -Ao pcpu,comm -r | sed -n 2,4p | awk "{print \$1, \$NF}" | tr "\n" ";")"
  caffeinate -is python3 compare_engines.py $C --only "$l" --agentic --skip greedy,prefill,tools,agent,cache,sustain --quiet-timeout 60 --out-dir $R/ab8 2>&1 | grep -E 'flag|FLAG|INVALID|invalid|error|Error|result|decode|sweep|quiet|warm'
done
echo "=== done $(date +%H:%M:%S)"
