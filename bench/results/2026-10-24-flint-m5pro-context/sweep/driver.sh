#!/bin/zsh
# driver.sh: the llama.cpp legs, one after another, after kap-rv98b is done (orchestrator's GPU freeze).
# Each leg takes the GPU lock itself; exit 75 (drex-wants-gpu / low memory) is retried after 120 s.
SP=/tmp/flint-ctx
until grep -q '^rv98b done' ~/.agents/jobs/kap-rv98b.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) rv98b done; starting llama.cpp legs"
S=1024,2048,4096,8192,16384,32768
run() {  # LEG PLAN
  while :; do
    while [[ -e /tmp/drex-wants-gpu ]]; do sleep 30; done   # don't even queue on the lock while drex wants the GPU
    lockf -k /tmp/kapitsa-gpu.lock zsh $SP/leg.sh llamacpp $1 $2; rc=$?
    echo "$(date +%T) leg $1 rc=$rc"
    [[ $rc -eq 75 ]] || break
    sleep 120
  done
  [[ $rc -eq 0 ]] || { echo "$(date +%T) leg $1 failed (rc=$rc): driver stops"; exit $rc; }
  sleep 90   # let other sessions' legs in between
}
run A1 $S,$S,$S,$S,$S
run B1 65536,65536,65536,131072
run C1 131072,131072
n128() { python3 -c "import json,glob;print(sum(1 for p in glob.glob('$SP/runs/llamacpp-*/runs.json') for r in json.load(open(p)) if r['size']==131072 and not r.get('skipped') and r.get('prefill_tps')))"; }
for leg in C2 C3; do
  [[ $(n128) -ge 3 ]] && break
  run $leg 131072,131072
done
echo "$(date +%T) driver done, 128K runs: $(n128)"
