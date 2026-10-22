#!/bin/zsh
# run_llamacpp.sh: run under `lockf -k /tmp/kapitsa-gpu.lock`. llama.cpp b11804 on the Q4_1 GGUF: poster pass 3
# (compare_engines flint.json --only flint-llamacpp), then its context sweep (1K-64K, then 128K on the same server if
# the machine allows it). Safeguards agreed with the orchestrator for going below the 30 GiB disk floor:
#  - df -g /System/Volumes/Data before each llama.cpp load; under 15 GiB free: stop (exit 4)
#  - the 128K point is skipped if memory free < 40% or swap used > 9 GB just before it (said so in the log)
SP=<scratch>
R=<repo>
export BENCH_MODELS=$HOME/Documents/nace/models
export LLAMA_SERVER=$SP/llamacpp/b11804/llama-b11804/llama-server
export NACE_GPU_LOCK_HELD=1
st() { python3 $SP/stage.py "$@"; }
memfree() { memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'; }
swapmb() { sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1; }
diskok() {
  local f=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
  echo "$(date +%T) disk free ${f} GiB before $1"
  st note "disk free ${f} GiB before $1"
  if [[ $f -lt 15 ]]; then st note "STOP: disk free ${f} GiB < 15 before $1"; echo "STOP disk ${f} GiB"; exit 4; fi
}
waitmem() {  # waitmem <stage>
  for i in {1..120}; do
    m=$(memfree); [[ $m -ge 40 ]] && break
    [[ $i == 1 ]] && st set $1 waiting "lock held; waiting for >= 40% free memory (now $m%)"
    sleep 15
  done
  echo "$(date +%T) memory free ${m}%, swap $(swapmb) MB"
  if [[ $m -lt 40 ]]; then st set $1 failed "memory never reached 40% free (last $m%)"; exit 3; fi
}
echo "$(date +%T) lock acquired"; st note "GPU lock acquired by run_llamacpp.sh (pid $$)"

# poster pass 3
waitmem p3l
diskok "llama.cpp poster load"
st set p3l running "compare_engines.py --only flint-llamacpp, memory free $m%, swap $(swapmb) MB"
cd $R/bench/harness
python3 compare_engines.py configs/flint.json --only flint-llamacpp --skip agent --quiet-timeout 30 --out-dir $SP/runs
rc=$?
st set p3l $([[ $rc == 0 ]] && echo done || echo failed) "exit $rc; $(ls -t $SP/runs/results/*.json | head -1 | xargs basename)"
python3 $SP/summarize.py >/dev/null 2>&1

# sweep: 1K-64K, then 128K on the same server if memory and swap allow
waitmem swl
diskok "llama.cpp sweep load"
st set swl running "sweep 1K-64K, memory free $m% at start, swap $(swapmb) MB"
SKIP128=1 $SP/sweep_engine.sh llamacpp 1024,2048,4096,8192,16384,32768,65536
rc=$?
n=$(python3 -c "import json;print(len(json.load(open('$SP/sweep/llamacpp.json'))))" 2>/dev/null)
st set swl $([[ $rc == 0 && -n $n ]] && echo done || echo failed) "exit $rc; $n points; $(sysctl -n vm.swapusage | awk '{print "swap used " $6}')"
echo "$(date +%T) llama.cpp done"
