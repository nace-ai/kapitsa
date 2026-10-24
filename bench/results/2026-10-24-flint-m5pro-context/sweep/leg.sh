#!/bin/zsh
# leg.sh ENGINE LEG PLAN: one GPU leg (<= 10 min): start ENGINE's server, a 2K warm-up, ctx_sweep.py over PLAN (sizes in
# order, cut to the leg budget), stop. Run under the GPU lock:
#   detach.py run --name <job> --cwd /tmp/flint-ctx -- lockf -k /tmp/kapitsa-gpu.lock zsh leg.sh ENGINE LEG PLAN
# ENGINE: kapitsa | llamacpp. Exits 75 without loading if /tmp/drex-wants-gpu exists or memory is short.
# A watchdog stops the server if swap use grows more than 256 MB over the leg's start.
SP=/tmp/flint-ctx
R=/tmp/kap-flint-ctx                       # kapitsa 71d68916, Release build
SRC=$SP/src                                # bench/corpus.txt (dereferenced), src/, tools/, docs/ of 71d68916
TOK=$SP/flint-tokenizer.json               # RecursiveSI/flint-int4 @3b93f905 tokenizer.json
M=$HOME/Documents/nace/models/flint-4bit   # RecursiveSI/flint-int4 @3b93f905 (MLX 4-bit)
G="/Volumes/Extreme Pro/downloads/flint-int4-gguf/flint-int4-q4_1.gguf"  # RecursiveSI/flint-int4-gguf @9c471510 (Q4_1),
# sha256 43c07bac...; read whole into memory at load (--no-mmap), so no timed request reads weights from a disk
LL=$SP/llama-b11840/llama-server
E=$1; LEG=$2; PLAN=$3
OUT=$SP/runs/$E-$LEG
mkdir -p $OUT
exec > >(tee -a $OUT/leg.log) 2>&1
echo "$(date '+%F %T') leg $E $LEG plan $PLAN"
[[ -e /tmp/drex-wants-gpu ]] && { echo "drex-wants-gpu present: not starting"; exit 75; }
mf=$(memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
SW0=$(sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1)
echo "memory free ${mf}%, swap used ${SW0} MB, disk free $(df -g /System/Volumes/Data | awk 'NR==2{print $4}') GiB"
[[ $mf -lt 40 ]] && { echo "memory free < 40%: not starting"; exit 75; }
# any other process over ~2 GB RSS (the agent host, `node ... bin.ts web`, always runs and is not counted)
big=$(ps -ax -o rss=,pid=,command= | awk '$1 > 2000000 && !/bin\.ts web/' | cut -c1-120)
[[ -n $big ]] && { echo "process over 2 GB RSS running, not starting: $big"; exit 75; }
LEG_START=$(python3 -c 'import time;print(time.time())')
case $E in
  kapitsa)
    NI_DISK_CACHE=0 $R/build/kapitsa serve --model $M --model-id flint --host 127.0.0.1 --port 8401 --ctx 262144 \
      --slots 1 > $OUT/server.log 2>&1 &
    URL=http://127.0.0.1:8401/v1 ;;
  llamacpp)
    [[ -f "$G" ]] || { echo "GGUF missing: $G"; exit 75; }
    $LL -m "$G" --host 127.0.0.1 --port 8411 -a flint -c 140000 -np 1 -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
      -lm none -lzm off --jinja --cache-ram 0 --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --no-webui > $OUT/server.log 2>&1 &
    URL=http://127.0.0.1:8411/v1 ;;
  *) echo "unknown engine $E"; exit 2 ;;
esac
PID=$!
python3 $R/bench/harness/gpu_hogs.py monitor $OUT/foreign_gpu.txt $PID $$ > /dev/null 2>&1 &
MON=$!
( while sleep 3; do s=$(sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1)
    if (( s > SW0 + 256 )); then echo "$(date +%T) SWAP GREW: $SW0 -> $s MB, stopping the server" | tee $OUT/SWAP_ABORT; kill $PID; exit; fi
  done ) &
WD=$!
for i in {1..300}; do kill -0 $PID 2>/dev/null || { echo "$(date +%T) server exited during load:"; tail -3 $OUT/server.log; kill $MON $WD 2>/dev/null; exit 3; }; curl -sf -m 2 -H "Authorization: Bearer local" $URL/models > /dev/null 2>&1 && break; sleep 1; done
echo "$(date +%T) $E ready after ${i}s"
{ vm_stat; sysctl vm.swapusage; } > $OUT/vmstat_start.txt
$SP/venv/bin/python $SP/ctx_sweep.py --url $URL --key local --out $OUT/warm.json --tok $TOK --src $SRC --plan 2048 \
  --max-tokens 32 --engine $E --leg $LEG-warm > /dev/null 2>&1
$SP/venv/bin/python $SP/ctx_sweep.py --url $URL --key local --out $OUT/runs.json --tok $TOK --src $SRC --plan $PLAN \
  --leg-start $LEG_START --budget-s 570 --foreign $OUT/foreign_gpu.txt --engine $E --leg $LEG
rc=$?
{ vm_stat; sysctl vm.swapusage; } > $OUT/vmstat_end.txt
footprint $PID 2>/dev/null | head -5 > $OUT/footprint_end.txt
kill $PID; wait $PID 2>/dev/null
kill $MON $WD 2>/dev/null
echo "$(date '+%F %T') end rc=$rc, leg $(python3 -c "import time;print(round(time.time()-$LEG_START))")s, $(sysctl -n vm.swapusage)"
