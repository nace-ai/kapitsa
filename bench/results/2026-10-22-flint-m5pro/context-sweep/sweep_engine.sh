#!/bin/zsh
# sweep_engine.sh ENGINE SIZES: one engine's server with a long context, ctx_sweep.py against it, then stop. Run it
# under the GPU lock (lockf -k /tmp/kapitsa-gpu.lock ...). ENGINE: kapitsa | rec | llamacpp
# As the M4 Pro's (bench/results/2026-10-22-flint-m4pro/context-sweep/sweep_engine.sh), plus a foreign-GPU-load
# monitor (gpu_hogs.py monitor) over the server's process tree for the whole sweep.
SP=<scratch>
R=<repo>
SRC=$SP/sweep-src            # bench/corpus.txt, src/, tools/, docs/ of 6c13588 (git archive), the prompt's source
M=$HOME/Documents/nace/models/flint-4bit
G=$HOME/Documents/nace/models/flint-int4-gguf/flint-int4-q4_1.gguf
TOK=$SP/flint-tokenizer.json # Flint's tokenizer.json (a copy, so the sweep does not need the MLX folder)
E=$1; SIZES=$2
mkdir -p $SP/sweep
case $E in
  kapitsa)
    NI_DISK_CACHE=0 $R/build/kapitsa serve --model $M --model-id flint --host 127.0.0.1 --port 8401 --ctx 262144 --slots 1 > $SP/sweep/$E$TAG.log 2>&1 &
    URL=http://127.0.0.1:8401/v1; KEY=local ;;
  rec)
    XDG_DATA_HOME=$HOME/.cache/kbr $SP/rec-bin/rec-darwin-arm64 serve --port 11511 --context 262144 > $SP/sweep/$E$TAG.log 2>&1 &
    URL=http://127.0.0.1:11511/v1; KEY=$(cat $HOME/.cache/kbr/rec/config/api-key) ;;
  llamacpp)
    $SP/llamacpp/b11804/llama-b11804/llama-server -m $G \
      --host 127.0.0.1 --port 8411 -a flint -c 140000 -np 1 -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -lm none \
      --jinja --cache-ram 0 --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --no-webui > $SP/sweep/$E$TAG.log 2>&1 &
    URL=http://127.0.0.1:8411/v1; KEY=local ;;
esac
PID=$!
python3 $R/bench/harness/gpu_hogs.py monitor $SP/sweep/${E}${TAG}_foreign_gpu.txt $PID $$ > /dev/null 2>&1 &
MON=$!
for i in {1..1800}; do curl -sf -m 2 -H "Authorization: Bearer $KEY" $URL/models > /dev/null 2>&1 && break; sleep 1; done
echo "$E ready after ${i}s"
vm_stat > $SP/sweep/${E}${TAG}_vmstat_start.txt; sysctl vm.swapusage >> $SP/sweep/${E}${TAG}_vmstat_start.txt
# a warm-up: a short prompt and a few tokens (kernel compilation, first allocations)
$SP/venv/bin/python -I $SP/ctx_sweep.py $URL $KEY $SP/sweep/${E}${TAG}_warm.json $TOK $SRC 2048 > /dev/null 2>&1
$SP/venv/bin/python -I $SP/ctx_sweep.py $URL $KEY $SP/sweep/$E$TAG.json $TOK $SRC $SIZES
if [[ -n $SKIP128 ]]; then
  # llama.cpp only (the step below the 30 GiB disk floor): 128K on the same server, and only if memory free >= 40% and
  # swap used <= 9 GB right now (the orchestrator's rule)
  mf=$(memory_pressure -Q | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
  sw=$(sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9.]*\)M.*/\1/p' | cut -d. -f1)
  echo "$(date +%T) before 128K: memory free ${mf}%, swap used ${sw} MB" | tee $SP/sweep/${E}${TAG}_128k_check.txt
  if [[ $mf -ge 40 && $sw -le 9216 ]]; then
    $SP/venv/bin/python -I $SP/ctx_sweep.py $URL $KEY $SP/sweep/${E}${TAG}_128k.json $TOK $SRC 131072
    python3 -c "import json;a=json.load(open('$SP/sweep/$E$TAG.json'));a+=json.load(open('$SP/sweep/${E}${TAG}_128k.json'));json.dump(a,open('$SP/sweep/$E$TAG.json','w'),indent=1)"
  else
    echo "128K skipped: memory free ${mf}% (< 40?) or swap ${sw} MB (> 9216?)" | tee -a $SP/sweep/${E}${TAG}_128k_check.txt
  fi
fi
vm_stat > $SP/sweep/${E}${TAG}_vmstat_end.txt; sysctl vm.swapusage >> $SP/sweep/${E}${TAG}_vmstat_end.txt
kill $PID; wait $PID 2>/dev/null
kill $MON 2>/dev/null
echo done
