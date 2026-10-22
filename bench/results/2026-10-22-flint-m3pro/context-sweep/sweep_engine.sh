#!/bin/zsh
# sweep_engine.sh ENGINE SIZES: one engine's server with a long context, ctx_sweep.py against it, then stop. Run it
# under the GPU lock (lockf -k /tmp/kapitsa-gpu.lock ...). ENGINE: kapitsa | rec | llamacpp
# LLAMA_CTX overrides llama.cpp's -c (140000 does not fit the M3 Pro 36 GB's GPU memory).
# The M4 Pro run's script (../../2026-10-22-flint-m4pro/context-sweep/sweep_engine.sh) with the M3 Pro's paths.
SP=$HOME/flint-m3/sweep
R=$HOME/kapitsa-flint
M=$HOME/models/flint-4bit
E=$1; SIZES=$2
case $E in
  kapitsa)
    NI_DISK_CACHE=0 $R/build/kapitsa serve --model $M --model-id flint --host 127.0.0.1 --port 8401 --ctx 262144 --slots 1 > $SP/sweep_$E.log 2>&1 &
    URL=http://127.0.0.1:8401/v1; KEY=local ;;
  rec)
    XDG_DATA_HOME=$HOME/.cache/kbr $HOME/flint-m3/engines/rec/rec serve --port 11511 --context 262144 > $SP/sweep_$E.log 2>&1 &
    URL=http://127.0.0.1:11511/v1; KEY=$(cat $HOME/.cache/kbr/rec/config/api-key) ;;
  llamacpp)
    $HOME/flint-m3/engines/llama/llama-b11804/llama-server -m $HOME/models/flint-int4-gguf/flint-int4-q4_1.gguf \
      --host 127.0.0.1 --port 8411 -a flint -c ${LLAMA_CTX:-140000} -np 1 -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -lm none \
      --jinja --cache-ram 0 --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --no-webui > $SP/sweep_$E.log 2>&1 &
    URL=http://127.0.0.1:8411/v1; KEY=local ;;
esac
PID=$!
for i in {1..900}; do curl -sf -m 2 -H "Authorization: Bearer $KEY" $URL/models > /dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 1; done
echo "$E ready after ${i}s"
# a warm-up: a short prompt and a few tokens (kernel compilation, first allocations)
$SP/venv/bin/python -I $SP/ctx_sweep.py $URL $KEY $SP/sweep_${E}_warm.json $M/tokenizer.json $R 2048 > /dev/null 2>&1
$SP/venv/bin/python -I $SP/ctx_sweep.py $URL $KEY $SP/sweep_$E.json $M/tokenizer.json $R $SIZES
kill $PID; wait $PID 2>/dev/null
echo done
