#!/bin/zsh
# sweep_engine.sh ENGINE SIZES: one engine's server with a long context, ctx_sweep.py against it, then stop. Run it
# under the GPU lock (lockf -k ~/.cache/nace-gpu.lock ...). ENGINE: kapitsa | rec | llamacpp
SP=/private/tmp/claude-501/-Users-flaminis-Desktop-kapitsa--claude-worktrees-kapitsa-flint-compatibility-592a31/87805e1d-d28b-43fa-9728-1d6a94aafce7/scratchpad
R=/Users/flaminis/Desktop/kapitsa/.claude/worktrees/kapitsa-flint-compatibility-592a31
M=$HOME/.cache/kapitsa/models/flint-4bit
E=$1; SIZES=$2
case $E in
  kapitsa)
    NI_DISK_CACHE=0 $R/build/kapitsa serve --model $M --model-id flint --host 127.0.0.1 --port 8401 --ctx 262144 --slots 1 > $SP/sweep_$E.log 2>&1 &
    URL=http://127.0.0.1:8401/v1; KEY=local ;;
  rec)
    XDG_DATA_HOME=$HOME/.cache/kbr $SP/rec-bin/rec-darwin-arm64 serve --port 11511 --context 262144 > $SP/sweep_$E.log 2>&1 &
    URL=http://127.0.0.1:11511/v1; KEY=$(cat $HOME/.cache/kbr/rec/config/api-key) ;;
  llamacpp)
    $SP/llamacpp/b11804/llama-b11804/llama-server -m $HOME/.cache/kapitsa/models/flint-int4-gguf/flint-int4-q4_1.gguf \
      --host 127.0.0.1 --port 8411 -a flint -c 140000 -np 1 -ngl 999 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 -lm none \
      --jinja --cache-ram 0 --temp 1 --top-p 0.95 --top-k 20 --min-p 0 --no-webui > $SP/sweep_$E.log 2>&1 &
    URL=http://127.0.0.1:8411/v1; KEY=local ;;
esac
PID=$!
for i in {1..900}; do curl -sf -m 2 -H "Authorization: Bearer $KEY" $URL/models > /dev/null 2>&1 && break; sleep 1; done
echo "$E ready after ${i}s"
# a warm-up: a short prompt and a few tokens (kernel compilation, first allocations)
$SP/venv-ref/bin/python -I $SP/scripts/ctx_sweep.py $URL $KEY $SP/sweep_${E}_warm.json $M/tokenizer.json $R 2048 > /dev/null 2>&1
$SP/venv-ref/bin/python -I $SP/scripts/ctx_sweep.py $URL $KEY $SP/sweep_$E.json $M/tokenizer.json $R $SIZES
kill $PID; wait $PID 2>/dev/null
echo done
