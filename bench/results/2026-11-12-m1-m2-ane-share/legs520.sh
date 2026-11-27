#!/bin/bash
# #520 leg driver (ec2-user on the box), one server at a time, every serve under the GPU lock.
# usage: legs520.sh PASS ARM [ARM ...]
#   PASS dec32: 8K warm, then dec32.py (greedy 400-token answers after 32K prompts at 5 corpus offsets)
#   PASS pf16:  8K warm, then the #514 short pass (3 greedy short answers + one cold 16K prompt)
#   ARM  k35def k35mlx (release 0.2.35) | k27main (origin/main build) k27pr (PR build) | s27v161
main() {
set -u
PASS=$1; shift
W=~/k520; R=$W/runs; mkdir -p $R/results $R/logs
H=~/kapitsa-0.2.35-darwin-arm64/bench/harness
REL=~/kapitsa-0.2.35-darwin-arm64/kapitsa
SPL="$HOME/Library/Application Support/Splash-M1"
CTX=${CTX:-65536}
snap() { ~/venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"; }
log() { echo "$(date -u +%T) $*" | tee -a $W/legs.log; }
arm_cmd() {
  local k=(--ctx $CTX --slots 1 --private --port 8091 --idle-unload 0)
  case $1 in
    k35def) CMD=("$REL" serve --model 35b "${k[@]}");;
    k35mlx) CMD=("$REL" serve --model "$(snap mlx-community/Qwen3.6-35B-A3B-4bit)" --dflash "$(snap incoai/Qwen3.6-35B-A3B-DFlash2)" "${k[@]}");;
    k27main) CMD=("$W/main/build/kapitsa" serve --model 27b "${k[@]}");;
    k27pr) CMD=("$W/pr/build/kapitsa" serve --model 27b "${k[@]}");;
    k27pr3) CMD=("$W/pr3/build/kapitsa" serve --model 27b "${k[@]}");;
    k4main) CMD=("$W/main/build/kapitsa" serve --model 4b "${k[@]}");;
    k4pr3) CMD=("$W/pr3/build/kapitsa" serve --model 4b "${k[@]}");;
    k27pr4) CMD=("$W/pr4/build/kapitsa" serve --model 27b "${k[@]}");;
    k4pr4) CMD=("$W/pr4/build/kapitsa" serve --model 4b "${k[@]}");;
    k27pr5) CMD=("$W/pr5/build/kapitsa" serve --model 27b "${k[@]}");;
    k4pr5) CMD=("$W/pr5/build/kapitsa" serve --model 4b "${k[@]}");;
    s27v161) CMD=("$SPL/1.6.1-m1/python/bin/python3" -u "$SPL/1.6.1-m1/install/launcher.py" serve --model mlx-community/Qwen3.8-27B-4bit --host 127.0.0.1 --port 8151 --no-webui --max-context $((CTX/1024))K --language-only --idle-release off);;
    *) echo "unknown arm $1"; exit 2;;
  esac
  case $1 in k*) PORT=8091;; s*) PORT=8151;; esac
  URL=http://127.0.0.1:$PORT/v1
}
quiet() {
  for i in $(seq 1 60); do l=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v l=$l 'BEGIN{exit !(l<2.5)}' && break; sleep 5; done
  log "  quiet: load $(sysctl -n vm.loadavg)"
}
leg() {
  local arm=$1 n=$2 tag
  tag=$PASS-$arm-$(date -u +%H%M%S)-$n
  arm_cmd $arm
  if lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; then log "port $PORT busy before $tag: abort"; exit 3; fi
  log "LEG $tag: ${CMD[*]}"
  /usr/bin/perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' lockf -k /tmp/kapitsa-gpu.lock "${CMD[@]}" > $R/logs/$tag.log 2>&1 < /dev/null &
  local pid=$! t0=$(date +%s) ok=0
  for i in $(seq 1 360); do
    curl -fsS -m 5 $URL/models > $W/models.json 2>/dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  local mid=$(python3 -c "import json;print(json.load(open('$W/models.json'))['data'][0]['id'])" 2>/dev/null)
  log "  ready=$ok after $(( $(date +%s) - t0 )) s, model id '$mid'"
  if [ $ok = 1 ]; then
    quiet
    local common=(--base-url $URL --model "$mid" --pid $pid --no-gpu-lock --decode-tokens 400 --temperature 0 --top-k 1 --top-p 1
                  --agentic --sweep-task code --sweep-repeats 1 --max-ctx $CTX --agent-trace none)
    python3 $H/bench.py "${common[@]}" --label warm-$tag --warmup full --adaptive-warm --greedy-runs 0 --skip decode,greedy,prefill,tools,agent,cache,sustain --sweep-sizes 8000 --out $R/warm-$tag.json > $R/logs/warm-$tag.out 2>&1
    log "  warm: $(grep -E 'sweep 8000' $R/logs/warm-$tag.out | tail -1)"
    if [ $PASS = dec32 ]; then
      python3 $W/dec32.py $URL "$mid" $R/results/$tag.json ${DEC_TARGET:-32000} > $R/logs/$tag.out 2>&1
      log "  rc=$? $(python3 -c "import json;d=json.load(open('$R/results/$tag.json'));print(' '.join(f\"{r['offset']//1000}k:{r['decode_tps']:.1f}\" for r in d['runs']))" 2>&1)"
      log "  cycles: $(grep -E 'chat: prompt [0-9]{5} ' $R/logs/$tag.log | tail -5 | grep -oE '[0-9.]+ tok/cycle' | tr '\n' ' ')"
    else
      python3 $H/bench.py "${common[@]}" --label $tag --warmup none --greedy-runs 3 --skip decode,prefill,tools,agent,cache,sustain --sweep-sizes ${PF_SIZES:-700,1500,16000} --out $R/results/$tag.json > $R/logs/$tag.out 2>&1
      log "  rc=$? $(grep -E 'decode T=0|sweep [0-9]+:|INVALID' $R/logs/$tag.out | tr '\n' '|')"
      log "  ane: $(grep -E 'NI_ANE' $R/logs/$tag.log | grep -oE 'FFN split \(wavefront\): [0-9]+/[0-9]+ units[^;]*|no calibration[^;]*|unavailable \([^)]{0,60}' | head -2 | tr '\n' '|')"
    fi
  else
    log "  NOT READY: $(tail -n 5 $R/logs/$tag.log | tr '\n' '|')"
  fi
  kill -TERM -$pid 2>/dev/null; for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -KILL -$pid 2>/dev/null; sleep 3
  pgrep -fl "launcher.py serve|engine/splash|kapitsa serve" && { log "  leftover processes after $tag"; pkill -KILL -f "launcher.py serve|engine/splash|kapitsa serve"; sleep 3; }
}
n=0
for arm in "$@"; do n=$((n+1)); leg $arm $n; done
log "PASS $PASS done: $*"
tar czf $W/res-$PASS-$(date -u +%H%M%S).tgz -C $W legs.log runs/results runs/logs && log "packed"
}; main "$@"
