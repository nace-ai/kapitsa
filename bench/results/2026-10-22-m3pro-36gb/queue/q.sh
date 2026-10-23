#!/bin/bash
# M3 Pro step 2: Kapitsa 0.2.8 (catalog defaults) on every fitting model, then the M4 Pro poster setups of the rivals.
# The Ollama pull of the 35B/27B GGUF tags (only the llama.cpp cells of those two read them) is stopped while anything
# is measured and resumed after phase B, so no run shares the Mac with a download.
. ~/kenv.sh; . ~/kq/m3/cfg.sh; cd ~/kq/rel/bench/harness
L=~/kq/m3/q.log
echo "$(date +%T) waiting for DL-DONE (dl3) and B318-DONE (step-3 builds)" >> $L
until sed -n "/dl3 start/,\$p" ~/kq/dl.log | grep -q DL-DONE && grep -q B318-DONE ~/kq/s3/build.log; do sleep 30; done
pgrep -f "curl|ollama|cmake|clang" >> $L && echo "$(date +%T) WARNING busy processes before phase A" >> $L
sleep 30
step() {
  pmset displaysleepnow; sleep 5
  while :; do
    fr=$(memory_pressure | tail -1 | grep -o "[0-9]*" | tail -1); dk=$(df -g ~ | tail -1 | awk '{print $4}')
    [ "$fr" -ge 40 ] && [ "$dk" -ge 30 ] && break
    echo "$(date +%T) wait: free ${fr}% disk ${dk}G" >> $L; sleep 60
  done
  vt=$(pgrep -x VTDecoderXPCService >/dev/null && ps -o %cpu= -p "$(pgrep -x VTDecoderXPCService | head -1)" || echo none)
  echo "$(date +%T) power $(pmset -g batt | head -1 | sed 's/Now drawing from //') free ${fr}% disk ${dk}G vtdecoder $vt :: $1" >> $L
  python3 compare_engines.py $C --skip agent --only "$1" >> $L 2>&1; echo "$(date +%T) rc=$? :: $1" >> $L
  pgrep -fl "llama-server|mlx_lm.server|mlx_vlm|bin/mtplx|ollama serve|kapitsa serve" >> $L && echo "$(date +%T) WARNING leftover process" >> $L
}
for m in q35-4b q38-27b q36-35b q35-9b dory nano-4b nano-9b lightning bolt; do step sw-$m-kapitsa; done
echo "$(date +%T) PHASE-A-DONE" >> $L
for x in ${RIV4//,/ } ${RIV27//,/ } ${RIV35//,/ } ${RIVO//,/ }; do
  case $x in bolt-mlxvlm|q36-35b-llamacpp-mtp1|q38-27b-llamacpp-mtp3) continue;; esac; step "$x"; done
echo "$(date +%T) PHASE-B-DONE" >> $L
echo "$(date +%T) resuming the Ollama pull" >> $L
: > ~/kq/opull.log; ~/kq/opull.sh
until grep -q OPULL-DONE ~/kq/opull.log; do sleep 30; done
sleep 30
step q36-35b-llamacpp-mtp1
step q38-27b-llamacpp-mtp3
echo "$(date +%T) PHASE-B2-DONE" >> $L
