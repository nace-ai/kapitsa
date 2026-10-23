#!/bin/bash
# M3 Pro step 2, Phases B and C (fresh cells only; RIVALS.md), on main's harness. Waits for PHASE-A-DONE.
. ~/kenv.sh; . ~/kq/m3/cfg.sh; cd ~/kq/head/bench/harness
L=~/kq/m3/q.log; P=~/flint-m3/sweep/venv/bin/python
until grep -q PHASE-A-DONE $L && ! pgrep -f "m3/q2.sh" >/dev/null; do sleep 30; done
wait_ok() {
  pmset displaysleepnow; sleep 5
  while :; do
    fr=$(memory_pressure | tail -1 | grep -o "[0-9]*" | tail -1); dk=$(df -g ~ | tail -1 | awk '{print $4}')
    [ "$fr" -ge 40 ] && [ "$dk" -ge 30 ] && break
    echo "$(date +%T) wait: free ${fr}% disk ${dk}G" >> $L; sleep 60
  done
  vt=$(pgrep -x VTDecoderXPCService >/dev/null && ps -o %cpu= -p "$(pgrep -x VTDecoderXPCService | head -1)" || echo none)
  echo "$(date +%T) power $(pmset -g batt | head -1 | sed 's/Now drawing from //') free ${fr}% disk ${dk}G vtdecoder $vt :: $*" >> $L
}
after() {
  echo "$(date +%T) rc=$1 :: $2" >> $L
  pgrep -fl "llama-server|mlx_lm.server|mlx_vlm|bin/mtplx|ollama serve|kapitsa serve" >> $L && echo "$(date +%T) WARNING leftover process" >> $L
}
touch -t 202610222337 /tmp/q3mark
done_() { [ -n "$(find ~/kq/m3/out/results -name "$1-2026*.json" -newer /tmp/q3mark 2>/dev/null | head -1)" ]; }
while pgrep -f compare_engines.py >/dev/null; do sleep 15; done   # a step the previous q3 started
step() { done_ "$1" && { echo "$(date +%T) already recorded :: $1" >> $L; return; }; wait_ok "$1"; python3 compare_engines.py $C --skip agent --only "$1" >> $L 2>&1; after $? "$1"; }
sweep() {
  wait_ok "sweep $1"
  python3 compare_engines.py $C --only "$1" --agentic \
    --bench-args "--skip decode,greedy,prefill,tools,agent,cache,sustain --sweep-repeats 1" >> $L 2>&1; after $? "sweep $1"
}
step sw-flint-kapitsa
for x in ${RIV4//,/ } ${RIV27//,/ } ${RIV35//,/ } ${RIVO//,/ }; do
  case $x in bolt-mlxvlm|q36-35b-llamacpp-mtp1|q38-27b-llamacpp-mtp3) continue;; esac; step "$x"; done
echo "$(date +%T) PHASE-B-DONE" >> $L
rival() {
  case $1 in
    q35-4b) $P ~/kq/m3/pick.py ${RIV4//,/ };; q38-27b) $P ~/kq/m3/pick.py ${RIV27//,/ };; q36-35b) $P ~/kq/m3/pick.py ${RIV35//,/ };;
    q35-9b) $P ~/kq/m3/pick.py q35-9b-llamacpp q35-9b-mlxlm;; dory) echo dory-mlxlm-4bit;; nano-4b) echo nano-4b-llamacpp;;
    nano-9b) echo nano-9b-mlxlm-4bit;; lightning) echo lightning-mlxvlm-dspark;; *) echo;;
  esac
}
# Nemotron-H (Dory, Nano 4B/9B, Lightning, Bolt): Kapitsa cannot compile its Metal 3.2 kernels on macOS 14, so no sweep pair
for m in q35-4b q38-27b q36-35b q35-9b flint; do
  r=$(rival $m); echo "$(date +%T) sweep rival $m: ${r:-none (reused or not measured)}" >> $L
  sweep sw-$m-kapitsa
  [ -n "$r" ] && sweep "$r"
done
echo "$(date +%T) PHASE-C-DONE" >> $L
# After Phase C: the rival-only downloads no later step needs (the Ollama MLX runner's models, the Qwen GGUFs, the
# 4B MTP drafter and MTPLX head). The Nemotron rivals' models stay until the PR is final: if fa880208's macOS 14 fix
# for the Nemotron-H kernels merges first, their rival sweeps run once against the new Kapitsa cells.
rm -rf ~/kq/bm/ollama-mlx ~/kq/bm/drafters/Qwen3.5-4B-MTP-4bit ~/kq/bm/qwen3.5-4b-4bit-mtplx
rm -f ~/kq/models/gguf/Qwen3.5-9B-Q4_K_M.gguf ~/kq/models/gguf/Qwen3.5-4B-Q4_K_M.gguf
echo "$(date +%T) cleanup after Phase C: $(df -g ~ | tail -1 | awk '{print $4}')G free" >> $L
# Phase B2: llama.cpp + MTP on the 35B and 27B, reading Ollama's MTP GGUFs. One at a time: pulled only now (no download
# beside a measurement), skipped if it would leave under 30 GiB free (the M3's floor), deleted once its cell is recorded.
O=~/engines/ollama/ollama
orun() { OLLAMA_NOPRUNE=1 OLLAMA_HOST=127.0.0.1:11434 $O serve >> ~/kq/oserve.log 2>&1 & OS=$!; sleep 5
         OLLAMA_HOST=127.0.0.1:11434 $O "$@" >> ~/kq/opull-cur.log 2>&1; local rc=$?; kill $OS; wait $OS 2>/dev/null; return $rc; }
b2() {  # tag size_G label
  local dk=$(df -g ~ | tail -1 | awk '{print $4}')
  if [ $((dk - $2)) -lt 30 ]; then echo "$(date +%T) B2 SKIPPED $3: ${dk}G free, its GGUF (~$2G) would leave under 30G" >> $L; return; fi
  echo "$(date +%T) B2 pull $1 (${dk}G free)" >> $L
  orun pull $1 && echo "$(date +%T) B2 pulled $1: $(df -g ~ | tail -1 | awk '{print $4}')G free" >> $L || { echo "$(date +%T) B2 PULL FAILED $1" >> $L; orun rm $1; return; }
  sleep 60; step $3
  orun rm $1; echo "$(date +%T) B2 deleted $1: $(df -g ~ | tail -1 | awk '{print $4}')G free" >> $L
}
b2 qwen3.6:35b-a3b-mtp-q4_K_M 25 q36-35b-llamacpp-mtp1
b2 qwen3.8:27b-mtp-q4_K_M 20 q38-27b-llamacpp-mtp3
orun rm qwen3.5:4b-mtp-q4_K_M   # the 4B llama.cpp cell's GGUF, recorded in Phase B (and swept in C if it won)
echo "$(date +%T) PHASE-B2-DONE" >> $L
echo "$(date +%T) CLEANUP-DONE: $(df -g ~ | tail -1 | awk '{print $4}')G free; ollama $(du -sh ~/.ollama/models | cut -f1); kept for a Nemotron rerun: ~/kq/models/mlx, ~/kq/models/gguf/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf" >> $L
