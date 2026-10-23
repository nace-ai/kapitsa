#!/bin/bash
# M3 Pro step 2, Nemotron-H after #351 (9fc7ff3): Kapitsa on origin/main c490f81 (~/kq/nh) for Dory, Nano 4B, Nano 9B,
# Lightning and Bolt (protocol, then the 1K->32K sweep), and each one's rival sweep once (Bolt has no rival cell).
# Waits for q3 to finish (its B2 downloads must not overlap a measurement). Same checks as q3.
. ~/kenv.sh; . ~/kq/m3/cfg.sh
# every leg (build, logit_dump, doctor, harness run, kapitsa bench) under the GPU lock; the tools then skip their own
G() { lockf -k /tmp/kapitsa-gpu.lock env NACE_GPU_LOCK_HELD=1 NI_LOCK_HELD=1 "$@"; }
L=~/kq/m3/q.log
until grep -qE "^[0-9:]{8} CLEANUP-DONE:" $L && ! pgrep -f "m3/q3.sh" >/dev/null; do sleep 30; done
cd ~/kq/nh/bench/harness
sed -e "s#^KAPITSA_BIN=.*#KAPITSA_BIN=$HOME/kq/nh/build/kapitsa#" ~/kq/head/bench/harness/local.env > local.env
echo "$(date +%T) PHASE-NH start: $(~/kq/nh/build/kapitsa --version 2>&1 | head -1), harness ~/kq/nh; $(grep KAPITSA_BIN local.env)" >> $L
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
step() { wait_ok "$1"; G python3 compare_engines.py $C --skip agent --only "$1" >> $L 2>&1; after $? "$1"; }
sweep() {
  wait_ok "sweep $1"
  G python3 compare_engines.py $C --only "$1" --agentic \
    --bench-args "--skip decode,greedy,prefill,tools,agent,cache,sustain --sweep-repeats 1" >> $L 2>&1; after $? "sweep $1"
}
# The two 9B rival cells ran while someone woke the M3's display (pmset log: on 01:11:53-01:19:22, the lock screen,
# VTDecoder idle): q35-9b-llamacpp's last minute and q35-9b-mlxlm's first minutes. Run them again with the display asleep.
for x in q35-9b-llamacpp q35-9b-mlxlm; do step "$x"; done
R=~/kq/m3/out/results; mkdir -p $R/display-woken
mv $R/q35-9b-llamacpp-20261023-010646.json $R/q35-9b-mlxlm-20261023-011328.json $R/display-woken/ 2>/dev/null
# nano-4b-llamacpp's first run was checked against the Qwen prompt reference (4133 / 16286); again with the Nemotron one
mkdir -p $R/qwen-ref; mv $R/nano-4b-llamacpp-20261023-012508.json $R/qwen-ref/ 2>/dev/null
step nano-4b-llamacpp
# Phase C's rival sweeps, which q3 skipped (pick.py read decode_summary.median, not median_tps): the fastest rival by
# decode for the 4B, 27B, 35B-A3B (Ollama's MLX runner, all three) and the 9B (picked from the rerun records)
cd ~/kq/bm && { [ -e ollama-mlx ] || ln -s ollama-mlx.keep ollama-mlx; }; cd ~/kq/nh/bench/harness
P=~/flint-m3/sweep/venv/bin/python
for m in RIV4 RIV27 RIV35; do r=$($P ~/kq/m3/pick.py ${!m//,/ }); echo "$(date +%T) sweep rival ($m): ${r:-none}" >> $L; [ -n "$r" ] && sweep "$r"; done
r=$($P ~/kq/m3/pick.py q35-9b-llamacpp q35-9b-mlxlm); echo "$(date +%T) sweep rival q35-9b: ${r:-none}" >> $L; [ -n "$r" ] && sweep "$r"
rm -f ~/kq/bm/ollama-mlx ~/kq/models/gguf/Qwen3.5-9B-Q4_K_M.gguf; rm -rf ~/kq/bm/ollama-mlx.keep; rm -f ~/kq/models/gguf/Qwen3.5-9B-Q4_K_M.gguf.keep
dk=$(df -g ~ | tail -1 | awk '{print $4}'); echo "$(date +%T) RIVAL-SWEEPS-DONE: Ollama MLX models + 9B GGUF deleted; ${dk}G free ($([ "$dk" -ge 30 ] && echo "floor 30G ok" || echo "UNDER the 30G floor"))" >> $L; df -h ~ | tail -1 >> $L
for m in dory nano-4b nano-9b lightning bolt; do step sw-$m-kapitsa; done
for p in dory:dory-mlxlm-4bit nano-4b:nano-4b-llamacpp nano-9b:nano-9b-mlxlm-4bit lightning:lightning-mlxvlm-dspark bolt:; do
  m=${p%%:*}; r=${p#*:}; sweep sw-$m-kapitsa; [ -n "$r" ] && sweep "$r"
done
echo "$(date +%T) PHASE-NH-DONE" >> $L
# the Nemotron rivals' downloads, now recorded (Lightning's folder is Kapitsa's own and stays)
rm -rf ~/kq/models/mlx/NVIDIA-Nemotron-Labs-Dory1.0-4B-mlx-4bit ~/kq/models/mlx/NVIDIA-Nemotron-Nano-9B-v2-4bits
rm -f ~/kq/models/gguf/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
dk=$(df -g ~ | tail -1 | awk '{print $4}'); echo "$(date +%T) NH-CLEANUP-DONE: ${dk}G free ($([ "$dk" -ge 30 ] && echo "floor 30G ok" || echo "UNDER the 30G floor"))" >> $L; df -h ~ | tail -1 >> $L
