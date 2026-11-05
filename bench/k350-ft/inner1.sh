#!/bin/bash
# #350 drafter fine-tune, GPU job 1 (untimed; caller holds /tmp/kapitsa-gpu.lock):
#   1. port check: 8 drafting passes of the shipped drafter (8-bit) with their contexts dumped
#   2. self-distillation: kapitsa serve answers items.jsonl (greedy, thinking off) until GEN_MIN minutes pass
#   3. render + tokenize, then k350_capt writes the drafter's context rows of every answered conversation
#   4. if train_inner.sh exists: runs it (training holds the same lock)
{
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
W=~/Documents/nace/kapitsa/.worktrees/k350-ft; D=/tmp/k350d; S=$W/bench/k350-ft
M=~/Documents/nace/models/qwen3.6-35b-a3b-4bit; PY=~/Documents/nace/drex-dlm/.venv/bin/python
GEN_MIN=${GEN_MIN:-50}
[ -d "/Volumes/Extreme Pro/downloads" ] && [ -r "$(readlink -f $M)/config.json" ] && [ -x $W/build/k350_capt ] || { echo PREFLIGHT FAIL; exit 9; }
echo "start $(date +%T) $(pmset -g batt | head -1) load $(sysctl -n vm.loadavg) commit $(git -C $W rev-parse --short HEAD)"
if [ ! -s $D/pc/pc.out ]; then
  NI_DFLASH_BITS=8 K350_RAW=1 timeout 600 $W/build/k350_capt $M $D/pc/pc.list > $D/pc/pc.out 2> $D/pc/pc.err; echo "portcheck rc=$? $(date +%T)"
  NI_DFLASH_BITS=4 timeout 600 $W/build/k350_capt $M $D/pc/pc.list > $D/pc/pc4.out 2> $D/pc/pc4.err; echo "portcheck4 rc=$? $(date +%T)"
fi
export KAPITSA_HOME=/tmp/k350-home; mkdir -p $KAPITSA_HOME
$W/build/kapitsa serve --model $M --port 8199 --private --slots 4 --disk-cache 0 --context 8k > $D/serve.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:8199/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:8199/health || { echo "serve did not start"; tail -20 $D/serve.log; exit 3; }
echo "serve up $(date +%T)"
$PY $S/gen.py $D/data/items.jsonl $D/data/results.jsonl 8199 4 640 $(( $(date +%s) + GEN_MIN * 60 ))
echo "gen rc=$? $(date +%T) $(pmset -g batt | head -1)"
kill $SP; wait $SP 2>/dev/null
mkdir -p $D/feat
$PY $S/render.py $D/data/items.jsonl $D/data/results.jsonl $D/data $D/feat
timeout 3600 $W/build/k350_capt $M $D/data/capt.list 2>&1 | grep -v "^\[" | tail -5
echo "capt rc=$? $(date +%T) $(pmset -g batt | head -1)"
du -sh $D/feat
if [ -f $S/train_inner.sh ]; then echo "train $(date +%T)"; bash $S/train_inner.sh; echo "train rc=$? $(date +%T)"; fi
echo DONE1
}
