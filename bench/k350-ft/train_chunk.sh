#!/bin/bash
# #350 GPU job 3.x (caller holds /tmp/kapitsa-gpu.lock): one chunk of training (train.py resumes from state.pt).
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1 PYTORCH_ENABLE_MPS_FALLBACK=0
# torch requires low <= high; its default low ratio is 1.4, so a 0.7 high alone fails at the first .to(mps) (6 Nov 16:12)
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${PYTORCH_MPS_HIGH_WATERMARK_RATIO:-0.6} PYTORCH_MPS_LOW_WATERMARK_RATIO=${PYTORCH_MPS_LOW_WATERMARK_RATIO:-0.5} K350_MAX_GB=${K350_MAX_GB:-30}
S=~/Documents/nace/kapitsa/.worktrees/k350-ft/bench/k350-ft; D=/tmp/k350d; PY=~/Documents/nace/drex-dlm/.venv/bin/python
TOTAL=${TOTAL:-80}; CHUNK=${CHUNK:-40}; LR=${LR:-3e-5}
[ -f $S/PORT_OK ] && [ -s $D/data/meta.jsonl ] || { echo PREFLIGHT FAIL; exit 9; }
echo "train chunk start $(date +%T) $(pmset -g batt | head -1) total $TOTAL chunk $CHUNK lr $LR"
$PY $S/train.py $D/data $D/feat $D/ckpt $TOTAL $CHUNK $LR 2>&1 | grep -v "^dbg"
rc=${PIPESTATUS[0]}; echo "train chunk rc=$rc $(date +%T) $(pmset -g batt | head -1)"; exit $rc
