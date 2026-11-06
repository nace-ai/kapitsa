#!/bin/bash
# #350 GPU job (caller holds /tmp/kapitsa-gpu.lock): one chunk (CHUNK, default 200) of the drafter's context rows
# (k350_capt, 4-bit drafter as deployed) for conversations whose feature file is missing or short. Exit 0 with
# "capture complete" once nothing is left. Features live on the external drive (/tmp/k350d -> it).
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
W=~/Documents/nace/kapitsa/.worktrees/k350-ft; D=/tmp/k350d; M=~/Documents/nace/models/qwen3.6-35b-a3b-4bit
CHUNK=${CHUNK:-200}
[ -d "/Volumes/Extreme Pro/downloads/k350-ft/feat" ] && [ -s $D/data/capt.real.list ] || { echo PREFLIGHT FAIL; exit 9; }
python3 - "$CHUNK" <<'PY'
import json, os, sys
D = '/tmp/k350d'
n = {str(m['id']): m['n'] for m in map(json.loads, open(D + '/data/meta.jsonl'))}
def ok(l):
    f = l.split()[2]
    return os.path.exists(f) and os.path.getsize(f) == min(n[f.split('/')[-1][:-2]], 2047) * 2048 * 2
todo = [l for l in open(D + '/data/capt.real.list') if not ok(l)]
open(D + '/data/capt.todo.list', 'w').writelines(todo[:int(sys.argv[1])])
print('capture todo', len(todo), 'this chunk', min(len(todo), int(sys.argv[1])))
PY
if [ ! -s $D/data/capt.todo.list ]; then echo "capture complete"; exit 0; fi
echo "capt chunk start $(date +%T) $(pmset -g batt | head -1)"
K350_MAX_GB=${K350_MAX_GB:-32} $W/build/k350_capt $M $D/data/capt.todo.list 2>&1 | grep -v "^\[kapitsa\]"
rc=${PIPESTATUS[0]}; echo "capt chunk rc=$rc $(date +%T)"; exit $rc
