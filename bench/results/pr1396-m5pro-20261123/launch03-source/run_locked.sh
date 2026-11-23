#!/bin/bash
set -euo pipefail
E=/Users/alisheramantay/Documents/nace/kapitsa-evidence/pr1396-m5pro-20261123T0925Z
export BENCH_GPU_LOCK=/tmp/kapitsa-gpu.lock
python3 - <<'PY'
import fcntl,os
p='/tmp/kapitsa-gpu.lock'
# The shared lock inode is immutable by policy; locking needs no write access.
with open(p,'r') as f:
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Independent flock probe confirms exclusive outer lock held.',flush=True)
    else:
        fcntl.flock(f,fcntl.LOCK_UN)
        raise RuntimeError('Outer exclusive GPU lock is not held')
PY
lsof /tmp/kapitsa-gpu.lock > "$E/launch03-lock.txt"
export NI_LOCK_HELD=1 NACE_GPU_LOCK_HELD=1
export KAPITSA_NO_AUTO_UPDATE=1 KAPITSA_NO_PROMPT=1
export KAPITSA_HOME="$E/scratch-home" KAPITSA_PROFILE=fast
export HF_ENDPOINT=http://127.0.0.1:9 KAPITSA_UPDATE_URL=http://127.0.0.1:9
python3 - <<'PY'
import json,os
from pathlib import Path
E=Path('/Users/alisheramantay/Documents/nace/kapitsa-evidence/pr1396-m5pro-20261123T0925Z')
r=json.loads((E/'internal-stage-receipt.json').read_text());assert r['completed']
M=E/'native-model';assert str(M)==r['destination']
for rel,v in r['files'].items():assert (M/rel).stat().st_size==v['bytes'],rel
import re,subprocess,time
raw=subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True)
models=[]
native={'kapitsa','ni_run','ink2_check','vit_check','logit_dump','logit_mask_check','fn_check','ni_v8check','llama-server','llama-cli'}
for line in raw.splitlines():
    parts=line.strip().split(None,2)
    if len(parts)<3:continue
    pid,exe,args=parts;name=Path(exe).name
    if name in native or (name.lower().startswith('python') and re.search(r'mlx_lm\.(?:server|generate)|llama_cpp\.server|vllm\.entrypoints|torchrun',args)):
        models.append({'pid':int(pid),'exe':exe,'args':args})
(E/'launch03-model-processes.json').write_text(json.dumps({'checked_at':time.time(),'processes':models},indent=2)+'\n')
assert not models, f'Another local model process is active: {models}'
keys=('NI_LOCK_HELD','NACE_GPU_LOCK_HELD','BENCH_GPU_LOCK','KAPITSA_NO_AUTO_UPDATE','KAPITSA_NO_PROMPT','KAPITSA_HOME','KAPITSA_PROFILE','HF_ENDPOINT','KAPITSA_UPDATE_URL')
(E/'launch03-env.json').write_text(json.dumps({k:os.environ.get(k) for k in keys},indent=2)+'\n')
# Inherited per-kernel controls would turn this into a different comparison.
extras=[k for k in os.environ if k.startswith(('NI_','KAPITSA_')) and k not in keys]
assert not extras, f'Unexpected inherited runtime controls: {extras}'
PY
T=/Users/alisheramantay/.agents/skills/session-orchestration/scripts/timed_leg.sh
cleanup() {
  rc=$?
  trap - EXIT
  bash "$T" off "$$" || true
  python3 - "$rc" <<'PY'
import json,os,subprocess,sys,time
from pathlib import Path
E=Path('/Users/alisheramantay/Documents/nace/kapitsa-evidence/pr1396-m5pro-20261123T0925Z')
r=subprocess.run(['lsof','-nP','-iTCP:18758','-sTCP:LISTEN'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
s=os.statvfs(E)
v={'checked_at':time.time(),'runner_returncode':int(sys.argv[1]),'private_port_probe_returncode':r.returncode,'private_port_probe':r.stdout,'no_private_listener':r.returncode==1,'internal_available_GiB':s.f_bavail*s.f_frsize/2**30}
(E/'cleanup-03.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({'cleanup':v}),flush=True)
PY
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
bash "$T" on "$$"
# The runner stops and joins its own server in a finally block before this lock is released.
PYTHONDONTWRITEBYTECODE=1 python3 "$E/run_ab.py" --model "$E/native-model" --output "$E/ab-run-02"
