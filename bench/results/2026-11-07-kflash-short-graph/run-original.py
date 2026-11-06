import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

P=Path(__file__).resolve().parent
ROOT=P.parent.parent
OLD=P.parent/'kflash-reduce-v8'
OUT=P/'graph-64'
OUT.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'kflash-reduce-v8/bench/harness'))
import gpu_hogs

state={'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting for local GPU lock','runs':[]}
def save():(OUT/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
save()
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','KAPITSA_','MTL_','REDUCE_','BENCH_'))}
env.update(NI_LOCK_HELD='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1',REDUCE_GRAPH_TIMING='1')
try:
    old=json.loads((OLD/'graph-results.json').read_text())
    identities={}
    for tag in ['base','candidate']:
        record=next(x for x in old if x['label']==f'graph-64-time-{0 if tag=="base" else 1}-{tag}')
        binary=OLD/('graph-'+tag)
        assert hashlib.sha256(binary.read_bytes()).hexdigest()==record['binary_sha256']
        identities[tag]={'binary_sha256':record['binary_sha256'],'source':'2fe7c3fa208c098b4e00b1bae9033172dfafaac7' if tag=='base' else '0be92def43f00c7c5f15b889daa71309dce4d5a7'}
    state['identities']=identities
    state['identity_note']='Reuses the original checked complete-graph binaries unchanged. Their original configure-time version labels predate the implementation commit; original source-file hashes and build evidence remain authoritative. No binary was rebuilt in this repeat.'
    state['prompt_sha256']=hashlib.sha256((OLD/'graph-64.ids').read_bytes()).hexdigest();save()
    with open('/tmp/kapitsa-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (OUT/'lock-acquired').touch()
        for arm,tag in enumerate(['base','candidate','base','candidate','base','candidate']):
            label=f'{arm}-{tag}';state['status']='quiet check '+label;save()
            assert gpu_hogs.wait_quiet(maxpct=1,quiet_s=10,timeout_s=120,exclude_roots=[os.getpid()]),'quiet precondition not met'
            run_env=dict(env,KAPITSA_HOME=str(OUT/('home-'+label)))
            cmd=[str(OLD/('graph-'+tag)),str(P.parent/'followup-1250/models/qwen3.5-4b-4bit'),str(OLD/'graph-64.ids'),str(OUT/label)]
            row={'label':label,'argv':cmd,'environment':{k:v for k,v in run_env.items() if k.startswith(('NI_','KAPITSA_','REDUCE_'))}}
            state['status']='running '+label;save()
            monitor=gpu_hogs.ContentionMonitor([os.getpid()],interval=0.5);monitor.start()
            try:
                with (OUT/(label+'.log')).open('x') as log:cp=subprocess.run(cmd,env=run_env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            finally:monitor.stop()
            samples=[float(x.split('us=')[1].split()[0]) for x in (OUT/(label+'.log')).read_text().splitlines() if x.startswith('graph sample=')]
            row.update(exit=cp.returncode,gpu_us=samples,median_last10_us=statistics.median(samples[2:]) if len(samples)==12 else None,contention=gpu_hogs.summarize_samples(monitor.samples),samples=monitor.samples)
            state['runs'].append(row);save();assert cp.returncode==0 and len(samples)==12
    state['status']='complete; inspect every contention sample before performance conclusion';save();(OUT/'passed').touch()
except BaseException as e:
    state['status']='failed';state['error']=repr(e);save();(OUT/'failed').write_text(traceback.format_exc());raise
