#!/usr/bin/env python3
"""Sequential locked experiment; all outputs stay outside the checkout."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = CODE = LC = None
MODELS = []


def configure():
    import argparse
    import shutil
    global ROOT, CODE, LC, MODELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for model in ("4b", "9b", "27b"):
        parser.add_argument("--model-" + model, type=Path, required=True)
    args = parser.parse_args()
    CODE = args.checkout.resolve()
    ROOT = args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).resolve().parent / "inputs", ROOT / "inputs")
    MODELS = [(m, getattr(args, "model_" + m).resolve(), units)
              for m, units in (("4b", 18), ("9b", 24), ("27b", 34))]
    spec = importlib.util.spec_from_file_location("lc", CODE / "tools/logit_compare.py")
    LC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(LC)
    assert LC.np is not None, "numpy is required for the published row metrics"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def environment(ane=0, units=18):
    env = {k:v for k,v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'MTL_'))}
    env.update({'NI_ANE':str(ane), 'NI_ANE_H4096':'1', 'NI_ANE_WAVE':'1', 'NI_ANE_ASYNC':'0',
                'NI_ANE_MIN_ROWS':'512', 'NI_ANE_VERIFY':'1', 'NI_ANE_KEEP_S':'0', 'NI_ANE_UNITS':str(units),
                'NI_BAL':'0', 'NI_KFLASH':'1', 'NI_LOCK_HELD':'1', 'NACE_GPU_LOCK_HELD':'1',
                'BENCH_GPU_LOCK':os.environ.get('BENCH_GPU_LOCK', '/tmp/kapitsa-gpu.lock'), 'KAPITSA_NO_AUTO_UPDATE':'1',
                'KAPITSA_HOME':str(ROOT/'scratch-home'), 'OPENBLAS_NUM_THREADS':'1', 'VECLIB_MAXIMUM_THREADS':'1'})
    return env


def run(name, cmd, env, timeout=2400):
    folder = ROOT/'runs'/name
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'command.json').exists(), f'Refusing to overwrite {name}'
    data = {'cmd':list(map(str,cmd)), 'cwd':str(CODE), 'env':{k:v for k,v in env.items() if k.startswith(('NI_', 'NACE_', 'MTL_', 'KAPITSA_', 'BENCH_'))}, 'started':time.time()}
    save(folder/'command.json', data)
    print(f'START {name}', flush=True)
    save(ROOT/'progress.json', {'stage':name, 'started':data['started']})
    with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
        p = subprocess.run(list(map(str,cmd)), cwd=CODE, env=env, stdout=out, stderr=err, timeout=timeout)
    data.update(rc=p.returncode, finished=time.time())
    save(folder/'command.json',data)
    if p.returncode:
        raise RuntimeError(f'{name} exited {p.returncode}: {(folder/"stderr.log").read_text()[-2500:]}')
    print(f'DONE {name}: {data["finished"]-data["started"]:.1f}s',flush=True)
    return folder


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def analyze(a,b,out):
    run(str(out.relative_to(ROOT/'runs')), [sys.executable,CODE/'tools/logit_compare.py',a,b,'--no-gate','--json',out/'comparison.json'], environment())
    allrows = {}
    for name,ref,cand in LC.pairs_of(str(a),str(b),None):
        rows=[]
        for i,j in LC.match(ref,cand):
            r=LC.row_metrics(ref.row(i),cand.row(j),ref.nexts[i])
            r.update(sequence=ref.keys[i][0],prefix=ref.keys[i][1])
            rows.append(r)
        allrows[name]=rows
    save(out/'rows.json',allrows)
    assert all(r['kl'] is not None for rows in allrows.values() for r in rows)


def logit_arm(model,path,units,sample,repeat,ane):
    label=f'{model}/{sample}-{repeat}-'+('B' if ane else 'A')
    target=ROOT/'runs'/label/'dump'
    folder=run(label,[CODE/'build/logit_dump',path,target,'--paths','step','--batch','1','--chunk','2048','--from',ROOT/'inputs'/sample],environment(ane,units))
    meta=json.loads((target/'step/meta.json').read_text())
    stats=meta['ane']
    assert stats['reruns']==0
    if ane:
        assert stats['chunks']>0 and stats['evals']>0, stats
    else:
        assert stats['chunks']==0 and stats['evals']==0,stats
    checks={str(p.relative_to(target)):digest(p) for p in target.rglob('logits.f32')}
    save(folder/'logit-sha256.json',checks)
    return target


def main():
    assert os.environ.get('NI_LOCK_HELD')=='1', 'Must be launched through vk gpu run'
    env=environment()
    env.update(MTL_SHADER_VALIDATION='1',MTL_DEBUG_LAYER='1')
    run('geometry-validation',[CODE/'build/ane_segment_check','--gpu'],env,300)
    for model,path,units in MODELS:
        for sample,count in [('primary',2),('long',1)]:
            for repeat in range(1,count+1):
                a=logit_arm(model,path,units,sample,repeat,0)
                b=logit_arm(model,path,units,sample,repeat,1)
                out=ROOT/'runs'/model/f'{sample}-{repeat}-comparison'
                analyze(a,b,out)
                # Only this task's raw logits are removed; keep all metadata, logs and metrics.
                for d in (a,b):
                    for p in d.rglob('logits.f32'): p.unlink()
        ref=ROOT/'runs'/model/'ppl-reference.bin'
        ids=['@'+str(ROOT/'inputs/ppl'/f'doc{i}.ids') for i in range(8)]
        base=[CODE/'build/ni_eval',path,'--chunk','1024','--skip','0']
        run(f'{model}/ppl-A',base+['--dump-ref',ref,'--topk','256']+ids,environment(0,units))
        folder=run(f'{model}/ppl-B',base+['--ref',ref]+ids,environment(1,units))
        err=(folder/'stderr.log').read_text()
        assert 'Neural Engine FFN split (wavefront)' in err and 'startup check' in err
        assert not any(s in err for s in ['GPU only','split unavailable','split failed','stopped ('])
        vals=[json.loads(line) for line in (folder/'stdout.log').read_text().splitlines() if line.startswith('{')]
        assert vals[-1]['positions']==8192, vals[-1]
        save(ROOT/'runs'/model/'ppl-summary.json',vals)
    save(ROOT/'experiment-complete.json',{'finished':time.time(),'status':'complete'})

if __name__=='__main__':
    assert os.environ.get('NI_LOCK_HELD') == '1', 'Run through vk gpu run to hold the GPU lock'
    configure()
    main()
