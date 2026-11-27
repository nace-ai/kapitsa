#!/usr/bin/env python3
"""Own #1022 full-checkout synthetic integration; no trained-checkpoint claim."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
E = ROOT / 'evidence/synthetic'
ENV = {k: v for k, v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_'))}
ENV.update(NI_LOCK_HELD='1', KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1', KAPITSA_HOME=str(ROOT / 'synthetic-home'))
commands = []


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def call(label, argv, cwd=None):
    log = E / (label + '.log')
    assert not log.exists(), 'Refuse overwriting ' + str(log)
    before = time.time()
    with log.open('w') as f:
        rc = subprocess.run(list(map(str, argv)), cwd=cwd or ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
    commands.append({'label': label, 'argv': list(map(str, argv)), 'cwd': str(cwd or ROOT), 'exit': rc, 'start': before, 'elapsed_seconds': time.time()-before})
    (E / 'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    assert rc == 0, (label, rc)


def floats(p):
    a = np.fromfile(p, dtype='<f4')
    assert a.size and np.isfinite(a).all() and np.count_nonzero(a), p
    return {'bytes': p.stat().st_size, 'sha256': digest(p), 'finite': True, 'nonzero': True}


def main():
    (ROOT / 'synthetic-gpu.pid').write_text(str(os.getpid())+'\n')
    while not (ROOT / 'cpu-passed').exists():
        if (ROOT / 'build-cpu.failed').exists():
            raise RuntimeError('Own CPU dependency failed; inspect retained evidence')
        time.sleep(15)
    E.mkdir(parents=True, exist_ok=False)
    identity = {}
    for variant, expected in [('base', '01b4c6d69fbe69e862c00358f88c94a363129592'), ('candidate', '8c0a7ec3bf72623f6c7f92524ef229f3643a9e67')]:
        tree = ROOT / variant
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tree, text=True).strip()
        assert head == expected
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=tree, text=True).strip()
        identity[variant] = {'head': head, 'version': subprocess.check_output([str(tree/'build/kapitsa'), '--version'], text=True).strip(), 'library_sha256': digest(tree/'build/libnaceinfer.a'), 'ni_run_sha256': digest(tree/'build/ni_run'), 'logit_dump_sha256': digest(tree/'build/logit_dump')}
    identity['scope'] = 'Own complete baseline and candidate checkouts; random synthetic 6-layer/hidden512/vocab2048 graph. No trained weights, language quality or performance claim.'
    identity['generator_sha256'] = digest(ROOT/'candidate/tools/fn_synth.py')
    identity['harness_sha256'] = digest(Path(__file__))
    (E/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    model = ROOT/'synthetic-sharded'
    assert not model.exists()
    call('generate-model', [sys.executable, ROOT/'candidate/tools/fn_synth.py', model, '--seed', '1'])
    call('make-table-variants', [sys.executable, ROOT/'make_table_variants.py', model, ROOT/'synthetic-variants'])
    models = {'sharded':model, 'bin-affine':ROOT/'synthetic-variants/synthetic-bin-affine', 'bin-bf16':ROOT/'synthetic-variants/synthetic-bin-bf16'}
    manifest = {name:{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in path.iterdir() if p.is_file()} for name,path in models.items()}
    (E/'model-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # The fixed teacher-forced windows need no tokenizer or meaningful language.
    fixed = ROOT/'synthetic-fixed-windows'
    fixed.mkdir()
    sequences = [[(i*13+j*19)%1900+1 for i in range(48)] for j in range(3)]
    (fixed/'sequences.txt').write_text(''.join(' '.join(map(str,s))+'\n' for s in sequences))
    (fixed/'records.txt').write_text(''.join(f'{j} {pos}\n' for j in range(3) for pos in range(17,41)))
    results=[]; controls={}; logits=[]
    lock=open('/tmp/kapitsa-gpu.lock','a')
    print('Waiting for remote GPU lock',flush=True)
    fcntl.flock(lock,fcntl.LOCK_EX)
    (ROOT/'synthetic-gpu-lock-acquired').write_text(str(time.time())+'\n')
    try:
        for name, path in models.items():
            cfg=json.loads((path/'config.json').read_text());cfg=cfg.get('text_config',cfg);vocab=cfg['vocab_size']
            for mode in ['prefill-step','tbt']:
                expected=None
                for n,variant in enumerate(['base','candidate','base','candidate'],1):
                    label=f'{name}-{mode}-{n:02d}-{variant}';out=ROOT/(label+'.f32')
                    args=[ROOT/variant/'build/ni_run',path,'1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16','64','0','--dflash','none','--dump-all-logits',out]
                    if mode=='tbt':args+=['--tbt']
                    call(label,args)
                    log=(E/(label+'.log')).read_text();ids=[int(x) for x in re.search(r'^ids:(.*)$',log,re.M).group(1).split()]
                    assert len(ids)==64 and re.search(r'n-gram',log,re.I),label
                    fp=floats(out);assert fp['bytes']%(4*vocab)==0
                    current=(ids,fp)
                    if expected is None:expected=current
                    assert current==expected,label
                    row={'label':label,'generated_ids':ids,'tokens':len(ids),'logit_rows':fp['bytes']//(4*vocab),'all_ids_equal':True,'all_logits_byte_equal':True,**fp};results.append(row)
                    (E/'generation-results.json').write_text(json.dumps(results,indent=2)+'\n')
                    out.unlink()
                controls[(name,mode)]=expected
            for variant in ['base','candidate']:
                dest=ROOT/f'synthetic-logits-{name}-{variant}'
                call(f'logits-{name}-{variant}',[ROOT/variant/'build/logit_dump',path,dest,'--paths','step,verify','--from',fixed],ROOT/variant)
            for kind in ['prefill','step','verify']:
                a=ROOT/f'synthetic-logits-{name}-base'/kind;b=ROOT/f'synthetic-logits-{name}-candidate'/kind
                for f in ['records.txt','sequences.txt']:assert (a/f).read_bytes()==(b/f).read_bytes()
                fa,fb=floats(a/'logits.f32'),floats(b/'logits.f32');assert fa==fb,(name,kind)
                for variant,folder in [('base',a),('candidate',b)]:
                    for f in ['records.txt','sequences.txt','meta.json']:(E/f'{name}-{variant}-{kind}-{f}').write_bytes((folder/f).read_bytes())
                logits.append({'model':name,'path':kind,'rows':json.loads((a/'meta.json').read_text())['records'],'byte_equal':True,**fa})
                (E/'teacher-forced-results.json').write_text(json.dumps(logits,indent=2)+'\n')
                (a/'logits.f32').unlink();(b/'logits.f32').unlink()
        for mode in ['prefill-step','tbt']:
            assert controls[('sharded',mode)]==controls[('bin-affine',mode)],mode
            assert controls[('sharded',mode)][1]['sha256']!=controls[('bin-bf16',mode)][1]['sha256'],mode
        (E/'controls.json').write_text(json.dumps({'identical_affine_layouts_agree':True,'different_bf16_payload_changes_logits':True,'scope':'Synthetic integration only; no tokenizer/language or inference speed claim.'},indent=2)+'\n')
        (ROOT/'synthetic-gpu-passed').write_text('Own full-library synthetic checks passed; actual model/HTTP/native bench still pending.\n')
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close();(ROOT/'synthetic-gpu-lock-released').write_text(str(time.time())+'\n')


if __name__=='__main__':
    try:main()
    except BaseException as e:
        (ROOT/'synthetic-gpu-failed.json').write_text(json.dumps({'error':str(e),'time':time.time()})+'\n')
        raise
