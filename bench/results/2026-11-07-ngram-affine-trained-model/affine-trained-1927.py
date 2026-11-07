import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path('/Users/bytedance/kapitsa-ngram-affine-20261107')
SHARED = Path('/Users/bytedance/kapitsa-ngram-remote-20261107')
OUT = ROOT / 'trained-1927'
OUT.mkdir(exist_ok=False)
MODEL = SHARED / 'model-sushi'
state = {'pid': os.getpid(), 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'status': 'waiting for PR1019 covered-row preparation and GPU completion', 'commands': [], 'results': []}
env = {k:v for k,v in os.environ.items() if not k.startswith(('NI_', 'NACE_', 'KAPITSA_', 'MTL_', 'BENCH_'))}
env.update(NI_LOCK_HELD='1', KAPITSA_NO_AUTO_UPDATE='1', KAPITSA_NO_PROMPT='1', KAPITSA_HOME=str(OUT / 'home'))
def save(): (OUT / 'STATE.json').write_text(json.dumps(state, indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()
def run(cmd, label, cwd=ROOT, check=True):
    cmd=[str(x) for x in cmd]; state['status']=label; save()
    with (OUT/(label+'.log')).open('x') as f:
        start=time.time()
        cp=subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
    state['commands'].append({'label':label,'argv':cmd,'cwd':str(cwd),'exit':cp.returncode,'elapsed_s':time.time()-start});save()
    if check: assert cp.returncode==0,(label,cp.returncode)
    return cp.returncode
def fp(p):
    import numpy as np
    a=np.memmap(p,dtype='<f4',mode='r'); nz=0
    for i in range(0,a.size,1<<20):
        v=a[i:i+(1<<20)];assert np.isfinite(v).all(),p;nz+=int(np.count_nonzero(v))
    assert nz>0,p
    return {'bytes':Path(p).stat().st_size,'sha256':sha(p),'finite':True,'nonzero_values':nz}
save()
try:
    while not (SHARED/'gpu-passed').exists():
        assert not (SHARED/'gpu-failed.json').exists(),'upstream row preparation GPU failed; inspect before using generated fixture'
        time.sleep(20)
    assert (ROOT/'cpu-passed').exists() and (SHARED/'weights-passed').exists()
    state['input_scope']='Shared pinned weights, covered corpus and generated-token fixture only. All outputs below freshly computed with this PR\'s own full libraries; no other PR outputs count as its reference.'
    state['source']={}
    for tag,expected in [('base','01b4c6d69fbe69e862c00358f88c94a363129592'),('candidate','8c0a7ec3bf72623f6c7f92524ef229f3643a9e67')]:
        tree=ROOT/tag; build=tree/'build'
        actual=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()
        assert actual==expected,(actual,expected)
        state['source'][tag]={'head':actual,'library_sha256':sha(build/'libnaceinfer.a'),'ni_run_sha256':sha(build/'ni_run'),'logit_dump_sha256':sha(build/'logit_dump')}
    sys.path.insert(0,str(SHARED));from table_rows import TableRows
    table=TableRows()
    corpus=json.loads((SHARED/'evidence/corpus-base.json').read_text())
    prompt=json.loads((SHARED/'evidence/chat-prompt.json').read_text())['sequences'][0]
    fixture=json.loads((SHARED/'evidence/generated.json').read_text())['sequences'][0]
    sys.path.insert(0,str(ROOT/'candidate/tools'));import qwen4_exp_pack as pack
    cfg=json.loads((MODEL/'config.json').read_text());ng=pack.ngram_spec(cfg.get('text_config',cfg))
    required=sorted({int(x) for x in pack.ngram_ids(prompt+fixture,ng=ng).reshape(-1)})
    state['coverage']={'corpus':table.check(corpus['rows']),'fixed_generation':table.check(required)}
    (OUT/'fixed-input.json').write_text(json.dumps({'prompt':prompt,'prepared_generation_fixture':fixture,'required_rows':required})+'\n')
    state['status']='waiting for GPU lock';save()
    with open('/tmp/kapitsa-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (OUT/'lock-acquired').touch()
        for tag in ['base','candidate']:
            build=ROOT/tag/'build'
            flags=(build/'CMakeFiles/ni_run.dir/flags.make').read_text();args=[]
            for key in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:args+=shlex.split(re.search(r'^'+key+r' = (.*)$',flags,re.M).group(1))
            obj=OUT/(tag+'.o');run(['/usr/bin/clang++',*args,ROOT/'ngram-gather-1927.cpp','-c','-o',obj],'compile-gather-'+tag,build)
            cmd=shlex.split((build/'CMakeFiles/ni_run.dir/link.txt').read_text());cmd=[str(obj) if x.endswith('tools/ni_run.cpp.o') else x for x in cmd];cmd[cmd.index('-o')+1]=str(OUT/('gather-'+tag));run(cmd,'link-gather-'+tag,build)
        reference=None
        for arm,tag in enumerate(['base','candidate','base','candidate']):
            label=f'gather-{arm}-{tag}';raw=OUT/(label+'.f32')
            run([OUT/('gather-'+tag),MODEL,SHARED/'evidence/corpus-base.json',raw],label)
            result=fp(raw)
            if reference is None:reference=result
            else:assert result==reference,'gather outputs differ'
            state['results'].append({'label':label,**result,'byte_equal':True});save();raw.unlink()
        for tag in ['base','candidate']:
            cmd=[ROOT/tag/'build/logit_dump',MODEL,OUT/('logits-'+tag),'--paths','step,verify']
            if tag=='candidate':cmd+=['--from',OUT/'logits-base']
            run(cmd,'logits-'+tag,ROOT/tag)
        for name in ['prefill','step','verify']:
            a,b=[OUT/('logits-'+tag)/name for tag in ['base','candidate']]
            assert (a/'sequences.txt').read_bytes()==(b/'sequences.txt').read_bytes()
            seqs=[[int(x) for x in re.split(r'[,\s]+',s.strip()) if x] for s in (a/'sequences.txt').read_text().splitlines() if s.strip()]
            assert seqs==corpus['sequences'],'uncovered corpus window'
            assert (a/'records.txt').read_bytes()==(b/'records.txt').read_bytes()
            fa,fb=fp(a/'logits.f32'),fp(b/'logits.f32');assert fa==fb,name
            state['results'].append({'path':name,**fa,'byte_equal':True,'records':json.loads((a/'meta.json').read_text())['records']});save()
        state['comparator_exit']=run([sys.executable,ROOT/'candidate/tools/logit_compare.py',OUT/'logits-base',OUT/'logits-candidate'],'logit-compare',check=False)
        save()
        for tag in ['base','candidate']:
            for name in ['prefill','step','verify']:(OUT/('logits-'+tag)/name/'logits.f32').unlink()
        reference=None; reference_ids=None
        cfgtext=cfg.get('text_config',cfg);eos=cfgtext.get('eos_token_id',cfg.get('eos_token_id',[248044,248046]));eos=set(eos if isinstance(eos,list) else [eos])|{248044,248046}
        for arm,tag in enumerate(['base','candidate','base','candidate']):
            table.check(required);label=f'greedy-{arm}-{tag}';raw=OUT/(label+'.f32')
            run([ROOT/tag/'build/ni_run',MODEL,'@'+str(SHARED/'evidence/chat-prompt.ids'),'256','0','--dflash','none','--dump-all-logits',raw],label)
            text=(OUT/(label+'.log')).read_text();ids=[int(x) for x in re.search(r'^ids:(.*)$',text,re.M).group(1).split()]
            actual_rows={int(x) for x in pack.ngram_ids(prompt+ids,ng=ng).reshape(-1)}
            assert actual_rows<=set(required),'generation used unprepared rows; reject this run'
            assert len(ids)==256 and not (set(ids)&eos)
            own_input=OUT/(label+'.ids.json');own_input.write_text(json.dumps({'sequences':[ids]})+'\n')
            run([ROOT/'real-rows-1857/probe-candidate','decode',MODEL,own_input,OUT/(label+'.text.json')],label+'-decode')
            assert json.loads((OUT/(label+'.text.json')).read_text())['texts'][0].strip()
            f=fp(raw)
            if reference is None:reference=f;reference_ids=ids
            else:assert f==reference and ids==reference_ids,'own A/B generated outputs differ'
            state['results'].append({'label':label,**f,'tokens':len(ids),'eos_count':0,'text_nonempty':True,'all_ids_equal':True,'byte_equal':True});save();raw.unlink()
    (OUT/'lock-released').touch()
    state['status']='complete; inspect evidence, no model speed claim; HTTP/native bench remain unverified';save();(OUT/'passed').touch()
except BaseException as e:
    state['status']='failed';state['error']=repr(e);save();(OUT/'failed').write_text(traceback.format_exc());raise
