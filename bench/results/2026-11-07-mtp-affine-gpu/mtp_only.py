import datetime, fcntl, hashlib, json, os, pathlib, re, signal, subprocess, time, traceback
P=pathlib.Path(__file__).resolve().parent
O=P/'m4-35b-mtp';O.mkdir(exist_ok=False)
cfg=json.loads((P/'local.json').read_text());builds={k:pathlib.Path(v) for k,v in cfg['builds'].items()}
model=pathlib.Path('/Users/bytedance/.cache/kapitsa/models/qwen3.6-35b-a3b-oq4e-mtp')
state={'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting for 27B checks','model':str(model),'runs':[]}
def save():(O/'results.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
import sys
sys.path.insert(0,str(pathlib.Path(cfg['trees']['candidate'])/'bench/harness'))
import gpu_hogs
save()
try:
 while not (P/'m4-27b/complete').exists():
  if (P/'m4-27b/failed').exists():raise RuntimeError('27B runner needs inspection first')
  time.sleep(5)
 previous=json.loads((P/'m4-27b/STATE.json').read_text())
 for tag in builds:assert sha(builds[tag]/'ni_run')==previous['identities'][tag]['binaries']['ni_run']
 state['identities']=previous['identities'];state['metadata']={f:sha(model/f) for f in ['config.json','tokenizer.json']};save()
 tok=P/'m4-27b/tokenizer';prompt=P/'m4-27b/prompt.txt'
 ids=subprocess.check_output([str(tok),str(model),'encode',str(prompt)],text=True);(O/'prompt.ids').write_text(ids);(O/'prompt.txt').write_bytes(prompt.read_bytes())
 env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','KAPITSA_','MTL_','BENCH_'))};env.update(NI_LOCK_HELD='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  state['status']='waiting for GPU lock';save();fcntl.flock(lock,fcntl.LOCK_EX)
  reference=None
  for i,tag in enumerate(['base','candidate','base','candidate']):
   label=f'35b-mtp-{i}-{tag}';state['status']=label;save()
   assert gpu_hogs.wait_quiet(maxpct=5,quiet_s=5,timeout_s=120,exclude_roots=[os.getpid()])
   cmd=[str(builds[tag]/'ni_run'),str(model),'@'+str(O/'prompt.ids'),'256','0','--dflash','none','--spec','2']
   env['KAPITSA_HOME']=str(O/('home-'+label));mon=gpu_hogs.ContentionMonitor([os.getpid()],interval=0.5);mon.start()
   with (O/(label+'.log')).open('x') as log:
    cp=subprocess.Popen(cmd,env=env,cwd=cfg['trees'][tag],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state['child_pid']=cp.pid;save()
    try:cp.wait(timeout=1800)
    except BaseException:
     os.killpg(cp.pid,signal.SIGTERM)
     try:cp.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(cp.pid,signal.SIGKILL);cp.wait()
     raise
    finally:mon.stop()
   text=(O/(label+'.log')).read_text();match=re.search(r'^ids:(.*)$',text,re.M);assert match
   out=list(map(int,match.group(1).split()));(O/(label+'.ids')).write_text(' '.join(map(str,out))+'\n')
   decoded=subprocess.check_output([str(tok),str(model),'decode',str(O/(label+'.ids'))],text=True);(O/(label+'.txt')).write_text(decoded)
   r={'label':label,'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith(('NI_','KAPITSA_'))},'exit':cp.returncode,'tokens':len(out),'text_nonempty':bool(decoded.strip()),'eos_count':sum(x in [248044,248046] for x in out),'before_first_eos':next((j for j,x in enumerate(out) if x in [248044,248046]),len(out)),'sha256':sha(O/(label+'.ids')),'mtp_load_lines':[x for x in text.splitlines() if 'MTP' in x or 'mtp' in x],'decode_lines':[x for x in text.splitlines() if 'tok/s' in x],'contention':gpu_hogs.summarize_samples(mon.samples),'samples':mon.samples}
   state['runs'].append(r);save()
   assert cp.returncode==0 and len(out)==256 and decoded.strip() and not r['eos_count'] and 'spec decode D=2:' in text
   if reference is None:reference=out
   else:assert out==reference
 state['status']='complete: additional real MoE MTP continuation coverage, not a full second 35B benchmark';state.pop('child_pid',None);save();(O/'complete').touch()
except BaseException as e:state['status']='failed';state['error']=repr(e);save();(O/'failed').write_text(traceback.format_exc());raise
