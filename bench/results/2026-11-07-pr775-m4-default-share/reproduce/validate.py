import os,pathlib,subprocess,json,time,fcntl,hashlib,shutil
root=pathlib.Path(__file__).resolve().parent; repo=root.parents[2]/'ane-pr775'; e=root/'runs';e.mkdir(exist_ok=True)
model=pathlib.Path('/Users/bytedance/.cache/kapitsa/models/qwen3.5-9b-4bit')
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_'))}
env.update(NI_LOCK_HELD='1',NACE_GPU_LOCK_HELD='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1',NI_ANE_ASYNC='0',NI_ANE_KEEP_S='0',NI_NO_MTP='1')
def state(p):
 print(p,flush=True);(e/'status.json').write_text(json.dumps({'phase':p,'time':time.time()}))
def run(label,cmd,extra=None):
 state(label)
 with (e/(label+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=repo,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
 (e/(label+'-exit.json')).write_text(json.dumps({'command':cmd,'returncode':p.returncode,'env':{k:v for k,v in {**env,**(extra or {})}.items() if k.startswith(('NI_','KAPITSA_'))}},indent=2))
 if p.returncode:raise RuntimeError(label+' failed '+str(p.returncode))
try:
 state('model_identity')
 rows=[]
 for p in sorted(model.iterdir()):
  if p.is_file() and p.suffix in ('.safetensors','.json','.jinja'):
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
   rows.append({'file':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
 (e/'identity.json').write_text(json.dumps({'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'model':str(model),'files':rows,'chip':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'macos':subprocess.check_output(['sw_vers'],text=True),'sdk':subprocess.check_output(['xcrun','--show-sdk-version'],text=True),'calibration_file_exists':pathlib.Path('/Users/bytedance/.cache/nace-infer-models/ane-cache/calibration.txt').exists()},indent=2))
 state('waiting_for_GPU')
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  run('segments',[str(repo/'build/ane_segment_check'),'--gpu'],{'MTL_SHADER_VALIDATION':'1','MTL_DEBUG_LAYER':'1'})
  for label in ('precise','fast'):
   if shutil.disk_usage(root).free<1800*1024**2:raise RuntimeError('insufficient free disk for logit dump')
   cmd=[str(repo/'build/logit_dump'),str(model),str(e/('logits-'+label)),'--chunk','2048']
   if label=='fast':cmd+=['--from',str(e/'logits-precise/step')]
   run(label,cmd,{'KAPITSA_PROFILE':label,'KAPITSA_HOME':str(e/('home-'+label))})
  meta={x:json.loads((e/('logits-'+x)/'step/meta.json').read_text()) for x in ('precise','fast')}
  (e/'policy-runtime.json').write_text(json.dumps(meta,indent=2))
  if not meta['fast']['ane']['evals']:raise RuntimeError('fast made zero ANE evaluations: not valid default ANE evidence')
  if meta['precise']['ane']['evals']:raise RuntimeError('precise unexpectedly used ANE')
 state('complete')
except Exception as ex:state('failed: '+str(ex));raise
