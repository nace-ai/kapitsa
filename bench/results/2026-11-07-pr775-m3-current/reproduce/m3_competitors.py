import pathlib,os,subprocess,json,time,fcntl,urllib.request,hashlib,plistlib,re,shutil,sys,statistics
root=pathlib.Path(__file__).resolve().parent;repo=root/'candidate-current-pr775';e=root/'evidence-m3-current-competitors';e.mkdir(exist_ok=True);py=str(root/'.venv/bin/python')
sys.path.insert(0,str(repo/'bench/harness'));import gpu_hogs
base={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_','MLX_'))};base['NI_LOCK_HELD']='1';base['DYLD_LIBRARY_PATH']=str(root/'llama-b12110')
model='/Users/bytedance/kapitsa-qmm9b-models-20261105/hub/models/qwen3.5-9b-4bit';revision='3885219b6810b007914f3a7950a8d1b469d598a5';name='Qwen3.5-9B-Q4_K_M.gguf';expected='03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8'
def state(p):
 print(p,flush=True);(e/'status.json').write_text(json.dumps({'phase':p,'time':time.time()}))
def run(label,cmd,timed=False):
 state(label);quiet=None;mon=None
 if timed:
  quiet,wait,busy=gpu_hogs.wait_quiet(quiet_s=15,timeout_s=300)
  if not quiet:raise RuntimeError('GPU did not become quiet before '+label)
 with (e/(label+'.log')).open('w') as out:
  proc=subprocess.Popen(cmd,cwd=repo,env=base,stdout=out,stderr=subprocess.STDOUT)
  if timed:mon=gpu_hogs.ContentionMonitor([proc.pid]);mon.start()
  start=time.time()
  try:rc=proc.wait()
  finally:
   end=time.time()
   if mon:mon.stop()
   if proc.poll() is None:proc.terminate();proc.wait(timeout=30)
 (e/(label+'-exit.json')).write_text(json.dumps({'command':cmd,'returncode':rc,'start':start,'end':end},indent=2))
 if mon:(e/(label+'-telemetry.json')).write_text(json.dumps({'quiet':quiet,'wait_seconds':wait,'busy':busy,'samples':mon.samples,'summary':gpu_hogs.summarize_samples(mon.samples,start,end)},indent=2))
 if rc:raise RuntimeError(label+' exit '+str(rc))
def layout(engine):
 out=e/('logits-'+engine);out.mkdir(exist_ok=True)
 for path in ('step','verify','batch','prefill'):
  ref=root/'evidence-m3-current/logits-precise'/path;dst=out/path;dst.mkdir(exist_ok=True)
  for f in ('sequences.txt','records.txt'):shutil.copyfile(ref/f,dst/f)
  raw=e/(engine+'-raw')/('prefill.f32' if path=='prefill' else 'step.f32')
  if path in ('verify','batch'):assert (ref/'records.txt').read_bytes()==(root/'evidence-m3-current/logits-precise/step/records.txt').read_bytes()
  os.link(raw,dst/'logits.f32')
  meta=json.loads((ref/'meta.json').read_text());meta.update(engine=engine,build='mlx-lm 0.32.2 / mlx 0.32.4' if engine=='mlx' else 'llama.cpp b12110 cd959bae9',model_dir=model if engine=='mlx' else 'unsloth/Qwen3.5-9B-GGUF@'+revision,env={},ane={'evals':0},native_path='single-token teacher forcing; same rows reused for Kapitsa path comparisons')
  for key in ('disk_format','batch','verify'):meta.pop(key,None)
  (dst/'meta.json').write_text(json.dumps(meta,indent=2))
state('waiting_for_M3_ABAB')
try:
 while True:
  p=root/'evidence-m3-current-performance/performance-status.json';s=json.loads(p.read_text())['phase'] if p.exists() else ''
  if s.startswith('failed'):raise RuntimeError(s)
  if s=='complete':break
  time.sleep(5)
 state('waiting_for_GPU')
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);device=None
  try:
   gguf=e/name
   assert not gguf.exists()
   if shutil.disk_usage(root).free<9*1024**3:raise RuntimeError('need 9 GiB free before GGUF and logits')
   state('downloading_GGUF')
   h=hashlib.sha256();url=f'https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/{revision}/{name}'
   with urllib.request.urlopen(url,timeout=120) as src,gguf.open('wb') as dst:
    while b:=src.read(8*1024**2):dst.write(b);h.update(b)
   assert h.hexdigest()==expected
   (e/'gguf-identity.json').write_text(json.dumps({'url':url,'bytes':gguf.stat().st_size,'sha256':h.hexdigest()},indent=2))
   for engine,cmd in [('mlx',[py,str(root/'mlx_probe.py'),model]),('llama',[str(root/'llama_probe'),str(gguf)])]:
    run(engine+'-speed',cmd+['speed',str(root/'speed-ids.txt'),str(e/(engine+'-speed.jsonl'))],True)
    raw=e/(engine+'-raw');raw.mkdir(exist_ok=True)
    run(engine+'-dump',cmd+['dump',str(root/'evidence-m3-current/logits-precise/step'),str(raw)])
    layout(engine)
   rates={x:statistics.median(json.loads(l)['tok_s'] for l in (e/(x+'-speed.jsonl')).read_text().splitlines()) for x in ('mlx','llama')}
   (e/'speed-selection.json').write_text(json.dumps({'median_decode_tok_s':rates,'faster':max(rates,key=rates.get)},indent=2))
  finally:
   if 'gguf' in locals() and gguf.exists():
    gguf.unlink()
    (e/'temporary-model-cleanup.json').write_text(json.dumps({'removed':str(gguf),'reason':'task-downloaded temporary model; original pinned URL and SHA256 retained'}))
 for engine in ('mlx','llama'):
  run('compare-'+engine,[py,str(repo/'tools/logit_compare.py'),str(root/'evidence-m3-current/logits-precise'),str(e/('logits-'+engine)),'--no-gate','--json',str(e/('compare-'+engine+'.json'))])
 run('compare-fast',[py,str(repo/'tools/logit_compare.py'),str(root/'evidence-m3-current/logits-precise'),str(root/'evidence-m3-current/logits-fast'),'--no-gate','--json',str(e/'compare-fast.json')])
 state('complete')
except Exception as ex:state('failed: '+str(ex));raise
