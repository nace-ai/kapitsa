import pathlib,os,subprocess,json,time,fcntl,shlex,signal
root=pathlib.Path(__file__).resolve().parent;repo=root.parents[2]/'ane-pr775';e=root/'long-results';e.mkdir(exist_ok=True);py=str(root/'venv/bin/python');model='/Users/bytedance/.cache/kapitsa/models/qwen3.5-9b-4bit';ids=['@'+str(root/'long-quality'/f'case-{i}.txt') for i in range(4)]
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_','MLX_'))};env.update(NI_ANE_ASYNC='0',NI_ANE_KEEP_S='0',NI_NO_MTP='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
def state(s):print(s,flush=True);(e/'status.json').write_text(json.dumps({'phase':s,'time':time.time()}))
def run(label,cmd,profile):
 state('waiting_for_GPU: '+label)
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state(label)
  with (e/(label+'.log')).open('w') as f:
   p=subprocess.Popen(cmd,cwd=repo,env={**env,'NI_LOCK_HELD':'1','NACE_GPU_LOCK_HELD':'1','KAPITSA_PROFILE':profile,'KAPITSA_HOME':str(e/('home-'+label))},stdout=f,stderr=subprocess.STDOUT)
   try:rc=p.wait()
   finally:
    if p.poll() is None:p.send_signal(signal.SIGINT);p.wait(timeout=30)
 (e/(label+'-exit.json')).write_text(json.dumps({'command':cmd,'returncode':rc,'profile':profile},indent=2))
 if rc:raise RuntimeError(label+' failed')
try:
 state('waiting_for_repeat_bar')
 while True:
  s=json.loads((root/'repeat-bar/status.json').read_text())['phase']
  if s=='complete' or s.startswith('failed'):break
  time.sleep(5)
 run('tf-precise',[str(root/'ni_eval_positions'),model,'--ref',str(root/'long-quality/bf16.ref'),'--dump-ref',str(e/'precise.ref'),'--topk','4096','--chunk','2048',*ids],'precise')
 run('tf-fast',[str(root/'ni_eval_positions'),model,'--ref',str(root/'long-quality/bf16.ref'),'--chunk','2048',*ids],'fast')
 run('tf-fast-vs-precise',[str(root/'ni_eval_positions'),model,'--ref',str(e/'precise.ref'),'--chunk','2048',*ids],'fast')
 items=[json.loads(x) for x in (root/'long-reasoning.jsonl').read_text().splitlines()]
 for profile in ('precise','fast'):
  for i,item in enumerate(items):
   folder=e/'inputs'/str(i);folder.mkdir(parents=True,exist_ok=True);problem=folder/'long-reasoning.jsonl';problem.write_text(json.dumps(item)+'\n')
   server=[str(repo/'build/kapitsa'),'serve','--private','--model',model,'--port','18779','--ctx','65536','--slots','1','--prefill-chunk','2048','--disk-cache','0','--dflash','none']
   cmd=[py,'bench/reasoning/reasoning.py','run','--problems',str(problem),'--max-tokens','32768','--samples','1','--label','m4-'+profile,'--base','http://127.0.0.1:18779/v1','--out',str(e/f'{profile}-case-{i}.jsonl'),'--serve',shlex.join(server)]
   run(f'reasoning-{profile}-{i}',cmd,profile)
 state('complete')
except Exception as ex:state('failed: '+str(ex));raise
