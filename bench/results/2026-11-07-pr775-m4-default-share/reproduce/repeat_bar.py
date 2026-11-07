import pathlib,subprocess,os,fcntl,json,time
root=pathlib.Path(__file__).resolve().parent;repo=root.parents[2]/'ane-pr775';e=root/'repeat-bar';e.mkdir(exist_ok=True);py=str(root/'venv/bin/python');model='/Users/bytedance/.cache/kapitsa/models/qwen3.5-9b-4bit'
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_','MLX_'))};env.update(NI_LOCK_HELD='1',NI_ANE_ASYNC='0',NI_ANE_KEEP_S='0',NI_NO_MTP='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
def state(s):(e/'status.json').write_text(json.dumps({'phase':s,'time':time.time()}))
def run(name,cmd,extra=None):
 state(name)
 with (e/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=repo,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
 (e/(name+'-exit.json')).write_text(json.dumps({'command':cmd,'returncode':r.returncode}))
 if r.returncode:raise RuntimeError(name)
try:
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  state('waiting_for_GPU');fcntl.flock(lock,fcntl.LOCK_EX)
  run('fast',[str(repo/'build/logit_dump'),model,str(e/'fast'),'--chunk','2048','--from',str(root/'runs/logits-precise/step')],{'KAPITSA_PROFILE':'fast','KAPITSA_HOME':str(e/'home')})
  out=e/'mlx';out.mkdir(exist_ok=True)
  run('mlx',[py,str(root/'mlx_probe.py'),model,'dump',str(root/'runs/logits-precise/step'),str(out)])
 run('fast-zero',[py,str(repo/'tools/logit_compare.py'),str(root/'runs/logits-fast'),str(e/'fast'),'--min-top1','1','--max-mean-kl','0','--max-p99-kl','0','--max-nll-change','0','--json',str(e/'fast-zero.json')])
 run('mlx-zero',[py,str(repo/'tools/logit_compare.py'),str(root/'competitors-resume/mlx-raw/step.f32'),str(e/'mlx/step.f32'),'--vocab','248320','--min-top1','1','--max-mean-kl','0','--max-p99-kl','0','--json',str(e/'mlx-zero.json')])
 state('complete')
except Exception as ex:state('failed: '+str(ex));raise
