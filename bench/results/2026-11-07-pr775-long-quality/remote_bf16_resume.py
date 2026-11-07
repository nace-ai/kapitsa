import pathlib,os,json,subprocess,fcntl,time,shlex
root=pathlib.Path(__file__).resolve().parent;repo=root/'candidate-review-policy';e=root/'evidence-m4-long-bf16-resume';e.mkdir(exist_ok=True);py=str(root/'.venv/bin/python');model=str(root/'models/qwen3.5-9b-bf16');inputs=root/'m4-long-inputs'
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_','MLX_'))}
env.update(KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
def state(s):print(s,flush=True);(e/'status.json').write_text(json.dumps({'phase':s,'time':time.time()}))
def run(label,cmd):
 state('waiting_for_GPU: '+label)
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state(label)
  with (e/(label+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=repo,env={**env,'NI_LOCK_HELD':'1','NACE_GPU_LOCK_HELD':'1'},stdout=f,stderr=subprocess.STDOUT)
 (e/(label+'-exit.json')).write_text(json.dumps({'command':cmd,'returncode':r.returncode},indent=2))
 if r.returncode:raise RuntimeError(label+' failed')
try:
 run('bf16-tf',[py,'bench/quality/eval_mlx.py',model,'--f32','--topk','4096','--dump-ref',str(e/'bf16.ref'),*['@'+str(inputs/'long-quality'/f'case-{i}.txt') for i in range(4)]])
 server=[py,'-m','mlx_lm.server','--model',model,'--host','127.0.0.1','--port','18778']
 run('bf16-reasoning',[py,'bench/reasoning/reasoning.py','run','--problems',str(inputs/'long-reasoning.jsonl'),'--max-tokens','32768','--samples','1','--label','bf16-m3','--base','http://127.0.0.1:18778/v1','--out',str(e/'bf16-reasoning.jsonl'),'--serve',shlex.join(server)])
 state('complete')
except Exception as ex:state('failed: '+str(ex));raise
