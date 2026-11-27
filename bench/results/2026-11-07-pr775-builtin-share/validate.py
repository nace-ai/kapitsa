import pathlib,subprocess,os,json,time,fcntl,hashlib,sys
root=pathlib.Path('/Users/bytedance/Documents/ChatGPT/kapitsa');repo=root/'ane-pr775';out=root/'work/pr775/builtin-share/run';out.mkdir(parents=True,exist_ok=True)
old=root/'work/pr775/m4-local/rebase-3f3b/validation-versioned';model=pathlib.Path('/Users/bytedance/.cache/kapitsa/models/qwen3.5-9b-4bit');venv=root/'work/pr775/m4-local/venv/bin'
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_','MLX_'))};env['PATH']=str(venv)+os.pathsep+env['PATH'];env['VK_EVIDENCE']=str(out/'vk')
def state(s):
 (out/'status.json').write_text(json.dumps({'phase':s,'time':time.time()})+'\n');print(s,flush=True)
def run(label,cmd,extra=None,allow_failure=False):
 state(label)
 with (out/(label+'.log')).open('w') as f:p=subprocess.run(list(map(str,cmd)),cwd=repo,env={**env,**(extra or {})},stdout=f,stderr=subprocess.STDOUT)
 (out/(label+'-exit.json')).write_text(json.dumps({'command':list(map(str,cmd)),'returncode':p.returncode,'extra_env':extra or {}},indent=2)+'\n')
 if p.returncode and not allow_failure:raise RuntimeError(label+' failed')
 return p.returncode
try:
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();assert head=='92689bea1bce8c956b84815fcebc72cf2d5088d0'
 run('configure',[venv/'cmake','-S','.', '-B','build','-DCMAKE_BUILD_TYPE=Release'])
 run('build',['.dsh/skills/verify-kapitsa/scripts/vk','build','kapitsa','logit_dump','ane_segment_check','device_profile_test'])
 version=subprocess.check_output([repo/'build/kapitsa','--version'],text=True).strip();assert head[:8] in version
 (out/'identity.json').write_text(json.dumps({'head':head,'version':version,'model':str(model),'old_evidence':str(old),'units_override':None,'calibration_file_exists':(pathlib.Path.home()/'.cache/nace-infer-models/ane-cache/calibration.txt').exists()},indent=2)+'\n')
 assert not (pathlib.Path.home()/'.cache/nace-infer-models/ane-cache/calibration.txt').exists()
 for label,cmd in [('device-profile',['build/device_profile_test']),('segments-cpu',['build/ane_segment_check']),('policy',[sys.executable,'tests/test_ane_profile.py','build/ane_segment_check'])]:run(label,cmd)
 state('waiting_for_GPU')
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  verified=[]
  for entry in json.loads((old/'weight-identity.json').read_text()):
   f=model/entry['file'];h=hashlib.sha256()
   with f.open('rb') as fp:
    for b in iter(lambda:fp.read(8*1024**2),b''):h.update(b)
   assert h.hexdigest()==entry['sha256'];verified.append({'file':entry['file'],'sha256':h.hexdigest()})
  (out/'weight-identity.json').write_text(json.dumps(verified,indent=2)+'\n')
  gpu={'NI_LOCK_HELD':'1','NACE_GPU_LOCK_HELD':'1','NI_ANE_ASYNC':'0','NI_ANE_KEEP_S':'0','NI_NO_MTP':'1','KAPITSA_PROFILE':'fast','KAPITSA_HOME':str(out/'home-fast'),'KAPITSA_NO_AUTO_UPDATE':'1','KAPITSA_NO_PROMPT':'1'}
  run('m4-fast',[repo/'build/logit_dump',model,out/'logits-fast','--chunk','2048','--from',old/'logits-precise/step'],gpu)
  meta=json.loads((out/'logits-fast/step/meta.json').read_text());assert meta['ane']['evals']>0 and meta['ane']['reruns']==0
  (out/'policy-runtime.json').write_text(json.dumps(meta,indent=2)+'\n')
 run('old-new-fast',[sys.executable,'tools/logit_compare.py',old/'logits-fast',out/'logits-fast','--min-top1','1','--max-mean-kl','0','--json',out/'old-new-fast.json'])
 rc=run('cpu',['.dsh/skills/verify-kapitsa/scripts/vk','cpu-tests'],allow_failure=True)
 for label,cmd in [('lint',[venv/'prek','run','--all-files']),('profile-gates',[sys.executable,'tests/test_profile_gates.py']),('comment-gates',[sys.executable,'tests/test_comment_gates.py']),('docs-links',[sys.executable,'tests/test_docs_links.py'])]:run(label,cmd)
 state('complete' if rc==0 else 'complete_with_CPU_failure')
except Exception as e:state('failed: '+str(e));raise
