import argparse,pathlib,os,sys,subprocess,json,time,fcntl,hashlib
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--old',type=pathlib.Path,required=True);p.add_argument('--competitors',type=pathlib.Path,required=True);p.add_argument('--model',type=pathlib.Path,required=True);p.add_argument('--cmake',required=True);p.add_argument('--units');a=p.parse_args()
a.out.mkdir(exist_ok=True,parents=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_','MLX_'))};env.update(NI_ANE_ASYNC='0',NI_ANE_KEEP_S='0',NI_NO_MTP='1',NI_LOCK_HELD='1',NACE_GPU_LOCK_HELD='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
env['PATH']=str(pathlib.Path(a.cmake).parent)+os.pathsep+env['PATH'];env['VK_EVIDENCE']=str(a.out/'vk')
if a.units:env['NI_ANE_UNITS']=a.units
def state(s):
 (a.out/'status.json').write_text(json.dumps({'phase':s,'time':time.time()}));print(s,flush=True)
def run(label,cmd,extra=None,gate=False):
 state(label);en={**env,**(extra or {})}
 with (a.out/(label+'.log')).open('w') as f:r=subprocess.run(list(map(str,cmd)),cwd=a.repo,env=en,stdout=f,stderr=subprocess.STDOUT)
 (a.out/(label+'-exit.json')).write_text(json.dumps({'command':list(map(str,cmd)),'returncode':r.returncode,'env':{k:v for k,v in en.items() if k.startswith(('NI_','KAPITSA_'))}},indent=2)+'\n')
 if r.returncode and not gate:raise RuntimeError(label+' failed')
 return r.returncode
try:
 if not a.units:
  state('waiting_for_raw_matrix_relocation')
  while not (a.out.parent/'validation-resume/matrix-location.json').exists():time.sleep(5)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip();assert head=='50fadecc932863bff4d3f81a7a20d3913be3e15b'
 calibration=pathlib.Path.home()/'.cache/nace-infer-models/ane-cache/calibration.txt'
 if not a.units:assert not calibration.exists(),'M4 calibration file appeared; built-in share condition not satisfied'
 (a.out/'identity.json').write_text(json.dumps({'head':head,'model':str(a.model),'units_override':a.units,'calibration_file_exists':calibration.exists(),'old_evidence':str(a.old),'external_engine_evidence_reused':str(a.competitors)},indent=2)+'\n')
 run('configure',[a.cmake,'-S','.', '-B','build','-DCMAKE_BUILD_TYPE=Release'])
 run('build',[a.repo/'.dsh/skills/verify-kapitsa/scripts/vk','build','kapitsa','logit_dump','ane_segment_check','device_profile_test'])
 version=subprocess.check_output([a.repo/'build/kapitsa','--version'],text=True).strip();assert head[:8] in version,version
 (a.out/'binary-version.txt').write_text(version+'\n')
 run('device-profile',[a.repo/'build/device_profile_test'])
 run('policy',[sys.executable,'tests/test_ane_profile.py','build/ane_segment_check'])
 state('waiting_for_GPU')
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);state('verifying_weights')
  old_identity=json.loads((a.old/'identity.json').read_text());verified=[]
  for entry in old_identity['files']:
   f=a.model/entry['file'];h=hashlib.sha256()
   with f.open('rb') as fp:
    while chunk:=fp.read(8*1024**2):h.update(chunk)
   actual=h.hexdigest();assert actual==entry['sha256'],f
   verified.append({'file':entry['file'],'sha256':actual,'match':True})
  (a.out/'weight-identity.json').write_text(json.dumps(verified,indent=2)+'\n')
  run('segments',[a.repo/'build/ane_segment_check','--gpu'],{'MTL_SHADER_VALIDATION':'1','MTL_DEBUG_LAYER':'1'})
  for profile in ('precise','fast'):
   run(profile,[a.repo/'build/logit_dump',a.model,a.out/('logits-'+profile),'--chunk','2048','--from',a.old/'logits-precise/step'],{'KAPITSA_PROFILE':profile,'KAPITSA_HOME':str(a.out/('home-'+profile))})
  meta={x:json.loads((a.out/('logits-'+x)/'step/meta.json').read_text()) for x in ('precise','fast')}
  assert meta['precise']['ane']['evals']==0 and meta['fast']['ane']['evals']>0 and meta['fast']['ane']['reruns']==0
  (a.out/'policy-runtime.json').write_text(json.dumps(meta,indent=2)+'\n')
 comparisons={}
 for profile in ('precise','fast'):
  comparisons[profile]=run('old-new-'+profile,[sys.executable,a.repo/'tools/logit_compare.py',a.old/('logits-'+profile),a.out/('logits-'+profile),'--min-top1','1','--max-mean-kl','0','--json',a.out/('old-new-'+profile+'.json')],gate=True)
 for engine in ('mlx','llama'):
  run('compare-'+engine,[sys.executable,a.repo/'tools/logit_compare.py',a.out/'logits-precise',a.competitors/('logits-'+engine),'--no-gate','--json',a.out/('compare-'+engine+'.json')])
 run('compare-fast',[sys.executable,a.repo/'tools/logit_compare.py',a.out/'logits-precise',a.out/'logits-fast','--no-gate','--json',a.out/'compare-fast.json'])
 engine=json.loads((a.competitors/'speed-selection.json').read_text())['faster'];bar=json.loads((a.out/('compare-'+engine+'.json')).read_text())['paths'];fast=json.loads((a.out/'compare-fast.json').read_text())['paths'];gates=[]
 for path,b in bar.items():
  f=fast[path];assert f['records']==b['records']
  rc=run('fast-gate-'+path,[sys.executable,a.repo/'tools/logit_compare.py',a.out/'logits-precise'/path,a.out/'logits-fast'/path,'--min-top1',str(b['top1']),'--max-mean-kl',str(b['kl_mean']),'--json',a.out/('fast-gate-'+path+'.json')],gate=True)
  gates.append({'path':path,'returncode':rc,'fast_flips':f['flips'],'bar_flips':b['flips'],'fast_mean_kl':f['kl_mean'],'bar_mean_kl':b['kl_mean']})
 (a.out/'gate-summary.json').write_text(json.dumps({'old_new_gate_exit_codes':comparisons,'external_engine':engine,'gates':gates},indent=2)+'\n')
 state('complete')
except Exception as ex:state('failed: '+str(ex));raise
