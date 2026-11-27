import datetime,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,time,traceback
P=pathlib.Path(__file__).resolve().parent;O=P/'m4-27b-quiet-v2';O.mkdir(exist_ok=False)
cfg=json.loads((P/'local.json').read_text());builds={k:pathlib.Path(v) for k,v in cfg['builds'].items()};model=cfg['models'][0]['path']
sys.path.insert(0,str(pathlib.Path(cfg['trees']['candidate'])/'bench/harness'));import gpu_hogs
state={'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting for preceding 27B/MoE checks','reason':'Original 27B A2 native bench flagged sustained foreign GPU activity during prefill; retain it and repeat its A/B pair under a fresh label.','runs':[]}
def save():(O/'results.json').write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 while not (P/'m4-35b-mtp/complete').exists():
  if (P/'m4-35b-mtp/failed').exists():raise RuntimeError('preceding MTP check failed; inspect it')
  time.sleep(5)
 old=json.loads((P/'m4-27b/STATE.json').read_text());state['identities']=old['identities']
 for tag,b in builds.items():assert hashlib.sha256((b/'kapitsa').read_bytes()).hexdigest()==old['identities'][tag]['binaries']['kapitsa']
 env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','KAPITSA_','MTL_','BENCH_'))};env.update(NI_LOCK_HELD='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
 with open('/tmp/kapitsa-gpu.lock','a') as lock:
  state['status']='waiting for GPU lock';save();fcntl.flock(lock,fcntl.LOCK_EX)
  for tag in ['base','candidate']:
   state['status']='waiting for quiet GPU before '+tag;save()
   quiet = gpu_hogs.wait_quiet(maxpct=0.1,quiet_s=30,timeout_s=600,exclude_roots=[os.getpid()])
   state.setdefault('quiet_checks',[]).append({'arm':tag,'result':quiet,'note':'gpu_hogs.diff_rows reports foreign processes above 0.5%; configured maxpct is 0.1%'});save()
   assert quiet[0], 'GPU did not become quiet'
   label='27b-repeat-'+tag;state['status']=label;save();env['KAPITSA_HOME']=str(O/('home-'+label));out=O/(label+'.json')
   cmd=[str(builds[tag]/'kapitsa'),'bench','--model',model,'--quick','--json','--out',str(out)]
   with (O/(label+'.log')).open('x') as log:
    cp=subprocess.Popen(cmd,cwd=cfg['trees'][tag],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);state['child_pid']=cp.pid;save()
    try:cp.wait(timeout=2400)
    except BaseException:
     os.killpg(cp.pid,signal.SIGTERM)
     try:cp.wait(timeout=25)
     except subprocess.TimeoutExpired:os.killpg(cp.pid,signal.SIGKILL);cp.wait()
     raise
   d=json.loads(out.read_text());r={'label':label,'argv':cmd,'exit':cp.returncode,'environment':{k:v for k,v in env.items() if k.startswith(('NI_','KAPITSA_'))},**{k:d[k] for k in ['kapitsa','model','results','conditions','flags','invalid']}};state['runs'].append(r);save()
 state['status']='complete; inspect both replacement arms and retain original flagged pair';state.pop('child_pid',None);save();(O/'complete').touch()
except BaseException as e:state['status']='failed';state['error']=repr(e);save();(O/'failed').write_text(traceback.format_exc());raise
