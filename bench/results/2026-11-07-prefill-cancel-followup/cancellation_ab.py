"""Interleaved cancellation regression with client timings and process RSS samples."""
import datetime, fcntl, hashlib, importlib.util, json, os, pathlib, subprocess, sys, time
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text())
p=pathlib.Path(cfg['output']);p.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('cancel_test',cfg['test']);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
sys.path.insert(0,str(pathlib.Path(cfg['test']).parents[1]/'bench/harness'))
import gpu_hogs
state={'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting','config':cfg,'test_sha256':hashlib.sha256(pathlib.Path(cfg['test']).read_bytes()).hexdigest(),'arms':[]}
def save():
 tmp=p/'STATE.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p/'STATE.json')
def rss(pid):
 return int(subprocess.check_output(['ps','-p',str(pid),'-o','rss='],text=True).strip())*1024
original_server=mod.Server
original_request=mod.request
active=None
telemetry=[]
class TrackedServer(original_server):
 def __init__(self,*a,**kw):
  global active
  super().__init__(*a,**kw);active=self
  telemetry.append({'event':'server_ready','pid':self.process.pid,'rss_bytes':rss(self.process.pid),'command':self.command})
 def cancel(self,*a,**kw):
  begin=time.monotonic();value=super().cancel(*a,**kw)
  telemetry.append({'event':'cancel_confirmed','name':a[2],'seconds':time.monotonic()-begin,'rss_bytes':rss(self.process.pid)})
  return value
 def stop(self):
  global active
  super().stop();active=None
def measured_request(*a,**kw):
 before=rss(active.process.pid);start=time.monotonic();value=original_request(*a,**kw)
 telemetry.append({'event':'request','name':a[2],'seconds':time.monotonic()-start,'rss_before':before,'rss_after':rss(active.process.pid),'usage':value.get('usage'),'cancelled':value['cancelled']})
 return value
mod.Server=TrackedServer;mod.request=measured_request
save()
with open(os.environ.get('BENCH_GPU_LOCK','/tmp/kapitsa-gpu.lock'),'a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);os.environ['NI_LOCK_HELD']='1'
 for model in cfg['models']:
  for arm,tag in enumerate(['base','candidate','base','candidate']):
   label=f"{model['name']}-{arm}-{tag}";state['status']=label;save()
   quiet=gpu_hogs.wait_quiet(maxpct=.1,quiet_s=15,timeout_s=600,exclude_roots=[os.getpid()]);assert quiet[0],quiet
   telemetry=[];monitor=gpu_hogs.ContentionMonitor([os.getpid()],interval=.5);monitor.start()
   sys.argv=['test_prefill_cancel.py','--cli',cfg['binaries'][tag],'--reference',cfg['binaries']['base'],'--model',model['path'],'--output',str(p/label),'--port','8298']
   try:code=mod.main()
   finally:monitor.stop()
   checks=json.loads((p/label/'summary.json').read_text())['checks'];failures=[x['name'] for x in checks if not x['pass']]
   row={'label':label,'exit':code,'checks':checks,'telemetry':telemetry,'quiet':quiet,'contention':gpu_hogs.summarize_samples(monitor.samples),'samples':monitor.samples}
   state['arms'].append(row);save()
   expected=['retry keeps the completed prefix'] if tag=='base' else []
   assert failures==expected,(label,failures)
state['status']='complete';save()
