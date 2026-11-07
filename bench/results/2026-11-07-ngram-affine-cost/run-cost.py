import datetime,fcntl,hashlib,json,os,pathlib,signal,subprocess,sys,threading,time,traceback
R=pathlib.Path(sys.argv[1]).resolve();MODEL=pathlib.Path(sys.argv[2]).resolve();O=R/'cost-results';O.mkdir(exist_ok=False)
s={'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'waiting for measurement lock','runs':[],'scope':'CPU-only resident trained-table cost decomposition; no GPU model, no cold I/O or full inference claim'}
child=None;lock=None

def save():(O/'STATE.json').write_text(json.dumps(s,indent=2)+'\n')
def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
def conditions():return {'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'power':command(['pmset','-g','batt']),'thermal_limits':command(['pmset','-g','therm']),'vm_stat':command(['vm_stat']),'loadavg':list(os.getloadavg())}
def stop():
 global child
 if child and child.poll() is None:
  os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
 child=None
def signal_exit(sig,frame):raise RuntimeError('signal '+str(sig))
signal.signal(signal.SIGTERM,signal_exit);signal.signal(signal.SIGINT,signal_exit)
def monitor(event,samples):
 while not event.wait(0.5):
  try:
   raw=command(['ps','-axo','pid=,ppid=,%cpu=']);rows=[tuple(float(x) for x in l.split()) for l in raw.splitlines() if len(l.split())==3];own={os.getpid()}
   for _ in range(8):own|={int(p) for p,pp,c in rows if int(pp) in own}
   samples.append({'time':time.time(),'own_cpu_pct':sum(c for p,pp,c in rows if int(p) in own),'foreign_cpu_pct':sum(c for p,pp,c in rows if int(p) not in own),'loadavg':list(os.getloadavg())})
  except Exception as e:samples.append({'error':repr(e)})
save()
try:
 s['hardware']={key:command(['sysctl','-n',key]).strip() for key in ['hw.model','hw.memsize','machdep.cpu.brand_string']};s['macos']=command(['sw_vers']);s['identity']=json.loads((R/'identities.json').read_text());s['fixture']=json.loads((R/'fixture/manifest.json').read_text());save()
 lock=open('/tmp/kapitsa-gpu.lock','a');deadline=time.monotonic()+600
 while True:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:
   assert time.monotonic()<deadline,'lock wait exceeded ten minutes'
   time.sleep(2)
 for mode in ['default','no-prefetch']:
  for arm,tag in enumerate(['base','candidate','base','candidate']):
   label=f'{mode}-{arm}-{tag}';s['status']=label;save();binary=R/('cost-'+tag);assert hashlib.sha256(binary.read_bytes()).hexdigest()==s['identity'][tag]['binary_sha256']
   env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','KAPITSA_','MTL_','BENCH_'))};env['NI_LOCK_HELD']='1'
   if mode=='no-prefetch':env['NI_PLE_PREFETCH']='0'
   cmd=[str(binary),str(MODEL),str(R/'fixture/corpus.json'),mode];before=conditions();samples=[];ev=threading.Event();thread=threading.Thread(target=monitor,args=(ev,samples));thread.start()
   try:
    with (O/(label+'.log')).open('x') as f:
     child=subprocess.Popen(cmd,cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);s['child_pid']=child.pid;save();rc=child.wait(timeout=180)
   finally:stop();ev.set();thread.join()
   row={'label':label,'argv':cmd,'environment':{k:v for k,v in env.items() if k.startswith('NI_')},'exit':rc,'before':before,'after':conditions(),'cpu_monitor':samples}
   row['cases']=[json.loads(l) for l in (O/(label+'.log')).read_text().splitlines() if l.startswith('{')]
   s['runs'].append(row);save();assert rc==0 and len(row['cases'])==(12 if mode=='default' else 4),'failed measurement'
 s['status']='complete';save()
except BaseException as e:s['status']='failed';s['error']=repr(e);s['traceback']=traceback.format_exc();save();raise
finally:
 stop()
 if lock:fcntl.flock(lock,fcntl.LOCK_UN);lock.close();s['lock_released']=True
 s['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
