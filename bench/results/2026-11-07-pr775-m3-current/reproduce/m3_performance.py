import os,sys,pathlib,time,json,subprocess,fcntl,random,urllib.request,hashlib,contextlib
root=pathlib.Path(__file__).resolve().parent;repo=root/'candidate-current-pr775';e=root/'evidence-m3-current-performance';e.mkdir(exist_ok=True);sys.path.insert(0,str(repo/'bench/harness'))
import bench,gpu_hogs
for k in list(os.environ):
    if k.startswith(('NI_','NACE_','BENCH_','KAPITSA_')):os.environ.pop(k)
os.environ.update(NI_LOCK_HELD='1',NACE_GPU_LOCK_HELD='1',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1',NI_ANE_ASYNC='0',NI_ANE_KEEP_S='0',NI_NO_MTP='1')
model='/Users/bytedance/kapitsa-qmm9b-models-20261105/hub/models/qwen3.5-9b-4bit'
def state(phase):
    print(phase,flush=True);(e/'performance-status.json').write_text(json.dumps({'phase':phase,'time':time.time()}))
try:
    state('waiting_for_validation')
    while True:
        status=json.loads((root/'evidence-m3-current/status.json').read_text())['phase'] if (root/'evidence-m3-current/status.json').exists() else ''
        if status.startswith('failed'):raise RuntimeError(status)
        if status=='complete':break
        time.sleep(5)
    state('waiting_for_GPU')
    with open('/tmp/kapitsa-gpu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        os.chdir(repo)
        units='16'
        legs=[]
        for label,build in [('A1','9cca6b0d'),('B1','9cca6b0d'),('A2','9cca6b0d'),('B2','9cca6b0d')]:
            state(label);proc=None;mon=None;requests=[]
            cmd=[str(repo/'build/kapitsa'),'serve','--private','--model',model,'--model-id','qwen3.5-9b','--port','18776','--ctx','32768','--slots','1','--prefill-chunk','2048','--disk-cache','0','--dflash','none']
            en={**os.environ,'KAPITSA_HOME':str(e/'runtime'/('perf-'+label))}
            en['KAPITSA_PROFILE']='precise' if label.startswith('A') else 'fast'
            en['NI_ANE_UNITS']=units
            args=['--base-url','http://127.0.0.1:18776/v1','--model','qwen3.5-9b','--label',label,'--sizes','4000','--skip','decode,greedy,tools,agent','--cached-samples','0','--warmup','none','--out',str(e/('perf-'+label+'.json'))]
            original=bench.stream_chat
            def capture(*a,**kw):
                body=a[1] if len(a)>1 else kw.get('body')
                requests.append(body)
                return original(*a,**kw)
            bench.stream_chat=capture
            try:
                with open(e/('perf-'+label+'-server.log'),'w') as sf,open(e/('perf-'+label+'.log'),'w') as lf:
                    proc=subprocess.Popen(cmd,env=en,stdout=sf,stderr=subprocess.STDOUT)
                    deadline=time.time()+600
                    while True:
                        if proc.poll() is not None:raise RuntimeError('server exited')
                        try:
                            urllib.request.urlopen('http://127.0.0.1:18776/health',timeout=2).close();break
                        except OSError:
                            if time.time()>deadline:raise RuntimeError('server readiness timeout')
                            time.sleep(.5)
                    args+=['--pid',str(proc.pid)]
                    with contextlib.redirect_stdout(lf):
                        random.seed(20261106);bench.load_corpus();bench.warmup(bench.parse_args(args),False)
                        quiet,waited,busy=gpu_hogs.wait_quiet(quiet_s=15,timeout_s=300,exclude_roots=[proc.pid],own_max=5.0)
                        mon=gpu_hogs.ContentionMonitor([proc.pid]);mon.start()
                        t0=time.time();sys.argv=['bench.py',*args];random.seed(20261106);bench.main();t1=time.time()
                        mon.stop()
                    result=json.loads((e/('perf-'+label+'.json')).read_text())
                    telemetry={'quiet':quiet,'quiet_wait_seconds':waited,'busy':busy,'samples':mon.samples,'overall':gpu_hogs.summarize_samples(mon.samples,t0,t1),'phases':{k:gpu_hogs.summarize_samples(mon.samples,*v) for k,v in result['phases'].items()},'ignored_processes':sorted(gpu_hogs.IGNORE),'ambient_processes':sorted(gpu_hogs.AMBIENT)}
                    (e/('perf-'+label+'-telemetry.json')).write_text(json.dumps(telemetry,indent=2))
                    (e/('perf-'+label+'-requests.json')).write_text(json.dumps(requests,ensure_ascii=False,indent=2))
                    legs.append({'label':label,'build':build,'profile':en['KAPITSA_PROFILE'],'command':cmd,'result':result,'telemetry':telemetry,'request_sha256':[hashlib.sha256(json.dumps(b,sort_keys=True,ensure_ascii=False).encode()).hexdigest() for b in requests]})
                    (e/'performance-legs.json').write_text(json.dumps(legs,indent=2))
            finally:
                bench.stream_chat=original
                if mon and mon.is_alive():mon.stop()
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    state('complete')
except Exception as ex:
    state('failed: '+str(ex));raise
