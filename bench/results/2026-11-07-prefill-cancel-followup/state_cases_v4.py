"""Exercise cancellation followed by memory eviction and SSD restart, with exact request logs."""
import base64, datetime, fcntl, hashlib, importlib.util, json, os, pathlib, re, struct, subprocess, sys, tempfile, time, zlib
from unittest.mock import patch
cfg=json.loads(pathlib.Path(sys.argv[1]).read_text());out=pathlib.Path(cfg['output']);out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('cancel_test',cfg['test']);t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
corpus=(pathlib.Path(cfg['test']).parents[1]/'bench/harness/corpus/bench_corpus.txt').read_text()
state={'pid':os.getpid(),'config':cfg,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':[]}
def save():
 tmp=out/'STATE.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'STATE.json')
def image():
 def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
 raw=b''.join(b'\0'+bytes([150,40,90])*32 for _ in range(32));png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>2I5B',32,32,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
 return 'data:image/png;base64,'+base64.b64encode(png).decode()
def first(with_image=False):
 content='Read this source:\n'+corpus[:8500]+'\nName one function in one sentence.'
 if with_image:content=[{'type':'image_url','image_url':{'url':image()}},{'type':'text','text':content}]
 return [{'role':'system','content':'You are a careful code reviewer, cancellation test.'},{'role':'user','content':content}]
def start(binary,home,log,extra=()):
 real=subprocess.Popen
 def launch(cmd,*a,**kw):
  cmd=list(cmd)+list(extra);kw['env']['NI_DISK_CACHE_IDLE_MS']='600000'
  log.with_suffix('.command.json').write_text(json.dumps({'argv':cmd,'environment':{k:v for k,v in kw['env'].items() if k.startswith(('NI_','KAPITSA_'))}},indent=2)+'\n')
  return real(cmd,*a,**kw)
 with patch.object(t.subprocess,'Popen',launch):return t.Server(pathlib.Path(binary),cfg['model'],8298,home,log)
def seed_and_cancel(server,p,with_image=False):
 f=first(with_image);seed=t.request(8298,p,'seed',f);assert seed['text'].strip()
 prefix=f+[{'role':'assistant','content':seed['text']}];time.sleep(1)
 short=t.request(8298,p,'short',prefix+[{'role':'user','content':'Name one possible failure in one sentence.'}]);assert t.cached(short)>=64
 request=prefix+[{'role':'user','content':'Review these additional files:\n'+corpus[10000:65000]+'\nName another function in one sentence.'}]
 c=server.cancel(8298,p,'cancel',request);cache=c['progress'][0]['cache'];assert cache>=64
 return request,cache
save()
with open(os.environ.get('BENCH_GPU_LOCK','/tmp/kapitsa-gpu.lock'),'a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);os.environ['NI_LOCK_HELD']='1'
 with tempfile.TemporaryDirectory(prefix='kapitsa-cancel-state-') as tmp:
  for case in ['image-ram-only']:
   replies={}
   for tag,binary in cfg['binaries'].items():
    label=case+'-'+tag;p=out/label;p.mkdir();home=pathlib.Path(tmp)/label
    extra=(['--slots','2','--ctx','16384','--mem-budget','4.9','--prefill-chunk','1024'] if case=='memory-eviction' else ['--dflash','none'] if case=='mtp-cancel' else ['--disk-cache','0'])
    state['status']=label;save();s=None
    try:
     s=start(binary,home,p/'server-first.log',extra);request,cache=seed_and_cancel(s,p,case.startswith('image'))
     if case=='image-ram-only':
      mark=len(s.path.read_text());reply=t.request(8298,p,'retry',request)
      row={'label':label,'initial_cache':cache,'retry_cache':t.cached(reply),'text':reply['text'],'drafter':reply['usage'].get('kapitsa',{}).get('kflash'),'checks':{'nonempty':bool(reply['text'].strip()),'candidate_prefix_retained':t.cached(reply)>=cache if tag=='candidate' else True}}
      row['image_log']=s.path.read_text()[mark:]
      row['checks']['candidate_cached_image_not_reencoded']=('0 encoded' in row['image_log'] and 'in the cached prefix' in row['image_log']) if tag=='candidate' else True
     elif case=='memory-eviction':
      mark=len(s.path.read_text());other=[{'role':'system','content':'A separate repository review.'},{'role':'user','content':corpus[:55000]+'\nName a function.'}]
      b=t.request(8298,p,'other',other);assert b['text'].strip();drop=[line for line in s.path.read_text()[mark:].splitlines() if 'memory: dropping the KV cache' in line]
      reply=t.request(8298,p,'retry',request)
      row={'label':label,'initial_cache':cache,'retry_cache':t.cached(reply),'memory_drops':drop,'text':reply['text'],'checks':{'nonempty':bool(reply['text'].strip()),'candidate_eviction_exercised':bool(drop) if tag=='candidate' else True}}
     else:
      s.stop();s=None;s=start(binary,home,p/'server-restart.log',extra);reply=t.request(8298,p,'retry',request)
      row={'label':label,'initial_cache':cache,'retry_cache':t.cached(reply),'text':reply['text'],'checks':{'nonempty':bool(reply['text'].strip()),'candidate_prefix_survives_restart':t.cached(reply)>=cache if tag=='candidate' else True}}
      if case.startswith('image'):
       row['cached_image_not_reencoded']='0 encoded' in s.path.read_text() and 'in the cached prefix' in s.path.read_text()
       row['checks']['candidate_cached_image_not_reencoded']=row['cached_image_not_reencoded'] if tag=='candidate' else True
     replies[tag]=reply['text'];state['cases'].append(row);save()
    except Exception as error:
     state['cases'].append({'label':label,'error':str(error),'checks':{'completed':False}});save()
    finally:
     if s:s.stop()
   equal='base' in replies and 'candidate' in replies and replies['base']==replies['candidate'];state.setdefault('comparisons',[]).append({'case':case,'same_returned_text':equal});save()
state['status']='complete';state['all_checks_pass']=all(all(x['checks'].values()) for x in state['cases']) and all(x['same_returned_text'] for x in state['comparisons']);save()
sys.exit(0 if state['all_checks_pass'] else 1)
