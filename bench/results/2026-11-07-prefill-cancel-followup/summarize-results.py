"""Summarize only this run's completed evidence; retain every arm and failure."""
import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);out={}
p=r/'gpu/STATE.json'
if p.exists():
 d=json.loads(p.read_text());out['gpu_status']=d['status'];out['identities']=d.get('identities');out['models']=[]
 for m in d['models']:
  paths=m.get('logits',{}).get('paths',{});row={'name':m['name'],'logits':{k:{x:v.get(x) for x in ('records','bitwise','non_finite')} for k,v in paths.items()},'http':m.get('http'),'http_same_outcomes':m.get('http_same_outcomes'),'continuations':[{k:x.get(k) for k in ('label','tokens','text_nonempty','eos_count','before_first_eos','ids_sha256','contention')} for x in m.get('continuations',[])],'native':[]}
  for b in m.get('benchmarks',[]):
   v=b['results'];row['native'].append({'label':b['label'],'exit':b['exit'],'source':b['kapitsa'],'sampled_tps':v['decode']['median_tps'],'greedy_tps':v['decode_greedy']['median_tps'],'prefill':v['prefill'],'peak_memory_gb':v['peak_memory_gb'],'conditions':b['conditions'],'flags':b['flags'],'invalid':b['invalid']})
  out['models'].append(row)
p=r/'cancellation-ab/STATE.json'
if p.exists():
 d=json.loads(p.read_text());out['cancel_status']=d['status'];out['cancellation_arms']=[]
 for arm in d['arms']:
  rows=arm['telemetry'];retry=next(x for x in rows if x['event']=='request' and x['name']=='cancel-reply');out['cancellation_arms'].append({'label':arm['label'],'exit':arm['exit'],'failures':[x['name'] for x in arm['checks'] if not x['pass']],'retry_seconds':retry['seconds'],'retry_usage':retry['usage'],'rss_before_retry':retry['rss_before'],'rss_after_retry':retry['rss_after'],'cancellations':[x for x in rows if x['event']=='cancel_confirmed'],'contention':arm['contention']})
p=r/'state-cases/STATE.json'
if p.exists():out['state_cases']=json.loads(p.read_text())
p=r/'state-cases-v2/STATE.json'
if p.exists():out['state_cases_v2']=json.loads(p.read_text())
p=r/'state-cases-ramfs-v3/STATE.json'
if p.exists():out['state_cases_ramfs_v3']=json.loads(p.read_text())
p=r/'state-cases-image-ram-v4/STATE.json'
if p.exists():out['state_cases_image_ram_v4']=json.loads(p.read_text())
print(json.dumps(out,indent=2))
