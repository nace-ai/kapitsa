import json, re, glob, subprocess, statistics as st, sys
O='/tmp/k350os/out'; rows=[]; log=open('/Users/alisheramantay/.agents/jobs/k350-selonline.log').read()
pat=re.compile(r'dflash decode .*?: (\d+) tok in ([\d.]+) ms .*?, (\d+) cycles, ([\d.]+) tok/cycle')
for i in range(19):
    r={'req':i}
    for arm in ('ctl','learn'):
        try: e=open(f'{O}/r{i}.{arm}.err').read(); ids=open(f'{O}/r{i}.{arm}.out').read()
        except FileNotFoundError: continue
        m=pat.search(e)
        if not m: continue
        r[arm]={'tok':int(m[1]),'ms':float(m[2]),'cycles':int(m[3]),'tpc':float(m[4])}
        r[arm+'_ids']=hash(ids.split('ids:')[-1])
    if 'ctl' in r and 'learn' in r: rows.append(r)
def agg(a): return sum(r[a]['tok'] for r in rows)/sum(r[a]['cycles'] for r in rows), sum(r[a]['tok'] for r in rows)/sum(r[a]['ms'] for r in rows)*1000
c,l=agg('ctl'),agg('learn')
v={'task':'#350 online selector codebook update, real 35B-A3B on M5 Pro (session of 19 OpenCode agent requests, state carried)',
 'commit':'2f6aa0f8 (k350-weaver-online)','model':'qwen3.6-35b-a3b-4bit + dflash2 drafter',
 'argv':'NI_SEL_ONLINE={0|0.003} NI_SEL_STATE=<file> build/ni_run <model> @r<i>.ids <n> 0 --dflash auto --ignore-eos (mode 2 both arms)',
 'n_requests':len(rows),'outputs_identical':sum(r['ctl_ids']==r['learn_ids'] for r in rows),
 'before':{'tok_per_cycle':round(c[0],4),'decode_tok_s':round(c[1],2)},'after':{'tok_per_cycle':round(l[0],4),'decode_tok_s':round(l[1],2)},
 'delta_tok_per_cycle_pct':round(100*(l[0]/c[0]-1),2),'delta_tok_s_pct':round(100*(l[1]/c[1]-1),2),
 'replay_prediction_pct':8.6,
 'per_request':[{'req':r['req'],'n':r['ctl']['tok'],'tpc_ctl':r['ctl']['tpc'],'tpc_learn':r['learn']['tpc'],'tps_ctl':round(r['ctl']['tok']/r['ctl']['ms']*1000,1),'tps_learn':round(r['learn']['tok']/r['learn']['ms']*1000,1)} for r in rows],
 'logs':{'job':'~/.agents/jobs/k350-selonline.log','legs':O+'/r<i>.{ctl,learn}.{out,err}'}}
json.dump(v,open('/tmp/k350os/VERDICT.json','w'),indent=1); print(json.dumps({k:v[k] for k in v if k!='per_request'},indent=1))
