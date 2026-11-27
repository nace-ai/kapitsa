#!/usr/bin/env python3
"""#325 verdict. A leg COUNTS only if every rule holds; anything unparsed is VOID (never dropped silently).
VOID when: any harness flag or invalid run; any phase flagged by the contention sampler; overall foreign GPU mean > 3%
(or contention data missing); power not proven AC (ac.log needs a sample <=90 s before start and >=1 after end, all 'AC Power').
usage: valid.py RESULTS_DIR [--ac ac.log] [--selftest]"""
import json,glob,sys,os,datetime as dt
MAXMEAN=3.0
def ts(s): return dt.datetime.strptime(s,'%Y-%m-%d %H:%M:%S')
def load_ac(p):
    out=[]
    if os.path.exists(p):
        for l in open(p):
            try: out.append((dt.datetime.strptime(l[:19],'%Y-%m-%dT%H:%M:%S'),'AC Power' in l))
            except ValueError: pass
    return out
def judge(r,ac):
    why=[]
    try: a,b=ts(r['when']),ts(r['when_end'])
    except Exception as e: return ['unparsed start/end time']
    fl=(r.get('flags') or [])+(r.get('invalid') or [])
    if 'flags' not in r or 'invalid' not in r: why.append('flags/invalid missing')
    if fl: why.append('harness flags: '+'; '.join(x[:50] for x in fl))
    c=r.get('contention')
    if not c or 'overall' not in c or not c.get('phases'): why.append('contention data missing')
    else:
        bad=[k for k,v in c['phases'].items() if v.get('flagged') or v.get('mean_foreign_gpu_pct',99)>5]
        if bad: why.append('contention in phase '+','.join(bad))
        if c['overall'].get('mean_foreign_gpu_pct',99)>MAXMEAN: why.append('foreign GPU mean %.1f%% > %.0f%%'%(c['overall'].get('mean_foreign_gpu_pct',99),MAXMEAN))
    win=[x for x in ac if a-dt.timedelta(seconds=90)<=x[0]<=b+dt.timedelta(seconds=90)]
    if not any(x[0]<=a for x in win) or not any(x[0]>=b for x in win): why.append('no AC samples around the leg')
    elif not all(x[1] for x in win): why.append('on battery during the leg')
    return why
def verdicts(d,ac):
    rows=[]
    for f in sorted(glob.glob(d+'/results/*.json')):
        try: r=json.load(open(f)); lab=r['label']
        except Exception: rows.append((os.path.basename(f),None,['unreadable json'])); continue
        rows.append((os.path.basename(f),r,judge(r,ac)))
    return rows
def main():
    d=sys.argv[1]; acp=sys.argv[sys.argv.index('--ac')+1] if '--ac' in sys.argv else os.path.join(os.path.dirname(os.path.abspath(__file__)),'ac.log')
    for n,r,why in verdicts(d,load_ac(acp)):
        dec=r and r.get('decode_summary',{}).get('median_tps')
        print(n[:46],'dec',dec,'->','COUNTS' if not why else 'VOID: '+' | '.join(why))
def selftest():
    import copy,tempfile
    base=json.load(open('harness-runs/results/q38-27b-mlxlm-20261103-181528.json'))
    clean=copy.deepcopy(base); clean['flags']=[]
    for v in clean['contention']['phases'].values(): v['flagged']=False; v['mean_foreign_gpu_pct']=1.0
    clean['contention']['overall']['mean_foreign_gpu_pct']=1.0
    a=ts(clean['when']); ac=[(a-dt.timedelta(seconds=30),True),(ts(clean['when_end'])+dt.timedelta(seconds=30),True)]
    bat=[(ac[0][0],True),(a+dt.timedelta(seconds=60),False),(ac[1][0],True)]
    cases={'clean+AC':(clean,ac,True),'real flagged leg':(base,ac,False),'battery mid-leg':(clean,bat,False),'no ac.log':(clean,[],False)}
    nc=copy.deepcopy(clean); nc.pop('contention'); cases['no contention data']=(nc,ac,False)
    bt=copy.deepcopy(clean); bt['when']='garbage'; cases['unparsed time']=(bt,ac,False)
    ph=copy.deepcopy(clean); ph['contention']['phases']['decode']['flagged']=True; cases['one phase flagged']=(ph,ac,False)
    mean=copy.deepcopy(clean); mean['contention']['overall']['mean_foreign_gpu_pct']=3.5; cases['mean 3.5%']=(mean,ac,False)
    bad=0
    for k,(r,a_,want) in cases.items():
        got=not judge(r,a_); print('%-20s %s'%(k,'ok' if got==want else 'FAIL (got %s)'%('COUNTS' if got else 'VOID'))); bad+=got!=want
    sys.exit(bad)
if __name__=='__main__':
    selftest() if '--selftest' in sys.argv else main()
