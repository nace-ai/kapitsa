import pathlib,json,time,subprocess
root=pathlib.Path(__file__).resolve().parent
repo=root/'candidate-current-pr775'
e=root/'evidence-m3-current-competitors'
while True:
 s=json.loads((e/'status.json').read_text())['phase']
 if s.startswith('failed'):raise RuntimeError(s)
 if s=='complete':break
 time.sleep(5)
engine=json.loads((e/'speed-selection.json').read_text())['faster']
bar=json.loads((e/('compare-'+engine+'.json')).read_text())['paths']
fast=json.loads((e/'compare-fast.json').read_text())['paths']
results=[];arithmetic={}
for path,b in bar.items():
 f=fast[path];assert f['records']==b['records']
 arithmetic[path]={'fast_flips':f['flips'],'bar_flips':b['flips'],'fast_mean_kl':f['kl_mean'],'bar_mean_kl':b['kl_mean'],'margin_mean_kl':b['kl_mean']-f['kl_mean'],'pass':f['flips']<=b['flips'] and f['kl_mean']<=b['kl_mean']}
 cmd=[str(root/'.venv/bin/python'),str(repo/'tools/logit_compare.py'),str(root/'evidence-m3-current/logits-precise'/path),str(root/'evidence-m3-current/logits-fast'/path),'--min-top1',str(b['top1']),'--max-mean-kl',str(b['kl_mean']),'--json',str(e/('fast-gate-'+path+'.json'))]
 with (e/('fast-gate-'+path+'.log')).open('w') as out:p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
 results.append({'path':path,'command':cmd,'returncode':p.returncode,'bar_from':'same-Mac measured faster engine '+engine+'; no floor modification'})
(e/'fast-gates.json').write_text(json.dumps(results,indent=2)+'\n')
(e/'admission-arithmetic.json').write_text(json.dumps(arithmetic,indent=2)+'\n')
cmd=[str(root/'.venv/bin/python'),str(repo/'tools/logit_compare.py'),str(root/'evidence-m3-current/logits-precise'),str(root/'evidence-m3-current/logits-fast'),'--min-top1','1','--max-mean-kl','0','--json',str(e/'zero-drift.json')]
with (e/'zero-drift.log').open('w') as out:p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
(e/'zero-drift-exit.json').write_text(json.dumps({'command':cmd,'returncode':p.returncode},indent=2)+'\n')
(e/'gate-status.json').write_text(json.dumps({'phase':'complete','engine':engine,'all_paths_pass':all(x['returncode']==0 for x in results),'zero_drift_returncode':p.returncode},indent=2)+'\n')
print(json.dumps(arithmetic,indent=2),flush=True)
