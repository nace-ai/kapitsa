import pathlib,sys,json,statistics
root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root/'candidate-current-pr775/tools'))
import logit_compare as compare
r=compare.Dump(str(root/'evidence-m3-current/logits-precise/prefill'))
f=compare.Dump(str(root/'evidence-m3-current/logits-fast/prefill'))
m=compare.Dump(str(root/'evidence-m3-current-competitors/logits-mlx/prefill'))
s=compare.Dump(str(root/'evidence-m3-current/logits-precise/step'))
assert compare.match(r,f)==compare.match(r,m)==[(i,i) for i in range(8)]
rows=[]
for i,(sid,length) in enumerate(r.keys):
 assert min(n for k,n in s.keys if k==sid)-1==length
 assert r.token_key(i)==f.token_key(i)==m.token_key(i)
 fm=compare.row_metrics(r.row(i),f.row(i),r.nexts[i]);mm=compare.row_metrics(r.row(i),m.row(i),r.nexts[i])
 rows.append({'sequence':sid,'prefix_tokens':length,'fast_kl':fm['kl'],'mlx_kl':mm['kl']})
print(json.dumps({'same_full_vocabulary':r.vocab==f.vocab==m.vocab==248320,'rows':rows,'prefix_is_one_token_before_first_step_record':True,'fast_mean':statistics.mean(x['fast_kl'] for x in rows),'mlx_mean':statistics.mean(x['mlx_kl'] for x in rows)},indent=2))
