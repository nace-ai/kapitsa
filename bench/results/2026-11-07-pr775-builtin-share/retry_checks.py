import pathlib,subprocess,json,os,time
root=pathlib.Path('/Users/bytedance/Documents/ChatGPT/kapitsa');repo=root/'ane-pr775';out=root/'work/pr775/builtin-share/retry';out.mkdir(exist_ok=True)
env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_'))};venv=root/'work/pr775/m4-local/venv/bin';env['PATH']=str(venv)+os.pathsep+env['PATH']
checks=[('test_update',['python3','tests/test_update.py','build']),('test_pr_size',['python3','tests/test_pr_size.py']),('test_logit_compare',['python3','tests/test_logit_compare.py']),('test_sampled_tokens_harness',['python3','tests/test_sampled_tokens_harness.py']),('test_structured_cli',['python3','tests/test_structured.py','--cli','build']),('lint',[str(venv/'prek'),'run','--all-files']),('profile-gates',['python3','tests/test_profile_gates.py']),('comment-gates',['python3','tests/test_comment_gates.py']),('docs-links',['python3','tests/test_docs_links.py'])]
rows=[]
for name,cmd in checks:
 print(name,flush=True);(out/'status.json').write_text(json.dumps({'phase':name,'time':time.time()}))
 with (out/(name+'.log')).open('w') as f:p=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
 rows.append({'check':name,'command':cmd,'returncode':p.returncode});(out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
 if p.returncode:raise RuntimeError(name+' failed')
(out/'status.json').write_text(json.dumps({'phase':'complete','time':time.time()}));print('complete',flush=True)
