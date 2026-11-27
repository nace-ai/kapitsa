import subprocess,pathlib,json,datetime
root=pathlib.Path('/Users/bytedance/kapitsa-stop-sequence-20261107'); out=root/'evidence/watch-baseline-control-0815';out.mkdir(exist_ok=False)
py='/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin/python';results=[]
for arm in ['base']:
 for i in range(24):
  p=out/f'{arm}-{i}.log'; started=datetime.datetime.now(datetime.timezone.utc).isoformat()
  with p.open('x') as log:r=subprocess.run([py,'-u',str(root/'update-watch-diagnostic.py'),str(root/arm/'tests/test_update.py'),str(root/arm/'build')],stdout=log,stderr=subprocess.STDOUT,cwd=root/arm,timeout=120)
  text=p.read_text();results.append({'arm':arm,'iteration':i,'exit':r.returncode,'started':started,'diagnostic':[x for x in text.splitlines() if x.startswith(('VERSION_DIAGNOSTIC','WATCH_RESULT','FAIL '))]})
  (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  if 'VERSION_DIAGNOSTIC' in text or r.returncode:break
