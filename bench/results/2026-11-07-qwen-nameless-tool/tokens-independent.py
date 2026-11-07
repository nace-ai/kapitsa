import json,subprocess,pathlib,hashlib,time,os
P=pathlib.Path('/Users/bytedance/kapitsa-nameless-tool-20261107'); C=P/'candidate'; E=P/'evidence';E.mkdir(exist_ok=True)
(P/'tokens.pid').write_text(str(os.getpid()))
assert (C/'build/parser_test').is_file() and (C/'build/libnaceinfer.a').is_file()
# Full target builds have completed; tokenizer-only checks can run alongside unrelated CLI CPU tests.
models=['/Users/bytedance/pr934-m3-validation-20261105/modelhome/models/qwen3.5-4b-4bit','/Users/bytedance/models/Qwen3.6-35B-A3B-4bit']
commands=[]
def run(label,args):
 with (E/(label+'.log')).open('w') as f:r=subprocess.run(args,cwd=C,stdout=f,stderr=subprocess.STDOUT)
 commands.append({'label':label,'argv':args,'cwd':str(C),'rc':r.returncode});(E/'token-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 return r.returncode
manifest=[]
for m in models:
 for name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']:
  f=pathlib.Path(m)/name
  if f.exists():manifest.append({'file':str(f),'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(E/'tokenizer-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
errors=[]
# The final regression adds a type guard so the known-bad baseline reports failures instead of indexing a missing text key.
args=['/usr/bin/c++','-O3','-std=gnu++20','-arch','arm64','-mmacosx-version-min=14.0','-I'+str(C/'src'),'-I'+str(C/'include'),'-I'+str(C/'third_party'),str(P/'api-regression.cpp'),str(C/'build/libnaceinfer.a'),'-o',str(P/'api-candidate')]
for fw in ['Metal','Foundation','CoreFoundation','IOSurface','IOKit','ImageIO','CoreGraphics','Security','SystemConfiguration','Accelerate']:args+=['-framework',fw]
errors.append(run('api-candidate-compile',args))
(E/'final-api-source-sha256.txt').write_text(hashlib.sha256((P/'api-regression.cpp').read_bytes()).hexdigest()+'\n')
for tag,m in zip(['4b','35b'],models):
 for label,args in [('parser',[str(C/'build/parser_test'),m]),('api',[str(P/'api-candidate'),m]),('structured',['/Users/bytedance/kapitsa-968-evidence-20261106/venv/bin/python3','tests/test_structured.py','--cli','build',m])]:
  errors.append(run(tag+'-'+label,args))
# Compile the new API regression against the unmodified base headers and base library.
B=pathlib.Path('/Users/bytedance/kapitsa-affine-rebase-20261107/base')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=B,text=True).strip()=='4d1d744538063a17bc85e88a08c08c8061511c95'
args=['/usr/bin/c++','-O3','-std=gnu++20','-arch','arm64','-mmacosx-version-min=14.0','-I'+str(B/'src'),'-I'+str(B/'include'),'-I'+str(B/'third_party'),str(P/'api-regression.cpp'),str(B/'build/libnaceinfer.a'),'-o',str(P/'api-base')]
for fw in ['Metal','Foundation','CoreFoundation','IOSurface','IOKit','ImageIO','CoreGraphics','Security','SystemConfiguration','Accelerate']:args+=['-framework',fw]
errors.append(run('api-base-compile',args))
if errors[-1]==0:
 run('api-base-4b',[str(P/'api-base'),models[0]])
(E/'tokens-result.json').write_text(json.dumps({'candidate_exit_codes':errors,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip()},indent=2)+'\n')
(P/'tokens.exit').write_text(str(int(any(errors)))+'\n')
