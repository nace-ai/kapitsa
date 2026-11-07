import pathlib,os,tempfile,subprocess,json,argparse
p=argparse.ArgumentParser();p.add_argument('--binary',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
with tempfile.TemporaryDirectory(prefix='pr775-apfs-rm-') as home:
 root=pathlib.Path(home);models=root/'models';x=models/'copy-a';y=models/'copy-b';x.mkdir(parents=True)
 (x/'config.json').write_text('{}');(x/'model.safetensors').write_bytes(os.urandom(20<<20));subprocess.run(['cp','-c','-R',str(x),str(y)],check=True)
 env={k:v for k,v in os.environ.items() if not k.startswith(('NI_','NACE_','BENCH_','KAPITSA_'))};env.update(KAPITSA_HOME=home,HF_ENDPOINT='http://127.0.0.1:9',NI_LOG='0',KAPITSA_NO_AUTO_UPDATE='1',KAPITSA_NO_PROMPT='1')
 rows=[]
 for n,needle in [('copy-b','nothing freed'),('copy-a','MB freed')]:
  cmd=[str(a.binary.resolve()),'rm',n];r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
  rows.append({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'expected_text':needle,'passed':r.returncode==0 and needle in r.stdout and not (models/n).exists()})
 (a.out/'results.json').write_text(json.dumps({'binary':str(a.binary.resolve()),'rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
