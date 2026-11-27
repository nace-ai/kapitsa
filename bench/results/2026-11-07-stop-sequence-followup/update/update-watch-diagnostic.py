import pathlib, sys, shutil, json, datetime
source=pathlib.Path(sys.argv[1]).resolve(); build=pathlib.Path(sys.argv[2]).resolve()
s=source.read_text(); s=s[:s.index("# a user's local.env in the harness survives an update")]
needle='return m.group(1) if m else ""'
assert s.count(needle)==1
s=s.replace(needle, 'if not m: print("VERSION_DIAGNOSTIC", repr({"path":path,"exit":r.returncode,"stdout":r.stdout,"stderr":r.stderr}), flush=True)\n    return m.group(1) if m else ""')
sys.argv=[str(source),str(build)]
ns={'__file__':str(source),'__name__':'__main__'}
try:
 exec(compile(s,str(source)+'[watch-only diagnostic]','exec'),ns)
 print('WATCH_RESULT',json.dumps({'fails':ns['FAILS'],'seen':sorted(ns['seen']),'bad':ns['bad']}),flush=True)
finally:
 if 'server' in ns: ns['server'].shutdown();ns['server'].server_close()
 if 'TMP' in ns: shutil.rmtree(ns['TMP'])
