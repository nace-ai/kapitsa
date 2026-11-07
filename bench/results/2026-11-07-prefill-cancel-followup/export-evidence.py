import hashlib, json, pathlib, sys, tarfile
root=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); assert not out.exists()
for marker in ["driver.rc","cancel-ab.rc","state-cases.rc","state-cases-v2.rc","state-cases-v3.rc","state-cases-v4.rc"]: assert (root/marker).exists(), marker
extensions={".json",".txt",".log",".rc",".cmd",".out",".err",".ids",".sh",".py",".cpp"}
files=[]
for p in sorted(root.rglob("*")):
 if not p.is_file() or p.is_symlink(): continue
 rel=p.relative_to(root)
 if any(part.startswith("home-") or part in {"__pycache__",".git","build"} for part in rel.parts): continue
 if p.suffix not in extensions and p.name not in {"stage","build-ready"}: continue
 files.append((p,rel))
manifest="".join(hashlib.sha256(p.read_bytes()).hexdigest()+"  "+str(rel)+"\n" for p,rel in files)
(root/"EXPORT-SHA256SUMS").write_text(manifest)
with tarfile.open(out,"w:gz") as t:
 for p,rel in files:t.add(p,arcname=str(rel),recursive=False)
 t.add(root/"EXPORT-SHA256SUMS",arcname="EXPORT-SHA256SUMS")
print(json.dumps({"files":len(files),"archive":str(out),"bytes":out.stat().st_size,"sha256":hashlib.sha256(out.read_bytes()).hexdigest()}))
