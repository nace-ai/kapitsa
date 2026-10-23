# Fastest rival by T=0.6 decode median among this run's valid Phase B records: pick.py LABEL [LABEL ...]
import glob, json, os, sys
best = None
for lab in sys.argv[1:]:
    for f in glob.glob(os.path.expanduser(f"~/kq/m3/out/results/{lab}-2026*.json")):
        d = json.load(open(f))
        med = (d.get("decode_summary") or {}).get("median_tps")
        if d.get("invalid") or not med:
            continue
        if best is None or med > best[0]:
            best = (med, lab)
print(best[1] if best else "")
