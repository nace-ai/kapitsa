import re,glob,statistics as st,sys,json
rows={}
for f in sorted(glob.glob('ab/*.err')):
    m=re.match(r'ab/(\w+)-(old|new)-(\d)\.err',f)
    t=open(f).read()
    p=re.search(r'prompt (\d+) tok in (\d+) ms \(([\d.]+) tok/s\); spec decode D=2: (\d+) tok in (\d+) ms \(([\d.]+) tok/s\), (\d+) cycles.*?([\d.]+) ms wall/cycle',t)
    if not p: continue
    ctx,arm,r=m.groups()
    rows.setdefault(ctx,{}).setdefault(arm,[]).append((int(r),float(p.group(3)),float(p.group(6)),float(p.group(8))))
out={}
for ctx in ['4k','32k','128k']:
    if ctx not in rows: continue
    d=rows[ctx]
    if 'old' not in d or 'new' not in d: continue
    n=min(len(d['old']),len(d['new']))
    res={}
    for i,name in ((1,'prefill'),(2,'decode')):
        o=[x[i] for x in sorted(d['old'])][:n]; w=[x[i] for x in sorted(d['new'])][:n]
        ratios=[b/a for a,b in zip(o,w)]
        res[name]=dict(old=o,new=w,old_med=st.median(o),new_med=st.median(w),ratio_pairs=[round(x,3) for x in ratios],pct_med=round((st.median(w)/st.median(o)-1)*100,1),pct_min=round((min(ratios)-1)*100,1),pct_max=round((max(ratios)-1)*100,1))
    wc_o=[x[3] for x in sorted(d['old'])][:n]; wc_n=[x[3] for x in sorted(d['new'])][:n]
    res['ms_per_cycle']=dict(old=wc_o,new=wc_n)
    out[ctx]=res
    print(ctx,'n=%d'%n,'decode old/new med %.1f/%.1f (%+.1f%%, pairs %+.1f..%+.1f%%)'%(res['decode']['old_med'],res['decode']['new_med'],res['decode']['pct_med'],res['decode']['pct_min'],res['decode']['pct_max']),'| prefill %.0f/%.0f (%+.1f%%, %+.1f..%+.1f%%)'%(res['prefill']['old_med'],res['prefill']['new_med'],res['prefill']['pct_med'],res['prefill']['pct_min'],res['prefill']['pct_max']))
json.dump(out,open('ab/summary.json','w'),indent=1)
