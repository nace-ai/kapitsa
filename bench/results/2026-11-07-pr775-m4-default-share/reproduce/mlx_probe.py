import pathlib,sys,os,fcntl,json,time
if not os.getenv('NI_LOCK_HELD'):
    lock=open('/tmp/kapitsa-gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
model_dir,mode,source,dest=sys.argv[1:]
model,tok=load(model_dir)
print('native model loaded; no dtype cast',flush=True)
def feed(ids,cache):
    for start in range(0,len(ids),2048):
        logits=model(mx.array(ids[start:start+2048])[None],cache=cache)[0,-1]
        mx.eval(logits,[c.state for c in cache])
    return logits
if mode=='dump':
    source=pathlib.Path(source);dest=pathlib.Path(dest);dest.mkdir(exist_ok=True)
    seqs=[[int(t) for t in line.split()] for line in (source/'sequences.txt').read_text().splitlines()]
    records=[tuple(map(int,line.split())) for line in (source/'records.txt').read_text().splitlines()]
    with (dest/'step.f32').open('wb') as f,(dest/'prefill.f32').open('wb') as pf:
        for sid,seq in enumerate(seqs):
            targets=[l for i,l in records if i==sid];cache=make_prompt_cache(model);first=min(targets)-1
            lg=feed(seq[:first],cache);np.asarray(lg.astype(mx.float32)).tofile(pf);pos=first
            for target in targets:
                lg=feed(seq[pos:target],cache);pos=target;a=np.asarray(lg.astype(mx.float32));assert a.size==248320 and np.isfinite(a).all();a.tofile(f)
            print('window',sid,'done',flush=True);del cache;mx.clear_cache()
elif mode=='speed':
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    prompt=list(map(int,pathlib.Path(source).read_text().split()))
    with open(dest,'w') as f:
        for rep in range(-1,3):
            start=time.perf_counter();ids=[]
            gen=generate_step(mx.array(prompt[:64] if rep<0 else prompt),model,max_tokens=8 if rep<0 else 256,sampler=make_sampler(0.0))
            for i,(t,_) in enumerate(gen):
                t=int(t);ids.append(t);last=time.perf_counter()
                if i==0:first=last
                if t in tok.eos_token_ids:break
            if rep>=0:
                row={'rep':rep,'prefill_s':first-start,'decode_s':last-first,'tokens':len(ids),'tok_s':(len(ids)-1)/(last-first),'ids':ids,'text':tok.decode(ids),'peak_bytes':mx.get_peak_memory()}
                f.write(json.dumps(row)+'\n');f.flush();assert len(ids)==256 and ids[-1] not in tok.eos_token_ids and row['text'].strip(),'short/EOS/empty sample'
            del gen;mx.clear_cache()
else:raise ValueError(mode)
