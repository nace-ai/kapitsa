#!/usr/bin/env python3
"""#350: fine-tune the 35B-A3B's DFlash2 drafter on self-distilled coding turns (MPS, holds the GPU lock).
Data: <data>/meta.jsonl, <data>/ids/<id>.ids, <feat>/<id>.c (k350_capt c rows, positions 0..n-1).
Objective (DFlash's block objective): anchors P in the answer, block [t_P, mask x 7] at positions P..P+7 sees the
context rows of positions < P (last 2047); position-weighted CE of t_{P+1..P+7} (w_k = exp(-(k-1)/4)) over the full
head, plus the selector's CE over each row's top 16 with the true predecessor. fc and hidden_norm are frozen (the c rows
are their output); the 6 layers, final norm and the selector train. 5% of conversations held out: loss and the
greedy-walk acceptance proxy (accepted prefix + 1, the tokens a full-depth chain cycle would emit).
  train.py <data> <feat> <out_dir> <total_minutes> <chunk_minutes> [lr]
Resumes from <out_dir>/state.pt (weights, optimizer, step, elapsed); a run stops after chunk_minutes (or when the
total is reached) and saves state.pt, so training runs as several short GPU-lock jobs. The LM head and its loss run
in 64-row chunks (no [B x 7 x V] logits in one buffer)."""
import json, math, os, random, re, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfl

data, feat, out, minutes, chunk = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
lr = float(sys.argv[6]) if len(sys.argv) > 6 else 3e-5
BA = int(os.environ.get('K350_ANCHORS', 48))
dev = os.environ.get('K350_DEV', 'mps')
os.makedirs(out, exist_ok=True)
if os.path.exists(out + '/DONE'):
    print('training already done'); sys.exit(0)
log = open(out + '/train.log', 'a')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()

meta = [json.loads(l) for l in open(data + '/meta.jsonl')]
meta = [m for m in meta if os.path.exists(f"{feat}/{m['id']}.c") and os.path.getsize(f"{feat}/{m['id']}.c") == m['n'] * dfl.H * 2]
rng = random.Random(350)
rng.shuffle(meta)
nv = max(8, len(meta) // 20)
val, tr = meta[:nv], meta[nv:]
say(f'data: {len(tr)} train / {len(val)} val conversations, answer tokens {sum(m["n"] - m["n_prompt"] for m in tr)} / {sum(m["n"] - m["n_prompt"] for m in val)}; device {dev}, lr {lr}, anchors/step {BA}')

W0 = dfl.st_read(dfl.M + '/dflash2/model.safetensors')
frozen = {k: W0[k] for k in ('fc.weight', 'hidden_norm.weight')}
d = dfl.Drafter({k: v for k, v in W0.items() if k not in frozen}, dev)
emb, head = dfl.target_head()
emb = emb.to(dev).float(); head = head.to(dev).float()
wpos = torch.tensor([math.exp(-k / 4.0) for k in range(7)], device=dev)
wpos = wpos / wpos.sum() * 7

def load(m):
    ids = torch.tensor([int(x) for x in open(f"{data}/ids/{m['id']}.ids").read().split(',')], dtype=torch.long)
    c = torch.frombuffer(bytearray(open(f"{feat}/{m['id']}.c", 'rb').read()), dtype=torch.bfloat16).view(-1, dfl.H)
    return ids, c

DBG = os.environ.get('K350_DBG')
def dbg(t):
    if DBG:
        torch.mps.synchronize(); print('dbg', t, flush=True)

def foot():  # this process's phys_footprint in GB (what mem-guard reads)
    out = os.popen(f'footprint -p {os.getpid()} 2>/dev/null').read()
    import re
    m_ = re.search(r'phys_footprint: ([\d.]+) (GB|MB)', out)
    return float(m_[1]) / (1024 if m_ and m_[2] == 'MB' else 1) if m_ else -1.0

PAD = int(os.environ.get('K350_PAD', 2048))  # 7 Nov: at 512 (4 shapes) allocated stayed flat at 11.4 GiB but driver grew 20.6 -> 23.1 GiB over 100 steps; every conversation is < 2048 rows, so one shape
def batch_loss(m, anchors=None, nmax=BA, stats=False, pad=None):
    ids, c = load(m)
    n, n0 = m['n'], m['n_prompt']
    pad = PAD if pad is None else pad
    if pad:  # MPS compiles and keeps a graph per new shape; every conversation's own length grew the footprint ~1 GB per 25 steps.
        # Zero rows at positions >= n: blocks() masks context positions >= P, and every anchor P < n, so they get weight exp(-inf) = 0.
        npad = -(-n // pad) * pad
        c = torch.cat([c, c.new_zeros(npad - n, c.shape[1])])
    cand = list(range(n0, n - 1))
    if anchors is None:
        anchors = rng.sample(cand, min(nmax, len(cand)))
    P = torch.tensor(anchors, dtype=torch.long)
    idsd = ids.to(dev); c = c.to(dev).float()
    cpos = torch.arange(c.shape[0], device=dev)
    kv = d.ctx_kv(c, cpos); dbg('kv')
    x = emb[torch.cat([idsd[P.to(dev)][:, None], torch.full((len(anchors), 7), dfl.MASK, device=dev)], 1)]
    dbg('x')
    hid = d.blocks(x, P.to(dev), kv, cpos)[:, 1:]  # [B][7][H]
    dbg('blocks')
    tpos = P[:, None] + torch.arange(1, 8)[None]
    valid = (tpos < n).to(dev)
    tgt = idsd[tpos.clamp(max=n - 1).to(dev)]
    hf = hid.contiguous().reshape(-1, dfl.H)
    tf = tgt.reshape(-1)
    ces, tvs, tis = [], [], []
    for r0 in range(0, hf.shape[0], 64):  # the LM head in 64-row chunks: [64][V] logits at a time
        lgc = hf[r0:r0 + 64] @ head.T
        ces.append(F.cross_entropy(lgc, tf[r0:r0 + 64], reduction='none'))
        tc = lgc.detach().topk(16, -1); tvs.append(tc.values); tis.append(tc.indices)
    dbg('lg')
    ce = torch.cat(ces).view(tgt.shape)
    class _T: pass
    top = _T(); top.values = torch.cat(tvs).view(*tgt.shape, 16); top.indices = torch.cat(tis).view(*tgt.shape, 16)
    dbg('ce')
    lce = (ce * wpos * valid).sum() / valid.sum().clamp(min=1)
    pred = torch.cat([idsd[P.to(dev)][:, None], tgt[:, :-1]], 1)
    sc = top.values + d.sel_scores(hid, top.indices, pred)
    hit = (top.indices == tgt[..., None])
    has = hit.any(-1) & valid
    lsel = (F.cross_entropy(sc.reshape(-1, 16), hit.float().argmax(-1).reshape(-1), reduction='none').view(tgt.shape) * has).sum() / has.sum().clamp(min=1)
    dbg('sel')
    res = (lce, lsel)
    if stats:  # greedy walk (no teacher): accepted prefix length per anchor
        with torch.no_grad():
            p = idsd[P.to(dev)]
            acc = torch.ones(len(anchors), dtype=torch.bool, device=dev)
            alen = torch.zeros(len(anchors), device=dev)
            for k in range(7):
                e = d.sel_scores(hid[:, k], top.indices[:, k], p)
                j = (top.values[:, k] + e).argmax(-1)
                p = top.indices[:, k].gather(1, j[:, None])[:, 0]
                acc = acc & (p == tgt[:, k]) & valid[:, k]
                alen += acc.float()
        res = res + (alen,)
    return res

def evaluate():
    d.eval()
    L = S = A = N = 0.0
    with torch.no_grad():
        for m in val:
            vr = random.Random(m['id'])
            cand = list(range(m['n_prompt'], m['n'] - 1))
            an = vr.sample(cand, min(32, len(cand)))
            lce, lsel, alen = batch_loss(m, an, stats=True)
            L += lce.item() * len(an); S += lsel.item() * len(an); A += alen.sum().item(); N += len(an)
    d.train()
    if dev == 'mps':
        torch.mps.empty_cache()
    return L / N, S / N, 1 + A / N

if os.environ.get('K350_CHECK_PAD'):  # CPU check: padded context rows change nothing
    worst = 0.0
    for m in val[:4]:
        vr = random.Random(m['id']); an = vr.sample(list(range(m['n_prompt'], m['n'] - 1)), min(8, m['n'] - 1 - m['n_prompt']))
        with torch.no_grad():
            a = batch_loss(m, an, stats=True, pad=0); b = batch_loss(m, an, stats=True, pad=512)
        df = max(abs(a[0].item() - b[0].item()), abs(a[1].item() - b[1].item()), (a[2] - b[2]).abs().max().item())
        worst = max(worst, df); say(f'pad check {m["id"]} n {m["n"]}: ce {a[0].item():.6f} vs {b[0].item():.6f}, sel {a[1].item():.6f} vs {b[1].item():.6f}, max diff {df:.2e}')
    say(f'pad check worst diff {worst:.2e}'); sys.exit(0 if worst < 1e-4 else 1)
trainable = [p for p in d.parameters()]
opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
t0 = time.time()
budget = minutes * 60
st = out + '/state.pt'
if os.path.exists(st):
    S0 = torch.load(st)
    with torch.no_grad():
        for k, v in S0['params'].items():
            d.w(k).copy_(v.to(dev))
    opt.load_state_dict(S0['opt'])
    step, done_s, best, v0 = S0['step'], S0['elapsed'], S0['best'], S0['v0']
    rng.setstate(S0['rng'])
    say(f'resumed at step {step}, {done_s:.0f} s done')
else:
    v0 = evaluate()
    say(f'step 0 val ce {v0[0]:.4f} sel {v0[1]:.4f} walk tok/cycle proxy {v0[2]:.4f} ({time.time() - t0:.0f} s)')
    best, step, done_s = v0[2], 0, 0.0
ema = None
STOP = int(os.environ.get('K350_STOP_STEP', 0))
tstart = time.time() - done_s
tchunk = time.time()
SWAP0 = float(re.search(r'used = ([\d.]+)M', os.popen('sysctl -n vm.swapusage').read()).group(1))  # swap at chunk start
order = []
def save_state():
    torch.save({'params': {k: d.w(k).detach().cpu() for k in W0 if k not in frozen}, 'opt': opt.state_dict(), 'step': step,
                'elapsed': time.time() - tstart, 'best': best, 'v0': v0, 'rng': rng.getstate()}, st + '.tmp')
    os.replace(st + '.tmp', st)
def save(tag):
    sd = {k: v.detach().to('cpu', torch.bfloat16) for k, v in ((k, d.w(k)) for k in W0 if k not in frozen)}
    sd.update(frozen)
    sd = {k: sd[k] for k in W0}
    os.makedirs(f'{out}/{tag}', exist_ok=True)
    dfl.st_write(f'{out}/{tag}/model.safetensors', sd)
    os.system(f"cp '{dfl.M}/dflash2/config.json' '{out}/{tag}/config.json'")
while True:
    el = time.time() - tstart
    if el > budget:
        break
    if time.time() - tchunk > chunk * 60:
        save_state(); say(f'chunk end at step {step}, {el:.0f} s of {budget:.0f}; state saved'); sys.exit(0)
    frac = el / budget
    cur = lr * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    for g in opt.param_groups:
        g['lr'] = cur
    if step % 25 == 0 and dev == 'mps':
        say(f'mem step {step}: allocated {torch.mps.current_allocated_memory() / 2**30:.2f} GiB, driver {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB, footprint {foot():.2f} GB')
    if STOP and step >= STOP:
        say(f'K350_STOP_STEP {STOP} reached: test run ends without saving'); sys.exit(0)
    if not order:
        order = tr[:]; rng.shuffle(order)
    m = order.pop()
    lce, lsel = batch_loss(m)
    loss = lce + 0.5 * lsel
    lv, lcev, lselv = loss.item(), lce.item(), lsel.item()
    ema = lv if ema is None else 0.98 * ema + 0.02 * lv
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    del loss, lce, lsel
    if dev == 'mps':
        torch.mps.empty_cache()  # varying sequence lengths fragment MPS's cache: 44 GB footprint by step 600 without this
    step += 1
    if step == 1 or (step < 50 and step % 10 == 0):
        torch.mps.synchronize() if dev == 'mps' else None
        mem = torch.mps.driver_allocated_memory() / 2**30 if dev == 'mps' else 0
        say(f'step {step} done: anchors {BA}, seq len {m["n"]}, driver memory {mem:.2f} GiB, footprint {foot():.2f} GB, {time.time() - tchunk:.1f} s into chunk')
    if step % 25 == 0:  # orchestrator 7 Nov: abort on swap over 8 GB; exit 75 tells pipeline.sh to re-gate and resume
        su = float(re.search(r'used = ([\d.]+)M', os.popen('sysctl -n vm.swapusage').read()).group(1))
        if su > 8192 and su > SWAP0 + 2048:  # swap stays high for hours after others' pressure ends: stop only if we added 2 GB
            save_state(); say(f'swap used {su / 1024:.1f} GB over 8 GB at step {step}: state saved, chunk stops'); sys.exit(75)
    if step % 25 == 0 and dev == 'mps' and foot() > float(os.environ.get('K350_MAX_GB', 30)):
        save_state(); say(f'footprint {foot():.1f} GB over K350_MAX_GB at step {step}: state saved, chunk ends early'); sys.exit(0)
    if step % 25 == 0:
        say(f'step {step} footprint {foot():.1f} GB loss {lv:.4f} ema {ema:.4f} ce {lcev:.4f} sel {lselv:.4f} lr {cur:.2e} {time.time() - tstart:.0f} s')
    if step % 300 == 0:
        v = evaluate()
        say(f'step {step} val ce {v[0]:.4f} sel {v[1]:.4f} walk tok/cycle proxy {v[2]:.4f}')
        if v[2] > best:
            best = v[2]; save('best'); say(f'saved best at step {step}')
        save_state()
v = evaluate()
say(f'final step {step} val ce {v[0]:.4f} sel {v[1]:.4f} walk tok/cycle proxy {v[2]:.4f} (shipped {v0[2]:.4f}) in {time.time() - tstart:.0f} s')
save('last')
if v[2] >= best:
    best = v[2]; save('best')
say(f'DONE best proxy {best:.4f}')
open(out + '/DONE', 'w').write(f'{best}\n')
