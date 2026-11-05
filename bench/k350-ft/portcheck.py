#!/usr/bin/env python3
"""Port check: the torch drafter's drafting pass against Kapitsa's on the same contexts (k350_capt mode d output).
  portcheck.py <pc_dir> <pc.out> [drafter_dir]
Per pass: chain agreement (positions equal), top-16 overlap, and c(j) from the raw captures vs Kapitsa's c rows."""
import sys, os, re
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dfl

pc, outf = sys.argv[1], sys.argv[2]
ddir = sys.argv[3] if len(sys.argv) > 3 else dfl.M + '/dflash2'
dev = 'cpu'
W = dfl.st_read(ddir + '/model.safetensors')
emb, head = dfl.target_head()
d = dfl.Drafter({k: v for k, v in W.items() if not k.startswith('fc.') and k != 'hidden_norm.weight'}, dev)
V = head.shape[0]
special0 = 248044
allowed = torch.zeros(V, dtype=torch.bool)
allowed[special0:] = True
allowed[: 98304 - (V - special0)] = True
headf = head.float()
tot_eq = tot = ov_tot = 0
for line in open(outf):
    if not line.startswith('DRAFT'):
        continue
    f = line.split()
    idf, P = f[1], int(f[3])
    chain_k = [int(x) for x in f[5:12]]
    cand_k = [int(x) for x in f[13:13 + 112]]
    ids = [int(x) for x in open(idf).read().strip().split(',')]
    base = idf[:-4]
    c = torch.frombuffer(bytearray(open(base + '.c', 'rb').read()), dtype=torch.bfloat16).view(-1, dfl.H).float()
    S = c.shape[0]
    if os.path.exists(base + '.c.raw'):
        raw = torch.frombuffer(bytearray(open(base + '.c.raw', 'rb').read()), dtype=torch.bfloat16).view(S, -1).float()
        cc = dfl.rms(raw[-64:] @ W['fc.weight'].float().T, W['hidden_norm.weight'].float())
        rel = ((cc - c[-64:]).norm() / c[-64:].norm()).item()
    else:
        rel = float('nan')
    with torch.no_grad():
        cpos = torch.arange(P - S, P)
        kv = d.ctx_kv(c, cpos)
        x = emb[torch.tensor([[ids[P]] + [dfl.MASK] * 7])].float()
        hid = d.blocks(x, torch.tensor([P]), kv, cpos)[0, 1:]
        lg = hid @ headf.T
        lg[:, ~allowed] = -1e30
        top = lg.topk(16, -1)
        pred = ids[P]
        chain = []
        for k in range(7):
            e = d.sel_scores(hid[k], top.indices[k], torch.tensor(pred))
            j = int((top.values[k] + e).argmax())
            pred = int(top.indices[k, j]); chain.append(pred)
    ov = sum(len(set(top.indices[k].tolist()) & set(cand_k[k * 16:(k + 1) * 16])) for k in range(7))
    eq = sum(a == b for a, b in zip(chain, chain_k))
    tot_eq += eq; tot += 7; ov_tot += ov
    print(f'P {P}: chain torch {chain}\n        kapitsa {chain_k}  equal {eq}/7, top16 overlap {ov}/112, fc rel err {rel:.4f}')
print(f'PORTCHECK chain agreement {tot_eq}/{tot} = {tot_eq / max(tot, 1):.3f}, top16 overlap {ov_tot / max(tot, 1) / 16:.3f}')
