"""DFlash2 drafter in torch (the 35B-A3B's dflash2/), after mlx_vlm's DFlash2DraftModel and Kapitsa's src/model/dflash2.cpp:
context rows c(j) = RMSNorm(fc(x_ctx); hidden_norm) (taken as given: k350_capt dumps them), 6 layers whose block rows
attend to the last 2047 context positions and to all 8 block rows (non-causal), grouped dynamic causal convolutions
around attention and MLP (kernel 2 over the block rows), final norm, the target's LM head, and the candidate selector's
greedy walk over each row's top 16. Training forward: many anchors of one sequence at once."""
import json, math, os, struct
import numpy as np
import torch
import torch.nn.functional as F

M = os.path.expanduser('~/Documents/nace/models/qwen3.6-35b-a3b-4bit')
H, NQ, NKV, HD, I, NL, B8, KEEP, G = 2048, 32, 8, 128, 6144, 6, 8, 2047, 16
MASK, THETA, EPS = 248077, 1e7, 1e-6


def st_read(path, names=None):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == '__metadata__' or (names is not None and k not in names):
                continue
            a, b = v['data_offsets']
            f.seek(base + a)
            buf = f.read(b - a)
            dt = {'BF16': np.uint16, 'U32': np.uint32, 'F32': np.float32, 'F16': np.float16}[v['dtype']]
            arr = np.frombuffer(buf, dtype=dt).reshape(v['shape'])
            t = torch.from_numpy(arr.copy())
            if v['dtype'] == 'BF16':
                t = t.view(torch.bfloat16)
            out[k] = t
        return out


def st_write(path, tensors):
    hdr, off, bufs = {}, 0, []
    for k, t in tensors.items():
        t = t.detach().to('cpu').contiguous()
        assert t.dtype == torch.bfloat16
        b = t.view(torch.uint16).numpy().tobytes()
        hdr[k] = {'dtype': 'BF16', 'shape': list(t.shape), 'data_offsets': [off, off + len(b)]}
        off += len(b); bufs.append(b)
    h = json.dumps(hdr).encode()
    h += b' ' * ((8 - len(h) % 8) % 8)
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(h))); f.write(h)
        for b in bufs:
            f.write(b)


def dequant4(w, s, b):  # MLX affine 4-bit group 64: w uint32 [N][K/8], s/b bf16 [N][K/64]
    N = w.shape[0]
    w = w.to(torch.int64)
    q = torch.stack([(w >> (4 * i)) & 15 for i in range(8)], dim=-1).reshape(N, -1).float()
    K = q.shape[1]
    q = q.view(N, K // 64, 64)
    return (q * s.float()[..., None] + b.float()[..., None]).view(N, K).to(torch.bfloat16)


def target_head(cache='/tmp/k350d/head.pt'):
    """(embed [V][H], lm_head [V][H]) bf16, dequantized from the target once and cached."""
    if os.path.exists(cache):
        return torch.load(cache)
    idx = json.load(open(M + '/model.safetensors.index.json'))['weight_map']
    out = []
    for p in ('language_model.model.embed_tokens', 'language_model.lm_head'):
        names = [p + '.weight', p + '.scales', p + '.biases']
        t = st_read(M + '/' + idx[names[0]], set(names))
        out.append(dequant4(t[names[0]], t[names[1]], t[names[2]]))
    torch.save(tuple(out), cache)
    return tuple(out)


def rms(x, w):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w


def rope_cs(pos, device):  # pos: long tensor of positions -> cos, sin [.., HD]
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, device=device, dtype=torch.float32) / HD))
    f = pos.float()[..., None] * inv
    emb = torch.cat([f, f], -1)
    return emb.cos(), emb.sin()


def rope(x, cos, sin):  # neox split-half; x [..., HD]
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    return (x.float() * cos + torch.cat([-x2, x1], -1).float() * sin).to(x.dtype)


def conv(x, dyn, base):
    """out[t] = sum_o (base[o] + dyn[t, o]) * x[t - o] within the block; x [..., L, H], dyn [..., L, 2, G], base [2, H]."""
    sh = x.shape
    L = sh[-2]
    xb = x.reshape(*sh[:-1], H // G, G)
    d = dyn.reshape(*sh[:-1], 2, H // G, 1)
    out = (base[0].reshape(H // G, G) + d[..., 0, :, :]) * xb
    prev = torch.cat([torch.zeros_like(xb[..., :1, :, :]), xb[..., : L - 1, :, :]], dim=-3)
    out = out + (base[1].reshape(H // G, G) + d[..., 1, :, :]) * prev
    return out.reshape(sh)


class Drafter(torch.nn.Module):
    def __init__(self, weights, device, dtype=torch.float32):
        super().__init__()
        self.p = torch.nn.ParameterDict({k.replace('.', '__'): torch.nn.Parameter(v.to(device=device, dtype=dtype))
                                         for k, v in weights.items()})
        self.device = device

    def w(self, k):
        return self.p[k.replace('.', '__')]

    def ctx_kv(self, c, pos):
        """per layer (K [NKV][S][HD] roped, V [NKV][S][HD]) of context rows c [S][H] at positions pos [S]."""
        cos, sin = rope_cs(pos, c.device)
        out = []
        for l in range(NL):
            a = f'layers.{l}.self_attn.'
            k = (c @ self.w(a + 'k_proj.weight').T).view(-1, NKV, HD)
            k = rope(rms(k, self.w(a + 'k_norm.weight')), cos[:, None], sin[:, None]).transpose(0, 1)
            v = (c @ self.w(a + 'v_proj.weight').T).view(-1, NKV, HD).transpose(0, 1)
            out.append((k, v))
        return out

    def _cprep(self, l, name, x):
        dyn = (x @ self.w(f'layers.{l}.{name}.kernel_projection.weight').T).view(*x.shape[:-1], 2, 2, H // G)
        base = self.w(f'layers.{l}.{name}.base_kernel')  # [2 (prepare/finish)][2 (offset)][H]
        return conv(x, dyn[..., 0, :, :], base[0]), (dyn[..., 1, :, :], base[1])

    def blocks(self, x, P, kv, cpos):
        """x [B][8][H] block embeddings at positions P[b] + 0..7; kv: ctx_kv of rows at positions cpos [S].
        Block row of anchor P sees context positions max(0, P - 2047) <= j < P. Returns final-normed hidden [B][8][H]."""
        Bn = x.shape[0]
        qpos = P[:, None] + torch.arange(B8, device=x.device)[None]
        cos, sin = rope_cs(qpos, x.device)  # [B][8][HD]
        allow = (cpos[None, :] < P[:, None]) & (cpos[None, :] >= P[:, None] - KEEP)  # [B][S]
        bias = torch.zeros(allow.shape, device=x.device, dtype=torch.float32).masked_fill(~allow, float('-inf'))
        for l in range(NL):
            a = f'layers.{l}.self_attn.'
            r = x
            h, fin = self._cprep(l, 'attention_conv', rms(x, self.w(f'layers.{l}.input_layernorm.weight')))
            q = (h @ self.w(a + 'q_proj.weight').T).view(Bn, B8, NQ, HD)
            q = rope(rms(q, self.w(a + 'q_norm.weight')), cos[:, :, None], sin[:, :, None]).permute(0, 2, 1, 3)
            kb = (h @ self.w(a + 'k_proj.weight').T).view(Bn, B8, NKV, HD)
            kb = rope(rms(kb, self.w(a + 'k_norm.weight')), cos[:, :, None], sin[:, :, None]).permute(0, 2, 1, 3)
            vb = (h @ self.w(a + 'v_proj.weight').T).view(Bn, B8, NKV, HD).permute(0, 2, 1, 3)
            kc, vc = kv[l]  # [NKV][S][HD]
            qg = q.reshape(Bn, NKV, NQ // NKV * B8, HD)  # heads grouped by their kv head
            sc = torch.einsum('bgqd,gsd->bgqs', qg.float(), kc.float()) / math.sqrt(HD) + bias[:, None, None, :]
            sb = torch.einsum('bgqd,bgkd->bgqk', qg.float(), kb.float()) / math.sqrt(HD)
            pr = torch.softmax(torch.cat([sc, sb], -1), -1)
            S = kc.shape[1]
            o = torch.einsum('bgqs,gsd->bgqd', pr[..., :S], vc.float()) + torch.einsum('bgqk,bgkd->bgqd', pr[..., S:], vb.float())
            o = o.to(x.dtype).reshape(Bn, NQ, B8, HD).permute(0, 2, 1, 3).reshape(Bn, B8, NQ * HD)
            o = o @ self.w(a + 'o_proj.weight').T
            x = r + conv(o, *fin)
            r = x
            h, fin = self._cprep(l, 'mlp_conv', rms(x, self.w(f'layers.{l}.post_attention_layernorm.weight')))
            m = f'layers.{l}.mlp.'
            h = (F.silu(h @ self.w(m + 'gate_proj.weight').T) * (h @ self.w(m + 'up_proj.weight').T)) @ self.w(m + 'down_proj.weight').T
            x = r + conv(h, *fin)
        return rms(x, self.w('norm.weight'))

    def sel_scores(self, hid, cand, pred):
        """edges for rows: hid [..][H] (final hidden), cand [..][16] ids, pred [..] predecessor ids -> [..][16]."""
        hp = hid @ self.w('candidate_selector.hidden_projection.weight').T  # [..][256]
        pc = F.embedding(pred, self.w('candidate_selector.predecessor_codebook'))
        sc = F.embedding(cand, self.w('candidate_selector.successor_codebook'))
        return ((pc * hp)[..., None, :] * sc).sum(-1)
