#!/usr/bin/env python3
"""Generate tests/fixtures/ink2_golden.json from a pinned thinkingmachines/Inkling-2-Mini checkpoint.

Usage:
  python3 tools/make_ink2_golden.py --model-dir /path/to/Inkling-2-Mini
  python3 tools/make_ink2_golden.py --download
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer

REPO = "thinkingmachines/Inkling-2-Mini"
REVISION = "1fb4134b9a74f9681054f4ef82e4bf1f458a2337"

E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Exact F8_E4M3FN lookup table over all 256 byte values.
_e4m3_vals = np.zeros(256, dtype=np.float32)
for _b in range(256):
    _sign = -1.0 if (_b & 0x80) else 1.0
    _exp = (_b >> 3) & 0xF
    _man = _b & 7
    if _exp == 0:
        _val = _sign * math.ldexp(float(_man), -9)
    elif _exp == 15 and _man == 7:
        _val = float("nan")
    else:
        _val = _sign * math.ldexp(1.0 + _man / 8.0, _exp - 7)
    _e4m3_vals[_b] = _val


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.sqrt(torch.sum(xf * xf, dim=-1, keepdim=True))
    return xf * torch.clamp(threshold / (norm + 1e-6), max=1.0)


def attn_res(states: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
    st = torch.stack([s.float() for s in states], dim=-2)  # [T, M, H]
    k = rms_norm(st, None, 1e-6)
    q = query.float()
    squeeze = q.ndim == 1
    if squeeze:
        q = q.unsqueeze(0)
    scores = torch.einsum("tmh,qh->tmq", k, q) * (k.shape[-1] ** -0.5)
    weights = F.softmax(scores, dim=1)
    out = torch.einsum("tmq,tmh->qth", weights, k)
    return out[0] if squeeze else out


def dequant_expert_nvfp4(st, layer: int, wname: str, e: int) -> torch.Tensor:
    base = f"model.llm.layers.{layer}.mlp.experts.{wname}"
    w_q = st.get_slice(base)[e].numpy().astype(np.uint8)
    sc = st.get_slice(base + "_scale")[e].view(torch.uint8).numpy()
    amax = float(st.get_slice(base + "_amax")[e].float().item())
    denom = float(st.get_tensor(base + "_outer_scale_denominator")[0].float().item())
    rows, half_cols = w_q.shape
    vals = np.empty((rows, half_cols * 2), dtype=np.float32)
    e2m1_np = E2M1_TABLE.numpy()
    vals[:, 0::2] = e2m1_np[w_q & 15]
    vals[:, 1::2] = e2m1_np[w_q >> 4]
    vals.reshape(rows, half_cols // 8, 16)[:] *= _e4m3_vals[sc][:, :, None]
    vals *= np.float32(amax) / np.float32(denom)
    return torch.from_numpy(vals)


def dequant_expert_mxfp8(st, layer: int, wname: str, e: int) -> torch.Tensor:
    base = f"model.llm.layers.{layer}.mlp.experts.{wname}"
    w_u8 = st.get_slice(base)[e].view(torch.uint8).numpy()
    sc = st.get_slice(base + "_scale")[e].numpy().astype(np.int32)
    vals = _e4m3_vals[w_u8].copy()
    bscale = np.ldexp(np.ones(sc.shape, dtype=np.float32), sc - 127)
    rows, cols = vals.shape
    vals.reshape(rows, cols // 32, 32)[:] *= bscale[:, :, None]
    return torch.from_numpy(vals)


class Inkling2Ref:
    def __init__(self, model_dir: Path):
        self.dir = model_dir
        self.cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
        self.st = safe_open(str(model_dir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu")
        self.H = self.cfg["hidden_size"]
        self.Hq = self.cfg["num_attention_heads"]
        self.Hkv = self.cfg["num_key_value_heads"]
        self.D = self.cfg["head_dim"]
        self.d_rel = self.cfg["d_rel"]
        self.rel_extent = self.cfg["rel_extent"]
        self.swa = self.cfg["sliding_window_size"]
        self.local_ids = set(self.cfg["local_layer_ids"])
        self.eps = float(self.cfg["rms_norm_eps"])
        self.block_sz = self.cfg["n_layers_per_attn_res_block"]
        loop = self.cfg["loop_config"]
        self.core_start = loop["core_start"]
        self.core_end = loop["core_end"]
        self.num_passes = loop["num_passes"]
        self.schedule = (
            list(range(self.core_start))
            + list(range(self.core_start, self.core_end)) * self.num_passes
            + list(range(self.core_end, self.cfg["num_hidden_layers"]))
        )
        self.w = {
            k: self.st.get_tensor(k)
            for k in self.st.keys()
            if k.startswith("model.llm.")
            and ".mlp.experts." not in k
            and k not in ("model.llm.embed.weight", "model.llm.unembed.weight")
        }
        q_am = self.w["model.llm.middle_cycle_cross_pass_queries"].float().reshape(
            self.core_end - self.core_start, 2, self.H
        )
        q_kv = self.w["model.llm.middle_cycle_cross_pass_kv_queries"].float().reshape(
            self.core_end - self.core_start, 2, self.H
        )
        self.msg_queries = torch.cat([q_am[:, :1], q_kv, q_am[:, 1:]], dim=1)
        self.expert_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def get_expert(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, e)
        if key not in self.expert_cache:
            if layer < 20:
                w13 = dequant_expert_nvfp4(self.st, layer, "w13_weight", e)
                w2 = dequant_expert_nvfp4(self.st, layer, "w2_weight", e)
            else:
                w13 = dequant_expert_mxfp8(self.st, layer, "w13_weight", e)
                w2 = dequant_expert_mxfp8(self.st, layer, "w2_weight", e)
            self.expert_cache[key] = (w13, w2)
        return self.expert_cache[key]

    def reset_cache(self):
        return [
            {
                "k": [],
                "v": [],
                "conv_k": torch.zeros((3, self.Hkv * self.D), dtype=torch.float32),
                "conv_v": torch.zeros((3, self.Hkv * self.D), dtype=torch.float32),
                "conv_a": torch.zeros((3, self.H), dtype=torch.float32),
                "conv_m": torch.zeros((3, self.H), dtype=torch.float32),
            }
            for _ in range(len(self.schedule))
        ]

    @staticmethod
    def sconv_static(x: torch.Tensor, w: torch.Tensor, hist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        wf = w.float().squeeze(1)
        ctx = torch.cat([hist, xf], dim=0)
        T = xf.shape[0]
        conv = (
            ctx[0:T] * wf[:, 0]
            + ctx[1 : T + 1] * wf[:, 1]
            + ctx[2 : T + 2] * wf[:, 2]
            + ctx[3 : T + 3] * wf[:, 3]
        )
        return (conv + xf) / math.sqrt(5.0), ctx[-3:].clone()

    @staticmethod
    def sconv_dynamic(
        x: torch.Tensor,
        dyn_in: torch.Tensor,
        static_w: torch.Tensor,
        weight_proj: torch.Tensor,
        hist: torch.Tensor,
        n_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xf = x.float()
        T, C = xf.shape
        head_dim = C // n_heads
        ctx = torch.cat([hist, xf], dim=0)
        dyn = (dyn_in.float() @ weight_proj.float().t()).reshape(T, n_heads, 1, 4)
        sw = static_w.float().t().reshape(1, n_heads, head_dim, 4)
        weights = sw + dyn
        out = torch.zeros((T, n_heads, head_dim), dtype=torch.float32)
        for lag in range(4):
            vals = ctx[3 - lag : 3 - lag + T].reshape(T, n_heads, head_dim)
            out = out + vals * weights[..., lag]
        return (out.reshape(T, C) + xf) / 3.0, ctx[-3:].clone()

    def message_gate(self, h: torch.Tensor, m: torch.Tensor, layer: int, name: str) -> torch.Tensor:
        hf, mf = h.float(), m.float()
        inp = torch.cat([rms_norm(hf, None, self.eps), rms_norm(mf, None, self.eps)], dim=-1)
        p = f"model.llm.layers.{layer}.controllers.{name}"
        g = inp @ self.w[f"{p}.down_weight"].float().t()
        fg = torch.tanh(g @ self.w[f"{p}.forget_weight"].float().t())
        gn = torch.tanh(g @ self.w[f"{p}.gain_weight"].float().t())
        return (1.0 - fg) * hf + gn * mf

    def forward(self, token_ids: list[int], caches: list[dict], pos0: int, record_dumps: bool = False, fp16_kv: bool = False):
        T = len(token_ids)
        emb_sl = self.st.get_slice("model.llm.embed.weight")
        tok_rows = torch.stack([emb_sl[tid].float() for tid in token_ids], dim=0)
        x = rms_norm(tok_rows, self.w["model.llm.embed_norm.weight"], self.eps)
        dumps = {}
        if record_dumps:
            dumps["after_embed"] = self._snap(x[-1])

        blocks: list[torch.Tensor] = []
        pre_loop = None
        previous = None
        positions = torch.arange(pos0, pos0 + T, dtype=torch.float32) + 1.0
        tau = 1.0 + 0.1 * torch.log(torch.clamp(positions / 8192.0, min=1.0))

        for v_idx, ip in enumerate(self.schedule):
            c = caches[v_idx]
            lp = f"model.llm.layers.{ip}"
            is_swa = ip in self.local_ids

            messages = None
            if ip == self.core_start:
                if pre_loop is None:
                    pre_loop = (x, list(blocks))
                else:
                    previous = (blocks[len(pre_loop[1]) + 1 :], x)
                    x, blocks = pre_loop[0], list(pre_loop[1])
            if previous is not None and ip < self.core_end:
                prev_blocks, prev_last = previous
                messages = attn_res([*prev_blocks, prev_last], self.msg_queries[ip - self.core_start])

            inp_sa = x.float()
            if ip == 0:
                inputs = [inp_sa, inp_sa, inp_sa]
            else:
                mixed = attn_res([*blocks, inp_sa], self.w[f"model.llm.attn_res.attn_queries.{ip - 1}"])
                inputs = [mixed, mixed, mixed] if mixed.ndim == 2 else [mixed[0], mixed[1], mixed[2]]

            if messages is not None:
                inputs = [
                    self.message_gate(inputs[0], messages[0], ip, "attn"),
                    self.message_gate(inputs[1], messages[1], ip, "k"),
                    self.message_gate(inputs[2], messages[2], ip, "v"),
                ]

            if is_swa and messages is None:
                n0 = rms_norm(inputs[0], self.w[f"{lp}.attn_norm.weight"], self.eps)
                normed = [n0, n0, n0]
            elif is_swa:
                nw = self.w[f"{lp}.attn_norm.weight"]
                normed = [rms_norm(p, nw, self.eps) for p in inputs]
            else:
                normed = [
                    rms_norm(inputs[0], self.w[f"{lp}.q_input_norm.weight"], self.eps),
                    rms_norm(inputs[1], self.w[f"{lp}.k_input_norm.weight"], self.eps),
                    rms_norm(inputs[2], self.w[f"{lp}.v_input_norm.weight"], self.eps),
                ]

            q = normed[0] @ self.w[f"{lp}.attn.wq_du.weight"].float().t()
            k = normed[1] @ self.w[f"{lp}.attn.wk_dv.weight"].float().t()
            v = normed[2] @ self.w[f"{lp}.attn.wv_dv.weight"].float().t()
            r = normed[0] @ self.w[f"{lp}.attn.wr_du.weight"].float().t()

            if is_swa:
                k, c["conv_k"] = self.sconv_static(k, self.w[f"{lp}.attn.k_sconv.weight"], c["conv_k"])
                v, c["conv_v"] = self.sconv_static(v, self.w[f"{lp}.attn.v_sconv.weight"], c["conv_v"])
            else:
                k, c["conv_k"] = self.sconv_dynamic(
                    k, normed[1], self.w[f"{lp}.attn.k_sconv.static_w"], self.w[f"{lp}.attn.k_sconv.weight_proj"], c["conv_k"], self.Hkv
                )
                v, c["conv_v"] = self.sconv_dynamic(
                    v, normed[2], self.w[f"{lp}.attn.v_sconv.static_w"], self.w[f"{lp}.attn.v_sconv.weight_proj"], c["conv_v"], self.Hkv
                )

            q = l2_clip(q.reshape(T, self.Hq, self.D), 22.627416997969522, self.w[f"{lp}.attn.q_l2_clip.scale"])
            k = l2_clip(k.reshape(T, self.Hkv, self.D), 22.627416997969522, None)
            v = rms_norm(v.reshape(T, self.Hkv, self.D), None, self.eps)
            r = l2_clip(r.reshape(T, self.Hq, self.d_rel), 8.0, self.w[f"{lp}.attn.rel_l2_clip.scale"])
            if fp16_kv:
                k = k.half().float()
                v = v.half().float()

            if not is_swa:
                q = q * tau[:, None, None]
                r = r * tau[:, None, None]

            c["k"].append(k)
            c["v"].append(v)
            k_all = torch.cat(c["k"], dim=0)
            v_all = torch.cat(c["v"], dim=0)

            sinks = normed[0] @ self.w[f"{lp}.attn.attn_sink_proj.weight"].float().t()
            rel_proj = self.w[f"{lp}.attn.rel_logits_proj.proj"].float()

            attn_out = torch.empty((T, self.Hq, self.D), dtype=torch.float32)
            for t in range(T):
                qpos = pos0 + t
                k0 = max(0, qpos - (self.swa - 1)) if is_swa else 0
                ks_rep = k_all[k0 : qpos + 1].repeat_interleave(3, dim=1)
                vs_rep = v_all[k0 : qpos + 1].repeat_interleave(3, dim=1)
                scores = torch.einsum("hd,lhd->hl", q[t], ks_rep) * (1.0 / math.sqrt(self.D))
                dists = torch.arange(qpos - k0, -1, -1, dtype=torch.long)
                valid = dists < self.rel_extent
                if valid.any():
                    rb = r[t] @ rel_proj
                    scores[:, valid] += rb[:, dists[valid]]
                scores_with_sink = torch.cat([scores, sinks[t].unsqueeze(-1)], dim=-1)
                probs = F.softmax(scores_with_sink, dim=-1)[:, :-1]
                attn_out[t] = torch.einsum("hl,lhd->hd", probs, vs_rep)

            attn_out = attn_out.reshape(T, self.Hq * self.D)
            gate_attn = torch.sigmoid(normed[0] @ self.w[f"{lp}.attn.gate_attn.weight"].float().t())
            attn_out = (attn_out * gate_attn) @ self.w[f"{lp}.attn.wo_ud.weight"].float().t()
            a_out, c["conv_a"] = self.sconv_static(attn_out, self.w[f"{lp}.attn_sconv.weight"], c["conv_a"])

            if ip % self.block_sz == 0:
                blocks.append(inp_sa)
                ffn_inp = a_out
            else:
                ffn_inp = inp_sa + a_out

            h_mlp = attn_res([*blocks, ffn_inp], self.w[f"model.llm.attn_res.mlp_queries.{ip}"])
            if messages is not None:
                h_mlp = self.message_gate(h_mlp, messages[3], ip, "mlp")
            x_mlp = rms_norm(h_mlp, self.w[f"{lp}.mlp_norm.weight"], self.eps)

            r_scores = torch.sigmoid(x_mlp @ self.w[f"{lp}.mlp.gate.weight"].float().t())
            r_bias = self.w[f"{lp}.mlp.gate.bias"].float()
            topk_idx = torch.topk(r_scores + r_bias, k=8, dim=-1).indices
            ew = torch.gather(r_scores, -1, topk_idx)
            ew = ew / (ew.sum(dim=-1, keepdim=True) + 1e-20) * 8.0

            moe_out = torch.zeros((T, self.H), dtype=torch.float32)
            for t in range(T):
                xt = x_mlp[t]
                for slot in range(8):
                    eid = int(topk_idx[t, slot].item())
                    w13, w2 = self.get_expert(ip, eid)
                    gu = w13 @ xt
                    gate = gu[0::2]
                    up = gu[1::2]
                    act = (4.0 * torch.tanh(gate / 4.0)) * torch.sigmoid(gate) * (25.0 * torch.tanh(up / 25.0))
                    moe_out[t] += ew[t, slot] * (w2 @ act)

            m_out, c["conv_m"] = self.sconv_static(moe_out, self.w[f"{lp}.mlp_sconv.weight"], c["conv_m"])
            x = ffn_inp + m_out

            if record_dumps and v_idx in (0, 5, 17, 18, 30, 44):
                dumps[f"v{v_idx}_out"] = self._snap(x[-1])

        h = attn_res([*blocks, x.float()], self.w["model.llm.attn_res.output_query"])
        h_norm = rms_norm(h, self.w["model.llm.norm.weight"], self.eps)
        if record_dumps:
            dumps["final_norm"] = self._snap(h_norm[-1])

        unpadded = self.cfg["unpadded_vocab_size"]
        unembed = self.st.get_slice("model.llm.unembed.weight")[:unpadded].float()
        logits = h_norm[-1] @ unembed.t()
        logits = 30.0 * torch.tanh(logits / 30.0)
        return logits, dumps

    @staticmethod
    def _snap(t: torch.Tensor) -> dict:
        arr = t.detach().cpu().float().numpy()
        return {
            "sum": float(arr.sum()),
            "l2": float(np.linalg.norm(arr)),
            "first8": [float(v) for v in arr[:8]],
            "sha256_f32": hashlib.sha256(arr.astype("<f4").tobytes()).hexdigest()[:16],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", help=f"local {REPO} directory (defaults to downloading pinned revision {REVISION[:8]})")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ink2_golden.json"))
    args = ap.parse_args()

    if args.model_dir:
        mdir = Path(args.model_dir)
    else:
        from huggingface_hub import snapshot_download

        mdir = Path(
            snapshot_download(
                repo_id=REPO,
                revision=REVISION,
                allow_patterns=["config.json", "generation_config.json", "tokenizer.json", "model*.safetensors"],
            )
        )

    tok = Tokenizer.from_file(str(mdir / "tokenizer.json"))
    p1_text = "The capital of France is"
    p2_text = (
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>Reply with one word: 2+2=<|end_message|>"
        "<|message_model|>"
    )
    p2_tpl_text = (
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        "<|message_user|><|content_text|>Reply with one word: 2+2=<|end_message|>"
    )

    ref = Inkling2Ref(mdir)
    prompts_out = []
    for name, text in [("p1_france", p1_text), ("p2_chat_2plus2", p2_text), ("p2_no_model_hdr", p2_tpl_text)]:
        ids = tok.encode(text, add_special_tokens=False).ids
        caches = ref.reset_cache()
        logits, dumps = ref.forward(ids, caches, 0, record_dumps=True)
        top10_vals, top10_idx = torch.topk(logits, 10)
        caches_h = ref.reset_cache()
        logits_h, _ = ref.forward(ids, caches_h, 0, record_dumps=False, fp16_kv=True)
        fp16_vals, fp16_idx = torch.topk(logits_h, 10)
        rollout = []
        pos = len(ids)
        next_tok = int(top10_idx[0].item())
        rollout.append(next_tok)
        for _ in range(23):
            step_logits, _ = ref.forward([next_tok], caches, pos, record_dumps=False)
            pos += 1
            next_tok = int(torch.argmax(step_logits).item())
            rollout.append(next_tok)
        decoded = tok.decode(rollout, skip_special_tokens=False)
        print(
            f"[{name}] top1={top10_idx[0].item()} (fp32 {top10_vals[0].item():.4f}, fp16 {fp16_vals[0].item():.4f}) "
            f"rollout={rollout[:8]}... text={decoded!r}",
            flush=True,
        )
        prompts_out.append(
            {
                "name": name,
                "text": text,
                "ids": ids,
                "dumps": dumps,
                "top10_ids": [int(i) for i in top10_idx.tolist()],
                "top10_vals": [float(v) for v in top10_vals.tolist()],
                "fp16_kv_top10_ids": [int(i) for i in fp16_idx.tolist()],
                "fp16_kv_top10_vals": [float(v) for v in fp16_vals.tolist()],
                "greedy_24": rollout,
                "decoded_24": decoded,
            }
        )

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps({"repo": REPO, "revision": REVISION, "prompts": prompts_out}, ensure_ascii=False, indent=2) + "\n"
    )
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
