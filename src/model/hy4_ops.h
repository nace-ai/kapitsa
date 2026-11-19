// Hy4-flash (tencent/Hy4-flash, model_type hy_v4_flash) GPU blocks that no other Kapitsa model has: the
// independent hyper-connections, the sigmoid router, the Engram mix, the gated MLA attention with sinks and a sliding
// window, and plain fp32 RMSNorm rows. Kernels in src/kernels/hy4/hy4.metal (the prefill attention is attn.metal's
// attn_prefill11_impl with its HY flag). The reference semantics are transformers' modeling_hy_v4_flash.py
// (revision c2ec8505); tools/hy4_ops_check.cpp checks every block against an fp64 CPU version of it.
// Rows are fp32 unless a parameter says half. R = rows (1 for decode, up to the prefill chunk). Weight tensors
// (norms, biases, hc params, sinks) may be fp32, bf16 or fp16 as stored: the kernels read them as they are.
// Nothing here inserts barriers: use a serial encoder (Gpu::begin()) or a grouped one (begin_grouped()). enc_attn's
// dispatches depend on each other (prep, then attention, then the decode reduce): never call it inside a group.
#pragma once
#include "loader/safetensors.h"
#include "metal/gpu.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ni::hy4 {

constexpr uint32_t kStreams = 4;  // hc_mult

// The kernels these blocks use (prewarm lists)
std::vector<std::string> kernel_names();

// ---- hyper-connections (HYV4FlashHyperConnection, HYV4FlashHyperHead) -----------------------------------------
// fn [nout, S*D], base [nout], scale [2] (pre, post; hc_head: [1]). nout 8 for a block's attn_hc / ffn_hc, 4 for
// model.hc_head.
struct Hc {
  const Tensor* fn = nullptr;
  const Tensor* base = nullptr;
  const Tensor* scale = nullptr;
  uint32_t nout = 8;
};
struct HcConst {
  uint32_t D = 3072;
  float rms_eps = 1e-5f;  // the unweighted input norm's eps (rms_norm_eps)
  float hc_eps = 1e-6f;
  float mag = 2.0f;       // hc_magnitude (post only)
};
// X fp32 [R, S*D] (stream s of row r at X[r*S*D + s*D]). When y is given, first X[s] += post_prev[r*S + s] * y[r]
// (y fp32 [R, D]: the previous block's pending injection) and X is written back. Then per row:
// inv = rsqrt(mean(X^2 over all S*D) + rms_eps); m_j = dot(fn_j, X) * inv; pre_s = sigmoid(m_s * scale[0] + base[s])
// + hc_eps; u = sum_s pre_s * X[s] (u fp32 [R, D], or half when u_half). nout 8: also post_out[r*S + s] =
// mag * sigmoid(m_{S+s} * scale[1] + base[S+s]) + hc_eps. nout 4 (hc_head): the same pre and u, no post (pass
// post_out nullptr); model.norm follows (enc_rmsnorm).
void enc_hc_pre(Encoder& e, const Hc& w, const HcConst& c, int R, const Buffer& X, const Buffer* y,
                const Buffer* post_prev, const Buffer& u, bool u_half, const Buffer* post_out);
// X[r][s] += post[r*S + s] * y[r]: flush a pending injection (before an Engram layer, and after the last block)
void enc_hc_inject(Encoder& e, uint32_t D, int R, const Buffer& X, const Buffer& y, const Buffer& post);
// X[r][s] = emb[r] for every stream (emb fp32 [R, D])
void enc_hc_expand(Encoder& e, uint32_t D, int R, const Buffer& emb, const Buffer& X);

// ---- RMSNorm rows, adds, SwiGLU ------------------------------------------------------------------------------------
// y[r] = w * (x[r] * rsqrt(mean(x[r]^2) + eps)) (w [D]); y may alias x. y_half: y is half [R, D].
void enc_rmsnorm(Encoder& e, uint32_t D, int R, const Buffer& x, const Tensor& w, float eps, const Buffer& y,
                 bool y_half = false);
// y[r] += x[r] (fp32)
void enc_add(Encoder& e, uint32_t n_per_row, int R, const Buffer& x, const Buffer& y);
// h[r, i] = silu(min(g, limit)) * clamp(u, -limit, limit) when limit > 0, silu(g) * u when limit <= 0.
// g, u fp32 [R, n]; h fp32 or half [R, n] (h may alias g or u when fp32).
void enc_swiglu(Encoder& e, uint32_t n, int R, const Buffer& g, const Buffer& u, const Buffer& h, float limit,
                bool h_half);

// ---- router (HYV4FlashTopkRouter, n_group 1) ---------------------------------------------------------------------
// logits fp32 [R, E] (from the bf16 router weight). s = sigmoid(logits); choose the top k of s + bias (bias [E],
// e_score_correction_bias); wts = s[chosen] / (sum s[chosen] + 1e-20) * scale. idx uint32 [R, k], wts fp32 [R, k],
// in descending order of s + bias (ties: the lower expert index first). E <= 512, k <= 16.
void enc_router(Encoder& e, uint32_t E, uint32_t k, float scale, int R, const Buffer& logits, const Tensor& bias,
                const Buffer& idx, const Buffer& wts);

// ---- Engram (HYV4FlashEngramLayer after the lookup) --------------------------------------------------------------
// The CPU side (hy4_engram.h) gives the normalized lookup rows; the caller runs value_proj and the four key_projs
// (without their biases) on them. This block adds the biases and does the rest, in place on X:
//   v = pre_gamma * rmsnorm_unweighted(vraw + vb)                    (eps: engram_eps)
//   per stream s: k = norm_key_s(kraw_s + kb_s), q = norm_query_s(X_s)   (weighted: w * (x * rstd), eps: engram_eps)
//   g = dot(k, q) / sqrt(D); g = sigmoid(sign(g) * sqrt(max(|g|, 1e-6))); X_s += g * v + per_hc_scale_s * v
// engram_eps is the reference's nn.RMSNorm(eps=None). It ran torch 2.15 on CUDA, where that is FLT_EPSILON
// (1.1920929e-07, aten layer_norm_kernel.cu), not bf16's 0.0078125: pass 1.1920929e-07.
// The pending hyper-connection injection must be flushed (enc_hc_inject) before this block.
struct EngramW {
  const Tensor* vb = nullptr;           // value_proj.bias [D]
  const Tensor* kb[kStreams] = {};      // key_projs.s.bias [D]
  const Tensor* nk[kStreams] = {};      // norm_key.s.weight [D]
  const Tensor* nq[kStreams] = {};      // norm_query.s.weight [D]
  const Tensor* pre_gamma = nullptr;    // unified_pre_gamma [D]
  const Tensor* hc_scale[kStreams] = {};  // unified_per_hc_scales.s [D]
};
// vraw fp32 [R, D]; kraw fp32 [S, R, D] (stream s's key for row r at kraw[(s*R + r)*D]); X fp32 [R, S*D];
// mask (optional) uint32 [R]: a row whose entry is 0 keeps X as it is (the reference's engram_mask)
void enc_engram(Encoder& e, const EngramW& w, uint32_t D, float engram_eps, int R, const Buffer& vraw,
                const Buffer& kraw, const Buffer& X, const Buffer* mask = nullptr);

// ---- attention (HYV4FlashAttention with expanded K/V) ------------------------------------------------------------
// Per head: q = [q_nope 192 | q_rope 64], k = [k_nope 192 | k_rope 64 shared by all heads], v 256. RoPE rotates the
// 64 rope dims as interleaved pairs (2i, 2i+1) by angle pos * theta^(-2i/64) (text only: the three M-RoPE positions
// are equal). Scores scaled by 1/sqrt(256), one learned sink logit per head joins the softmax denominator (then
// dropped), causal, and on window > 0 a query at position p sees keys at positions > p - window (window keys
// including itself). The output row is attn * sigmoid(gate), per element of each head's 256.
struct AttnLayer {
  uint32_t H = 32;          // query heads = KV heads (no grouping)
  float theta = 1e7f;       // 1e7 on deepseek_sparse_attention layers, 1e4 on sliding_attention ones
  uint32_t window = 0;      // 0: full causal; 512 on sliding layers
  const Tensor* sinks = nullptr;  // learnable_sink_param [H] (required)
  // The reference casts its cos/sin tables to the activations' dtype (bf16) before rotating; true rounds them the
  // same way (to bf16, nearest even).
  bool rope_bf16 = true;
};
// The layer's K/V cache: post-RoPE K and V rows, [H][cap] rows of kv_row_bytes(256) bytes each (fp16, or int8 rows
// under NI_KV_I8, as the Qwen caches). Linear (ring false): position p in row p, p < cap. Ring (a window layer's
// cache, fixed size): position p in row p % cap; enc_attn then needs cap >= window + T - 1 for a call of T rows
// (decode: cap >= window), so the chunk's own writes never overwrite a key one of its queries still needs.
struct KvCache {
  Buffer k, v;
  uint32_t H = 0, cap = 0;
  bool ring = false;
};
KvCache alloc_kv(uint32_t H, uint32_t cap, const char* label, bool ring = false);
void free_kv(KvCache& kv);
size_t kv_bytes_per_pos(uint32_t H);  // K and V rows of one position, all heads
// Grows a linear cache to cap positions keeping rows [0, keep): its own command buffer, waited for, then the old
// buffers are released. Call between passes (no work in flight may still use kv).
void grow_kv(KvCache& kv, uint32_t cap, uint32_t keep);
// Scratch for enc_attn at up to Tmax rows of up to Hmax heads (any cache length; cap is not used).
size_t attn_scratch_bytes(uint32_t Hmax, int Tmax, uint32_t cap = 0);
// T rows at positions pos0 .. pos0+T-1 (decode T = 1, prefill T <= the chunk). Inputs:
//   q    fp32 [T, H, 256]   q_b_proj output (rope dims not yet rotated)
//   kvb  fp32 [T, H, 448]   kv_b_proj output: k_nope 192 | v 256 per head
//   kr   fp32 rows of kr_ld floats; floats [kr_off, kr_off + 64) of row t are its k_rope, not yet rotated
//   gate fp32 [T, H * 256]  linear_gate output (raw logits)
// Writes the T rows' K/V to the cache at pos0.., then out [T, H * 256] (fp32, or half when out_half).
// T = 1 runs split-K decode kernels; T > 1 the prefill kernel (keys before pos0 come from the cache).
void enc_attn(Encoder& e, const AttnLayer& L, KvCache& kv, int pos0, int T, const Buffer& q, const Buffer& kvb,
              const Buffer& kr, uint32_t kr_ld, uint32_t kr_off, const Buffer& gate, const Buffer& out, bool out_half,
              const Buffer& scratch);

// ---- latent MLA: the cache holds the 576-wide latent rows, not the expanded K/V -------------------------------------
// Per layer: half [cap][576] rows = kv_a_norm latent 512 | roped k_rope 64, every value rounded to bf16 first, so a row
// holds exact bf16 values as the reference's bf16 cache does (bf16 values in [2^-17, 65504) are exact in half; half
// storage lets the kernels use half MMA on every Apple GPU). Position p sits in row p (linear) or p % cap (ring). cap
// is rounded up to a multiple of 16.
struct LatentCache {
  Buffer rows;
  uint32_t cap = 0;
  bool ring = false;
};
constexpr size_t kLatentRowBytes = 1152;
constexpr uint32_t kLatentDim = 576;
LatentCache alloc_latent(uint32_t cap, const char* label, bool ring = false);  // ring: cap >= window + Tmax - 1
void free_latent(LatentCache& c);
// Grows a linear cache keeping rows [0, keep): its own command buffer, waited for. Call between passes.
void grow_latent(LatentCache& c, uint32_t cap, uint32_t keep);
// Writes T rows at pos0..: lat fp32 rows of lat_ld floats (their first 512: the kv_a_norm output), kr fp32 rows of
// kr_ld floats with the unroped k_rope at [kr_off, kr_off + 64). (L: theta and rope_bf16.)
void enc_latent_write(Encoder& e, const AttnLayer& L, LatentCache& c, int pos0, int T, const Buffer& lat,
                      uint32_t lat_ld, const Buffer& kr, uint32_t kr_ld, uint32_t kr_off);
// Scratch for enc_attn_latent at up to Tmax rows of up to Hmax heads; max_keys: the most keys one call with T > 16
// attends (window layers: Tmax + window - 1; full layers: the largest pos0 + T).
size_t latent_scratch_bytes(uint32_t Hmax, int Tmax, uint32_t max_keys);
// enc_latent_write, then attention with the same semantics as enc_attn. q fp32 rows of q_ld floats ([H, 256] each:
// q_b output, rope unapplied); kv_b: the layer's kv_b_proj.weight, bf16 [H*448, 512] (per head 192 K-nope rows, then
// 256 V rows); gate fp32 [T, H*256]; out [T, H*256] fp32 (or half when out_half). H % 16 == 0.
// T <= 16 (decode, verify, the drafter's block): absorbed (q_nope through W_UK, one split-K pass over the 576-wide rows
// for all heads, the latent output through W_UV); kv_b and sinks come only from the arguments. T > 16: the attended
// keys' K/V are expanded from the latent rows into scratch and enc_attn's prefill kernel runs on them.
// NI_HY4_MLA_PREFILL=absorbed runs the absorbed path there instead (a measuring switch; no A/B of the two is recorded).
// Never inside a group_begin/group_end: the dispatches depend on each other.
void enc_attn_latent(Encoder& e, const AttnLayer& L, LatentCache& c, int pos0, int T, const Buffer& q, uint32_t q_ld,
                     const Buffer& lat, uint32_t lat_ld, const Buffer& kr, uint32_t kr_ld, uint32_t kr_off,
                     const Tensor& kv_b, const Buffer& gate, const Buffer& out, bool out_half, const Buffer& scratch,
                     const Buffer* sel = nullptr, bool sel_one = false);

// ---- compressed indexer (the deepseek_sparse_attention layers: 0, 1, 5, 9, ..., 41) ---------------------------------
// Reference: HYV4FlashCompressedIndexer and HYV4FlashAttention._mask_unrecalled_tokens. Semantics, per layer:
//  token key (every position s): tk = bf16(w_tkn * bf16(x * rsqrt(mean(x^2) + 1e-5))), x = bf16(token_key_projection
//    (hidden)) [128]; w_tkn = token_key_norm.weight (bf16). Cached per position.
//  group key (group g = slots 4g .. 4g+3, made once all 4 exist): from the 4 token keys x_i (fp32 from bf16),
//    v_i = Wv x_i, a_i = Wg x_i + ape[i] (fp32 [128,128] weights, y = W x; ape [4,128]), w = softmax over i of a_i per
//    dim, p = sum_i w_i v_i; n = w_kn * (p * rsqrt(mean(p^2) + 1e-5)) (key_norm.weight fp32, fp32 throughout);
//    k_g = bf16(rope(bf16(n))) with the last 64 dims rotated at position 4g (theta 1e7, bf16 cos/sin, the pairing of
//    the attention's k_rope). Cached roped.
//  query (row t at position s = pos0 + t): qi = bf16(rope(bf16(query_projection(q_resid)))) [16 heads, 128], last 64
//    dims of each head rotated at s; head weights hw = float(bf16(head_weight_projection(hidden))) [16] / sqrt(16*128).
//  logit(t, g) = sum_h hw[h] * relu(qi[h] . k_g) (fp32), for the groups g < G = s / 4 (complete before s's group).
//  selection: G <= 511: every group < G (dense); else the 511 groups of highest logit (ties: lower g first; the
//    reference's topk leaves tie order unspecified). Plus s's own group G, of which only slots 4G .. s exist (causal).
//    At most 511 * 4 + 4 = 2048 keys (candidate_topk). Equivalent to the reference's "kv_length <= 2048: dense" rule:
//    every query at s < 2048 has G <= 511 and recalls everything, whatever the call's length.
//  attention: the selected keys only, causal, sinks unchanged (they join the denominator as in dense).
// Positions are token slots (text only; the reference's vision M-RoPE positions are not handled, as in enc_attn).
struct IndexerWeights {
  const Tensor* tk_norm = nullptr;   // token_key_norm.weight bf16 [128]
  const Tensor* wv = nullptr;        // key_value_projection.weight f32 [128, 128]
  const Tensor* wg = nullptr;        // key_gate_projection.weight f32 [128, 128]
  const Tensor* ape = nullptr;       // intra_group_ape f32 [4, 128]
  const Tensor* key_norm = nullptr;  // key_norm.weight f32 [128]
};
// Per layer: tok half [cap][128] token keys (normed, unroped); grp half [cap/4][128] group keys (pooled, normed,
// roped); every value a bf16 value, as in LatentCache. Linear (indexed layers are full-context). cap is rounded up to a
// multiple of 16. 320 B per position.
struct IndexerCache {
  Buffer tok, grp;
  uint32_t cap = 0;
};
constexpr uint32_t kIndexRatio = 4, kIndexGroups = 512, kIndexDense = 2048;  // ratio, group_topk, candidate_topk
IndexerCache alloc_indexer(uint32_t cap, const char* label);
void free_indexer(IndexerCache& c);
void grow_indexer(IndexerCache& c, uint32_t cap, uint32_t keep);  // as grow_latent
// Every call of an indexed layer, before its attention (whatever pos0 + T): tk fp32 rows of tk_ld floats, the raw
// token_key_projection output at [tk_off, tk_off + 128). Writes the T token keys at pos0.., then pools every group
// whose last slot is in [pos0, pos0 + T) (reading up to 3 earlier token keys from the cache). Rollback-safe: a group
// is re-pooled whenever its last slot is written again.
void enc_index_write(Encoder& e, const AttnLayer& L, const IndexerWeights& w, IndexerCache& c, int pos0, int T,
                     const Buffer& tk, uint32_t tk_ld, uint32_t tk_off);
// Scratch for enc_index_select at up to Tmax rows over contexts of up to max_pos positions.
size_t index_scratch_bytes(int Tmax, uint32_t max_pos);
// When pos0 + T > kIndexDense (else the layer attends densely: call enc_attn_latent with sel null; calling it anyway
// gives every row its dense list). qi fp32 rows
// of qi_ld floats, the raw query_projection output [16 * 128] at [qi_off, ..); hw fp32 rows of hw_ld floats, the raw
// head_weight_projection output [16] at [hw_off, ..). Writes sel uint32 [T][kIndexGroups]: row t's selected group ids,
// ascending, n_t = min(G, 511) + 1 of them (G = (pos0 + t) / 4), the last one G itself; the rest of the row unused.
void enc_index_select(Encoder& e, const AttnLayer& L, const IndexerCache& c, int pos0, int T, const Buffer& qi,
                      uint32_t qi_ld, uint32_t qi_off, const Buffer& hw, uint32_t hw_ld, uint32_t hw_off,
                      const Buffer& sel, const Buffer& scratch);
// enc_attn_latent's sel (an indexed layer, linear cache, window 0; any T, always the absorbed path; latent_scratch_bytes
// covers it):
//   sel_one false (the target's rows, sel from enc_index_select of the same call): row t attends only to the keys of
//     its selected groups, slots <= pos0 + t.
//   sel_one true (the DFly2 drafter's block at pos0 = a, sel: one row [512] from enc_index_pick, the selection of the
//     row at position a - 1): every row t attends to that selection's positions < a, then to a .. a + t (the block's
//     own rows, written to the cache by this call). Pass it only when a > kIndexDense; below, sel null is the same set
//     (every position < a, then the block causally).
// The drafter's selection: after a verify or prefill whose call ran enc_index_select, copy the row of the last accepted
// token into dst [512] per drafter layer. row = idx ? idx[0] (int, clamped to T - 1: the verify's accepted count, as
// hy4_pick_row) : row (e.g. T - 1 after a prefill). sel is enc_index_select's [T][512].
void enc_index_pick(Encoder& e, const Buffer& sel, int T, const Buffer* idx, int row, const Buffer& dst);

}  // namespace ni::hy4
