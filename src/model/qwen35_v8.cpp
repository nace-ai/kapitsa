// 8-row target verify (DFlash2 block verification: anchor/pending token + up to 7 drafts).
// Every projection runs on the qmm8 MMA GEMM (src/kernels/verify8.metal): the weights are read once
// for all rows and the 8 rows fill one 8x8 simdgroup_matrix dimension. Inputs reach the GEMMs as
// "X8 tables" (half rows in the MMA operand order plus group sums) written by x8_prep, or directly by
// the producer (the SwiGLU gate/up kernel writes the down projection's table). The MoE layers route
// each row's top-8, group the (row, expert) pairs by distinct expert (moe8_plan) and read every
// touched expert once for all its rows; the shared expert is one more 8-row tile.
// Taken by encode_verify for R >= 5 (NI_V8=1: R >= 2); MTP verifies with R <= 4 keep their kernels.
#include "model/kparams.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace ni {

static Pipeline pipe8(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

static int env_int(const char* n, int def) {
  const char* v = getenv(n);
  return v && *v ? atoi(v) : def;
}

// gate/up kernel: NI_V8_GU = 1 (default: 8 simdgroups x 1+1 fragments) or 2 (4 x 2+2); NI_V8_GU_PF=1
// for the software-pipelined loop.
static int gu_np() {
  static const int v = std::clamp(env_int("NI_V8_GU", 1), 1, 2);
  return v;
}
static std::string gu_kernel() {
  static const bool pf = env_int("NI_V8_GU_PF", 0) != 0;
  return std::string(gu_np() == 1 ? "qmm8_gu1" : "qmm8_gu") + (pf ? "_pf" : "") + "_b4";
}
static uint32_t gu_threads() { return 32u * (8u / uint32_t(gu_np())); }

void Qwen35::v8_alloc() {
  if (v8_ready_) return;
  Gpu& g = Gpu::get();
  const int D = cfg_.hidden, VD = cfg_.lin_hv * cfg_.lin_dv;
  const int W = std::max({D, VD, cfg_.hq * cfg_.head_dim, cfg_.dense_I, cfg_.moe_I, cfg_.shared_I, 64});
  if (W % 64) fatal("8-row verify: widths must be multiples of 64");
  v8t_ = g.alloc(size_t(W) * 8 * 2, "x8_table");
  v8s_ = g.alloc(size_t(W) / 64 * 8 * 4, "x8_sums");
  v8tl_ = g.alloc(size_t(W) * 8 * 2, "x8_table_lo");
  v8sl_ = g.alloc(size_t(W) / 64 * 8 * 4, "x8_sums_lo");
  const int tiles = cfg_.moe ? kMaxVerify * cfg_.top_k + 1 : 1;
  const int Ih = cfg_.moe ? std::max(cfg_.moe_I, cfg_.shared_I) : std::max(cfg_.dense_I, 64);
  v8h_ = g.alloc(size_t(tiles) * Ih * 8 * 2, "x8_h_tables");
  v8hs_ = g.alloc(size_t(tiles) * Ih / 64 * 8 * 4, "x8_h_sums");
  if (cfg_.moe) {
    v8tiles_ = g.alloc(size_t(tiles) * 4, "moe8_tiles");
    v8nt_ = g.alloc(16, "moe8_ntiles");
    v8slot_ = g.alloc(size_t(tiles) * 8 * 4, "moe8_slots");
    v8args_ = g.alloc(64, "moe8_args");
    v8out_ = g.alloc(size_t(kMaxVerify) * cfg_.top_k * D * 4, "moe8_out");
    v8sh_ = g.alloc(size_t(kMaxVerify) * D * 4, "moe8_shared_out");
    v8ustat_ = g.alloc(size_t(cfg_.layers) * 4, "moe8_stats");
    v8sm_ = g.alloc(320 * 4, "moe8_scalar_tiles");
    v8hsc_ = g.alloc(size_t(2 * 64) * Ih * 4, "moe8_scalar_h");
    v8xn_ = g.alloc(size_t(8) * D * 4, "x8_rows");
  }
  v8_ready_ = true;
}

void Qwen35::set_verify_capture(const std::vector<int>& layers, Buffer capt) {
  for (int l : layers)
    if (l < 0 || l >= cfg_.layers) fatal("capture layer %d out of range", l);
  capt_layers_ = layers;
  capt_ = capt;
}

void Qwen35::enc_x8(Encoder& e, const Buffer& x, int R, int K, const Tensor* norm, bool lo, int x_ld,
                    const Buffer* xn_out) {
  if (K % 64 || R < 1 || R > 8) fatal("x8_prep: K=%d R=%d", K, R);
  kp::X8Prep p{uint32_t(K), uint32_t(R), uint32_t(x_ld ? x_ld : K), norm ? 1u : 0u, xn_out ? 1u : 0u, cfg_.eps,
               uint32_t(K), lo ? 1u : 0u};
  // x8_prep2 (default): K/256 threadgroups, coalesced table stores; NI_V8_PREP=1: x8_prep (8 threadgroups)
  static const bool prep1 = env_int("NI_V8_PREP", 2) == 1;
  const bool v2 = !prep1 && K % 256 == 0;
  e.use(pipe8(v2 ? "x8_prep2" : "x8_prep"));
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, v8t_);
  e.buf(3, v8s_);
  e.buf(4, xn_out ? *xn_out : x);
  e.val(5, p);
  e.buf(6, v8tl_);
  e.buf(7, v8sl_);
  e.dispatch(v2 ? uint32_t(K / 256) : 8u, 1, 1, 256);
}

// Tiling of the qmm8 grid: nf fragments (8 rows each) per simdgroup, cs column blocks per threadgroup,
// ks k-slices per column block (fixed-order sum in threadgroup memory), cs * ks <= 8 simdgroups.
// k-slices are added until the grid has ~1280 simdgroups (64 per core) or a slice would drop below
// 4 groups of 64 inputs. NI_V8_NF / NI_V8_CS / NI_V8_KS override (tuning).
struct V8Plan {
  int nf, cs, ks;
};
static V8Plan v8_plan(uint32_t N, uint32_t K, int nf_def) {
  static const int nf_env = env_int("NI_V8_NF", 0), cs_env = env_int("NI_V8_CS", 0), ks_env = env_int("NI_V8_KS", 0);
  V8Plan p{nf_env ? nf_env : nf_def, cs_env ? cs_env : 4, 1};
  const uint32_t blocks = (N + 8 * p.nf - 1) / (8 * p.nf), gpr = K / 64;
  if (ks_env) p.ks = ks_env;
  else
    while (p.ks < 8 && blocks * p.ks < 1280 && gpr / (p.ks * 2) >= 4) p.ks *= 2;
  p.cs = std::max(1, std::min(p.cs, 8 / p.ks));
  return p;
}

void Qwen35::enc_qmm8(Encoder& e, const std::vector<Seg8>& segs, int R, const Buffer& table, const Buffer& sums,
                      bool x2) {
  if (segs.empty() || segs.size() > 4) fatal("qmm8: %zu segments", segs.size());
  const uint32_t K = segs[0].L->K;
  const int bits = segs[0].L->bits;
  uint32_t Ntot = 0;
  for (auto& sg : segs) {
    if (sg.L->K != K || sg.L->bits != bits || sg.L->gs != 64) fatal("qmm8: segments must share K, bits, group 64");
    Ntot += sg.L->N;
  }
  if (bits != 4 && bits != 8) fatal("qmm8: %d-bit weights", bits);
  V8Plan pl = v8_plan(Ntot, K, bits == 8 ? 1 : 2);
  if (x2) pl.nf = 1;
  else if (bits == 8) pl.nf = std::min(pl.nf, 2);
  // NI_V8_GEMM picks the 4-bit inner loop: pf (default: software-pipelined), base, s (split chains), pfs
  static const std::string var = getenv("NI_V8_GEMM") && *getenv("NI_V8_GEMM") ? getenv("NI_V8_GEMM") : "pf";
  const std::string pre = x2 ? "qmm8x2" : bits == 4 && var != "base" ? "qmm8" + var : "qmm8";
  const std::string name = pre + "_b" + std::to_string(bits) + "_nf" + std::to_string(pl.nf);
  kp::Qmm8 p{};
  p.K = K;
  p.R = uint32_t(R);
  p.nseg = uint32_t(segs.size());
  p.ks = uint32_t(pl.ks);
  uint32_t tg = 0;
  e.use(pipe8(name));
  for (size_t i = 0; i < 4; ++i) {
    const Seg8& sg = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = sg.L->N;
      p.y_ld[i] = sg.y_ld ? sg.y_ld : sg.L->N;
      p.add[i] = sg.add ? 1u : 0u;
      p.tg_begin[i] = tg;
      const uint32_t blocks = (sg.L->N + 8 * pl.nf - 1) / (8 * pl.nf);
      tg += (blocks + pl.cs - 1) / pl.cs;
    }
    e.buf(int(i * 4 + 0), sg.L->w->buf);
    e.buf(int(i * 4 + 1), sg.L->s->buf);
    e.buf(int(i * 4 + 2), sg.L->b->buf);
    e.buf(int(i * 4 + 3), sg.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.buf(16, table);
  e.buf(17, sums);
  e.val(18, p);
  e.buf(19, x2 ? v8tl_ : table);
  e.buf(20, x2 ? v8sl_ : sums);
  e.dispatch(tg, 1, 1, 32 * pl.cs * pl.ks);
}

// Attention block for R rows at positions pos..pos+R-1: x_ += o_proj(attn(norm(x_))).
void Qwen35::enc_attn8(Encoder& e, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos, int R) {
  const int D = cfg_.hidden, HD = cfg_.hq * cfg_.head_dim;
  enc_x8(e, x_, R, D, L.in_norm);
  enc_qmm8(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, R, v8t_, v8s_);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                  uint32_t(cfg_.rot_dims), uint32_t(pos), uint32_t(cap), cfg_.eps, cfg_.theta};
  e.use(pipe8("attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, kc); e.buf(7, vc);
  e.val(8, ap);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, R, 1, cfg_.head_dim);
  // NI_V8_ATTN: 1 = attn8m_partial (MMA, all rows and heads of a kv head per threadgroup),
  // 0 = attn8_partial (scalar, RB rows per threadgroup)
  static const int attn_mma = env_int("NI_V8_ATTN", 1);
  const int G = cfg_.hq / cfg_.hkv;
  const bool mma = attn_mma && G <= 16 && cfg_.head_dim == 256;
  // history chunks (flash decoding): ~8 threadgroups per core over the kv heads, >= 64 keys each,
  // at most 1024 / R (partials buffer)
  static const int tg_target = env_int("NI_V8_ATTN_TGS", 160);
  const uint32_t Lmax = uint32_t(pos + R), gran = mma ? 32 : 16;
  const uint32_t want = std::max<uint32_t>(1, (uint32_t(tg_target) + cfg_.hkv - 1) / cfg_.hkv);
  uint32_t chunk = std::max<uint32_t>(64, (Lmax + want - 1) / want);
  chunk = (chunk + gran - 1) / gran * gran;
  while ((Lmax + chunk - 1) / chunk * uint32_t(R) > 1024) chunk += gran;
  const uint32_t nch = (Lmax + chunk - 1) / chunk;
  kp::AttnRows rp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos), chunk, uint32_t(cap),
                  1.0f / std::sqrt(float(cfg_.head_dim)), nch};
  const uint32_t Ru = uint32_t(R);
  if (mma) {
    e.use(pipe8("attn8m_partial"));
    e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
    e.val(5, rp);
    e.val(6, Ru);
    e.dispatch(nch, cfg_.hkv, 1, 64 * G);
  } else {
    static const int rb_env = env_int("NI_V8_ATTN_RB", 0);
    const int RB = std::max(1, std::min(rb_env ? rb_env : 32 / G, R));
    e.use(pipe8("attn8_partial"));
    e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
    e.val(5, rp);
    e.val(6, Ru);
    e.dispatch(nch, cfg_.hkv, (R + RB - 1) / RB, 32 * G * RB);
  }
  e.use(pipe8("attn_rows_reduce"));
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, rp);
  e.dispatch(cfg_.hq, R, 1, cfg_.head_dim);
  enc_x8(e, attn_out_, R, HD, nullptr);
  enc_qmm8(e, {{&L.o, x_, uint32_t(D), true}}, R, v8t_, v8s_);
}

// Dense SwiGLU MLP for R rows: x_ += down(silu(gate(n)) * up(n)), n = norm(x_).
void Qwen35::enc_mlp8(Encoder& e, const Layer& L, int R) {
  const int D = cfg_.hidden;
  const uint32_t I = L.mg.N;
  if (I % 64 || L.mu.N != I || L.mg.bits != 4 || L.mu.bits != 4 || L.mg.K != uint32_t(D)) fatal("mlp8: shapes");
  enc_x8(e, x_, R, D, L.post_norm);
  kp::Qmm8Gu p{uint32_t(D), I, 0, 0, 0, 0, 0};
  e.use(pipe8(gu_kernel()));
  e.buf(0, L.mg.w->buf); e.buf(1, L.mg.s->buf); e.buf(2, L.mg.b->buf);
  e.buf(3, L.mu.w->buf); e.buf(4, L.mu.s->buf); e.buf(5, L.mu.b->buf);
  e.buf(6, L.mg.w->buf); e.buf(7, L.mg.s->buf); e.buf(8, L.mg.b->buf);
  e.buf(9, L.mu.w->buf); e.buf(10, L.mu.s->buf); e.buf(11, L.mu.b->buf);
  e.buf(12, v8t_); e.buf(13, v8s_); e.buf(14, v8h_); e.buf(15, v8hs_);
  e.val(16, p);
  e.buf(17, v8s_); e.buf(18, v8s_);
  e.dispatch(I / 64, 1, 1, gu_threads());
  enc_qmm8(e, {{&L.md, x_, uint32_t(D), true}}, R, v8h_, v8hs_);
}

// MoE block for R rows: x_ += sum_j w_j expert_j(n) + w_s shared(n), n = norm(x_).
void Qwen35::enc_moe8(Encoder& e, const Layer& L, int R, int li) {
  const int D = cfg_.hidden, k = cfg_.top_k, E = cfg_.experts;
  const uint32_t I = uint32_t(cfg_.moe_I);
  if (L.eg.gs != 64 || L.eg.bits != 4 || L.eu.bits != 4 || L.ed.bits != 4 || L.sg.bits != 4 || L.su.bits != 4 ||
      L.sd.bits != 4 || L.sg.gs != 64 || L.sd.gs != 64 || cfg_.shared_I != cfg_.moe_I || I % 64 || E > 1024 ||
      k > 8 || D % 64)
    fatal("moe8: unsupported expert layout");
  // two-term (~22-bit) inputs and exact codes for the router: keeps the top-8 choice at fp32 level
  static const bool precise = env_int("NI_V8_ROUTER_X2", 1) != 0;
  // experts used by <= small rows run as register-dequant matvecs over just those rows instead of an
  // 8-row MMA tile (35B p700: 24.0 vs 24.9 ms per verify; docs/PERF.md)
  static const int small = std::clamp(env_int("NI_V8_MOE_SMALL", 2), 0, 2);
  enc_x8(e, x_, R, D, L.post_norm, precise, 0, small ? &v8xn_ : nullptr);
  enc_qmm8(e, {{&L.router, rlog_, uint32_t(E)}, {&L.shared_gate, slog_, 1u}}, R, v8t_, v8s_, precise);
  kp::Topk tp{uint32_t(E), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  e.use(pipe8(E == 256 && k <= 8 ? "moe_topk256_rows" : "moe_topk_rows"));
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  e.val(4, tp);
  e.dispatch(R, 1, 1, 32);
  static const int dn_nf = env_int("NI_V8_MOE_NF", 2), dn_cs = 4;
  if (D % (8 * dn_nf * dn_cs)) fatal("moe8: hidden %d", D);
  // scalar passes: 8 simdgroups x 4 columns per threadgroup, inputs staged (hidden <= 2048)
  constexpr uint32_t s_nsg = 8, SR = 4;
  if (small && D > 2048) fatal("moe8s2: hidden %d > 2048", D);
  kp::Moe8Plan pp{uint32_t(R), uint32_t(k), uint32_t(E), I / 64, uint32_t(D / (8 * dn_nf * dn_cs)), uint32_t(li),
                  uint32_t(small), I / (SR * s_nsg), uint32_t(D) / (SR * s_nsg)};
  e.use(pipe8("moe8_plan"));
  e.buf(0, ridx_); e.buf(1, v8tiles_); e.buf(2, v8nt_); e.buf(3, v8slot_); e.buf(4, v8args_);
  e.val(5, pp);
  e.buf(6, v8ustat_); e.buf(7, v8sm_);
  e.dispatch(1, 1, 1, uint32_t(std::max(32, (E + 31) / 32 * 32)));
  // gate/up + SwiGLU per touched expert (+ the shared expert), one h table per tile
  const uint32_t wstride = I * (uint32_t(D) / 8), sstride = I * (uint32_t(D) / 64);
  kp::Qmm8Gu gp{uint32_t(D), I, wstride, sstride, 1u, I * 8, I / 64 * 8};
  e.use(pipe8(gu_kernel()));
  e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
  e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
  e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
  e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
  e.buf(12, v8t_); e.buf(13, v8s_); e.buf(14, v8h_); e.buf(15, v8hs_);
  e.val(16, gp);
  e.buf(17, v8tiles_); e.buf(18, v8nt_);
  e.dispatch_indirect(v8args_, 0, gu_threads());
  // down per tile -> per (row, slot) outputs, then the fixed-order combine
  const uint32_t dw = uint32_t(D) * (I / 8), ds = uint32_t(D) * (I / 64);
  kp::Moe8Down dp{I, uint32_t(D), dw, ds, I * 8, I / 64 * 8, uint32_t(k)};
  e.use(pipe8("moe8_down_nf" + std::to_string(dn_nf)));
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, v8h_); e.buf(7, v8hs_); e.buf(8, v8out_); e.buf(9, v8sh_);
  e.val(10, dp);
  e.buf(11, v8tiles_); e.buf(12, v8nt_); e.buf(13, v8slot_);
  e.dispatch_indirect(v8args_, 12, 32 * dn_cs);
  if (small) {
    kp::Moe8s sp{uint32_t(D), I, wstride, sstride, uint32_t(k), uint32_t(D)};
    e.use(pipe8("moe8s2_gu"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, v8xn_); e.buf(7, v8hsc_);
    e.val(8, sp);
    e.buf(9, v8sm_); e.buf(10, v8nt_);
    e.dispatch_indirect(v8args_, 24, 32 * s_nsg);
    kp::Moe8s dq{I, uint32_t(D), dw, ds, uint32_t(k), I};
    e.use(pipe8("moe8s2_down"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, v8hsc_); e.buf(4, v8out_);
    e.val(5, dq);
    e.buf(6, v8sm_); e.buf(7, v8nt_);
    e.dispatch_indirect(v8args_, 36, 32 * s_nsg);
  }
  kp::Moe8Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe8("moe8_combine"));
  e.buf(0, v8out_); e.buf(1, v8sh_); e.buf(2, rwts_); e.buf(3, x_);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, R, 1, 64);
}

void Qwen35::encode_verify8(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step) {
  if (R < 1 || R > 8) fatal("encode_verify8: R=%d", R);
  if (pos + R + 1 >= s.cap) fatal("context full (%d)", s.cap);
  v8_alloc();
  const int D = cfg_.hidden;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  kp::Embed ep{uint32_t(D), uint32_t(embed_.gs), uint32_t(pos)};
  e.use(pipe8("embed_q4"));
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, s.tok); e.buf(4, x_);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, R, 1, 64);
  kp::GdnVerify gv{uint32_t(R), uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps};
  const size_t ncapt = capt_layers_.size();
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[i];
    if (L.linear) {
      enc_x8(e, x_, R, D, L.in_norm);
      enc_qmm8(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, R, v8t_, v8s_);
      e.use(pipe8("gdn_prep_verify"));
      e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
      e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
      e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
      e.buf(12, s.pend_qkv[i]);
      e.val(13, gv);
      e.dispatch(C / 128, 1, 1, 128);
      static const bool gdn8 = env_int("NI_V8_GDN", 1) != 0;  // gdn_step_verify8: one norm pass
      e.use(pipe8(gdn8 ? "gdn_step_verify8" : "gdn_step_verify"));
      e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
      e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
      e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
      e.val(12, gv);
      e.dispatch(cfg_.lin_hv, 1, 1, 1024);
      enc_x8(e, gout_, R, VD, nullptr);
      enc_qmm8(e, {{&L.out, x_, uint32_t(D), true}}, R, v8t_, v8s_);
    } else {
      enc_attn8(e, L, s.kc[i], s.vc[i], s.cap, pos, R);
    }
    if (cfg_.moe) enc_moe8(e, L, R, i);
    else enc_mlp8(e, L, R);
    for (size_t c = 0; c < ncapt; ++c)
      if (capt_layers_[c] == i) {
        kp::Capture8 cp{uint32_t(D), uint32_t(ncapt * D), uint32_t(c)};
        e.use(pipe8("capture8"));
        e.buf(0, x_); e.buf(1, capt_); e.val(2, cp);
        e.dispatch((D + 255) / 256, R, 1, 256);
      }
  }
  s.n_pend = 0;  // folded by this verify; commit_rows() sets the new count
  enc_x8(e, x_, R, D, final_norm_);
  enc_qmm8(e, {{&lm_head_, logits_}}, R, v8t_, v8s_);
  const uint32_t V = uint32_t(cfg_.vocab), chunk = 2048, ntg = (V + chunk - 1) / chunk;
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  kp::TopkStage ts{V, K, chunk};
  e.use(pipe8("topk_stage1_rows"));
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, R, 1, 256);
  kp::SampleRows sr{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, 0, 0, 0};
  e.use(pipe8("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, samp_);
  e.val(3, sr);
  e.dispatch(R, 1, 1, 1024);
}

} // namespace ni
