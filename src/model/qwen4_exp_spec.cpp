// Speculative decoding for Qwen3.8-Flash-Next: the MTP head (one gated-residual QSA layer fed by the trunk's pre-mix
// 4-stream state) drafts tokens; the trunk verifies the pending token plus drafts in one small-batch pass, DeltaNet
// layers roll back through Qwen35's tape and the n-gram conv window through its position-indexed ring (a rejected
// row's slot is rewritten when its position is fed again).
#include "model/kparams.h"
#include "model/qwen4_exp.h"
#include "util/log.h"
#include "vision/embed.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace ni {

namespace {

Pipeline pipe4(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

struct HcParams {
  uint32_t R, D, S, has_y, has_inj;
  float eps;
};

}  // namespace

void Qwen4Exp::enc_rows(Encoder& e, const std::vector<RowSeg>& segs, const Buffer& x, int R, uint32_t x_ld) {
  if (segs.empty()) return;
  const uint32_t K = segs[0].L->K;
  // NI_ROWS_OLD=1: Qwen35's verify GEMVs (qmvrc1 / qmvr2 / qmvr4 / the 8-row MMA GEMM)
  static const bool old = env_on("NI_ROWS_OLD");
  auto fallback = [&](const RowSeg& sg) {
    const size_t xl = x_ld ? x_ld : K, yl = sg.y_ld ? sg.y_ld : sg.L->N;
    for (int r0 = 0; r0 < R; r0 += kMaxVerify) {
      const int n = std::min(kMaxVerify, R - r0);
      enc_qmvr(e, *sg.L, x.view(size_t(r0) * xl * 4, size_t(n) * xl * 4), sg.y.view(size_t(r0) * yl * 4, size_t(n) * yl * 4),
               n, sg.add, x_ld, sg.y_ld);
    }
  };
  std::vector<const RowSeg*> rest;
  for (const RowSeg& sg : segs) rest.push_back(&sg);
  while (!rest.empty()) {
    const QLinear& f0 = *rest[0]->L;
    const bool sp0 = f0.q4() || f0.q8();
    const bool wf0 = f0.gs == 64 && (f0.bits == 3 || f0.bits == 5 || f0.bits == 6);
    std::vector<const RowSeg*> same, other;
    for (const RowSeg* sg : rest) {
      const QLinear& L = *sg->L;
      const bool match = sp0 ? (L.q4() || L.q8()) : L.same_format(f0);
      (match && same.size() < 4 ? same : other).push_back(sg);
    }
    rest = other;
    bool ok = !old && (sp0 || wf0) && K % 64 == 0;
    for (const RowSeg* sg : same) ok = ok && sg->L->K == K;
    if (!ok) {
      for (const RowSeg* sg : same) fallback(*sg);
      continue;
    }
    // 4 rows of 4/8-bit: two token pairs per weight block (qmvp: 60 vs 69 us for 16384 x 2560, 832 vs 990 us for
    // the LM head on the M5 Max); else a simdgroup per token row
    const bool qmvp = sp0 && R == 4;
    e.use(pipe4(qmvp ? "qmvp_multi" : sp0 ? "qmvs_multi" : "qmvs_multi_" + f0.fmt()));
    struct {
      uint32_t K, R, nseg, x_ld, N[4], tg_begin[5], bits[4], y_ld[4], add[4];
    } p{};
    p.K = K;
    p.R = uint32_t(R);
    p.nseg = uint32_t(same.size());
    p.x_ld = x_ld ? x_ld : K;
    uint32_t tg = 0;
    for (size_t i = 0; i < 4; ++i) {
      const RowSeg& sg = *same[std::min(i, same.size() - 1)];
      if (i < same.size()) {
        p.N[i] = sg.L->N;
        p.bits[i] = uint32_t(sg.L->bits);
        p.y_ld[i] = sg.y_ld ? sg.y_ld : sg.L->N;
        p.add[i] = sg.add ? 1u : 0u;
        p.tg_begin[i] = tg;
        tg += (sg.L->N + 7) / 8;
      }
      e.buf(int(i * 4 + 0), sg.L->w->buf);
      e.buf(int(i * 4 + 1), sg.L->s->buf);
      e.buf(int(i * 4 + 2), sg.L->b->buf);
      e.buf(int(i * 4 + 3), sg.y);
    }
    p.tg_begin[same.size()] = tg;
    e.buf(16, x);
    e.val(17, p);
    if (qmvp) e.dispatch(tg, 1, 1, 128);
    else e.dispatch(uint32_t(R), tg, 1, 64);
  }
}

// The MoE of R rows (input xrn_, output added to yb_): router and shared-expert gate on the row kernels, the top-k
// of the decode step (moe_topk512) per row, then Qwen35's per-row expert kernels.
void Qwen4Exp::enc_moe4_rows(Encoder& e, const Layer& L, int R, bool mtp_layer) {
  const int D = cfg_.hidden, k = cfg_.top_k;
  static const bool old = env_on("NI_ROWS_OLD");
  if (old || cfg_.experts != 512 || k > 16) {
    enc_moe_rows(e, L, R, yb_, mtp_layer, &xrn_);
    return;
  }
  // streamed experts: the next layer's router on the same rows too (its top-k is read ahead during this layer)
  const int li = layer_index(L);
  const bool pred = L.eg.q20 && store_ && !store_->complete() && li >= 0 && li + 1 < hi_;
  std::vector<RowSeg> segs = {{&L.router, rlog_}, {&L.shared_gate, slog_}};
  if (pred) segs.push_back({&layers_[size_t(li + 1)].router, plog_});
  enc_rows(e, segs, xrn_, R);
  kp::Topk tp{uint32_t(cfg_.experts), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  e.use(pipe4("moe_topk512"));
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  e.val(4, tp);
  e.dispatch(uint32_t(R), 1, 1, 32);
  if (pred) {
    e.use(pipe4("moe_topk512"));
    e.buf(0, plog_); e.buf(1, slog_); e.buf(2, pidx_); e.buf(3, pwts_);
    e.val(4, tp);
    e.dispatch(uint32_t(R), 1, 1, 32);
  }
  if (L.eg.q20) {
    // Q2_0 experts (check_formats: a 4-bit shared expert): unique experts, gate/up, down for R <= kMaxVerify rows
    q20::enc_verify(e, q20_rows_fetch(e, L, R, ridx_, pred), q20_shared(L), q20_shape(), R, xrn_, ridx_, rwts_, eh_,
                    yb_, ue_, uw_.view(0, 512), uw_.view(1024, 1024), un_);
    return;
  }
  kp::Moe mp{uint32_t(D), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(k), cfg_.eps};
  const uint32_t Imax = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
  const bool all4 = L.eg.q4() && L.eu.q4() && L.ed.q4() && L.sg.q4() && L.su.q4() && L.sd.q4();
  // 4-bit experts (the trunk): the decode step's arithmetic per row, each distinct expert's tiles for its rows
  // adjacent (NI_MOE4=0: Qwen35's per-row kernels)
  static const bool moe4 = !(getenv("NI_MOE4") && !strcmp(getenv("NI_MOE4"), "0"));
  if (moe4 && all4 && R <= 4 && D <= 4096 && (k + 1) * Imax <= 7168 && cfg_.moe_I % 16 == 0) {
    const struct { uint32_t R, k; } rk{uint32_t(R), uint32_t(k)};
    const uint32_t Ru = uint32_t(R);
    const Buffer umask = uw_.view(0, 256), uslot = uw_.view(1024, 256);
    e.use(pipe4("moe4_unique"));
    e.buf(0, ridx_); e.buf(1, ue_); e.buf(2, umask); e.buf(3, uslot); e.buf(4, un_);
    e.val(5, rk);
    e.dispatch(1, 1, 1, 32);
    e.use(pipe4("moe4_gate_up"));
    e.buf(16, umask); e.buf(17, uslot); e.buf(18, un_);
    e.val(19, Ru);
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, xrn_); e.buf(13, eh_);
    e.val(14, mp);
    e.buf(15, ue_);
    e.dispatch(Ru, (Imax + 15) / 16, Ru * uint32_t(k) + 1, 256);
    enc_moe4_down(e, L, R, ridx_, rwts_);
    return;
  }
  const std::string gu_k = moe_rows_kernel(true, L.eg, L.sg), dn_k = moe_rows_kernel(false, L.ed, L.sd, moe_wide());
  if (gu_k.empty() || dn_k.empty()) fatal("MoE rows: no kernels for these expert formats");
  e.use(pipe4(gu_k));
  e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
  e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
  e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
  e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
  e.buf(12, xrn_); e.buf(13, eh_);
  e.val(14, mp);
  e.buf(15, ridx_);
  constexpr uint32_t rows = 2, nsg = 8;
  const uint32_t I = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
  e.dispatch((I + rows * nsg - 1) / (rows * nsg), k + 1, R, 32 * nsg);
  kp::MoeRowsDown dp{uint32_t(D), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(k), uint32_t(L.sd.bits)};
  e.use(pipe4(dn_k));
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, ridx_); e.buf(8, rwts_); e.buf(9, yb_);
  e.val(10, dp);
  e.dispatch((D + rows * nsg - 1) / (rows * nsg), R, 1, 32 * nsg);
}

// The MoE's down projection of R <= 4 rows (h in eh_ rows of k + 1 slots, routing idx / wts), added to yb_:
// moe_down2_w's arithmetic per row (moe4_down)
void Qwen4Exp::enc_moe4_down(Encoder& e, const Layer& L, int R, const Buffer& idx, const Buffer& wts) {
  const int D = cfg_.hidden, k = cfg_.top_k;
  kp::Moe mp{uint32_t(D), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(k), cfg_.eps};
  e.use(pipe4("moe4_down"));
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, idx); e.buf(8, wts); e.buf(9, yb_);
  e.val(10, mp);
  e.dispatch(uint32_t(R), (uint32_t(D) + 15) / 16, 1, 256);
}

// DeltaNet of R rows (u in xrn_) without committing the state: the rows go to the tape
void Qwen4Exp::enc_gdn4_rows(Encoder& e, Seq& s, int li, const Layer& L, int R) {
  const int D = cfg_.hidden, VD = cfg_.lin_hv * cfg_.lin_dv;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  enc_rows(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, xrn_, R);
  kp::GdnVerify gv{uint32_t(R), uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps};
  e.use(pipe4("gdn_prep_verify"));
  e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[size_t(li)]);
  e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
  e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
  e.buf(12, s.pend_qkv[size_t(li)]);
  e.val(13, gv);
  e.dispatch(C / 128, 1, 1, 128);
  e.use(pipe4(gdn_sigmoid_ ? "gdn_step_verify_sg" : "gdn_step_verify"));
  e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
  e.buf(5, s.rec[size_t(li)]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
  e.buf(9, s.pend_k[size_t(li)]); e.buf(10, s.pend_g[size_t(li)]); e.buf(11, s.pend_d[size_t(li)]);
  e.val(12, gv);
  e.dispatch(uint32_t(cfg_.lin_hv), 1, 1, 1024);
  enc_rows(e, {{&L.out, yb_, false, uint32_t(D)}}, gout_, R, uint32_t(VD));
}

// Attention of R rows (u in xrn_) at positions pos.. against the cache (kc, vc) and the indexer's caches (ik, bk)
void Qwen4Exp::enc_attn4_rows(Encoder& e, const Seq& s, const Layer& L, const Qwen4Layer& F, const Buffer& kc,
                              const Buffer& vc, const Buffer& ik, const Buffer& bk, int pos, int R) {
  enc_rows(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}, {&F.iqk, iqk_}}, xrn_, R);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(pipe4(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, kc); e.buf(7, vc);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, uint32_t(R), 1, cfg_.head_dim);
  enc_qsa(e, s, F, iqk_, F.iqk.N, R, pos, ik, bk);
  if (qsa_sparse(pos + R - 1)) {  // rows that see > budget + ratio - 1 tokens: their selected blocks (the others: all)
    enc_qsa_select(e, bk, R, pos, 0);
    enc_attn_sparse(e, kc, vc, s.cap, R, true, 0);
    enc_rows(e, {{&L.o, yb_}}, attn_out_, R);
    return;
  }
  const uint32_t Lmax = uint32_t(pos + R);
  const uint32_t cores = uint32_t(std::max(1, Gpu::get().core_count()));
  uint32_t chunk = std::max<uint32_t>(64, (Lmax + cores - 1) / cores);
  chunk = (chunk + 15) / 16 * 16;
  uint32_t nch = (Lmax + chunk - 1) / chunk;
  while (nch * uint32_t(R) > 1024) {  // the partials' room (part_o_ / part_ml_)
    chunk *= 2;
    nch = (Lmax + chunk - 1) / chunk;
  }
  kp::AttnRows rp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos), chunk, uint32_t(s.cap),
                  1.0f / std::sqrt(float(cfg_.head_dim)), nch};
  e.use(pipe4("attn_rows_partial"));
  e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
  e.val(5, rp);
  e.dispatch(nch, cfg_.hkv, uint32_t(R), 32 * (cfg_.hq / cfg_.hkv));
  e.use(pipe4("attn_rows_reduce"));
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, rp);
  e.dispatch(cfg_.hq, uint32_t(R), 1, cfg_.head_dim);
  enc_rows(e, {{&L.o, yb_}}, attn_out_, R);
}

void Qwen4Exp::encode_verify(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step, bool v8) {
  (void)v8;  // no 8-row path (no block drafter binds to this model)
  if (R < 1 || R > kMaxVerify) fatal("encode_verify: R=%d", R);
  if (pos + R + 1 >= s.cap) fatal("context full (%d)", s.cap);
  const int D = cfg_.hidden;
  // The rows' n-gram rows: the drafts were sampled by the work encoded so far (the MTP head), so that command buffer
  // ends here and, once complete, a Metal thread gathers every row's n-grams; the verify waits for them before the
  // n-gram block (its first layer overlaps the gather)
  Buffer rows;
  uint64_t ticket = 0;
  if (c4_.ple_layer >= 0) {
    const int slot = ple_dec_next_;
    ple_dec_next_ = (ple_dec_next_ + 1) % 4;
    rows = ple_dec_[slot];
    ticket = ++ple_ticket_;
    const Seq* sq = &s;
    void* ev = ple_ev_;
    e.on_completed([this, sq, pos, R, rows, ticket, ev] {
      gather_rows(const_cast<Seq*>(sq)->tokens(), pos, R, rows.ptr(), false);
      Gpu::signal_event_cpu(ev, ticket);
    });
    e.split();
    pend_ = Pending{};
  }
  enc_embed(e, s.tok, pos, R, yb_);
  enc_expand(e, R);
  enc_verify_layers(e, s, pos, R, rows, ticket);
  enc_hop(e, Remote::Kind::verify, s, pos, R);
  s.n_pend = 0;  // folded by this verify; commit_rows() sets the new count
  enc_gr_rows(e, mix_, R, x_, 0, true, injM_, injA_, xrn_, uint32_t(D));
  enc_rows(e, {{&lm_head_, logits_}}, xrn_, R);
  const uint32_t V = uint32_t(cfg_.vocab), chunk = 2048, ntg = (V + chunk - 1) / chunk;
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  kp::TopkStage ts{V, K, chunk};
  e.use(pipe4("topk_stage1_rows"));
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, uint32_t(R), 1, 256);
  kp::SampleRows sr{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, 0, 0, 0};
  e.use(pipe4("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, samp_);
  e.val(3, sr);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void Qwen4Exp::enc_verify_layers(Encoder& e, Seq& s, int pos, int R, const Buffer& rows, uint64_t ticket) {
  const int D = cfg_.hidden;
  for (int i = lo_; i < hi_; ++i) {
    const Layer& L = layers_[size_t(i)];
    const Qwen4Layer& F = l4_[size_t(i)];
    if (i == c4_.ple_layer) {
      if (i > 0) enc_inject(e, R, x_, injM_);
      e.wait_event(ple_ev_, ticket);
      enc_ple_rows(e, s, R, pos, rows);
    }
    enc_gr_rows(e, F.ga, R, x_, 0, i > 0 && i != c4_.ple_layer, injM_, injA_, xrn_, uint32_t(D));
    if (L.linear) enc_gdn4_rows(e, s, i, L, R);
    else enc_attn4_rows(e, s, L, F, s.kc[size_t(i)], s.vc[size_t(i)], s.ixk[size_t(i)], s.ixb[size_t(i)], pos, R);
    enc_gr_rows(e, F.gm, R, x_, 0, true, injA_, injM_, xrn_, uint32_t(D));
    enc_moe4_rows(e, L, R, false);
    enc_capture4(e, i, pos, R);
  }
}

void Qwen4Exp::encode_flush(Encoder& e, Seq& s) {
  const int n = s.n_pend;  // (the flush zeroes it)
  Qwen35::encode_flush(e, s);
  if (remote_ && n > 0) remote_->flush(s.rid, n);
}

// The MTP layer over R rows of mtpX_ (positions mpos0..): attention on its own caches, then (moe) its MoE; the
// injections leave the layer's output in mtpX_ (with moe) or the attention's (without: history passes only fill the
// caches).
void Qwen4Exp::enc_mtp_layer_rows(Encoder& e, Seq& s, int mpos0, int R, bool moe) {
  const uint32_t D = uint32_t(cfg_.hidden);
  enc_gr_rows(e, mtp4_.ga, R, mtpX_, 0, false, injM_, injA_, xrn_, D);
  enc_attn4_rows(e, s, mtp_, mtp4_, s.mtp_kc, s.mtp_vc, s.mtp_ixk, s.mtp_ixb, mpos0, R);
  if (!moe) {
    enc_inject(e, R, mtpX_, injA_);
    return;
  }
  enc_gr_rows(e, mtp4_.gm, R, mtpX_, 0, true, injA_, injM_, xrn_, D);
  enc_moe4_rows(e, mtp_, R, true);
  enc_inject(e, R, mtpX_, injM_);
}

void Qwen4Exp::encode_mtp(Encoder& e, Seq& s, int mpos0, int R, int tok0, const Buffer& h, int h_row0, int draft_out,
                          bool h_trunk) {
  (void)h_trunk;  // trunk rows and the head's own rows are both pre-mix 4-stream states
  if (!mtp_loaded_) fatal("no MTP head");
  if (R > kMaxVerify) fatal("encode_mtp: R=%d", R);
  const uint32_t D = uint32_t(cfg_.hidden), S = uint32_t(c4_.hc), SD = S * D;
  enc_embed(e, s.tok, tok0, R, mtp_e_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, tok0, R, mtp_e_, int(D));  // image rows (prompt history)
  const HcParams p{uint32_t(R), D, S, 0, 0, cfg_.eps};
  e.use(pipe4("mtp4_pre"));
  e.buf(0, mtp_e_); e.buf(1, h, size_t(h_row0) * SD * 4); e.buf(2, mtp_ne_->buf); e.buf(3, mtp_nh_->buf);
  e.buf(4, xrn_); e.buf(5, xn4_);
  e.val(6, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
  enc_rows(e, {{&mtp_fce_, mtp_x_}}, xrn_, R);
  enc_rows(e, {{&mtp_fch_, hcm_}}, xn4_, int(R * S));  // fc_hidden of every stream: R * S rows of D
  e.use(pipe4("mtp4_fuse"));
  e.buf(0, mtp_x_); e.buf(1, hcm_); e.buf(2, mtpX_);
  e.val(3, p);
  e.dispatch((D + 255) / 256, uint32_t(R), 1, 256);
  const bool draft = draft_out >= 0;
  enc_mtp_layer_rows(e, s, mpos0, R, draft);
  if (!draft) return;
  enc_copy(e, mtpX_, mtp_post_);  // the rows' states, for the next chained draft
  // the draft: the last row through the head's own mix and the (reduced) LM head
  enc_gr_rows(e, mtp_mix_, 1, mtpX_, size_t(R - 1), false, injM_, injA_, xrn_, D);
  uint32_t n0 = draft_head_[0].N, n1 = draft_head_[1].N;
  if (full_draft_) {
    enc_qmv(e, lm_head_, xrn_, dlog_, nullptr, nullptr);
    n0 = lm_head_.N;
    n1 = 0;
  } else {
    enc_qmv(e, draft_head_[0], xrn_, dlog_, nullptr, nullptr);
    enc_qmv(e, draft_head_[1], xrn_, dlog_.view(size_t(n0) * 4, size_t(n1) * 4), nullptr, nullptr);
  }
  const uint32_t V = n0 + n1, chunk = 2048, ntg = (V + chunk - 1) / chunk;
  kp::TopkStage ts{V, 1, chunk};
  e.use(pipe4("topk_stage1_rows"));
  e.buf(0, dlog_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  kp::SampleRows sr{ntg, 1, 0.0f, 1.0f, 0, 0, uint32_t(draft_out), full_draft_ ? 0u : n0, draft_special0_};
  e.use(pipe4("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, s.tok);
  e.val(3, sr);
  e.dispatch(1, 1, 1, 1024);
}

// MTP history of prompt rows [pos0, pos0 + R): pairs (pre-mix state of row h_row0 + r of the last prefill chunk,
// token pos0 + 1 + r) through the head's attention, which fills its caches (nothing else of the layer is needed)
void Qwen4Exp::encode_mtp_prefill(Encoder& e, Seq& s, int pos0, int R, int h_row0) {
  if (!mtp_loaded_ || R <= 0) return;
  if (R <= kMaxMtpRows) {
    encode_mtp(e, s, pos0, R, pos0 + 1, x_, h_row0, -1);
    return;
  }
  const uint32_t D = uint32_t(cfg_.hidden), S = uint32_t(c4_.hc), SD = S * D;
  enc_embed(e, s.tok, pos0 + 1, R, mtp_e_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0 + 1, R, mtp_e_, int(D));
  const HcParams p{uint32_t(R), D, S, 0, 0, cfg_.eps};
  e.use(pipe4("mtp4_pre_h"));
  e.buf(0, mtp_e_); e.buf(1, x_, size_t(h_row0) * SD * 4); e.buf(2, mtp_ne_->buf); e.buf(3, mtp_nh_->buf);
  e.buf(4, xh_); e.buf(5, xn4h_);
  e.val(6, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
  enc_qmm(e, mtp_fce_, xh_, mtp_x_, R, false);
  enc_qmm(e, mtp_fch_, xn4h_, hcm_, int(R * S), false);
  e.use(pipe4("mtp4_fuse"));
  e.buf(0, mtp_x_); e.buf(1, hcm_); e.buf(2, mtpX_);
  e.val(3, p);
  e.dispatch((D + 255) / 256, uint32_t(R), 1, 256);
  enc_gr_prefill(e, mtp4_.ga, R, mtpX_, false, injM_, injA_);
  enc_proj(e, {{&mtp_.q, qg_, false}, {&mtp_.k, kr_, false}, {&mtp_.v, vr_, false}, {&mtp4_.iqk, iqk_, false}}, xh_, R);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos0), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(pipe4(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, mtp_.q_norm->buf); e.buf(4, mtp_.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.mtp_kc); e.buf(7, s.mtp_vc);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, uint32_t(R), 1, cfg_.head_dim);
  enc_qsa(e, s, mtp4_, iqk_, mtp4_.iqk.N, R, pos0, s.mtp_ixk, s.mtp_ixb);
}

}  // namespace ni
