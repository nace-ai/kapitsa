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

// DeltaNet of R rows (u in xrn_) without committing the state: the rows go to the tape
void Qwen4Exp::enc_gdn4_rows(Encoder& e, Seq& s, int li, const Layer& L, int R) {
  const int D = cfg_.hidden, VD = cfg_.lin_hv * cfg_.lin_dv;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  enc_qmvr_multi(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, xrn_, nullptr, R);
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
  enc_qmvr(e, L.out, gout_, yb_, R, false, uint32_t(VD), uint32_t(D));
}

// Attention of R rows (u in xrn_) at positions pos.. against the cache (kc, vc) and the indexer's caches (ik, bk)
void Qwen4Exp::enc_attn4_rows(Encoder& e, const Seq& s, const Layer& L, const Qwen4Layer& F, const Buffer& kc,
                              const Buffer& vc, const Buffer& ik, const Buffer& bk, int pos, int R) {
  enc_qmvr_multi(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}, {&F.iqk, iqk_}}, xrn_, nullptr, R);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(pipe4(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, kc); e.buf(7, vc);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, uint32_t(R), 1, cfg_.head_dim);
  enc_qsa(e, F, iqk_, F.iqk.N, R, pos, ik, bk);
  if (qsa_sparse(pos + R - 1)) {  // rows that see > budget + ratio - 1 tokens: their selected blocks (the others: all)
    enc_qsa_select(e, bk, R, pos, 0);
    enc_attn_sparse(e, kc, vc, s.cap, R, true, 0);
    enc_qmvr(e, L.o, attn_out_, yb_, R, false);
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
  enc_qmvr(e, L.o, attn_out_, yb_, R, false);
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
  for (int i = 0; i < cfg_.layers; ++i) {
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
    enc_moe_rows(e, L, R, yb_, false, &xrn_);
  }
  s.n_pend = 0;  // folded by this verify; commit_rows() sets the new count
  enc_gr_rows(e, mix_, R, x_, 0, true, injM_, injA_, xrn_, uint32_t(D));
  enc_qmvr(e, lm_head_, xrn_, logits_, R, false);
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

void Qwen4Exp::encode_flush(Encoder& e, Seq& s) { Qwen35::encode_flush(e, s); }

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
  enc_moe_rows(e, mtp_, R, yb_, true, &xrn_);
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
  enc_qmvr(e, mtp_fce_, xrn_, mtp_x_, R, false);
  for (int r0 = 0; r0 < int(R * S); r0 += kMaxVerify) {  // fc_hidden of every stream: R * S rows of D
    const int n = std::min<int>(kMaxVerify, int(R * S) - r0);
    enc_qmvr(e, mtp_fch_, xn4_.view(size_t(r0) * D * 4, size_t(n) * D * 4), hcm_.view(size_t(r0) * D * 4, size_t(n) * D * 4),
             n, false);
  }
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
  enc_qsa(e, mtp4_, iqk_, mtp4_.iqk.N, R, pos0, s.mtp_ixk, s.mtp_ixb);
}

}  // namespace ni
