#include "model/ink2_dflare.h"

#include "model/inkling2_kernels.h"
#include "model/nemotron_h.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ni {

using json = nlohmann::json;

namespace {

Pipeline dfl_pipe(const std::string& name) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
  }
  const Pipeline p = ink2::pipeline(name);
  std::lock_guard<std::mutex> lk(mu);
  return cache.emplace(name, p).first->second;
}

struct DflMixP {
  uint32_t R, H, row_off, total_rows;
  float a0, a1, a2, a3, eps;
};
struct DflQ8P {
  uint32_t M, N, K, add;
};
struct DflCtxKvP {
  uint32_t R, Hkv, D, pos0, cap;
  float eps, theta;
};
struct DflBlockP {
  uint32_t B, H, anchor_idx, V;
  float eps;
};
struct DflQkRopeP {
  uint32_t B, Hq, Hkv, D, pos0;
  float eps, theta;
};
struct DflAttnP {
  uint32_t B, Hq, Hkv, D, pos0, ctx_len, ctx_cap;
  float scale;
};
struct DflSwigluP {
  uint32_t n;
};
struct DflTopkP {
  uint32_t V, K;
};
struct DflMarkovP {
  uint32_t R, H, K, step, tok_prev_idx, tok_out_idx;
  float softcap, inv_softcap;
};
struct Ink2NormP {
  uint32_t T, H, has_w, w_f32;
  float eps;
};

constexpr int kCommitBatch = 64;

}  // namespace

bool Ink2DFlare::is_available(const std::string& dir) {
  namespace fs = std::filesystem;
  if (!fs::is_regular_file(dir + "/mtp.safetensors")) return false;
  const auto jp = json_file(dir + "/config.json");
  if (!jp || !jp->is_object() || !jp->contains("mtp_config") || !(*jp)["mtp_config"].is_object()) return false;
  const json& m = (*jp)["mtp_config"];
  return m.value("markov_head_type", std::string()) == "rnn_hidden";
}

Ink2DFlare::Ink2DFlare(Inkling2& target, const std::string& dir) : tg_(target), dir_(dir) {
  const auto jp = json_file(dir + "/config.json");
  if (!jp || !jp->is_object() || !jp->contains("mtp_config")) fatal("ink2_dflare: missing mtp_config");
  const json& m = (*jp)["mtp_config"];
  H_ = m.value("hidden_size", 2304);
  NL_ = m.value("num_hidden_layers", 3);
  hq_ = m.value("num_attention_heads", 24);
  hkv_ = m.value("num_key_value_heads", 8);
  D_ = m.value("head_dim", 128);
  I_ = m.value("intermediate_size", 9216);
  eps_ = m.value("rms_norm_eps", 1e-6f);
  theta_ = m.value("rope_theta", 1000000.0f);
  block_size_ = m.value("block_size", 8);
  ctx_cap_ = std::max(1, m.value("sliding_window_size", 1024) - 1);
  rank_ = m.value("markov_rank", 256);

  ink2::compile_kernels();
  load();
  alloc_buffers();
  Gpu::get().make_resident(owned_bufs_);
}

Ink2DFlare::~Ink2DFlare() {
  Gpu& g = Gpu::get();
  for (Buffer& b : owned_bufs_) g.release(b);
}

void Ink2DFlare::load() {
  const double t0 = now_ms();
  WeightStore ws;
  ws.open_dir(
      dir_, nullptr, [](const std::string& fn) { return fn == "mtp.safetensors"; }, nullptr);
  Gpu& g = Gpu::get();

  auto req = [&](const std::string& name) -> const Tensor& {
    const Tensor* t = ws.find(name);
    if (!t) fatal("ink2_dflare: missing tensor %s in mtp.safetensors", name.c_str());
    return *t;
  };
  auto to_f32_buf = [&](const std::string& name, const char* lbl) -> Buffer {
    const Tensor& t = req(name);
    const size_t n = size_t(t.numel());
    Buffer b = g.alloc(n * 4, lbl);
    owned_bufs_.push_back(b);
    float* dst = b.as<float>();
    if (t.dtype == DType::F32) {
      memcpy(dst, t.buf.ptr(), n * 4);
    } else if (t.dtype == DType::BF16) {
      const uint16_t* src = t.buf.as<uint16_t>();
      for (size_t i = 0; i < n; ++i) dst[i] = bf16_to_f32(src[i]);
    } else {
      fatal("ink2_dflare: unsupported dtype for %s", name.c_str());
    }
    return b;
  };
  auto copy_bf16_buf = [&](const std::string& name, const char* lbl) -> Buffer {
    const Tensor& t = req(name);
    if (t.dtype != DType::BF16) fatal("ink2_dflare: expected BF16 for %s", name.c_str());
    const size_t bytes = size_t(t.numel()) * 2;
    Buffer b = g.alloc(bytes, lbl);
    owned_bufs_.push_back(b);
    memcpy(b.ptr(), t.buf.ptr(), bytes);
    return b;
  };

  const unsigned nthr = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  size_t q8_bytes = 0;
  auto to_q8 = [&](const std::string& name) -> Q8Mat {
    const Tensor& t = req(name);
    if (t.shape.size() != 2 || t.dtype != DType::BF16) fatal("ink2_dflare: expected 2D BF16 for %s", name.c_str());
    Q8Mat m;
    m.N = uint32_t(t.shape[0]);
    m.K = uint32_t(t.shape[1]);
    const size_t qsz = size_t(m.N) * m.K;
    const size_t ssz = size_t(m.N) * (m.K / 32) * 2;
    m.q = g.alloc(qsz, "dfl_q8_q");
    m.s = g.alloc(ssz, "dfl_q8_s");
    owned_bufs_.push_back(m.q);
    owned_bufs_.push_back(m.s);
    q8_bytes += qsz + ssz;
    const uint16_t* src = t.buf.as<uint16_t>();
    uint8_t* qdst = m.q.as<uint8_t>();
    uint8_t* sdst = m.s.as<uint8_t>();
    const size_t K = m.K, srow = K / 32 * 2;
    auto worker = [&](size_t r0, size_t r1) {
      std::vector<float> frow(K);
      for (size_t r = r0; r < r1; ++r) {
        const uint16_t* sr = src + r * K;
        for (size_t c = 0; c < K; ++c) frow[c] = bf16_to_f32(sr[c]);
        nh_quant_q8(frow.data(), int64_t(K), qdst + r * K, sdst + r * srow);
      }
    };
    const size_t use_thr = std::min<size_t>(nthr, std::max<size_t>(1, m.N / 64));
    if (use_thr <= 1) {
      worker(0, m.N);
    } else {
      std::vector<std::thread> ts;
      for (size_t ti = 0; ti < use_thr; ++ti) ts.emplace_back(worker, m.N * ti / use_thr, m.N * (ti + 1) / use_thr);
      for (auto& th : ts) th.join();
    }
    return m;
  };

  // fuse_weight [4, 4] -> row-wise softmax alpha[4][4]
  {
    const Tensor& fw = req("fuse_weight");
    const uint16_t* p = fw.buf.as<uint16_t>();
    for (int r = 0; r < 4; ++r) {
      float mx = -1e30f, v[4];
      for (int c = 0; c < 4; ++c) {
        v[c] = bf16_to_f32(p[r * 4 + c]);
        mx = std::max(mx, v[c]);
      }
      float sum = 0.0f;
      for (int c = 0; c < 4; ++c) {
        v[c] = std::exp(v[c] - mx);
        sum += v[c];
      }
      for (int c = 0; c < 4; ++c) alpha_[r][c] = v[c] / sum;
    }
  }

  mask_embed_ = to_f32_buf("mask_embed", "dfl_mask_embed");
  norm_w_ = to_f32_buf("norm.weight", "dfl_norm_w");

  layers_.resize(size_t(NL_));
  for (int l = 0; l < NL_; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    Layer& L = layers_[size_t(l)];
    L.dflare_norm = to_f32_buf(p + "dflare_norm.weight", "dfl_dn");
    L.input_norm = to_f32_buf(p + "input_layernorm.weight", "dfl_in");
    L.post_norm = to_f32_buf(p + "post_attention_layernorm.weight", "dfl_pn");
    L.q_norm = to_f32_buf(p + "self_attn.q_norm.weight", "dfl_qn");
    L.k_norm = to_f32_buf(p + "self_attn.k_norm.weight", "dfl_kn");
    L.k_norm_ctx = to_f32_buf(p + "self_attn.k_norm_ctx.weight", "dfl_knc");
    L.q_proj = to_q8(p + "self_attn.q_proj.weight");
    L.k_proj = to_q8(p + "self_attn.k_proj.weight");
    L.v_proj = to_q8(p + "self_attn.v_proj.weight");
    L.o_proj = to_q8(p + "self_attn.o_proj.weight");
    L.k_proj_ctx = to_q8(p + "self_attn.k_proj_ctx.weight");
    L.v_proj_ctx = to_q8(p + "self_attn.v_proj_ctx.weight");
    L.gate_proj = to_q8(p + "mlp.gate_proj.weight");
    L.up_proj = to_q8(p + "mlp.up_proj.weight");
    L.down_proj = to_q8(p + "mlp.down_proj.weight");
  }

  mh_dflare_norm_ = to_f32_buf("markov_head.dflare_norm.weight", "dfl_mh_dn");
  mh_w1_ = copy_bf16_buf("markov_head.markov_w1.weight", "dfl_mh_w1");
  mh_rnn_w_ = copy_bf16_buf("markov_head.rnn_proj.weight", "dfl_mh_rnnw");
  mh_rnn_b_ = to_f32_buf("markov_head.rnn_proj.bias", "dfl_mh_rnnb");

  // Fold target unembed [V, H] @ hidden_out.weight [H, R] -> Q8_0 [V, R] + unembed @ hidden_out.bias -> F32 [V]
  {
    const Tensor& hw = req("markov_head.hidden_out.weight");
    Buffer hb = to_f32_buf("markov_head.hidden_out.bias", "dfl_mh_hb_tmp");
    const uint32_t V = uint32_t(tg_.vocab());
    fold_q_ = g.alloc(size_t(V) * size_t(rank_), "dfl_fold_q");
    fold_s_ = g.alloc(size_t(V) * size_t(rank_ / 32) * 2, "dfl_fold_s");
    fold_b_ = g.alloc(size_t(V) * 4, "dfl_fold_b");
    owned_bufs_.push_back(fold_q_);
    owned_bufs_.push_back(fold_s_);
    owned_bufs_.push_back(fold_b_);

    auto e = g.begin();
    struct { uint32_t V, H, R; } fp{V, uint32_t(H_), uint32_t(rank_)};
    e->use(dfl_pipe("dflare_fold_mm_q8"));
    e->buf(0, tg_.unembed_tensor().buf);
    e->buf(1, hw.buf);
    e->buf(2, fold_q_);
    e->buf(3, fold_s_);
    e->val(4, fp);
    e->dispatch(uint32_t(rank_ / 32), (V + 31u) / 32u, 1, 128);
    tg_.enc_unembed(*e, hb, fold_b_, 1);
    e->commit_and_wait();
  }

  logi("loaded Inkling-2 DFlare drafter (%d layers Q8, block %d, rank %d, ctx %d): %.2f GB in %.0f ms",
       NL_, block_size_, rank_, ctx_cap_,
       (q8_bytes + mh_w1_.size + mh_rnn_w_.size + fold_q_.size + fold_s_.size + fold_b_.size) / 1e9,
       now_ms() - t0);
}

void Ink2DFlare::alloc_buffers() {
  Gpu& g = Gpu::get();
  auto alloc = [&](size_t bytes, const char* lbl) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), lbl);
    owned_bufs_.push_back(b);
    return b;
  };
  const size_t B = size_t(block_size_);
  const size_t QD = size_t(hq_) * size_t(D_);
  const size_t KVD = size_t(hkv_) * size_t(D_);
  const size_t V = size_t(tg_.vocab());

  ctx_h_ = alloc(size_t(kCommitBatch) * size_t(H_) * 4, "dfl_ctx_h");
  ctx_k_ = alloc(size_t(kCommitBatch) * KVD * 4, "dfl_ctx_k");
  ctx_v_ = alloc(size_t(kCommitBatch) * KVD * 4, "dfl_ctx_v");

  bx_ = alloc(B * size_t(H_) * 4, "dfl_bx");
  bn_ = alloc(B * size_t(H_) * 4, "dfl_bn");
  bq_ = alloc(B * QD * 4, "dfl_bq");
  bk_ = alloc(B * KVD * 4, "dfl_bk");
  bv_ = alloc(B * KVD * 4, "dfl_bv");
  batt_ = alloc(B * QD * 4, "dfl_batt");
  bgate_ = alloc(B * size_t(I_) * 4, "dfl_bgate");
  bup_ = alloc(B * size_t(I_) * 4, "dfl_bup");
  bact_ = alloc(B * size_t(I_) * 4, "dfl_bact");
  draft_h_ = alloc(size_t(kMaxDrafts) * size_t(H_) * 4, "dfl_draft_h");

  const size_t nchunks = (V + 1023) / 1024;
  base_logits_ = alloc(size_t(kMaxDrafts) * V * 4, "dfl_base_logits");
  pick_v_ = alloc(nchunks * 4, "dfl_pick_v");
  pick_i_ = alloc(nchunks * 4, "dfl_pick_i");

  mh_state_ = alloc(size_t(rank_) * 4, "dfl_mh_state");
  mh_rnn_in_ = alloc((2 * size_t(rank_) + 2 * size_t(H_)) * 4, "dfl_mh_in");
  mh_latent_ = alloc(size_t(rank_) * 4, "dfl_mh_lat");
}

std::unique_ptr<Ink2DFlareSeq> Ink2DFlare::new_seq() {
  Gpu& g = Gpu::get();
  auto ds = std::make_unique<Ink2DFlareSeq>();
  const size_t kv_bytes = size_t(ctx_cap_) * size_t(hkv_) * size_t(D_) * 2;
  ds->kc.resize(size_t(NL_));
  ds->vc.resize(size_t(NL_));
  std::vector<Buffer> res;
  for (int l = 0; l < NL_; ++l) {
    ds->kc[size_t(l)] = g.alloc(kv_bytes, "dfl_seq_kc");
    ds->vc[size_t(l)] = g.alloc(kv_bytes, "dfl_seq_vc");
    memset(ds->kc[size_t(l)].ptr(), 0, kv_bytes);
    memset(ds->vc[size_t(l)].ptr(), 0, kv_bytes);
    res.push_back(ds->kc[size_t(l)]);
    res.push_back(ds->vc[size_t(l)]);
  }
  ds->rows_f = g.alloc(size_t(H_) * 4, "dfl_seq_rows_f");
  memset(ds->rows_f.ptr(), 0, ds->rows_f.size);
  res.push_back(ds->rows_f);
  g.make_resident(res);
  return ds;
}

void Ink2DFlare::reset(Ink2DFlareSeq& ds) {
  ds.ctx_len = 0;
  if (ds.rows_f) memset(ds.rows_f.ptr(), 0, ds.rows_f.size);
}

void Ink2DFlare::enc_q8(Encoder& e, const Q8Mat& w, const Buffer& x, const Buffer& y, int M, bool add) {
  const DflQ8P p{uint32_t(M), w.N, w.K, add ? 1u : 0u};
  e.use(dfl_pipe("dflare_q8_mm"));
  e.buf(0, x);
  e.buf(1, w.q);
  e.buf(2, w.s);
  e.buf(3, y);
  e.val(4, p);
  e.dispatch((w.N + 3) / 4, (uint32_t(M) + 7) / 8, 1, 128);
}

void Ink2DFlare::enc_rmsnorm(Encoder& e, const Buffer& x, const Buffer& w, const Buffer& out, int T) {
  const Ink2NormP p{uint32_t(T), uint32_t(H_), 1u, 1u, eps_};
  e.use(dfl_pipe("ink2_rmsnorm"));
  e.buf(0, x);
  e.buf(1, w);
  e.buf(2, out);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 512);
}

void Ink2DFlare::encode_commit(Encoder& e, Ink2DFlareSeq& ds, const Buffer& taps, int n_rows, int pos0,
                               int taps_stride_rows) {
  if (n_rows <= 0) return;
  const int stride = taps_stride_rows > 0 ? taps_stride_rows : n_rows;
  const int keep = std::min(n_rows, ctx_cap_);
  const int start_row = n_rows - keep;
  const int start_pos = pos0 + start_row;

  for (int l = 0; l < NL_; ++l) {
    const Layer& L = layers_[size_t(l)];
    for (int off = 0; off < keep; off += kCommitBatch) {
      const int batch = std::min(kCommitBatch, keep - off);
      {
        const DflMixP mp{uint32_t(batch), uint32_t(H_), uint32_t(start_row + off), uint32_t(stride),
                         alpha_[l][0],    alpha_[l][1], alpha_[l][2],              alpha_[l][3],     eps_};
        e.use(dfl_pipe("dflare_mix_norm"));
        e.buf(0, taps);
        e.buf(1, L.dflare_norm);
        e.buf(2, ctx_h_);
        e.val(3, mp);
        e.dispatch(uint32_t(batch), 1, 1, 512);
      }
      enc_q8(e, L.k_proj_ctx, ctx_h_, ctx_k_, batch, false);
      enc_q8(e, L.v_proj_ctx, ctx_h_, ctx_v_, batch, false);
      {
        const DflCtxKvP kp{uint32_t(batch), uint32_t(hkv_), uint32_t(D_), uint32_t(start_pos + off),
                           uint32_t(ctx_cap_), eps_, theta_};
        e.use(dfl_pipe("dflare_ctx_kv_write"));
        e.buf(0, ctx_k_);
        e.buf(1, ctx_v_);
        e.buf(2, L.k_norm_ctx);
        e.buf(3, ds.kc[size_t(l)]);
        e.buf(4, ds.vc[size_t(l)]);
        e.val(5, kp);
        e.dispatch(uint32_t(hkv_), uint32_t(batch), 1, uint32_t(D_));
      }
    }
  }
  // Update markov_head rows_f from the last committed row (n_rows - 1)
  {
    const DflMixP mp{1u,          uint32_t(H_), uint32_t(n_rows - 1), uint32_t(stride),
                     alpha_[3][0], alpha_[3][1], alpha_[3][2],         alpha_[3][3], eps_};
    e.use(dfl_pipe("dflare_mix_norm"));
    e.buf(0, taps);
    e.buf(1, mh_dflare_norm_);
    e.buf(2, ds.rows_f);
    e.val(3, mp);
    e.dispatch(1, 1, 1, 512);
  }
  ds.ctx_len = std::min(ctx_cap_, ds.ctx_len + n_rows);
}

void Ink2DFlare::encode_draft(Encoder& e, Seq& s, Ink2DFlareSeq& ds, int pos, int depth) {
  if (depth < 1 || depth > kMaxDrafts) fatal("ink2_dflare: depth %d out of range", depth);
  const int B = depth + 1;
  const uint32_t V = uint32_t(tg_.vocab());
  const float scale = 1.0f / std::sqrt(float(D_));
  const uint32_t ctx_len = uint32_t(std::min(ds.ctx_len, std::min(pos, ctx_cap_)));

  // 1. Build block [anchor_embed, mask_embed x depth] -> bx_
  {
    const DflBlockP bp{uint32_t(B), uint32_t(H_), uint32_t(pos), uint32_t(tg_.embed_tensor().shape[0]), eps_};
    e.use(dfl_pipe("dflare_build_block"));
    e.buf(0, s.tok);
    e.buf(1, tg_.embed_tensor().buf);
    e.buf(2, tg_.embed_norm_tensor().buf);
    e.buf(3, mask_embed_);
    e.buf(4, bx_);
    e.val(5, bp);
    e.dispatch(uint32_t(B), 1, 1, 512);
  }

  // 2. 3 draft transformer layers
  for (int l = 0; l < NL_; ++l) {
    const Layer& L = layers_[size_t(l)];
    enc_rmsnorm(e, bx_, L.input_norm, bn_, B);
    enc_q8(e, L.q_proj, bn_, bq_, B, false);
    enc_q8(e, L.k_proj, bn_, bk_, B, false);
    enc_q8(e, L.v_proj, bn_, bv_, B, false);
    {
      const DflQkRopeP rp{uint32_t(B), uint32_t(hq_), uint32_t(hkv_), uint32_t(D_), uint32_t(pos), eps_, theta_};
      e.use(dfl_pipe("dflare_qk_rope"));
      e.buf(0, bq_);
      e.buf(1, bk_);
      e.buf(2, L.q_norm);
      e.buf(3, L.k_norm);
      e.val(4, rp);
      e.dispatch(uint32_t(hq_ + hkv_), uint32_t(B), 1, uint32_t(D_));
    }
    {
      const DflAttnP ap{uint32_t(B), uint32_t(hq_), uint32_t(hkv_), uint32_t(D_),
                        uint32_t(pos), ctx_len, uint32_t(ctx_cap_), scale};
      e.use(dfl_pipe("dflare_attn"));
      e.buf(0, bq_);
      e.buf(1, ds.kc[size_t(l)]);
      e.buf(2, ds.vc[size_t(l)]);
      e.buf(3, bk_);
      e.buf(4, bv_);
      e.buf(5, batt_);
      e.val(6, ap);
      e.dispatch(uint32_t(hq_), uint32_t(B), 1, 128);
    }
    enc_q8(e, L.o_proj, batt_, bx_, B, true);

    enc_rmsnorm(e, bx_, L.post_norm, bn_, B);
    enc_q8(e, L.gate_proj, bn_, bgate_, B, false);
    enc_q8(e, L.up_proj, bn_, bup_, B, false);
    {
      const DflSwigluP sp{uint32_t(B) * uint32_t(I_)};
      e.use(dfl_pipe("dflare_swiglu"));
      e.buf(0, bgate_);
      e.buf(1, bup_);
      e.buf(2, bact_);
      e.val(3, sp);
      e.dispatch((sp.n + 255) / 256, 1, 1, 256);
    }
    enc_q8(e, L.down_proj, bact_, bx_, B, true);
  }

  // 3. Final norm on mask positions [1 .. depth] -> draft_h_ [depth, H]
  enc_rmsnorm(e, bx_.view(size_t(H_) * 4, size_t(depth) * size_t(H_) * 4), norm_w_, draft_h_, depth);

  // 4. Base logits via target unembed [depth, V] (raw, pre-softcap)
  tg_.enc_unembed(e, draft_h_, base_logits_, depth);

  // 5. Left-to-right RNN Markov head (step 0..depth-1) with folded Q8_0 [V, 256] full-vocab rescore
  const float softcap = tg_.config().logit_softcap;
  const float inv_sc = softcap > 0.0f ? (1.0f / softcap) : 0.0f;
  const uint32_t rnn_in_dim = uint32_t(2 * rank_ + 2 * H_);
  const uint32_t nchunks = (V + 1023u) / 1024u;
  for (int step = 0; step < depth; ++step) {
    const DflMarkovP mp{uint32_t(rank_), uint32_t(H_), uint32_t(kCandK), uint32_t(step),
                        uint32_t(pos + step), uint32_t(pos + step + 1), softcap, inv_sc};
    e.use(dfl_pipe("dflare_markov_concat"));
    e.buf(0, mh_state_);
    e.buf(1, mh_w1_);
    e.buf(2, draft_h_);
    e.buf(3, ds.rows_f);
    e.buf(4, s.tok);
    e.buf(5, mh_rnn_in_);
    e.val(6, mp);
    e.dispatch((rnn_in_dim + 255) / 256, 1, 1, 256);

    e.use(dfl_pipe("dflare_markov_gru"));
    e.buf(0, mh_rnn_in_);
    e.buf(1, mh_rnn_w_);
    e.buf(2, mh_rnn_b_);
    e.buf(3, mh_state_);
    e.buf(4, mh_latent_);
    e.val(5, mp);
    e.dispatch((uint32_t(rank_) + 3) / 4, 1, 1, 128);

    struct { uint32_t V, R, step, tok_out_idx; } fp{V, uint32_t(rank_), uint32_t(step), uint32_t(pos + step + 1)};
    e.use(dfl_pipe("dflare_fold_pick_s1"));
    e.buf(0, mh_latent_);
    e.buf(1, fold_q_);
    e.buf(2, fold_s_);
    e.buf(3, fold_b_);
    e.buf(4, base_logits_);
    e.buf(5, pick_v_);
    e.buf(6, pick_i_);
    e.val(7, fp);
    e.dispatch(nchunks, 1, 1, 128);

    e.use(dfl_pipe("dflare_fold_pick_s2"));
    e.buf(0, pick_v_);
    e.buf(1, pick_i_);
    e.buf(2, s.tok);
    e.val(3, fp);
    e.dispatch(1, 1, 1, 256);
  }
}

}  // namespace ni
