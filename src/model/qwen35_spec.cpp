// Speculative decoding for Qwen3.5/3.6/3.8: the model's own MTP head drafts tokens, the trunk
// verifies the pending token plus drafts in one small-batch forward, and Gated DeltaNet layers roll
// back through a tape (rows are only folded into the recurrent state once accepted).
#include "loader/safetensors.h"
#include "model/kparams.h"
#include "model/qwen35.h"
#include "util/bf16.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include <nlohmann/json.hpp>

namespace ni {

static Pipeline pipe2(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

// ---- loading ----

const Tensor* Qwen35::own_copy(const Tensor& t) {
  Buffer b = Gpu::get().alloc(t.buf.size, "mtp", /*tracked=*/false);
  memcpy(b.ptr(), t.buf.ptr(), t.buf.size);
  owned_bufs_.push_back(b);
  Tensor c = t;
  c.buf = b;
  owned_.push_back(c);
  return &owned_.back();
}

const Tensor* Qwen35::bf16_param(const Tensor* t, int& lossy) {
  if (!t || t->dtype == DType::BF16) return t;
  if (t->dtype != DType::F32 && t->dtype != DType::F16) fatal("parameter tensor of unsupported dtype");
  const size_t n = size_t(t->numel());
  Buffer b = Gpu::get().alloc(n * 2, "param_bf16", /*tracked=*/false);
  uint16_t* d = b.as<uint16_t>();
  for (size_t i = 0; i < n; ++i) {
    const float v = t->dtype == DType::F32 ? t->buf.as<float>()[i] : float(t->buf.as<_Float16>()[i]);
    d[i] = f32_to_bf16(v);
    if (bf16_to_f32(d[i]) != v) ++lossy;
  }
  owned_bufs_.push_back(b);
  Tensor c = *t;
  c.buf = b;
  c.dtype = DType::BF16;
  owned_.push_back(c);
  return &owned_.back();
}

// bf16 [N, K] -> MLX affine 8-bit, group 64 (near lossless; halves the bytes the head reads).
QLinear Qwen35::quantize8(const Tensor& w) {
  if (w.dtype != DType::BF16 || w.shape.size() != 2) fatal("quantize8: expected a 2-D bf16 tensor");
  const uint32_t N = uint32_t(w.shape[0]), K = uint32_t(w.shape[1]);
  if (K % 64) fatal("quantize8: K=%u not a multiple of 64", K);
  Gpu& g = Gpu::get();
  Buffer wq = g.alloc(size_t(N) * K, "mtp_q8", false);
  Buffer sc = g.alloc(size_t(N) * K / 64 * 2, "mtp_q8s", false);
  Buffer bi = g.alloc(size_t(N) * K / 64 * 2, "mtp_q8b", false);
  const uint16_t* src = w.buf.as<uint16_t>();
  uint8_t* q = wq.as<uint8_t>();  // 4 values per uint32, value j at bits 8j == byte j (little endian)
  uint16_t* s = sc.as<uint16_t>();
  uint16_t* b = bi.as<uint16_t>();
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g0 = 0; g0 < K; g0 += 64) {
      float lo = INFINITY, hi = -INFINITY;
      for (uint32_t k = 0; k < 64; ++k) {
        float v = bf16_to_f32(src[size_t(n) * K + g0 + k]);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      float scale = (hi - lo) / 255.0f;
      if (scale == 0.0f) scale = 1e-8f;
      uint16_t sb = f32_to_bf16(scale), bb = f32_to_bf16(lo);
      float sr = bf16_to_f32(sb), br = bf16_to_f32(bb);
      s[size_t(n) * (K / 64) + g0 / 64] = sb;
      b[size_t(n) * (K / 64) + g0 / 64] = bb;
      for (uint32_t k = 0; k < 64; ++k) {
        float v = bf16_to_f32(src[size_t(n) * K + g0 + k]);
        long qv = std::lround((v - br) / sr);
        q[size_t(n) * K + g0 + k] = uint8_t(std::clamp(qv, 0L, 255L));
      }
    }
  owned_bufs_.push_back(wq);
  owned_bufs_.push_back(sc);
  owned_bufs_.push_back(bi);
  Tensor tw{wq, DType::U32, {int64_t(N), int64_t(K / 4)}};
  Tensor ts{sc, DType::BF16, {int64_t(N), int64_t(K / 64)}};
  Tensor tb{bi, DType::BF16, {int64_t(N), int64_t(K / 64)}};
  owned_.push_back(tw);
  QLinear L;
  L.w = &owned_.back();
  owned_.push_back(ts);
  L.s = &owned_.back();
  owned_.push_back(tb);
  L.b = &owned_.back();
  L.N = N;
  L.K = K;
  L.bits = 8;
  L.gs = 64;
  return L;
}

// An already-quantized linear from the MTP file (copied into engine-owned buffers).
QLinear Qwen35::own_qlin(const WeightStore& w, const std::string& name) {
  QLinear L;
  L.w = own_copy(w.get(name + ".weight"));
  L.s = own_copy(w.get(name + ".scales"));
  L.b = own_copy(w.get(name + ".biases"));
  const int64_t words = L.w->shape.back(), groups = L.s->shape.back();
  L.N = uint32_t(L.w->shape[L.w->shape.size() - 2]);
  // 4-bit g64 is the only sidecar format in use; derive and check
  L.gs = 64;
  L.K = uint32_t(groups * 64);
  L.bits = int(words * 32 / L.K);
  if (L.bits != 4 && L.bits != 8) fatal("%s: unsupported quantization", name.c_str());
  return L;
}

// Numbered per-expert tensors (mtp.layers.0.mlp.experts.<e>.<proj>.{weight,scales,biases}) ->
// one stacked [E, N, ...] tensor each, as the trunk's switch_mlp layout.
QLinear Qwen35::stack_experts(const WeightStore& w, const std::string& prefix, const std::string& proj) {
  const int E = cfg_.experts;
  Gpu& g = Gpu::get();
  QLinear L;
  const char* parts[3] = {"weight", "scales", "biases"};
  const Tensor* first[3];
  for (int p = 0; p < 3; ++p) first[p] = &w.get(prefix + "0." + proj + "." + parts[p]);
  for (int p = 0; p < 3; ++p) {
    size_t each = first[p]->buf.size;
    Buffer b = g.alloc(each * E, "mtp_experts", false);
    for (int e = 0; e < E; ++e) {
      const Tensor& t = w.get(prefix + std::to_string(e) + "." + proj + "." + parts[p]);
      if (t.buf.size != each) fatal("MTP expert %d %s: size mismatch", e, proj.c_str());
      memcpy(b.as<uint8_t>() + each * e, t.buf.ptr(), each);
    }
    owned_bufs_.push_back(b);
    Tensor t = *first[p];
    t.buf = b;
    t.shape.insert(t.shape.begin(), E);
    owned_.push_back(t);
    (p == 0 ? L.w : p == 1 ? L.s : L.b) = &owned_.back();
  }
  const int64_t words = first[0]->shape.back(), groups = first[1]->shape.back();
  L.N = uint32_t(first[0]->shape[0]);
  // bits/group from shapes: K = words * 32 / bits = groups * gs; the MTP experts are 4-bit
  L.bits = 4;
  L.K = uint32_t(words * 32 / L.bits);
  L.gs = int(L.K / groups);
  return L;
}

void Qwen35::load_mtp(const std::string& dir) {
  WeightStore w;
  w.open_dir(dir, nullptr, [](const std::string& f) { return f == "mtp.safetensors"; });
  if (!w.has("mtp.fc.weight")) {
    logi("no MTP head in %s; speculative decoding off", dir.c_str());
    return;
  }
  double t0 = now_ms();
  auto get = [&](const std::string& n) -> const Tensor& { return w.get(n); };
  // Which trunk hidden the head was trained on: MTPLX packs record it ("base_hidden_variant").
  {
    auto read = [](const std::string& p) -> nlohmann::json {
      std::ifstream f(p);
      if (!f) return nlohmann::json();
      try { return nlohmann::json::parse(f); } catch (...) { return nlohmann::json(); }
    };
    std::string variant;
    for (auto& fn : {"config.json", "pack_config.json"}) {
      auto j = read(dir + "/" + fn);
      if (j.contains("mtplx_mtp_contract")) variant = j["mtplx_mtp_contract"].value("base_hidden_variant", variant);
    }
    if (variant.empty()) {
      auto j = read(dir + "/mtplx_runtime.json");
      if (j.contains("forge_provenance") && j["forge_provenance"].contains("mtp_contract"))
        variant = j["forge_provenance"]["mtp_contract"].value("base_hidden_variant", "");
    }
    mtp_post_norm_hidden_ = variant == "post_norm";
  }
  auto lin = [&](const std::string& n) {  // quantized in the file, or bf16 -> 8-bit here
    QLinear L = w.has(n + ".scales") ? own_qlin(w, n) : quantize8(get(n + ".weight"));
    L.mtp = true;
    return L;
  };
  // norms: the kernels read bf16 (an f32/f16 file gets an owned bf16 copy)
  int lossy = 0;
  auto norm = [&](const std::string& n) {
    const Tensor& t = get(n);
    return t.dtype == DType::BF16 ? own_copy(t) : bf16_param(&t, lossy);
  };
  mtp_pre_e_ = norm("mtp.pre_fc_norm_embedding.weight");
  mtp_pre_h_ = norm("mtp.pre_fc_norm_hidden.weight");
  mtp_norm_ = norm("mtp.norm.weight");
  mtp_fc_ = lin("mtp.fc");
  const std::string p = "mtp.layers.0.";
  Layer& L = mtp_;
  L.linear = false;
  L.in_norm = norm(p + "input_layernorm.weight");
  L.post_norm = norm(p + "post_attention_layernorm.weight");
  L.q = lin(p + "self_attn.q_proj");
  L.k = lin(p + "self_attn.k_proj");
  L.v = lin(p + "self_attn.v_proj");
  L.o = lin(p + "self_attn.o_proj");
  L.q_norm = norm(p + "self_attn.q_norm.weight");
  L.k_norm = norm(p + "self_attn.k_norm.weight");
  if (cfg_.moe) {
    L.router = lin(p + "mlp.gate");
    L.shared_gate = lin(p + "mlp.shared_expert_gate");
    L.sg = lin(p + "mlp.shared_expert.gate_proj");
    L.su = lin(p + "mlp.shared_expert.up_proj");
    L.sd = lin(p + "mlp.shared_expert.down_proj");
    L.eg = stack_experts(w, p + "mlp.experts.", "gate_proj");
    L.eu = stack_experts(w, p + "mlp.experts.", "up_proj");
    L.ed = stack_experts(w, p + "mlp.experts.", "down_proj");
    if (L.eg.gs != 32 && L.eg.gs != 64) fatal("MTP experts: group size %d unsupported", L.eg.gs);
  } else {
    L.mg = lin(p + "mlp.gate_proj");
    L.mu = lin(p + "mlp.up_proj");
    L.md = lin(p + "mlp.down_proj");
  }

  // Draft head: the 32768 lowest token ids (byte-level BPE ids are roughly in merge order, i.e.
  // by frequency) plus every added/special token, so the drafter can still propose <tool_call>,
  // </think> or <|im_end|>. 286 MB per draft step becomes ~37 MB.
  draft_split_ = std::min<uint32_t>(32768, uint32_t(cfg_.vocab));
  // Dense 27B: the full-vocabulary draft raises acceptance (depth 2: 0.73 -> 0.83) for ~1% of the
  // cycle; the 35B-A3B's cycle is short enough that the 286 MB lm_head pass outweighs the gain.
  full_draft_ = getenv("NI_FULL_DRAFT") && *getenv("NI_FULL_DRAFT") ? env_on("NI_FULL_DRAFT") : !cfg_.moe;
  draft_special0_ = uint32_t(cfg_.eos.empty() ? cfg_.vocab : *std::min_element(cfg_.eos.begin(), cfg_.eos.end()));
  auto rows_view = [&](const QLinear& src, uint32_t r0, uint32_t n) {
    QLinear v = src;
    size_t wrow = size_t(src.K) * src.bits / 8, srow = size_t(src.K / src.gs) * 2;
    owned_.push_back(Tensor{src.w->buf.view(r0 * wrow, n * wrow), src.w->dtype, {int64_t(n), int64_t(wrow / 4)}});
    v.w = &owned_.back();
    owned_.push_back(Tensor{src.s->buf.view(r0 * srow, n * srow), src.s->dtype, {int64_t(n), int64_t(src.K / src.gs)}});
    v.s = &owned_.back();
    owned_.push_back(Tensor{src.b->buf.view(r0 * srow, n * srow), src.b->dtype, {int64_t(n), int64_t(src.K / src.gs)}});
    v.b = &owned_.back();
    v.N = n;
    return v;
  };
  draft_head_[0] = rows_view(lm_head_, 0, draft_split_);
  draft_head_[1] = rows_view(lm_head_, draft_special0_, uint32_t(cfg_.vocab) - draft_special0_);
  mtp_loaded_ = true;
  logi("MTP head loaded in %.0f ms (%s, trunk hidden %s)", now_ms() - t0, cfg_.moe ? "MoE" : "dense",
       mtp_post_norm_hidden_ ? "post-norm" : "pre-norm");
}

// ---- small-batch encoders ----

void Qwen35::enc_rows_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& y, int R, int D) {
  e.use(pipe2("rows_norm_f32"));
  kp::RowsNorm p{uint32_t(D), cfg_.eps};
  e.buf(0, x);
  e.buf(1, w->buf);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(R, 1, 1, 256);
}

void Qwen35::enc_qmvr(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, int R, bool add, uint32_t x_ld,
                      uint32_t y_ld) {
  if (L.gs != 64) fatal("qmvr: group %d", L.gs);
  if (R < 1 || R > kMaxVerify) fatal("qmvr: R=%d", R);
  if (R > 4) {  // 5-8 rows (deep MTP drafts): the 8-row MMA GEMM
    v8_alloc();
    enc_x8(e, x, R, int(L.K), nullptr, false, int(x_ld ? x_ld : L.K));
    enc_qmm8(e, {{&L, y, y_ld ? y_ld : L.N, add}}, R, v8t_, v8s_);
    return;
  }
  // Two rows: one small single-row threadgroup per (row, 8-row weight block), rows fastest. The pair
  // shares each weight block through the core's cache, so verifying 2 rows costs ~1 row
  // (bench_qmvr: 17408x5120 208 us for 1 or 2 rows, vs 240 us for the 2-rows-in-registers kernel).
  static const bool pair_off = env_on("NI_NO_RC1");
  if (R == 2 && !pair_off) {
    std::string n = std::string("qmvrc1_b") + std::to_string(L.bits) + (add ? "_add" : "");
    e.use(pipe2(n));
    kp::Qmvr p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, add ? 1u : 0u};
    e.buf(0, L.w->buf); e.buf(1, L.s->buf); e.buf(2, L.b->buf); e.buf(3, x); e.buf(4, y);
    e.val(5, p);
    e.dispatch(R, (L.N + 7) / 8, 1, 64);
    return;
  }
  // Three rows: half-precision inner products with the dequantization shared across rows (the
  // fp32 version is ALU-bound at 1.5x a single row; this one measured 1.17-1.32x).
  static const bool half3_off = env_on("NI_NO_QMVH");
  if (R == 3 && !half3_off && L.bits == 4) {
    e.use(pipe2("qmvht_t3_r4"));
    kp::Qmvr p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, add ? 1u : 0u};
    e.buf(0, L.w->buf); e.buf(1, L.s->buf); e.buf(2, L.b->buf); e.buf(3, x); e.buf(4, y);
    e.val(5, p);
    constexpr uint32_t rows = 4, nsg = 8;
    e.dispatch((L.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
    return;
  }
  static const bool rowcache = env_on("NI_RC");
  if (rowcache && R > 1) {
    // one single-row threadgroup per (row, block), rows fastest: weights read from DRAM once
    const bool huge = L.N >= 65536 && !add;
    const uint32_t rows = huge ? 4 : 2, nsg = 8;
    std::string n = "qmvrc_b" + std::to_string(L.bits) + "_r" + std::to_string(rows) + (add ? "_add" : "");
    e.use(pipe2(n));
    kp::Qmvr p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, add ? 1u : 0u};
    e.buf(0, L.w->buf); e.buf(1, L.s->buf); e.buf(2, L.b->buf); e.buf(3, x); e.buf(4, y);
    e.val(5, p);
    e.buf(6, y); e.buf(7, y);
    e.dispatch(R, (L.N + rows * nsg - 1) / (rows * nsg), 1, 32 * nsg);
    return;
  }
  if (L.K > 4096 || (R == 1 && L.K > 2048 && false)) {  // wide inputs: tiled staging
    // 3-4 rows of 4-bit: 8 values per lane keeps the per-lane x copies in registers
    std::string kn = "qmvr4_b" + std::to_string(L.bits) + "_t" + std::to_string(R);
    if (L.bits == 4 && R >= 3) kn += "_v8";
    e.use(pipe2(kn));
    kp::Qmvr p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, add ? 1u : 0u};
    e.buf(0, L.w->buf); e.buf(1, L.s->buf); e.buf(2, L.b->buf); e.buf(3, x); e.buf(4, y);
    e.val(5, p);
    constexpr uint32_t rows = 4, nsg = 8;
    e.dispatch((L.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
    return;
  }
  // R > 1: x rows staged as half in threadgroup memory (R * K <= 4096 * R), 4 rows/simdgroup
  static const int ver = getenv("NI_QMVR") ? atoi(getenv("NI_QMVR")) : 3;
  // v3 (simdgroup-matrix kernel) stages R*K halfs (<= 12288) and K/64*R group sums (<= 256)
  if (ver == 3 && R > 2 && R * L.K <= 12288 && R * (L.K / 64) <= 256 && getenv("NI_QMVM")) {
    static const bool m1 = env_on("NI_QMVM1");
    e.use(pipe2(L.bits == 8 ? (m1 ? "qmvm_b8_g64" : "qmvm2_b8_g64") : (m1 ? "qmvm_b4_g64" : "qmvm2_b4_g64")));
    kp::Qmvr p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, add ? 1u : 0u};
    e.buf(0, L.w->buf); e.buf(1, L.s->buf); e.buf(2, L.b->buf); e.buf(3, x); e.buf(4, y);
    e.val(5, p);
    constexpr uint32_t nsg = 8;
    e.dispatch((L.N + 8 * nsg - 1) / (8 * nsg), 1, 1, 32 * nsg);
    return;
  }
  const bool v2 = R > 1 && ver >= 2;
  std::string n = std::string(v2 ? "qmvr2_b" : "qmvr_b") + std::to_string(L.bits) + "_g64_t" + std::to_string(R);
  e.use(pipe2(n));
  kp::Qmvr p{L.N, L.K, uint32_t(R), x_ld ? x_ld : L.K, y_ld ? y_ld : L.N, add ? 1u : 0u};
  e.buf(0, L.w->buf);
  e.buf(1, L.s->buf);
  e.buf(2, L.b->buf);
  e.buf(3, x);
  e.buf(4, y);
  e.val(5, p);
  const uint32_t rows = v2 ? 4 : 2, nsg = 8;
  e.dispatch((L.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

void Qwen35::enc_qmvr_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm, int R,
                            const Buffer* xn_out, std::vector<bool> add) {
  const uint32_t K = segs[0].L->K;
  static const bool rowcache = env_on("NI_RC");
  if (rowcache && R > 1 && segs.size() <= 4 && (add.empty() || std::none_of(add.begin(), add.end(), [](bool b) { return b; }))) {
    e.use(pipe2("qmvrc_multi_r2"));
    constexpr uint32_t rows = 2, nsg = 8;
    kp::QmvrcMulti p{};
    p.K = K;
    p.R = uint32_t(R);
    p.nseg = uint32_t(segs.size());
    p.norm = norm ? 1u : 0u;
    p.write_xn = xn_out ? 1u : 0u;
    p.eps = cfg_.eps;
    uint32_t tg = 0;
    for (size_t i = 0; i < 4; ++i) {
      const Seg& sgm = segs[std::min(i, segs.size() - 1)];
      if (i < segs.size()) {
        if (sgm.L->K != K || sgm.L->gs != 64) fatal("qmvrc_multi: segments must share K, group 64");
        p.N[i] = sgm.L->N;
        p.bits[i] = uint32_t(sgm.L->bits);
        p.tg_begin[i] = tg;
        tg += (sgm.L->N + rows * nsg - 1) / (rows * nsg);
      }
      e.buf(int(i * 4 + 0), sgm.L->w->buf);
      e.buf(int(i * 4 + 1), sgm.L->s->buf);
      e.buf(int(i * 4 + 2), sgm.L->b->buf);
      e.buf(int(i * 4 + 3), sgm.y);
    }
    p.tg_begin[segs.size()] = tg;
    e.buf(16, x);
    e.val(17, p);
    e.buf(18, norm ? norm->buf : x);
    e.buf(19, xn_out ? *xn_out : x);
    e.dispatch(R, tg, 1, 32 * nsg);
    return;
  }
  static const bool off = env_on("NI_NO_QMVR_MULTI");
  if (off || (R != 2 && R != 3) || K > 2048 || segs.size() > 4) {
    const Buffer* xin = &x;
    if (norm) {
      enc_rows_norm(e, x, norm, xrn_, R, int(K));
      xin = &xrn_;
    }
    if (xn_out && xin != xn_out) fatal("enc_qmvr_multi: xn_out fallback needs xn_out == xrn_");
    for (size_t i = 0; i < segs.size(); ++i) enc_qmvr(e, *segs[i].L, *xin, segs[i].y, R, i < add.size() && add[i]);
    return;
  }
  e.use(pipe2(R == 2 ? "qmvr_multi_t2" : "qmvr_multi_t3"));
  constexpr uint32_t rows = 4, nsg = 8;
  kp::QmvrMulti p{};
  p.K = K;
  p.R = uint32_t(R);
  p.nseg = uint32_t(segs.size());
  p.norm = norm ? 1u : 0u;
  p.write_xn = xn_out ? 1u : 0u;
  p.eps = cfg_.eps;
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const Seg& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      if (s.L->K != K || s.L->gs != 64) fatal("qmvr_multi: segments must share K, group 64");
      p.N[i] = s.L->N;
      p.bits[i] = uint32_t(s.L->bits);
      p.add[i] = i < add.size() && add[i] ? 1u : 0u;
      p.tg_begin[i] = tg;
      tg += (s.L->N + rows * nsg - 1) / (rows * nsg);
    }
    e.buf(int(i * 4 + 0), s.L->w->buf);
    e.buf(int(i * 4 + 1), s.L->s->buf);
    e.buf(int(i * 4 + 2), s.L->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.buf(16, x);
  e.val(17, p);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, xn_out ? *xn_out : x);
  e.dispatch(tg, 1, 1, 32 * nsg);
}

// Attention block for R rows at positions pos..pos+R-1; x += o_proj(attn(norm(x))).
void Qwen35::enc_attn_rows(Encoder& e, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos, int R,
                           const Buffer& x) {
  enc_qmvr_multi(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, x, L.in_norm, R, &xrn_);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                  uint32_t(cfg_.rot_dims), uint32_t(pos), uint32_t(cap), cfg_.eps, cfg_.theta};
  e.use(pipe2("attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, kc); e.buf(7, vc);
  e.val(8, ap);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, R, 1, cfg_.head_dim);
  const uint32_t Lmax = uint32_t(pos + R);
  uint32_t chunk = std::max<uint32_t>(64, (Lmax + 19) / 20);
  chunk = (chunk + 15) / 16 * 16;
  uint32_t nch = (Lmax + chunk - 1) / chunk;
  if (nch * R > 1024) fatal("attention rows: too many chunks");
  kp::AttnRows rp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos), chunk, uint32_t(cap),
                  1.0f / std::sqrt(float(cfg_.head_dim)), nch};
  e.use(pipe2("attn_rows_partial"));
  e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
  e.val(5, rp);
  e.dispatch(nch, cfg_.hkv, R, 32 * (cfg_.hq / cfg_.hkv));
  e.use(pipe2("attn_rows_reduce"));
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, rp);
  e.dispatch(cfg_.hq, R, 1, cfg_.head_dim);
  enc_qmvr(e, L.o, attn_out_, x, R, true);
}

// MoE block for R rows: x += moe(norm(x)).
void Qwen35::enc_moe_rows(Encoder& e, const Layer& L, int R, const Buffer& x, bool mtp_layer) {
  const int D = cfg_.hidden, k = cfg_.top_k;
  enc_qmvr_multi(e, {{&L.router, rlog_}, {&L.shared_gate, slog_}}, x, L.post_norm, R, &xrn_);
  if (cfg_.experts != 256 || k > 8) fatal("MoE rows: E must be 256 and k <= 8");
  kp::Topk tp{uint32_t(cfg_.experts), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  e.use(pipe2("moe_topk256_rows"));
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  e.val(4, tp);
  e.dispatch(R, 1, 1, 32);
  kp::Moe mp{uint32_t(D), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(k), cfg_.eps};
  // opt-in: measured slower than per-row expert reads on M4 Pro (see docs/PERF.md)
  static const bool no_dedupe = !env_on("NI_DEDUPE");
  static const bool rc_moe = env_on("NI_RC");
  if (!mtp_layer && rc_moe && R > 1 && L.eg.gs == 64 && L.sg.bits == 4 && L.sd.bits == 4 && D <= 4096) {
    struct { uint32_t R, k; } up{uint32_t(R), uint32_t(k)};
    e.use(pipe2("moe_unique_rows"));
    e.buf(0, ridx_); e.buf(1, rwts_); e.buf(2, ue_); e.buf(3, uw_); e.buf(4, un_);
    e.val(5, up);
    e.dispatch(1, 1, 1, 32);
    const uint32_t Ru = uint32_t(R);
    constexpr uint32_t rows = 2, nsg = 8;
    const uint32_t I = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
    e.use(pipe2("moe_gate_up_rc"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, xrn_); e.buf(13, eh_);
    e.val(14, mp);
    e.buf(15, ue_); e.buf(16, uw_); e.buf(17, un_); e.val(18, Ru);
    e.dispatch(R, (I + rows * nsg - 1) / (rows * nsg), R * k + 1, 32 * nsg);
    e.use(pipe2("moe_down_rc"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
    e.buf(6, eh_); e.buf(7, ue_); e.buf(8, uw_); e.buf(9, un_); e.buf(10, rwts_); e.buf(11, x);
    e.val(12, mp); e.val(13, Ru);
    e.dispatch(R, (D + rows * nsg - 1) / (rows * nsg), 1, 32 * nsg);
    return;
  }
  if (!mtp_layer && !no_dedupe && (R == 2 || R == 3) && L.eg.gs == 64 && L.sg.bits == 4 && L.sd.bits == 4 &&
      D <= 2048) {
    // tokens being verified share experts: read each distinct expert once for all rows
    struct { uint32_t R, k; } up{uint32_t(R), uint32_t(k)};
    e.use(pipe2("moe_unique_rows"));
    e.buf(0, ridx_); e.buf(1, rwts_); e.buf(2, ue_); e.buf(3, uw_); e.buf(4, un_);
    e.val(5, up);
    e.dispatch(1, 1, 1, 32);
    const std::string sfx = R == 2 ? "t2" : "t3";
    const uint32_t Ru = uint32_t(R);
    constexpr uint32_t rows = 4, nsg = 8;
    const uint32_t I = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
    e.use(pipe2("moe_gate_up_dedupe_" + sfx));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, xrn_); e.buf(13, eh_);
    e.val(14, mp);
    e.buf(15, ue_); e.buf(16, un_); e.val(17, Ru);
    e.dispatch((I + rows * nsg - 1) / (rows * nsg), R * k + 1, 1, 32 * nsg);
    e.use(pipe2("moe_down_dedupe_" + sfx));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
    e.buf(6, eh_); e.buf(7, ue_); e.buf(8, uw_); e.buf(9, un_); e.buf(10, rwts_); e.buf(11, x);
    e.val(12, mp); e.val(13, Ru);
    e.dispatch((D + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
    return;
  }
  const bool g32 = L.eg.gs == 32;
  if (mtp_layer && L.sg.bits != 8) fatal("MTP shared expert must be 8-bit");
  e.use(pipe2(g32 ? "moe_gate_up_rows_g32_s8" : "moe_gate_up_rows_g64"));
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
  e.use(pipe2(g32 ? "moe_down_rows_g32_s8" : "moe_down_rows_g64"));
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, ridx_); e.buf(8, rwts_); e.buf(9, x);
  e.val(10, dp);
  e.dispatch((D + rows * nsg - 1) / (rows * nsg), R, 1, 32 * nsg);
}

// Dense SwiGLU MLP for R rows: x += down(silu(gate(n)) * up(n)), n = norm(x).
void Qwen35::enc_mlp_rows(Encoder& e, const Layer& L, int R, const Buffer& x) {
  const int D = cfg_.hidden;
  enc_rows_norm(e, x, L.post_norm, xrn_, R, D);
  enc_qmvr(e, L.mg, xrn_, hg_, R, false);
  enc_qmvr(e, L.mu, xrn_, hu_, R, false);
  uint32_t n = uint32_t(R) * L.mg.N;
  e.use(pipe2("swiglu_f32"));
  e.buf(0, hg_); e.buf(1, hu_); e.buf(2, mh_); e.val(3, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  enc_qmvr(e, L.md, mh_, x, R, true);
}

// ---- MTP head ----

void Qwen35::encode_mtp(Encoder& e, Seq& s, int mpos0, int R, int tok0, const Buffer& h, int h_row0, int draft_out,
                        bool h_trunk) {
  if (!mtp_loaded_) fatal("no MTP head");
  if (R > kMaxVerify) fatal("encode_mtp: R=%d", R);
  const int D = cfg_.hidden;
  kp::Embed ep{uint32_t(D), uint32_t(embed_.gs), uint32_t(tok0)};
  e.use(pipe2("embed_q4"));
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, s.tok); e.buf(4, mtp_e_);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, R, 1, 64);
  kp::RowsNorm np{uint32_t(D), cfg_.eps};
  const bool post = h_trunk && mtp_post_norm_hidden_;
  e.use(pipe2(post ? "mtp_concat_norm_post" : "mtp_concat_norm"));
  e.buf(0, mtp_e_); e.buf(1, h, size_t(h_row0) * D * 4); e.buf(2, mtp_pre_e_->buf); e.buf(3, mtp_pre_h_->buf);
  e.buf(4, mtp_u_);
  e.val(5, np);
  if (post) e.buf(6, final_norm_->buf);
  e.dispatch(R, 1, 1, 256);
  enc_qmvr(e, mtp_fc_, mtp_u_, mtp_x_, R, false);
  enc_attn_rows(e, mtp_, s.mtp_kc, s.mtp_vc, s.cap, mpos0, R, mtp_x_);
  if (cfg_.moe) enc_moe_rows(e, mtp_, R, mtp_x_, true);
  else enc_mlp_rows(e, mtp_, R, mtp_x_);
  enc_rows_norm(e, mtp_x_, mtp_norm_, mtp_post_, R, D);
  if (draft_out < 0) return;
  // greedy draft over the reduced vocabulary from the last row
  const Buffer last = mtp_post_.view(size_t(R - 1) * D * 4, size_t(D) * 4);
  const bool full_vocab = full_draft_;
  uint32_t n0 = draft_head_[0].N, n1 = draft_head_[1].N;
  if (full_vocab) {
    enc_qmv(e, lm_head_, last, dlog_, nullptr, nullptr);
    n0 = lm_head_.N;
    n1 = 0;
  } else {
    enc_qmv(e, draft_head_[0], last, dlog_, nullptr, nullptr);
    enc_qmv(e, draft_head_[1], last, dlog_.view(size_t(n0) * 4, size_t(n1) * 4), nullptr, nullptr);
  }
  const uint32_t V = n0 + n1, chunk = 2048, ntg = (V + chunk - 1) / chunk;
  kp::TopkStage ts{V, 1, chunk};
  e.use(pipe2("topk_stage1_rows"));
  e.buf(0, dlog_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  kp::SampleRows sr{ntg, 1, 0.0f, 1.0f, 0, 0, uint32_t(draft_out), full_vocab ? 0u : n0, draft_special0_};
  e.use(pipe2("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, s.tok);
  e.val(3, sr);
  e.dispatch(1, 1, 1, 1024);
}

void Qwen35::encode_mtp_prefill(Encoder& e, Seq& s, int pos0, int R, int h_row0) {
  if (!mtp_loaded_ || R <= 0) return;
  const int D = cfg_.hidden;
  if (R <= kMaxMtpRows) {
    encode_mtp(e, s, pos0, R, pos0 + 1, x_, h_row0, -1);
    return;
  }
  kp::Embed ep{uint32_t(D), uint32_t(embed_.gs), uint32_t(pos0 + 1)};
  e.use(pipe2("embed_q4"));
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, s.tok); e.buf(4, mtp_e_);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, R, 1, 64);
  kp::RowsNorm np{uint32_t(D), cfg_.eps};
  e.use(pipe2(mtp_post_norm_hidden_ ? "mtp_concat_norm_post" : "mtp_concat_norm"));
  e.buf(0, mtp_e_); e.buf(1, x_, size_t(h_row0) * D * 4); e.buf(2, mtp_pre_e_->buf); e.buf(3, mtp_pre_h_->buf);
  e.buf(4, mtp_u_);
  e.val(5, np);
  if (mtp_post_norm_hidden_) e.buf(6, final_norm_->buf);
  e.dispatch(R, 1, 1, 256);
  enc_rows_half(e, mtp_u_, nullptr, hh_, R, 2 * D);
  enc_qmm(e, mtp_fc_, hh_, mtp_x_, R, false);
  // attention block (batched)
  enc_rows_half(e, mtp_x_, mtp_.in_norm, xh_, R, D);
  enc_proj(e, {{&mtp_.q, qg_, false}, {&mtp_.k, kr_, false}, {&mtp_.v, vr_, false}}, xh_, R);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                  uint32_t(cfg_.rot_dims), uint32_t(pos0), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(pipe2("attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, mtp_.q_norm->buf); e.buf(4, mtp_.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.mtp_kc); e.buf(7, s.mtp_vc);
  e.val(8, ap);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, R, 1, cfg_.head_dim);
  enc_attn_prefill(e, s.mtp_kc, s.mtp_vc, R, pos0, s.cap);
  enc_rows_half(e, attn_out_, nullptr, hh_, R, cfg_.hq * cfg_.head_dim);
  enc_qmm(e, mtp_.o, hh_, mtp_x_, R, true);
  if (cfg_.moe) enc_moe_prefill(e, mtp_, R, mtp_x_);
  else enc_mlp_prefill(e, mtp_, R, mtp_x_);
}

// ---- trunk verify ----

uint32_t Qwen35::verify_candidates(const SamplingParams& sp) const {
  const uint32_t V = uint32_t(cfg_.vocab), ntg = (V + 2047) / 2048;
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  return ntg * K;
}

void Qwen35::encode_verify(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step, bool v8) {
  if (R < 1 || R > kMaxVerify) fatal("encode_verify: R=%d", R);
  // NI_V8=1: the 8-row path for any R >= 2 (A/B); by default only R >= 5 takes it, so MTP verifies
  // (R <= 4) run exactly the kernels they always did.
  static const bool v8_all = env_on("NI_V8");
  if (R > 4 || v8 || (v8_all && R > 1)) {
    encode_verify8(e, s, pos, R, sp, step);
    return;
  }
  if (pos + R + 1 >= s.cap) fatal("context full (%d)", s.cap);
  const int D = cfg_.hidden;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  kp::Embed ep{uint32_t(D), uint32_t(embed_.gs), uint32_t(pos)};
  e.use(pipe2("embed_q4"));
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, s.tok); e.buf(4, x_);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, R, 1, 64);
  kp::GdnVerify gv{uint32_t(R), uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps};
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[i];
    if (L.linear) {
      enc_qmvr_multi(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, x_, L.in_norm, R, &xrn_);
      e.use(pipe2("gdn_prep_verify"));
      e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
      e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
      e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
      e.buf(12, s.pend_qkv[i]);
      e.val(13, gv);
      e.dispatch(C / 128, 1, 1, 128);
      e.use(pipe2("gdn_step_verify"));
      e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
      e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
      e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
      e.val(12, gv);
      e.dispatch(cfg_.lin_hv, 1, 1, 1024);
      enc_qmvr(e, L.out, gout_, x_, R, true, uint32_t(VD), uint32_t(D));
    } else {
      enc_attn_rows(e, L, s.kc[i], s.vc[i], s.cap, pos, R, x_);
    }
    if (cfg_.moe) enc_moe_rows(e, L, R, x_, false);
    else enc_mlp_rows(e, L, R, x_);
    if (cap_) enc_capture(e, i, pos, R);
  }
  s.n_pend = 0;  // folded by this verify; commit_rows() sets the new count
  enc_rows_norm(e, x_, final_norm_, xrn_, R, D);
  enc_qmvr(e, lm_head_, xrn_, logits_, R, false);
  const uint32_t V = uint32_t(cfg_.vocab), chunk = 2048, ntg = (V + chunk - 1) / chunk;
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  kp::TopkStage ts{V, K, chunk};
  e.use(pipe2("topk_stage1_rows"));
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, R, 1, 256);
  kp::SampleRows sr{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, 0, 0, 0};
  e.use(pipe2("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, samp_);
  e.val(3, sr);
  e.dispatch(R, 1, 1, 1024);
}

void Qwen35::encode_flush(Encoder& e, Seq& s) {
  if (s.n_pend == 0) return;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  kp::GdnVerify gv{0, uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps};
  for (int i = 0; i < cfg_.layers; ++i) {
    if (!cfg_.is_linear[i]) continue;
    const Layer& L = layers_[i];
    e.use(pipe2("gdn_prep_verify"));
    e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
    e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
    e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
    e.buf(12, s.pend_qkv[i]);
    e.val(13, gv);
    e.dispatch(C / 128, 1, 1, 128);
    e.use(pipe2("gdn_step_verify"));
    e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
    e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
    e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
    e.val(12, gv);
    e.dispatch(cfg_.lin_hv, 1, 1, 1024);
  }
  s.n_pend = 0;
}

} // namespace ni
