// Speculative decoding for Qwen3.5/3.6/3.8: the model's own MTP head drafts tokens, the trunk
// verifies the pending token plus drafts in one small-batch forward, and Gated DeltaNet layers roll
// back through a tape (rows are only folded into the recurrent state once accepted).
#include "loader/safetensors.h"
#include "model/dflash2.h"
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

static int env_int(const char* n, int def) {
  const char* v = getenv(n);
  return v && *v ? atoi(v) : def;
}

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

// bf16 [N, K] -> MLX affine 8-bit (near lossless; halves the bytes the head reads) or 4-bit (NI_MTP_BITS=4: a
// quarter), group 64, min/max per group.
QLinear Qwen35::quantize8(const Tensor& w, int bits) {
  if (w.dtype != DType::BF16 || w.shape.size() != 2) fatal("quantize8: expected a 2-D bf16 tensor");
  if (bits != 8 && bits != 4) fatal("quantize8: %d bits", bits);
  const uint32_t N = uint32_t(w.shape[0]), K = uint32_t(w.shape[1]);
  if (K % 64) fatal("quantize8: K=%u not a multiple of 64", K);
  Gpu& g = Gpu::get();
  const uint32_t qmax = (1u << bits) - 1;
  Buffer wq = g.alloc(size_t(N) * K * bits / 8, bits == 8 ? "mtp_q8" : "mtp_q4", false);
  Buffer sc = g.alloc(size_t(N) * K / 64 * 2, "mtp_q8s", false);
  Buffer bi = g.alloc(size_t(N) * K / 64 * 2, "mtp_q8b", false);
  const uint16_t* src = w.buf.as<uint16_t>();
  uint8_t* q = wq.as<uint8_t>();  // value k of a row at bits [bits*k, bits*k + bits) of the little-endian stream
  if (bits == 4) memset(q, 0, wq.size);
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
      float scale = (hi - lo) / float(qmax);
      if (scale == 0.0f) scale = 1e-8f;
      uint16_t sb = f32_to_bf16(scale), bb = f32_to_bf16(lo);
      float sr = bf16_to_f32(sb), br = bf16_to_f32(bb);
      s[size_t(n) * (K / 64) + g0 / 64] = sb;
      b[size_t(n) * (K / 64) + g0 / 64] = bb;
      for (uint32_t k = 0; k < 64; ++k) {
        float v = bf16_to_f32(src[size_t(n) * K + g0 + k]);
        long qv = std::lround((v - br) / sr);
        const uint8_t c = uint8_t(std::clamp(qv, 0L, long(qmax)));
        const size_t i = size_t(n) * K + g0 + k;
        if (bits == 8) q[i] = c;
        else q[i / 2] |= uint8_t(c << (4 * (i % 2)));
      }
    }
  owned_bufs_.push_back(wq);
  owned_bufs_.push_back(sc);
  owned_bufs_.push_back(bi);
  Tensor tw{wq, DType::U32, {int64_t(N), int64_t(K * bits / 32)}};
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
  L.bits = bits;
  L.gs = 64;
  return L;
}

QLinear Qwen35::quantize_rows4(const QLinear& src, uint32_t r0, uint32_t n) {
  const uint32_t K = src.K, gpr = K / 64;
  if (K % 64 || n == 0) fatal("quantize_rows4: [%u, %u]", n, K);
  std::vector<uint16_t> tmp(size_t(n) * K);
  std::vector<const uint16_t*> rowp(n);
  for (uint32_t i = 0; i < n; ++i) {
    dfl_row_bf16(src, r0 + i, tmp.data() + size_t(i) * K);
    rowp[i] = tmp.data() + size_t(i) * K;
  }
  Gpu& g = Gpu::get();
  Buffer w = g.alloc(size_t(n) * K / 2, "draft_q4_w", false), sc = g.alloc(size_t(n) * gpr * 2, "draft_q4_s", false),
         bi = g.alloc(size_t(n) * gpr * 2, "draft_q4_b", false);
  dfl_quantize_cpu(rowp, K, 4, false, w.as<uint32_t>(), sc.as<uint16_t>(), bi.as<uint16_t>());
  QLinear L;
  for (const Buffer& b : {w, sc, bi}) owned_bufs_.push_back(b);
  owned_.push_back(Tensor{w, DType::U32, {int64_t(n), int64_t(K / 8)}});
  L.w = &owned_.back();
  owned_.push_back(Tensor{sc, DType::BF16, {int64_t(n), int64_t(gpr)}});
  L.s = &owned_.back();
  owned_.push_back(Tensor{bi, DType::BF16, {int64_t(n), int64_t(gpr)}});
  L.b = &owned_.back();
  L.N = n;
  L.K = K;
  L.bits = 4;
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
  // Kapitsa's sidecar (mtp.safetensors next to the model: MLX conventions), else the head inside the checkpoint's
  // own shards (the official Hugging Face checkpoints keep it there, norms zero-centred)
  bool in_shards = false;
  {
    std::ifstream sidecar(dir + "/mtp.safetensors");
    if (sidecar) {
      w.open_dir(dir, nullptr, [](const std::string& f) { return f == "mtp.safetensors"; });
    } else {
      w.open_dir(dir, [](const std::string& n) { return n.rfind("mtp.", 0) == 0; });
      in_shards = true;
    }
  }
  if (!w.has("mtp.fc.weight")) {
    logi("no MTP head in %s; speculative decoding off", dir.c_str());
    return;
  }
  if (cfg_.moe && !w.has("mtp.layers.0.mlp.experts.0.gate_proj.weight")) {
    logi("MTP head in %s: experts not in the per-expert layout Kapitsa reads (fused Hugging Face experts?); "
         "speculative decoding with the MTP head off", dir.c_str());
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
    if (variant.empty()) {  // a sidecar made by tools/mtp_sidecar.py records it in its own header
      std::ifstream f(dir + "/mtp.safetensors", std::ios::binary);
      uint64_t n = 0;
      if (f.read(reinterpret_cast<char*>(&n), 8) && n < (64u << 20)) {
        std::string h(n, '\0');
        if (f.read(h.data(), std::streamsize(n))) {
          try {
            auto j = nlohmann::json::parse(h);
            if (j.contains("__metadata__")) variant = j["__metadata__"].value("base_hidden_variant", "");
          } catch (...) {
          }
        }
      }
    }
    // a head read from a Hugging Face checkpoint's shards: the official models feed it the final-norm hidden
    if (variant.empty() && in_shards && hf_norms_) variant = "post_norm";
    if (const char* v = getenv("NI_MTP_HIDDEN"); v && *v) variant = std::string(v) + "_norm";  // pre | post
    mtp_post_norm_hidden_ = variant == "post_norm";
  }
  // bf16 linears of the head (Hugging Face checkpoints, tools/mtp_sidecar.py) are quantized here (mtp_head_bits).
  const int mtp_bits = mtp_head_bits(cfg_.moe);
  auto lin = [&](const std::string& n) {  // quantized in the file, or bf16 -> 4-bit here
    QLinear L = w.has(n + ".scales") ? own_qlin(w, n) : quantize8(get(n + ".weight"), mtp_bits);
    L.mtp = true;
    return L;
  };
  // norms: the kernels read bf16 (an f32/f16 file gets an owned bf16 copy)
  int lossy = 0;
  const bool zc = in_shards && hf_norms_;  // zero-centred norms (Hugging Face): 1 + w
  auto norm = [&](const std::string& n) {
    const Tensor& t = get(n);
    if (zc) return bf16_copy(&t, 1.0f);
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
  draft_split_ = std::min<uint32_t>(uint32_t(std::max(1024, env_int("NI_DRAFT_IDS", 32768))), uint32_t(cfg_.vocab));
  // Dense 27B: the full-vocabulary draft raises acceptance (depth 2: 0.73 -> 0.83) for ~1% of the
  // cycle; the 35B-A3B's cycle is short enough that the 286 MB lm_head pass outweighs the gain. So does
  // the 4B's: its lm_head is 15% of the bytes a decode step reads (27B: 5%), and the reduced draft head
  // measured faster at depth 1 (code 119.5 vs 115.8 tok/s, prose 120.9 vs 112.0) though it accepts
  // fewer drafts (0.67 vs 0.76 on code). Full vocabulary for dense models whose lm_head is < 10% of a step.
  double trunk_bytes = 0;  // the linears a decode step reads (any format; bf16 has no scales / biases)
  auto lbytes = [](const QLinear& q) {
    return q ? double(q.w->buf.size) + (q.dense() ? 0.0 : double(q.s->buf.size + q.b->buf.size)) : 0.0;
  };
  for (const Layer& L : layers_)
    for (const QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out, &L.q, &L.k, &L.v, &L.o, &L.mg, &L.mu, &L.md, &L.eg, &L.eu,
                             &L.ed, &L.sg, &L.su, &L.sd, &L.router, &L.shared_gate})
      trunk_bytes += lbytes(*q);
  const double head_bytes = lbytes(lm_head_);
  trunk_bytes += head_bytes;
  full_draft_ = getenv("NI_FULL_DRAFT") && *getenv("NI_FULL_DRAFT") ? env_on("NI_FULL_DRAFT")
                                                                     : !cfg_.moe && head_bytes < 0.1 * trunk_bytes;
  draft_special0_ = uint32_t(cfg_.eos.empty() ? cfg_.vocab : *std::min_element(cfg_.eos.begin(), cfg_.eos.end()));
  auto rows_view = [&](const QLinear& src, uint32_t r0, uint32_t n) {
    QLinear v = src;
    const size_t wrow = src.row_bytes(), srow = src.group_row_bytes();
    owned_.push_back(Tensor{src.w->buf.view(r0 * wrow, n * wrow), src.w->dtype,
                            {int64_t(n), int64_t(src.dense() ? src.K : wrow / 4)}});
    v.w = &owned_.back();
    if (src.dense()) {
      v.s = v.b = v.w;
    } else {
      owned_.push_back(Tensor{src.s->buf.view(r0 * srow, n * srow), src.s->dtype, {int64_t(n), int64_t(src.K / src.gs)}});
      v.s = &owned_.back();
      owned_.push_back(Tensor{src.b->buf.view(r0 * srow, n * srow), src.b->dtype, {int64_t(n), int64_t(src.K / src.gs)}});
      v.b = &owned_.back();
    }
    v.N = n;
    return v;
  };
  draft_head_[0] = rows_view(lm_head_, 0, draft_split_);
  draft_head_[1] = rows_view(lm_head_, draft_special0_, uint32_t(cfg_.vocab) - draft_special0_);
  // An output layer in another format than 4-bit group 64 (bf16, 6- or 8-bit variants) drafts from a 4-bit copy of
  // those rows (NI_DRAFT_Q4=0: from its own): drafting only, the verify reads the output layer itself. bf16 4B: the
  // reduced head reads 45 instead of 168 MB per draft pass; depth 3, code / prose, 256 tokens: 64.0 -> 65.5 / 52.0 ->
  // 56.2 tok/s.
  draft_q4_ = !full_draft_ && !lm_head_.q4() && env_int("NI_DRAFT_Q4", 1) != 0;
  if (draft_q4_) {
    draft_head_[0] = quantize_rows4(lm_head_, 0, draft_split_);
    draft_head_[1] = quantize_rows4(lm_head_, draft_special0_, uint32_t(cfg_.vocab) - draft_special0_);
  }
  mtp_loaded_ = true;
  logi("MTP head loaded in %.0f ms (%s, trunk hidden %s, drafts over %s%s)", now_ms() - t0, cfg_.moe ? "MoE" : "dense",
       mtp_post_norm_hidden_ ? "post-norm" : "pre-norm",
       full_draft_ ? "the full vocabulary" : ("ids < " + std::to_string(draft_split_) + " + specials").c_str(),
       draft_q4_ ? ", a 4-bit copy of the output layer's rows" : "");
}

// Bits of the MTP head's bf16 linears as loaded. Dense models: 4 (NI_MTP_BITS=8: 8 bits, the earlier default). The
// drafts may differ, the verified output does not. 4B, 512-token greedy runs, depth 3: code 148.3 -> 152.5 tok/s,
// prose 142.1 -> 147.0 (2.37 -> 2.31 / 2.26 -> 2.22 tokens per cycle for a 0.26 ms cheaper draft pass); depth 1
// 124.8 -> 128.3 / 124.8 -> 127.0; bf16 trunk, depth 4, 256 tokens: 68.1 -> 69.4 / 55.8 -> 57.4. MoE models: 8, as
// before (the MoE row kernels pair group-32 experts, as in MTPLX's 35B sidecar, with an 8-bit shared expert only).
int Qwen35::mtp_head_bits(bool moe) {
  static const int env = env_int("NI_MTP_BITS", 4) == 8 ? 8 : 4;
  return moe ? 8 : env;
}

// ---- the draft head's per-conversation rows ----

struct Qwen35::DraftVocab {
  std::vector<uint8_t> in;    // per vocabulary id: already gathered
  std::vector<uint32_t> ids;  // gathered ids in order of appearance (row i of the buffers)
  int scanned = 0;            // tokens [0, scanned) of the Seq have been looked at
  uint32_t cap = 0;
  Buffer w, sc, bi, table;    // gathered output-layer rows (the output layer's format) and their ids
  Tensor tw, ts, tb;
  QLinear L;                  // the gathered rows as a linear of N = ids.size() rows
};

// NI_DRAFT_CTX (read once; default 1) and NI_DRAFT_CTX_ROWS (row capacity per conversation, default 8192: 11.8 MB
// for the 4B's 4-bit output layer)
static bool draft_ctx_on() {
  static const bool v = env_int("NI_DRAFT_CTX", 1) != 0;
  return v;
}

// Scans the Seq's tokens [scanned, upto) (all final: written by the host or by command buffers that completed) and
// gathers the output-layer rows of ids outside the static draft rows. Rows already gathered are never rewritten while
// the set grows, so a command buffer still reading the first N rows is unaffected by appending row N. A prefill from
// position 0 (a new conversation in this slot) starts the set over; one from p > 0 rescans from p.
Qwen35::DraftVocab* Qwen35::draft_vocab_update(Seq& s, int upto) {
  if (!draft_ctx_on() || full_draft_ || !mtp_loaded_) return nullptr;
  if (!s.draft_vocab) {
    auto v = std::make_shared<DraftVocab>();
    const uint32_t outside = draft_special0_ - std::min(draft_split_, draft_special0_);
    v->cap = std::min<uint32_t>(uint32_t(std::clamp(env_int("NI_DRAFT_CTX_ROWS", 8192), 64, 65536)), outside);
    if (v->cap == 0) return nullptr;
    Gpu& g = Gpu::get();
    const QLinear& H = draft_head_[0];  // the draft head's format (the output layer's, or its 4-bit copy)
    v->w = g.alloc(size_t(v->cap) * H.row_bytes(), "draft_ctx_w", false);
    if (!H.dense()) {
      v->sc = g.alloc(size_t(v->cap) * H.group_row_bytes(), "draft_ctx_s", false);
      v->bi = g.alloc(size_t(v->cap) * H.group_row_bytes(), "draft_ctx_b", false);
    }
    v->table = g.alloc(size_t(v->cap) * 4, "draft_ctx_ids", false);
    v->in.assign(size_t(cfg_.vocab), 0);
    v->tw = Tensor{v->w, H.w->dtype, {int64_t(v->cap), int64_t(H.dense() ? H.K : H.row_bytes() / 4)}};
    v->L = H;
    v->L.w = &v->tw;
    if (H.dense()) {
      v->L.s = v->L.b = v->L.w;
    } else {
      v->ts = Tensor{v->sc, H.s->dtype, {int64_t(v->cap), int64_t(H.K / H.gs)}};
      v->tb = Tensor{v->bi, H.b->dtype, {int64_t(v->cap), int64_t(H.K / H.gs)}};
      v->L.s = &v->ts;
      v->L.b = &v->tb;
    }
    v->L.N = 0;
    s.draft_vocab = v;
  }
  DraftVocab& v = *static_cast<DraftVocab*>(s.draft_vocab.get());
  if (upto < v.scanned) v.scanned = 0;  // rewound below what was scanned: look again
  const int* t = s.tokens();
  for (int i = v.scanned; i < upto; ++i) {
    const int id = t[i];
    if (id < int(draft_split_) || id >= int(draft_special0_) || v.in[size_t(id)]) continue;
    if (!draft_vocab_add(v, id)) break;
  }
  v.scanned = std::max(v.scanned, upto);
  v.L.N = uint32_t(v.ids.size());
  return &v;
}

bool Qwen35::draft_vocab_add(DraftVocab& v, int id) {
  if (v.ids.size() >= v.cap) return false;
  const QLinear& H = lm_head_;
  const size_t wrow = H.row_bytes(), srow = H.group_row_bytes();
  const size_t r = v.ids.size();
  if (draft_q4_) {  // the draft head's 4-bit copy of this row
    std::vector<uint16_t> row(H.K);
    dfl_row_bf16(H, uint32_t(id), row.data());
    const uint32_t gpr = H.K / 64;
    dfl_quantize_cpu({row.data()}, H.K, 4, false, v.w.as<uint32_t>() + r * (H.K / 8), v.sc.as<uint16_t>() + r * gpr,
                     v.bi.as<uint16_t>() + r * gpr);
  } else {
    memcpy(v.w.as<uint8_t>() + r * wrow, H.w->buf.as<uint8_t>() + size_t(id) * wrow, wrow);
    if (!H.dense()) {
      memcpy(v.sc.as<uint8_t>() + r * srow, H.s->buf.as<uint8_t>() + size_t(id) * srow, srow);
      memcpy(v.bi.as<uint8_t>() + r * srow, H.b->buf.as<uint8_t>() + size_t(id) * srow, srow);
    }
  }
  v.table.as<uint32_t>()[r] = uint32_t(id);
  v.in[size_t(id)] = 1;
  v.ids.push_back(uint32_t(id));
  return true;
}

std::vector<uint32_t> Qwen35::draft_vocab_ids(const Seq& s, int* scanned) const {
  *scanned = 0;
  if (!s.draft_vocab) return {};
  const DraftVocab& v = *static_cast<const DraftVocab*>(s.draft_vocab.get());
  *scanned = v.scanned;
  return v.ids;
}

void Qwen35::draft_vocab_restore(Seq& s, const std::vector<uint32_t>& ids, int scanned) {
  draft_vocab_rewind(s, 0);  // (the set starts over)
  if (ids.empty() && !s.draft_vocab) return;
  DraftVocab* v = draft_vocab_update(s, 0);
  if (!v) return;
  for (uint32_t id : ids)
    if (id < uint32_t(cfg_.vocab) && !v->in[id] && !draft_vocab_add(*v, int(id))) break;
  v->scanned = std::max(0, scanned);
  v->L.N = uint32_t(v->ids.size());
}

// A prefill at pos0 may replace the tokens from pos0 on: the next scan starts there (pos0 == 0: a new conversation in
// this slot, the gathered set starts over). Nothing of this Seq that reads the rows is in flight during a prefill.
void Qwen35::draft_vocab_rewind(Seq& s, int pos0) {
  if (!s.draft_vocab) return;
  DraftVocab& v = *static_cast<DraftVocab*>(s.draft_vocab.get());
  if (pos0 <= 0) {
    for (uint32_t id : v.ids) v.in[id] = 0;
    v.ids.clear();
    v.L.N = 0;
    v.scanned = 0;
  } else {
    v.scanned = std::min(v.scanned, pos0);
  }
}

// ---- small-batch encoders ----

// Rows from which verifies and the MTP passes' matvecs (enc_qmvr, enc_qmvr_multi, enc_mlp_rows) take the 8-row MMA
// GEMMs: 3 on dense models, 5 on MoE models. NI_V8=1: 2, NI_V8=0: 5. mtp: for an MTP pass (NI_MTP_V8=0 keeps those on
// the small-batch kernels below 5 rows and one-row passes on qmvr, as before).
int Qwen35::v8_rows_min(bool mtp) const {
  static const int v8_env = getenv("NI_V8") && *getenv("NI_V8") ? atoi(getenv("NI_V8")) : -1;
  static const bool mtp_v8 = env_int("NI_MTP_V8", 1) != 0;
  if (mtp && !mtp_v8) return 5;
  return v8_env == 1 ? 2 : v8_env == 0 || cfg_.moe ? 5 : 3;
}

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
  if (R < 1 || R > kMaxVerify) fatal("qmvr: R=%d", R);
  // one row (a chained MTP draft pass): the decode matvecs (qmv2 / qmv3, in-place add)
  static const bool mtp_v8 = env_int("NI_MTP_V8", 1) != 0;
  if (mtp_v8 && R == 1 && (L.q4() || L.q8()) && (!x_ld || x_ld == L.K) && (!y_ld || y_ld == L.N)) {
    enc_qmv(e, L, x, y, add ? &y : nullptr, nullptr);
    return;
  }
  // 5-8 rows (deep MTP drafts; 3-4 on dense models: MTP passes over a cycle's history rows), and formats without
  // qmvr kernels (bf16, 3/5/6-bit, group 32): the 8-row MMA GEMM
  if (R >= v8_rows_min(true) || !(L.q4() || L.q8())) {
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
  // fp32 version is ALU-bound at 1.5x a single row; this one measured 1.17-1.32x). Apple10 GPUs (M5) take the
  // shifted-nibble build: their fast-math compiler turns qmvht_t3_r4 into half sums that overflow (NaN rows).
  static const bool half3_off = env_on("NI_NO_QMVH");
  static const char* half3 = Gpu::get().apple10() ? "qmvhts_t3_r4" : "qmvht_t3_r4";
  if (R == 3 && !half3_off && !no_qmvh_ && L.bits == 4) {
    e.use(pipe2(half3));
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
  if (enc_qmvf(e, segs, x, R, norm, xn_out, add)) return;
  const uint32_t K = segs[0].L->K;
  if (R >= v8_rows_min(true) && R <= kMaxVerify && segs.size() <= 4 && K % 64 == 0) {
    // one X8 table (RMSNorm folded, the normalized rows to xn_out) for every matrix, one GEMM dispatch
    v8_alloc();
    enc_x8(e, x, R, int(K), norm, false, 0, xn_out);
    std::vector<Seg8> s8;
    for (size_t i = 0; i < segs.size(); ++i) s8.push_back({segs[i].L, segs[i].y, 0, i < add.size() && add[i]});
    enc_qmm8(e, s8, R, v8t_, v8s_);
    return;
  }
  bool special = true;  // the fused multi-row kernels read 4/8-bit group-64 weights
  for (auto& sg : segs) special = special && (sg.L->q4() || sg.L->q8());
  static const bool rowcache = env_on("NI_RC");
  if (special && rowcache && R > 1 && segs.size() <= 4 &&
      (add.empty() || std::none_of(add.begin(), add.end(), [](bool b) { return b; }))) {
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
  if (off || !special || (R != 2 && R != 3) || K > 2048 || segs.size() > 4) {
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
void Qwen35::enc_attn_rows(Encoder& e, const Seq& s, const Layer& L, const Buffer& kc, const Buffer& vc, int cap,
                           int pos, int R, const Buffer& x) {
  enc_qmvr_multi(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, x, L.in_norm, R, &xrn_);
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                  uint32_t(cfg_.rot_dims), uint32_t(pos), uint32_t(cap), cfg_.eps, cfg_.theta};
  e.use(pipe2(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, kc); e.buf(7, vc);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, R, 1, cfg_.head_dim);
  const uint32_t Lmax = uint32_t(pos + R);
  // split-K as in Qwen35::enc_attn: one chunk per GPU core and kv head
  const uint32_t cores = uint32_t(std::max(1, Gpu::get().core_count()));
  uint32_t chunk = std::max<uint32_t>(64, (Lmax + cores - 1) / cores);
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
void Qwen35::enc_moe_rows(Encoder& e, const Layer& L, int R, const Buffer& x, bool mtp_layer, const Buffer* in) {
  const int D = cfg_.hidden, k = cfg_.top_k;
  if (in && in->mtl != xrn_.mtl) fatal("enc_moe_rows: the input rows must be xrn_");
  // IQ experts (GGUF imports) at one row: their bf16 router and shared gate on the decode matvec with the norm folded
  // (qmvg_multi_f16, 14 us per layer on the M1 Pro) instead of a row norm and two X8 tables and 8-row GEMMs (54 us)
  if (L.eg.iq && R == 1 && !in) enc_qmv_multi(e, {{&L.router, rlog_}, {&L.shared_gate, slog_}}, x, L.post_norm, &xrn_);
  else
    enc_qmvr_multi(e, {{&L.router, rlog_}, {&L.shared_gate, slog_}}, in ? *in : x, in ? nullptr : L.post_norm, R, &xrn_);
  kp::Topk tp{uint32_t(cfg_.experts), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  // E = 256, k <= 8 (Qwen3.5/3.6): the unrolled form; any other E <= 512, k <= 16: one simdgroup per row
  e.use(pipe2(cfg_.experts == 256 && k <= 8 ? "moe_topk256_rows" : "moe_topk_rows"));
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  e.val(4, tp);
  e.dispatch(R, 1, 1, 32);
  if (L.eg.iq) {  // llama.cpp's IQ blocks (qwen35_iq.cpp)
    enc_moe_iq_rows(e, L, R, x);
    return;
  }
  kp::Moe mp{uint32_t(D), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(k), cfg_.eps};
  // opt-in: measured slower than per-row expert reads on M4 Pro
  static const bool no_dedupe = !env_on("NI_DEDUPE");
  static const bool rc_moe = env_on("NI_RC");
  const bool all4 = L.eg.q4() && L.eu.q4() && L.ed.q4() && L.sg.q4() && L.su.q4() && L.sd.q4();
  if (!mtp_layer && rc_moe && R > 1 && all4 && D <= 4096) {
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
  if (!mtp_layer && !no_dedupe && (R == 2 || R == 3) && all4 && D <= 2048) {
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
  const std::string gu_k = moe_rows_kernel(true, L.eg, L.sg), dn_k = moe_rows_kernel(false, L.ed, L.sd, moe_wide());
  if (gu_k.empty() || dn_k.empty())
    fatal("MoE rows: no kernels for %s experts with a %s shared expert%s", format_name(L.eg.bits, L.eg.gs).c_str(),
          format_name(L.sg.bits, L.sg.gs).c_str(), mtp_layer ? " (MTP head)" : "");
  e.use(pipe2(gu_k));
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
  e.use(pipe2(dn_k));
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, ridx_); e.buf(8, rwts_); e.buf(9, x);
  e.val(10, dp);
  e.dispatch((D + rows * nsg - 1) / (rows * nsg), R, 1, 32 * nsg);
}

// Dense SwiGLU MLP for R rows: x += down(silu(gate(n)) * up(n)), n = norm(x).
void Qwen35::enc_mlp_rows(Encoder& e, const Layer& L, int R, const Buffer& x) {
  const int D = cfg_.hidden;
  if (enc_qmvf(e, {{&L.mg, mh_}, {&L.mu, mh_}}, x, R, L.post_norm, nullptr, {}, 0, 0, true)) {
    enc_qmvr(e, L.md, mh_, x, R, true);
    return;
  }
  if (R >= v8_rows_min(true) && R <= kMaxVerify && L.mg.N % 256 == 0 && L.mg.N <= uint32_t(std::max(cfg_.dense_I, 0)) &&
      L.mg.same_format(L.mu)) {
    v8_alloc();
    enc_x8(e, x, R, D, L.post_norm);
    enc_gu8(e, L.mg, L.mu, v8t_, v8s_, v8h_, v8hs_);
    enc_qmm8(e, {{&L.md, x, uint32_t(D), true}}, R, v8h_, v8hs_);
    return;
  }
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
  enc_embed(e, s.tok, tok0, R, mtp_e_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, tok0, R, mtp_e_, D);  // image rows (prompt history)
  kp::RowsNorm np{uint32_t(D), cfg_.eps};
  const bool post = h_trunk && mtp_post_norm_hidden_;
  e.use(pipe2(post ? "mtp_concat_norm_post" : "mtp_concat_norm"));
  e.buf(0, mtp_e_); e.buf(1, h, size_t(h_row0) * D * 4); e.buf(2, mtp_pre_e_->buf); e.buf(3, mtp_pre_h_->buf);
  e.buf(4, mtp_u_);
  e.val(5, np);
  if (post) e.buf(6, final_norm_->buf);
  e.dispatch(R, 1, 1, 256);
  enc_qmvr(e, mtp_fc_, mtp_u_, mtp_x_, R, false);
  enc_attn_rows(e, s, mtp_, s.mtp_kc, s.mtp_vc, s.cap, mpos0, R, mtp_x_);
  if (cfg_.moe) enc_moe_rows(e, mtp_, R, mtp_x_, true);
  else enc_mlp_rows(e, mtp_, R, mtp_x_);
  enc_rows_norm(e, mtp_x_, mtp_norm_, mtp_post_, R, D);
  if (draft_out < 0) return;
  // greedy draft over the reduced vocabulary from the last row
  const Buffer last = mtp_post_.view(size_t(R - 1) * D * 4, size_t(D) * 4);
  const bool full_vocab = full_draft_;
  uint32_t n0 = draft_head_[0].N, n1 = draft_head_[1].N, n2 = 0;
  DraftVocab* dv = nullptr;
  if (full_vocab) {
    enc_qmv(e, lm_head_, last, dlog_, nullptr, nullptr);
    n0 = lm_head_.N;
    n1 = 0;
  } else {
    // the conversation's own ids beyond the static rows: gathered on the first pass of a cycle (its tokens are
    // final); the chained passes of deeper drafts reuse that set
    dv = h_trunk ? draft_vocab_update(s, tok0 + R) : static_cast<DraftVocab*>(s.draft_vocab.get());
    n2 = dv ? dv->L.N : 0;
    std::vector<Seg> segs = {{&draft_head_[0], dlog_}, {&draft_head_[1], dlog_.view(size_t(n0) * 4, size_t(n1) * 4)}};
    if (n2) segs.push_back({&dv->L, dlog_.view(size_t(n0 + n1) * 4, size_t(n2) * 4)});
    if (!enc_qmvf(e, segs, last, 1, nullptr))
      for (const Seg& sg : segs) enc_qmv(e, *sg.L, last, sg.y, nullptr, nullptr);
  }
  const uint32_t V = n0 + n1 + n2, chunk = 2048, ntg = (V + chunk - 1) / chunk;
  kp::TopkStage ts{V, 1, chunk};
  e.use(pipe2("topk_stage1_rows"));
  e.buf(0, dlog_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  kp::SampleRows sr{ntg, 1, 0.0f, 1.0f, 0, 0, uint32_t(draft_out), full_vocab ? 0u : n0, draft_special0_,
                    n2 ? n0 + n1 : 0u};
  e.use(pipe2("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, s.tok);
  e.val(3, sr);
  e.buf(4, n2 ? dv->table : cand_i_);
  e.dispatch(1, 1, 1, 1024);
}

void Qwen35::encode_mtp_prefill(Encoder& e, Seq& s, int pos0, int R, int h_row0) {
  if (!mtp_loaded_ || R <= 0) return;
  const int D = cfg_.hidden;
  if (R <= 4) {  // a few history rows: the small-batch kernels; more: the prefill GEMMs below
    encode_mtp(e, s, pos0, R, pos0 + 1, x_, h_row0, -1);
    return;
  }
  enc_embed(e, s.tok, pos0 + 1, R, mtp_e_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0 + 1, R, mtp_e_, D);  // image rows
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
  e.use(pipe2(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, mtp_.q_norm->buf); e.buf(4, mtp_.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.mtp_kc); e.buf(7, s.mtp_vc);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
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
  // a tree verify may be wider (16 / 32 rows: verify_wide_ok, set_tree_rows)
  if (R < 1 || R > (v8_tree_ ? kMaxTreeRows : kMaxVerify)) fatal("encode_verify: R=%d", R);
  // The 8-row MMA path: R >= 5; dense models also R = 3-4 (it reads the weights once for up to 8 rows at ~1.1-1.25x a
  // decode step, where the FMA kernels cost 1.2-1.9x from 3 rows on: 4B MTP depth 2 117.7 -> 138.3 tok/s). MoE
  // models keep the small-batch kernels below 5 rows (their verify unions the rows' experts). NI_V8=1: any R >= 2;
  // NI_V8=0: R >= 5 only (the earlier rule).
  if (R >= v8_rows_min() || v8) {
    encode_verify8(e, s, pos, R, sp, step);
    return;
  }
  if (pos + R + 1 >= s.cap) fatal("context full (%d)", s.cap);
  if (v8_tree_) fatal("encode_verify: a verify tree needs the 8-row path");
  if (s.pend_tree) encode_flush(e, s);  // a tree cycle's accepted rows (DFlash2 tree drafting) fold on their own
  const int D = cfg_.hidden;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  enc_embed(e, s.tok, pos, R, x_);
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
      enc_attn_rows(e, s, L, s.kc[i], s.vc[i], s.cap, pos, R, x_);
    }
    if (cfg_.moe) enc_moe_rows(e, L, R, x_, false);
    else enc_mlp_rows(e, L, R, x_);
    if (cap_) enc_capture(e, i, pos, R);
  }
  s.n_pend = 0;  // folded by this verify; commit_rows() sets the new count
  if (!enc_qmvf(e, {{&lm_head_, logits_}}, x_, R, final_norm_)) {
    enc_rows_norm(e, x_, final_norm_, xrn_, R, D);
    enc_qmvr(e, lm_head_, xrn_, logits_, R, false);
  }
  const uint32_t V = uint32_t(cfg_.vocab), chunk = 2048, ntg = (V + chunk - 1) / chunk;
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  enc_logit_mask(e, s, logits_, R, int(V), int(V));  // structured output (nothing without masks)
  kp::TopkStage ts{V, K, chunk};
  e.use(pipe2("topk_stage1_rows"));
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, R, 1, 256);
  kp::SampleRows sr{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, 0, 0, 0};
  e.use(pipe2("sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, samp_);
  e.val(3, sr);
  e.buf(4, cand_i_);  // (no remap table)
  e.dispatch(R, 1, 1, 1024);
}

void Qwen35::encode_flush(Encoder& e, Seq& s) {
  if (s.n_pend == 0) {
    s.pend_tree = false;
    return;
  }
  // a tree cycle's accepted rows (DFlash2 tree drafting): tape rows pend_path[0 .. n_pend)
  const bool pt = s.pend_tree;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  kp::GdnVerify gv{0, uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps, 0u, pt ? 1u : 0u};
  for (int i = lo_; i < hi_; ++i) {
    if (!cfg_.is_linear[i]) continue;
    const Layer& L = layers_[i];
    e.use(pipe2(pt ? "gdn_prep_verify_tree" : "gdn_prep_verify"));
    e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
    e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
    e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
    e.buf(12, s.pend_qkv[i]);
    e.val(13, gv);
    if (pt) {
      e.buf(14, s.pend_path);
      e.buf(15, s.pend_path);
    }
    e.dispatch(C / 128, 1, 1, 128);
    e.use(pipe2(pt ? "gdn_step_verify_pidx" : "gdn_step_verify"));
    e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
    e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
    e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
    e.val(12, gv);
    if (pt) e.buf(13, s.pend_path);
    e.dispatch(cfg_.lin_hv, 1, 1, 1024);
  }
  s.n_pend = 0;
  s.pend_tree = false;
}

} // namespace ni
