#include "model/inkling2.h"
#include "model/inkling2_gguf.h"
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "kernels/inkling2/gguf_blocks.h"
#include "util/json_file.h"
#include "util/log.h"

#include <algorithm>
#include <cstring>

namespace ni {
void Inkling2::load_gguf() {
  const double start = now_ms();
  const auto config = json_file(path_ + "/config.json");
  gguf::File f(path_ + "/" + config->at("gguf_file").get<std::string>());
  if (f.str("general.architecture") != "inkling2_loop") throw gguf::Error("Not an Inkling2 GGUF");
  auto alloc = [&](size_t bytes) {
    Buffer b = Gpu::get().alloc(std::max<size_t>(16, bytes), "ink2_gguf");
    owned_bufs_.push_back(b);
    return b;
  };
  auto tensor = [&](Buffer b, DType dt, std::vector<int64_t> shape) -> const Tensor* {
    gguf_tensors_.push_back(Tensor{b, dt, std::move(shape)});
    return &gguf_tensors_.back();
  };
  auto require = [&](const std::string& name, std::initializer_list<uint64_t> shape) -> const gguf::Tensor& {
    const auto* t = f.find(name);
    if (!t || t->ne != std::vector<uint64_t>(shape)) throw gguf::Error("Inkling2: missing or unexpected shape: " + name);
    return *t;
  };
  auto small = [&](const std::string& name, std::initializer_list<uint64_t> shape) -> const Tensor* {
    const auto& t = require(name, shape);
    if (t.type != gguf::kF32 && t.type != gguf::kBF16) throw gguf::Error("Inkling2: expected F32/BF16: " + name);
    Buffer b = alloc(t.bytes);
    f.read(t, 0, b.ptr(), t.bytes);
    return tensor(b, t.type == gguf::kF32 ? DType::F32 : DType::BF16, std::vector<int64_t>(t.ne.rbegin(), t.ne.rend()));
  };
  auto scalar = [](const Tensor* t, size_t i) {
    return t->dtype == DType::F32 ? t->buf.as<float>()[i] : gguf::from_bf16(t->buf.as<uint16_t>()[i]);
  };
  auto blocks = [&](const gguf::Tensor& t) {
    if (!ink2_gguf::block_bytes(t.type) || t.ne[0] % ink2_gguf::block_values(t.type))
      throw gguf::Error("Inkling2: unsupported block type or row: " + t.name);
    Ink2Blocks b{alloc(t.bytes), t.type, uint32_t(t.row_bytes())};
    f.read(t, 0, b.data.ptr(), t.bytes);
    return b;
  };
  auto mat = [&](const std::string& name, uint64_t N, uint64_t K) {
    const auto& t = require(name, {K, N});
    Ink2Mat m;
    m.N = uint32_t(N); m.K = uint32_t(K);
    if (t.type == gguf::kBF16 || t.type == gguf::kF32) m.raw = small(name, {K, N});
    else if (t.type == gguf::kQ8_0) {
      if (K % 32) throw gguf::Error("Inkling2: Q8_0 row is not block aligned: " + name);
      m.q = alloc(N * K); m.s = alloc(N * (K / 32) * 2);
      std::vector<uint8_t> rows(size_t(t.row_bytes()) * 256);
      for (size_t r = 0; r < N; r += 256) {
        const size_t nr = std::min<size_t>(256, N - r);
        f.read(t, r * t.row_bytes(), rows.data(), nr * t.row_bytes());
        for (size_t j = 0; j < nr; ++j)
          ink2::unpack_q8_row(rows.data() + j * t.row_bytes(), K, m.q.as<uint8_t>() + (r + j) * K,
                             m.s.as<uint16_t>() + (r + j) * (K / 32));
      }
    } else m.gguf = blocks(t);
    return m;
  };
  const uint64_t H = cfg_.hidden, QD = uint64_t(cfg_.hq) * cfg_.head_dim;
  const uint64_t KD = uint64_t(cfg_.hkv) * cfg_.head_dim, E = cfg_.n_experts, I = cfg_.moe_I;
  gguf_embed_ = mat("token_embd.weight", cfg_.vocab_padded, H);
  if (gguf_embed_.raw) throw gguf::Error("Inkling2 GGUF embedding must use block quantization");
  embed_norm_ = small("token_embd_norm.weight", {H});
  final_norm_ = small("output_norm.weight", {H});
  unembed_ = mat("output.weight", cfg_.vocab_padded, H);
  output_query_ = small("output_res_query.weight", {H});
  attn_queries_.resize(size_t(cfg_.num_layers - 1));
  mlp_queries_.resize(size_t(cfg_.num_layers));
  const size_t ncore = size_t(cfg_.core_end - cfg_.core_start);
  msg_queries_ = alloc(ncore * 4 * H * 4);
  layers_.resize(size_t(cfg_.num_layers));
  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    auto& L = layers_[size_t(layer)];
    const std::string p = "blk." + std::to_string(layer) + ".";
    L.is_swa = cfg_.is_local[size_t(layer)];
    L.has_ctrl = layer >= cfg_.core_start && layer < cfg_.core_end;
    if (layer) attn_queries_[size_t(layer - 1)] = L.is_swa ? small(p + "attn_res_query.weight", {H})
                                                                          : small(p + "attn_res_query.weight", {H, 3});
    mlp_queries_[size_t(layer)] = small(p + "ffn_res_query.weight", {H});
    L.wq_du = mat(p + "attn_q.weight", QD, H);
    L.wk_dv = mat(p + "attn_k.weight", KD, H);
    L.wv_dv = mat(p + "attn_v.weight", KD, H);
    L.wr_du = mat(p + "attn_rel.weight", uint64_t(cfg_.hq) * cfg_.d_rel, H);
    L.gate_attn = mat(p + "attn_gate.weight", QD, H);
    L.wo_ud = mat(p + "attn_output.weight", H, QD);
    L.attn_sink_proj = small(p + "attn_sink_proj.weight", {H, uint64_t(cfg_.hq)});
    L.q_scale = small(p + "attn_q_clip.weight", {uint64_t(cfg_.head_dim), uint64_t(cfg_.hq)});
    L.rel_scale = small(p + "attn_rel_clip.weight", {uint64_t(cfg_.d_rel), uint64_t(cfg_.hq)});
    if (L.q_scale->dtype != L.rel_scale->dtype) throw gguf::Error("Inkling2 clip dtypes differ");
    const auto* rp = small(p + "attn_rel_bias.weight", {uint64_t(cfg_.d_rel), uint64_t(cfg_.rel_extent)});
    Buffer rel = alloc(size_t(rp->numel()) * 4);
    for (int r = 0; r < cfg_.d_rel; ++r)
      for (int c = 0; c < cfg_.rel_extent; ++c)
        rel.as<float>()[size_t(r) * cfg_.rel_extent + c] = scalar(rp, size_t(c) * cfg_.d_rel + r);
    L.rel_proj = tensor(rel, DType::F32, {cfg_.d_rel, cfg_.rel_extent});
    L.mlp_norm = small(p + "ffn_norm.weight", {H});
    L.attn_sconv = small(p + "attn_post_conv.weight", {4, H});
    L.mlp_sconv = small(p + "ffn_post_conv.weight", {4, H});
    if (L.is_swa) {
      L.attn_norm = small(p + "attn_norm.weight", {H});
      L.k_sconv = small(p + "attn_k_conv.weight", {4, KD});
      L.v_sconv = small(p + "attn_v_conv.weight", {4, KD});
    } else {
      L.q_norm = small(p + "attn_q_inp_norm.weight", {H});
      L.k_norm = small(p + "attn_k_inp_norm.weight", {H});
      L.v_norm = small(p + "attn_v_inp_norm.weight", {H});
      auto dynamic_conv = [&](const std::string& name) {
        const auto* s = small(name, {4, KD});
        Buffer b = alloc(4 * KD * 4);
        for (size_t tap = 0; tap < 4; ++tap)
          for (size_t c = 0; c < KD; ++c) b.as<float>()[tap * KD + c] = scalar(s, c * 4 + (3 - tap));
        return tensor(b, DType::F32, {4, int64_t(KD)});
      };
      L.k_static = dynamic_conv(p + "attn_k_conv.weight");
      L.v_static = dynamic_conv(p + "attn_v_conv.weight");
      L.k_proj = small(p + "attn_k_conv_proj.weight", {H, uint64_t(4 * cfg_.hkv)});
      L.v_proj = small(p + "attn_v_conv_proj.weight", {H, uint64_t(4 * cfg_.hkv)});
    }
    if (L.has_ctrl) {
      auto ctrl = [&](const char* name) {
        const std::string c = p + "loop_" + name + "_ctrl_";
        Ink2Ctrl out{small(c + "down.weight", {2 * H, uint64_t(cfg_.gate_rank)}),
                     small(c + "forget.weight", {uint64_t(cfg_.gate_rank), H}),
                     small(c + "gain.weight", {uint64_t(cfg_.gate_rank), H})};
        if (out.down->dtype != out.forget->dtype || out.down->dtype != out.gain->dtype)
          throw gguf::Error("Inkling2 controller dtypes differ");
        return out;
      };
      L.ctrl_attn = ctrl("q"); L.ctrl_k = ctrl("k"); L.ctrl_v = ctrl("v"); L.ctrl_mlp = ctrl("ffn");
      const auto* q = small(p + "loop_msg_query.weight", {H, 4});
      float* dst = msg_queries_.as<float>() + size_t(layer - cfg_.core_start) * 4 * H;
      for (size_t j = 0; j < 4 * H; ++j) dst[j] = scalar(q, j);
    }
    L.router_w = small(p + "ffn_gate_inp.weight", {H, E});
    L.router_bias = small(p + "exp_probs_b.bias", {E});
    const auto experts = ink2::read_expert_sources(f, p, H, I, E);
    auto projection = [&](const ink2::ExpertSource& source, bool gate_up, const Tensor** q, const Tensor** sc,
                          Buffer& outer, Ink2Blocks& retained) {
      const auto& t = *source.weight;
      outer = alloc(E * sizeof(float));
      if (source.outer.empty()) std::fill_n(outer.as<float>(), E, 1.0f);
      else std::memcpy(outer.ptr(), source.outer.data(), E * sizeof(float));
      if (!source.native_nvfp4()) {
        retained = blocks(t);
        return;
      }
      const size_t K = t.ne[0], R = t.ne[1], rows = R * E;
      Buffer qb = alloc(rows * K / 2), sb = alloc(rows * K / 16);
      std::vector<uint8_t> buf(256 * t.row_bytes());
      for (size_t r = 0; r < rows; r += 256) {
        const size_t nr = std::min<size_t>(256, rows - r);
        f.read(t, r * t.row_bytes(), buf.data(), nr * t.row_bytes());
        for (size_t j = 0; j < nr; ++j) {
          const size_t src_row = r + j, expert = src_row / R, local = src_row % R;
          const size_t dst = expert * R + (gate_up ? ink2::native_gu_row(local, I) : local);
          ink2::unpack_nvfp4_row(buf.data() + j * t.row_bytes(), K, qb.as<uint8_t>() + dst * K / 2,
                                sb.as<uint8_t>() + dst * K / 16);
        }
      }
      *q = tensor(qb, DType::U8, {int64_t(E), int64_t(R), int64_t(K / 2)});
      *sc = tensor(sb, DType::F8E4M3, {int64_t(E), int64_t(R), int64_t(K / 16)});
    };
    projection(experts.gu, true, &L.w13, &L.w13_sc, L.w13_outer, L.gguf_gu);
    projection(experts.down, false, &L.w2, &L.w2_sc, L.w2_outer, L.gguf_down);
  }
  size_t weight_bytes = 0;
  for (const auto& b : owned_bufs_) weight_bytes += b.size;
  alloc_buffers();
  Gpu::get().make_resident(owned_bufs_);
  logi("loaded %s: Inkling2 mixed GGUF, %.2f GB text weights in %.0f ms", path_.c_str(), weight_bytes / 1e9, now_ms() - start);
}
}  // namespace ni
