#include "model/inkling2.h"
#include "engine/device_profile.h"

#include "engine/device_profile.h"
#include "engine/profile.h"
#include "model/inkling2_kernels.h"
#include "model/kparams.h"
#include "model/kv_grow.h"
#include "model/quant_q8.h"
#include "model/sampler.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

namespace ni {

extern const char* const kTensorKernelSource;

using json = nlohmann::json;

int g_ink2_ab_phase = 0;
int ink2_ab(const char* name, int v) {
  static const char* ab = getenv("BENCH_AB");
  if (!g_ink2_ab_phase || !ab) return v;
  const size_t n = strlen(name);
  for (const char* p = ab; p && *p; p = strchr(p, ',') ? strchr(p, ',') + 1 : nullptr)
    if (!strncmp(p, name, n) && p[n] == '=') return atoi(p + n + 1);
  return v;
}

namespace {

constexpr uint32_t kMaxAttnChunks = 64;
constexpr uint32_t kAttnChunkCap = 128;  // (lab12: partials capacity; NI_INK2_KW_MAXCH up to it)

// NI_INK2_SCFOLD mask (unset = 3: wo / w2 and SWA k / v folds on), A/B-able
int ink2_scfold() {
  static const int v = ink2_knob("NI_INK2_SCFOLD", 3, 0);
  return ink2_ab("NI_INK2_SCFOLD", v);
}

size_t visit_conv_floats(const Ink2Config& c) {
  const size_t kvd = size_t(c.hkv) * size_t(c.head_dim);
  return 3 * (2 * kvd + 2 * size_t(c.hidden));  // conv_k, conv_v, conv_a, conv_m
}

Pipeline ink2_pipe(const std::string& name) {
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

Pipeline base_pipe(const std::string& name) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  {
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
  }
  const Pipeline p = Gpu::get().pipeline(name);
  std::lock_guard<std::mutex> lk(mu);
  return cache.emplace(name, p).first->second;
}

// Prefill-size passes: one compute pass per dispatch (Encoder::set_pass_per_dispatch) from NI_INK2_PASS_MIN rows
// (default 256; 0 = off). ink2_bench 8K prefill 312 -> 380 tok/s on the M1 Max (one big encoder, grouped or serial,
// ran 311); decode / verify rows keep one encoder (a pass costs ~13 us there: 26 -> 40 ms a step).
struct Ink2PassEach {
  Encoder& e;
  const bool prev;
  Ink2PassEach(Encoder& enc, int T) : e(enc), prev(enc.pass_per_dispatch()) {
    static const int min_rows = ink2_knob("NI_INK2_PASS_MIN", 256, 0);
    if (min_rows > 0 && T >= min_rows && !enc.recording()) e.set_pass_per_dispatch(true);
  }
  ~Ink2PassEach() { e.set_pass_per_dispatch(prev); }
};

struct Ink2Grouped {
  Encoder& e;
  const bool on;
  Ink2Grouped(Encoder& enc, bool want)
      : e(enc), on(want && !enc.grouped() && !Encoder::profiling()) {
    if (on) e.set_grouped(true);
  }
  ~Ink2Grouped() {
    if (on) e.set_grouped(false);
  }
};

int ink2_fa_nt() {
  static const int v = ink2_knob("NI_INK2_FA", 16, 0);
  return v;
}

// lab12: g3kw_rc variant. NI_INK2_ATTL=1 (A/B-able): ink2_attn_decode_g3kw_rc_ml (ring slots without a per-key modulo +
// whole blocks by a constant V stride; bit-identical).
// The tiled K (NI_INK2_KT) is read only by the _mlt variant, so the decode binds it only when this is on.
static bool ink2_attl() {
  static const int attl = ink2_knob("NI_INK2_ATTL", 1, 0);
  return ink2_ab("NI_INK2_ATTL", attl) != 0;
}
std::string ink2_rc_name(bool kt) {
  return ink2_attl() ? (kt ? "ink2_attn_decode_g3kw_rc_mlt" : "ink2_attn_decode_g3kw_rc_ml") : "ink2_attn_decode_g3kw_rc";
}

// lab13 NI_INK2_KT=1: a tiled copy of every visit's K (Seq::kt, ink2_kt_idx in ink2.metal) read by the decode attention
// (ink2_attn_decode_g3kw_rc_mlt, with NI_INK2_ATTL=1 NI_INK2_REDLAST=3; bit-identical). Costs K's bytes again:
// +16 KB per position (8 global visits) and +50% on the SWA rings. Allocation and the writes follow the env at start;
// the read is A/B-able (BENCH_AB NI_INK2_KT=0).
static bool ink2_kt_alloc() {
  static const bool v = ink2_knob("NI_INK2_KT", 1, 0) != 0;
  return v;
}
static size_t ink2_kt_rows(size_t rows) { return (rows + 31) / 32 * 32; }

void prewarm_pipelines(bool tensor_pf, bool q8_dense) {
  if (const char* v = getenv("NI_PREWARM"); v && !strcmp(v, "0")) return;
  static const char* const kInk2Names[] = {
      "ink2_copy_bytes",       "ink2_bf16_mv",           "ink2_bf16_mv4",         "ink2_bf16_mv4_multi6",
      "ink2_bf16_mv8",         "ink2_bf16_mm", "ink2_mv_sk",           "ink2_rmsnorm",          "ink2_attn_res",
      "ink2_msg_gate",         "ink2_msg_gate_prep_s",   "ink2_msg_gate_down_s",  "ink2_msg_gate_up_mix_s", "ink2_msg_gate_down_f", "ink2_msg_gate_up_mix_c",
      "ink2_embed_norm",       "ink2_sconv_static",      "ink2_sconv_static_f",    "ink2_sconv_static_pf",  "ink2_sconv_dynamic",    "ink2_qkvr_prep",         "ink2_rel_bias",         "ink2_rel_bias_pf",
      "ink2_attn_decode",      "ink2_attn_decode_reduce", "ink2_attn_decode_g3", "ink2_attn_decode_g3_fx", "ink2_attn_decode_g3kw", "ink2_attn_res1", "ink2_msg_gate_mix","ink2_attn_prefill",     "ink2_moe_route",
      "ink2_moe_nvfp4_w13_situ","ink2_moe_nvfp4_w2_acc", "ink2_moe_nvfp4_w13_q4", "ink2_moe_nvfp4_w2_f2", "ink2_moe_mxfp8_w13_situ","ink2_moe_mxfp8_w2_acc", "ink2_moe_mxfp8_w13_situ_v", "ink2_moe_mxfp8_w2_acc_v",
      "ink2_softcap",           "ink2_moe_bucket",       "ink2_moe_situ_half",
      "ink2_moe_combine",       "ink2_fa_t16k32",        "ink2_fa_t16k16",        "ink2_sum_parts",
  };
  for (const char* n : kInk2Names) (void)ink2_pipe(n);
  if (tensor_pf) {
    static const char* const kTensorNames[] = {
        "ink2_t_bf16_m64",          "ink2_t_bf16_sk_m64",
        "ink2_t_nvfp4_moe_w13_m16", "ink2_t_nvfp4_moe_w13_m32", "ink2_t_nvfp4_moe_w13_m64",
        "ink2_t_nvfp4_moe_w2_m16",  "ink2_t_nvfp4_moe_w2_m32",  "ink2_t_nvfp4_moe_w2_m64",
        "ink2_t_mxfp8_moe_w13_m16", "ink2_t_mxfp8_moe_w13_m32", "ink2_t_mxfp8_moe_w13_m64",
        "ink2_t_mxfp8_moe_w2_m16",  "ink2_t_mxfp8_moe_w2_m32",  "ink2_t_mxfp8_moe_w2_m64",
        "qattn_ink2_s6k32p",        "qattn_ink2_s3k16",        "qattn_ink2_s6k32p_l1",     "qattn_ink2_s3k16_l1",
    };
    for (const char* n : kTensorNames) (void)base_pipe(n);
  }
  if (q8_dense) {
    for (const char* n : {"ink2_q8_mv1", "ink2_q8_mv", "ink2_q8_mm"}) (void)ink2_pipe(n);
  }
  (void)base_pipe("topk_stage1");
  (void)base_pipe("sample_stage2");
}

struct Ink2EmbedP {
  uint32_t T, H, tok_off, V;
  float eps;
};
struct Ink2NormP {
  uint32_t T, H, has_w, w_f32;
  float eps;
};
struct Ink2AttnResP {
  uint32_t T, H, M, Q, q_f32;
  float eps, scale;
};
struct Ink2MsgGateP {
  uint32_t T, H, R, w_f32;
  float eps;
};
struct Ink2GgufMmP { uint32_t M, N, K, type, row_bytes, add; };
struct Ink2GgufMoeP { uint32_t T, H, I, top, type, row_bytes; };
struct Ink2GgufEmbedP { uint32_t T, H, tok_off, V, norm_f32; float eps; uint32_t type, row_bytes; };
struct Ink2MmP {
  uint32_t M, N, K, x_ld, y_ld, w_f32, add;
};
struct Ink2MultiMmP {
  uint32_t M, K, N_tot, n_segs;
  uint32_t offs[7];
};
// The fused short conv epilogues round with explicit fmas; on the M1 stack the standalone conv matches them
static std::string ink2_sconv_name() {
  static const bool f = ink2_knob("NI_INK2_SCFMA", 1, 0) != 0;
  return f ? "ink2_sconv_static_f" : "ink2_sconv_static";
}
struct Ink2SconvStaticP {
  uint32_t T, C, w_f32, add_res;
  float scale;
  uint32_t snap_stride;
};
struct Ink2SconvDynP {
  uint32_t T, C, n_heads, sw_f32, snap_stride;
};
struct Ink2QkvrP {
  uint32_t T, Hq, Hkv, D, d_rel, pos0, kv_cap, is_swa, scale_f32;
  float qk_clip, r_clip, v_eps, log_floor, log_alpha;
  uint32_t kt;
};
struct Ink2KtP {
  uint32_t a, Hkv, D;
};
struct Ink2RelP {
  uint32_t T, Hq, d_rel, extent, proj_f32;
};
struct Ink2AttnDecP {
  uint32_t pos, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, chunk, nchunks;
  float scale;
};
struct Ink2Q8P {
  uint32_t M, N, K, x_ld, y_ld, add, n_segs;
  uint32_t offs[7];
};
struct Ink2AttnVerP {
  uint32_t pos0, T, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa, k_first, k_end, chunk, nchunks;
  float scale;
};
struct Ink2AttnPfP {
  uint32_t T, pos0, kv_cap, Hq, Hkv, D, rel_extent, is_swa, swa;
  float scale;
};
struct Ink2RouteP {
  uint32_t T, E, K, bias_f32;
  float route_scale;
};
struct Ink2MoeP {
  uint32_t T, H, I, K_top;
};
struct Ink2SoftcapP {
  uint32_t n;
  float cap, inv_cap;
};
struct Ink2CapP {
  uint32_t N, K;
  float cap, inv_cap;
};
struct Ink2BucketP {
  uint32_t P, E, BM;
};
struct Ink2SituP {
  uint32_t n;
};
struct Ink2CombineP {
  uint32_t T, H, K_top;
};
struct Ink2tGemmP {
  uint32_t M, N, K;
  uint32_t x_ld, y_ld;
  uint32_t add;
  uint32_t q_estride, s_estride;
  uint32_t k;
};

const json& text_cfg(const json& root) { return root.contains("text_config") ? root["text_config"] : root; }

float read_elem_f32(const Tensor& t, size_t idx) {
  if (t.dtype == DType::F32) return t.buf.as<float>()[idx];
  if (t.dtype == DType::BF16) return bf16_to_f32(t.buf.as<uint16_t>()[idx]);
  fatal("inkling2: unsupported scalar/vec dtype");
}

void enc_copy_buf(Encoder& e, const Buffer& src, const Buffer& dst, size_t bytes = 0) {
  const uint64_t n = uint64_t(bytes ? bytes : std::min(src.size, dst.size));
  if (!n) return;
  e.use(ink2_pipe("ink2_copy_bytes"));
  e.buf(0, src);
  e.buf(1, dst);
  e.val(2, n);
  const uint32_t threads = uint32_t((n + 15) / 16);
  e.dispatch((threads + 255) / 256, 1, 1, 256);
}

size_t safetensors_llm_bytes(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  size_t total = 0;
  for (const auto& ent : fs::directory_iterator(dir, ec)) {
    const std::string p = ent.path().string();
    if (p.size() <= 12 || p.compare(p.size() - 12, 12, ".safetensors") != 0) continue;
    std::ifstream in(p, std::ios::binary);
    if (!in) continue;
    uint64_t hlen = 0;
    in.read(reinterpret_cast<char*>(&hlen), 8);
    if (!in || hlen == 0 || hlen > (64u << 20)) continue;
    std::string hdr(size_t(hlen), '\0');
    in.read(hdr.data(), std::streamsize(hlen));
    const json j = json::parse(hdr, nullptr, false);
    if (!j.is_object()) continue;
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.key().rfind("model.llm.", 0) != 0 || !it.value().is_object()) continue;
      if (it.value().contains("data_offsets") && it.value()["data_offsets"].is_array() &&
          it.value()["data_offsets"].size() == 2) {
        const size_t a = it.value()["data_offsets"][0].get<size_t>();
        const size_t b = it.value()["data_offsets"][1].get<size_t>();
        if (b >= a) total += (b - a);
      }
    }
  }
  return total;
}

[[noreturn]] __attribute__((format(printf, 1, 2))) void cfg_error(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  throw std::runtime_error(buf);
}

}  // namespace

bool Inkling2::is_model(const std::string& dir) {
  const auto jp = json_file(dir + "/config.json");
  if (!jp || !jp->is_object()) return false;
  if (jp->value("model_type", std::string()) == "inkling2_loop") return true;
  return jp->contains("text_config") && (*jp)["text_config"].is_object() &&
         (*jp)["text_config"].value("model_type", std::string()) == "inkling2_loop_text";
}

Ink2Config Inkling2::read_config(const std::string& dir) {
  const auto rp = json_file(dir + "/config.json");
  if (!rp || !rp->is_object()) cfg_error("%s: cannot read config.json", dir.c_str());
  const json& root = *rp;
  const json& t = text_cfg(root);
  if (!t.is_object()) cfg_error("%s: missing text_config object", dir.c_str());

  auto req_int = [&](const json& obj, const char* key) -> int {
    if (!obj.contains(key) || !obj[key].is_number_integer())
      cfg_error("%s: missing or non-integer %s in config.json", dir.c_str(), key);
    return obj[key].get<int>();
  };
  auto opt_int = [&](const json& obj, const char* key, int def) -> int {
    if (!obj.contains(key)) return def;
    if (!obj[key].is_number_integer()) cfg_error("%s: non-integer %s in config.json", dir.c_str(), key);
    return obj[key].get<int>();
  };
  auto opt_float = [&](const json& obj, const char* key, float def) -> float {
    if (!obj.contains(key)) return def;
    if (!obj[key].is_number()) cfg_error("%s: non-numeric %s in config.json", dir.c_str(), key);
    return obj[key].get<float>();
  };
  auto opt_bool = [&](const json& obj, const char* key, bool def) -> bool {
    if (!obj.contains(key)) return def;
    if (!obj[key].is_boolean()) cfg_error("%s: non-boolean %s in config.json", dir.c_str(), key);
    return obj[key].get<bool>();
  };

  Ink2Config c;
  c.hidden = req_int(t, "hidden_size");
  c.num_layers = req_int(t, "num_hidden_layers");
  c.vocab_padded = req_int(t, "vocab_size");
  c.vocab = opt_int(t, "unpadded_vocab_size", c.vocab_padded);
  c.hq = req_int(t, "num_attention_heads");
  c.hkv = req_int(t, "num_key_value_heads");
  c.head_dim = req_int(t, "head_dim");
  c.d_rel = opt_int(t, "d_rel", 16);
  c.rel_extent = opt_int(t, "rel_extent", 1024);
  c.swa = opt_int(t, "sliding_window_size", 1024);
  c.block_sz = opt_int(t, "n_layers_per_attn_res_block", 6);
  c.n_experts = opt_int(t, "n_routed_experts", 128);
  c.top_k = opt_int(t, "num_experts_per_tok", 8);
  c.moe_I = opt_int(t, "intermediate_size", 1152);
  c.rms_eps = opt_float(t, "rms_norm_eps", 1e-6f);
  c.qk_clip = opt_float(t, "qk_l2_clip_threshold", 22.627416997969522f);
  c.rel_clip = opt_float(t, "rel_l2_clip_threshold", 8.0f);
  c.log_floor = float(opt_int(t, "log_scaling_n_floor", 8192));
  c.log_alpha = opt_float(t, "log_scaling_alpha", 0.1f);
  c.route_scale = opt_float(t, "route_scale", 8.0f);
  c.logit_softcap = opt_float(t, "final_logit_softcapping", 30.0f);

  if (!t.contains("loop_config") || !t["loop_config"].is_object())
    cfg_error("%s: missing loop_config in config.json", dir.c_str());
  const json& loop = t["loop_config"];
  c.core_start = req_int(loop, "core_start");
  c.core_end = req_int(loop, "core_end");
  c.num_passes = req_int(loop, "num_passes");
  c.gate_rank = opt_int(loop, "gate_rank", 64);

  if (opt_bool(t, "q_bias", false) || opt_bool(t, "o_bias", false))
    cfg_error("%s: Inkling-2 with attention q_bias/o_bias is not supported", dir.c_str());
  if (!opt_bool(t, "use_value_norm_no_gamma", true) || !opt_bool(t, "attn_res_normalize_values", true) ||
      !opt_bool(t, "use_gate_bias", true))
    cfg_error("%s: Inkling-2 requires use_value_norm_no_gamma, attn_res_normalize_values, and use_gate_bias", dir.c_str());
  if (opt_bool(loop, "is_varloop", false))
    cfg_error("%s: Inkling-2 varloop is not supported", dir.c_str());
  if (opt_int(t, "sconv_kernel_size", 4) != 4 || opt_int(t, "n_shared_experts", 0) != 0)
    cfg_error("%s: Inkling-2 requires sconv_kernel_size 4 and n_shared_experts 0", dir.c_str());
  if (opt_int(t, "swa_head_dim", c.head_dim) != c.head_dim ||
      opt_int(t, "swa_num_attention_heads", c.hq) != c.hq ||
      opt_int(t, "swa_num_key_value_heads", c.hkv) != c.hkv)
    cfg_error("%s: Inkling-2 heterogeneous SWA head geometry is not supported", dir.c_str());
  if (std::fabs(opt_float(t, "mlp_gate_softcap", 4.0f) - 4.0f) > 1e-4f ||
      std::fabs(opt_float(t, "mlp_up_softcap", 25.0f) - 25.0f) > 1e-4f)
    cfg_error("%s: Inkling-2 requires mlp_gate_softcap 4 and mlp_up_softcap 25", dir.c_str());

  if (c.hidden <= 0 || c.hidden > 2304 || c.hidden % 128 != 0 || c.num_layers < 2 || c.vocab <= 0 ||
      c.vocab > c.vocab_padded || c.head_dim != 128 || c.hq <= 0 || c.hkv <= 0 || c.hq % c.hkv != 0 ||
      c.d_rel <= 0 || c.d_rel > 128 || c.d_rel % 4 != 0 || c.rel_extent <= 0 || c.swa <= 0 ||
      c.n_experts != 128 || c.top_k < 1 || c.top_k > 8 || c.moe_I <= 0 || c.moe_I % 32 != 0 ||
      c.gate_rank != 64 || c.block_sz <= 0 || (c.num_layers + c.block_sz - 1) / c.block_sz > 4 ||
      c.core_start <= 0 || c.core_start % c.block_sz != 0 || c.core_end <= c.core_start ||
      c.core_end > c.num_layers || c.core_end % c.block_sz != 0 ||
      c.core_end - c.core_start != 2 * c.block_sz || c.num_passes < 1 ||
      !(c.rms_eps > 0.0f && c.qk_clip > 0.0f && c.rel_clip > 0.0f && c.log_floor > 0.0f &&
        c.logit_softcap > 0.0f && std::isfinite(c.log_alpha) && std::isfinite(c.route_scale)))
    cfg_error("%s: an Inkling-2 shape Kapitsa does not run (hidden %d, layers %d, heads %d/%d of %d, "
              "%d experts top-%d I=%d, gate_rank=%d, block=%d core=%d..%d, vocab=%d/%d)",
              dir.c_str(), c.hidden, c.num_layers, c.hq, c.hkv, c.head_dim, c.n_experts, c.top_k, c.moe_I,
              c.gate_rank, c.block_sz, c.core_start, c.core_end, c.vocab, c.vocab_padded);

  c.is_local.assign(size_t(c.num_layers), false);
  if (t.contains("local_layer_ids")) {
    if (!t["local_layer_ids"].is_array())
      cfg_error("%s: local_layer_ids must be an array", dir.c_str());
    for (const auto& lid : t["local_layer_ids"]) {
      if (!lid.is_number_integer()) cfg_error("%s: local_layer_ids must contain integers", dir.c_str());
      const int idx = lid.get<int>();
      if (idx < 0 || idx >= c.num_layers)
        cfg_error("%s: local_layer_ids contains out-of-range layer %d", dir.c_str(), idx);
      c.is_local[size_t(idx)] = true;
    }
  }
  if (!c.is_local[0])
    cfg_error("%s: Inkling-2 layer 0 must be a local layer", dir.c_str());

  for (int i = 0; i < c.core_start; ++i) c.schedule.push_back(i);
  for (int pass = 0; pass < c.num_passes; ++pass)
    for (int i = c.core_start; i < c.core_end; ++i) c.schedule.push_back(i);
  for (int i = c.core_end; i < c.num_layers; ++i) c.schedule.push_back(i);

  const auto gp = json_file(dir + "/generation_config.json");
  auto push_eos = [&](const json& v) {
    if (v.is_number_integer()) {
      const int id = v.get<int>();
      if (std::find(c.eos.begin(), c.eos.end(), id) == c.eos.end()) c.eos.push_back(id);
    } else if (v.is_array()) {
      for (const auto& x : v)
        if (x.is_number_integer()) {
          const int id = x.get<int>();
          if (std::find(c.eos.begin(), c.eos.end(), id) == c.eos.end()) c.eos.push_back(id);
        }
    }
  };
  if (gp && gp->is_object()) {
    if (gp->contains("eos_token_id")) push_eos((*gp)["eos_token_id"]);
  }
  if (root.contains("eos_token_id")) push_eos(root["eos_token_id"]);
  if (c.eos.empty()) c.eos = {200006, 199999};
  else if (std::find(c.eos.begin(), c.eos.end(), 199999) == c.eos.end()) c.eos.push_back(199999);
  return c;
}

size_t Inkling2::kv_per_pos(const Ink2Config& c) {
  int n_global = 0;
  for (int ip : c.schedule)
    if (!c.is_local[size_t(ip)]) ++n_global;
  // (2 (K, V) + the tiled K copy) * hkv * head_dim * sizeof(half)
  return size_t(n_global) * (ink2_kt_alloc() ? 3 : 2) * size_t(c.hkv) * size_t(c.head_dim) * 2;
}

size_t Inkling2::conv_bytes_per_seq(const Ink2Config& c) {
  return c.schedule.size() * visit_conv_floats(c) * 4;
}

size_t Inkling2::ring_bytes_per_seq(const Ink2Config& c, int tmax) {
  int n_swa = 0;
  for (int ip : c.schedule)
    if (c.is_local[size_t(ip)]) ++n_swa;
  const int rcap = (c.swa + std::max(1, tmax) + 15) / 16 * 16;
  return size_t(n_swa) * (2 * size_t(rcap) + (ink2_kt_alloc() ? ink2_kt_rows(size_t(rcap)) : 0)) * size_t(c.hkv) *
         size_t(c.head_dim) * 2;
}

static Profile ink2_profile() {
  return profile(default_profile(current_device(), ModelFacts{.inkling2 = true}));
}

// The M1 stack (device profile row ink2_m1_stack: the fast profile on the legacy class, Apple7 = M1): the decode and
// prefill changes measured on the M1 Max default to their served values there; on every other Mac, and under
// precise, they default to the path #1379 / #1381 / #1387 shipped. A switch's env var, when set, always wins.
bool ink2_m1_stack() {  // NI_INK2_M1_STACK=0|1 overrides the row (0: every switch at its shipped default)
  static const char* env = getenv("NI_INK2_M1_STACK");
  static const bool v = env && *env ? atoi(env) != 0
                                    : ink2_profile() == Profile::kFast && default_on(Knob::kInk2M1Stack, ModelFacts{.inkling2 = true});
  return v;
}
int ink2_knob(const char* env, int m1_value, int off_value) {
  if (const char* v = getenv(env); v && *v) return atoi(v);
  return ink2_m1_stack() ? m1_value : off_value;
}

static bool ink2_tensor_pf_want() {
  if (__builtin_available(macOS 26.0, *)) {
    const Profile selected = ink2_profile();
    const bool here = default_on(Knob::kInk2TensorPf, ModelFacts{}) || ink2_m1_stack();
    return profile_switch("NI_INK2_TENSOR", here, "Inkling-2-Mini prefill GEMMs on the matrix units (NI_INK2_TENSOR)", selected);
  }
  return false;
}

bool Inkling2::fast_profile_available() {
  if (ink2_profile() != Profile::kPrecise) return false;
  auto unset = [](const char* env) {
    const char* v = std::getenv(env);
    return v == nullptr || *v == '\0';
  };
  if (__builtin_available(macOS 26.0, *)) {
    if (unset("NI_INK2_TENSOR") && default_on(Knob::kInk2TensorPf, ModelFacts{})) return true;
  }
  return unset("NI_INK2_Q8") && default_on(Knob::kInk2Q8Fast, ModelFacts{});
}

MemPlan Inkling2::plan(const std::string& dir, int prefill_chunk) {
  const Ink2Config c = read_config(dir);
  const int chunk = prefill_chunk > 0 ? prefill_chunk : 512;
  const size_t T = size_t(std::max(8, chunk));
  const size_t H = size_t(c.hidden);
  const size_t QD = size_t(c.hq) * size_t(c.head_dim);
  const size_t KVD = size_t(c.hkv) * size_t(c.head_dim);
  const size_t V = size_t(c.vocab);
  const size_t per_row = 33 * H * 4 + size_t(c.gate_rank) * 4 + 3 * QD * 4 + 2 * KVD * 4 + size_t(c.hq) * size_t(c.d_rel) * 4 +
                         size_t(c.hq) * 4 + 4 * size_t(c.hkv) * 4 + size_t(c.hq) * size_t(c.rel_extent) * 4 +
                         size_t(c.n_experts) * 4 + 2 * size_t(c.top_k) * 4 + size_t(c.top_k) * size_t(c.moe_I) * 4;
  const size_t fixed = size_t(c.core_end - c.core_start) * 4 * H * 4 +
                       size_t(c.num_layers) * 2 * size_t(c.n_experts) * 4 +
                       size_t(c.hq) * kAttnChunkCap * kMaxVerify * (2 + size_t(c.head_dim)) * 4 + V * 4 +
                       2 * (V / 2048 + 2) * 64 * 4 + 2 * 64 * 4;
  const size_t P = T * size_t(c.top_k);
  const size_t moe_tmp_bytes = ink2_tensor_pf_want()
                                   ? (P * std::max(H, size_t(2) * size_t(c.moe_I)) * 4 + P * size_t(c.moe_I) * 2 +
                                      2 * P * 4 + (P / 16 + size_t(c.n_experts) + 2) * 16 + 16)
                                   : 0;
  MemPlan m;
  const auto model_config = json_file(dir + "/config.json");
  const bool gguf = model_config->contains("gguf_file");
  m.weights = gguf ? model_config->at("gguf_tensor_bytes").get<size_t>() + (16u << 20) : safetensors_llm_bytes(dir);
  m.kv_per_pos = kv_per_pos(c);
  m.slot_state = 3 * (ring_bytes_per_seq(c, chunk) + conv_bytes_per_seq(c));
  m.activations = T * per_row + fixed + moe_tmp_bytes + (16u << 20);
  m.hidden = c.hidden;
  m.layers = int(c.schedule.size());
  m.formats = gguf ? "mixed GGUF blocks + native NVFP4 (text-only)"
                   : "NVFP4 (L0-19) + MXFP8 (L20-23) + BF16 dense (text-only)";
  return m;
}

Inkling2::Inkling2(const std::string& path, int prefill_chunk) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not an Inkling-2 checkpoint folder", path.c_str());
  cfg_ = read_config(path_);
  profile_ = ink2_profile();
  tmax_ = std::max(8, prefill_chunk > 0 ? prefill_chunk : 512);
  ring_rows_ = (cfg_.swa + tmax_ + 15) / 16 * 16;
  q8_dense_ = profile_switch("NI_INK2_Q8", default_on(Knob::kInk2Q8Fast, {}),
                             "Inkling-2-Mini Q8-packed dense projections and unembed (NI_INK2_Q8)", profile_);
  logi("Inkling-2 profile: %s", profile_ == Profile::kFast ? "fast" : "precise");
  ink2::compile_kernels();
  if (ink2_tensor_pf_want()) {
    if (__builtin_available(macOS 26.0, *)) {
      Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
      tensor_pf_ = Gpu::get().pipeline("ink2_t_bf16_m64", false).pso != nullptr;
    }
  }
  q8m_ = q8_dense_ && ink2_knob("NI_INK2_Q8M", 1, 0) != 0;
  q8_bf16_ = q8_dense_ && tensor_pf_ && ink2_knob("NI_INK2_Q8_BF16", 1, 0) != 0;
  prewarm_pipelines(tensor_pf_, q8_dense_);
  load();
}

Inkling2::~Inkling2() {
  Gpu& g = Gpu::get();
  for (Buffer& b : owned_bufs_) g.release(b);
}

void Inkling2::load() {
  if (json_file(path_ + "/config.json")->contains("gguf_file")) {
    load_gguf();
    return;
  }
  const double t0 = now_ms();
  auto is_q8_target = [&](const std::string& n) -> bool {
    if (!q8_dense_) return false;
    if (n == "model.llm.unembed.weight") return true;
    static const char* const kSuffixes[] = {
        ".attn.wq_du.weight", ".attn.wk_dv.weight", ".attn.wv_dv.weight",
        ".attn.wr_du.weight", ".attn.gate_attn.weight", ".attn.wo_ud.weight",
    };
    for (const char* suf : kSuffixes) {
      const size_t sl = strlen(suf);
      if (n.size() > sl && n.compare(n.size() - sl, sl, suf) == 0) return true;
    }
    return false;
  };
  ws_.open_dir(
      path_, [](const std::string& n) { return n.rfind("model.llm.", 0) == 0; }, nullptr,
      q8_dense_ && !q8_bf16_ ? WeightStore::Filter(is_q8_target) : WeightStore::Filter(nullptr));
  auto req_shape = [&](const std::string& name, std::initializer_list<int64_t> want) -> const Tensor* {
    const Tensor* t = ws_.find(name);
    if (!t) fatal("%s: missing tensor %s", path_.c_str(), name.c_str());
    if (t->shape.size() != want.size() || !std::equal(want.begin(), want.end(), t->shape.begin()))
      fatal("%s: %s has an unexpected shape", path_.c_str(), name.c_str());
    return t;
  };
  auto req_float = [&](const std::string& name, std::initializer_list<int64_t> want) -> const Tensor* {
    const Tensor* t = req_shape(name, want);
    if (t->dtype != DType::BF16 && t->dtype != DType::F32)
      fatal("%s: %s must be BF16 or F32", path_.c_str(), name.c_str());
    return t;
  };
  auto req_raw = [&](const std::string& name, DType dtype, std::initializer_list<int64_t> want) -> const Tensor* {
    const Tensor* t = req_shape(name, want);
    if (t->dtype != dtype) fatal("%s: %s has an unexpected dtype", path_.c_str(), name.c_str());
    return t;
  };
  auto req_any = [&](const std::string& name) -> const Tensor* {
    const Tensor* t = ws_.find(name);
    if (!t) fatal("%s: missing tensor %s", path_.c_str(), name.c_str());
    return t;
  };

  size_t q8_bytes = 0;
  const unsigned nthr = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  auto load_mat = [&](const std::string& name, int64_t want_n, int64_t want_k, bool allow_ge_n = false) -> Ink2Mat {
    const Tensor* t = req_any(name);
    if ((t->dtype != DType::BF16 && t->dtype != DType::F32) || t->shape.size() != 2 || t->shape[1] != want_k ||
        (allow_ge_n ? t->shape[0] < want_n : t->shape[0] != want_n))
      fatal("%s: %s must be BF16/F32 [%s%lld, %lld]", path_.c_str(), name.c_str(), allow_ge_n ? ">=" : "",
            (long long)want_n, (long long)want_k);
    Ink2Mat m;
    m.N = uint32_t(want_n);
    m.K = uint32_t(want_k);
    std::vector<uint8_t> host_raw;
    const uint8_t* host = nullptr;
    if (!ws_.deferred(name)) {
      m.raw = t;
      if (!q8_bf16_ || !is_q8_target(name)) return m;
      host = t->buf.as<uint8_t>();  // NI_INK2_Q8_BF16: the BF16 copy stays for the prefill GEMMs, Q8 beside it
    } else {
      host_raw.resize(size_t(t->numel()) * dtype_size(t->dtype));
      ws_.read_deferred(name, host_raw.data());
      host = host_raw.data();
    }
    const size_t qsz = size_t(want_n) * size_t(want_k);
    const size_t ssz = size_t(want_n) * size_t(want_k / 32) * 2;
    m.q = Gpu::get().alloc(qsz, "ink2_q8_q");
    m.s = Gpu::get().alloc(ssz, "ink2_q8_s");
    owned_bufs_.push_back(m.q);
    owned_bufs_.push_back(m.s);
    q8_bytes += qsz + ssz;
    uint8_t* qdst = m.q.as<uint8_t>();
    uint8_t* sdst = m.s.as<uint8_t>();
    const bool is_f32 = t->dtype == DType::F32;
    const size_t rows = size_t(want_n);
    const size_t K = size_t(want_k);
    const size_t srow = K / 32 * 2;
    auto worker = [&](size_t r0, size_t r1) {
      std::vector<float> frow(K);
      for (size_t r = r0; r < r1; ++r) {
        if (is_f32) {
          const float* src = reinterpret_cast<const float*>(host) + r * K;
          nh_quant_q8(src, want_k, qdst + r * K, sdst + r * srow);
        } else {
          const uint16_t* src = reinterpret_cast<const uint16_t*>(host) + r * K;
          for (size_t c = 0; c < K; ++c) frow[c] = bf16_to_f32(src[c]);
          nh_quant_q8(frow.data(), want_k, qdst + r * K, sdst + r * srow);
        }
      }
    };
    const size_t use_thr = std::min<size_t>(nthr, std::max<size_t>(1, rows / 64));
    if (use_thr <= 1) {
      worker(0, rows);
    } else {
      std::vector<std::thread> ts;
      ts.reserve(use_thr);
      for (size_t ti = 0; ti < use_thr; ++ti) {
        const size_t r0 = rows * ti / use_thr;
        const size_t r1 = rows * (ti + 1) / use_thr;
        ts.emplace_back(worker, r0, r1);
      }
      for (auto& th : ts) th.join();
    }
    return m;
  };

  const int64_t H64 = cfg_.hidden;
  const int64_t QD64 = int64_t(cfg_.hq) * cfg_.head_dim;
  const int64_t KVD64 = int64_t(cfg_.hkv) * cfg_.head_dim;
  const int64_t E64 = cfg_.n_experts;
  const int64_t I64 = cfg_.moe_I;

  embed_ = req_any("model.llm.embed.weight");
  if (embed_->dtype != DType::BF16 || embed_->shape.size() != 2 || embed_->shape[0] < cfg_.vocab ||
      embed_->shape[1] != H64)
    fatal("%s: model.llm.embed.weight must be BF16 [>=%d, %lld]", path_.c_str(), cfg_.vocab, (long long)H64);
  embed_norm_ = req_raw("model.llm.embed_norm.weight", DType::BF16, {H64});
  final_norm_ = req_float("model.llm.norm.weight", {H64});
  unembed_ = load_mat("model.llm.unembed.weight", cfg_.vocab, H64, true);

  attn_queries_.resize(size_t(cfg_.num_layers - 1));
  for (int i = 0; i < cfg_.num_layers - 1; ++i) {
    const std::string qn = "model.llm.attn_res.attn_queries." + std::to_string(i);
    attn_queries_[size_t(i)] = cfg_.is_local[size_t(i + 1)] ? req_float(qn, {H64}) : req_float(qn, {3, H64});
  }
  mlp_queries_.resize(size_t(cfg_.num_layers));
  for (int i = 0; i < cfg_.num_layers; ++i)
    mlp_queries_[size_t(i)] = req_float("model.llm.attn_res.mlp_queries." + std::to_string(i), {H64});
  output_query_ = req_float("model.llm.attn_res.output_query", {H64});

  // Pre-interleave [12, 4, H] middle_cycle cross-pass queries: (attn, k, v, mlp)
  const int ncore = cfg_.core_end - cfg_.core_start;
  const Tensor* qam = req_float("model.llm.middle_cycle_cross_pass_queries", {int64_t(2) * ncore, H64});
  const Tensor* qkv = req_float("model.llm.middle_cycle_cross_pass_kv_queries", {int64_t(2) * ncore, H64});
  const size_t H = size_t(cfg_.hidden);
  msg_queries_ = Gpu::get().alloc(size_t(ncore) * 4 * H * 4, "ink2_msg_queries");
  owned_bufs_.push_back(msg_queries_);
  {
    float* dst = msg_queries_.as<float>();
    for (int c = 0; c < ncore; ++c) {
      for (size_t j = 0; j < H; ++j) {
        dst[(size_t(c) * 4 + 0) * H + j] = read_elem_f32(*qam, (size_t(c) * 2 + 0) * H + j);
        dst[(size_t(c) * 4 + 1) * H + j] = read_elem_f32(*qkv, (size_t(c) * 2 + 0) * H + j);
        dst[(size_t(c) * 4 + 2) * H + j] = read_elem_f32(*qkv, (size_t(c) * 2 + 1) * H + j);
        dst[(size_t(c) * 4 + 3) * H + j] = read_elem_f32(*qam, (size_t(c) * 2 + 1) * H + j);
      }
    }
  }

  auto make_outer = [&](const std::string& base) -> Buffer {
    const Tensor* amax = req_float(base + "_amax", {E64});
    const Tensor* denom = req_any(base + "_outer_scale_denominator");
    if ((denom->dtype != DType::BF16 && denom->dtype != DType::F32) || denom->numel() != 1)
      fatal("%s: %s_outer_scale_denominator must be a 1-element BF16/F32 tensor", path_.c_str(), base.c_str());
    const float d = read_elem_f32(*denom, 0);
    if (!std::isfinite(d) || d <= 0.0f)
      fatal("%s: invalid %s_outer_scale_denominator (%g)", path_.c_str(), base.c_str(), d);
    Buffer b = Gpu::get().alloc(size_t(cfg_.n_experts) * 4, "ink2_outer");
    owned_bufs_.push_back(b);
    float* out = b.as<float>();
    for (int e = 0; e < cfg_.n_experts; ++e) out[e] = read_elem_f32(*amax, size_t(e)) / d;
    return b;
  };

  layers_.assign(size_t(cfg_.num_layers), Ink2Layer{});
  for (int i = 0; i < cfg_.num_layers; ++i) {
    Ink2Layer& L = layers_[size_t(i)];
    const std::string p = "model.llm.layers." + std::to_string(i) + ".";
    L.is_swa = cfg_.is_local[size_t(i)];
    L.has_ctrl = (i >= cfg_.core_start && i < cfg_.core_end);
    L.wq_du = load_mat(p + "attn.wq_du.weight", QD64, H64);
    L.wk_dv = load_mat(p + "attn.wk_dv.weight", KVD64, H64);
    L.wv_dv = load_mat(p + "attn.wv_dv.weight", KVD64, H64);
    L.wr_du = load_mat(p + "attn.wr_du.weight", int64_t(cfg_.hq) * cfg_.d_rel, H64);
    L.attn_sink_proj = req_float(p + "attn.attn_sink_proj.weight", {cfg_.hq, H64});
    L.gate_attn = load_mat(p + "attn.gate_attn.weight", QD64, H64);
    L.wo_ud = load_mat(p + "attn.wo_ud.weight", H64, QD64);
    L.q_scale = req_float(p + "attn.q_l2_clip.scale", {cfg_.hq, cfg_.head_dim});
    L.rel_scale = req_float(p + "attn.rel_l2_clip.scale", {cfg_.hq, cfg_.d_rel});
    if (L.q_scale->dtype != L.rel_scale->dtype)
      fatal("%s: layer %d q_l2_clip.scale and rel_l2_clip.scale dtypes differ", path_.c_str(), i);
    L.rel_proj = req_float(p + "attn.rel_logits_proj.proj", {cfg_.d_rel, cfg_.rel_extent});
    L.mlp_norm = req_float(p + "mlp_norm.weight", {H64});
    L.attn_sconv = req_float(p + "attn_sconv.weight", {H64, 1, 4});
    L.mlp_sconv = req_float(p + "mlp_sconv.weight", {H64, 1, 4});
    if (L.is_swa) {
      L.attn_norm = req_float(p + "attn_norm.weight", {H64});
      L.k_sconv = req_float(p + "attn.k_sconv.weight", {KVD64, 1, 4});
      L.v_sconv = req_float(p + "attn.v_sconv.weight", {KVD64, 1, 4});
    } else {
      L.q_norm = req_float(p + "q_input_norm.weight", {H64});
      L.k_norm = req_float(p + "k_input_norm.weight", {H64});
      L.v_norm = req_float(p + "v_input_norm.weight", {H64});
      L.k_static = req_float(p + "attn.k_sconv.static_w", {4, KVD64});
      L.k_proj = req_float(p + "attn.k_sconv.weight_proj", {int64_t(4) * cfg_.hkv, H64});
      L.v_static = req_float(p + "attn.v_sconv.static_w", {4, KVD64});
      L.v_proj = req_float(p + "attn.v_sconv.weight_proj", {int64_t(4) * cfg_.hkv, H64});
    }
    if (L.has_ctrl) {
      auto load_ctrl = [&](const char* name) {
        const std::string c = p + "controllers." + name + ".";
        Ink2Ctrl ctrl{req_float(c + "down_weight", {cfg_.gate_rank, 2 * H64}),
                      req_float(c + "forget_weight", {H64, cfg_.gate_rank}),
                      req_float(c + "gain_weight", {H64, cfg_.gate_rank})};
        if (ctrl.down->dtype != ctrl.forget->dtype || ctrl.down->dtype != ctrl.gain->dtype)
          fatal("%s: layer %d controller %s weight dtypes differ", path_.c_str(), i, name);
        return ctrl;
      };
      L.ctrl_attn = load_ctrl("attn");
      L.ctrl_k = load_ctrl("k");
      L.ctrl_v = load_ctrl("v");
      L.ctrl_mlp = load_ctrl("mlp");
    }
    L.router_w = req_float(p + "mlp.gate.weight", {E64, H64});
    L.router_bias = req_float(p + "mlp.gate.bias", {E64});
    const std::string m = p + "mlp.experts.";
    L.w13 = req_any(m + "w13_weight");
    if (L.w13->dtype == DType::U8) {
      L.is_mxfp8 = false;
      L.w13 = req_raw(m + "w13_weight", DType::U8, {E64, 2 * I64, H64 / 2});
      L.w13_sc = req_raw(m + "w13_weight_scale", DType::F8E4M3, {E64, 2 * I64, H64 / 16});
      L.w2 = req_raw(m + "w2_weight", DType::U8, {E64, H64, I64 / 2});
      L.w2_sc = req_raw(m + "w2_weight_scale", DType::F8E4M3, {E64, H64, I64 / 16});
      L.w13_outer = make_outer(m + "w13_weight");
      L.w2_outer = make_outer(m + "w2_weight");
    } else if (L.w13->dtype == DType::F8E4M3) {
      L.is_mxfp8 = true;
      L.w13 = req_raw(m + "w13_weight", DType::F8E4M3, {E64, 2 * I64, H64});
      L.w13_sc = req_raw(m + "w13_weight_scale", DType::U8, {E64, 2 * I64, H64 / 32});
      L.w2 = req_raw(m + "w2_weight", DType::F8E4M3, {E64, H64, I64});
      L.w2_sc = req_raw(m + "w2_weight_scale", DType::U8, {E64, H64, I64 / 32});
    } else {
      fatal("%s: unexpected w13 dtype in layer %d", path_.c_str(), i);
    }
  }
  if (cfg_.n_experts > 256 || cfg_.top_k > 8)
    fatal("%s: n_experts=%d top_k=%d exceeds route kernel bounds (256, 8)", path_.c_str(), cfg_.n_experts, cfg_.top_k);

  // Pack [wq_du; wr_du; attn_sink_proj; gate_attn; wk_dv; wv_dv] per layer for fused BF16 MV dispatches (precise only).
  if (!q8_dense_ && (!mem_budget_ || (ws_.total_bytes() + 1050ull * 1024ull * 1024ull <= mem_budget_))) {
    for (int i = 0; i < cfg_.num_layers; ++i) {
      Ink2Layer& L = layers_[size_t(i)];
      const Tensor* segs[6] = {L.wq_du.raw, L.wr_du.raw, L.attn_sink_proj, L.gate_attn.raw, L.wk_dv.raw, L.wv_dv.raw};
      bool ok = (cfg_.hidden & 7) == 0;
      uint32_t off = 0;
      L.qkvrsg_offs[0] = 0;
      for (int s_idx = 0; s_idx < 6; ++s_idx) {
        const Tensor* t = segs[s_idx];
        if (!t || t->dtype != DType::BF16 || (t->shape[0] & 7) != 0 || t->shape[1] != H64) ok = false;
        off += uint32_t(t->shape[0]);
        L.qkvrsg_offs[s_idx + 1] = off;
      }
      if (ok) {
        L.qkvrsg_w = Gpu::get().alloc(size_t(off) * size_t(cfg_.hidden) * 2, "ink2_qkvrsg");
        owned_bufs_.push_back(L.qkvrsg_w);
        uint8_t* dst = L.qkvrsg_w.as<uint8_t>();
        for (int s_idx = 0; s_idx < 6; ++s_idx) {
          const size_t bytes = size_t(segs[s_idx]->shape[0]) * size_t(cfg_.hidden) * 2;
          std::memcpy(dst + size_t(L.qkvrsg_offs[s_idx]) * size_t(cfg_.hidden) * 2, segs[s_idx]->buf.as<uint8_t>(), bytes);
        }
      }
    }
  }

  // The fused qkvrsg rows' Q8 copy for the decode / verify rows (NI_INK2_Q8M): the segments' own Q8 rows end to end
  // (Q8_0 is per row, so this is the packing of the fused BF16 matrix), attn_sink_proj packed here.
  if (q8m_ && cfg_.hidden % 256 == 0) {
    const size_t H = size_t(cfg_.hidden), srow = H / 32 * 2;
    for (int i = 0; i < cfg_.num_layers; ++i) {
      Ink2Layer& L = layers_[size_t(i)];
      const Ink2Mat* segs[6] = {&L.wq_du, &L.wr_du, nullptr, &L.gate_attn, &L.wk_dv, &L.wv_dv};
      const Tensor* sink = L.attn_sink_proj;
      bool ok = sink && sink->dtype == DType::BF16 && (sink->shape[0] & 7) == 0;
      uint32_t off = 0;
      L.qkvrsg_offs[0] = 0;
      for (int k = 0; k < 6; ++k) {
        const uint32_t n = segs[k] ? segs[k]->N : uint32_t(sink->shape[0]);
        if (segs[k] && (!segs[k]->is_q8() || (n & 7) != 0)) ok = false;
        off += n;
        L.qkvrsg_offs[k + 1] = off;
      }
      if (!ok) continue;
      L.qkvrsg_q8.N = off;
      L.qkvrsg_q8.K = uint32_t(H);
      L.qkvrsg_q8.q = Gpu::get().alloc(size_t(off) * H, "ink2_q8_q");
      L.qkvrsg_q8.s = Gpu::get().alloc(size_t(off) * srow, "ink2_q8_s");
      owned_bufs_.push_back(L.qkvrsg_q8.q);
      owned_bufs_.push_back(L.qkvrsg_q8.s);
      q8_bytes += L.qkvrsg_q8.q.size + L.qkvrsg_q8.s.size;
      uint8_t* qd = L.qkvrsg_q8.q.as<uint8_t>();
      uint8_t* sd = L.qkvrsg_q8.s.as<uint8_t>();
      for (int k = 0; k < 6; ++k) {
        const size_t r0 = L.qkvrsg_offs[k], n = L.qkvrsg_offs[k + 1] - r0;
        if (segs[k]) {
          std::memcpy(qd + r0 * H, segs[k]->q.as<uint8_t>(), n * H);
          std::memcpy(sd + r0 * srow, segs[k]->s.as<uint8_t>(), n * srow);
          continue;
        }
        std::vector<float> f(H);
        const uint16_t* src = sink->buf.as<uint16_t>();
        for (size_t r = 0; r < n; ++r) {
          for (size_t c = 0; c < H; ++c) f[c] = bf16_to_f32(src[r * H + c]);
          nh_quant_q8(f.data(), int64_t(H), qd + (r0 + r) * H, sd + (r0 + r) * srow);
        }
      }
    }
  }
  repack_moe();
  Gpu::get().make_resident(ws_.buffers());
  alloc_buffers();
  Gpu::get().make_resident(owned_bufs_);
  logi("loaded %s: Inkling-2-Mini (24 layers / 48 visits, vocab %d%s), %.2f GB text weights in %.0f ms",
       path_.c_str(), cfg_.vocab, q8_dense_ ? ", Q8 dense" : "", (ws_.total_bytes() + q8_bytes) / 1e9, now_ms() - t0);
}

namespace {
// lab10: per 16-bit half, nibble 0 -> sign 15 / eem 9-11, 1 -> 12 / 6-8, 2 -> 13 / 3-5, 3 -> 14 / 0-2 (ink2_dot8d DEC 1;
// inkling/lab10/repack_check.cpp: identical half bits to ink2_bt_dot8 for all 2^32 words; ink2t_unrp inverts it)
inline uint16_t ink2_repack16(uint16_t v) {
  const uint32_t q0 = v & 15u, q1 = (v >> 4) & 15u, q2 = (v >> 8) & 15u, q3 = v >> 12u;
  return uint16_t(((q0 & 8u) << 12) | ((q0 & 7u) << 9) | ((q1 & 8u) << 9) | ((q1 & 7u) << 6) | ((q2 & 8u) << 10) |
                  ((q2 & 7u) << 3) | ((q3 & 8u) << 11) | (q3 & 7u));
}
template <class F>
void ink2_par(size_t n, const F& f) {  // f(begin, end) over 8 threads
  std::vector<std::thread> th;
  const size_t step = (n + 7) / 8;
  for (size_t b = 0; b < n; b += step) th.emplace_back([&f, b, step, n] { f(b, std::min(n, b + step)); });
  for (auto& t : th) t.join();
}
int ink2_env(const char* name, int dflt) { return getenv(name) ? atoi(getenv(name)) : dflt; }
}  // namespace

void Inkling2::repack_moe() {
  static const char* ab = getenv("BENCH_AB");
  repack_ = ink2_knob("NI_INK2_REPACK", 1, 0);
  const bool want_sc = ink2_knob("NI_INK2_SCPAIR", 1, 0) != 0 || (ab && strstr(ab, "NI_INK2_SCPAIR"));
  const bool bt = ink2_knob("NI_INK2_MOEBT", 1, 0) != 0 && cfg_.hidden % 256 == 0 && cfg_.moe_I % 16 == 0 && cfg_.moe_I >= 512;
  if (repack_ && !bt) {
    fprintf(stderr, "[kapitsa] NI_INK2_REPACK needs the bt MoE kernels (NI_INK2_MOEBT, H %% 256, I %% 16): off\n");
    repack_ = 0;
  }
  if (!repack_ && !want_sc) return;
  size_t bytes = 0, bad_sc = 0;
  for (Ink2Layer& L : layers_) {
    if (L.is_mxfp8 || !L.w13) continue;
    for (int k = 0; k < 2 && want_sc; ++k) {  // E4M3 block scales: DEC 2 drops the sign term
      const Tensor* t = k ? L.w2_sc : L.w13_sc;
      const uint8_t* b = t->buf.as<uint8_t>();
      std::atomic<size_t> nb{0};
      ink2_par(size_t(t->numel()), [&](size_t i0, size_t i1) {
        size_t c = 0;
        for (size_t i = i0; i < i1; ++i) c += b[i] >> 7;
        nb += c;
      });
      bad_sc += nb;
    }
    if (!repack_) continue;
    for (int k = 0; k < 2; ++k) {
      const Tensor* t = k ? L.w2 : L.w13;
      const size_t n = size_t(t->numel()) / 2;  // 16-bit halves
      Buffer dst = t->buf;
      if (repack_ == 2) {
        dst = Gpu::get().alloc(n * 2, k ? "ink2_w2_rp" : "ink2_w13_rp");
        owned_bufs_.push_back(dst);
        bytes += n * 2;
      }
      const uint16_t* src = t->buf.as<uint16_t>();
      uint16_t* d = dst.as<uint16_t>();
      ink2_par(n, [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) d[i] = ink2_repack16(src[i]);
      });
      (k ? L.w2_rp : L.w13_rp) = dst;
    }
  }
  sc_pos_ = want_sc && bad_sc == 0;
  if (want_sc && !sc_pos_)
    fprintf(stderr, "[kapitsa] NI_INK2_SCPAIR: %zu NVFP4 block scales have the sign bit set: off\n", bad_sc);
  fprintf(stderr, "[kapitsa] ink2 MoE repack %d (%.2f GB extra), scale pair %s\n", repack_, bytes / 1e9,
          sc_pos_ ? "ok" : "off");
}

// lean: repacked weights this step (NI_INK2_REPACK=1 always; =2 per NI_INK2_LEAN, default 1, BENCH_AB-able)
bool Inkling2::moe_lean() const {
  static const int lean_env = ink2_env("NI_INK2_LEAN", 1);
  return repack_ == 1 || (repack_ == 2 && ink2_ab("NI_INK2_LEAN", lean_env) != 0);
}

// DEC mask for this step: 1 lean (repacked), 2 scale pair (NI_INK2_SCPAIR, sign-0 scales checked), probes
// NI_INK2_LITE (BENCH_AB only; wrong math): 1 / 2 w13 decode-lite / x-lite, 4 / 8 w2 decode-lite / act-lite
uint32_t Inkling2::moe_dec(bool w2) const {
  static const int scp_env = ink2_knob("NI_INK2_SCPAIR", 1, 0);
  static const int lean_mask = ink2_env("NI_INK2_LEANMASK", 3);  // (debug: 1 w13 only, 2 w2 only; needs REPACK=2)
  const int lite = ink2_ab("NI_INK2_LITE", 0) >> (w2 ? 2 : 0);
  return (moe_lean() && (lean_mask >> (w2 ? 1 : 0) & 1) ? 1u : 0u) | (sc_pos_ && ink2_ab("NI_INK2_SCPAIR", scp_env) != 0 ? 2u : 0u) |
         (lite & 1 ? 4u : 0u) | (lite & 2 ? 8u : 0u);
}

bool Inkling2::enc_q8(Encoder& e, const Ink2Mat& q, const Buffer& x, const Buffer& y, int M, uint32_t N, bool add) {
  if (!q8m_ || !q.is_q8() || q.K % 256 != 0 || M < 1 || M > 8) return false;
  const Ink2Q8P p{uint32_t(M), N, q.K, q.K, N, add ? 1u : 0u, 0u, {}};
  // NI_INK2_Q8R=R (default 2; 4 = off, 1): R weight rows per simdgroup for one-row matvecs with N <= NI_INK2_Q8R_MAXN
  // (8k decode: 24.30 -> 24.07 ms/step GPU at R = 2, R = 1 worse; same per-row sums, bit-identical)
  static const uint32_t r1 = getenv("NI_INK2_Q8R") ? uint32_t(atoi(getenv("NI_INK2_Q8R"))) : 2u;
  static const uint32_t r1n = getenv("NI_INK2_Q8R_MAXN") ? uint32_t(atoi(getenv("NI_INK2_Q8R_MAXN"))) : 4096u;
  const uint32_t R = (M == 1 && N <= r1n && (r1 == 1u || r1 == 2u)) ? r1 : 4u;
  e.use(ink2_pipe(R == 4u ? "ink2_q8_mvm" + std::to_string(M) : "ink2_q8_mv1r" + std::to_string(R)));
  e.buf(0, x);
  e.buf(1, q.q);
  e.buf(2, q.s);
  e.buf(3, y);
  e.val(4, p);
  e.dispatch((N + 4 * R - 1) / (4 * R), 1, 1, 128);
  return true;
}

void Inkling2::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t T = size_t(tmax_), H = size_t(cfg_.hidden);
  const size_t QD = size_t(cfg_.hq) * size_t(cfg_.head_dim);
  const size_t KVD = size_t(cfg_.hkv) * size_t(cfg_.head_dim);
  const size_t V = size_t(cfg_.vocab);
  auto alloc = [&](size_t bytes, const char* lbl) {
    Buffer b = g.alloc(std::max<size_t>(bytes, 16), lbl);
    owned_bufs_.push_back(b);
    return b;
  };
  x_ = alloc(T * H * 4, "ink2_x");
  x_pre_ = alloc(T * H * 4, "ink2_x_pre");
  inp_sa_ = alloc(T * H * 4, "ink2_inp_sa");
  ffn_inp_ = alloc(T * H * 4, "ink2_ffn_inp");
  h_norm_ = alloc(T * H * 4, "ink2_h_norm");
  for (int i = 0; i < 6; ++i) blocks_[i] = alloc(T * H * 4, "ink2_block");
  for (int i = 0; i < 4; ++i) prelude_blocks_[i] = alloc(T * H * 4, "ink2_pre_block");
  ar_out_ = alloc(4 * T * H * 4, "ink2_ar");
  msg_out_ = alloc(4 * T * H * 4, "ink2_msg");
  mg_cat_ = alloc(3 * 16 * 2 * H * 4, "ink2_mg_cat");
  mg_lat_ = alloc(3 * 16 * size_t(cfg_.gate_rank) * 4, "ink2_mg_lat");
  mg_catb_ = alloc(T * 2 * H * 4, "ink2_mg_catb");
  mg_latb_ = alloc(T * size_t(cfg_.gate_rank) * 4, "ink2_mg_latb");
  mg_gb_ = alloc(T * H * 4, "ink2_mg_gb");
  n0_ = alloc(T * H * 4, "ink2_n0");
  n1_ = alloc(T * H * 4, "ink2_n1");
  n2_ = alloc(T * H * 4, "ink2_n2");
  q_ = alloc(T * QD * 4, "ink2_q");
  k_ = alloc(T * KVD * 4, "ink2_k");
  v_ = alloc(T * KVD * 4, "ink2_v");
  r_ = alloc(T * size_t(cfg_.hq) * size_t(cfg_.d_rel) * 4, "ink2_r");
  sinks_ = alloc(T * size_t(cfg_.hq) * 4, "ink2_sinks");
  gate_attn_ = alloc(T * QD * 4, "ink2_gate_attn");
  att_out_ = alloc(T * QD * 4, "ink2_att_out");
  a_out_ = alloc(T * H * 4, "ink2_a_out");
  dyn_ = alloc(T * 4 * size_t(cfg_.hkv) * 4, "ink2_dyn");
  rb_ = alloc(T * size_t(cfg_.hq) * size_t(cfg_.rel_extent) * 4, "ink2_rb");
  part_m_ = alloc(size_t(cfg_.hq) * kAttnChunkCap * kMaxVerify * 4, "ink2_part_m");
  part_l_ = alloc(size_t(cfg_.hq) * kAttnChunkCap * kMaxVerify * 4, "ink2_part_l");
  part_o_ = alloc(size_t(cfg_.hq) * kAttnChunkCap * kMaxVerify * size_t(cfg_.head_dim) * 4, "ink2_part_o");
  // (+ 16 bytes: the router's arrival counter, ink2_mv_sk_route)
  attn_cnt_ = alloc(size_t(kMaxVerify) * size_t(cfg_.hkv) * 4 + 16, "ink2_attn_cnt");
  memset(attn_cnt_.base + attn_cnt_.offset, 0, attn_cnt_.size);
  h_mlp_ = alloc(T * H * 4, "ink2_h_mlp");
  x_mlp_ = alloc(T * H * 4, "ink2_x_mlp");
  rlog_ = alloc(T * size_t(cfg_.n_experts) * 4, "ink2_rlog");
  eid_ = alloc(T * size_t(cfg_.top_k) * 4, "ink2_eid");
  ew_ = alloc(T * size_t(cfg_.top_k) * 4, "ink2_ew");
  moe_act_ = alloc(T * size_t(cfg_.top_k) * size_t(cfg_.moe_I) * 4, "ink2_moe_act");
  moe_out_ = alloc(T * H * 4, "ink2_moe_out");
  if (tensor_pf_) {
    const size_t P = T * size_t(cfg_.top_k);
    const size_t max_tiles = P / 16 + size_t(cfg_.n_experts) + 2;
    moe_sorted_ = alloc(P * 4, "ink2_moe_sorted");
    moe_rank_ = alloc(P * 4, "ink2_moe_rank");
    moe_tiles_ = alloc(max_tiles * 16, "ink2_moe_tiles");
    moe_ntiles_ = alloc(16, "ink2_moe_ntiles");
    moe_tmp_ = alloc(P * std::max(H, size_t(2) * size_t(cfg_.moe_I)) * 4, "ink2_moe_tmp");
    moe_act_h_ = alloc(P * size_t(cfg_.moe_I) * 2, "ink2_moe_act_h");
  }
  logits_ = alloc(V * 4, "ink2_logits");
  cand_v_ = alloc((V / 2048 + 2) * 64 * 4, "ink2_cand_v");
  cand_i_ = alloc((V / 2048 + 2) * 64 * 4, "ink2_cand_i");
  sprob_ = alloc(64 * 4, "ink2_sprob");
  samp_ = alloc(64 * 4, "ink2_samp");
}

// ---- sequences, KV caches, checkpoints ----
std::unique_ptr<Seq> Inkling2::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "ink2_tokens");
  const size_t nv = cfg_.schedule.size();
  s->kc.resize(nv);
  s->vc.resize(nv);
  if (ink2_kt_alloc()) s->kt.resize(nv);
  s->conv.push_back(g.alloc(conv_bytes_per_seq(cfg_), "ink2_conv"));
  g.make_resident(s->conv);
  reset(*s);
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token Inkling-2 KV cache", cap);
  return s;
}

bool Inkling2::ensure_rings(Seq& s) {
  if (s.cap <= 0 && !s.kc[0]) {
    // Rings may be allocated alongside first linear KV capacity inside reserve(); if all visits were SWA they are handled below.
  }
  Gpu& g = Gpu::get();
  const size_t row = size_t(cfg_.hkv) * size_t(cfg_.head_dim) * 2;
  const size_t nv = cfg_.schedule.size();
  size_t need = 0;
  for (size_t v = 0; v < nv; ++v)
    if (cfg_.is_local[size_t(cfg_.schedule[v])] && !s.kc[v])
      need += 2 * row * size_t(ring_rows_) + (s.kt.empty() ? 0 : row * ink2_kt_rows(size_t(ring_rows_)));
  if (!need) return true;
  if (mem_budget_ && g.allocated_bytes() + need > mem_budget_) return false;
  std::vector<Buffer> add;
  for (size_t v = 0; v < nv; ++v)
    if (cfg_.is_local[size_t(cfg_.schedule[v])] && !s.kc[v]) {
      s.kc[v] = g.alloc(row * size_t(ring_rows_), "ink2_kv_ring");
      s.vc[v] = g.alloc(row * size_t(ring_rows_), "ink2_kv_ring");
      memset(s.kc[v].ptr(), 0, s.kc[v].size);
      memset(s.vc[v].ptr(), 0, s.vc[v].size);
      add.push_back(s.kc[v]);
      add.push_back(s.vc[v]);
      if (!s.kt.empty()) {
        s.kt[v] = g.alloc(row * ink2_kt_rows(size_t(ring_rows_)), "ink2_kt_ring");
        memset(s.kt[v].ptr(), 0, s.kt[v].size);
        add.push_back(s.kt[v]);
      }
    }
  g.make_resident(add);
  return true;
}

size_t Inkling2::kv_grow_unit() const {
  return size_t(cfg_.hkv) * size_t(cfg_.head_dim) * 2u;
}

size_t Inkling2::kv_grow_held(const Seq& s) const {
  return kvgrow::held(s.cap, kv_bytes_per_pos(), kv_grow_unit());
}

bool Inkling2::reserve(Seq& s, int need) {
  if (need <= s.cap) return ensure_rings(s);
  if (need > s.limit) return false;
  Gpu& g = Gpu::get();
  const size_t per = kv_per_pos(cfg_);
  const size_t row = kv_grow_unit();
  const size_t nv = cfg_.schedule.size();
  size_t rings = 0;
  for (size_t v = 0; v < nv; ++v)
    if (cfg_.is_local[size_t(cfg_.schedule[v])] && !s.kc[v]) rings += 2 * row * size_t(ring_rows_);
  const size_t alloc = g.allocated_bytes();
  const size_t room = !mem_budget_ ? SIZE_MAX : (mem_budget_ > alloc + rings ? mem_budget_ - alloc - rings : 0);
  const kvgrow::Plan p = kvgrow::plan(s.cap, need, s.limit, per, row, room);
  if (!p.target) return false;
  const int target = p.target;

  const size_t nb = row * size_t(target);
  const int old = s.cap;
  std::vector<Buffer> olds, news;
  std::unique_ptr<Encoder> e;
  size_t batch = 0;
  auto flush = [&] {
    if (e) e->commit_and_wait();
    e.reset();
    for (Buffer& b : olds) g.release(b);
    olds.clear();
    batch = 0;
  };
  for (size_t v = 0; v < nv; ++v) {
    if (cfg_.is_local[size_t(cfg_.schedule[v])]) continue;
    for (Buffer* b : {&s.kc[v], &s.vc[v], s.kt.empty() ? nullptr : &s.kt[v]}) {
      if (!b) continue;
      const bool tiled = !s.kt.empty() && b == &s.kt[v];  // (whole 32-row tiles; old tiles keep their offsets)
      const size_t nbb = tiled ? row * ink2_kt_rows(size_t(target)) : nb;
      if (batch && batch + nbb > p.batch) flush();
      Buffer nbuf = g.alloc(nbb, tiled ? "ink2_kt_global" : "ink2_kv_global");
      if (tiled) memset(nbuf.ptr(), 0, nbuf.size);  // (tiles zeroed as the lab ran them; kc / vc unzeroed as #1396)
      if (old > 0 && *b) {
        if (!e) e = g.begin();
        enc_copy_buf(*e, *b, nbuf, tiled ? row * ink2_kt_rows(size_t(old)) : size_t(old) * row);
      }
      if (*b) olds.push_back(*b);
      *b = nbuf;
      news.push_back(nbuf);
      batch += nbb;
    }
  }
  flush();
  s.cap = target;
  g.make_resident(news);
  return ensure_rings(s);
}

void Inkling2::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc, &s.kt})
    for (Buffer& b : *v) {
      g.release(b);
      b = Buffer{};
    }
  s.cap = 0;
}

void Inkling2::reset(Seq& s) {
  if (!s.conv.empty() && s.conv[0]) memset(s.conv[0].ptr(), 0, s.conv[0].size);
}

Checkpoint Inkling2::new_checkpoint() { return Checkpoint{}; }

void Inkling2::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Gpu& g = Gpu::get();
  c.pos = -1;
  const size_t nv = cfg_.schedule.size();
  if (c.conv.empty() && !s.conv.empty() && s.conv[0]) {
    if (mem_budget_ && g.allocated_bytes() + s.conv[0].size > mem_budget_) return;
    c.conv.push_back(g.alloc(s.conv[0].size, "ink2_ckpt_conv"));
    g.make_resident(c.conv);
  }
  if (!c.conv.empty() && !s.conv.empty() && s.conv[0]) enc_copy_buf(e, s.conv[0], c.conv[0]);
  if (c.rec.size() != 2 * nv) c.rec.assign(2 * nv, Buffer{});
  for (size_t v = 0; v < nv; ++v) {
    const bool swa = cfg_.is_local[size_t(cfg_.schedule[v])];
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = kv ? s.vc[v] : s.kc[v];
      Buffer& dst = c.rec[2 * v + size_t(kv)];
      if (!swa || !src) {
        if (dst) g.release(dst), dst = Buffer{};
        continue;
      }
      if (!dst) {
        if (mem_budget_ && g.allocated_bytes() + src.size > mem_budget_) return;
        dst = g.alloc(src.size, "ink2_ckpt_ring");
        g.make_resident({dst});
      }
      enc_copy_buf(e, src, dst);
    }
  }
  c.pos = pos;
}

void Inkling2::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  if (!c.conv.empty() && c.conv[0]) {
    if (s.conv.empty() || !s.conv[0] || s.conv[0].size < c.conv[0].size)
      fatal("inkling2: restore missing or undersized conv destination buffer");
    enc_copy_buf(e, c.conv[0], s.conv[0], c.conv[0].size);
  }
  const size_t nv = cfg_.schedule.size();
  bool need_rings = false;
  for (size_t v = 0; v < nv && 2 * v + 1 < c.rec.size(); ++v) {
    if ((c.rec[2 * v] && (v >= s.kc.size() || !s.kc[v])) ||
        (c.rec[2 * v + 1] && (v >= s.vc.size() || !s.vc[v]))) {
      need_rings = true;
      break;
    }
  }
  if (need_rings && !ensure_rings(s)) fatal("inkling2: restore could not allocate missing ring buffers");
  for (size_t v = 0; v < nv && 2 * v + 1 < c.rec.size(); ++v) {
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = c.rec[2 * v + size_t(kv)];
      if (!src) continue;
      const auto& dsts = kv ? s.vc : s.kc;
      if (v >= dsts.size() || !dsts[v] || dsts[v].size < src.size)
        fatal("inkling2: restore missing or undersized ring destination buffer for visit %zu (%s)", v, kv ? "v" : "k");
      enc_copy_buf(e, src, dsts[v], src.size);
      if (!kv && v < s.kt.size() && s.kt[v]) enc_kt_tile(e, s, v, 0, ring_rows_);  // (the ring's tiled K again)
    }
  }
}

void Inkling2::enc_kt_tile(Encoder& e, Seq& s, size_t v, int a, int b) {
  if (b <= a) return;
  const Ink2KtP p{uint32_t(a), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim)};
  e.use(ink2_pipe("ink2_kt_tile"));
  e.buf(0, s.kc[v]);
  e.buf(1, s.kt[v]);
  e.val(2, p);
  e.dispatch(uint32_t(b - a), uint32_t(cfg_.hkv), 1, uint32_t(cfg_.head_dim));
}

std::string Inkling2::disk_format() const {
  char b[256];
  snprintf(b, sizeof b, "inkling2; %zu visits; K,V f16 [cap][%d*%d]; SWA rings %d; conv %zu B",
           cfg_.schedule.size(), cfg_.hkv, cfg_.head_dim, ring_rows_, conv_bytes_per_seq(cfg_));
  return b;
}

void Inkling2::disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const {
  (void)mtp;
  const size_t nv = cfg_.schedule.size();
  const size_t rb = size_t(cfg_.hkv) * size_t(cfg_.head_dim) * 2;
  for (size_t v = 0; v < nv; ++v) {
    if (cfg_.is_local[size_t(cfg_.schedule[v])]) continue;
    out.push_back({s.kc[v], 1, rb});
    out.push_back({s.vc[v], 1, rb});
  }
}

void Inkling2::disk_state(const Seq& s, const Checkpoint* ck, std::vector<Buffer>& out) const {
  out.push_back(!ck ? (s.conv.empty() ? Buffer{} : s.conv[0]) : (ck->conv.empty() ? Buffer{} : ck->conv[0]));
  const size_t nv = cfg_.schedule.size();
  for (size_t v = 0; v < nv; ++v) {
    if (!cfg_.is_local[size_t(cfg_.schedule[v])]) continue;
    for (size_t kv = 0; kv < 2; ++kv) {
      out.push_back(!ck ? (kv ? s.vc[v] : s.kc[v])
                        : (2 * v + kv < ck->rec.size() ? ck->rec[2 * v + kv] : Buffer{}));
    }
  }
}

// ---- primitives ----
void Inkling2::enc_mm(Encoder& e, const Tensor& w, const Buffer& x, const Buffer& y, int M, uint32_t N_limit,
                      bool add, bool exact_f32) {
  const uint32_t N = N_limit ? N_limit : uint32_t(w.shape[0]);
  const uint32_t K = uint32_t(w.shape[1]);
  if (exact_f32 && M > 8 && w.dtype == DType::BF16 && (K & 127u) == 0u) {
    const Ink2MmP p{uint32_t(M), N, K, K, N, 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_bf16_mv8"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, y);
    e.val(3, p);
    e.dispatch((N + 15) / 16, (uint32_t(M) + 7) / 8, 1, 128);
    return;
  }
  static const uint32_t sk_max = ink2_knob("NI_INK2_MVSK", 256, 0);
  if (M == 1 && N <= sk_max && (K & 3u) == 0u && (w.dtype == DType::BF16 || w.dtype == DType::F32)) {
    const Ink2MmP p{1u, N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_mv_sk"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, y);
    e.val(3, p);
    e.dispatch(N, 1, 1, 128);
    return;
  }
  // 2..8 rows of the same small-N matvecs (2-slot decode / verify rows; lab6): ink2_mv_sk_m<M>, each row bit-identical
  // to its one-row ink2_mv_sk result (8k 2-slot: 64 BF16 calls 2.68 ms as bf16_mvm2). NI_INK2_MVSKM=0: the old path
  static const bool skm = ink2_knob("NI_INK2_MVSKM", 1, 0) != 0;
  if (skm && M >= 2 && M <= 8 && N <= sk_max && (K & 3u) == 0u && (w.dtype == DType::BF16 || w.dtype == DType::F32)) {
    const Ink2MmP p{uint32_t(M), N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_mv_sk_m" + std::to_string(M)));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, y);
    e.val(3, p);
    e.dispatch(N, 1, 1, 128);
    return;
  }
  if (M <= mv_rows() || exact_f32) {
    for (int m0 = 0; m0 < M; m0 += 8) {
      const uint32_t mc = uint32_t(std::min(8, M - m0));
      const Ink2MmP p{mc, N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
      const bool aligned_k = w.dtype == DType::F32 ? ((K & 3u) == 0u) : ((K & 7u) == 0u);
      if (mc >= 2u && aligned_k && mvm()) {
        e.use(ink2_pipe("ink2_bf16_mvm" + std::to_string(mc)));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 15) / 16, 1, 1, 128);
      } else if (mc <= 4u && aligned_k) {
        e.use(ink2_pipe("ink2_bf16_mv4"));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 15) / 16, 1, 1, 128);
      } else if (w.dtype == DType::BF16 && (K & 127u) == 0u) {
        e.use(ink2_pipe("ink2_bf16_mv8"));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 15) / 16, 1, 1, 128);
      } else {
        e.use(ink2_pipe("ink2_bf16_mv"));
        e.buf(0, x.view(size_t(m0) * K * 4, size_t(mc) * K * 4));
        e.buf(1, w.buf);
        e.buf(2, y.view(size_t(m0) * N * 4, size_t(mc) * N * 4));
        e.val(3, p);
        e.dispatch((N + 3) / 4, 1, 1, 128);
      }
    }
  } else if (tensor_pf_ && w.dtype == DType::BF16 && K % 32 == 0) {
    // Split-K for small N (lab5): N <= 64 ran (M / 64) threadgroups through all of K (msg gate down [64, 4608] at
    // M = 500: 0.97 ms); NI_INK2_SK = splits (default 8, 0 = off). Partials in moe_tmp_ (free outside the MoE).
    static const uint32_t sk = ink2_knob("NI_INK2_SK", 8, 0);
    if (sk > 1 && N <= 64 && K >= 1024 && K % (sk * 32) == 0 && M >= 64 && !e.in_group() &&
        size_t(sk) * M * N * 4 <= moe_tmp_.size) {
      const Ink2tGemmP p{uint32_t(M), N, K, K, N, 0u, K / sk, 0u, 0u};
      e.use(base_pipe("ink2_t_bf16_sk_m64"));
      e.buf(0, x);
      e.buf(1, w.buf);
      e.buf(3, moe_tmp_);
      e.val(4, p);
      e.dispatch((N + 63) / 64, (uint32_t(M) + 63) / 64, sk, 128);
      const struct { uint32_t n, S, add; } sp{uint32_t(M) * N, sk, add ? 1u : 0u};
      e.use(ink2_pipe("ink2_sum_parts"));
      e.buf(0, moe_tmp_);
      e.buf(1, y);
      e.val(2, sp);
      e.dispatch((sp.n + 255) / 256, 1, 1, 256);
      return;
    }
    const Ink2tGemmP p{uint32_t(M), N, K, K, N, add ? 1u : 0u, 0u, 0u, 0u};
    e.use(base_pipe("ink2_t_bf16_m64"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, w.buf);
    e.buf(3, y);
    e.val(4, p);
    e.buf(5, moe_tiles_);
    e.buf(6, moe_ntiles_);
    e.buf(7, moe_sorted_);
    e.buf(8, x);
    e.dispatch((N + 63) / 64, (uint32_t(M) + 63) / 64, 1, 128);
  } else {
    const Ink2MmP p{uint32_t(M), N, K, K, N, w.dtype == DType::F32 ? 1u : 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_bf16_mm"));
    e.buf(0, x);
    e.buf(1, w.buf);
    e.buf(2, y);
    e.val(3, p);
    e.dispatch((N + 63) / 64, (uint32_t(M) + 31) / 32, 1, 128);
  }
}

void Inkling2::enc_mm(Encoder& e, const Ink2Mat& w, const Buffer& x, const Buffer& y, int M, uint32_t N_limit,
                      bool add, bool exact_f32) {
  if (w.gguf.data) {
    const uint32_t N = N_limit ? N_limit : w.N;
    const Ink2GgufMmP p{uint32_t(M), N, w.K, w.gguf.type, w.gguf.row_bytes, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_gguf_mm"));
    e.buf(0, x); e.buf(1, w.gguf.data); e.buf(2, y); e.val(3, p);
    e.dispatch((N + 3) / 4, uint32_t(M), 1, 128);
    return;
  }
  if (!exact_f32 && M <= 8 && enc_q8(e, w, x, y, M, N_limit ? N_limit : w.N, add)) return;
  if (w.raw && (!w.is_q8() || M > 8)) {  // not Q8-packed, or prefill rows on the BF16 copy kept beside it (NI_INK2_Q8_BF16)
    enc_mm(e, *w.raw, x, y, M, N_limit, add, exact_f32);
    return;
  }
  const uint32_t N = N_limit ? N_limit : w.N;
  const uint32_t K = w.K;
  if (M == 1) {
    const Ink2MmP p{1u, N, K, K, N, 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_q8_mv1"));
    e.buf(0, x);
    e.buf(1, w.q);
    e.buf(2, w.s);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((N + 7) / 8, 1, 1, 128);
  } else if (M <= 32 || exact_f32) {
    const Ink2MmP p{uint32_t(M), N, K, K, N, 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_q8_mv"));
    e.buf(0, x);
    e.buf(1, w.q);
    e.buf(2, w.s);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((N + 3) / 4, (uint32_t(M) + 7) / 8, 1, 128);
  } else {
    const Ink2MmP p{uint32_t(M), N, K, K, N, 0u, add ? 1u : 0u};
    e.use(ink2_pipe("ink2_q8_mm"));
    e.buf(0, x);
    e.buf(1, w.q);
    e.buf(2, w.s);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((N + 63) / 64, (uint32_t(M) + 31) / 32, 1, 128);
  }
}

void Inkling2::enc_norm(Encoder& e, const Buffer& x, const Tensor* w, const Buffer& out, int T, uint32_t dim) {
  const Ink2NormP p{uint32_t(T), dim, w ? 1u : 0u, (w && w->dtype == DType::F32) ? 1u : 0u, cfg_.rms_eps};
  e.use(ink2_pipe("ink2_rmsnorm"));
  e.buf(0, x);
  e.buf(1, w ? w->buf : x);
  e.buf(2, out);
  e.val(3, p);
  e.dispatch(uint32_t(T), 1, 1, 512);
}

void Inkling2::enc_attn_res(Encoder& e, const Buffer* const* states, int M, const void* query_ptr, bool q_f32,
                            uint32_t Q, const Buffer& out, int T) {
  if (M < 1 || M > 6 || Q < 1 || Q > 4) fatal("inkling2: attn_res M=%d Q=%u out of range", M, Q);
  const Ink2AttnResP p{uint32_t(T),
                       uint32_t(cfg_.hidden),
                       uint32_t(M),
                       Q,
                       q_f32 ? 1u : 0u,
                       cfg_.rms_eps,
                       1.0f / std::sqrt(float(cfg_.hidden))};
  static const bool res1 = ink2_knob("NI_INK2_RES1", 1, 0) != 0;
  static const bool resmq = ink2_knob("NI_INK2_RESMQ", 1, 0) != 0;
  e.use(ink2_pipe(!res1  ? std::string("ink2_attn_res")
                  : resmq ? "ink2_attn_res_m" + std::to_string(M) + "q" + std::to_string(Q)
                          : std::string("ink2_attn_res1")));
  for (int m = 0; m < 6; ++m) e.buf(m, m < M ? *states[m] : *states[0]);
  e.buf(6, *static_cast<const Buffer*>(query_ptr));
  e.buf(7, out);
  e.val(8, p);
  e.dispatch(uint32_t(T), 1, 1, 512);
}

struct Ink2ResNormP {
  uint32_t src[3], has_w[3], w_f32[3];
};

bool Inkling2::enc_attn_res_norm(Encoder& e, const Buffer* const* states, int M, const void* query_ptr, bool q_f32,
                                 uint32_t Q, const Buffer& out, int T, int nn, const uint32_t* src,
                                 const Tensor* const* w, const Buffer* const* nout) {
  // NI_INK2_NFOLD=0: separate ink2_rmsnorm stages (lab7: the fold is bit-identical)
  // (bit mask: 1 attention input, 2 MLP input, 4 final norm)
  static const int nfold_env = ink2_knob("NI_INK2_NFOLD", 7, 0);
  const int nfold = ink2_ab("NI_INK2_NFOLD", nfold_env);
  static const bool res1 = ink2_knob("NI_INK2_RES1", 1, 0) != 0;
  static const bool resmq = ink2_knob("NI_INK2_RESMQ", 1, 0) != 0;
  const int site = nout[0] == &n0_ ? 1 : nout[0] == &x_mlp_ ? 2 : 4;
  if (!(nfold & site) || !res1 || !resmq || M < 1 || M > 6 || (Q != 1 && Q != 3) || (nn != 1 && nn != 3) ||
      cfg_.hidden > 5 * 512)
    return false;
  const Ink2AttnResP p{uint32_t(T), uint32_t(cfg_.hidden), uint32_t(M), Q, q_f32 ? 1u : 0u, cfg_.rms_eps,
                       1.0f / std::sqrt(float(cfg_.hidden))};
  Ink2ResNormP np{};
  for (int j = 0; j < nn; ++j) {
    np.src[j] = src[j];
    np.has_w[j] = w[j] ? 1u : 0u;
    np.w_f32[j] = (w[j] && w[j]->dtype == DType::F32) ? 1u : 0u;
  }
  e.use(ink2_pipe("ink2_attn_res_n_m" + std::to_string(M) + "q" + std::to_string(Q) + "n" + std::to_string(nn)));
  for (int m = 0; m < 6; ++m) e.buf(m, m < M ? *states[m] : *states[0]);
  e.buf(6, *static_cast<const Buffer*>(query_ptr));
  e.buf(7, out);
  e.val(8, p);
  for (int j = 0; j < 3; ++j) {
    const int k = j < nn ? j : 0;
    e.buf(9 + j, w[k] ? w[k]->buf : out);
    e.buf(12 + j, *nout[k]);
  }
  e.val(15, np);
  e.dispatch(uint32_t(T), 1, 1, 512);
  return true;
}

void Inkling2::enc_msg_gate(Encoder& e, const Ink2Ctrl& c, const Buffer& h, const Buffer& m, const Buffer& out, int T) {
  const Ink2MsgGateP p{uint32_t(T), uint32_t(cfg_.hidden), uint32_t(cfg_.gate_rank),
                       c.down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
  if (T <= 16 && msg_fused()) {
    enc_msg_gate_f(e, c, h, m, mg_lat_, out, T, 0);  // (outside a group: the grouped encoder puts a barrier between)
    enc_msg_gate_f(e, c, h, m, mg_lat_, out, T, 1);
    return;
  }
  if (T <= 16) {
    const uint32_t H = uint32_t(cfg_.hidden), R = uint32_t(cfg_.gate_rank);
    e.use(ink2_pipe("ink2_msg_gate_prep_s"));
    e.buf(0, h); e.buf(1, m); e.buf(2, mg_cat_); e.val(3, p);
    e.dispatch(uint32_t(T), 1, 1, 512);
    e.use(ink2_pipe("ink2_msg_gate_down_s"));
    e.buf(0, mg_cat_); e.buf(1, c.down->buf); e.buf(2, mg_lat_); e.val(3, p);
    e.dispatch((R + 3) / 4, (uint32_t(T) + 3) / 4, 1, 128);
    e.use(ink2_pipe("ink2_msg_gate_up_mix_s"));
    e.buf(0, h); e.buf(1, m); e.buf(2, mg_lat_);
    e.buf(3, c.forget->buf); e.buf(4, c.gain->buf); e.buf(5, out); e.val(6, p);
    e.dispatch((H + 127) / 128, (uint32_t(T) + 7) / 8, 1, 128);
    return;
  }
  if (msg_mm()) {
    // T > 16 as GEMMs (the per-token kernel re-reads the gate's 1.2 MB of weights for every row: 14 ms per 2112-row
    // call on the M1 Max): rmsnorm both inputs -> [T, 2H], down -> [T, R], forget / gain -> [T, H], tanh mix.
    const uint32_t H = uint32_t(cfg_.hidden);
    e.use(ink2_pipe("ink2_msg_gate_prep_s"));
    e.buf(0, h); e.buf(1, m); e.buf(2, mg_catb_); e.val(3, p);
    e.dispatch(uint32_t(T), 1, 1, 512);
    enc_mm(e, *c.down, mg_catb_, mg_latb_, T);
    enc_mm(e, *c.forget, mg_latb_, mg_catb_, T);  // (cat is consumed: forget logits go there; out may alias h)
    enc_mm(e, *c.gain, mg_latb_, mg_gb_, T);
    const Ink2SituP mp{uint32_t(T) * H};
    e.use(ink2_pipe("ink2_msg_gate_mix"));
    e.buf(0, h); e.buf(1, m); e.buf(2, mg_catb_); e.buf(3, mg_gb_); e.buf(4, out); e.val(5, mp);
    e.dispatch((mp.n + 255) / 256, 1, 1, 256);
    return;
  }
  e.use(ink2_pipe("ink2_msg_gate"));
  e.buf(0, h);
  e.buf(1, m);
  e.buf(2, c.down->buf);
  e.buf(3, c.forget->buf);
  e.buf(4, c.gain->buf);
  e.buf(5, out);
  e.val(6, p);
  e.dispatch(uint32_t(T), 1, 1, 512);
}

// stage 0: fused norms + down -> lat; stage 1: forget / gain up + tanh mix -> out (out may alias h)
void Inkling2::enc_msg_gate_f(Encoder& e, const Ink2Ctrl& c, const Buffer& h, const Buffer& m, const Buffer& lat,
                              const Buffer& out, int T, int stage) {
  const uint32_t H = uint32_t(cfg_.hidden), R = uint32_t(cfg_.gate_rank);
  const Ink2MsgGateP p{uint32_t(T), H, R, c.down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
  if (stage == 0) {
    e.use(ink2_pipe("ink2_msg_gate_down_f"));
    e.buf(0, h); e.buf(1, m); e.buf(2, c.down->buf); e.buf(3, lat); e.val(4, p);
    e.dispatch(R, (uint32_t(T) + 3) / 4, 1, 256);
  } else {
    e.use(ink2_pipe("ink2_msg_gate_up_mix_c"));
    e.buf(0, h); e.buf(1, m); e.buf(2, lat); e.buf(3, c.forget->buf); e.buf(4, c.gain->buf); e.buf(5, out);
    e.val(6, p);
    e.dispatch((H + 31) / 32, (uint32_t(T) + 7) / 8, 1, 256);
  }
}

void Inkling2::enc_msg_gate3(Encoder& e, const Ink2Layer& L, const Buffer* const* in_ptrs, const Buffer& m0,
                             const Buffer& m1, const Buffer& m2, int T) {
  const Ink2Ctrl* cs[3] = {&L.ctrl_attn, &L.ctrl_k, &L.ctrl_v};
  const Buffer* ms[3] = {&m0, &m1, &m2};
  const Buffer* outs[3] = {&n0_, &n1_, &n2_};
  if (T <= 16 && msg_fused()) {
    const size_t lat_sz = size_t(T) * uint32_t(cfg_.gate_rank) * 4;
    for (int stage = 0; stage < 2; ++stage) {
      e.group_begin();
      for (int i = 0; i < 3; ++i)
        enc_msg_gate_f(e, *cs[i], *in_ptrs[i], *ms[i], mg_lat_.view(size_t(i) * lat_sz, lat_sz), *outs[i], T, stage);
      e.group_end();
    }
    return;
  }
  if (T <= 16) {
    const uint32_t H = uint32_t(cfg_.hidden), R = uint32_t(cfg_.gate_rank);
    const size_t cat_sz = size_t(T) * 2 * H * 4, lat_sz = size_t(T) * R * 4;
    e.group_begin();
    for (int i = 0; i < 3; ++i) {
      const Ink2MsgGateP p{uint32_t(T), H, R, cs[i]->down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
      e.use(ink2_pipe("ink2_msg_gate_prep_s"));
      e.buf(0, *in_ptrs[i]); e.buf(1, *ms[i]); e.buf(2, mg_cat_.view(size_t(i) * cat_sz, cat_sz)); e.val(3, p);
      e.dispatch(uint32_t(T), 1, 1, 512);
    }
    e.group_end();
    e.group_begin();
    for (int i = 0; i < 3; ++i) {
      const Ink2MsgGateP p{uint32_t(T), H, R, cs[i]->down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
      e.use(ink2_pipe("ink2_msg_gate_down_s"));
      e.buf(0, mg_cat_.view(size_t(i) * cat_sz, cat_sz)); e.buf(1, cs[i]->down->buf);
      e.buf(2, mg_lat_.view(size_t(i) * lat_sz, lat_sz)); e.val(3, p);
      e.dispatch((R + 3) / 4, (uint32_t(T) + 3) / 4, 1, 128);
    }
    e.group_end();
    e.group_begin();
    for (int i = 0; i < 3; ++i) {
      const Ink2MsgGateP p{uint32_t(T), H, R, cs[i]->down->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps};
      e.use(ink2_pipe("ink2_msg_gate_up_mix_s"));
      e.buf(0, *in_ptrs[i]); e.buf(1, *ms[i]); e.buf(2, mg_lat_.view(size_t(i) * lat_sz, lat_sz));
      e.buf(3, cs[i]->forget->buf); e.buf(4, cs[i]->gain->buf); e.buf(5, *outs[i]); e.val(6, p);
      e.dispatch((H + 127) / 128, (uint32_t(T) + 7) / 8, 1, 128);
    }
    e.group_end();
    return;
  }
  if (msg_mm()) {  // shared staging buffers: one gate after another
    for (int i = 0; i < 3; ++i) enc_msg_gate(e, *cs[i], *in_ptrs[i], *ms[i], *outs[i], T);
    return;
  }
  e.group_begin();
  for (int i = 0; i < 3; ++i) enc_msg_gate(e, *cs[i], *in_ptrs[i], *ms[i], *outs[i], T);
  e.group_end();
}

void Inkling2::enc_forward(Encoder& e, Seq& s, int pos0, int T, bool dump_logits, const Buffer* out_logits,
                           bool need_all_logits, const Buffer* diag_rows, const Buffer* taps_out,
                           bool save_conv_snap) {
  Ink2Grouped gp(e, true);
  Ink2PassEach pe(e, T);
  const uint32_t H = uint32_t(cfg_.hidden);
  const uint32_t QD = uint32_t(cfg_.hq) * uint32_t(cfg_.head_dim);
  const uint32_t KVD = uint32_t(cfg_.hkv) * uint32_t(cfg_.head_dim);
  const size_t row_bytes = size_t(T) * H * 4;
  const Buffer& conv = s.conv[0];
  // One sequence: the per-sequence stages keep the loop shape over ng = 1 group (row views at gi) that the verify rows'
  // decode attention (NI_INK2_VDEC, aseq / arv below) runs over T groups.
  const int ng = 1;
  const uint32_t Tg = uint32_t(T);
  auto gseq = [&](int) -> Seq& { return s; };
  auto gpos = [&](int) { return pos0; };
  auto rv = [&](const Buffer& b, size_t, int) { return b; };
  for (int gi = 0; gi < ng; ++gi) {  // (NI_INK2_KT: kc rows the SSD cache wrote; rows >= pos are written before read)
    Seq& sg = gseq(gi);
    if (!sg.kt_stale) continue;
    for (size_t v = 0; v < cfg_.schedule.size(); ++v)
      if (sg.kt[v]) enc_kt_tile(e, sg, v, 0, cfg_.is_local[size_t(cfg_.schedule[v])] ? ring_rows_ : std::min(gpos(gi), sg.cap));
    sg.kt_stale = false;
  }
  // (verify rows' decode attention dispatches run concurrently: distinct rows and partials; NI_INK2_MCONC=0 serial)
  static const bool mconc = !getenv("NI_INK2_MCONC") || atoi(getenv("NI_INK2_MCONC")) != 0;
  const float sconv_static_scale = 1.0f / std::sqrt(5.0f);
  // Q8 one-row matvec + ink2_sconv1 epilogue (the fused short convs; R as enc_q8 picks it)
  auto enc_q8_sc = [&](Encoder& e, const Ink2Mat& w, const Buffer& x, const Buffer& y, const Buffer& cw,
                       const Buffer& hist, const Buffer& cout, const Ink2SconvStaticP& cp, const Buffer& res,
                       const Buffer& snap) {
    const Ink2Mat& q = w;
    const Ink2Q8P qp{1u, q.N, q.K, q.K, q.N, 0u, 0u, {}};
    static const uint32_t r1 = getenv("NI_INK2_Q8R") ? uint32_t(atoi(getenv("NI_INK2_Q8R"))) : 2u;
    static const uint32_t r1n = getenv("NI_INK2_Q8R_MAXN") ? uint32_t(atoi(getenv("NI_INK2_Q8R_MAXN"))) : 4096u;
    const uint32_t R = (q.N <= r1n && (r1 == 1u || r1 == 2u)) ? r1 : 4u;
    e.use(ink2_pipe("ink2_q8_mv1r_sc" + std::to_string(R)));
    e.buf(0, x);
    e.buf(1, q.q);
    e.buf(2, q.s);
    e.buf(3, y);
    e.val(4, qp);
    e.buf(5, cw);
    e.buf(6, hist);
    e.buf(7, cout);
    e.val(8, cp);
    e.buf(9, res);
    e.buf(10, snap);
    e.dispatch((q.N + 4 * R - 1) / (4 * R), 1, 1, 128);
  };
  const uint32_t snap_stride = (save_conv_snap && T <= kMaxVerify) ? uint32_t(conv_bytes_per_seq(cfg_) / 4) : 0u;
  if (snap_stride && !conv_snap_) {
    conv_snap_ = Gpu::get().alloc(size_t(kMaxVerify) * conv_bytes_per_seq(cfg_), "ink2_conv_snap");
    owned_bufs_.push_back(conv_snap_);
    Gpu::get().make_resident({conv_snap_});
  }

  // Embed + norm -> x_
  for (int gi = 0; gi < ng; ++gi) {
    if (gguf_embed_.is_q8() || gguf_embed_.gguf.data) {
      const auto& blocks = gguf_embed_.gguf;
      const Ink2GgufEmbedP ep{Tg, H, uint32_t(gpos(gi)), gguf_embed_.N,
                             embed_norm_->dtype == DType::F32 ? 1u : 0u, cfg_.rms_eps,
                             blocks.type, blocks.row_bytes};
      e.use(ink2_pipe("ink2_gguf_embed"));
      e.buf(0, gseq(gi).tok);
      e.buf(1, blocks.data ? blocks.data : gguf_embed_.q);
      e.buf(2, blocks.data ? blocks.data : gguf_embed_.s);
      e.buf(3, embed_norm_->buf); e.buf(4, rv(x_, H * 4, gi)); e.val(5, ep);
      e.dispatch(Tg, 1, 1, 512);
      continue;
    }
    const Ink2EmbedP ep{Tg, H, uint32_t(gpos(gi)), uint32_t(embed_->shape[0]), cfg_.rms_eps};
    e.use(ink2_pipe("ink2_embed_norm"));
    e.buf(0, gseq(gi).tok);
    e.buf(1, embed_->buf);
    e.buf(2, embed_norm_->buf);
    e.buf(3, rv(x_, H * 4, gi));
    e.val(4, ep);
    e.dispatch(Tg, 1, 1, 512);
  }
  if (diag_rows) enc_copy_buf(e, x_.view(size_t(T - 1) * H * 4, H * 4), diag_rows->view(0, H * 4), H * 4);

  // Ping-pong active residual stream between x_ and inp_sa_ with zero extra copies.
  const Buffer* cur_x = &x_;
  const Buffer* alt_x = &inp_sa_;
  std::vector<const Buffer*> blocks_vec;
  bool have_prelude = false;
  bool have_prev = false;
  int n_prelude_blocks = 0;

  for (size_t v_idx = 0; v_idx < cfg_.schedule.size(); ++v_idx) {
    const int ip = cfg_.schedule[v_idx];
    const Ink2Layer& L = layers_[size_t(ip)];
    const size_t cbase = v_idx * visit_conv_floats(cfg_) * 4;
    const Buffer conv_k = conv.view(cbase + 0, 3 * KVD * 4);
    const Buffer conv_v = conv.view(cbase + 3 * KVD * 4, 3 * KVD * 4);
    const Buffer conv_a = conv.view(cbase + 6 * KVD * 4, 3 * H * 4);
    const Buffer conv_m = conv.view(cbase + 6 * KVD * 4 + 3 * H * 4, 3 * H * 4);
    auto gconv = [&](int gi, size_t off, size_t bytes) { return gseq(gi).conv[0].view(cbase + off, bytes); };

    static const int cpgrp_env = ink2_knob("NI_INK2_CPGRP", 1, 0);
    const bool cpgrp = ink2_ab("NI_INK2_CPGRP", cpgrp_env) != 0;
    if (ip == cfg_.core_start) {
      if (!have_prelude) {
        enc_copy_buf(e, *cur_x, x_pre_, row_bytes);
        n_prelude_blocks = int(blocks_vec.size());
        for (int b = 0; b < n_prelude_blocks; ++b) {
          if (blocks_vec[size_t(b)] != &prelude_blocks_[b])
            enc_copy_buf(e, *blocks_vec[size_t(b)], prelude_blocks_[b], row_bytes);
        }
        have_prelude = true;
      } else {
        // previous = (blocks[len(pre_loop[1]) + 1 :], x): blocks[2] into blocks_[4], *cur_x into blocks_[5].
        // NI_INK2_CPGRP (lab8, default 1): the two independent copies in one step
        if (cpgrp) e.group_begin();
        enc_copy_buf(e, *blocks_vec[size_t(n_prelude_blocks + 1)], blocks_[4], row_bytes);
        enc_copy_buf(e, *cur_x, blocks_[5], row_bytes);
        if (cpgrp) e.group_end();
        cur_x = &x_pre_;
        alt_x = &inp_sa_;
        blocks_vec.clear();
        for (int b = 0; b < n_prelude_blocks; ++b) blocks_vec.push_back(&prelude_blocks_[b]);
        have_prev = true;
      }
    }

    if (taps_out) {
      int tslot = -1;
      if (v_idx == 5) tslot = 0;
      else if (v_idx == 23) tslot = 1;
      else if (v_idx == 41) tslot = 2;
      if (tslot >= 0) enc_copy_buf(e, x_, taps_out->view(size_t(tslot) * row_bytes, row_bytes), row_bytes);
    }

    const bool has_msgs = have_prev && ip < cfg_.core_end;
    const Buffer msg0 = msg_out_.view(0 * row_bytes, row_bytes);
    const Buffer msg1 = msg_out_.view(1 * row_bytes, row_bytes);
    const Buffer msg2 = msg_out_.view(2 * row_bytes, row_bytes);
    const Buffer msg3 = msg_out_.view(3 * row_bytes, row_bytes);
    if (has_msgs) {
      const Buffer* bank_states[2] = {&blocks_[4], &blocks_[5]};
      const Buffer qv = msg_queries_.view(size_t(ip - cfg_.core_start) * 4 * H * 4, 4 * H * 4);
      enc_attn_res(e, bank_states, 2, &qv, true, 4, msg_out_, T);
    }

    // Attention input AttnRes (*cur_x stays untouched until end-of-visit ping-pong write)
    const Buffer ar0 = ar_out_.view(0 * row_bytes, row_bytes);
    const Buffer ar1 = ar_out_.view(1 * row_bytes, row_bytes);
    const Buffer ar2 = ar_out_.view(2 * row_bytes, row_bytes);
    const Buffer* in_ptrs[3];
    bool attn_normed = false;
    if (ip == 0) {
      in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = cur_x;
    } else {
      std::vector<const Buffer*> st = blocks_vec;
      st.push_back(cur_x);
      const Tensor* aq = attn_queries_[size_t(ip - 1)];
      const uint32_t Q = aq->shape.size() == 2 ? uint32_t(aq->shape[0]) : 1u;
      // (no message gate: the attention norms fold into the AttnRes kernel)
      const uint32_t srcs[3] = {0u, Q == 1 ? 0u : 1u, Q == 1 ? 0u : 2u};
      const Tensor* nws[3] = {L.is_swa ? L.attn_norm : L.q_norm, L.k_norm, L.v_norm};
      const Buffer* nos[3] = {&n0_, &n1_, &n2_};
      attn_normed = !has_msgs && enc_attn_res_norm(e, st.data(), int(st.size()), &aq->buf, aq->dtype == DType::F32, Q,
                                                   ar_out_, T, L.is_swa ? 1 : 3, srcs, nws, nos);
      if (!attn_normed) enc_attn_res(e, st.data(), int(st.size()), &aq->buf, aq->dtype == DType::F32, Q, ar_out_, T);
      if (Q == 1) {
        in_ptrs[0] = in_ptrs[1] = in_ptrs[2] = &ar_out_;
      } else {
        in_ptrs[0] = &ar0;
        in_ptrs[1] = &ar1;
        in_ptrs[2] = &ar2;
      }
    }

    if (has_msgs) {
      enc_msg_gate3(e, L, in_ptrs, msg0, msg1, msg2, T);
      e.group_begin();
      if (L.is_swa) {
        enc_norm(e, n0_, L.attn_norm, n0_, T, H);
        enc_norm(e, n1_, L.attn_norm, n1_, T, H);
        enc_norm(e, n2_, L.attn_norm, n2_, T, H);
      } else {
        enc_norm(e, n0_, L.q_norm, n0_, T, H);
        enc_norm(e, n1_, L.k_norm, n1_, T, H);
        enc_norm(e, n2_, L.v_norm, n2_, T, H);
      }
      e.group_end();
    } else if (attn_normed) {
    } else if (L.is_swa) {
      enc_norm(e, *in_ptrs[0], L.attn_norm, n0_, T, H);
    } else {
      e.group_begin();
      enc_norm(e, *in_ptrs[0], L.q_norm, n0_, T, H);
      enc_norm(e, *in_ptrs[1], L.k_norm, n1_, T, H);
      enc_norm(e, *in_ptrs[2], L.v_norm, n2_, T, H);
      e.group_end();
    }

    const Buffer& kn = (has_msgs || !L.is_swa) ? n1_ : n0_;
    const Buffer& vn = (has_msgs || !L.is_swa) ? n2_ : n0_;

    const bool pf_sconv_kv = L.is_swa && T >= 16 && KVD <= H;
    bool kv_sc_fused = false;
    if ((L.qkvrsg_w || L.qkvrsg_q8) && T <= std::min(mvm() ? 8 : 4, mv_rows()) && (H & 7u) == 0u) {
      const uint32_t nsegs = (!has_msgs && L.is_swa) ? 6u : 4u;
      Ink2MultiMmP mp{uint32_t(T), H, L.qkvrsg_offs[nsegs], nsegs, {}};
      for (uint32_t i = 0; i < 7u; ++i) mp.offs[i] = L.qkvrsg_offs[i];
      if (nsegs == 4u) e.group_begin();
      // one row, one sequence, Q8 (NI_INK2_SCFOLD): SWA k / v short convs in the projections' epilogues
      const bool scfold_kv = (ink2_scfold() & 2) != 0;
      kv_sc_fused = scfold_kv && L.is_swa && T == 1 && ng == 1 && L.qkvrsg_q8 && !pf_sconv_kv &&
                    (nsegs == 6u || ((q8m_ && L.wk_dv.is_q8()) && (q8m_ && L.wv_dv.is_q8())));
      const bool kf32 = kv_sc_fused && L.k_sconv->dtype == DType::F32, vf32 = kv_sc_fused && L.v_sconv->dtype == DType::F32;
      const Ink2SconvStaticP ksp{1u, KVD, kf32 ? 1u : 0u, 0u, sconv_static_scale, snap_stride};
      const Ink2SconvStaticP vsp{1u, KVD, vf32 ? 1u : 0u, 0u, sconv_static_scale, snap_stride};
      const Buffer ksnap = snap_stride ? conv_snap_.view(cbase + 0, conv_snap_.size - (cbase + 0)) : conv_k;
      const Buffer vsnap =
          snap_stride ? conv_snap_.view(cbase + 3 * KVD * 4, conv_snap_.size - (cbase + 3 * KVD * 4)) : conv_v;
      if (L.qkvrsg_q8 && kv_sc_fused && nsegs == 6u) {
        Ink2Q8P qp{1u, mp.N_tot, H, H, 0u, 0u, nsegs, {}};
        for (uint32_t i = 0; i < 7u; ++i) qp.offs[i] = L.qkvrsg_offs[i];
        e.use(ink2_pipe("ink2_q8_multi6_sckv"));
        e.buf(0, n0_);
        e.buf(1, L.qkvrsg_q8.q);
        e.buf(2, L.qkvrsg_q8.s);
        e.buf(3, q_);
        e.val(4, qp);
        e.buf(5, r_);
        e.buf(6, sinks_);
        e.buf(7, gate_attn_);
        e.buf(8, k_);
        e.buf(9, v_);
        e.buf(10, L.k_sconv->buf);
        e.buf(11, gconv(0, 0, 3 * KVD * 4));
        e.val(12, ksp);
        e.buf(13, ksnap);
        e.buf(14, L.v_sconv->buf);
        e.buf(15, gconv(0, 3 * KVD * 4, 3 * KVD * 4));
        e.val(16, vsp);
        e.buf(17, vsnap);
      } else if (L.qkvrsg_q8) {
        Ink2Q8P qp{uint32_t(T), mp.N_tot, H, H, 0u, 0u, nsegs, {}};
        for (uint32_t i = 0; i < 7u; ++i) qp.offs[i] = L.qkvrsg_offs[i];
        e.use(ink2_pipe("ink2_q8_mvm_multi6_" + std::to_string(T)));
        e.buf(0, n0_);
        e.buf(1, L.qkvrsg_q8.q);
        e.buf(2, L.qkvrsg_q8.s);
        e.buf(3, q_);
        e.val(4, qp);
        e.buf(5, r_);
        e.buf(6, sinks_);
        e.buf(7, gate_attn_);
        e.buf(8, k_);
        e.buf(9, v_);
      } else {
        e.use(ink2_pipe(T >= 2 && mvm() ? "ink2_bf16_mvm_multi6_" + std::to_string(T) : std::string("ink2_bf16_mv4_multi6")));
        e.buf(0, n0_);
        e.buf(1, L.qkvrsg_w);
        e.buf(2, q_);
        e.buf(3, r_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, k_);
        e.buf(7, v_);
        e.val(8, mp);
      }
      e.dispatch((mp.N_tot + 15) / 16, 1, 1, 128);
      if (nsegs == 4u) {
        if (kv_sc_fused) {
          enc_q8_sc(e, L.wk_dv, kn, k_, L.k_sconv->buf, gconv(0, 0, 3 * KVD * 4), k_, ksp, k_, ksnap);
          enc_q8_sc(e, L.wv_dv, vn, v_, L.v_sconv->buf, gconv(0, 3 * KVD * 4, 3 * KVD * 4), v_, vsp, v_, vsnap);
        } else {
          enc_mm(e, L.wk_dv, kn, k_, T);
          enc_mm(e, L.wv_dv, vn, v_, T);
        }
        e.group_end();
      }
    } else {
      e.group_begin();
      enc_mm(e, L.wq_du, n0_, q_, T);
      enc_mm(e, L.wk_dv, kn, pf_sconv_kv ? moe_out_ : k_, T);
      enc_mm(e, L.wv_dv, vn, pf_sconv_kv ? h_mlp_ : v_, T);
      enc_mm(e, L.wr_du, n0_, r_, T);
      static const bool sk_sinks = ink2_knob("NI_INK2_SK", 8, 0) > 1;
      if (!sk_sinks) enc_mm(e, *L.attn_sink_proj, n0_, sinks_, T);
      enc_mm(e, L.gate_attn, n0_, gate_attn_, T);
      e.group_end();
      if (sk_sinks) enc_mm(e, *L.attn_sink_proj, n0_, sinks_, T);  // (outside the group: split-K)
    }

    // K / V short convolutions
    if (L.is_swa && kv_sc_fused) {
    } else if (L.is_swa) {
      if (pf_sconv_kv) {
        e.group_begin();
        const Ink2SconvStaticP sp{uint32_t(T), KVD, L.k_sconv->dtype == DType::F32 ? 1u : 0u, 0u, sconv_static_scale};
        e.use(ink2_pipe("ink2_sconv_static_pf"));
        e.buf(0, moe_out_);
        e.buf(1, L.k_sconv->buf);
        e.buf(2, conv_k);
        e.buf(3, k_);
        e.val(4, sp);
        e.buf(5, k_);
        e.dispatch((KVD + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);

        const Ink2SconvStaticP svp{uint32_t(T), KVD, L.v_sconv->dtype == DType::F32 ? 1u : 0u, 0u, sconv_static_scale};
        e.use(ink2_pipe("ink2_sconv_static_pf"));
        e.buf(0, h_mlp_);
        e.buf(1, L.v_sconv->buf);
        e.buf(2, conv_v);
        e.buf(3, v_);
        e.val(4, svp);
        e.buf(5, v_);
        e.dispatch((KVD + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);
        e.group_end();

        e.group_begin();
        enc_copy_buf(e, moe_out_.view(size_t(T - 3) * KVD * 4, 3 * size_t(KVD) * 4), conv_k, 3 * size_t(KVD) * 4);
        enc_copy_buf(e, h_mlp_.view(size_t(T - 3) * KVD * 4, 3 * size_t(KVD) * 4), conv_v, 3 * size_t(KVD) * 4);
        e.group_end();
      } else {
        e.group_begin();
        for (int gi = 0; gi < ng; ++gi) {
          const Ink2SconvStaticP sp{Tg, KVD, L.k_sconv->dtype == DType::F32 ? 1u : 0u, 0u, sconv_static_scale,
                                    snap_stride};
          e.use(ink2_pipe(ink2_sconv_name()));
          e.buf(0, rv(k_, KVD * 4, gi));
          e.buf(1, L.k_sconv->buf);
          e.buf(2, gconv(gi, 0, 3 * KVD * 4));
          e.buf(3, rv(k_, KVD * 4, gi));
          e.val(4, sp);
          e.buf(5, rv(k_, KVD * 4, gi));
          e.buf(6, snap_stride ? conv_snap_.view(cbase + 0, conv_snap_.size - (cbase + 0)) : conv_k);
          e.dispatch((KVD + 255) / 256, 1, 1, 256);

          const Ink2SconvStaticP svp{Tg, KVD, L.v_sconv->dtype == DType::F32 ? 1u : 0u, 0u, sconv_static_scale,
                                     snap_stride};
          e.use(ink2_pipe(ink2_sconv_name()));
          e.buf(0, rv(v_, KVD * 4, gi));
          e.buf(1, L.v_sconv->buf);
          e.buf(2, gconv(gi, 3 * KVD * 4, 3 * KVD * 4));
          e.buf(3, rv(v_, KVD * 4, gi));
          e.val(4, svp);
          e.buf(5, rv(v_, KVD * 4, gi));
          e.buf(6, snap_stride ? conv_snap_.view(cbase + 3 * KVD * 4, conv_snap_.size - (cbase + 3 * KVD * 4)) : conv_v);
          e.dispatch((KVD + 255) / 256, 1, 1, 256);
        }
        e.group_end();
      }
    } else {
      // NI_INK2_KVGRP (lab8, default 1; Kapitsa #1393): k_proj + v_proj in one step (v's dynamic weights in a second
      // dyn_ region), then both dynamic convs in one step: 4 -> 2 stages per global visit (one-dispatch matvecs, T <= 8)
      static const int kvgrp_env = ink2_knob("NI_INK2_KVGRP", 1, 0);
      const bool kvgrp = ink2_ab("NI_INK2_KVGRP", kvgrp_env) != 0 && T <= 8 && 2 * T <= tmax_;
      const size_t dynb = size_t(T) * 4 * size_t(cfg_.hkv) * 4;
      const Buffer dyn_v = kvgrp ? dyn_.view(dynb, dynb) : dyn_;
      auto dyn_conv = [&](int gi, bool is_v) {
        const Tensor* sw = is_v ? L.v_static : L.k_static;
        const Ink2SconvDynP dp{Tg, KVD, uint32_t(cfg_.hkv), sw->dtype == DType::F32 ? 1u : 0u, snap_stride};
        const size_t co = is_v ? 3 * KVD * 4 : 0;
        const Buffer& xb = is_v ? v_ : k_;
        e.use(ink2_pipe("ink2_sconv_dynamic"));
        e.buf(0, rv(xb, KVD * 4, gi));
        e.buf(1, rv(is_v ? dyn_v : dyn_, 4 * size_t(cfg_.hkv) * 4, gi));
        e.buf(2, sw->buf);
        e.buf(3, gconv(gi, co, 3 * KVD * 4));
        e.buf(4, rv(xb, KVD * 4, gi));
        e.val(5, dp);
        e.buf(6, snap_stride ? conv_snap_.view(cbase + co, conv_snap_.size - (cbase + co)) : (is_v ? conv_v : conv_k));
        e.dispatch((KVD + 255) / 256, 1, 1, 256);
      };
      if (kvgrp) {
        e.group_begin();
        enc_mm(e, *L.k_proj, kn, dyn_, T);
        enc_mm(e, *L.v_proj, vn, dyn_v, T);
        e.group_end();
        e.group_begin();
        for (int gi = 0; gi < ng; ++gi) dyn_conv(gi, false), dyn_conv(gi, true);
        e.group_end();
      } else {
        enc_mm(e, *L.k_proj, kn, dyn_, T);
        for (int gi = 0; gi < ng; ++gi) dyn_conv(gi, false);
        enc_mm(e, *L.v_proj, vn, dyn_, T);
        for (int gi = 0; gi < ng; ++gi) dyn_conv(gi, true);
      }
    }

    // Q/K/R clipping + V norm + log scaling + KV cache write
    const uint32_t kv_cap = L.is_swa ? uint32_t(ring_rows_) : uint32_t(s.cap);
    const size_t rrow = size_t(cfg_.hq) * size_t(cfg_.d_rel) * 4;
    // NI_INK2_RBFOLD (lab8, default 1): the decode-size rel bias (ink2_rel_bias rows, T < 8) in qkvr_prep's r heads
    static const int rbfold_env = ink2_knob("NI_INK2_RBFOLD", 1, 0);
    const bool rb_pf = cfg_.d_rel == 16 && T >= 8;
    const bool rbfold = ink2_ab("NI_INK2_RBFOLD", rbfold_env) != 0 && !rb_pf && cfg_.d_rel <= cfg_.head_dim;
    const Ink2RelP rbp{Tg, uint32_t(cfg_.hq), uint32_t(cfg_.d_rel), uint32_t(cfg_.rel_extent),
                       L.rel_proj->dtype == DType::F32 ? 1u : 0u};
    const size_t rbrow = size_t(cfg_.hq) * size_t(cfg_.rel_extent) * 4;
    for (int gi = 0; gi < ng; ++gi) {
      const uint32_t g_cap = L.is_swa ? uint32_t(ring_rows_) : uint32_t(gseq(gi).cap);
      const Ink2QkvrP qp{Tg,
                         uint32_t(cfg_.hq),
                         uint32_t(cfg_.hkv),
                         uint32_t(cfg_.head_dim),
                         uint32_t(cfg_.d_rel),
                         uint32_t(gpos(gi)),
                         g_cap,
                         L.is_swa ? 1u : 0u,
                         L.q_scale->dtype == DType::F32 ? 1u : 0u,
                         cfg_.qk_clip,
                         cfg_.rel_clip,
                         cfg_.rms_eps,
                         cfg_.log_floor,
                         cfg_.log_alpha,
                         gseq(gi).kt.empty() ? 0u : 1u};
      e.use(ink2_pipe(rbfold ? "ink2_qkvr_prep_rb" : "ink2_qkvr_prep"));
      if (rbfold) {
        e.buf(9, L.rel_proj->buf);
        e.buf(10, rv(rb_, rbrow, gi));
        e.val(11, rbp);
      }
      e.buf(0, rv(q_, QD * 4, gi));
      e.buf(1, rv(k_, KVD * 4, gi));
      e.buf(2, rv(v_, KVD * 4, gi));
      e.buf(3, rv(r_, rrow, gi));
      e.buf(4, L.q_scale->buf);
      e.buf(5, L.rel_scale->buf);
      e.buf(6, gseq(gi).kc[v_idx]);
      e.buf(7, gseq(gi).vc[v_idx]);
      e.val(8, qp);
      e.buf(12, gseq(gi).kt.empty() ? gseq(gi).kc[v_idx] : gseq(gi).kt[v_idx]);
      const uint32_t rj = rbfold ? (uint32_t(cfg_.rel_extent) + uint32_t(cfg_.head_dim) - 1) / uint32_t(cfg_.head_dim) : 1u;
      e.dispatch(uint32_t(cfg_.hq) * (1 + rj) + uint32_t(2 * cfg_.hkv), Tg, 1, uint32_t(cfg_.head_dim));
    }

    // Rel bias [T, Hq, rel_extent]
    if (!rbfold) {
      const Ink2RelP rp{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.d_rel), uint32_t(cfg_.rel_extent),
                        L.rel_proj->dtype == DType::F32 ? 1u : 0u};
      if (cfg_.d_rel == 16 && T >= 8) {
        e.use(ink2_pipe("ink2_rel_bias_pf"));
        e.buf(0, r_);
        e.buf(1, L.rel_proj->buf);
        e.buf(2, rb_);
        e.val(3, rp);
        e.dispatch((uint32_t(cfg_.rel_extent) + 31) / 32, (uint32_t(T) + 7) / 8, (uint32_t(cfg_.hq) + 7) / 8, 128);
      } else {
        e.use(ink2_pipe("ink2_rel_bias"));
        e.buf(0, r_);
        e.buf(1, L.rel_proj->buf);
        e.buf(2, rb_);
        e.val(3, rp);
        e.dispatch((uint32_t(cfg_.rel_extent) + 31) / 32, uint32_t(cfg_.hq), uint32_t(T), 32);
      }
    }

    // Attention
    const float attn_scale = 1.0f / std::sqrt(float(cfg_.head_dim));
    // lab15 NI_INK2_VDEC=1 (A/B-able): verify rows (T = 2..8, one sequence) run the decode attention once per row
    // (row t at position pos0 + t, its own partials), concurrent, instead of ink2_attn_verify_g3 (8k: 37.3 ms at T = 8
    // vs 2.9 for one decode row: simd_sum per (key, row, head) made it ALU-bound). Every row = a decode step's math.
    static const int vdec_env = ink2_knob("NI_INK2_VDEC", 1, 0);
    const bool vdec = T > 1 && T <= kMaxVerify && ink2_ab("NI_INK2_VDEC", vdec_env) != 0;
    if (T == 1 || vdec) {
      const int ang = vdec ? T : ng;
      const bool aconc = vdec && mconc;
      auto aseq = [&](int gi) -> Seq& { return vdec ? s : gseq(gi); };
      auto apos = [&](int gi) { return vdec ? pos0 + gi : gpos(gi); };
      auto arv = [&](const Buffer& b, size_t rb, int gi) { return vdec ? b.view(size_t(gi) * rb, rb) : rv(b, rb, gi); };
      // (multi rows: one decode attention per sequence, each into its own partials; then the reduces)
      const size_t pm = size_t(cfg_.hq) * kAttnChunkCap * 4, rbr = size_t(cfg_.hq) * size_t(cfg_.rel_extent) * 4;
      std::vector<Ink2AttnDecP> adps;
      bool redlast = false;
      if (aconc) e.group_begin();
      for (int gi = 0; gi < ang; ++gi) {
      const int pos0 = apos(gi);
      const uint32_t kv_cap = L.is_swa ? uint32_t(ring_rows_) : uint32_t(aseq(gi).cap);
      const uint32_t lk = L.is_swa ? std::min(uint32_t(pos0 + 1), uint32_t(cfg_.swa)) : uint32_t(pos0 + 1);
      static const int dec_g_mode = ink2_knob("NI_INK2_DEC_G", 3, 0);
      const bool kw = dec_g_mode == 3;
      // (keys-across-lanes kernel: chunks of whole 32-key blocks; NI_INK2_KW_MINCHUNK for the SWA window's)
      static const int kw_min_env = getenv("NI_INK2_KW_MINCHUNK") ? atoi(getenv("NI_INK2_KW_MINCHUNK")) : 64;
      static const int kw_maxch_env = ink2_knob("NI_INK2_KW_MAXCH", 48, int(kMaxAttnChunks));
      const uint32_t kw_min = uint32_t(ink2_ab("NI_INK2_KW_MINCHUNK", kw_min_env));
      // (lab12 NI_INK2_KW_MAXCH: chunks of a long visit, <= kAttnChunkCap; ulp-level when changed)
      const uint32_t maxch = kw ? std::min(kAttnChunkCap, uint32_t(ink2_ab("NI_INK2_KW_MAXCH", kw_maxch_env))) : kMaxAttnChunks;
      // (lab12 NI_INK2_KW_SWACH: keys per threadgroup of the SWA visits (0: KW_MINCHUNK's); 128 = 8 chunks of the
      //  1024 window with all 4 simdgroups busy, vs 16 x 64 with 2 idle: -0.19 ms at 8k on ATTP=8704; ulp-level)
      static const int kw_swach_env = ink2_knob("NI_INK2_KW_SWACH", 128, 0);
      const uint32_t swach = kw && L.is_swa ? uint32_t(ink2_ab("NI_INK2_KW_SWACH", kw_swach_env)) : 0u;
      uint32_t chunk = std::max(swach ? swach : kw ? kw_min : 32u, (lk + maxch - 1) / maxch);
      chunk = kw ? (chunk + 31) / 32 * 32 : (chunk + 3) / 4 * 4;
      const uint32_t nchunks = std::max(1u, (lk + chunk - 1) / chunk);
      const Ink2AttnDecP adp{uint32_t(pos0),
                             kv_cap,
                             uint32_t(cfg_.hq),
                             uint32_t(cfg_.hkv),
                             uint32_t(cfg_.head_dim),
                             uint32_t(cfg_.rel_extent),
                             L.is_swa ? 1u : 0u,
                             uint32_t(cfg_.swa),
                             chunk,
                             nchunks,
                             attn_scale};
      const bool dec_g = dec_g_mode != 0;
      const bool g3 = dec_g && cfg_.hq == 3 * cfg_.hkv && cfg_.head_dim == 128;
      // NI_INK2_DEC_G: 1 GQA kernel with fast::exp (decbench 40K: 1.83 -> 0.80 ms a global visit), 2 the
      // same with precise::exp (1.37 ms), 0 the per-head kernel; 3 (default, round 3) keys across lanes (0.59 ms)
      static const int redlast_env = ink2_knob("NI_INK2_REDLAST", 3, 0);
      static const bool red_p0 = ink2_knob("NI_INK2_REDP", 1, 0) != 0;
      // NI_INK2_REDLAST=3 (lab11): the split-K reduce by the last threadgroup of each KV head, its partials device
      // coherent(device) memory + device-scope fences (plain stores and loads); other values: the reduce stage
      const int redlast_v = ink2_ab("NI_INK2_REDLAST", redlast_env);
      redlast = redlast_v == 3 && g3 && kw && red_p0;
      static const int kt_env = ink2_kt_alloc() ? 1 : 0;
      const bool kt = redlast && ink2_attl() && !aseq(gi).kt.empty() && ink2_ab("NI_INK2_KT", kt_env) != 0;
      e.use(ink2_pipe(!g3 ? "ink2_attn_decode" : dec_g_mode == 2 ? "ink2_attn_decode_g3" : kw ? (redlast ? ink2_rc_name(kt).c_str() : "ink2_attn_decode_g3kw")
                                                                                               : "ink2_attn_decode_g3_fx"));
      if (redlast) {
        e.buf(8, arv(sinks_, size_t(cfg_.hq) * 4, gi));
        e.buf(9, arv(gate_attn_, QD * 4, gi));
        e.buf(10, arv(att_out_, QD * 4, gi));
        e.buf(11, attn_cnt_.view(size_t(gi) * size_t(cfg_.hkv) * 4, size_t(cfg_.hkv) * 4));
      }
      e.buf(0, arv(q_, QD * 4, gi));
      e.buf(1, kt ? aseq(gi).kt[v_idx] : aseq(gi).kc[v_idx]);
      e.buf(2, aseq(gi).vc[v_idx]);
      e.buf(3, arv(rb_, rbr, gi));
      e.buf(4, arv(part_m_, pm, gi));
      e.buf(5, arv(part_l_, pm, gi));
      e.buf(6, arv(part_o_, pm * cfg_.head_dim, gi));
      e.val(7, adp);
      e.dispatch(uint32_t(g3 ? cfg_.hkv : cfg_.hq), nchunks, 1, 128);
      adps.push_back(adp);
      }
      if (aconc) e.group_end(), e.group_begin();
      for (int gi = 0; gi < ang && !redlast; ++gi) {
      const Ink2AttnDecP& adp = adps[size_t(gi)];

      static const bool red_p = ink2_knob("NI_INK2_REDP", 1, 0) != 0;
      e.use(ink2_pipe(red_p ? "ink2_attn_decode_reduce_p" : "ink2_attn_decode_reduce"));
      e.buf(0, arv(part_m_, pm, gi));
      e.buf(1, arv(part_l_, pm, gi));
      e.buf(2, arv(part_o_, pm * cfg_.head_dim, gi));
      e.buf(3, arv(sinks_, size_t(cfg_.hq) * 4, gi));
      e.buf(4, arv(gate_attn_, QD * 4, gi));
      e.buf(5, arv(att_out_, QD * 4, gi));
      e.val(6, adp);
      e.dispatch(uint32_t(cfg_.hq), 1, 1, 32);
      }
      if (aconc) e.group_end();
    } else {
      const Ink2AttnPfP apf{uint32_t(T),
                            uint32_t(pos0),
                            kv_cap,
                            uint32_t(cfg_.hq),
                            uint32_t(cfg_.hkv),
                            uint32_t(cfg_.head_dim),
                            uint32_t(cfg_.rel_extent),
                            L.is_swa ? 1u : 0u,
                            uint32_t(cfg_.swa),
                            attn_scale};
      static const bool qat_env = !getenv("NI_INK2_ATTN_T") || atoi(getenv("NI_INK2_ATTN_T")) != 0;
      static const bool qat_small = getenv("MTL_SHADER_VALIDATION") && atoi(getenv("MTL_SHADER_VALIDATION")) != 0;
      static const int qat_min = getenv("NI_INK2_QAT_MIN") ? atoi(getenv("NI_INK2_QAT_MIN")) : 16;
      const bool use_qat = qat_env && tensor_pf_ && T >= qat_min && cfg_.head_dim == 128 && cfg_.hq == cfg_.hkv * 3;
      static const bool aver = ink2_knob("NI_INK2_AVER", 1, 0) != 0;
      if (!use_qat && aver && T <= kMaxVerify && cfg_.hq == 3 * cfg_.hkv && cfg_.head_dim == 128) {
        // verify rows: split-K over the union key range, K / V read once for all rows (ink2_attn_verify_g3)
        const uint32_t k_first =
            L.is_swa ? uint32_t(pos0 + 1) - std::min(uint32_t(pos0 + 1), uint32_t(cfg_.swa)) : 0u;
        const uint32_t k_end = uint32_t(pos0 + T);
        const uint32_t lk = k_end - k_first;
        uint32_t chunk = std::max(32u, (lk + kMaxAttnChunks - 1) / kMaxAttnChunks);
        chunk = (chunk + 3) / 4 * 4;
        const uint32_t nchunks = std::max(1u, (lk + chunk - 1) / chunk);
        const Ink2AttnVerP vp{uint32_t(pos0), uint32_t(T), kv_cap, uint32_t(cfg_.hq), uint32_t(cfg_.hkv),
                              uint32_t(cfg_.head_dim), uint32_t(cfg_.rel_extent), L.is_swa ? 1u : 0u,
                              uint32_t(cfg_.swa), k_first, k_end, chunk, nchunks, attn_scale};
        const uint32_t tr = T <= 4 ? uint32_t(T) : uint32_t(T + 1) / 2;
        e.use(ink2_pipe("ink2_attn_verify_g3_" + std::to_string(tr)));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, part_m_);
        e.buf(5, part_l_);
        e.buf(6, part_o_);
        e.val(7, vp);
        e.dispatch(uint32_t(cfg_.hkv), nchunks, (uint32_t(T) + tr - 1) / tr, 128);
        static const bool vred_p = ink2_knob("NI_INK2_REDP", 1, 0) != 0;
        e.use(ink2_pipe(vred_p ? "ink2_attn_verify_reduce_p" : "ink2_attn_verify_reduce"));
        e.buf(0, part_m_);
        e.buf(1, part_l_);
        e.buf(2, part_o_);
        e.buf(3, sinks_);
        e.buf(4, gate_attn_);
        e.buf(5, att_out_);
        e.val(6, vp);
        e.dispatch(uint32_t(cfg_.hq), uint32_t(T), 1, 32);
      } else if (use_qat && ink2_fa_nt()) {
        // simdgroup_matrix flash attention (lab5): 1.5x the tensor kernel at long context (attnbench 500 @ 40K global
        // 53.7 vs 81.0 ms); NI_INK2_FA=0 = qattn_ink2 below, NI_INK2_FA=16 (default) / 1616 = BK 32 / 16 keys
        e.use(ink2_pipe(ink2_fa_nt() == 1616 ? "ink2_fa_t16k16" : "ink2_fa_t16k32"));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, att_out_);
        e.val(7, apf);
        e.dispatch((uint32_t(T) + 15) / 16, uint32_t(cfg_.hkv), 1, 192);
      } else if (use_qat) {
        const uint32_t qat_tok = qat_small ? 16u : 32u;
        // Cooperative-tensor lane layout: Apple10+ (written for M5) vs the simdgroup_matrix layout of older GPUs (M1;
        // the M5 layout there gave wrong attention, ink2_check v0_out off). NI_INK2_ATTN_LAY=0|1 overrides.
        // (the M5 layout on the chips without matrix units: those the ink2_tensor_pf row leaves out)
        static const bool lay1 = getenv("NI_INK2_ATTN_LAY") ? atoi(getenv("NI_INK2_ATTN_LAY")) != 0
                                                             : !default_on(Knob::kInk2TensorPf, ModelFacts{});
        e.use(base_pipe(std::string(qat_small ? "qattn_ink2_s3k16" : "qattn_ink2_s6k32p") +
                        (lay1 ? "_l1" : "")));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, att_out_);
        e.val(7, apf);
        e.dispatch((uint32_t(T) + qat_tok - 1) / qat_tok, uint32_t(cfg_.hkv), 1, qat_small ? 96 : 192);
      } else {
        e.use(ink2_pipe("ink2_attn_prefill"));
        e.buf(0, q_);
        e.buf(1, s.kc[v_idx]);
        e.buf(2, s.vc[v_idx]);
        e.buf(3, rb_);
        e.buf(4, sinks_);
        e.buf(5, gate_attn_);
        e.buf(6, att_out_);
        e.val(7, apf);
        e.dispatch(uint32_t(cfg_.hq), uint32_t(T), 1, 128);
      }
    }

    // wo_ud + attn_sconv (+ *cur_x residual on non-boundary layers -> ffn_inp_)
    const bool is_boundary = (ip % cfg_.block_sz == 0);
    const Ink2SconvStaticP asp{uint32_t(T), H, L.attn_sconv->dtype == DType::F32 ? 1u : 0u, is_boundary ? 0u : 1u,
                               sconv_static_scale, snap_stride};
    // one row, one sequence, Q8 wo: attn_sconv in the matvec's epilogue (NI_INK2_SCFOLD=0: own stage)
    const bool scfold_a = (ink2_scfold() & 5) == 1;
    const Ink2Mat& woq = L.wo_ud;
    const bool attn_sc_fused = scfold_a && T == 1 && ng == 1 && (q8m_ && woq.is_q8() && woq.K % 256 == 0);
    // NI_INK2_CPGRP: a block boundary's copy of *cur_x runs beside the fused wo (wo reads *cur_x only as its residual,
    // which a boundary does not add; nothing here reads the block slot)
    const bool bcopy_grp = cpgrp && is_boundary && attn_sc_fused;
    const Buffer& bdst = (ip / cfg_.block_sz == 0) ? prelude_blocks_[0] : blocks_[ip / cfg_.block_sz];
    if (attn_sc_fused) {
      if (bcopy_grp) e.group_begin();
      enc_q8_sc(e, L.wo_ud, att_out_, a_out_, L.attn_sconv->buf, gconv(0, 6 * KVD * 4, 3 * H * 4), ffn_inp_, asp,
                *cur_x, snap_stride ? conv_snap_.view(cbase + 6 * KVD * 4, conv_snap_.size - (cbase + 6 * KVD * 4))
                                    : conv_a);
      if (bcopy_grp) enc_copy_buf(e, *cur_x, bdst, row_bytes), e.group_end();
    } else {
      enc_mm(e, L.wo_ud, att_out_, a_out_, T);
    }
    for (int gi = 0; gi < ng && !attn_sc_fused; ++gi) {
    Ink2SconvStaticP gasp = asp;
    gasp.T = Tg;
    e.use(ink2_pipe(T >= 16 ? std::string("ink2_sconv_static_pf") : ink2_sconv_name()));
    e.buf(0, rv(a_out_, H * 4, gi));
    e.buf(1, L.attn_sconv->buf);
    e.buf(2, gconv(gi, 6 * KVD * 4, 3 * H * 4));
    e.buf(3, rv(ffn_inp_, H * 4, gi));
    e.val(4, gasp);
    e.buf(5, rv(*cur_x, H * 4, gi));
    e.buf(6, snap_stride ? conv_snap_.view(cbase + 6 * KVD * 4, conv_snap_.size - (cbase + 6 * KVD * 4)) : conv_a);
    if (T >= 16) {
      e.dispatch((H + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);
      enc_copy_buf(e, a_out_.view(size_t(T - 3) * H * 4, 3 * size_t(H) * 4), conv_a, 3 * size_t(H) * 4);
    } else {
      e.dispatch((H + 255) / 256, 1, 1, 256);
    }
    }
    if (is_boundary) {
      if (!bcopy_grp) enc_copy_buf(e, *cur_x, bdst, row_bytes);
      blocks_vec.push_back(&bdst);
    }

    // MLP AttnRes + optional MessageGate + norm
    bool mlp_normed = false;
    {
      std::vector<const Buffer*> st = blocks_vec;
      st.push_back(&ffn_inp_);
      const Tensor* mq = mlp_queries_[size_t(ip)];
      const uint32_t src0[3] = {0u, 0u, 0u};
      const Tensor* nw[3] = {L.mlp_norm, nullptr, nullptr};
      const Buffer* no[3] = {&x_mlp_, nullptr, nullptr};
      mlp_normed = !has_msgs && enc_attn_res_norm(e, st.data(), int(st.size()), &mq->buf, mq->dtype == DType::F32, 1,
                                                  h_mlp_, T, 1, src0, nw, no);
      if (!mlp_normed) enc_attn_res(e, st.data(), int(st.size()), &mq->buf, mq->dtype == DType::F32, 1, h_mlp_, T);
    }
    if (has_msgs) enc_msg_gate(e, L.ctrl_mlp, h_mlp_, msg3, h_mlp_, T);
    if (!mlp_normed) enc_norm(e, h_mlp_, L.mlp_norm, x_mlp_, T, H);

    bool mlp_sc_fused = false;
    // Router + MoE (router uses exact FP32 math on all M so prefill top-k matches decode)
    // NI_INK2_RTLAST=1 (lab11; BENCH_AB-able): one row, the router matvec's last threadgroup runs the route
    // (ink2_mv_sk_route: same logits, same route math), one stage less per visit
    static const int rtlast_env = ink2_knob("NI_INK2_RTLAST", 1, 0);
    static const uint32_t sk_max_r = ink2_knob("NI_INK2_MVSK", 256, 0);
    const uint32_t rN = uint32_t(L.router_w->shape[0]), rK = uint32_t(L.router_w->shape[1]);
    const bool rtlast = ink2_ab("NI_INK2_RTLAST", rtlast_env) != 0 && T == 1 && rN <= sk_max_r && (rK & 3u) == 0u &&
                        (L.router_w->dtype == DType::BF16 || L.router_w->dtype == DType::F32) &&
                        uint32_t(cfg_.n_experts) == rN && cfg_.top_k <= 8 && cfg_.n_experts <= 256;
    if (rtlast) {
      const Ink2MmP mp{1u, rN, rK, rK, rN, L.router_w->dtype == DType::F32 ? 1u : 0u, 0u};
      const Ink2RouteP rp{1u, uint32_t(cfg_.n_experts), uint32_t(cfg_.top_k),
                          L.router_bias->dtype == DType::F32 ? 1u : 0u, cfg_.route_scale};
      e.use(ink2_pipe("ink2_mv_sk_route"));
      e.buf(0, x_mlp_);
      e.buf(1, L.router_w->buf);
      e.buf(2, rlog_);
      e.val(3, mp);
      e.buf(4, L.router_bias->buf);
      e.buf(5, eid_);
      e.buf(6, ew_);
      e.val(7, rp);
      e.buf(8, attn_cnt_.view(size_t(kMaxVerify) * size_t(cfg_.hkv) * 4, 16));
      e.dispatch(rN, 1, 1, 128);
    } else {
      enc_mm(e, *L.router_w, x_mlp_, rlog_, T, 0, false, true);
    }
    {
      const Ink2RouteP rp{uint32_t(T), uint32_t(cfg_.n_experts), uint32_t(cfg_.top_k),
                          L.router_bias->dtype == DType::F32 ? 1u : 0u, cfg_.route_scale};
      if (!rtlast) {
      e.use(ink2_pipe("ink2_moe_route"));
      e.buf(0, rlog_);
      e.buf(1, L.router_bias->buf);
      e.buf(2, eid_);
      e.buf(3, ew_);
      e.val(4, rp);
      e.dispatch(uint32_t(T), 1, 1, 32);
      }
    }
    {
      if (L.gguf_gu.data || L.gguf_down.data) {
        const Ink2MoeP mp{uint32_t(T), H, uint32_t(cfg_.moe_I), uint32_t(cfg_.top_k)};
        if (L.gguf_gu.data) {
          const Ink2GgufMoeP gu{uint32_t(T), H, uint32_t(cfg_.moe_I), uint32_t(cfg_.top_k),
                                L.gguf_gu.type, L.gguf_gu.row_bytes};
          e.use(ink2_pipe("ink2_gguf_gu"));
          e.buf(0, x_mlp_); e.buf(1, eid_); e.buf(2, L.gguf_gu.data); e.buf(3, moe_act_);
          e.buf(4, L.w13_outer); e.val(5, gu);
          e.dispatch((gu.I + 3) / 4, gu.top, uint32_t(T), 128);
        } else {
          e.use(ink2_pipe("ink2_moe_nvfp4_w13_situ"));
          e.buf(0, x_mlp_); e.buf(1, eid_); e.buf(2, L.w13->buf); e.buf(3, L.w13_sc->buf);
          e.buf(4, L.w13_outer); e.buf(5, moe_act_); e.val(6, mp);
          e.dispatch((mp.I + 15) / 16, mp.K_top, uint32_t(T), 128);
        }
        if (L.gguf_down.data) {
          const Ink2GgufMoeP dn{uint32_t(T), H, uint32_t(cfg_.moe_I), uint32_t(cfg_.top_k),
                                L.gguf_down.type, L.gguf_down.row_bytes};
          e.use(ink2_pipe("ink2_gguf_down"));
          e.buf(0, moe_act_); e.buf(1, eid_); e.buf(2, ew_); e.buf(3, L.gguf_down.data);
          e.buf(4, moe_out_); e.buf(5, L.w2_outer); e.val(6, dn);
          e.dispatch((H + 3) / 4, uint32_t(T), 1, 128);
        } else {
          e.use(ink2_pipe("ink2_moe_nvfp4_w2_acc"));
          e.buf(0, moe_act_); e.buf(1, eid_); e.buf(2, ew_); e.buf(3, L.w2->buf); e.buf(4, L.w2_sc->buf);
          e.buf(5, L.w2_outer); e.buf(6, moe_out_); e.val(7, mp);
          e.dispatch((H + 15) / 16, uint32_t(T), 1, 128);
        }
      } else if (tensor_pf_ && T > 32) {
        const uint32_t P = uint32_t(T) * uint32_t(cfg_.top_k);
        const uint32_t E = uint32_t(cfg_.n_experts);
        const uint32_t I = uint32_t(cfg_.moe_I);
        const uint32_t per = P / E;
        // 64-row expert tiles lose to 32 in-model (lab5: routing is skewed, so 64-row tiles are mostly padding;
        // 8k prefill 391.6 -> 431.1 tok/s at m32). NI_INK2_MOE_M64=1 = old rule (64 from 40 rows/expert),
        // NI_INK2_MOE_BM=16|32|64 forces one tile height
        static const uint32_t bm_env = getenv("NI_INK2_MOE_BM") ? uint32_t(atoi(getenv("NI_INK2_MOE_BM"))) : 0u;
        static const bool m64 = ink2_knob("NI_INK2_MOE_M64", 0, 1) != 0;
        const uint32_t bm = bm_env ? bm_env : (m64 && per >= 40) ? 64u : per >= 12 ? 32u : 16u;
        const std::string ms = "_m" + std::to_string(bm);
        const uint32_t max_tiles = P / bm + E + 1;
        {
          const Ink2BucketP bp{P, E, bm};
          e.use(ink2_pipe("ink2_moe_bucket"));
          e.buf(0, eid_);
          e.buf(1, moe_sorted_);
          e.buf(2, moe_tiles_);
          e.buf(3, moe_ntiles_);
          e.buf(4, moe_rank_);
          e.val(5, bp);
          e.dispatch(1, 1, 1, 256);
        }
        const uint32_t w13_q_stride = L.is_mxfp8 ? (2 * I * H) : (2 * I * (H / 2));
        const uint32_t w13_s_stride = L.is_mxfp8 ? (2 * I * (H / 32)) : (2 * I * (H / 16));
        {
          const Ink2tGemmP gp{P, 2 * I, H, H, 2 * I, 0u, w13_q_stride, w13_s_stride, uint32_t(cfg_.top_k)};
          const bool rp = !L.is_mxfp8 && moe_lean();
          e.use(base_pipe((L.is_mxfp8 ? "ink2_t_mxfp8_moe_w13" : "ink2_t_nvfp4_moe_w13") + ms + (rp ? "_rp" : "")));
          e.buf(0, x_mlp_);
          e.buf(1, rp ? L.w13_rp : L.w13->buf);
          e.buf(2, L.w13_sc->buf);
          e.buf(3, moe_tmp_);
          e.val(4, gp);
          e.buf(5, moe_tiles_);
          e.buf(6, moe_ntiles_);
          e.buf(7, moe_sorted_);
          e.buf(8, L.is_mxfp8 ? x_mlp_ : L.w13_outer);
          e.dispatch((2 * I + 63) / 64, max_tiles, 1, 128);
        }
        {
          const Ink2SituP sp{P * I};
          e.use(ink2_pipe("ink2_moe_situ_half"));
          e.buf(0, moe_tmp_);
          e.buf(1, moe_act_h_);
          e.val(2, sp);
          e.dispatch((sp.n + 255) / 256, 1, 1, 256);
        }
        const uint32_t w2_q_stride = L.is_mxfp8 ? (H * I) : (H * (I / 2));
        const uint32_t w2_s_stride = L.is_mxfp8 ? (H * (I / 32)) : (H * (I / 16));
        {
          const Ink2tGemmP gp{P, H, I, I, H, 0u, w2_q_stride, w2_s_stride, uint32_t(cfg_.top_k)};
          const bool rp = !L.is_mxfp8 && moe_lean();
          e.use(base_pipe((L.is_mxfp8 ? "ink2_t_mxfp8_moe_w2" : "ink2_t_nvfp4_moe_w2") + ms + (rp ? "_rp" : "")));
          e.buf(0, moe_act_h_);
          e.buf(1, rp ? L.w2_rp : L.w2->buf);
          e.buf(2, L.w2_sc->buf);
          e.buf(3, moe_tmp_);
          e.val(4, gp);
          e.buf(5, moe_tiles_);
          e.buf(6, moe_ntiles_);
          e.buf(7, moe_sorted_);
          e.buf(8, L.is_mxfp8 ? x_mlp_ : L.w2_outer);
          e.dispatch((H + 63) / 64, max_tiles, 1, 128);
        }
        {
          const Ink2CombineP cp{uint32_t(T), H, uint32_t(cfg_.top_k)};
          e.use(ink2_pipe("ink2_moe_combine"));
          e.buf(0, moe_tmp_);
          e.buf(1, ew_);
          e.buf(2, moe_out_);
          e.val(3, cp);
          e.dispatch(uint32_t(T), 1, 1, 256);
        }
      } else {
        const Ink2MoeP mp{uint32_t(T), H, uint32_t(cfg_.moe_I), uint32_t(cfg_.top_k)};
        if (!L.is_mxfp8) {
          // NI_INK2_MOEBT=0: the per-slot LUT kernels (bit-decoded wide kernels: lab4, -2.5 ms/token at 8k)
          static const bool moebt = ink2_knob("NI_INK2_MOEBT", 1, 0) != 0;
          const bool bt = moebt && H % 256 == 0 && cfg_.moe_I % 16 == 0 && cfg_.moe_I >= 512;
          const uint32_t dec13 = bt ? moe_dec(false) : 0u, dec2 = bt ? moe_dec(true) : 0u;
          const std::string d13 = dec13 ? "_d" + std::to_string(dec13) : "", d2 = dec2 ? "_d" + std::to_string(dec2) : "";
          e.use(ink2_pipe(bt ? "ink2_moe_nvfp4_w13_q4" + d13 : moe_lut() ? "ink2_moe_nvfp4_w13_situ_l" : "ink2_moe_nvfp4_w13_situ"));
          e.buf(0, x_mlp_);
          e.buf(1, eid_);
          e.buf(2, dec13 & 1u ? L.w13_rp : L.w13->buf);
          e.buf(3, L.w13_sc->buf);
          e.buf(4, L.w13_outer);
          e.buf(5, moe_act_);
          e.val(6, mp);
          e.dispatch((uint32_t(cfg_.moe_I) + 15) / 16, uint32_t(cfg_.top_k), uint32_t(T), 128);

          // NI_INK2_W2R=R (2|4|8) rows per simdgroup, NI_INK2_W2B=B (1|2) blocks per lane step (lab7: act re-read per
          // 2 rows was L2-bound; R at B = 1 keeps f2's per-row sums bit-identical)
          static const uint32_t w2r = getenv("NI_INK2_W2R") ? uint32_t(atoi(getenv("NI_INK2_W2R"))) : 2u;
          static const uint32_t w2b = getenv("NI_INK2_W2B") ? uint32_t(atoi(getenv("NI_INK2_W2B"))) : 1u;
          const bool wr = bt && (w2r == 2u || w2r == 4u || w2r == 8u) && (w2r != 2u || w2b == 2u) &&
                          (w2b == 1u || (w2b == 2u && cfg_.moe_I % 32 == 0));
          // NI_INK2_SCFOLD (mask 1 wo / w2, 2 SWA k / v, +4 / +8 drop wo / w2; 0 = own stages): mlp_sconv in the w2 epilogue
          // (one row, one sequence). W2R 4 / 8 and B = 2 were slower in-model (lab7: +0.6 / +1.8 ms), R = 1 +1.9 ms
          const bool scfold = (ink2_scfold() & 9) == 1;
          const bool rok = (w2r == 2u || w2r == 4u || w2r == 8u) && (w2b == 1u || (w2b == 2u && cfg_.moe_I % 32 == 0));
          mlp_sc_fused = bt && rok && scfold && T == 1 && ng == 1;
          const bool wr2 = wr || mlp_sc_fused;
          const uint32_t w2rows = wr2 ? w2r : 2u;
          e.use(ink2_pipe(wr2 ? "ink2_moe_nvfp4_w2_r" + std::to_string(w2r) + "b" + std::to_string(w2b) +
                                    (mlp_sc_fused ? "_sc" : "") + d2
                          : bt ? "ink2_moe_nvfp4_w2_f2" + d2 : moe_lut() ? "ink2_moe_nvfp4_w2_acc_l" : "ink2_moe_nvfp4_w2_acc"));
          e.buf(0, moe_act_);
          e.buf(1, eid_);
          e.buf(2, ew_);
          e.buf(3, dec2 & 1u ? L.w2_rp : L.w2->buf);
          e.buf(4, L.w2_sc->buf);
          e.buf(5, L.w2_outer);
          e.buf(6, moe_out_);
          e.val(7, mp);
          if (mlp_sc_fused) {
            const Ink2SconvStaticP msp{1u, H, L.mlp_sconv->dtype == DType::F32 ? 1u : 0u, 1u, sconv_static_scale,
                                       snap_stride};
            e.buf(8, L.mlp_sconv->buf);
            e.buf(9, gconv(0, 6 * KVD * 4 + 3 * H * 4, 3 * H * 4));
            e.buf(10, *alt_x);
            e.val(11, msp);
            e.buf(12, ffn_inp_);
            e.buf(13, snap_stride ? conv_snap_.view(cbase + 6 * KVD * 4 + 3 * H * 4,
                                                    conv_snap_.size - (cbase + 6 * KVD * 4 + 3 * H * 4))
                                  : conv_m);
          }
          e.dispatch(bt ? (H + 4 * w2rows - 1) / (4 * w2rows) : (H + 15) / 16, uint32_t(T), 1, 128);
        } else {
          static const bool mxv = ink2_knob("NI_INK2_MXV", 1, 0) != 0;
          const bool v8 = mxv && H % 256 == 0 && cfg_.moe_I % 128 == 0;  // (16-byte loads + LUT; NI_INK2_MXV=0 old)
          e.use(ink2_pipe(v8 ? "ink2_moe_mxfp8_w13_situ_v" : "ink2_moe_mxfp8_w13_situ"));
          e.buf(0, x_mlp_);
          e.buf(1, eid_);
          e.buf(2, L.w13->buf);
          e.buf(3, L.w13_sc->buf);
          e.buf(4, moe_act_);
          e.val(5, mp);
          e.dispatch((uint32_t(cfg_.moe_I) + 3) / 4, uint32_t(cfg_.top_k), uint32_t(T), 128);

          e.use(ink2_pipe(v8 ? "ink2_moe_mxfp8_w2_acc_v" : "ink2_moe_mxfp8_w2_acc"));
          e.buf(0, moe_act_);
          e.buf(1, eid_);
          e.buf(2, ew_);
          e.buf(3, L.w2->buf);
          e.buf(4, L.w2_sc->buf);
          e.buf(5, moe_out_);
          e.val(6, mp);
          e.dispatch(v8 ? (H + 15) / 16 : (H + 3) / 4, uint32_t(T), 1, 128);
        }
      }
    }

    // mlp_sconv + residual add -> *alt_x, then swap(cur_x, alt_x)
    if (!mlp_sc_fused) {
      for (int gi = 0; gi < ng; ++gi) {
      const Ink2SconvStaticP msp{Tg, H, L.mlp_sconv->dtype == DType::F32 ? 1u : 0u, 1u, sconv_static_scale,
                                 snap_stride};
      e.use(ink2_pipe(T >= 16 ? std::string("ink2_sconv_static_pf") : ink2_sconv_name()));
      e.buf(0, rv(moe_out_, H * 4, gi));
      e.buf(1, L.mlp_sconv->buf);
      e.buf(2, gconv(gi, 6 * KVD * 4 + 3 * H * 4, 3 * H * 4));
      e.buf(3, rv(*alt_x, H * 4, gi));
      e.val(4, msp);
      e.buf(5, rv(ffn_inp_, H * 4, gi));
      e.buf(6, snap_stride ? conv_snap_.view(cbase + 6 * KVD * 4 + 3 * H * 4,
                                             conv_snap_.size - (cbase + 6 * KVD * 4 + 3 * H * 4))
                           : conv_m);
      if (T >= 16) {
        e.dispatch((H + 127) / 128, (uint32_t(T) + 15) / 16, 1, 128);
        enc_copy_buf(e, moe_out_.view(size_t(T - 3) * H * 4, 3 * size_t(H) * 4), conv_m, 3 * size_t(H) * 4);
      } else {
        e.dispatch((H + 255) / 256, 1, 1, 256);
      }
      }
    }
    std::swap(cur_x, alt_x);
    if (alt_x == &x_pre_) alt_x = &x_;

    if (diag_rows) {
      int dslot = -1;
      if (v_idx == 0) dslot = 1;
      else if (v_idx == 5) dslot = 2;
      else if (v_idx == 17) dslot = 3;
      else if (v_idx == 18) dslot = 4;
      else if (v_idx == 30) dslot = 5;
      else if (v_idx == 44) dslot = 6;
      if (dslot >= 0)
        enc_copy_buf(e, cur_x->view(size_t(T - 1) * H * 4, H * 4), diag_rows->view(size_t(dslot) * H * 4, H * 4), H * 4);
    }

    constexpr uint32_t split_every = 8u;
    if (T >= 256 && (v_idx % split_every) == split_every - 1 && v_idx + 1 < cfg_.schedule.size()) e.split();
  }

  if (!dump_logits && !need_all_logits && !taps_out) return;

  // Final AttnRes + RMSNorm
  bool final_normed = false;
  {
    std::vector<const Buffer*> st = blocks_vec;
    st.push_back(cur_x);
    const uint32_t src0[3] = {0u, 0u, 0u};
    const Tensor* nw[3] = {final_norm_, nullptr, nullptr};
    const Buffer* no[3] = {&h_norm_, nullptr, nullptr};
    final_normed = enc_attn_res_norm(e, st.data(), int(st.size()), &output_query_->buf,
                                     output_query_->dtype == DType::F32, 1, ar_out_, T, 1, src0, nw, no);
    if (!final_normed)
      enc_attn_res(e, st.data(), int(st.size()), &output_query_->buf, output_query_->dtype == DType::F32, 1, ar_out_, T);
  }
  if (!final_normed) enc_norm(e, ar_out_, final_norm_, h_norm_, T, H);
  if (taps_out) enc_copy_buf(e, h_norm_, taps_out->view(3 * row_bytes, row_bytes), row_bytes);
  if (diag_rows) enc_copy_buf(e, h_norm_.view(size_t(T - 1) * H * 4, H * 4), diag_rows->view(7 * H * 4, H * 4), H * 4);
  if (!dump_logits && !need_all_logits) return;

  const uint32_t V = uint32_t(cfg_.vocab);
  const Buffer& dst = out_logits ? *out_logits : logits_;
  const int rows_out = need_all_logits ? T : 1;
  const Buffer h_in = need_all_logits ? h_norm_ : h_norm_.view(size_t(T - 1) * H * 4, size_t(H) * 4);
  // NI_INK2_CAPFOLD (lab8, default 1): one row, Q8 unembed: the softcap in the matvec's epilogue (ink2_q8_mv1_cap)
  static const int capfold_env = ink2_knob("NI_INK2_CAPFOLD", 1, 0);
  static const uint32_t r1n = getenv("NI_INK2_Q8R_MAXN") ? uint32_t(atoi(getenv("NI_INK2_Q8R_MAXN"))) : 4096u;
  const Ink2Mat& uq = unembed_;
  const bool capfold = ink2_ab("NI_INK2_CAPFOLD", capfold_env) != 0 && rows_out == 1 && (q8m_ && uq.is_q8() && uq.K % 256 == 0) && V > r1n;
  if (capfold) {
    const Ink2CapP cp{V, uq.K, cfg_.logit_softcap, 1.0f / cfg_.logit_softcap};
    e.use(ink2_pipe("ink2_q8_mv1_cap"));
    e.buf(0, h_in);
    e.buf(1, uq.q);
    e.buf(2, uq.s);
    e.buf(3, dst);
    e.val(4, cp);
    e.dispatch((V + 15) / 16, 1, 1, 128);
    return;
  }
  enc_mm(e, unembed_, h_in, dst, rows_out, V, false);
  {
    const Ink2SoftcapP sp{uint32_t(rows_out) * V, cfg_.logit_softcap, 1.0f / cfg_.logit_softcap};
    e.use(ink2_pipe("ink2_softcap"));
    e.buf(0, dst);
    e.val(1, sp);
    e.dispatch((sp.n + 255) / 256, 1, 1, 256);
  }
}

void Inkling2::enc_sample(Encoder& e, Seq& s, const Buffer& lrow, const Buffer& tokens, int out_index,
                          const SamplingParams& sp, uint32_t step) {
  enc_logit_mask(e, s, lrow, 1, cfg_.vocab, cfg_.vocab);
  if (full_vocab(sp)) {
    enc_sample_full(e, lrow, uint32_t(cfg_.vocab), tokens, out_index, sprob_, sp, step);
    return;
  }
  const uint32_t K = topk_k(sp);
  const kp::TopkStage ts{uint32_t(cfg_.vocab), K, topk_chunk(K)};
  const uint32_t ntg = topk_lists(uint32_t(cfg_.vocab), K);
  e.use(base_pipe("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, topk_threads(K));
  const kp::Sample p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(base_pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void Inkling2::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos, 1, sample, nullptr, false);
  if (!sample) return;
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);
  enc_sample(e, s, logits_, s.tok, pos + 1, sp, step);
}

void Inkling2::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                              Checkpoint* snap, int) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  if (snap) fatal("inkling2: in-pass snapshots are not supported");
  if (s.embeds) fatal("inkling2: image inputs are not supported in Stage 1");
  enc_forward(e, s, pos0, T, sample, nullptr, false);
  if (!sample) return;
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);
  enc_sample(e, s, logits_, s.tok, pos0 + T, sp, step);
}

void Inkling2::encode_verify(Encoder& e, Seq& s, int pos0, int T, const Buffer& out_logits, const Buffer* taps_out,
                             bool save_conv_snap) {
  if (T < 1 || T > kMaxVerify) fatal("inkling2: verify T=%d out of range", T);
  if (pos0 + T > s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos0, T, true, &out_logits, true, nullptr, taps_out, save_conv_snap);
  enc_logit_mask(e, s, out_logits, T, cfg_.vocab, cfg_.vocab);
}

void Inkling2::encode_conv_rollback(Encoder& e, Seq& s, int keep_idx, int total_rows) {
  if (keep_idx < 0 || keep_idx >= total_rows || !conv_snap_) return;
  if (keep_idx == total_rows - 1) return;
  const size_t cb = conv_bytes_per_seq(cfg_);
  enc_copy_buf(e, conv_snap_.view(size_t(keep_idx) * cb, cb), s.conv[0], cb);
}

void Inkling2::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  if (T > tmax_) fatal("logits_rows chunk %d > %d", T, tmax_);
  if (pos0 + T > s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos0, T, true, &out, true);
}

void Inkling2::encode_prefill_diag(Encoder& e, Seq& s, int pos0, int T, const Buffer& diag_rows) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  enc_forward(e, s, pos0, T, true, nullptr, false, &diag_rows);
}

}  // namespace ni
