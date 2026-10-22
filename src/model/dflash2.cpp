// DFlash2 block drafter: loading, load-time quantization, context ring and the drafting pass.
// Kernels in src/kernels/spec_dflash.metal.
// Techniques: MLX af4g64 quantization of the drafter at load, the per-row window bound of the draft
// attention, exact-code MMA operands, a separate block K/V scratch.
#include "model/dflash2.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <mach/mach.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unordered_map>

namespace ni {

extern const char* const kTensorKernelSource;
using json = nlohmann::json;

static bool default_v8(size_t hidden, bool moe, bool head_q4) {
  static const bool m1_max32 = Gpu::get().name() == "Apple M1 Max" && Gpu::get().core_count() == 32;
  return hidden >= 2560 || (m1_max32 && hidden == 2048 && moe && head_q4);
}

// process physical footprint (what Activity Monitor / bench.py report), bytes
static double footprint_gb() {
  task_vm_info_data_t info;
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return 0;
  return double(info.phys_footprint) / 1e9;
}

static Pipeline dpipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

// SPEC §1 checks of a drafter's config.json against the target (the kernels hard-code this geometry): the
// failed checks, empty for a supported drafter. find() applies them too, so the default ("auto") only picks
// drafters that load.
static std::vector<std::string> config_problems(const json& c, const Qwen35Config& t) {
  std::vector<std::string> bad;
  auto check = [&](bool ok, const char* what) {
    if (!ok) bad.push_back(what);
  };
  const json dc = c.value("dflash_config", json::object());
  // How a target layer's output becomes the hidden row the draft reads, as it was trained (Splash 1.4.0's
  // dflash_config.target_hidden_reduction): a hyper-connected target's streams by their mean (stream_mean: Qwen3.8-
  // Flash-Next, Qwen4Exp::enc_capture4), another's row as it is (none). A config without it takes its target's.
  {
    const std::string want = DFlash2::target_reduction(t);
    if (dc.contains("target_hidden_reduction") &&
        !(dc["target_hidden_reduction"].is_string() && dc["target_hidden_reduction"].get<std::string>() == want))
      bad.push_back("target_hidden_reduction " + want + " (the target's)");
  }
  if (c.value("architectures", json::array()) == json::array({"DFlashDraftModel"})) {
    // DFlash (v1): Qwen3 decoder layers over [context features; block], no conv / selector (see DFlash2Config::v1)
    const int NL = c.value("num_hidden_layers", 0), H = c.value("hidden_size", 0), I = c.value("intermediate_size", 0);
    const int vocab = c.value("vocab_size", 0), nq = c.value("num_attention_heads", 0),
              nkv = c.value("num_key_value_heads", 0), hd = c.value("head_dim", nq ? H / nq : 0);
    const int mask = dc.value("mask_token_id", -1);
    double theta = 0;
    std::string rope_type = "default";
    if (c.contains("rope_parameters")) {
      theta = c["rope_parameters"].value("rope_theta", 0.0);
      rope_type = c["rope_parameters"].value("rope_type", "default");
    } else {
      theta = c.value("rope_theta", 0.0);
    }
    check(c.contains("dflash_config"), "dflash_config");
    check(!(dc.contains("causal") && dc["causal"] == true) && !(c.contains("is_causal") && c["is_causal"] == true),
          "non-causal block");
    bool types = NL > 0 && (!c.contains("layer_types") || c["layer_types"].size() == size_t(NL));
    if (types && c.contains("layer_types"))
      for (auto& lt : c["layer_types"]) types = types && (lt == "sliding_attention" || lt == "full_attention");
    check(types, "layer types sliding / full attention");
    check(dc.value("block_size", DFlash2::kB) >= DFlash2::kB, "block >= 8");
    check(theta > 0 && rope_type == "default", "default rope");
    check(nq == 32 && nkv == 8 && hd == 128, "heads 32/8 x 128");
    check(c.value("rms_norm_eps", 1e-6f) == 1e-6f, "eps 1e-6");
    check(c.value("final_logit_softcapping", json()).is_null() && !dc.value("attention_sink_bias", false),
          "no softcap, no attention sinks");
    check(H == t.hidden, "draft hidden == target hidden");
    check(vocab == t.vocab, "vocab == target vocab");
    bool in_range = dc.contains("target_layer_ids") && dc["target_layer_ids"].is_array() && !dc["target_layer_ids"].empty();
    if (in_range)
      for (auto& v : dc["target_layer_ids"]) in_range = in_range && v.is_number_integer() && v >= 0 && v < t.layers;
    check(in_range, "target_layer_ids in range");
    check(H % 64 == 0 && I % 64 == 0, "hidden/intermediate multiples of 64");
    check(mask >= 0 && mask < vocab, "mask id");
    return bad;
  }
  const int NL = c.value("num_hidden_layers", 0), H = c.value("hidden_size", 0), I = c.value("intermediate_size", 0);
  const int vocab = c.value("vocab_size", 0), nq = c.value("num_attention_heads", 0), nkv = c.value("num_key_value_heads", 0);
  const int hd = c.value("head_dim", nq ? H / nq : 0), window = c.value("sliding_window", 0);
  const int mask = dc.value("mask_token_id", -1);
  double theta = 0;
  std::string rope_type = "default";
  if (c.contains("rope_parameters")) {
    theta = c["rope_parameters"].value("rope_theta", 0.0);
    rope_type = c["rope_parameters"].value("rope_type", "default");
  } else {
    theta = c.value("rope_theta", 0.0);
  }
  check(c.value("architectures", json::array()) == json::array({"DFlash2DraftModel"}), "architecture");
  check(c.contains("dflash_config"), "dflash_config");
  check(c.contains("is_causal") && c["is_causal"] == false, "is_causal false");
  bool sliding = NL > 0 && c.contains("layer_types") && c["layer_types"].size() == size_t(NL);
  if (sliding)
    for (auto& lt : c["layer_types"]) sliding = sliding && lt == "sliding_attention";
  check(sliding && window == DFlash2::kWin, "all layers sliding, window 2048");
  check(int(dc.value("conv_kernel_size", 0)) == 2, "conv taps 2");
  check(dc.value("conv_group_size", 0) == 16, "conv group 16");
  check(dc.value("selector_rank", 0) == DFlash2::kRank && dc.value("selector_top_k", 0) == DFlash2::kTopK,
        "selector rank 256 / top-k 16");
  check(dc.value("block_size", 0) == DFlash2::kB, "block 8");
  // any theta: the inverse frequencies come from it at load (inv_freq_; incoai's Flash-Next draft has 1e4, the others 1e7)
  check(theta > 0 && rope_type == "default", "default rope");
  check(nq == 32 && nkv == 8 && hd == 128, "heads 32/8 x 128");
  check(c.value("rms_norm_eps", 1e-6f) == 1e-6f, "eps 1e-6");
  check(dc.value("input_embedding_scale", 1.0) == 1.0 && dc.value("output_multiplier", 1.0) == 1.0 &&
            dc.value("final_logit_softcapping", json()).is_null() && c.value("final_logit_softcapping", json()).is_null(),
        "neutral embedding/output scales, no softcap");
  check(!c.value("tie_word_embeddings", false), "untied");
  check(H == t.hidden, "draft hidden == target hidden");
  check(vocab == t.vocab, "vocab == target vocab");
  bool in_range = dc.contains("target_layer_ids") && dc["target_layer_ids"].is_array() && !dc["target_layer_ids"].empty();
  if (in_range)
    for (auto& v : dc["target_layer_ids"]) in_range = in_range && v.is_number_integer() && v >= 0 && v < t.layers;
  check(in_range, "target_layer_ids in range");
  check(H % 64 == 0 && I % 64 == 0, "hidden/intermediate multiples of 64");
  check(mask >= 0 && mask < vocab, "mask id");
  return bad;
}

std::string DFlash2::target_reduction(const Qwen35Config& t) { return t.hc > 1 ? "stream_mean" : "none"; }

std::vector<std::string> DFlash2::config_check(const std::string& config_json, const Qwen35Config& target) {
  const json c = json::parse(config_json, nullptr, false);
  if (c.is_discarded() || !c.is_object()) return {"config.json is not a JSON object"};
  if (!c.contains("dflash_config")) return {"dflash_config"};
  return config_problems(c, target);
}

void DFlash2::load_config(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) fatal("DFlash2: cannot read %s/config.json", dir.c_str());
  json c = json::parse(f);
  if (!c.contains("dflash_config")) fatal("DFlash2: %s has no dflash_config", dir.c_str());
  const std::vector<std::string> bad = config_problems(c, tg_.config());
  if (!bad.empty()) {
    std::string s;
    for (auto& b : bad) s += " [" + b + "]";
    fatal("DFlash2: unsupported drafter config in %s:%s", dir.c_str(), s.c_str());
  }
  const json& dc = c["dflash_config"];
  cfg_.v1 = c.value("architectures", json::array()) == json::array({"DFlashDraftModel"});
  cfg_.H = c["hidden_size"];
  cfg_.NL = c["num_hidden_layers"];
  cfg_.I = c["intermediate_size"];
  cfg_.vocab = c["vocab_size"];
  cfg_.nq = c["num_attention_heads"];
  cfg_.nkv = c["num_key_value_heads"];
  cfg_.hd = c.value("head_dim", cfg_.H / cfg_.nq);
  cfg_.eps = c.value("rms_norm_eps", 1e-6f);
  cfg_.mask_id = dc["mask_token_id"];
  for (auto& v : dc["target_layer_ids"]) cfg_.target_layers.push_back(v);
  cfg_.stream_mean = target_reduction(tg_.config()) == "stream_mean";
  if (cfg_.v1) {  // runs with this pipeline's 8-row block and 2048-position ring (DFlash2Config::v1)
    cfg_.trained_block = dc.value("block_size", 16);
    cfg_.trained_window = c.value("sliding_window", 0);
    cfg_.block = kB;
    cfg_.window = kWin;
  } else {
    cfg_.block = dc["block_size"];
    cfg_.conv_gs = dc["conv_group_size"];
    cfg_.rank = dc["selector_rank"];
    cfg_.topk = dc["selector_top_k"];
    cfg_.window = c.value("sliding_window", 0);
  }
  if (c.contains("rope_parameters")) cfg_.theta = float(c["rope_parameters"].value("rope_theta", 0.0));
  else cfg_.theta = float(c.value("rope_theta", 0.0));
}

// The target's embedding table is read by dfl_embed_norm (4-bit group 64) or its wfmt.metal form for the other
// formats; the draft head is the target's LM head (4/8-bit group 64), or an 8-bit copy of its rows for the other
// formats (bf16, 6-bit, ...: drafting only, the verify uses the target's own head).
bool DFlash2::supports(const Qwen35& t) {
  const QLinear& emb = t.embed_table();
  const QLinear& head = t.lm_head();
  return (emb.q4() || format_supported(emb.bits, emb.gs)) && format_supported(head.bits, head.gs) &&
         head.K == uint32_t(t.config().hidden) && head.K % 64 == 0;
}

// One row of a linear in any format as bf16 (CPU; draft-head conversion).
void dfl_row_bf16(const QLinear& H, uint32_t row, uint16_t* out) {
  if (H.dense()) {
    memcpy(out, H.w->buf.as<uint8_t>() + size_t(row) * H.row_bytes(), size_t(H.K) * 2);
    return;
  }
  const uint32_t* W = reinterpret_cast<const uint32_t*>(H.w->buf.as<uint8_t>() + size_t(row) * H.row_bytes());
  const uint16_t* S = H.s->buf.as<uint16_t>() + size_t(row) * (H.K / H.gs);
  const uint16_t* B = H.b->buf.as<uint16_t>() + size_t(row) * (H.K / H.gs);
  const uint32_t mask = (1u << H.bits) - 1u;
  for (uint32_t k = 0; k < H.K; ++k) {
    const uint32_t bit = k * uint32_t(H.bits), sh = bit % 32;
    uint32_t v = W[bit / 32] >> sh;
    if (sh + uint32_t(H.bits) > 32) v |= W[bit / 32 + 1] << (32 - sh);
    out[k] = f32_to_bf16(bf16_to_f32(S[k / H.gs]) * float(v & mask) + bf16_to_f32(B[k / H.gs]));
  }
}

const Tensor* DFlash2::own(const Tensor& t, const char* label) {
  Buffer b = Gpu::get().alloc(t.buf.size, label, /*tracked=*/false);
  memcpy(b.ptr(), t.buf.ptr(), t.buf.size);
  owned_bufs_.push_back(b);
  resident_ += b.size;
  Tensor c = t;
  c.buf = b;
  owned_.push_back(c);
  return &owned_.back();
}

// MLX affine quantization of bf16 rows (see DFlash2::quantize); multithreaded over rows.
void dfl_quantize_cpu(const std::vector<const uint16_t*>& rowp, uint32_t K, int bits, bool even, uint32_t* W,
                      uint16_t* S, uint16_t* B) {
  const uint32_t N = uint32_t(rowp.size()), wpr = K * bits / 32, gpr = K / 64;
  auto rnd = [even](float x) { return even ? std::nearbyint(x) : std::round(x); };
  const float n_bins = float((1 << bits) - 1);
  auto work = [&](uint32_t r0, uint32_t r1) {
    float v[64];
    for (uint32_t n = r0; n < r1; ++n) {
      const uint16_t* src = rowp[n];
      for (uint32_t gi = 0; gi < gpr; ++gi) {
        float lo = 3.38953139e38f, hi = 0.0f;  // MLX starts from (Limits<bf16>::max, 0)
        for (int k = 0; k < 64; ++k) {
          v[k] = bf16_to_f32(src[gi * 64 + k]);
          lo = std::min(lo, v[k]);
          hi = std::max(hi, v[k]);
        }
        float scale = std::max((hi - lo) / n_bins, 1e-7f);
        const bool side = std::fabs(lo) > std::fabs(hi);
        scale = side ? scale : -scale;
        const float edge = side ? lo : hi;
        const float q0 = rnd(edge / scale);
        const bool at_zero = q0 == 0.0f;
        scale = at_zero ? scale : edge / q0;
        const float bias = at_zero ? 0.0f : edge;
        S[size_t(n) * gpr + gi] = f32_to_bf16(scale);
        B[size_t(n) * gpr + gi] = f32_to_bf16(bias);
        uint32_t* wr = W + size_t(n) * wpr + gi * (64 * bits / 32);
        const int per = 32 / bits;
        for (int w = 0; w < 64 / per; ++w) {
          uint32_t word = 0;
          for (int j = 0; j < per; ++j) {
            float c = std::min(rnd((v[w * per + j] - bias) / scale), n_bins);
            c = std::max(c, 0.0f);
            word |= uint32_t(c) << (bits * j);
          }
          wr[w] = word;
        }
      }
    }
  };
  if (N < 64) {  // a few rows (single rows gathered while decoding): no threads
    work(0, N);
    return;
  }
  const unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  const uint32_t per = (N + nth - 1) / nth;
  for (unsigned t = 0; t < nth; ++t) {
    uint32_t a = std::min(N, t * per), b = std::min(N, a + per);
    if (a < b) th.emplace_back(work, a, b);
  }
  for (auto& t : th) t.join();
}

// bf16 rows (the parts stacked along N) -> MLX affine `bits_`-bit, group 64, with exactly
// mx.quantize's rounding (MLX Metal affine_quantize: lo = min(w), hi = max(max(w), 0),
// scale = max((hi - lo) / n_bins, 1e-7) signed toward the larger-magnitude edge, rescaled so the
// edge is exact, bias = edge; codes = min(round((w - bias) / scale), n_bins) in IEEE fp32 with
// round-half-away; scale/bias stored as bf16 RNE). MLX's CPU backend rounds halves to even
// instead (NI_DFLASH_QROUND=even reproduces that one).
QLinear DFlash2::quantize(const std::vector<const Tensor*>& parts, const char* label, const std::vector<uint32_t>& perm) {
  const uint32_t K = uint32_t(parts[0]->shape[1]);
  uint32_t N = 0;
  for (auto* t : parts) {
    if (t->dtype != DType::BF16 || t->shape.size() != 2 || uint32_t(t->shape[1]) != K)
      fatal("DFlash2 quantize: %s parts must be 2-D bf16 with equal K", label);
    N += uint32_t(t->shape[0]);
  }
  if (K % 64 || N % 8) fatal("DFlash2 quantize: %s [%u, %u] not a multiple of (8, 64)", label, N, K);
  const int bits = bits_;
  const uint32_t wpr = K * bits / 32, gpr = K / 64;
  Gpu& g = Gpu::get();
  Buffer wq = g.alloc(size_t(N) * wpr * 4, label, false);
  Buffer sc = g.alloc(size_t(N) * gpr * 2, label, false);
  Buffer bi = g.alloc(size_t(N) * gpr * 2, label, false);
  static const bool even = getenv("NI_DFLASH_QROUND") && std::string(getenv("NI_DFLASH_QROUND")) == "even";
  std::vector<const uint16_t*> rowp(N);
  {
    uint32_t r = 0;
    for (auto* t : parts)
      for (int64_t i = 0; i < t->shape[0]; ++i) rowp[r++] = t->buf.as<uint16_t>() + size_t(i) * K;
  }
  if (!perm.empty()) {  // stored row i = source row perm[i]
    std::vector<const uint16_t*> src = rowp;
    for (uint32_t i = 0; i < N; ++i) rowp[i] = src[perm[i]];
  }
  dfl_quantize_cpu(rowp, K, bits, even, wq.as<uint32_t>(), sc.as<uint16_t>(), bi.as<uint16_t>());
  owned_bufs_.push_back(wq);
  owned_bufs_.push_back(sc);
  owned_bufs_.push_back(bi);
  resident_ += wq.size + sc.size + bi.size;
  QLinear L;
  owned_.push_back(Tensor{wq, DType::U32, {int64_t(N), int64_t(wpr)}});
  L.w = &owned_.back();
  owned_.push_back(Tensor{sc, DType::BF16, {int64_t(N), int64_t(gpr)}});
  L.s = &owned_.back();
  owned_.push_back(Tensor{bi, DType::BF16, {int64_t(N), int64_t(gpr)}});
  L.b = &owned_.back();
  L.N = N;
  L.K = K;
  L.bits = bits;
  L.gs = 64;
  return L;
}

QLinear DFlash2::row_view(const QLinear& src, uint32_t r0, uint32_t n) {
  QLinear v = src;
  const size_t wrow = size_t(src.K) * src.bits / 8, srow = size_t(src.K / src.gs) * 2;
  owned_.push_back(Tensor{src.w->buf.view(r0 * wrow, n * wrow), src.w->dtype, {int64_t(n), int64_t(wrow / 4)}});
  v.w = &owned_.back();
  owned_.push_back(Tensor{src.s->buf.view(r0 * srow, n * srow), src.s->dtype, {int64_t(n), int64_t(src.K / src.gs)}});
  v.s = &owned_.back();
  owned_.push_back(Tensor{src.b->buf.view(r0 * srow, n * srow), src.b->dtype, {int64_t(n), int64_t(src.K / src.gs)}});
  v.b = &owned_.back();
  v.N = n;
  return v;
}

static bool dfl_tw_supported();  // below: the drafter's GEMMs through Metal 4 tensor_ops

bool DFlash2::rows16_tensor_ok() const {
  const char* av = getenv("NI_DFLASH_ATTN");
  const bool fused_attn = !(av && std::string(av) == "scalar") && !(v8_knobs().dfl & 1);
  bool tw = !cfg_.v1 && fused_attn && dfl_tw_supported() && v8_knobs().tw && !(v8_knobs().dfl & (32 | 128));
  for (const DLayer& L : layers_)
    for (const QLinear* q : {&L.q, &L.kv, &L.o, &L.gate, &L.up, &L.down})
      tw = tw && q->q4() && q->gs == 64 && q->K % 64 == 0;
  return tw;
}

bool DFlash2::rows16_auto_ok() const {
  // the same conditions the load-time gate below refuses NI_DFLASH_BLOCK=16 under, read from the environment here
  // because set_tree() runs after the block must be known: trees on, and a walk deep enough to use all 16 rows (a
  // 15-row lattice is what the block drafts; set_tree keeps the tree only where verify_depth() == block() - 1)
  if (!tg_.block_drafter_ok() || !tg_.verify_tree_ok() || !tg_.verify_wide_ok()) return false;
  const char* kf = getenv("NI_KFLASH");
  const char* tv = getenv("NI_DFLASH_TREE");
  if (!(tv && *tv ? atoi(tv) != 0 : (!(kf && *kf) || atoi(kf) != 0))) return false;
  const char* vv = getenv("NI_DFLASH_VERIFY");
  if (vv && *vv && atoi(vv) < 15) return false;
  return rows16_tensor_ok();
}

DFlash2::DFlash2(Qwen35& target, const std::string& dir, int bits, bool rows16) : tg_(target), dir_(dir) {
  const double t0 = now_ms(), fp0 = footprint_gb();
  bits_ = bits ? bits : (getenv("NI_DFLASH_BITS") ? atoi(getenv("NI_DFLASH_BITS")) : 4);
  if (bits_ != 4 && bits_ != 8) fatal("DFlash2: bits must be 4 or 8 (got %d)", bits_);
  load_config(dir);
  if (!supports(tg_)) fatal("DFlash2: the target's embedding / LM head format is not supported");
  {
    WeightStore w;  // bf16 checkpoint, released after quantization
    w.open_dir(dir);
    auto get = [&](const std::string& n) -> const Tensor& { return w.get(n); };
    auto expect = [&](const std::string& n, std::vector<int64_t> shape) -> const Tensor& {
      const Tensor& t = get(n);
      if (t.dtype != DType::BF16 || t.shape != shape) fatal("DFlash2: %s has an unexpected dtype/shape", n.c_str());
      return t;
    };
    const int64_t H = cfg_.H, I = cfg_.I, V = cfg_.vocab, G4 = 4 * (cfg_.H / cfg_.conv_gs);
    const int64_t nH = int64_t(cfg_.target_layers.size()) * H;
    const bool v1 = cfg_.v1;
    fc_ = quantize({&expect("fc.weight", {H, nH})}, "dfl_fc");
    if (!v1) sel_w_ = quantize({&expect("candidate_selector.hidden_projection.weight", {kRank, H})}, "dfl_sel");
    hnorm_ = own(expect("hidden_norm.weight", {H}), "dfl_norm");
    norm_ = own(expect("norm.weight", {H}), "dfl_norm");
    if (!v1) {
      cbA_ = own(expect("candidate_selector.predecessor_codebook", {V, kRank}), "dfl_codebook");
      cbB_ = own(expect("candidate_selector.successor_codebook", {V, kRank}), "dfl_codebook");
    }
    // k|v of every layer stacked: one GEMM projects a committed context row for all layers.
    std::vector<const Tensor*> kv_parts;
    for (int l = 0; l < cfg_.NL; ++l) {
      const std::string p = "layers." + std::to_string(l) + ".self_attn.";
      kv_parts.push_back(&expect(p + "k_proj.weight", {1024, H}));
      kv_parts.push_back(&expect(p + "v_proj.weight", {1024, H}));
    }
    kv_all_ = quantize(kv_parts, "dfl_kv");
    {
      Buffer kn = Gpu::get().alloc(size_t(cfg_.NL) * 128 * 2, "dfl_kn", false);
      for (int l = 0; l < cfg_.NL; ++l) {
        const Tensor& t = expect("layers." + std::to_string(l) + ".self_attn.k_norm.weight", {128});
        memcpy(kn.as<uint8_t>() + l * 256, t.buf.ptr(), 256);
      }
      owned_bufs_.push_back(kn);
      owned_.push_back(Tensor{kn, DType::BF16, {int64_t(cfg_.NL), 128}});
      kn_all_ = &owned_.back();
    }
    layers_.resize(cfg_.NL);
    for (int l = 0; l < cfg_.NL; ++l) {
      DLayer& L = layers_[l];
      const std::string p = "layers." + std::to_string(l) + ".";
      L.in_w = own(expect(p + "input_layernorm.weight", {H}), "dfl_norm");
      L.post_w = own(expect(p + "post_attention_layernorm.weight", {H}), "dfl_norm");
      L.qn = own(expect(p + "self_attn.q_norm.weight", {128}), "dfl_norm");
      owned_.push_back(Tensor{kn_all_->buf.view(size_t(l) * 256, 256), DType::BF16, {128}});
      L.kn = &owned_.back();
      if (!v1) {
        L.ac_base = own(expect(p + "attention_conv.base_kernel", {2, 2, H}), "dfl_conv");
        L.mc_base = own(expect(p + "mlp_conv.base_kernel", {2, 2, H}), "dfl_conv");
        // rows permuted for dfl_qmm8_cp: block b < H / 64 = stage 0's taps 0 and 1 of conv groups 4b .. 4b + 3
        std::vector<uint32_t> perm(static_cast<size_t>(G4));
        const uint32_t G = uint32_t(H / cfg_.conv_gs);
        for (uint32_t i = 0; i < uint32_t(G4); ++i)
          perm[i] = i >= 2 * G ? i : (i % 8 < 4 ? 4 * (i / 8) + i % 8 : G + 4 * (i / 8) + i % 8 - 4);
        L.ac = quantize({&expect(p + "attention_conv.kernel_projection.weight", {G4, H})}, "dfl_ac", perm);
        L.mc = quantize({&expect(p + "mlp_conv.kernel_projection.weight", {G4, H})}, "dfl_mc", perm);
      }
      L.q = quantize({&expect(p + "self_attn.q_proj.weight", {4096, H})}, "dfl_q");
      L.o = quantize({&expect(p + "self_attn.o_proj.weight", {H, 4096})}, "dfl_o");
      L.gate = quantize({&expect(p + "mlp.gate_proj.weight", {I, H})}, "dfl_gate");
      L.up = quantize({&expect(p + "mlp.up_proj.weight", {I, H})}, "dfl_up");
      L.down = quantize({&expect(p + "mlp.down_proj.weight", {H, I})}, "dfl_down");
      L.kv = row_view(kv_all_, uint32_t(l) * 2048, 2048);
    }
  }
  // RoPE inverse frequencies in fp32 as HF computes them: 1 / theta^(2i/128).
  inv_freq_ = Gpu::get().alloc(64 * 4, "dfl_inv_freq", false);
  for (int i = 0; i < 64; ++i)
    inv_freq_.as<float>()[i] = 1.0f / std::pow(cfg_.theta, float(2 * i) / 128.0f);
  owned_bufs_.push_back(inv_freq_);
  // capture slots: target layer -> slot
  for (CaptureHook* hk : {&hook_pf_, &hook_vf_}) {
    hk->slot.assign(tg_.config().layers, -1);
    for (size_t i = 0; i < cfg_.target_layers.size(); ++i) hk->slot[cfg_.target_layers[i]] = int(i);
    hk->ld = uint32_t(cfg_.target_layers.size() * cfg_.H);
  }
  // The block, before anything is sized (alloc_buffers() and the reduced head read the rows): NI_DFLASH_BLOCK says 8 or
  // 16, `auto` and unset leave it to the default. Engine asks for 16 on a single-slot engine, where the wider block's
  // cost is paid back by the deeper lattice it drafts (one weight read drafts 15 guesses instead of 7) and there is no
  // batched cycle to lose; a target, a drafter or a knob that takes the drafting pass off the tensor path keeps 8 rows.
  {
    const char* v = getenv("NI_DFLASH_BLOCK");
    const bool said = v && *v && std::string(v) != "auto";  // `auto`: the default, one slot or not
    if (said) {
      block_ = atoi(v);
      if (block_ != kB && block_ != 16) fatal("NI_DFLASH_BLOCK=%s: 8, 16 or auto", v);
    } else if (rows16 && rows16_auto_ok()) {
      block_ = 16;
      logi("DFlash2: 16-row drafting block (one slot: the drafter drafts 15 guesses per weight read and the tree verify "
           "runs 16 rows; NI_DFLASH_BLOCK=8 keeps 8)");
    }
    const char* vv = getenv("NI_DFLASH_VERIFY");
    vdepth_ = std::clamp(vv && *vv ? atoi(vv) : block_ - 1, 1, block_ - 1);
  }
  alloc_buffers();
  {  // reduced draft head (default on): NI_DFLASH_HEAD=n rows (0 = the full vocabulary)
    const char* v = getenv("NI_DFLASH_HEAD");
    if (v && *v) build_head(atoi(v), true);
    else build_head(0, false);
  }
  Gpu::get().make_resident(owned_bufs_);
  // Dynamic verify depth on MoE targets (default tau 0.5, NI_DFLASH_DYN=tau overrides, 0 = off): the expert
  // union, the 35B verify's main cost, shrinks with the rows (35B, 512 tokens: code +7%, prose +20% over
  // verifying all 7; tau 0.4-0.7 swept). Dense targets read every weight once whatever the row count, so
  // there it would only cost acceptance.
  dyn_tau_ = tg_.config().moe ? 0.5f : 0.0f;
  if (const char* v = getenv("NI_DFLASH_DYN")) dyn_tau_ = tg_.config().moe ? float(atof(v)) : 0.0f;
  // T > 0 uses an exact rule (dfl_depth): default tau 0.3 (offline replay, 35B T = 0.6, 4 prompts x 6 seeds: 0.25 /
  // 0.3 / 0.35 within 0.4%); NI_DFLASH_DYN_T overrides. NI_DFLASH_DYN_RULE=2: the earlier T > 0 rule (exclusive
  // product of the walk's q; 35B code_35b 512, T = 0.6: 139.2 tok/s without, 149.4 / 147.8 / 140.3 at 0.3 / 0.5 / 0.7)
  // Rule 0 is the default on the M5 (Apple10), where it was measured; other Macs keep rule 2, which they were tuned
  // with, until it is measured there.
  dyn_tau_t_ = dyn_tau_ > 0 ? 0.3f : 0.0f;
  dyn_rule_ = Gpu::get().apple10() ? 0u : 2u;
  if (const char* v = getenv("NI_DFLASH_DYN_RULE")) {  // rule 1 is biased at T > 0 (dflash_accept_test only)
    const int r = atoi(v);
    if (r != 0 && r != 2) fatal("NI_DFLASH_DYN_RULE=%s: 0 or 2 (rule 1 is biased at T > 0, for tests only)", v);
    dyn_rule_ = uint32_t(r);
  }
  if (const char* v = getenv("NI_DFLASH_DYN_T")) dyn_tau_t_ = tg_.config().moe ? float(atof(v)) : 0.0f;
  if (dyn_tau_ <= 0) dyn_tau_t_ = 0.0f;
  if (const char* v = getenv("NI_DFLASH_EXACT_MOE")) exact_moe_ = atoi(v) != 0;
  if (const char* v = getenv("NI_DFLASH_QT")) walk_tf_ = std::max(0.0f, float(atof(v)));
  // KFlash on every target. The budget's cache key: this model, drafter, bit depth, drafting block and chip (a 16-row
  // block drafts a deeper lattice and verifies wider rows, so its learned capacity must not be the 8-row block's).
  {
    std::string key = tg_.model_dir().substr(tg_.model_dir().find_last_of('/') + 1) + "-" +
                      dir.substr(dir.find_last_of('/') + 1) + "-b" + std::to_string(bits_) +
                      (block() > kB ? "-blk16" : "") + "-" + Gpu::get().name();
    for (char& c : key)
      if (!isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '.') c = '_';
    kf_budget_ = std::make_unique<KfBudget>(key);
    // batched decode (Engine::batch_cycle): the rows of a batch's verify by the same rule; NI_BATCH_KF_FLOOR=1: every
    // member verifies at least one draft
    const char* fl = getenv("NI_BATCH_KF_FLOOR");
    kf_batch_ = std::make_unique<KfBatch>(key + "-batch", fl && *fl ? atoi(fl) : 0);
    // the verify's row counts: whole 8-row tiles. Dense targets: 8 | 16 | 32 where they run wide tree verifies
    // (Qwen35::verify_wide_ok), else one tile (every weight is read once whatever the rows); MoE targets: every count
    // 1 .. 8, a cycle's tree as large as its expected tokens per ms make it (the verify routes experts, the shared
    // expert and the GDN step over the live rows alone: the experts of n rows are most of its cost).
    // NI_KFLASH_ROWS=<list of 8, 16, 32> restricts the dense counts (e.g. "8" = M1's single tile, "16" = always 16 rows)
    const bool moe = tg_.config().moe;
    // the row counts: one 8-row tile on the dense targets that cannot run wide verifies, 8 | 16 | 32 where they can and
    // the knobs (NI_KFLASH, NI_DFLASH_TREE, NI_KFLASH_ROWS) allow them (Qwen35::kf_dense_rows, which plan() reads too,
    // and which rejects a bad NI_KFLASH_ROWS)
    uint32_t allowed = moe ? (1u << kB) - 1 : Qwen35::kf_dense_rows(tg_.config());
    // One rule on every target (here and on Nemotron-H): trees. NI_KFLASH=0 keeps the chains everywhere;
    // NI_DFLASH_TREE=0/1 overrides it here.
    const char* kf = getenv("NI_KFLASH");
    const char* v = getenv("NI_DFLASH_TREE");
    set_tree(v && *v ? atoi(v) != 0 : (!(kf && *kf) || atoi(kf) != 0));
    if (const char* r = getenv("NI_KFLASH_ROWS"); r && *r && !moe) {
      uint32_t want = 0;
      for (const char* c = r; *c;) {
        const int n = atoi(c);
        if (n == 8 || n == 16 || n == 32) want |= 1u << (n - 1);
        while (*c && *c != ',') ++c;
        if (*c == ',') ++c;
      }
      if (want & ~allowed) logi("KFlash: NI_KFLASH_ROWS=%s: this target's tree verify runs 8 rows", r);
    }
    // a drafting block of 16 rows drafts a depth-15 lattice: its verify must run 16+ rows, so the narrow tile goes
    // and the block's own width comes in whatever NI_KFLASH_ROWS asked for
    if (block() > kB) {
      if (!tree_) fatal("NI_DFLASH_BLOCK=%d needs KFlash trees (NI_KFLASH=1)", block());
      if (!tg_.verify_wide_ok()) fatal("NI_DFLASH_BLOCK=%d: this target has no wide tree verify", block());
      // and the drafting pass itself must run on the matrix units: a 16-row block drafts its two tiles with the pair
      // kernel and the tensor-path GEMMs, and has no 8-row fallback (encode_draft_impl and enc_qmm8 fatal otherwise).
      // Said here, while the model loads, not on the first request. The knobs that take the drafting pass or the
      // attention off that path (V8Knobs::dfl bit 1, the old attention; bit 128, the MLP's own qmm8_gu; and
      // NI_DFLASH_ATTN=scalar) count as missing too, so they are refused here rather than on the first cycle.
      if (!rows16_tensor_ok())
        fatal("NI_DFLASH_BLOCK=%d: the drafting pass on a 16-row block needs the tensor path and the fused attention "
              "(not the v1 drafter, the matrix units on, 4-bit group-64 projections, no NI_DFLASH_ATTN=scalar or "
              "V8Knobs::dfl 1 / 128)", block());
      allowed |= 1u << (block() - 1);
      allowed &= ~(1u << (kB - 1));
      // and nothing wider: a 16-row block drafts a 15-guess lattice, so a 32-row verify has nothing more to check,
      // while pin_wide() would pin every cycle to the widest allowed count (2x the rows for the same tree)
      allowed &= (1u << block()) - 1;
    }
    tree_rows_ = 32 - __builtin_clz(allowed);  // the widest allowed count
    kf_budget_->set_rows(allowed, tree_rows_);
    kf_budget_->set_wide(tree_rows_ > kB);  // counts past 8: one wide verify; the budget also picks its capacity
    // a block wider than the 8-row tile drafts the whole lattice in one weight read, so a cycle that verified only 8
    // rows would leave half the draft on the floor: the block pins the capacity (and the size rule's choice of 8 rows,
    // which measured 69.3 tok/s against 90.6 on this Mac, is not a decision the block leaves open)
    if (block() > kB) kf_budget_->pin_wide();
    if (tree_ && tree_rows_ > kB) {
      tg_.set_tree_rows(tree_rows_);
      logi("KFlash: tree verifies of %s rows (the size rule picks per cycle)",
           allowed == ((1u << 7) | (1u << 15) | (1u << 31)) ? "8, 16 or 32" : r_list(allowed).c_str());
    }
  }
  if (dyn_tau_ > 0 && !tree_)
    logi("DFlash2: dynamic verify depth, tau %.3f (T = 0) / %.3f (T > 0)", dyn_tau_, dyn_tau_t_);
  if (cfg_.v1)
    logi("DFlash drafter (v1) %s: blocks of %d rows (trained %d), context window %d positions (trained %d)",
         dir.c_str(), kB, cfg_.trained_block, kCtxKeep, cfg_.trained_window);
  logi("DFlash2 drafter %s: %d layers, hidden %d, %d-bit projections, %.2f GB resident (process footprint %+.2f GB), "
       "loaded in %.0f ms", dir.c_str(), cfg_.NL, cfg_.H, bits_, resident_ / 1e9, footprint_gb() - fp0, now_ms() - t0);
}

void DFlash2::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t H = cfg_.H, I = cfg_.I, V = cfg_.vocab, G4 = 4 * (cfg_.H / cfg_.conv_gs);
  const size_t nH = cfg_.target_layers.size() * H;
  const size_t Kx = std::max({nH, size_t(4096), H, I});
  const size_t KV = size_t(cfg_.NL) * 2048;
  const size_t B = size_t(block()), T = B / 8, D = B - 1;  // block rows, 8-row tiles, lattice depth
  h_ = g.alloc(B * H * 4, "dfl_h");
  n_ = g.alloc(B * H * 4, "dfl_n");
  xh_ = g.alloc(16 * T * Kx * 2, "dfl_xh");  // + the T table (dfl_put2)
  xs_ = g.alloc(T * Kx / 64 * 8 * 4, "dfl_xs");
  xh2_ = g.alloc(16 * T * H * 2, "dfl_xh2");
  xs2_ = g.alloc(T * H / 64 * 8 * 4, "dfl_xs2");
  hh_ = g.alloc(16 * T * I * 2, "dfl_hh");
  gu_g_ = g.alloc(B * I * 4, "dfl_gu_g");  // the matrix-unit gate / up rows (fp32)
  gu_u_ = g.alloc(B * I * 4, "dfl_gu_u");
  hs_ = g.alloc(T * I / 64 * 8 * 4, "dfl_hs");
  dyn_ = g.alloc(B * G4 * 4, "dfl_dyn");
  q_ = g.alloc(B * 4096 * 4, "dfl_q");
  kv_ = g.alloc(B * 2048 * 4, "dfl_kv");
  Q_ = g.alloc(32 * B * 128 * 4, "dfl_Q");
  Kb_ = g.alloc(8 * B * 128 * 4, "dfl_Kb");
  Vb_ = g.alloc(8 * B * 128 * 4, "dfl_Vb");
  part_o_ = g.alloc(size_t(max_splits_) * 32 * B * 128 * 4, "dfl_part_o");
  part_ml_ = g.alloc(size_t(max_splits_) * 32 * B * 2 * 4, "dfl_part_ml");
  u_ = g.alloc(B * H * 4, "dfl_u");
  logits_ = g.alloc(B * V * 4, "dfl_logits");
  sel_ = g.alloc(B * kRank * 4, "dfl_sel");
  const size_t ntg = (V + 2047) / 2048;
  tk_v_ = g.alloc(D * ntg * 16 * 4, "dfl_tk_v");
  tk_i_ = g.alloc(D * ntg * 16 * 4, "dfl_tk_i");
  cand_v_ = g.alloc(D * 16 * 4, "dfl_cand_v");
  cand_i_ = g.alloc(D * 16 * 4, "dfl_cand_i");
  edges_ = g.alloc(D * 256 * 4, "dfl_edges");
  memset(edges_.ptr(), 0, edges_.size);  // v1 (no candidate selector): the walk's edge scores stay 0
  rope_cs_ = g.alloc(B * 64 * 8, "dfl_rope_cs");
  walk_s_ = g.alloc(D * 16 * 4, "dfl_walk_s");
  qprob_ = g.alloc(D * 16 * 4, "dfl_qprob");
  walk_k_ = g.alloc(B * 4, "dfl_walk_k");
  dist_id_ = g.alloc(B * 32 * 4, "dfl_dist_id");
  dist_p_ = g.alloc(B * 32 * 4, "dfl_dist_p");
  dist_keep_ = g.alloc(B * 4, "dfl_dist_keep");
  acc_ = g.alloc(16, "dfl_acc");
  memset(acc_.ptr(), 0, acc_.size);
  {
    // the target's 8-row verify GEMMs for the drafter's projections (NI_DFLASH_V8=0/1 overrides): the 27B's drafter
    // (H 5120) and the 4B's (H 2560: 4-bit 4B, code / prose, 178.2 -> 179.8 / 142.1 -> 144.2 tok/s); the 35B's
    // (H 2048) keeps dfl_qmm8 except for the measured 32-core M1 Max 4-bit MoE case
    const char* v = getenv("NI_DFLASH_V8");
    v8_ = B == 8 && (v && *v ? atoi(v) != 0 : default_v8(H, tg_.config().moe, tg_.lm_head().q4()));
  }
  if (v8_) {
    v8t_ = g.alloc(8 * H * 2, "dfl_v8_table");
    v8h_ = g.alloc(8 * I * 2, "dfl_v8_h");
    v8hs_ = g.alloc(I / 64 * 8 * 4, "dfl_v8_hs");
    resident_ += v8t_.size + v8h_.size + v8hs_.size;
  }
  gst_ = g.alloc(B == 8 ? 64 : 256, "dfl_gpu_state");  // [0] anchor, [1] rows to commit, [2] their first position, [4 + i] tree path
  memset(gst_.ptr(), 0, gst_.size);
  tree_buf_ = g.alloc(size_t(kf::kWords) * 4, "dfl_kf_tree");
  memset(tree_buf_.ptr(), 0, tree_buf_.size);
  kf_clq_ = g.alloc(D * 16 * 16 * 4, "dfl_kf_clq");
  kf_tok_ = g.alloc(D * 16 * 16 * 4, "dfl_kf_tok");
  kf_next_ = g.alloc(D * 16 * 16 * 4, "dfl_kf_next");
  {
    const std::vector<int32_t> nx = kf::lattice_next(int(D), 16);
    memcpy(kf_next_.ptr(), nx.data(), nx.size() * 4);
  }
  kf_live_ = g.alloc(16, "dfl_kf_live");
  kf_live_.as<uint32_t>()[0] = uint32_t(B);
  kdyn_ = g.alloc(16, "dfl_kdyn");
  kdyn_.as<uint32_t>()[0] = uint32_t(D);
  kdyn_.as<uint32_t>()[1] = uint32_t(B);
  capt_ = g.alloc(size_t(kf::kMaxRows) * nH * 2, "dfl_capt");  // a tree verify's rows (up to 32)
  y8_ = g.alloc(B * H * 4, "dfl_y8");
  c8_ = g.alloc(B * H * 4, "dfl_c8");
  kv8_ = g.alloc(B * KV * 4, "dfl_kv8");
  cdbg_ = g.alloc(size_t(kCtxKeep) * H * 4, "dfl_c");
  memset(capt_.ptr(), 0, capt_.size);
  for (Buffer* b : {&h_, &n_, &xh_, &xs_, &xh2_, &xs2_, &hh_, &gu_g_, &gu_u_, &hs_, &dyn_, &q_, &kv_, &Q_, &Kb_, &Vb_, &part_o_, &part_ml_, &u_,
                    &logits_, &sel_, &tk_v_, &tk_i_, &cand_v_, &cand_i_, &edges_, &walk_s_, &qprob_, &walk_k_, &capt_,
                    &y8_, &c8_, &kv8_, &cdbg_, &rope_cs_, &dist_id_, &dist_p_, &dist_keep_, &acc_, &kdyn_, &gst_,
                    &tree_buf_, &kf_clq_, &kf_tok_, &kf_next_, &kf_live_})
    resident_ += b->size;
}

void DFlash2::build_head(int n_req, bool explicit_n) {
  const uint32_t V = uint32_t(cfg_.vocab);
  const QLinear& H = tg_.lm_head();
  // The ranking: NI_DFLASH_HEAD_IDS=<file> (uint32 token ids, most useful first, e.g. counts from a corpus), else
  // Kapitsa's own: token id order, which for the byte-level BPE tokenizer is its merge order, a frequency proxy
  // (the same idea as the MTP draft head). Id order needs more rows than a frequency ranking: at 65536 it drafts
  // worse (27B code_27b 512: 4.52 tok/cycle vs 4.78 for the full head), at 98304 it covers 99.8-100% of the
  // generated tokens and drafts like the full head (tok/cycle equal on code / p700 / prose, both models).
  std::vector<uint32_t> ranked;
  const char* path = getenv("NI_DFLASH_HEAD_IDS");
  if (path && *path) {
    if (std::ifstream f{path, std::ios::binary}) {
      f.seekg(0, std::ios::end);
      ranked.resize(size_t(f.tellg()) / 4);
      f.seekg(0);
      f.read(reinterpret_cast<char*>(ranked.data()), std::streamsize(ranked.size() * 4));
    }
    if (ranked.empty()) logi("DFlash2: NI_DFLASH_HEAD_IDS=%s holds no token ids; the draft head uses id order", path);
  }
  if (!explicit_n) n_req = ranked.empty() ? kHeadRowsIdOrder : kHeadRows;
  // a head without 4/8-bit group-64 kernels on the drafter's path gets an 8-bit copy (the full one if asked)
  const bool conv = !(H.q4() || H.q8());
  if (n_req <= 0 || uint32_t(n_req) >= V) {
    if (!conv) return;
    n_req = int(V);
  }
  const auto& eos = tg_.config().eos;
  const uint32_t special0 = eos.empty() ? V : uint32_t(*std::min_element(eos.begin(), eos.end()));
  const size_t n = std::max<size_t>(size_t(n_req), V - std::min(special0, V));
  std::vector<char> in(V, 0);
  std::vector<uint32_t> ids;
  ids.reserve(n);
  auto add = [&](uint32_t t) {
    if (t < V && !in[t] && ids.size() < n) {
      in[t] = 1;
      ids.push_back(t);
    }
  };
  for (uint32_t t = special0; t < V; ++t) add(t);  // added/special tokens (and the padding rows past them)
  for (uint32_t t : ranked) add(t);
  for (uint32_t t = 0; t < V && ids.size() < n; ++t) add(t);  // no list (or a short one): id order
  std::sort(ids.begin(), ids.end());
  Gpu& g = Gpu::get();
  if (conv) {  // 8-bit group 64 with mx.quantize's rounding (dfl_quantize_cpu) from the rows as bf16
    const uint32_t K = H.K, N = uint32_t(ids.size()), gpr = K / 64;
    std::vector<uint16_t> tmp(size_t(N) * K);
    std::vector<const uint16_t*> rowp(N);
    for (uint32_t i = 0; i < N; ++i) {
      dfl_row_bf16(H, ids[i], tmp.data() + size_t(i) * K);
      rowp[i] = tmp.data() + size_t(i) * K;
    }
    Buffer w = g.alloc(size_t(N) * K, "dfl_head_w8", false), sc = g.alloc(size_t(N) * gpr * 2, "dfl_head_s8", false),
           bi = g.alloc(size_t(N) * gpr * 2, "dfl_head_b8", false);
    dfl_quantize_cpu(rowp, K, 8, false, w.as<uint32_t>(), sc.as<uint16_t>(), bi.as<uint16_t>());
    head_ids_ = g.alloc(size_t(N) * 4, "dfl_head_ids", false);
    for (uint32_t i = 0; i < N; ++i) head_ids_.as<uint32_t>()[i] = ids[i];
    for (Buffer* b : {&w, &sc, &bi, &head_ids_}) {
      owned_bufs_.push_back(*b);
      resident_ += b->size;
    }
    head_sub_ = QLinear{};
    owned_.push_back(Tensor{w, DType::U32, {int64_t(N), int64_t(K / 4)}});
    head_sub_.w = &owned_.back();
    owned_.push_back(Tensor{sc, DType::BF16, {int64_t(N), int64_t(gpr)}});
    head_sub_.s = &owned_.back();
    owned_.push_back(Tensor{bi, DType::BF16, {int64_t(N), int64_t(gpr)}});
    head_sub_.b = &owned_.back();
    head_sub_.N = N;
    head_sub_.K = K;
    head_sub_.bits = 8;
    head_sub_.gs = 64;
    head_n_ = N;
    logi("DFlash2: draft head of %u of %u ids as an 8-bit copy of the %s LM head (%s), %.0f MB", head_n_, V,
         format_name(H.bits, H.gs).c_str(), ranked.empty() ? "id order = BPE merge order" : path,
         (w.size + sc.size + bi.size) / 1e6);
    return;
  }
  const size_t wrow = size_t(H.K) * H.bits / 8, srow = size_t(H.K / H.gs) * 2;
  Buffer w = g.alloc(ids.size() * wrow, "dfl_head_w", false), sc = g.alloc(ids.size() * srow, "dfl_head_s", false),
         bi = g.alloc(ids.size() * srow, "dfl_head_b", false);
  head_ids_ = g.alloc(ids.size() * 4, "dfl_head_ids", false);
  for (size_t i = 0; i < ids.size(); ++i) {
    memcpy(w.as<uint8_t>() + i * wrow, H.w->buf.as<uint8_t>() + size_t(ids[i]) * wrow, wrow);
    memcpy(sc.as<uint8_t>() + i * srow, H.s->buf.as<uint8_t>() + size_t(ids[i]) * srow, srow);
    memcpy(bi.as<uint8_t>() + i * srow, H.b->buf.as<uint8_t>() + size_t(ids[i]) * srow, srow);
    head_ids_.as<uint32_t>()[i] = ids[i];
  }
  for (Buffer* b : {&w, &sc, &bi, &head_ids_}) {
    owned_bufs_.push_back(*b);
    resident_ += b->size;
  }
  head_sub_ = H;
  owned_.push_back(Tensor{w, H.w->dtype, {int64_t(ids.size()), int64_t(wrow / 4)}});
  head_sub_.w = &owned_.back();
  owned_.push_back(Tensor{sc, H.s->dtype, {int64_t(ids.size()), int64_t(H.K / H.gs)}});
  head_sub_.s = &owned_.back();
  owned_.push_back(Tensor{bi, H.b->dtype, {int64_t(ids.size()), int64_t(H.K / H.gs)}});
  head_sub_.b = &owned_.back();
  head_sub_.N = uint32_t(ids.size());
  head_n_ = uint32_t(ids.size());
  logi("DFlash2: reduced draft head, %u of %u ids (%s), %.0f MB", head_n_, V,
       ranked.empty() ? "id order = BPE merge order" : path, (w.size + sc.size + bi.size) / 1e6);
}

std::unique_ptr<DFlashSeq> DFlash2::new_seq() {
  auto d = std::make_unique<DFlashSeq>();
  const size_t bytes = size_t(cfg_.NL) * 8 * kWin * 128 * 2;
  d->ring_k = Gpu::get().alloc(bytes, "dfl_ring_k");
  d->ring_v = Gpu::get().alloc(bytes, "dfl_ring_v");
  memset(d->ring_k.ptr(), 0, bytes);
  memset(d->ring_v.ptr(), 0, bytes);
  d->ctx = 0;
  d->valid_lo = 0;
  return d;
}

bool DFlash2::tensor_path_ok() {
  return dfl_tw_supported() && v8_knobs().tw && !(v8_knobs().dfl & (1 | 32 | 128));
}

int DFlash2::plan_block_rows(const Qwen35Config& target, int slots, const char* block_env, bool tw_ok) {
  const char* v = block_env ? block_env : getenv("NI_DFLASH_BLOCK");
  if (v && *v && std::string(v) != "auto")  // 8 or 16; anything else fatals when the drafter loads
    return atoi(v) == 16 ? 16 : kB;
  // The default, and the same conditions the load refuses NI_DFLASH_BLOCK=16 under, as far as they can be read
  // without the drafter: a one-slot engine on a dense target, trees on, a walk deep enough for all 16 rows, and the
  // matrix units with the fused attention the drafting pass needs (tw_ok, which the caller fills from its GPU and its
  // knobs). What is left out is the drafter's own weights (a v1 drafter, projections that are not 4-bit group 64), so
  // on such a drafter this is an upper bound: the plan reserves 16 rows and the load keeps 8, ~30 MB on the 27B. A MoE
  // target has no wide verify, and a multi-slot engine co-decodes requests, so both keep the 8-row tile.
  if (slots != 1 || target.moe) return kB;
  const char* kf = getenv("NI_KFLASH");
  const char* tv = getenv("NI_DFLASH_TREE");
  if (!(tv && *tv ? atoi(tv) != 0 : (!(kf && *kf) || atoi(kf) != 0))) return kB;
  const char* vv = getenv("NI_DFLASH_VERIFY");
  if (vv && *vv && atoi(vv) < 15) return kB;
  const char* av = getenv("NI_DFLASH_ATTN");
  if (av && std::string(av) == "scalar") return kB;
  if (!tw_ok) return kB;
  return 16;
}

size_t DFlash2::plan_bytes(const std::string& dir, const Qwen35Config& target, size_t head_row_bytes,
                           size_t* per_slot, size_t* at_prefill, int block_rows) {
  namespace fs = std::filesystem;
  json c;
  if (std::ifstream f(dir + "/config.json"); f) {
    try {
      c = json::parse(f);
    } catch (...) {
    }
  }
  const json dc = c.contains("dflash_config") ? c["dflash_config"] : json::object();
  const size_t H = c.value("hidden_size", 2048), NL = c.value("num_hidden_layers", 5), I = c.value("intermediate_size", 6144),
               V = c.value("vocab_size", 248320), nT = dc.contains("target_layer_ids") ? dc["target_layer_ids"].size() : 5,
               gs = dc.value("conv_group_size", 16);
  const size_t bits = getenv("NI_DFLASH_BITS") ? size_t(atoi(getenv("NI_DFLASH_BITS"))) : 4;
  auto ends = [](const std::string& a, const char* b) {
    const size_t n = strlen(b);
    return a.size() >= n && a.compare(a.size() - n, n, b) == 0;
  };
  // weights: the 2-D projections quantized at load (codes + bf16 scale and bias per 64), the rest as stored
  size_t w = 0;
  std::error_code ec;
  for (auto& de : fs::directory_iterator(dir, ec)) {
    if (de.path().extension() != ".safetensors") continue;
    std::ifstream f(de.path(), std::ios::binary);
    uint64_t n = 0;
    if (!f.read(reinterpret_cast<char*>(&n), 8) || n > (64u << 20)) continue;
    std::string h(n, '\0');
    if (!f.read(h.data(), std::streamsize(n))) continue;
    json j;
    try {
      j = json::parse(h);
    } catch (...) {
      continue;
    }
    for (auto& [name, v] : j.items()) {
      if (name == "__metadata__" || !v.contains("shape")) continue;
      const auto& sh = v["shape"];
      if (sh.size() == 2 && (ends(name, "_proj.weight") || name == "fc.weight" || ends(name, "kernel_projection.weight"))) {
        const size_t N = sh[0], K = sh[1];
        w += N * K * bits / 8 + 2 * N * (K / 64) * 2;
      } else if (name.find("norm.weight") != std::string::npos || name.find("codebook") != std::string::npos ||
                 name.find("base_kernel") != std::string::npos) {
        w += v["data_offsets"][1].get<size_t>() - v["data_offsets"][0].get<size_t>();
      }
    }
  }
  // alloc_buffers(), term for term, at the rows the block drafts: Bk rows, Tk 8-row tiles, Dk lattice depth
  const size_t nH = nT * H, Kx = std::max({nH, size_t(4096), H, I}), KV = NL * 2048, G4 = 4 * (H / std::max<size_t>(gs, 1)),
               ntg = (V + 2047) / 2048, Bk = size_t(std::max(block_rows, int(kB))), Tk = Bk / 8, Dk = Bk - 1;
  size_t b = 5 * Bk * H * 4 + 2 * Bk * I * 4 + 16 * Tk * Kx * 2 + Tk * Kx / 64 * 8 * 4 + 16 * Tk * H * 2 + Tk * H / 64 * 8 * 4 +
             16 * Tk * I * 2 + Tk * I / 64 * 8 * 4 + Bk * G4 * 4 + Bk * 4096 * 4 + Bk * 2048 * 4 + 32 * Bk * 128 * 4 +
             2 * 8 * Bk * 128 * 4 + 32 * (32 * Bk * 128 * 4 + 32 * Bk * 2 * 4) + Bk * V * 4 + Bk * kRank * 4 +
             2 * Dk * ntg * 16 * 4 + 8 * nH * 2 + Bk * KV * 4 + size_t(kCtxKeep) * H * 4 + (64 << 10);
  const char* v8 = getenv("NI_DFLASH_V8");
  // A native 4-bit group-64 head row includes its packed codes, scales and biases.
  const bool head_q4 = head_row_bytes == size_t(target.hidden) / 2 + size_t(target.hidden) / 64 * 4;
  if (v8 && *v8 ? atoi(v8) != 0 : default_v8(H, target.moe, head_q4))
    b += 8 * H * 2 + 8 * I * 2 + I / 64 * 8 * 4;
  // build_head(): rows of the target's LM head (the ranked or id-order rows plus every id from the first
  // end-of-sequence id on)
  {
    const char* hv = getenv("NI_DFLASH_HEAD");
    const char* ids = getenv("NI_DFLASH_HEAD_IDS");
    const size_t req = hv && *hv ? size_t(std::max(0, atoi(hv))) : ids && *ids ? size_t(kHeadRows) : size_t(kHeadRowsIdOrder);
    const size_t special0 = target.eos.empty() ? V : std::min<size_t>(V, size_t(*std::min_element(target.eos.begin(), target.eos.end())));
    if (req > 0 && req < V) b += std::max(req, V - special0) * (head_row_bytes + 4);
  }
  b += size_t(4) << 20;  // the target's 8-row verify scratch (Qwen35::v8_alloc): ~1-3 MB
  // begin_prefill(): the prompt captures and the 512-row context-commit buffers
  const size_t pf = size_t(kCtxKeep) * nH * 2 + 512 * nH * 2 + 512 * H * 4 + 512 * H * 2 + 512 * KV * 4;
  if (at_prefill) *at_prefill = pf;
  // ring: [layers][8 kv heads][2048 slots][128] bf16, K and V (DFlash2::new_seq)
  if (per_slot) *per_slot = NL * 8 * kWin * 128 * 2 * 2;
  // Above the 8-row tile the target's verify buffers and every slot's tape are taller than the plan's baseline: the
  // captures, the target's logits, candidates and X8 tables (Qwen35::set_tree_rows), and per slot the verify tape's
  // extra rows (Qwen35::grow_tape). Two ways to get there: KFlash's wide tree verifies on a dense target (32 rows),
  // only where the knobs allow them (kf_dense_rows reads the same knobs the drafting pass picks its rows from, so a
  // target that opted out reserves nothing here), or this drafter drafting a 16-row block, which Engine asks for on a
  // one-slot engine and which pins the verify to 16 rows (KfBudget::pin_wide).
  const uint32_t wide = Qwen35::kf_dense_rows(target);
  // a block wider than the 8-row tile bounds the verify at its own width: the constructor masks the wider counts away
  // there, so a wide KFlash row set cannot make the verify taller than the lattice the block drafts
  const size_t rows = block_rows > Qwen35::kMaxVerify ? size_t(block_rows)
                      : wide > (1u << (Qwen35::kMaxVerify - 1)) ? size_t(Qwen35::kMaxTreeRows)
                                                                : size_t(Qwen35::kMaxVerify);
  if (rows > size_t(Qwen35::kMaxVerify)) {
    const size_t extra = rows - Qwen35::kMaxVerify;
    b += extra * nH * 2 + extra * V * 4 + 2 * extra * (V / 2048 + 2) * 64 * 4 + (size_t(extra) << 19);
    size_t lin = 0;
    for (const bool l : target.is_linear) lin += l ? 1 : 0;
    const size_t C = 2 * size_t(target.lin_hk) * target.lin_dk + size_t(target.lin_hv) * target.lin_dv;
    const size_t row = C + size_t(target.lin_hv) * (target.lin_dk + target.lin_dv + 1);
    if (per_slot) *per_slot += lin * extra * row * 4;
  }
  return w + b + pf;
}

std::string DFlash2::find(const std::string& model_dir, const Qwen35Config& tc) {
  namespace fs = std::filesystem;
  auto lower = [](std::string x) {
    for (auto& c : x) c = char(std::tolower(static_cast<unsigned char>(c)));
    return x;
  };
  auto fits = [&](const fs::path& d) {
    std::ifstream f(d / "config.json");
    if (!f) return false;
    try {
      json c = json::parse(f);
      return c.value("num_target_layers", tc.layers) == tc.layers && config_problems(c, tc).empty();
    } catch (...) {
      return false;
    }
  };
  std::error_code ec;
  fs::path md = fs::path(model_dir);
  if (md.filename().empty()) md = md.parent_path();
  for (const char* sub : {"dflash2", "drafter"})
    if (fits(md / sub)) return (md / sub).string();
  const std::string name = lower(md.filename().string());
  std::vector<fs::path> roots = {md.parent_path(), md.parent_path() / "drafters"};
  if (const char* home = getenv("HOME")) roots.push_back(fs::path(home) / ".cache/nace-infer-models/drafters");
  std::string best;
  size_t best_len = 0;
  bool best_v2 = false;
  for (auto& r : roots) {
    if (!fs::is_directory(r, ec)) continue;
    for (auto& ent : fs::directory_iterator(r, ec)) {
      if (!ent.is_directory(ec)) continue;
      const std::string dn = lower(ent.path().filename().string());
      // <model>-DFlash2 (DFlash2DraftModel) or <model>-DFlash (DFlashDraftModel, v1); DFlash2 wins a tie
      for (const std::string suffix : {"-dflash2", "-dflash"}) {
        if (dn.size() <= suffix.size() || dn.compare(dn.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        const std::string base = dn.substr(0, dn.size() - suffix.size());
        const bool v2 = suffix == "-dflash2";
        const bool match = name == base || (name.size() > base.size() && name.compare(0, base.size(), base) == 0 &&
                                            name[base.size()] == '-');
        if (match && (base.size() > best_len || (base.size() == best_len && v2 && !best_v2)) && fits(ent.path())) {
          best = ent.path().string();
          best_len = base.size();
          best_v2 = v2;
        }
        break;
      }
    }
  }
  return best;
}

void DFlash2::rewind(DFlashSeq& d, int pos) {
  if (pos >= d.ctx) {
    if (pos > d.ctx) d.valid_lo = pos;  // positions [ctx, pos) were never committed: nothing before pos is usable
    d.ctx = pos;
    return;
  }
  // entries still in the ring: positions [max(valid_lo, ctx - 2048), ctx); keep those below pos
  const int oldest = std::max(d.valid_lo, d.ctx - kWin);
  d.ctx = pos;
  d.valid_lo = std::min(oldest, pos);
}

void DFlash2::encode_ring_copy(Encoder& e, const DFlashSeq& src, DFlashSeq& dst) {
  for (int kv = 0; kv < 2; ++kv) {
    const Buffer& a = kv ? src.ring_v : src.ring_k;
    const Buffer& b = kv ? dst.ring_v : dst.ring_k;
    const uint32_t n = uint32_t(std::min(a.size, b.size) / 16);
    e.use(dpipe("copy_u32"));
    e.buf(0, a);
    e.buf(1, b);
    e.val(2, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
  dst.ctx = src.ctx;
  dst.valid_lo = src.valid_lo;
}

// ---- encoders ----

void DFlash2::dbg(Encoder& e, const std::string& name, const Buffer& src, size_t bytes) {
  if (!debug_) return;
  auto it = dbg_.find(name);
  if (it == dbg_.end()) it = dbg_.emplace(name, Gpu::get().alloc((bytes + 15) / 16 * 16, "dfl_dbg")).first;
  const uint32_t n = uint32_t(bytes / 16);
  e.use(dpipe("copy_u32"));
  e.buf(0, src);
  e.buf(1, it->second);
  e.val(2, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
}

// The drafter's 4-bit GEMMs on the core before dfl_q8_run4: dfl_q8_run4 (two groups per step, the next step's loads
// first, subnormal codes) is measured on M5 and the 32-core M1 Max. Other Macs
// keep the earlier core; NI_DFLASH_CORE4=1 / 0 forces the new / earlier one.
static bool dfl_old4() {
  static const bool old = getenv("NI_DFLASH_CORE4") ? atoi(getenv("NI_DFLASH_CORE4")) == 0
      : !(Gpu::get().apple10() || (Gpu::get().name() == "Apple M1 Max" && Gpu::get().core_count() == 32));
  return old;
}

// K split (bench_qmm8 sweep, M4 Pro): no split once there are >= 1024 threadgroups (LM head) on its 20 GPU cores,
// ~51 per core (Gpu::core_count); otherwise the largest split that leaves every simdgroup >= 8 groups of 64.
static uint32_t dfl_qmm8_ks(uint32_t rbs_total, uint32_t K, int ks_req) {
  static const int ks_env = getenv("NI_DFLASH_KS") ? atoi(getenv("NI_DFLASH_KS")) : 0;
  static const uint32_t many = 1024u * uint32_t(std::max(1, Gpu::get().core_count())) / 20u;
  uint32_t ks = 1;
  if (ks_req > 0) ks = uint32_t(ks_req);
  else if (ks_env > 0) ks = uint32_t(ks_env);
  else if (rbs_total / 8 < many)
    while (ks < 8 && (K / 64) / (ks * 2) >= 8) ks *= 2;
  return ks;
}

void dfl_enc_qmm8(Encoder& e, const std::vector<DflSeg8>& segs, const Buffer& xh, const Buffer& xs, uint32_t K,
                  uint32_t x_ld, int ks_req, int rf_req) {
  if (segs.empty() || segs.size() > 4) fatal("enc_qmm8: 1-4 segments");
  kp::DflQmm8 p{};
  p.old4 = dfl_old4() ? 1u : 0u;
  p.K = K;
  p.nseg = uint32_t(segs.size());
  p.x_ld = x_ld;
  uint32_t rbs_total = 0;
  for (auto& s : segs) {
    if (s.L->K != K || s.L->gs != 64 || (s.L->bits != 4 && s.L->bits != 8)) fatal("enc_qmm8: K/group/bits mismatch");
    rbs_total += (s.L->N + 7) / 8;
  }
  const uint32_t ks = dfl_qmm8_ks(rbs_total, K, ks_req);
  p.ks = ks;
  // NI_DFLASH_RF=2: 16 weight rows per simdgroup (the X gather shared by two fragments)
  static const int rf_env = getenv("NI_DFLASH_RF") ? atoi(getenv("NI_DFLASH_RF")) : 1;
  const uint32_t rf = (rf_req ? rf_req : rf_env) == 2 ? 2 : 1;
  const uint32_t rbs = 8 / ks * rf;  // 8-row blocks per threadgroup
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const DflSeg8& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = s.L->N;
      p.bits[i] = uint32_t(s.L->bits);
      p.y_ld[i] = s.y_ld;
      p.oscale[i] = s.oscale;
      p.tg_begin[i] = tg;
      tg += ((s.L->N + 7) / 8 + rbs - 1) / rbs;
    }
    e.buf(int(i * 4 + 0), s.L->w->buf);
    e.buf(int(i * 4 + 1), s.L->s->buf);
    e.buf(int(i * 4 + 2), s.L->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.use(dpipe(rf == 2 ? "dfl_qmm8_r2" : "dfl_qmm8"));
  e.buf(16, xh);
  e.buf(17, xs);
  e.val(18, p);
  e.dispatch(tg, 1, 1, 256);
}

void dfl_enc_qmm8_gu(Encoder& e, const QLinear& g, const QLinear& u, const Buffer& xh, const Buffer& xs,
                     const Buffer& hh, const Buffer& hs, int ks_req) {
  if (g.N != u.N || g.K != u.K || g.bits != u.bits || g.N % 64) fatal("enc_qmm8_gu: shapes");
  // K split inside the 64-column threadgroup (8 row blocks x ks simdgroups); default: the largest
  // of 1/2/4 that keeps >= 8 groups per simdgroup (NI_DFLASH_GU_KS overrides)
  static const int ks_env = getenv("NI_DFLASH_GU_KS") ? atoi(getenv("NI_DFLASH_GU_KS")) : 0;
  uint32_t ks = ks_req > 0 ? uint32_t(ks_req) : ks_env > 0 ? uint32_t(ks_env) : 0;
  if (!ks) {
    ks = 1;
    while (ks < 4 && (g.K / 64) / (ks * 2) >= 8) ks *= 2;
  }
  if (ks != 1 && ks != 2 && ks != 4) fatal("enc_qmm8_gu: ks %u", ks);
  kp::DflQmm8 p{};
  p.old4 = dfl_old4() ? 1u : 0u;
  p.K = g.K;
  p.nseg = 1;
  p.ks = ks;
  p.x_ld = g.K;
  p.N[0] = g.N;
  p.bits[0] = uint32_t(g.bits);
  p.y_ld[0] = g.N;
  p.oscale[0] = 1.0f;
  e.use(dpipe(ks == 1 ? "dfl_qmm8_gu" : ks == 2 ? "dfl_qmm8_gu_ks2" : "dfl_qmm8_gu_ks4"));
  e.buf(0, g.w->buf); e.buf(1, g.s->buf); e.buf(2, g.b->buf); e.buf(3, hh);
  e.buf(4, u.w->buf); e.buf(5, u.s->buf); e.buf(6, u.b->buf); e.buf(7, hs);
  e.buf(16, xh);
  e.buf(17, xs);
  e.val(18, p);
  e.dispatch(g.N / 64, 1, 1, 256 * ks);
}

// The matrix units through Metal 4 tensor_ops (qmm8t): Apple10 (M5) on macOS 26, the "tensor" library (compiled here
// if the target has not yet), V8Knobs::tw on and V8Knobs::dfl bit 32 off (the drafter alone back on dfl_qmm8)
static bool dfl_tw_supported() {
  static const bool ok = [] {
    if (Gpu::get().apple_family() < 10) return false;
    if (__builtin_available(macOS 26.0, *)) {
      Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
      return Gpu::get().pipeline("qmm8t_b4_nt16", false).pso != nullptr;
    }
    return false;
  }();
  return ok;
}

bool DFlash2::tw_for(const std::vector<DflSeg8>& segs, uint32_t K) const {
  if (!dfl_tw_supported() || !v8_knobs().tw || (v8_knobs().dfl & 32) || K % 64) return false;
  for (auto& sg : segs)
    if (!sg.L->q4() || sg.L->gs != 64 || sg.L->K != K || sg.oscale != 1.0f) return false;
  return !segs.empty() && segs.size() <= 4;
}

// The drafter's GEMMs on the matrix units: qmm8t with the target's tiling rule, except 32 rows per simdgroup also
// for K > 4096 (bench_tq on the M5 Max, us: q|kv 6144x2048 17.7 vs dfl_qmm8 20.8, o 2048x4096 13.2 vs 15.5, down
// 2048x6144 19.2 vs 22.0, commit k|v 12288x2048 31.2 vs 37.3, commit fc 2048x16384 46.6 vs 54.1, reduced LM head
// 98304x2048 216.7 vs ~254).
void DFlash2::enc_qmm8(Encoder& e, const std::vector<DflSeg8>& segs, const Buffer& xh, const Buffer& xs, uint32_t K,
                       uint32_t x_ld, int tiles) {
  if (x_ld != K || !tw_for(segs, K)) {
    if (tiles > 1) fatal("DFlash2 block %d: a GEMM (K %u) needs the tensor path", tiles * 8, K);
    dfl_enc_qmm8(e, segs, xh, xs, K, x_ld);
    return;
  }
  uint32_t Ntot = 0;
  for (auto& sg : segs) Ntot += sg.L->N;
  const uint32_t gpr = K / 64;
  const int nt = ((Ntot + 15) / 16 > 8192 || gpr > 64) ? 32 : 16;
  uint32_t ks = gpr > 32 && gpr <= 64 ? 4 : 8;
  while (ks > 1 && gpr < ks) ks /= 2;
  const bool pf = nt == 32 || gpr > 32;
  // A 16-row drafting block is two 8-row tiles (dfl_put2): qmm8t2d takes the pair from one weight load, where the
  // per-tile loop below streams the weights once per tile. NI_DFL_PAIR=0 keeps the per-tile path (A/B of the pair).
  static const bool pair_env = !getenv("NI_DFL_PAIR") || std::string(getenv("NI_DFL_PAIR")) != "0";
  const bool pair = tiles >= 2 && tiles % 2 == 0 && pair_env;  // whole tiles only (a block of 8 or 16 rows)
  kp::Qmm8 p{};
  p.K = K;
  p.R = pair ? 16 : 8;
  p.nseg = uint32_t(segs.size());
  p.ks = ks;
  uint32_t tg = 0;
  e.use(dpipe(std::string(pf ? (pair ? "qmm8t2dp" : "qmm8tp") : (pair ? "qmm8t2d" : "qmm8t")) + "_b4_nt" +
              std::to_string(nt)));
  for (size_t i = 0; i < segs.size(); ++i) {
    p.N[i] = segs[i].L->N;
    p.y_ld[i] = segs[i].y_ld;
    p.add[i] = 0;
    p.tg_begin[i] = tg;
    tg += (segs[i].L->N + uint32_t(nt) - 1) / uint32_t(nt);
  }
  p.tg_begin[segs.size()] = tg;
  if (pair) {  // pairs of tiles, 16 rows each: the first tile's rows 0-7, the second's 8-15 of one contiguous Y
    for (int t = 0; t + 1 < tiles; t += 2) {
      for (size_t i = 0; i < 4; ++i) {
        const DflSeg8& sg = segs[std::min(i, segs.size() - 1)];
        e.buf(int(i * 4 + 0), sg.L->w->buf);
        e.buf(int(i * 4 + 1), sg.L->s->buf);
        e.buf(int(i * 4 + 2), sg.L->b->buf);
        e.buf(int(i * 4 + 3), sg.y, size_t(t) * 8 * sg.y_ld * 4);
      }
      e.buf(16, xh, (size_t(t) * 16 + 8) * K * 2);     // tile t's T table
      e.buf(17, xs, size_t(t) * (K / 8) * 4);          // tile t's group sums
      e.buf(18, xh, (size_t(t) * 16 + 24) * K * 2);    // tile t + 1's T table
      e.buf(19, xs, size_t(t + 1) * (K / 8) * 4);      // tile t + 1's group sums
      e.val(20, p);
      e.dispatch(tg, 1, 1, 32 * ks);
    }
    return;
  }
  for (int t = 0; t < tiles; ++t) {
    for (size_t i = 0; i < 4; ++i) {
      const DflSeg8& sg = segs[std::min(i, segs.size() - 1)];
      e.buf(int(i * 4 + 0), sg.L->w->buf);
      e.buf(int(i * 4 + 1), sg.L->s->buf);
      e.buf(int(i * 4 + 2), sg.L->b->buf);
      e.buf(int(i * 4 + 3), sg.y, size_t(t) * 8 * sg.y_ld * 4);
    }
    e.buf(16, xh, (size_t(t) * 16 + 8) * K * 2);  // the T table after the 8 rows (dfl_put2)
    e.buf(17, xs, size_t(t) * (K / 8) * 4);
    e.val(18, p);
    e.dispatch(tg, 1, 1, 32 * ks);
  }
}

// Conv projection L (rows permuted, see the constructor) + dfl_conv_prep: D -> dyn, the conv's output -> (xh2, xs2).
void DFlash2::enc_qmm8_cp(Encoder& e, const QLinear& L, const Tensor* base, const Buffer& xh, const Buffer& xs) {
  if (L.K != uint32_t(cfg_.H) || L.N % 8 || L.gs != 64) fatal("enc_qmm8_cp: shapes");
  kp::DflQmm8 p{};
  p.old4 = dfl_old4() ? 1u : 0u;
  p.K = L.K;
  p.nseg = 1;
  p.x_ld = L.K;
  p.ks = dfl_qmm8_ks(L.N / 8, L.K, 0);
  p.N[0] = L.N;
  p.bits[0] = uint32_t(L.bits);
  p.y_ld[0] = L.N;
  p.oscale[0] = 1.0f;
  const uint32_t tg = (L.N / 8 + 8 / p.ks - 1) / (8 / p.ks);
  p.tg_begin[1] = tg;
  const uint32_t H = uint32_t(cfg_.H);
  e.use(dpipe("dfl_qmm8_cp"));
  // 16-row blocks: the second tile is the same dispatch on offset buffers (its row 0 takes the conv's x[-1] from the
  // first tile's row 7: prev0)
  for (int t = 0; t < block() / 8; ++t) {
    const kp::DflConv cp{H, uint32_t(cfg_.H / cfg_.conv_gs), cfg_.eps, t > 0 ? 1u : 0u};
    e.buf(0, L.w->buf); e.buf(1, L.s->buf); e.buf(2, L.b->buf); e.buf(3, dyn_, size_t(t) * 8 * L.N * 4);
    e.buf(16, xh, size_t(t) * 16 * L.K * 2); e.buf(17, xs, size_t(t) * (L.K / 8) * 4);
    e.val(18, p);
    e.buf(19, n_, size_t(t) * 8 * H * 4); e.buf(20, base->buf); e.buf(21, xh2_, size_t(t) * 16 * H * 2);
    e.buf(22, xs2_, size_t(t) * (H / 8) * 4);
    e.val(23, cp);
    e.dispatch(tg, 1, 1, 256);
  }
}

// Prefill-size GEMM (context commit of the prompt): the target's prefill kernels, half X rows.
void DFlash2::enc_qmm_rows(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M) {
  const bool big = M >= 64;
  const int BM = big ? 64 : 32, BN = big ? 64 : 32;
  const std::string n = "qmm_b" + std::to_string(L.bits) + (big ? "_m64_n64" : "_m32_n32");
  e.use(dpipe(n));
  kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, 0u, uint32_t(L.gs), 0, 0};
  e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
  e.val(5, p);
  e.dispatch((L.N + BN - 1) / BN, (M + BM - 1) / BM, 1, 128);
}

void DFlash2::enc_to_v8(Encoder& e, const Buffer& xh, uint32_t K, const Buffer& table) {
  kp::DflPerm p{K};
  e.use(dpipe("dfl_x8_to_v8"));
  e.buf(0, xh); e.buf(1, table);
  e.val(2, p);
  e.dispatch((K / 64 + 7) / 8, 1, 1, 256);
}

void DFlash2::enc_commit_kv(Encoder& e, DFlashSeq& d, const Buffer& kv, int rows, int pos0) {
  // rows < 0: rows and positions from the drafter state (GPU-driven; the grid covers 8 rows)
  const bool gpu = rows < 0;
  kp::DflCommit p{gpu ? 0xffffffffu : uint32_t(rows), uint32_t(pos0), uint32_t(cfg_.NL) * 2048, cfg_.eps};
  e.use(dpipe("dfl_ctx_commit"));
  e.buf(0, kv); e.buf(1, kn_all_->buf); e.buf(2, inv_freq_); e.buf(3, d.ring_k); e.buf(4, d.ring_v);
  e.val(5, p);
  e.buf(6, gst_);
  e.dispatch(gpu ? uint32_t(block()) : uint32_t(rows), uint32_t(cfg_.NL) * 8, 1, 128);
}

void DFlash2::begin_prefill(int n, int start) {
  const size_t nH = cfg_.target_layers.size() * cfg_.H;
  pf_n_ = n;
  pf_rows_ = std::max(0, n - std::max(start, n - kCtxKeep));
  Gpu& g = Gpu::get();
  if (!pcap_) {
    pcap_ = g.alloc(size_t(kCtxKeep) * nH * 2, "dfl_prompt_capt");
    constexpr int C = 512;  // rows per context-commit chunk
    xph_ = g.alloc(size_t(C) * nH * 2, "dfl_pf_x");
    yph_ = g.alloc(size_t(C) * cfg_.H * 4, "dfl_pf_y");
    cph_ = g.alloc(size_t(C) * cfg_.H * 2, "dfl_pf_c");
    kvph_ = g.alloc(size_t(C) * cfg_.NL * 2048 * 4, "dfl_pf_kv");
  }
  hook_pf_.dst = pcap_;
  hook_pf_.pos_base = n - pf_rows_;
  hook_pf_.rows = pf_rows_;
  tg_.set_capture(&hook_pf_);
}

void DFlash2::set_prompt_captures(const uint16_t* rows_bf16, int rows) {
  if (!pcap_ || rows != pf_rows_) fatal("set_prompt_captures: call begin_prefill(n) first (rows %d vs %d)", rows, pf_rows_);
  memcpy(pcap_.ptr(), rows_bf16, size_t(rows) * hook_pf_.ld * 2);
}

void DFlash2::encode_prefill_commit(Encoder& e, DFlashSeq& d) {
  const uint32_t H = uint32_t(cfg_.H), nH = hook_pf_.ld;
  if (pf_rows_ == 0) {
    d.ctx = pf_n_;
    return;
  }
  // a gap between the ring's last position and the new rows: nothing older is contiguous with them
  // (with d.ctx == start the gap, if any, lies below n - 2047 and is never visible anyway)
  if (d.ctx < pf_n_ - pf_rows_) d.valid_lo = std::max(d.valid_lo, pf_n_ - pf_rows_);
  // Short prompts: 8-row chunks through the 8-row GEMM (it streams each weight once per 8 rows at
  // ~180 GB/s; the prefill GEMM pays 64-row tiles). NI_DFLASH_PF8 = largest row count for this path.
  static const int pf8_env = getenv("NI_DFLASH_PF8") ? atoi(getenv("NI_DFLASH_PF8")) : 256;
  if (pf_rows_ <= (pf8_ >= 0 ? pf8_ : pf8_env)) {
    for (int c0 = 0; c0 < pf_rows_; c0 += kB) {
      const int cn = std::min(kB, pf_rows_ - c0);
      kp::DflX8 xp{nH, uint32_t(cn), nH, 1.0f / 16.0f};
      e.use(dpipe("dfl_x8_from_bf16"));
      e.buf(0, pcap_, size_t(c0) * nH * 2); e.buf(1, xh_); e.buf(2, xs_);
      e.val(3, xp);
      e.buf(4, gst_);
      e.dispatch(8, (nH / 64 + 7) / 8, 1, 256);
      // the captures' 1/16 is undone by the GEMM (oscale) or, on the matrix units, by the norm (exact either way)
      const float fs = tw_for({{&fc_, y8_, H, 1.0f}}, nH) ? 16.0f : 1.0f;
      enc_qmm8(e, {{&fc_, y8_, H, 16.0f / fs}}, xh_, xs_, nH, nH);
      kp::DflNorm np{H, cfg_.eps, fs, 1u};
      e.use(dpipe("dfl_norm_x8"));
      e.buf(0, y8_); e.buf(1, hnorm_->buf); e.buf(2, c8_); e.buf(3, xh_); e.buf(4, xs_);
      e.val(5, np);
      e.dispatch(8, 1, 1, 256);
      if (debug_) {  // keep c(j) of the whole prompt for tests
        e.use(dpipe("copy_u32"));
        const uint32_t nq = uint32_t(cn) * H / 4;
        e.buf(0, c8_); e.buf(1, cdbg_, size_t(c0) * H * 4); e.val(2, nq);
        e.dispatch((nq + 255) / 256, 1, 1, 256);
      }
      enc_qmm8(e, {{&kv_all_, kv8_, uint32_t(cfg_.NL) * 2048, 1.0f}}, xh_, xs_, H, H);
      enc_commit_kv(e, d, kv8_, cn, pf_n_ - pf_rows_ + c0);
    }
    d.ctx = pf_n_;
    return;
  }
  constexpr int C = 512;
  for (int c0 = 0; c0 < pf_rows_; c0 += C) {
    const int cn = std::min(C, pf_rows_ - c0);
    {  // bf16 captures -> half / 16 (exact for |x| in [2^-10, 2^20])
      kp::DflRows p{nH, 1.0f / 16.0f};
      e.use(dpipe("dfl_rows_bf16_half"));
      e.buf(0, pcap_, size_t(c0) * nH * 2);
      e.buf(1, xph_);
      e.val(2, p);
      e.dispatch((nH / 4 + 255) / 256, uint32_t(cn), 1, 256);
    }
    enc_qmm_rows(e, fc_, xph_, yph_, cn);
    {
      kp::DflNorm p{H, cfg_.eps, 16.0f, 1u};
      e.use(dpipe("dfl_rows_norm_half"));
      e.buf(0, yph_); e.buf(1, hnorm_->buf); e.buf(2, cph_); e.buf(3, cdbg_, size_t(c0) * H * 4);
      e.val(4, p);
      e.dispatch(uint32_t(cn), 1, 1, 256);
    }
    enc_qmm_rows(e, kv_all_, cph_, kvph_, cn);
    enc_commit_kv(e, d, kvph_, cn, pf_n_ - pf_rows_ + c0);
  }
  d.ctx = pf_n_;
}

void DFlash2::end_prefill() { tg_.set_capture(nullptr); }

void DFlash2::arm_verify(int P, int rows) {
  hook_vf_.dst = capt_;
  hook_vf_.pos_base = P;
  hook_vf_.rows = rows;
  tg_.set_capture(&hook_vf_);
}

void DFlash2::disarm() { tg_.set_capture(nullptr); }

void DFlash2::encode_commit(Encoder& e, DFlashSeq& d, int m) { enc_commit_from(e, d, m, capt_, gst_tree_); }

std::string DFlash2::r_list(uint32_t allowed) {
  std::string r;
  for (int n = 1; n <= kf::kMaxRows; ++n)
    if ((allowed >> (n - 1)) & 1u) r += (r.empty() ? "" : ", ") + std::to_string(n);
  return r;
}

void DFlash2::set_tree(bool on) {
  // NI_DFLASH_VERIFY below 7 keeps the chain
  tree_ = on && tg_.block_drafter_ok() && tg_.verify_tree_ok() && verify_depth() == block() - 1;
  if (on && !tree_)
    logi("DFlash2: tree drafting needs all 8 verify rows and the 8-row verify's attn8m and split GDN paths; the chain "
         "stays");
  if (tree_) logi("KFlash: tree drafting on the DFlash2 lattice (budget cache %s)", kf_budget_ ? "on" : "off");
}

void DFlash2::observe(double ms) {
  if (tree_ && kf_budget_) kf_budget_->observe(tree_buf_.as<int32_t>(), ms);
}

void DFlash2::stage_init(DFlashSeq& d) {
  if (d.stage) return;
  d.walk = Gpu::get().alloc(1408, "dfl_walk");
  d.stage = Gpu::get().alloc(size_t(block()) * hook_vf_.ld * 2, "dfl_stage");
  d.staged = 0;
  d.stage_hook.dst = d.stage;
  d.stage_hook.ld = hook_vf_.ld;
  d.stage_hook.slot = hook_vf_.slot;
  d.stage_hook.pos_base = 0;
  d.stage_hook.rows = block();
}

void DFlash2::encode_save_walk(Encoder& e, DFlashSeq& d) {
  auto copy = [&](const Buffer& src, const Buffer& dst) {
    const uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
    e.use(dpipe("copy_u32"));
    e.buf(0, src);
    e.buf(1, dst);
    e.val(2, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  };
  copy(cand_i_, d.walk.view(0, 448));
  copy(qprob_, d.walk.view(448, 448));
  copy(walk_k_, d.walk.view(896, 32));
}

void DFlash2::encode_save_scores(Encoder& e, DFlashSeq& d) {
  e.use(dpipe("copy_u32"));
  e.buf(0, walk_s_);
  e.buf(1, d.walk.view(960, 448));
  e.val(2, uint32_t(448 / 16));
  e.dispatch(1, 1, 1, 256);
}

void DFlash2::encode_accept_rows(Encoder& e, Seq& s, DFlashSeq& d, int P, int K, const SamplingParams& sp, int r0,
                                 uint32_t n) {
  if (K < 1 || K > kB - 1 || sp.temperature <= 0) fatal("DFlash2 accept_rows: K=%d T=%g", K, sp.temperature);
  const size_t R = size_t(K) + 1;
  kp::DflDist dp{n, uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)), sp.temperature, sp.top_p};
  e.use(dpipe("dfl_dist_rows"));
  e.buf(0, tg_.verify_cand_vals().view(size_t(r0) * n * 4, R * n * 4));
  e.buf(1, tg_.verify_cand_ids().view(size_t(r0) * n * 4, R * n * 4));
  e.buf(2, dist_id_); e.buf(3, dist_p_); e.buf(4, dist_keep_);
  e.val(5, dp);
  e.dispatch(uint32_t(R), 1, 1, 1024);
  kp::DflAccept ap{uint32_t(P), uint32_t(K), sp.temperature, sp.seed, 0u};
  e.use(dpipe("dfl_accept"));
  e.buf(0, s.tok); e.buf(1, tg_.verify_samples().view(size_t(r0) * 4, R * 4)); e.buf(2, dist_id_); e.buf(3, dist_p_);
  e.buf(4, d.walk.view(0, 448)); e.buf(5, d.walk.view(448, 448)); e.buf(6, d.walk.view(896, 32));
  e.buf(7, d.walk.view(928, 16));
  e.val(8, ap);
  e.buf(9, kdyn_);
  e.buf(10, d.walk.view(944, 16));
  e.dispatch(1, 1, 1, 32);
}

void DFlash2::encode_commit_staged(Encoder& e, DFlashSeq& d) {
  if (d.staged <= 0) return;
  enc_commit_from(e, d, d.staged, d.stage);
  d.staged = 0;
}

void DFlash2::enc_commit_from(Encoder& e, DFlashSeq& d, int m, const Buffer& src, bool gather) {
  // m < 0: GPU-driven (rows and positions from the drafter state; d.ctx is advanced by the caller)
  const bool gpu = m < 0;
  if (!gpu && (m < 1 || m > block())) fatal("DFlash2 commit: m=%d", m);
  const uint32_t B = uint32_t(block()), BT = B / 8;
  if (v8_knobs().skip & 16384) {  // timing only (V8Knobs::skip)
    if (!gpu) d.ctx += m;
    return;
  }
  const uint32_t H = uint32_t(cfg_.H), nH = hook_vf_.ld;
  {
    kp::DflX8 p{nH, gpu ? 0xffffffffu : uint32_t(m), nH, 1.0f / 16.0f, gather ? 1u : 0u};
    // one group per simdgroup: 8 x 32 threadgroups for the 35B's 8 captured layers (12.5 -> ~3 us isolated, vs 8)
    e.use(dpipe("dfl_x8_from_bf16"));
    e.buf(0, src); e.buf(1, xh_); e.buf(2, xs_);
    e.val(3, p);
    e.buf(4, gst_);
    e.dispatch(B, (nH / 64 + 7) / 8, 1, 256);
  }
  const float fs = tw_for({{&fc_, y8_, H, 1.0f}}, nH) ? 16.0f : 1.0f;  // (see encode_prefill_commit)
  enc_qmm8(e, {{&fc_, y8_, H, 16.0f / fs}}, xh_, xs_, nH, nH, int(BT));
  {
    kp::DflNorm p{H, cfg_.eps, fs, 1u};
    e.use(dpipe("dfl_norm_x8"));
    e.buf(0, y8_); e.buf(1, hnorm_->buf); e.buf(2, cdbg_); e.buf(3, xh_); e.buf(4, xs_);
    e.val(5, p);
    e.dispatch(B, 1, 1, 256);
  }
  enc_qmm8(e, {{&kv_all_, kv8_, uint32_t(cfg_.NL) * 2048, 1.0f}}, xh_, xs_, H, H, int(BT));
  enc_commit_kv(e, d, kv8_, gpu ? -1 : m, d.ctx);
  if (!gpu) d.ctx += m;
}

void dfl_attn_plan(int P, uint32_t max_splits, uint32_t min_chunk, uint32_t* lo, uint32_t* chunk, uint32_t* nsplit,
                   int valid_lo) {
  static const uint32_t env = getenv("NI_DFLASH_ATTN_CHUNK") ? uint32_t(atoi(getenv("NI_DFLASH_ATTN_CHUNK"))) : 0;
  *lo = P > DFlash2::kCtxKeep ? uint32_t(P - DFlash2::kCtxKeep) : 0u;
  if (valid_lo > 0 && uint32_t(valid_lo) > *lo) *lo = std::min(uint32_t(valid_lo), uint32_t(P));
  const uint32_t L = uint32_t(P) - *lo;
  // min_chunk 0 (and no NI_DFLASH_ATTN_CHUNK): the smallest multiple of 32 keys that needs at most max_splits
  // splits (dfl_attn_geom's plan for a GPU-side P)
  const uint32_t mc = std::max<uint32_t>(32, min_chunk ? min_chunk : env ? env : ((L + max_splits - 1) / max_splits + 31) / 32 * 32);
  uint32_t ns = std::max<uint32_t>(1, std::min<uint32_t>(max_splits, (L + mc - 1) / mc));
  uint32_t ch = ((L + ns - 1) / ns + 31) / 32 * 32;
  if (ch == 0) ch = 32;
  *chunk = ch;
  *nsplit = std::max<uint32_t>(1, (L + ch - 1) / ch);
}

void DFlash2::set_h0(const float* rows) {
  memcpy(h_.ptr(), rows, size_t(block()) * cfg_.H * 4);
  h0_set_ = true;
}

void DFlash2::encode_draft(Encoder& e, Seq& s, DFlashSeq& d, int P, const SamplingParams& sp) {
  encode_draft_impl(e, s, d, P, sp, false);
}

void DFlash2::encode_draft_impl(Encoder& e, Seq& s, DFlashSeq& d, int P, const SamplingParams& sp, bool gpu,
                                bool tree) {
  // gpu: the anchor position comes from the drafter state (gst_); P is only its upper bound here
  if (!gpu && P != d.ctx) fatal("DFlash2 draft: anchor position %d but the ring holds %d positions", P, d.ctx);
  const uint32_t B = uint32_t(block()), BT = B / 8, D = B - 1;  // block rows, 8-row tiles, lattice depth
  if (P + int(B) >= s.cap) fatal("DFlash2 draft: context full (%d)", s.cap);
  if (B > 8 && (cfg_.v1 || !tree)) fatal("DFlash2 block %u: KFlash tree cycles on the lattice only", B);
  const uint32_t Pk = gpu ? 0xffffffffu : uint32_t(P);  // kDflGpu in the kernels
  const uint32_t H = uint32_t(cfg_.H), G = H / uint32_t(cfg_.conv_gs), I = uint32_t(cfg_.I);
  const uint32_t V = uint32_t(cfg_.vocab), NL = uint32_t(cfg_.NL);
  // block rows: embedding (or the test override) + the first input norm
  if (h0_set_) {
    kp::DflNorm p{H, cfg_.eps, 1.0f, 1u};
    e.use(dpipe("dfl_norm_x8"));
    e.buf(0, h_); e.buf(1, layers_[0].in_w->buf); e.buf(2, n_); e.buf(3, xh_); e.buf(4, xs_);
    e.val(5, p);
    e.dispatch(B, 1, 1, 256);
  } else {
    const QLinear& E = tg_.embed_table();
    kp::DflEmbed p{H, uint32_t(E.gs), Pk, uint32_t(cfg_.mask_id), cfg_.eps};
    e.use(dpipe(E.q4() ? std::string("dfl_embed_norm") : "dfl_embed_norm_" + E.fmt()));
    e.buf(0, E.w->buf); e.buf(1, E.s->buf); e.buf(2, E.b->buf); e.buf(3, s.tok); e.buf(4, h_);
    e.buf(5, layers_[0].in_w->buf); e.buf(6, n_); e.buf(7, xh_); e.buf(8, xs_);
    e.val(9, p);
    e.buf(10, gst_);
    e.dispatch(B, 1, 1, 256);
  }
  dbg(e, "h_0", h_, size_t(8) * H * 4);
  {
    e.use(dpipe("dfl_rope_table"));
    e.buf(0, inv_freq_); e.buf(1, rope_cs_);
    e.val(2, Pk);
    e.buf(3, gst_);
    const uint32_t nthr = B * 64;
    e.val(4, nthr);
    e.dispatch(1, 1, 1, nthr);
  }
  // Draft attention: dfl_attn_qk (q / k norm + RoPE fused in) over at most attn_splits_ context splits of >= 32 keys.
  // The previous dfl_qk_prep + dfl_attn_mma over 256-key splits ran 8 kv heads x 1-3 threadgroups at short contexts,
  // most of the GPU idle (M5 Max, 35B, 512 tokens from an 85-token prompt: attention + reduce 0.24 -> 0.17 ms per
  // drafting pass, the pass 1.48 -> 1.38 ms; 8 / 16 / 32 splits equal there). V8Knobs::dfl (NI_AB A/B): 1 = the
  // previous attention, 2 / 4 = 8 / 32 splits.
  const int dk = v8_knobs().dfl;
  const bool old_attn = dk & 1;
  const uint32_t splits = (dk & 2) ? 8u : (dk & 4) ? 32u : uint32_t(attn_splits_);
  uint32_t lo, chunk, nsplit;
  dfl_attn_plan(P, old_attn ? uint32_t(max_splits_) : splits, old_attn ? 256u : 0u, &lo, &chunk, &nsplit, d.valid_lo);
  if (gpu) {  // the kernels derive the plan from the GPU-side P: [max(P - 2047, valid_lo), P)
    lo = uint32_t(std::max(d.valid_lo, 0));
    chunk = old_attn ? kGpuChunk : 0u;  // 0: dfl_attn_geom's plan (at most `nsplit` splits)
    nsplit = old_attn ? (kCtxKeep + kGpuChunk - 1) / kGpuChunk : splits;  // the grid; empty splits exit
  }
  const kp::DflConv cp{H, G, cfg_.eps};
  // timing only: 1024 MLP gate/up, 2048 MLP down, 4096 LM head, 8192 projections, 32768 attention, 262144 its
  // reduce (35B, 512 tokens: 0.04 ms per pass)
  const int skip = v8_knobs().skip;
  auto proj = [&](const std::vector<DflSeg8>& segs, uint32_t K) {  // conv / attention projections
    if (!(skip & 8192)) enc_qmm8(e, segs, xh_, xs_, K, K, int(BT));
  };
  const bool v1 = cfg_.v1;
  // v1 (no block conv): h += u, then the next norm into n_ and the X8 table
  auto res_norm = [&](const Tensor* w) {
    kp::DflNorm np{H, cfg_.eps, 1.0f, 1u};
    e.use(dpipe("dfl_res_norm"));
    e.buf(0, h_); e.buf(1, u_); e.buf(2, w->buf); e.buf(3, n_); e.buf(4, xh_); e.buf(5, xs_);
    e.val(6, np);
    e.dispatch(B, 1, 1, 256);
  };
  for (uint32_t l = 0; l < NL; ++l) {
    const DLayer& Ly = layers_[l];
    const std::string sfx = "_" + std::to_string(l);
    dbg(e, "n" + sfx, n_, size_t(8) * H * 4);
    // non-v1: the attention conv's projection with dfl_conv_prep fused in (dfl_qmm8_cp) writes the q|kv input to the
    // second X8 table; v1: the X8 table already holds the input norm's rows (dfl_embed_norm / the previous res_norm)
    if (!v1 && !(skip & 8192)) enc_qmm8_cp(e, Ly.ac, Ly.ac_base, xh_, xs_);
    dbg(e, "D" + sfx, dyn_, size_t(8) * 4 * G * 4);
    const Buffer& ah = v1 ? xh_ : xh2_;
    const Buffer& as = v1 ? xs_ : xs2_;
    dbg(e, "a" + sfx, ah, size_t(8) * H * 2);
    if (!(skip & 8192)) enc_qmm8(e, {{&Ly.q, q_, 4096, 1.0f}, {&Ly.kv, kv_, 2048, 1.0f}}, ah, as, H, H, int(BT));
    // simdgroup-matrix attention by default (35B full window: 3.60 vs 4.52 ms per pass);
    // NI_DFLASH_ATTN=scalar selects the fp32 scalar kernel
    static const bool mma_env = !(getenv("NI_DFLASH_ATTN") && std::string(getenv("NI_DFLASH_ATTN")) == "scalar");
    const bool mma = attn_variant_ ? attn_variant_ == 2 : mma_env;
    const bool fused = mma && !old_attn && !debug_;  // debug_ keeps Q / Kb / Vb for the dumps
    if (!fused && B > 8) fatal("DFlash2 block %u: the fused attention only (no NI_DFLASH_ATTN=scalar / debug)", B);
    if (!fused) {
      kp::DflQk p{uint32_t(P), cfg_.eps, 4096, 2048};
      e.use(dpipe("dfl_qk_prep"));
      e.buf(0, q_); e.buf(1, kv_); e.buf(2, Ly.qn->buf); e.buf(3, Ly.kn->buf); e.buf(4, rope_cs_);
      e.buf(5, Q_); e.buf(6, Kb_); e.buf(7, Vb_);
      e.val(8, p);
      e.dispatch(40, 8, 1, 128);
    }
    dbg(e, "q" + sfx, Q_, size_t(32) * 8 * 128 * 4);
    dbg(e, "kb" + sfx, Kb_, size_t(8) * 8 * 128 * 4);
    dbg(e, "vb" + sfx, Vb_, size_t(8) * 8 * 128 * 4);
    if (!(skip & 32768)) {
      const size_t lstride = size_t(8) * kWin * 128 * 2;
      if (fused) {
        kp::DflAttnQk p{Pk, lo, chunk, nsplit, 1.0f / std::sqrt(128.0f), cfg_.eps, 4096, 2048};
        e.use(dpipe(B > 8 ? "dfl_attn_qk16" : "dfl_attn_qk"));
        e.buf(0, q_); e.buf(1, kv_); e.buf(2, Ly.qn->buf); e.buf(3, Ly.kn->buf); e.buf(4, rope_cs_);
        e.buf(5, d.ring_k, l * lstride); e.buf(6, d.ring_v, l * lstride); e.buf(7, part_o_); e.buf(8, part_ml_);
        e.val(9, p);
        e.buf(10, gst_);
        e.dispatch(nsplit, 8, BT, 128);
      } else {
        kp::DflAttn p{Pk, lo, chunk, nsplit, 1.0f / std::sqrt(128.0f)};
        e.use(dpipe(mma ? "dfl_attn_mma" : "dfl_attn"));
        e.buf(0, Q_); e.buf(1, d.ring_k, l * lstride); e.buf(2, d.ring_v, l * lstride); e.buf(3, Kb_); e.buf(4, Vb_);
        e.buf(5, part_o_); e.buf(6, part_ml_);
        e.val(7, p);
        e.buf(8, gst_);
        e.dispatch(nsplit, 8, 1, 128);
      }
      if (!(skip & 262144)) {
        kp::DflAttnRed rp{nsplit, Pk, lo, chunk, B};
        e.use(dpipe("dfl_attn_reduce"));
        e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, xh_); e.buf(3, xs_);
        e.val(4, rp);
        e.buf(5, gst_);
        e.dispatch(32, B, 1, 128);
      }
    }
    dbg(e, "o" + sfx, xh_, size_t(8) * 4096 * 2);
    proj({{&Ly.o, u_, H, 1.0f}}, 4096);
    dbg(e, "u" + sfx, u_, size_t(8) * H * 4);
    if (v1) {
      res_norm(Ly.post_w);
    } else {
      e.use(dpipe("dfl_conv_fin_norm"));
      e.buf(0, h_); e.buf(1, u_); e.buf(2, dyn_); e.buf(3, Ly.ac_base->buf); e.buf(4, Ly.post_w->buf);
      e.buf(5, n_); e.buf(6, xh_); e.buf(7, xs_);
      e.val(8, cp);
      e.dispatch(B, 1, 1, 256);
    }
    dbg(e, "h_mid" + sfx, h_, size_t(8) * H * 4);
    dbg(e, "n2" + sfx, n_, size_t(8) * H * 4);
    if (!v1 && !(skip & 8192)) enc_qmm8_cp(e, Ly.mc, Ly.mc_base, xh_, xs_);  // the MLP conv, likewise
    dbg(e, "D2" + sfx, dyn_, size_t(8) * 4 * G * 4);
    const Buffer& mh = v1 ? xh_ : xh2_;
    const Buffer& ms = v1 ? xs_ : xs2_;
    dbg(e, "mm" + sfx, mh, size_t(8) * H * 2);
    if (v8_) {  // the target's verify GEMMs (same math; the sums layout is shared, only the table permutes)
      enc_to_v8(e, mh, H, v8t_);
      if (!(skip & 1024)) tg_.enc_gu8(e, Ly.gate, Ly.up, v8t_, ms, v8h_, v8hs_);
      if (debug_) {
        kp::DflPerm pp{I};
        e.use(dpipe("dfl_v8_to_x8"));
        e.buf(0, v8h_); e.buf(1, hh_);
        e.val(2, pp);
        e.dispatch((I / 64 + 7) / 8, 1, 1, 256);
      }
      dbg(e, "hgu" + sfx, hh_, size_t(8) * I * 2);
      if (!(skip & 2048)) tg_.enc_qmm8(e, {{&Ly.down, u_, H, false}}, 8, v8h_, v8hs_);
    } else {
      if (skip & 1024) {
      } else if (!(dk & 128) && tw_for({{&Ly.gate, gu_g_, I, 1.0f}, {&Ly.up, gu_u_, I, 1.0f}}, H)) {
        // on the matrix units: gate and up as two segments of one qmm8t GEMM, then dfl_swiglu_x8 (35B drafting pass
        // -0.046 to -0.066 ms against dfl_qmm8_gu, NI_AB=dfl=128, 3 runs; V8Knobs::dfl 128 keeps dfl_qmm8_gu)
        enc_qmm8(e, {{&Ly.gate, gu_g_, I, 1.0f}, {&Ly.up, gu_u_, I, 1.0f}}, mh, ms, H, H, int(BT));
        e.use(dpipe("dfl_swiglu_x8"));
        e.buf(0, gu_g_); e.buf(1, gu_u_); e.buf(2, hh_); e.buf(3, hs_);
        e.val(4, I);
        e.dispatch(B, (I / 64 + 7) / 8, 1, 256);
      } else {
        if (B > 8) fatal("DFlash2 block %u: the MLP needs the tensor path", B);
        enc_qmm8_gu(e, Ly.gate, Ly.up, mh, ms);
      }
      dbg(e, "hgu" + sfx, hh_, size_t(8) * I * 2);
      if (!(skip & 2048)) enc_qmm8(e, {{&Ly.down, u_, H, 1.0f}}, hh_, hs_, I, I, int(BT));
    }
    dbg(e, "f" + sfx, u_, size_t(8) * H * 4);
    const Tensor* next = l + 1 < NL ? layers_[l + 1].in_w : norm_;
    if (v1) {
      res_norm(next);
    } else {
      e.use(dpipe("dfl_conv_fin_norm"));
      e.buf(0, h_); e.buf(1, u_); e.buf(2, dyn_); e.buf(3, Ly.mc_base->buf); e.buf(4, next->buf);
      e.buf(5, n_); e.buf(6, xh_); e.buf(7, xs_);
      e.val(8, cp);
      e.dispatch(B, 1, 1, 256);
    }
    dbg(e, "h_" + std::to_string(l + 1), h_, size_t(8) * H * 4);
  }
  dbg(e, "hn", n_, size_t(8) * H * 4);
  // target LM head (fp32 logits, all 8 rows; row 0 is never used) + selector hidden
  const QLinear& head = draft_head();
  const uint32_t Vd = draft_vocab();  // draft logit columns: the vocabulary or the reduced head
  if (skip & 4096) {
  } else if (v1) {  // no candidate selector
    if (v8_) {
      enc_to_v8(e, xh_, H, v8t_);
      tg_.enc_qmm8(e, {{&head, logits_, Vd, false}}, 8, v8t_, xs_);
    } else {
      enc_qmm8(e, {{&head, logits_, Vd, 1.0f}}, xh_, xs_, H, H, int(BT));
    }
  } else if (v8_ && head.bits == sel_w_.bits) {
    enc_to_v8(e, xh_, H, v8t_);
    tg_.enc_qmm8(e, {{&head, logits_, Vd, false}, {&sel_w_, sel_, uint32_t(kRank), false}}, 8, v8t_, xs_);
  } else {
    enc_qmm8(e, {{&head, logits_, Vd, 1.0f}, {&sel_w_, sel_, uint32_t(kRank), 1.0f}}, xh_, xs_, H, H, int(BT));
  }
  const uint32_t ntg = (Vd + 2047) / 2048;
  static const bool old_topk = getenv("NI_DFLASH_TOPK") && atoi(getenv("NI_DFLASH_TOPK")) == 0;
  if (old_topk) {  // reference path: 16 rounds of threadgroup argmax per stage
    if (head_n_) fatal("NI_DFLASH_TOPK=0 needs the full draft head (NI_DFLASH_HEAD=0)");
    kp::TopkStage ts{V, 16, 2048};
    e.use(dpipe("topk_stage1_rows"));
    e.buf(0, logits_, size_t(V) * 4); e.buf(1, tk_v_); e.buf(2, tk_i_);
    e.val(3, ts);
    e.dispatch(ntg, D, 1, 256);
    kp::DflMerge p{ntg * 16};
    e.use(dpipe("dfl_merge16"));
    e.buf(0, tk_v_); e.buf(1, tk_i_); e.buf(2, cand_v_); e.buf(3, cand_i_);
    e.val(4, p);
    e.dispatch(D, 1, 1, 256);
  } else {
    kp::DflTop p1{Vd, Vd, ntg};
    const uint32_t no = head_n_ ? 2u : 0u, yes = 1;  // stage 1 ids: the index, or the reduced head's id map
    e.use(dpipe("dfl_top16"));
    e.buf(0, logits_, size_t(Vd) * 4); e.buf(1, head_n_ ? head_ids_ : tk_i_); e.buf(2, tk_v_); e.buf(3, tk_i_);
    e.val(4, p1);
    e.val(5, no);
    e.dispatch(ntg, D, 1, 256);
    kp::DflTop p2{ntg * 16, ntg * 16, 1};
    e.use(dpipe("dfl_top16"));  // every dispatch sets its pipeline (NI_PROFILE splits encoders)
    e.buf(0, tk_v_); e.buf(1, tk_i_); e.buf(2, cand_v_); e.buf(3, cand_i_);
    e.val(4, p2);
    e.val(5, yes);
    e.dispatch(1, D, 1, 256);
  }
  {
    if (!v1) {  // v1: no selector, the edge scores stay 0 (each proposal from its own row's top-16)
      e.use(dpipe("dfl_lattice"));
      e.buf(0, cand_i_); e.buf(1, sel_); e.buf(2, cbA_->buf); e.buf(3, cbB_->buf); e.buf(4, s.tok); e.buf(5, edges_);
      e.val(6, Pk);
      e.buf(7, gst_);
      e.dispatch(D, 8, 1, 256);  // 8 threadgroups per position (7 x 1: +0.006 to +0.018 ms per pass)
    }
    if (tree) {  // KFlash: the lattice's children tables, then the shared best-first builder (no walk)
      e.use(dpipe("dfl_kf_table"));
      e.buf(0, cand_i_); e.buf(1, cand_v_); e.buf(2, edges_); e.buf(3, kf_clq_); e.buf(4, kf_tok_);
      e.dispatch(D * 16, 1, 1, 32);
      KfParams kp{};
      kf_budget_->write(kp);
      // the verify's capacity this cycle: the budget's (8, or the widest count when the per-cycle choice among them
      // pays), as far as the context holds it (rows P .. P + R - 1 of a verify at P < s.cap - R - 1; the build's rows
      // too: their tokens go to tokens[P + r]); the verify encodes it (tree_cap_), the builder's rule picks the live rows
      const int room = std::max(kB, (s.cap - P - 2) / kB * kB);
      if (kp.max_rows > room) kp.max_rows = room;
      tree_cap_ = std::min(kp.cap > 0 ? int(kp.cap) : int(kp.max_rows), int(kp.max_rows));
      if (tree_cap_ < kf::kMaxRows) kp.allowed &= (1u << tree_cap_) - 1u;
      if (!kp.allowed) kp.allowed = 1u << (tree_cap_ - 1);
      if (kp.force_n > tree_cap_) kp.force_n = 0;
      kp.cap = tree_cap_;
      // the GPU-driven anchor: the builder's P is unknown to the host, so the tokens' base comes from the drafter state
      // (kf_build reads P from its params: the host passes P; a GPU-driven pass passes the sentinel and the kernel reads
      // gst) -- see kf_build_gpu below
      const kp::KfBuild bp{16u, Pk, 0u, 0u};
      e.use(dpipe(gpu ? "kf_build_gpu" : "kf_build"));
      e.buf(0, kf_clq_); e.buf(1, kf_tok_); e.buf(2, kf_next_);
      e.val(3, kp);
      e.buf(4, tree_buf_); e.buf(5, s.tok); e.buf(6, kf_live_);
      e.val(7, bp);
      if (gpu) e.buf(8, gst_);
      e.dispatch(1, 1, 1, 32);
      return;
    }
    kp::DflSelect p{Pk, sp.temperature, sp.seed, uint32_t(P + 1), sp.temperature * walk_tf_};
    e.use(dpipe("dfl_walk"));
    e.buf(0, cand_i_); e.buf(1, cand_v_); e.buf(2, edges_); e.buf(3, s.tok); e.buf(4, walk_s_); e.buf(5, qprob_);
    e.buf(6, walk_k_);
    e.val(7, p);
    e.buf(8, gst_);
    e.dispatch(1, 1, 1, 32);
  }
}

// ---- acceptance and the decode cycle ----

void DFlash2::encode_dist(Encoder& e, int R, const SamplingParams& sp) {
  if (R < 1 || R > kB || sp.temperature <= 0) fatal("DFlash2 dist: R=%d T=%g", R, sp.temperature);
  kp::DflDist dp{tg_.verify_candidates(sp), uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)), sp.temperature,
                 sp.top_p};
  e.use(dpipe("dfl_dist_rows"));
  e.buf(0, tg_.verify_cand_vals()); e.buf(1, tg_.verify_cand_ids()); e.buf(2, dist_id_); e.buf(3, dist_p_);
  e.buf(4, dist_keep_);
  e.val(5, dp);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void DFlash2::encode_accept(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp) {
  if (K < 1 || K > kB - 1) fatal("DFlash2 accept: K=%d", K);
  const bool sampled = sp.temperature > 0;
  if (sampled) encode_dist(e, K + 1, sp);
  kp::DflAccept ap{uint32_t(P), uint32_t(K), sampled ? sp.temperature : 0.0f, sp.seed, dyn_tau_ > 0 ? 1u : 0u};
  e.use(dpipe("dfl_accept"));
  e.buf(0, s.tok); e.buf(1, tg_.verify_samples()); e.buf(2, dist_id_); e.buf(3, dist_p_); e.buf(4, cand_i_);
  e.buf(5, qprob_); e.buf(6, walk_k_); e.buf(7, acc_);
  e.val(8, ap);
  e.buf(9, kdyn_);
  e.buf(10, gst_);
  e.dispatch(1, 1, 1, 32);
}

void DFlash2::encode_cycle_draft(Encoder& e, Seq& s, DFlashSeq& d, int P, int m_prev, const SamplingParams& sp) {
  if (m_prev) encode_commit(e, d, m_prev);
  encode_draft_impl(e, s, d, P, sp, false, tree_);
}

void DFlash2::encode_cycle_verify(Encoder& e, Seq& s, int P, int K, const SamplingParams& sp) {
  if (tree_) {
    // the tree's rows (the drafting pass's capacity tree_cap_: 8, or 16 / 32 rows whose live tiles the GPU runs, as the
    // size rule chose them) with captures; the shared acceptance walks the rows' own tokens (argmax, or the plain
    // sampler's draw keyed by the token's position), writes the path and the budget's records, and leaves the path for
    // the K/V compaction, the target's next fold (s.pend_tree) and the next ring commit (gst[4 ..]). The builder's live
    // rows n (kf_live_, read by the GPU) limit the experts' routing on MoE targets as the chain's dynamic depth does;
    // the tree's own kernels run the live rows alone (tree[kKfN]).
    if (!s.pend_path) s.pend_path = Gpu::get().alloc(size_t(kf::kPathWords) * 4, "pend_path");
    const int R = tree_cap_;
    tg_.set_verify_tree(&tree_buf_);
    tg_.set_verify_row_limit(&kf_live_);
    // a row's logits must not depend on how the other rows route (Qwen35::set_verify_moe_exact): at T > 0 for the
    // acceptance, and greedy too, because the live rows follow the budget's state and a reply must not
    tg_.set_verify_moe_exact(exact_moe_);
    arm_verify(P, R);
    tg_.encode_verify(e, s, P, R, sp, uint32_t(P), /*v8=*/true);
    disarm();
    tg_.set_verify_moe_exact(false);
    tg_.set_verify_row_limit(nullptr);
    tg_.set_verify_tree(nullptr);
    const kp::KfAccept ap{uint32_t(P), uint32_t(R), 1u, 0u};
    e.use(dpipe("kf_accept"));
    e.buf(0, tg_.verify_samples()); e.buf(1, s.tok); e.buf(2, tree_buf_); e.buf(3, acc_);
    e.val(4, ap);
    e.buf(5, kf_live_); e.buf(6, s.pend_path); e.buf(7, gst_);
    e.dispatch(1, 1, 1, 32);
    tg_.encode_kv_tree_compact(e, s, P, tree_buf_);
    s.pend_tree = true;
    gst_tree_ = true;
    return;
  }
  gst_tree_ = false;
  const Buffer rows = kdyn_.view(4, 4);
  if (dyn_tau_ > 0) {
    kp::DflDepth dp{uint32_t(K), sp.temperature > 0 ? dyn_tau_t_ : dyn_tau_, sp.temperature > 0 ? sp.temperature : 0.0f,
                    dyn_rule_};
    e.use(dpipe("dfl_depth"));
    e.buf(0, walk_s_); e.buf(1, walk_k_); e.buf(2, qprob_); e.buf(3, kdyn_);
    e.val(4, dp);
    e.dispatch(1, 1, 1, 32);
    tg_.set_verify_row_limit(&rows);
  }
  // At T > 0 the acceptance needs each row's p(. | prefix) itself: on MoE targets keep a row's numerics
  // independent of how the other (drafted) rows route (see Qwen35::set_verify_moe_exact).
  tg_.set_verify_moe_exact(exact_moe_ && sp.temperature > 0);
  arm_verify(P);
  tg_.encode_verify(e, s, P, K + 1, sp, uint32_t(P), /*v8=*/true);
  disarm();
  tg_.set_verify_row_limit(nullptr);
  tg_.set_verify_moe_exact(false);
  encode_accept(e, s, P, K, sp);
}

void DFlash2::gpu_state_reset(int P) {
  uint32_t* g = gst_.as<uint32_t>();
  g[0] = uint32_t(P);
  g[1] = 0;
  g[2] = uint32_t(P);
}

void DFlash2::encode_cycle_draft_gpu(Encoder& e, Seq& s, DFlashSeq& d, int P_max, const SamplingParams& sp) {
  encode_commit(e, d, -1);  // the rows the last acceptance kept (none after gpu_state_reset)
  encode_draft_impl(e, s, d, P_max, sp, true, tree_);
}

void DFlash2::encode_cycle(Encoder& e, Seq& s, DFlashSeq& d, int P, int m_prev, int K, const SamplingParams& sp) {
  encode_cycle_draft(e, s, d, P, m_prev, sp);
  encode_cycle_verify(e, s, P, K, sp);
}

void DFlash2::prewarm_pipelines() {
  if (getenv("NI_PREWARM") && !strcmp(getenv("NI_PREWARM"), "0")) return;
  const double t0 = now_ms();
  Gpu& g = Gpu::get();
  // spec_dflash.metal (dfl_*), verify8.metal (qmm8*, x8_*, moe8*, attn8*, gdn_step_verify8) and the verify's
  // shared kernels; the prompt commit's prefill GEMMs
  static const char* prefixes[] = {"dfl_", "qmm8", "x8_", "moe8", "attn8", "gdn_step_verify8", "gdn_prep_verify", "v8_tree",
                                   "attn_prep", "attn_rows_reduce", "topk_stage1_rows", "sample_stage2_rows",
                                   "qmm_b4_m64_n64", "qmm_b4_m32_n32", "qmm_b8_m64_n64", "qmm_b8_m32_n32", "copy_u32"};
  std::vector<std::string> v;
  for (const std::string& n : g.function_names())
    for (const char* pre : prefixes)
      if (n.rfind(pre, 0) == 0) {
        v.push_back(n);
        break;
      }
  if (v.empty()) return;
  const unsigned nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; ++i)
    th.emplace_back([&, i] {
      for (size_t j = i; j < v.size(); j += nt) g.pipeline(v[j], false);
    });
  for (auto& t : th) t.join();
  logi("DFlash2: prewarmed %zu pipelines in %.0f ms", v.size(), now_ms() - t0);
}

} // namespace ni
