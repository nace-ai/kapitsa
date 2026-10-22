#include "model/qwen4_exp.h"
#include "loader/safetensors.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"
#include "vision/embed.h"

#include <nlohmann/json.hpp>

#include <CoreFoundation/CoreFoundation.h>
#include <dirent.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <unordered_map>

namespace ni {

using json = nlohmann::json;

namespace {

Pipeline pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  return json::parse(f);
}

// kernel parameter blocks (src/kernels/qwen4.metal)
struct HcParams {
  uint32_t R, D, S, has_y, has_inj;
  float eps;
};
struct PleParams {
  uint32_t R, D, S, pos0;
  float eps;
  uint32_t ring, dil;
};
struct QsaParams {
  uint32_t R, NH, HD, rot, pos0, ratio, ld;
  float eps, theta;
};

constexpr int kRing = 32;  // positions in the n-gram conv window (a power of two >= kMaxVerify + 9)

void parse4(const json& root, Qwen4ExpConfig& c) {
  const json t = root.contains("text_config") ? root["text_config"] : root;
  c.hc = t.value("hc_count", 4);
  c.hc_rank = t.value("hc_lowrank", 320);
  c.ix_heads = t.value("indexer_n_heads", 4);
  c.ix_dim = t.value("indexer_head_dim", 128);
  c.ix_budget = t.value("indexer_budget", 2048);
  c.ix_ratio = t.value("indexer_compress_ratio", 4);
  c.ple_layer = -1;
  if (t.contains("ple_layer_ids") && t["ple_layer_ids"].is_array() && !t["ple_layer_ids"].empty()) {
    if (t["ple_layer_ids"].size() > 1) fatal("qwen4_exp: more than one n-gram embedding layer is not supported");
    c.ple_layer = t["ple_layer_ids"][0].get<int>() - 1;  // one-indexed in the config
  }
  c.ngram = t.value("ngram_size", 3);
  c.per_ngram = t.value("heads_per_ngram", 8);
  c.seed = t.value("seed", 1234);
  c.ple_conv = t.value("ple_conv_kernel_size", 4);
  c.ngram_base = t.value("ngram_vocab_size_base", int64_t(20000000));
  const json eos = t.contains("eos_token_id") ? t["eos_token_id"] : root.value("eos_token_id", json(248044));
  c.eos = eos.is_array() ? eos[0].get<int>() : eos.get<int>();
}

// a trunk layer's routed experts (the MTP head's, mtp.layers.*, stay with the head)
bool is_trunk_expert(const std::string& n) {
  return n.find(".mlp.switch_mlp.") != std::string::npos && n.rfind("mtp.", 0) != 0;
}

// config.json's quantization gives the trunk's routed experts Q2_0 (tools/qwen4_exp_q20.py)
bool q20_experts_in(const json& root) {
  for (const char* key : {"quantization", "quantization_config"}) {
    if (!root.contains(key) || !root[key].is_object()) continue;
    for (auto& [k, v] : root[key].items())
      if (v.is_object() && is_trunk_expert(k) && v.value("mode", std::string()) == "q2_0") return true;
  }
  return false;
}

}  // namespace

bool Qwen4Exp::is_model(const std::string& dir) {
  std::ifstream f(dir + "/config.json");
  if (!f) return false;
  json r;
  try {
    r = json::parse(f);
  } catch (...) {
    return false;
  }
  if (r.value("model_type", std::string()) == "qwen4_exp") return true;
  return r.contains("text_config") && r["text_config"].is_object() &&
         r["text_config"].value("model_type", std::string()) == "qwen4_exp_text";
}

Qwen4Exp::Qwen4Exp(const std::string& dir, int prefill_chunk, size_t mem_budget)
    : Qwen35(dir, prefill_chunk, NoLoad{}), mem_budget_arg_(mem_budget) {
  load(dir);
}

std::unique_ptr<Qwen35> load_qwen(const std::string& dir, int prefill_chunk) {
  if (Qwen4Exp::is_model(dir)) return std::make_unique<Qwen4Exp>(dir, prefill_chunk);
  return std::make_unique<Qwen35>(dir, prefill_chunk);
}

Qwen4Exp::~Qwen4Exp() {
  if (ple_ev_) CFRelease(ple_ev_);
}

const Tensor* Qwen4Exp::norm32(const Tensor* t, float offset) {
  if (!t) return nullptr;
  const size_t n = size_t(t->numel());
  Buffer b = Gpu::get().alloc(n * 4 + 16, "norm_f32", /*tracked=*/false);
  float* d = b.as<float>();
  for (size_t i = 0; i < n; ++i) {
    const float v = t->dtype == DType::F32 ? t->buf.as<float>()[i]
                    : t->dtype == DType::F16 ? float(t->buf.as<_Float16>()[i])
                                             : bf16_to_f32(t->buf.as<uint16_t>()[i]);
    d[i] = v + offset;
  }
  owned_bufs_.push_back(b);
  Tensor c = *t;
  c.buf = b.view(0, n * 4);
  c.dtype = DType::F32;
  owned_.push_back(c);
  return &owned_.back();
}

GatedRes Qwen4Exp::load_gr(const std::string& p, bool inject, const WeightStore*) {
  GatedRes g;
  g.norm = norm32(&ws_.get(tname(p + ".hc_norm.weight")), 1.0f);
  g.down = qlin(p + ".input_mix_weight_down");
  g.up = qlin(p + ".input_mix_weight_up");
  if (inject) g.inj = qlin(p + ".block_inject_weight");
  const uint32_t SD = uint32_t(c4_.hc * cfg_.hidden), r = uint32_t(c4_.hc_rank);
  if (g.down.N != r || g.down.K != SD || g.up.N != SD || g.up.K != r)
    fatal("%s: unexpected hyper-connection shapes", p.c_str());
  if (inject && (!g.inj.dense() || g.inj.N != uint32_t(c4_.hc) || g.inj.K != SD))
    fatal("%s.block_inject_weight: expected bf16 [%d, %u]", p.c_str(), c4_.hc, SD);
  return g;
}

int Qwen4Exp::attn_layers() const {
  int a = 0;
  for (int i = 0; i < cfg_.layers; ++i) a += !cfg_.is_linear[i];
  return a;
}

void Qwen4Exp::load(const std::string& dir) {
  const double t0 = now_ms();
  load_config(dir);
  const json root = read_json(dir + "/config.json");
  parse4(root, c4_);
  {
    const json t = root.contains("text_config") ? root["text_config"] : root;
    gdn_sigmoid_ = t.value("output_gate_type", std::string("silu")) == "sigmoid";
  }
  no_qmvh_ = true;
  if (c4_.hc != 4) fatal("qwen4_exp: %d residual streams unsupported (4 are)", c4_.hc);
  if (c4_.ix_dim != 128 || c4_.ix_heads > 8 || c4_.ix_ratio != 4) fatal("qwen4_exp: indexer shape unsupported");
  if (c4_.ple_conv != 4) fatal("qwen4_exp: n-gram conv kernel %d unsupported", c4_.ple_conv);
  if (cfg_.hidden % 128 || cfg_.hidden > 4096) fatal("qwen4_exp: hidden size %d unsupported", cfg_.hidden);
  // the trunk and the MTP head (mtp.safetensors, read by load_mtp4): every tensor but the vision tower and the n-gram
  // table (mapped below); with NI_NO_MTP the head is left out too, as plan() leaves it out
  const bool no_mtp = getenv("NI_NO_MTP") != nullptr;
  // Q2_0 routed experts go to the expert store (init_store): left unread here (deferred: listed, their bytes in the
  // files), unless NI_EXPERT_STORE=off keeps them as stacked planes
  q20_pack_ = q20_experts_in(root);
  {
    const char* es = getenv("NI_EXPERT_STORE");
    stream_ = q20_pack_ && !(es && (!strcmp(es, "off") || !strcmp(es, "0")));
  }
  WeightStore::Filter defer = transform_defer();
  if (stream_)
    defer = [td = defer](const std::string& n) { return is_trunk_expert(n) || (td && td(n)); };
  ws_.open_dir(
      dir,
      [no_mtp](const std::string& n) {
        return n.find("vision_tower") == std::string::npos && n.find("visual.") == std::string::npos &&
               n.find(".ngram_embedding.") == std::string::npos && !(no_mtp && n.rfind("mtp.", 0) == 0);
      },
      [no_mtp](const std::string& f) { return f != "ple.safetensors" && !(no_mtp && f == "mtp.safetensors"); },
      defer);
  if (ws_.has("language_model.model.embed_tokens.weight")) prefix_ = "language_model.";
  else if (ws_.has("model.embed_tokens.weight")) prefix_ = "";
  else fatal("%s: unrecognized weight naming (expected the MLX qwen4_exp layout)", dir.c_str());
  embed_ = qlin("model.embed_tokens");
  lm_head_ = qlin("lm_head");
  final_norm_ = nullptr;  // the final normalization is the hyper-connection mix
  layers_.assign(size_t(cfg_.layers), Layer{});
  l4_.assign(size_t(cfg_.layers), Qwen4Layer{});
  int lossy = 0;
  auto param = [&](const std::string& n) { return bf16_param(&ws_.get(tname(n)), lossy); };
  for (int i = 0; i < cfg_.layers; ++i) {
    Layer& L = layers_[size_t(i)];
    Qwen4Layer& F = l4_[size_t(i)];
    const std::string p = "model.layers." + std::to_string(i) + ".";
    L.linear = cfg_.is_linear[size_t(i)];
    if (L.linear) {
      const std::string a = p + "linear_attn.";
      L.qkv = qlin(a + "in_proj_qkv");
      L.z = qlin(a + "in_proj_z");
      L.a = qlin(a + "in_proj_a");
      L.b = qlin(a + "in_proj_b");
      L.out = qlin(a + "out_proj");
      L.conv_w = param(a + "conv1d.weight");
      L.A_log = param(a + "A_log");
      L.dt_bias = param(a + "dt_bias");
      L.gnorm = param(a + "norm.weight");  // a plain weight (the one norm that is not zero-centred)
      const Tensor& cw = ws_.get(tname(a + "conv1d.weight"));
      if (cw.shape.size() != 3 || cw.shape[1] != 4 || cw.shape[2] != 1) fatal("%s: conv1d is not [C, 4, 1]", a.c_str());
    } else {
      const std::string a = p + "self_attn.";
      L.q = qlin(a + "q_proj");
      L.k = qlin(a + "k_proj");
      L.v = qlin(a + "v_proj");
      L.o = qlin(a + "o_proj");
      L.q_norm = bf16_copy(&ws_.get(tname(a + "q_norm.weight")), 1.0f);
      L.k_norm = bf16_copy(&ws_.get(tname(a + "k_norm.weight")), 1.0f);
      F.iqk = qlin(a + "indexer.index_qk_proj");
      F.iqn = norm32(&ws_.get(tname(a + "indexer.q_layernorm.weight")), 1.0f);
      F.ikn = norm32(&ws_.get(tname(a + "indexer.k_layernorm.weight")), 1.0f);
      if (F.iqk.N != uint32_t((c4_.ix_heads + 1) * c4_.ix_dim)) fatal("%s: index_qk_proj shape", a.c_str());
    }
    const std::string m = p + "mlp.";
    L.router = qlin(m + "gate");
    L.shared_gate = qlin(m + "shared_expert_gate");
    L.router.gate = L.shared_gate.gate = true;
    L.eg = qlin(m + "switch_mlp.gate_proj", true, true);
    L.eu = qlin(m + "switch_mlp.up_proj", true, true);
    L.ed = qlin(m + "switch_mlp.down_proj", true, true);
    L.sg = qlin(m + "shared_expert.gate_proj");
    L.su = qlin(m + "shared_expert.up_proj");
    L.sd = qlin(m + "shared_expert.down_proj");
    F.ga = load_gr(p + "attn_hyper_connection", true);
    F.gm = load_gr(p + "mlp_hyper_connection", true);
    if (i == c4_.ple_layer) {
      const std::string q = p + "ple.";
      ple_key_ = qlin(q + "key_proj");
      ple_val_ = qlin(q + "value_proj");
      ple_nk_ = norm32(&ws_.get(tname(q + "norm_key.weight")), 1.0f);
      ple_nq_ = norm32(&ws_.get(tname(q + "norm_query.weight")), 1.0f);
      ple_nc_ = norm32(&ws_.get(tname(q + "norm_conv.weight")), 1.0f);
      ple_conv_ = param(q + "conv1d.weight");
      const Tensor& cw = ws_.get(tname(q + "conv1d.weight"));
      if (cw.shape.size() != 3 || cw.shape[1] != 4 || cw.shape[2] != 1) fatal("%s: conv1d is not [C, 4, 1]", q.c_str());
    }
  }
  if (c4_.ple_layer >= 0 && !ple_key_) fatal("qwen4_exp: no n-gram embedding at layer %d", c4_.ple_layer);
  mix_ = load_gr("model.hyper_connection_mixer", false);
  if (lossy) logi("qwen4_exp: %d parameter values not exactly representable in bf16", lossy);
  for (const QLinear* q : {&embed_, &lm_head_})
    if (*q && !q->dense() && (q->s->dtype != DType::BF16 || q->b->dtype != DType::BF16))
      fatal("quantized weights with non-bf16 scales/biases are not supported");
  check_shapes();
  check_formats();

  // the n-gram table: ple.safetensors' row shards, mapped (never in GPU memory)
  if (c4_.ple_layer >= 0) {
    hash_ = NgramHash::make(cfg_.vocab, c4_.seed, 0, c4_.ngram_base, c4_.per_ngram, c4_.ngram, c4_.eos);
    const std::string tp = prefix_ + "model.layers." + std::to_string(c4_.ple_layer) + ".ple.ple_embedding.";
    // the hash constants the checkpoint carries must be the ones computed here
    if (const Tensor* lm = ws_.find(tp + "layer_multipliers"); lm && lm->dtype == DType::I64)
      for (int i = 0; i < std::min<int>(c4_.ngram, int(lm->numel())); ++i)
        if (lm->buf.as<int64_t>()[i] != hash_.mult[i]) fatal("qwen4_exp: the n-gram multipliers differ from the config's");
    if (const Tensor* hs = ws_.find(tp + "ngram_heads_vocab_sizes"); hs && hs->dtype == DType::I64)
      for (int i = 0; i < std::min<int>(hash_.heads, int(hs->numel())); ++i)
        if (hs->buf.as<int64_t>()[i] != hash_.mod[size_t(i)]) fatal("qwen4_exp: the n-gram head sizes differ from the config's");
    std::vector<NgramTable::Part> parts;
    const std::string file = dir + "/ple.safetensors";
    if (!std::ifstream(file)) fatal("%s: missing (the n-gram table)", file.c_str());
    for (int k = 0;; ++k) {
      const std::string n = tp + "ngram_embedding.shards." + std::to_string(k);
      // the shard count from the header: the parts are listed until one is missing
      NgramTable::Part pt{file, n + ".weight", n + ".scales", n + ".biases"};
      if (!NgramTable::has_tensor(file, pt.w)) break;
      parts.push_back(pt);
    }
    if (parts.empty()) fatal("%s: no n-gram table shards", file.c_str());
    table_.open(parts, NgramTable::Fmt::Affine, cfg_.hidden / hash_.heads);
    if (table_.rows() < hash_.rows) fatal("qwen4_exp: the n-gram table has %lld rows, the hash needs %lld",
                                          (long long)table_.rows(), (long long)hash_.rows);
    if (table_.dim() * hash_.heads != cfg_.hidden) fatal("qwen4_exp: n-gram rows of %d values", table_.dim());
  }
  test_transforms();  // NI_REQUANT / NI_DEQUANT (e.g. NI_REQUANT=6:gdn,shared,head: the all-dense-6-bit recipe's speed)
  if (!no_mtp) load_mtp4(dir);
  Gpu::get().make_resident(ws_.buffers());
  Gpu::get().make_resident(owned_bufs_);
  alloc_state();
  alloc4();
  if (stream_) init_store(dir);
  prewarm_pipelines();
  {
    std::vector<std::string> v = {"hc_pre", "hc_pre_h", "hc_act", "hc_act_h", "hc_mix", "hc_mix_h", "hc_expand", "hc_inject",
                                  "ple_rows", "ple_gate_rows", "ple_conv_rows", "ple_ring_rows", "qsa_prep", "qsa_block",
                                  "gdn_step_sg", "gdn_gnorm_sg", "gdn_step_verify_sg", "moe_topk", "moe_topk_rows",
                                  "moe_down2_w", "moe_down_rows_g64_w"};
    for (const std::string& n : v) Gpu::get().pipeline(n, false);
  }
  ple_ev_ = Gpu::get().new_shared_event();
  logi("loaded %s: Qwen3.8-Flash-Next, %d layers (%d attention), %.1f GB of weights in %.0f ms; %s; n-gram table %s%s%s",
       dir.c_str(), cfg_.layers, attn_layers(), ws_.total_bytes() / 1e9, now_ms() - t0, formats_summary().c_str(),
       table_.ok() ? table_.describe().c_str() : "none", store_ ? "; routed experts: " : "",
       store_ ? store_->describe().c_str() : "");
}

void Qwen4Exp::load_mtp4(const std::string& dir) {
  if (!std::ifstream(dir + "/mtp.safetensors")) {
    logi("no MTP head in %s (mtp.safetensors); speculative decoding off", dir.c_str());
    return;
  }
  if (!ws_.has("mtp.fc_embedding.weight")) {
    logi("mtp.safetensors without mtp.fc_embedding; speculative decoding off");
    return;
  }
  const double t0 = now_ms();
  // the head's tensors are named mtp.* (no language_model. prefix)
  const std::string keep = prefix_;
  prefix_ = "";
  int lossy = 0;
  (void)lossy;
  mtp_ne_ = norm32(&ws_.get("mtp.pre_fc_norm_embedding.weight"), 1.0f);
  mtp_nh_ = norm32(&ws_.get("mtp.pre_fc_norm_hidden.weight"), 1.0f);
  mtp_fce_ = qlin("mtp.fc_embedding");
  mtp_fch_ = qlin("mtp.fc_hidden");
  mtp_fce_.mtp = mtp_fch_.mtp = true;
  const std::string p = "mtp.layers.0.";
  Layer& L = mtp_;
  L.linear = false;
  L.q = qlin(p + "self_attn.q_proj");
  L.k = qlin(p + "self_attn.k_proj");
  L.v = qlin(p + "self_attn.v_proj");
  L.o = qlin(p + "self_attn.o_proj");
  L.q_norm = bf16_copy(&ws_.get(p + "self_attn.q_norm.weight"), 1.0f);
  L.k_norm = bf16_copy(&ws_.get(p + "self_attn.k_norm.weight"), 1.0f);
  mtp4_.iqk = qlin(p + "self_attn.indexer.index_qk_proj");
  mtp4_.iqn = norm32(&ws_.get(p + "self_attn.indexer.q_layernorm.weight"), 1.0f);
  mtp4_.ikn = norm32(&ws_.get(p + "self_attn.indexer.k_layernorm.weight"), 1.0f);
  L.router = qlin(p + "mlp.gate");
  L.shared_gate = qlin(p + "mlp.shared_expert_gate");
  L.eg = qlin(p + "mlp.switch_mlp.gate_proj");
  L.eu = qlin(p + "mlp.switch_mlp.up_proj");
  L.ed = qlin(p + "mlp.switch_mlp.down_proj");
  L.sg = qlin(p + "mlp.shared_expert.gate_proj");
  L.su = qlin(p + "mlp.shared_expert.up_proj");
  L.sd = qlin(p + "mlp.shared_expert.down_proj");
  for (QLinear* q : {&L.q, &L.k, &L.v, &L.o, &mtp4_.iqk, &L.router, &L.shared_gate, &L.eg, &L.eu, &L.ed, &L.sg, &L.su, &L.sd})
    q->mtp = true;
  mtp4_.ga = load_gr(p + "attn_hyper_connection", true);
  mtp4_.gm = load_gr(p + "mlp_hyper_connection", true);
  mtp_mix_ = load_gr("mtp.hyper_connection_mixer", false);
  prefix_ = keep;
  if (!L.eg.same_format(L.eu) || !L.eg.same_format(L.ed) || moe_rows_kernel(true, L.eg, L.sg).empty() ||
      moe_rows_kernel(false, L.ed, L.sd, moe_wide()).empty())
    fatal("MTP head: no MoE kernels for %s experts with a %s shared expert", format_name(L.eg.bits, L.eg.gs).c_str(),
          format_name(L.sg.bits, L.sg.gs).c_str());
  // the draft head: the lowest 32768 token ids (BPE merge order, a frequency proxy) plus the added / special tokens
  draft_split_ = std::min<uint32_t>(32768, uint32_t(cfg_.vocab));
  draft_special0_ = uint32_t(cfg_.eos.empty() ? cfg_.vocab : *std::min_element(cfg_.eos.begin(), cfg_.eos.end()));
  full_draft_ = getenv("NI_FULL_DRAFT") && *getenv("NI_FULL_DRAFT") ? env_on("NI_FULL_DRAFT") : false;
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
  mtp_loaded_ = true;
  logi("MTP head loaded in %.0f ms (drafts over %s)", now_ms() - t0, full_draft_ ? "the full vocabulary" : "ids < 32768 + specials");
}

// Buffers beside Qwen35's (alloc_state): the 4-stream residual replaces x_, the mix's scratch, the indexer's rows,
// the n-gram rows. qwen4_act_bytes (plan) adds up the same.
static size_t qwen4_act_bytes(const Qwen35Config& c, const Qwen4ExpConfig& c4, int T, int maxv, bool mtp) {
  const size_t D = size_t(c.hidden), SD = size_t(c4.hc) * D, Tz = size_t(T), V = size_t(maxv), R = size_t(c4.hc_rank);
  const size_t IQ = size_t(c4.ix_heads + 1) * c4.ix_dim;
  return Tz * SD * 4 - Tz * D * 4  // x_ grows from [T, D] to [T, S*D]
         + Tz * D * 4 + 2 * Tz * c4.hc * 4 + V * SD * 4 + Tz * SD * 2 + Tz * R * 4 + Tz * R * 2 + Tz * SD * 4 +
         Tz * IQ * 4 + Tz * c4.ix_heads * c4.ix_dim * 4 + Tz * SD * 4 + Tz * D * 4 + 2 * Tz * D * 2 + 4 * V * D * 4 +
         (mtp ? Tz * SD * 4 + V * SD * 4 - V * D * 4 : 0)  // the MTP head's state; mtp_post_ widened
         + Tz * size_t(c4.ix_budget + c4.ix_ratio + 3) / 4 * 4 * 4 + Tz * 4  // sparse attention's key lists
         + SD * 4 + 64 + (SD / 1024 + 1) * V * R * 4;  // the one-row gated residual's row, hc_down_sk's partials
}

void Qwen4Exp::alloc4() {
  Gpu& g = Gpu::get();
  const size_t D = size_t(cfg_.hidden), SD = size_t(c4_.hc) * D, T = size_t(tmax_), V = size_t(kMaxVerify);
  const size_t R = size_t(c4_.hc_rank), IQ = size_t(c4_.ix_heads + 1) * c4_.ix_dim;
  g.release(x_);
  x_ = g.alloc(T * SD * 4, "x4");
  yb_ = g.alloc(T * D * 4, "block_out");
  memset(yb_.ptr(), 0, yb_.size);  // accumulated into: zero between blocks (the injections zero it again)
  injA_ = g.alloc(T * c4_.hc * 4, "inject_a");
  injM_ = g.alloc(T * c4_.hc * 4, "inject_m");
  xn4_ = g.alloc(V * SD * 4, "hc_xn");
  xn4h_ = g.alloc(T * SD * 2, "hc_xn_half");
  hch_ = g.alloc(T * R * 4, "hc_h");
  hchh_ = g.alloc(T * R * 2, "hc_h_half");
  hcm_ = g.alloc(T * SD * 4, "hc_m");
  iqk_ = g.alloc(T * IQ * 4, "index_qk");
  iq_ = g.alloc(T * c4_.ix_heads * c4_.ix_dim * 4, "index_q");
  ple_k_ = g.alloc(T * SD * 4, "ple_key");
  ple_v_ = g.alloc(T * D * 4, "ple_value");
  for (Buffer& b : ple_pf_) b = g.alloc(T * D * 2, "ple_rows");
  for (Buffer& b : ple_dec_) b = g.alloc(V * D * 4, "ple_rows_decode");
  hpart_ = g.alloc((SD / 1024 + 1) * V * R * 2 * 4, "hc_part");
  hx2_ = g.alloc(V * SD * 4, "hc_rows");
  hstat_ = g.alloc((SD / 1024 + 1) * V * 10 * 4, "hc_stat");
  xr2_ = g.alloc(SD * 4, "hc_row");
  hinv_ = g.alloc(64, "hc_inv");
  qidx_ld_ = uint32_t((c4_.ix_budget + c4_.ix_ratio + 3) / 4 * 4);  // budget + ratio - 1 entries, rounded
  qidx_ = g.alloc(T * qidx_ld_ * 4, "qsa_idx");
  qcnt_ = g.alloc(T * 4 + 16, "qsa_cnt");
  if (mtp_loaded_) {
    mtpX_ = g.alloc(T * SD * 4, "mtp_x4");
    g.release(mtp_post_);
    mtp_post_ = g.alloc(V * SD * 4, "mtp_post4");
  }
}

// ---- memory ----

std::unique_ptr<Seq> Qwen4Exp::new_seq(int limit, int initial) {
  auto s = Qwen35::new_seq(limit, 0);
  Gpu& g = Gpu::get();
  s->ixk.assign(size_t(cfg_.layers), Buffer{});
  s->ixb.assign(size_t(cfg_.layers), Buffer{});
  s->pconv = g.alloc(size_t(kRing) * c4_.hc * cfg_.hidden * 4, "ple_conv_state");
  memset(s->pconv.ptr(), 0, s->pconv.size);
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0) grow_kv(*s, cap);
  return s;
}

void Qwen4Exp::grow_kv(Seq& s, int cap) {
  const int old = s.cap;
  Qwen35::grow_kv(s, cap);
  // the indexer's caches grow alongside: rows by position, so the old buffer is the new one's prefix
  Gpu& g = Gpu::get();
  const size_t kb = size_t(cap) * c4_.ix_dim * 2, bb = (size_t(cap) / c4_.ix_ratio + 1) * c4_.ix_dim * 4;
  std::vector<std::pair<Buffer*, size_t>> bufs;
  for (int i = 0; i < cfg_.layers; ++i)
    if (!cfg_.is_linear[size_t(i)]) {
      bufs.push_back({&s.ixk[size_t(i)], kb});
      bufs.push_back({&s.ixb[size_t(i)], bb});
    }
  if (mtp_loaded_) {
    bufs.push_back({&s.mtp_ixk, kb});
    bufs.push_back({&s.mtp_ixb, bb});
  }
  std::vector<Buffer> olds;
  std::unique_ptr<Encoder> e;
  for (auto& [b, n] : bufs) {
    Buffer nb = g.alloc(n, "index_cache");
    if (old > 0 && *b) {
      if (!e) e = g.begin();
      enc_copy(*e, *b, nb);
    }
    olds.push_back(*b);
    *b = nb;
  }
  if (e) e->commit_and_wait();
  for (Buffer& b : olds) g.release(b);
}

void Qwen4Exp::drop_kv(Seq& s) {
  Qwen35::drop_kv(s);
  Gpu& g = Gpu::get();
  for (auto* v : {&s.ixk, &s.ixb})
    for (Buffer& b : *v) g.release(b);
  g.release(s.mtp_ixk);
  g.release(s.mtp_ixb);
}

size_t Qwen4Exp::kv_bytes_per_pos() const {
  const size_t layers = size_t(attn_layers() + (mtp_loaded_ ? 1 : 0));
  return layers * (size_t(2) * cfg_.hkv * cfg_.head_dim * 2 + size_t(c4_.ix_dim) * 2 + size_t(c4_.ix_dim) * 4 / c4_.ix_ratio);
}

void Qwen4Exp::reset(Seq& s) {
  Qwen35::reset(s);
  if (s.pconv) memset(s.pconv.ptr(), 0, s.pconv.size);
  if (pend_.s == &s) pend_ = Pending{};
}

Checkpoint Qwen4Exp::new_checkpoint() {
  Checkpoint c = Qwen35::new_checkpoint();
  c.pconv = Gpu::get().alloc(size_t(kRing) * c4_.hc * cfg_.hidden * 4, "ckpt_ple_conv");
  return c;
}

void Qwen4Exp::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Qwen35::save(e, s, c, pos);
  enc_copy(e, s.pconv, c.pconv);
}

void Qwen4Exp::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  Qwen35::restore(e, s, c);
  enc_copy(e, c.pconv, s.pconv);
}

// ---- the n-gram rows ----

void Qwen4Exp::gather_rows(const int* tok, int pos0, int R, void* dst, bool half_rows) const {
  if (!table_.ok()) fatal("qwen4_exp: no n-gram table");
  const int H = hash_.heads, D = cfg_.hidden;
  std::vector<int64_t> ids(size_t(R) * H);
  for (int r = 0; r < R; ++r) hash_.ids(tok, pos0 + r, ids.data() + size_t(r) * H);
  if (!half_rows) {
    // NI_PLE_STATS=1: the decode / verify gathers' time (on the GPU's path: the next pass waits for it)
    static const bool stats = env_on("NI_PLE_STATS");
    const double t0 = stats ? now_ms() : 0.0;
    table_.gather(ids.data(), R, H, static_cast<float*>(dst));
    if (stats) {
      static double sum = 0, mx = 0;
      static int n = 0;
      const double dt = now_ms() - t0;
      sum += dt, mx = std::max(mx, dt);
      if (++n % 128 == 0) {
        fprintf(stderr, "[kapitsa] n-gram gathers: %d, mean %.3f ms, max %.3f ms\n", n, sum / 128, mx);
        sum = 0, mx = 0;
      }
    }
    return;
  }
  static const bool stats = env_on("NI_PLE_STATS");
  const double t0 = stats ? now_ms() : 0.0;
  std::vector<float> tmp(size_t(R) * D);
  table_.gather(ids.data(), R, H, tmp.data());
  _Float16* h = static_cast<_Float16*>(dst);
  for (size_t i = 0; i < tmp.size(); ++i) h[i] = _Float16(tmp[i]);
  if (stats) fprintf(stderr, "[kapitsa] n-gram gather of %d prefill rows: %.1f ms\n", R, now_ms() - t0);
}

Qwen4Exp::PleWait Qwen4Exp::ple_rows_for(Seq& s, int pos) {
  if (pend_.s == &s && pend_.pos == pos && pend_.ticket) {
    const PleWait w{ple_dec_[pend_.slot], pend_.ticket};
    pend_ = Pending{};
    return w;
  }
  const int slot = ple_dec_next_;
  ple_dec_next_ = (ple_dec_next_ + 1) % 4;
  gather_rows(s.tokens(), pos, 1, ple_dec_[slot].ptr(), false);
  return PleWait{ple_dec_[slot], 0};
}

// The pass being encoded samples tokens[pos] on the GPU: once it has completed, a Metal thread gathers that token's
// n-gram rows into a staging slot and signals the event the next pass (encode_step at pos) waits on before the
// n-gram block (its first layer overlaps the gather).
void Qwen4Exp::ple_after_sample(Encoder& e, Seq& s, int pos) {
  if (!table_.ok()) return;
  const int slot = ple_dec_next_;
  ple_dec_next_ = (ple_dec_next_ + 1) % 4;
  const uint64_t t = ++ple_ticket_;
  pend_ = Pending{&s, pos, t, slot};
  const Buffer dst = ple_dec_[slot];
  const Seq* sp = &s;
  void* ev = ple_ev_;
  e.on_completed([this, sp, pos, dst, t, ev] {
    gather_rows(const_cast<Seq*>(sp)->tokens(), pos, 1, dst.ptr(), false);
    Gpu::signal_event_cpu(ev, t);
  });
}

// ---- streamed experts (src/model/expert_store.h) ----

int Qwen4Exp::layer_index(const Layer& L) const {
  const Layer* b = layers_.data();
  return &L >= b && &L < b + layers_.size() ? int(&L - b) : -1;
}

// The store over the deferred Q2_0 planes. Its size: NI_EXPERT_STORE=<GB> or all; by default what the budget leaves
// after the weights and buffers allocated so far (and the room set apart below).
void Qwen4Exp::init_store(const std::string& dir) {
  (void)dir;
  const uint32_t E = uint32_t(cfg_.experts);
  std::vector<std::string> files;
  std::map<std::string, int> fidx;
  std::vector<q20::LayerSrc> src(size_t(cfg_.layers));
  static const char* const kMat[3] = {"gate", "up", "down"};
  for (int i = 0; i < cfg_.layers; ++i)
    for (int m = 0; m < 3; ++m)
      for (int part = 0; part < 2; ++part) {
        const std::string n = tname("model.layers." + std::to_string(i) + ".mlp.switch_mlp." + kMat[m] + "_proj") +
                              (part ? ".scales" : ".weight");
        std::string path;
        size_t off = 0, len = 0;
        if (!ws_.deferred_place(n, &path, &off, &len) || len % E) fatal("expert store: %s is not a stacked plane", n.c_str());
        auto it = fidx.find(path);
        const int f = it != fidx.end() ? it->second : int(files.size());
        if (it == fidx.end()) {
          fidx[path] = f;
          files.push_back(path);
        }
        q20::PlaneSrc& p = src[size_t(i)].p[m * 2 + part];
        p.file = f;
        p.off = off;
        p.stride = len / E;
        p.len = uint32_t(len / E);
      }
  const q20::SlotLayout lay = q20::slot_layout(uint32_t(cfg_.hidden), uint32_t(cfg_.moe_I));
  const size_t all = size_t(cfg_.layers) * E * lay.bytes;
  q20::ExpertStore::Options o;
  const char* es = getenv("NI_EXPERT_STORE");
  if (es && !strcmp(es, "all")) {
    o.bytes = all;
  } else if (es && *es && strcmp(es, "auto")) {
    o.bytes = size_t(atof(es) * 1e9);
  } else {
    // what the budget leaves: the engine's (mem_budget, its conversations' KV caches and slots' state already set
    // apart) less 0.5 GB, or the default budget less a 16K-token KV cache and 1.5 GB (tools)
    size_t budget = mem_budget_arg_, keep = size_t(1) << 29;
    if (!budget) {
      const char* v = getenv("NI_MEM_BUDGET");
      const size_t rws = Gpu::get().recommended_working_set();
      budget = v && *v ? size_t(atof(v) * 1e9) : rws - std::max<size_t>(size_t(512) << 20, rws / 25);
      keep = kv_bytes_per_pos() * 16384 + (size_t(3) << 29);
    }
    const size_t used = Gpu::get().allocated_bytes();
    const size_t left = budget > used + keep ? budget - used - keep : 0;
    uint64_t ram = 0;
    size_t len = sizeof ram;
    sysctlbyname("hw.memsize", &ram, &len, nullptr, 0);
    // A store short of all the experts waits on the CPU every layer. On a 48 GB M4 Pro (macOS 27, PR #213) every
    // command buffer then took 0.5-3 s longer (its submission blocked in IOGPU) once the process's GPU allocations
    // passed ~45% of the RAM (15-24 GB stores), so the automatic size kept them within 40% of the RAM. A 48 GB M5 Pro
    // (macOS 26.5) ran 12-30 GB stores (peak footprint up to 71% of the RAM) at full speed, the larger the faster
    // (85-token prompt 29.9 -> 120.6 tok/s, decode 24.5 -> 31.9 at 12 -> 30 GB), so the cap stays for chips before
    // Apple10 only. That split is by chip, not by OS: the two runs also differ in macOS (27 vs 26.5), and which of
    // the two explains the stall is not measured. NI_EXPERT_CAP=<share of the RAM, 0..1> sets it on any chip (0 = none). The available-memory guard
    // (q20::auto_store_size) applies on every chip: the budget assumes the Mac to itself.
    const char* cv = getenv("NI_EXPERT_CAP");
    q20::AutoStoreIn in;
    in.all = all, in.least = size_t(3) * E * lay.bytes, in.left = left, in.used = used, in.keep = keep;
    in.ram = size_t(ram), in.avail = q20::available_memory();
    in.cap = Gpu::get().apple10() ? 0.0 : 0.40;
    if (cv && *cv) {
      char* end = nullptr;
      const double x = strtod(cv, &end);
      if (end != cv && !*end && x >= 0 && x <= 1) in.cap = x;
      else logi("NI_EXPERT_CAP=%s is not a share of the RAM between 0 and 1, ignored: using %.2f", cv, in.cap);
    }
    const q20::AutoStore st = q20::auto_store_size(in);
    o.bytes = st.bytes;
    const double pct = ram ? 100.0 * double(used + st.bytes) / double(ram) : 0;
    switch (st.why) {
      case q20::AutoStoreWhy::All:
        logi("expert store: all %.1f GB of experts fit the %.1f GB the budget leaves: resident, no SSD reads",
             all / 1e9, left / 1e9);
        break;
      case q20::AutoStoreWhy::Budget:
        logi("expert store: %.1f GB of the %.1f GB of experts, what the budget leaves (%s; the process's GPU memory "
             "%.0f%% of the RAM)",
             st.bytes / 1e9, all / 1e9, in.cap > 0 ? "within the RAM-share cap" : "no RAM-share cap on this chip", pct);
        break;
      case q20::AutoStoreWhy::Avail:
        logi("expert store: %.1f GB rather than the %.1f GB the budget leaves (the system's available memory %.1f GB "
             "less %.1f GB headroom: other processes hold the rest)",
             st.bytes / 1e9, std::min(left, all) / 1e9, in.avail / 1e9, st.headroom / 1e9);
        break;
      case q20::AutoStoreWhy::Cap:
        logi("expert store: %.1f GB rather than the %.1f GB the budget leaves (the process's GPU memory within %.0f%% "
             "of the RAM: beyond ~45%%, every command buffer of a streamed store took seconds longer on a 48 GB M4 "
             "Pro; NI_EXPERT_CAP=0 lifts it)",
             st.bytes / 1e9, std::min(left, all) / 1e9, in.cap * 100);
        break;
    }
  }
  o.bytes = std::min(o.bytes, all);
  if (const char* r = getenv("NI_EXPERT_READERS"); r && *r) o.cache.readers = atoi(r);
  if (const char* io = getenv("NI_EXPERT_IO"); io && !strcmp(io, "direct")) o.cache.direct = true;
  if (const char* fl = getenv("NI_EXPERT_FILL"); fl && !strcmp(fl, "0")) o.cache.fill = false;
  store_ = std::make_unique<q20::ExpertStore>(files, src, E, lay, o);
  Gpu::get().make_resident(store_->buffers());
  if (!store_->complete()) {
    Gpu& g = Gpu::get();
    plog_ = g.alloc(size_t(kMaxVerify) * E * 4, "pred_logits");
    pidx_ = g.alloc(size_t(kMaxVerify) * cfg_.top_k * 4, "pred_idx");
    pwts_ = g.alloc(size_t(kMaxVerify) * (cfg_.top_k + 1) * 4, "pred_wts");
  }
}

// a prefill chunk's MoE: a chunk of 64+ rows routes most of the next layer's experts, so every one of them not in
// memory is read during this layer; shorter ones only read what they route
q20::Experts Qwen4Exp::q20_fetch(Encoder& e, const Layer& L, const Buffer& ids, uint32_t n, int T,
                                 const std::function<void()>& between) {
  const int li = layer_index(L);
  if (!store_ || li < 0) return Qwen35::q20_fetch(e, L, ids, n, T, between);
  const int nl = li + 1 < cfg_.layers ? li + 1 : -1;
  const bool all = T >= 64 && nl >= 0;
  store_->begin(e, li, ids, n, nullptr, 0, all ? nl : -1, all, /*scan=*/T > 16);
  if (between) between();
  store_->end(e);
  return store_->experts(li);
}

q20::Experts Qwen4Exp::q20_rows_fetch(Encoder& e, const Layer& L, int R, const Buffer& idx, bool pred) {
  const int li = layer_index(L);
  if (!store_ || li < 0) return q20_experts(L);
  const uint32_t n = uint32_t(R * cfg_.top_k);
  store_->begin(e, li, idx, n, pred ? &pidx_ : nullptr, n, pred ? li + 1 : -1, false, false, /*warm=*/true);
  store_->end(e);
  return store_->experts(li);
}

// ---- block-drafter captures (DFlash2, target_hidden_reduction stream_mean) ----

// Rows [pos0, pos0 + R) of layer `layer`'s output, the mean of their 4 streams with the MoE's output (yb_, weights
// injM_) as the next block's mix will inject it, into the armed hook's slot for that layer (Qwen35::set_capture; rows
// outside its positions are skipped). Reads only, so the target's outputs do not change.
void Qwen4Exp::enc_capture4(Encoder& e, int layer, int pos0, int R) {
  if (!cap_ || layer >= int(cap_->slot.size()) || cap_->slot[size_t(layer)] < 0) return;
  const int lo = std::max(pos0, cap_->pos_base), hi = std::min(pos0 + R, cap_->pos_base + cap_->rows);
  if (lo >= hi) return;
  const uint32_t D = uint32_t(cfg_.hidden), S = uint32_t(c4_.hc), r0 = uint32_t(lo - pos0);
  const struct {
    uint32_t R, D, S, ld, col0, dst_row0, has_y;
  } p{uint32_t(hi - lo), D, S, cap_->ld, uint32_t(cap_->slot[size_t(layer)]) * D, uint32_t(lo - cap_->pos_base), 1u};
  e.use(pipe("hc_capture_mean"));
  e.buf(0, x_, size_t(r0) * S * D * 4); e.buf(1, yb_, size_t(r0) * D * 4); e.buf(2, injM_, size_t(r0) * S * 4);
  e.buf(3, cap_->dst);
  e.val(4, p);
  e.dispatch((D + 255) / 256, p.R, 1, 256);
}

// ---- encoders ----

void Qwen4Exp::enc_expand(Encoder& e, int R) {
  const HcParams p{uint32_t(R), uint32_t(cfg_.hidden), uint32_t(c4_.hc), 0, 0, cfg_.eps};
  e.use(pipe("hc_expand"));
  e.buf(0, x_); e.buf(1, yb_);
  e.val(2, p);
  e.dispatch((p.D + 255) / 256, uint32_t(R), 1, 256);
}

void Qwen4Exp::enc_inject(Encoder& e, int R, const Buffer& X, const Buffer& inj) {
  const HcParams p{uint32_t(R), uint32_t(cfg_.hidden), uint32_t(c4_.hc), 1, 0, cfg_.eps};
  e.use(pipe("hc_inject"));
  e.buf(0, X); e.buf(1, yb_); e.buf(2, inj);
  e.val(3, p);
  e.dispatch((p.D + 255) / 256, uint32_t(R), 1, 256);
}

// The gated residual of R rows of X from row x_row0 (fp32): optional injection of the previous block (yb_ rows,
// weights inj_prev), the per-stream norm, the low-rank mix on the row GEMVs, u = the block input (fp32 rows u_ld
// apart) and the block's injection weights (inj_out).
void Qwen4Exp::enc_gr_rows(Encoder& e, const GatedRes& g, int R, const Buffer& X, size_t x_row0, bool inject,
                           const Buffer& inj_prev, const Buffer& inj_out, const Buffer& u, uint32_t u_ld) {
  const uint32_t D = uint32_t(cfg_.hidden), S = uint32_t(c4_.hc), SD = S * D;
  // one row in two dispatches, hc_down + hc_up (NI_HC_ROW=1; off: 4-layer synthetic decode 640 vs 790 tok/s)
  static const bool fused = env_on("NI_HC_ROW");
  const uint32_t rank = uint32_t(c4_.hc_rank);
  if (fused && R == 1 && (g.down.q4() || g.down.q8()) && (g.up.q4() || g.up.q8()) && rank % 64 == 0 && SD % 2048 == 0) {
    const struct {
      uint32_t D, S, rank, has_y, has_inj;
      float eps;
    } p{D, S, rank, inject ? 1u : 0u, g.inj ? 1u : 0u, cfg_.eps};
    e.use(pipe(g.down.bits == 8 ? "hc_down_b8" : "hc_down_b4"));
    e.buf(0, X, x_row0 * SD * 4); e.buf(1, yb_); e.buf(2, inj_prev); e.buf(3, g.norm->buf);
    e.buf(4, g.down.w->buf); e.buf(5, g.down.s->buf); e.buf(6, g.down.b->buf); e.buf(7, g.inj ? g.inj.w->buf : g.norm->buf);
    e.buf(8, xr2_); e.buf(9, hch_); e.buf(10, hinv_); e.buf(11, inj_out);
    e.val(12, p);
    e.dispatch(rank / 8 + 1, 1, 1, 256);
    e.use(pipe(g.up.bits == 8 ? "hc_up_b8" : "hc_up_b4"));
    e.buf(0, hch_); e.buf(1, xr2_); e.buf(2, hinv_); e.buf(3, g.norm->buf);
    e.buf(4, g.up.w->buf); e.buf(5, g.up.s->buf); e.buf(6, g.up.b->buf); e.buf(7, u); e.buf(8, X, x_row0 * SD * 4);
    e.buf(9, yb_);
    e.val(10, p);
    e.dispatch(D / 8, 1, 1, 256);
    return;
  }
  // the whole mix in two dispatches: hc_pre folded into the split-K GEMVs (NI_HC_FOLD=0: hc_pre + the two below;
  // real pack, M5 Max, p700 decode, 3 interleaved rounds: 74.0 vs 71.2 tok/s, same greedy ids)
  static const bool fold = !(getenv("NI_HC_FOLD") && !strcmp(getenv("NI_HC_FOLD"), "0"));
  const bool sk_fmt = g.down.gs == 64 && (g.down.bits == 4 || g.down.bits == 6 || g.down.bits == 8);
  if (fold && g.down.same_format(g.up) && sk_fmt && SD % 1024 == 0 && rank % 64 == 0 && D % 32 == 0 && S == 4) {
    const struct {
      uint32_t R, D, S, rank, kc, u_ld, has_y, has_inj;
      float eps;
    } fp{uint32_t(R), D, S, rank, SD / 1024, u_ld, inject ? 1u : 0u, g.inj ? 1u : 0u, cfg_.eps};
    const std::string b = "_b" + std::to_string(g.down.bits);
    // hc_down_sk3 / hc_up_mix3 (zqwen4v.metal): the same arithmetic with more work per threadgroup, every row
    // (<= 4) in each threadgroup: 2 weight rows per simdgroup of the down projection, 2 blocks of 16 channels per
    // threadgroup of the up projection; decode logits bitwise equal to hc_down_sk2 / hc_up_mix2's (NI_HC3=0: those)
    static const bool hc3 = !(getenv("NI_HC3") && !strcmp(getenv("NI_HC3"), "0"));
    constexpr int nr = 2, cpt = 2;
    if (hc3 && R <= 4 && rank <= 512 && rank % (8 * nr) == 0 && D % (16 * cpt) == 0) {
      e.use(pipe("hc_down_sk3_n" + std::to_string(nr) + b));
      e.buf(0, X, x_row0 * SD * 4); e.buf(1, yb_); e.buf(2, inj_prev); e.buf(3, g.norm->buf);
      e.buf(4, g.inj ? g.inj.w->buf : g.norm->buf); e.buf(5, g.down.w->buf); e.buf(6, g.down.s->buf);
      e.buf(7, g.down.b->buf); e.buf(8, hpart_); e.buf(9, hx2_); e.buf(10, hstat_);
      e.val(11, fp);
      e.dispatch(rank / (8 * nr), SD / 1024, 1, 256);
      e.use(pipe("hc_up_mix3_c" + std::to_string(cpt) + b));
      e.buf(0, hpart_); e.buf(1, hx2_); e.buf(2, hstat_); e.buf(3, g.norm->buf);
      e.buf(4, g.up.w->buf); e.buf(5, g.up.s->buf); e.buf(6, g.up.b->buf); e.buf(7, u);
      e.buf(8, X, x_row0 * SD * 4); e.buf(9, yb_); e.buf(10, inj_out);
      e.val(11, fp);
      e.dispatch(D / (16 * cpt), 1, 1, 256);
      return;
    }
    e.use(pipe("hc_down_sk2" + b));
    e.buf(0, X, x_row0 * SD * 4); e.buf(1, yb_); e.buf(2, inj_prev); e.buf(3, g.norm->buf);
    e.buf(4, g.inj ? g.inj.w->buf : g.norm->buf); e.buf(5, g.down.w->buf); e.buf(6, g.down.s->buf);
    e.buf(7, g.down.b->buf); e.buf(8, hpart_); e.buf(9, hx2_); e.buf(10, hstat_);
    e.val(11, fp);
    e.dispatch(rank / 8, SD / 1024, uint32_t(R), 256);
    e.use(pipe("hc_up_mix2" + b));
    e.buf(0, hpart_); e.buf(1, hx2_); e.buf(2, hstat_); e.buf(3, g.norm->buf);
    e.buf(4, g.up.w->buf); e.buf(5, g.up.s->buf); e.buf(6, g.up.b->buf); e.buf(7, u);
    e.buf(8, X, x_row0 * SD * 4); e.buf(9, yb_); e.buf(10, inj_out);
    e.val(11, fp);
    e.dispatch(D / 16, uint32_t(R), 1, 256);
    return;
  }
  const HcParams p{uint32_t(R), D, S, inject ? 1u : 0u, g.inj ? 1u : 0u, cfg_.eps};
  e.use(pipe("hc_pre"));
  e.buf(0, X, x_row0 * SD * 4); e.buf(1, yb_); e.buf(2, inj_prev); e.buf(3, g.norm->buf);
  e.buf(4, g.inj ? g.inj.w->buf : g.norm->buf); e.buf(5, xn4_); e.buf(6, inj_out);
  e.val(7, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
  // the low-rank mix on two parallel GEMVs (hc_down_sk: split over the inputs; hc_up_mix: up projection, sigmoid and
  // the streams' mean per channel); NI_HC_SK=0: the qmv GEMVs below and hc_mix
  static const bool sk = !(getenv("NI_HC_SK") && !strcmp(getenv("NI_HC_SK"), "0"));
  if (sk && g.down.same_format(g.up) && sk_fmt && SD % 1024 == 0 && rank % 64 == 0 && D % 16 == 0) {
    const struct {
      uint32_t R, D, S, rank, kc, u_ld;
    } mp{uint32_t(R), D, S, rank, SD / 1024, u_ld};
    const std::string b = "_b" + std::to_string(g.down.bits);
    e.use(pipe("hc_down_sk" + b));
    e.buf(0, xn4_); e.buf(1, g.down.w->buf); e.buf(2, g.down.s->buf); e.buf(3, g.down.b->buf); e.buf(4, hpart_);
    e.val(5, mp);
    e.dispatch(rank / 8, SD / 1024, uint32_t(R), 256);
    e.use(pipe("hc_up_mix" + b));
    e.buf(0, hpart_); e.buf(1, xn4_); e.buf(2, g.up.w->buf); e.buf(3, g.up.s->buf); e.buf(4, g.up.b->buf); e.buf(5, u);
    e.val(6, mp);
    e.dispatch(D / 16, uint32_t(R), 1, 256);
    return;
  }
  if (R == 1) enc_qmv(e, g.down, xn4_, hch_, nullptr, nullptr);
  else enc_qmvr(e, g.down, xn4_, hch_, R, false);
  const uint32_t n = uint32_t(R) * uint32_t(c4_.hc_rank);
  const float inv_s = 1.0f / float(S);
  e.use(pipe("hc_act"));
  e.buf(0, hch_); e.val(1, n); e.val(2, inv_s);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  if (R == 1) enc_qmv(e, g.up, hch_, hcm_, nullptr, nullptr);
  else enc_qmvr(e, g.up, hch_, hcm_, R, false);
  e.use(pipe("hc_mix"));
  e.buf(0, hcm_); e.buf(1, xn4_); e.buf(2, u);
  e.val(3, p); e.val(4, u_ld);
  e.dispatch((D + 255) / 256, uint32_t(R), 1, 256);
}

// The gated residual of T prefill rows of X on the GEMMs: u = the block input as half rows in xh_.
void Qwen4Exp::enc_gr_prefill(Encoder& e, const GatedRes& g, int T, const Buffer& X, bool inject,
                              const Buffer& inj_prev, const Buffer& inj_out) {
  const uint32_t D = uint32_t(cfg_.hidden), S = uint32_t(c4_.hc);
  const HcParams p{uint32_t(T), D, S, inject ? 1u : 0u, g.inj ? 1u : 0u, cfg_.eps};
  e.use(pipe("hc_pre_h"));
  e.buf(0, X); e.buf(1, yb_); e.buf(2, inj_prev); e.buf(3, g.norm->buf);
  e.buf(4, g.inj ? g.inj.w->buf : g.norm->buf); e.buf(5, xn4h_); e.buf(6, inj_out);
  e.val(7, p);
  e.dispatch(uint32_t(T), 1, 1, 1024);
  enc_qmm(e, g.down, xn4h_, hch_, T, false);
  const uint32_t n = uint32_t(T) * uint32_t(c4_.hc_rank);
  const float inv_s = 1.0f / float(S);
  e.use(pipe("hc_act_h"));
  e.buf(0, hch_); e.buf(1, hchh_); e.val(2, n); e.val(3, inv_s);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  enc_qmm(e, g.up, hchh_, hcm_, T, false);
  const uint32_t ld = D;
  e.use(pipe("hc_mix_h"));
  e.buf(0, hcm_); e.buf(1, xn4h_); e.buf(2, xh_);
  e.val(3, p); e.val(4, ld);
  e.dispatch((D + 255) / 256, uint32_t(T), 1, 256);
}

void Qwen4Exp::enc_gdn4(Encoder& e, Seq& s, int li, const Layer& L, const Buffer& u) {
  enc_qmv_multi(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, u, nullptr);
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  kp::GdnPrep pp{1, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C};
  e.use(pipe("gdn_prep"));
  e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[size_t(li)]);
  e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
  e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
  e.val(12, pp);
  e.dispatch(C / 128, 1, 1, 128);
  kp::GdnStep sp{1, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), cfg_.eps};
  e.use(pipe(gdn_sigmoid_ ? "gdn_step_sg" : "gdn_step"));
  e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
  e.buf(5, s.rec[size_t(li)]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
  e.val(9, sp);
  e.dispatch(uint32_t(cfg_.lin_hv), 1, 1, 1024);
  enc_qmv(e, L.out, gout_, yb_, nullptr, nullptr);
}

void Qwen4Exp::enc_qsa(Encoder& e, const Seq& s, const Qwen4Layer& F, const Buffer& iqk, uint32_t ld, int R, int pos0,
                       const Buffer& ik, const Buffer& bk) {
  const QsaParams p{uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), uint32_t(cfg_.rot_dims), uint32_t(pos0),
                    uint32_t(c4_.ix_ratio), ld, cfg_.eps, cfg_.theta};
  // images: the rows' M-RoPE positions (as attn_prep_m's); text: mn = delta = 0
  const struct {
    uint32_t mn;
    int32_t delta;
    uint32_t sec1, sec2;
  } m{uint32_t(s.rope_n), s.rope_delta, uint32_t(3 * cfg_.mrope[1]), uint32_t(3 * cfg_.mrope[2])};
  const Buffer& mpos = s.rope_pos ? s.rope_pos : iq_;
  e.use(pipe("qsa_prep"));
  e.buf(0, iqk); e.buf(1, F.iqn->buf); e.buf(2, iq_); e.buf(3, ik);
  e.val(4, p);
  e.buf(5, mpos); e.val(6, m);
  e.dispatch(p.NH + 1, uint32_t(R), 1, p.HD);
  e.use(pipe("qsa_block"));
  e.buf(0, ik); e.buf(1, F.ikn->buf); e.buf(2, bk);
  e.val(3, p);
  e.buf(4, mpos); e.val(5, m);
  e.dispatch(uint32_t(R), 1, 1, p.HD);
}

namespace {
struct QsaSelParams {
  uint32_t R, NH, HD, pos0, ratio, topb, s_ld, i_ld;
};
struct AttnSparseParams {
  uint32_t Hq, Hkv, chunk, max_seq;
  float scale;
  uint32_t nchunks, i_ld;
};
}  // namespace

// A scratch buffer grown while a prompt is encoded: the command buffers already committed for it (the earlier layers
// of this chunk, the chunk before) may still read the old one, and they hold no reference to it, so it is released
// only once the command buffer being encoded, the last of them, has completed.
void Qwen4Exp::regrow(Encoder& e, Buffer& b, size_t bytes, const char* label) {
  if (b) {
    Buffer old = b;
    e.on_completed([old]() mutable { Gpu::get().release(old); });
  }
  b = Gpu::get().alloc(bytes, label);
}

void Qwen4Exp::enc_qsa_select(Encoder& e, const Buffer& bk, int R, int pos0, int iq_row0) {
  const uint32_t nb = uint32_t(pos0 + R) / uint32_t(c4_.ix_ratio);  // blocks of the last row
  const size_t need = size_t(R) * nb * 4;
  if (qscore_.size < need) regrow(e, qscore_, std::max(need, size_t(4) << 20), "qsa_scores");
  const QsaSelParams p{uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), uint32_t(pos0), uint32_t(c4_.ix_ratio),
                       uint32_t(c4_.ix_budget / c4_.ix_ratio), nb, qidx_ld_};
  e.use(pipe("qsa_score"));
  e.buf(0, iq_, size_t(iq_row0) * c4_.ix_heads * c4_.ix_dim * 4); e.buf(1, bk); e.buf(2, qscore_);
  e.val(3, p);
  e.dispatch((nb + 255) / 256, uint32_t(R), 1, 256);
  e.use(pipe("qsa_select"));
  e.buf(0, qscore_); e.buf(1, qidx_); e.buf(2, qcnt_);
  e.val(3, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void Qwen4Exp::enc_qsa_mask(Encoder& e, const Buffer& bk, int R, int pos0, int row0, uint32_t mwords) {
  const uint32_t nb = uint32_t(pos0 + R) / uint32_t(c4_.ix_ratio);
  const size_t need = size_t(R) * nb * 4, mneed = size_t(row0 + R) * mwords * 4;
  if (qscore_.size < need) regrow(e, qscore_, std::max(need, size_t(4) << 20), "qsa_scores");
  if (qmask_.size < mneed) regrow(e, qmask_, std::max(size_t(tmax_) * mwords * 4, mneed), "qsa_mask");
  const QsaSelParams p{uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), uint32_t(pos0), uint32_t(c4_.ix_ratio),
                       uint32_t(c4_.ix_budget / c4_.ix_ratio), nb, mwords};
  e.use(pipe("qsa_score"));
  e.buf(0, iq_, size_t(row0) * c4_.ix_heads * c4_.ix_dim * 4); e.buf(1, bk); e.buf(2, qscore_);
  e.val(3, p);
  e.dispatch((nb + 255) / 256, uint32_t(R), 1, 256);
  e.use(pipe("qsa_select_mask"));
  e.buf(0, qscore_); e.buf(1, qmask_, size_t(row0) * mwords * 4); e.buf(2, qcnt_);
  e.val(3, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void Qwen4Exp::enc_attn_sparse(Encoder& e, const Buffer& kc, const Buffer& vc, int cap, int R, bool partials, int q_row0) {
  const uint32_t G = uint32_t(cfg_.hq / cfg_.hkv), maxn = qidx_ld_;
  const float scale = 1.0f / std::sqrt(float(cfg_.head_dim));
  const size_t qoff = size_t(q_row0) * cfg_.hq * cfg_.head_dim * 4;
  if (!partials) {
    const AttnSparseParams p{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), maxn, uint32_t(cap), scale, 1, qidx_ld_};
    e.use(pipe("attn_sparse_rows"));
    e.buf(0, qo_, qoff); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_, qoff * 2); e.buf(4, attn_out_, qoff);
    e.val(5, p);
    e.buf(6, qidx_); e.buf(7, qcnt_);
    e.dispatch(uint32_t(cfg_.hkv), uint32_t(R), 1, 32 * G);
    return;
  }
  // split-K over the lists (decode / verify rows): chunks of >= 64 entries, about one per GPU core and kv head
  const uint32_t cores = uint32_t(std::max(1, Gpu::get().core_count()));
  uint32_t chunk = std::max<uint32_t>(64, (maxn + cores - 1) / cores);
  chunk = (chunk + 15) / 16 * 16;
  uint32_t nch = (maxn + chunk - 1) / chunk;
  while (nch * uint32_t(R) > 1024) {
    chunk *= 2;
    nch = (maxn + chunk - 1) / chunk;
  }
  const AttnSparseParams p{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), chunk, uint32_t(cap), scale, nch, qidx_ld_};
  e.use(pipe("attn_rows_sparse"));
  e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
  e.val(5, p);
  e.buf(6, qidx_); e.buf(7, qcnt_);
  e.dispatch(nch, uint32_t(cfg_.hkv), uint32_t(R), 32 * G);
  kp::AttnRows rp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), 0, chunk, uint32_t(cap), scale, nch};
  e.use(pipe("attn_rows_reduce"));
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, rp);
  e.dispatch(cfg_.hq, uint32_t(R), 1, cfg_.head_dim);
}

void Qwen4Exp::enc_attn4(Encoder& e, Seq& s, int li, const Layer& L, const Qwen4Layer& F, int pos, const Buffer& u) {
  enc_qmv_multi(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}, {&F.iqk, iqk_}}, u, nullptr);
  kp::AttnPrep ap{1, uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(pipe(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.kc[size_t(li)]); e.buf(7, s.vc[size_t(li)]);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, 1, 1, cfg_.head_dim);
  enc_qsa(e, s, F, iqk_, F.iqk.N, 1, pos, s.ixk[size_t(li)], s.ixb[size_t(li)]);
  if (qsa_sparse(pos)) {  // the selected blocks only
    enc_qsa_select(e, s.ixb[size_t(li)], 1, pos, 0);
    enc_attn_sparse(e, s.kc[size_t(li)], s.vc[size_t(li)], s.cap, 1, true, 0);
    enc_qmv(e, L.o, attn_out_, yb_, nullptr, nullptr);
    return;
  }
  // dense attention over [0, pos] (exact while pos + 1 <= budget + ratio - 1)
  const uint32_t Lk = uint32_t(pos + 1);
  const uint32_t cores = uint32_t(std::max(1, Gpu::get().core_count()));
  uint32_t chunk = std::max<uint32_t>(64, (Lk + cores - 1) / cores);
  chunk = (chunk + 15) / 16 * 16;
  const uint32_t nch = (Lk + chunk - 1) / chunk;
  kp::AttnDec dp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), Lk, chunk, uint32_t(s.cap), 1.0f / std::sqrt(float(cfg_.head_dim)), nch};
  e.use(pipe("attn_decode_partial"));
  e.buf(0, qo_); e.buf(1, s.kc[size_t(li)]); e.buf(2, s.vc[size_t(li)]); e.buf(3, part_o_); e.buf(4, part_ml_);
  e.val(5, dp);
  e.dispatch(nch, cfg_.hkv, 1, 32 * (cfg_.hq / cfg_.hkv));
  e.use(pipe("attn_decode_reduce"));
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, dp);
  e.dispatch(cfg_.hq, 1, 1, cfg_.head_dim);
  enc_qmv(e, L.o, attn_out_, yb_, nullptr, nullptr);
}

// MoE of one row (u = the block input, fp32): router + shared gate, top-k, the fused 4-bit expert kernels (Q2_0
// experts: q20::enc_decode); other formats take the per-row MoE (the verify path at one row; u must then be in xrn_).
void Qwen4Exp::enc_moe4(Encoder& e, const Layer& L, const Buffer& u) {
  const bool q20 = L.eg.q20;  // check_formats: all three Q2_0 and a 4-bit shared expert
  if (!q20 && !(L.eg.q4() && L.eu.q4() && L.ed.q4() && L.sg.q4() && L.su.q4() && L.sd.q4())) {
    if (u.mtl != xrn_.mtl) fatal("enc_moe4: the input must be xrn_");
    enc_moe_rows(e, L, 1, yb_, false, &xrn_);
    return;
  }
  constexpr uint32_t nsg = 8, gu_rows = 2, rows = 2;
  // streamed experts: the next layer's router on this input too (its top-k is read ahead during this layer)
  const int li = layer_index(L);
  const bool pred = q20 && store_ && !store_->complete() && li >= 0 && li + 1 < cfg_.layers;
  std::vector<Seg> segs = {{&L.router, rlog_}, {&L.shared_gate, slog_}};
  if (pred) segs.push_back({&layers_[size_t(li + 1)].router, plog_});
  enc_qmv_multi(e, segs, u, nullptr);
  const uint32_t E = uint32_t(cfg_.experts);
  kp::Moe mp{uint32_t(cfg_.hidden), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(cfg_.top_k), cfg_.eps};
  kp::Topk tp{E, uint32_t(cfg_.top_k), cfg_.norm_topk ? 1u : 0u};
  // 512 experts: moe_topk512 (selects on the logits; NI_TOPK512=0: moe_topk)
  static const bool t512 = !(getenv("NI_TOPK512") && !strcmp(getenv("NI_TOPK512"), "0"));
  const Pipeline topk = pipe(t512 && E == 512 && cfg_.top_k <= 16 ? "moe_topk512" : E == 256 && cfg_.top_k <= 8 ? "moe_topk256" : "moe_topk");
  e.use(topk);
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, eidx_); e.buf(3, ewts_);
  e.val(4, tp);
  e.dispatch(1, 1, 1, 32);
  if (pred) {
    e.use(topk);
    e.buf(0, plog_); e.buf(1, slog_); e.buf(2, pidx_); e.buf(3, pwts_);
    e.val(4, tp);
    e.dispatch(1, 1, 1, 32);
  }
  if (q20) {
    q20::enc_decode(e, q20_rows_fetch(e, L, 1, eidx_, pred), q20_shared(L), q20_shape(), u, eidx_, ewts_, eh_, yb_);
    return;
  }
  e.use(pipe("moe_gate_up3_r2"));
  e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
  e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
  e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
  e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
  e.buf(12, u); e.buf(13, rlog_); e.buf(14, slog_); e.buf(15, eh_);
  e.val(16, mp);
  e.buf(17, eidx_); e.buf(18, ewts_); e.val(19, E);
  const uint32_t I = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
  e.dispatch((I + gu_rows * nsg - 1) / (gu_rows * nsg), cfg_.top_k + 1, 1, 32 * nsg);
  e.use(pipe(moe_wide() ? "moe_down2_w" : "moe_down2"));
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, eidx_); e.buf(8, ewts_); e.buf(9, yb_);
  e.val(10, mp);
  e.dispatch((cfg_.hidden + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

// The n-gram block of R rows (decode / verify): key / value projections of the staged rows, then ple_rows
void Qwen4Exp::enc_ple_rows(Encoder& e, Seq& s, int R, int pos0, const Buffer& rows) {
  if (R == 1) {
    enc_qmv_multi(e, {{&ple_key_, ple_k_}, {&ple_val_, ple_v_}}, rows, nullptr);
  } else {
    enc_rows(e, {{&ple_key_, ple_k_}, {&ple_val_, ple_v_}}, rows, R);
  }
  const PleParams p{uint32_t(R), uint32_t(cfg_.hidden), uint32_t(c4_.hc), uint32_t(pos0), cfg_.eps, kRing,
                    uint32_t(c4_.ngram)};
  e.use(pipe("ple_rows"));
  e.buf(0, ple_k_); e.buf(1, ple_v_); e.buf(2, x_); e.buf(3, ple_nk_->buf); e.buf(4, ple_nq_->buf);
  e.buf(5, ple_nc_->buf); e.buf(6, ple_conv_->buf); e.buf(7, s.pconv);
  e.val(8, p);
  e.dispatch(p.S, 1, 1, p.D / 4);
}

void Qwen4Exp::enc_ple_prefill(Encoder& e, Seq& s, int T, int pos0, const Buffer& rows, Checkpoint* snap, int snap_row) {
  enc_proj(e, {{&ple_key_, ple_k_, false}, {&ple_val_, ple_v_, false}}, rows, T);
  const PleParams p{uint32_t(T), uint32_t(cfg_.hidden), uint32_t(c4_.hc), uint32_t(pos0), cfg_.eps, kRing,
                    uint32_t(c4_.ngram)};
  const uint32_t SD = p.S * p.D;
  // gv into hcm_ (free until the next mix), gvn over the key rows (each threadgroup rewrites only what it read)
  e.use(pipe("ple_gate_rows"));
  e.buf(0, ple_k_); e.buf(1, ple_v_); e.buf(2, x_); e.buf(3, ple_nk_->buf); e.buf(4, ple_nq_->buf);
  e.buf(5, ple_nc_->buf); e.buf(6, hcm_); e.buf(7, ple_k_);
  e.val(8, p);
  e.dispatch(uint32_t(T), p.S, 1, p.D / 4);
  e.use(pipe("ple_conv_rows"));
  e.buf(0, hcm_); e.buf(1, ple_k_); e.buf(2, x_); e.buf(3, ple_conv_->buf); e.buf(4, s.pconv);
  e.val(5, p);
  e.dispatch((SD + 255) / 256, uint32_t(T), 1, 256);
  if (snap) {  // the window as it stands after snap_row rows: before the live one moves on
    const uint32_t end = uint32_t(pos0 + snap_row);
    e.use(pipe("ple_ring_rows"));
    e.buf(0, ple_k_); e.buf(1, s.pconv); e.buf(2, snap->pconv);
    e.val(3, p); e.val(4, end);
    e.dispatch((SD + 255) / 256, kRing, 1, 256);
  }
  const uint32_t end = uint32_t(pos0 + T);
  e.use(pipe("ple_ring_rows"));
  e.buf(0, ple_k_); e.buf(1, s.pconv); e.buf(2, s.pconv);
  e.val(3, p); e.val(4, end);
  e.dispatch((SD + 255) / 256, kRing, 1, 256);
}

void Qwen4Exp::enc_attn4_prefill(Encoder& e, Seq& s, int li, const Layer& L, const Qwen4Layer& F, int pos0, int T) {
  enc_proj(e, {{&L.q, qg_, false}, {&L.k, kr_, false}, {&L.v, vr_, false}, {&F.iqk, iqk_, false}}, xh_, T);
  kp::AttnPrep ap{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos0), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(pipe(mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.kc[size_t(li)]); e.buf(7, s.vc[size_t(li)]);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, uint32_t(T), 1, cfg_.head_dim);
  enc_qsa(e, s, F, iqk_, F.iqk.N, T, pos0, s.ixk[size_t(li)], s.ixb[size_t(li)]);
  if (qsa_sparse(pos0 + T - 1)) {
    // rows in passes whose score scratch stays within ~64 MB (every row of a 2048-row chunk up to a 32k context)
    const int nbmax = (pos0 + T) / c4_.ix_ratio;
    const int per = std::max(1, std::min(T, int((size_t(64) << 20) / (size_t(nbmax) * 4))));
    // Up to 8192 keys: bit rows and the prefill flash attention skipping unselected keys and tiles (attn_prefill11m:
    // its cost grows with the union of a 32-query block's selections, at worst the context); past that, per-row key
    // lists and a gather kernel without MMA (attn_sparse_rows: ~budget keys per row whatever the context). Synthetic
    // 4-layer pack, a 2048-row chunk: 11.7 vs 24.8 ms at 4k keys, 35 vs 24 ms at 16k. NI_QSA_PF=mask / list forces one.
    static const char* pf = getenv("NI_QSA_PF");
    const bool lists = pf && *pf ? !strcmp(pf, "list") : pos0 + T > 8192;
    if (lists) {
      for (int r0 = 0; r0 < T; r0 += per) {
        const int n = std::min(per, T - r0);
        enc_qsa_select(e, s.ixb[size_t(li)], n, pos0 + r0, r0);
        enc_attn_sparse(e, s.kc[size_t(li)], s.vc[size_t(li)], s.cap, n, false, r0);
      }
    } else {
      const uint32_t mwords = uint32_t(pos0 + T + 31) / 32;
      for (int r0 = 0; r0 < T; r0 += per) enc_qsa_mask(e, s.ixb[size_t(li)], std::min(per, T - r0), pos0 + r0, r0, mwords);
      kp::AttnPref fp{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos0), uint32_t(s.cap),
                      1.0f / std::sqrt(float(cfg_.head_dim))};
      e.use(pipe("attn_prefill11m"));
      e.buf(0, qo_); e.buf(1, s.kc[size_t(li)]); e.buf(2, s.vc[size_t(li)]); e.buf(3, qg_); e.buf(4, attn_out_);
      e.val(5, fp);
      e.buf(6, qmask_); e.val(7, mwords);
      e.dispatch(uint32_t(T + 31) / 32, uint32_t(cfg_.hq), 1, 256);
    }
  } else {
    enc_attn_prefill(e, s.kc[size_t(li)], s.vc[size_t(li)], T, pos0, s.cap);
  }
  enc_rows_half(e, attn_out_, nullptr, hh_, T, cfg_.hq * cfg_.head_dim);
  enc_qmm(e, L.o, hh_, yb_, T, false);
}

// ---- passes ----

void Qwen4Exp::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  const int D = cfg_.hidden;
  const PleWait pw = c4_.ple_layer >= 0 ? ple_rows_for(s, pos) : PleWait{};
  enc_embed(e, s.tok, pos, 1, yb_);
  enc_expand(e, 1);
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[size_t(i)];
    const Qwen4Layer& F = l4_[size_t(i)];
    if (i == c4_.ple_layer) {
      if (i > 0) enc_inject(e, 1, x_, injM_);
      if (pw.ticket) e.wait_event(ple_ev_, pw.ticket);
      enc_ple_rows(e, s, 1, pos, pw.rows);
    }
    enc_gr_rows(e, F.ga, 1, x_, 0, i > 0 && i != c4_.ple_layer, injM_, injA_, xrn_, uint32_t(D));
    if (L.linear) enc_gdn4(e, s, i, L, xrn_);
    else enc_attn4(e, s, i, L, F, pos, xrn_);
    enc_gr_rows(e, F.gm, 1, x_, 0, true, injA_, injM_, xrn_, uint32_t(D));
    enc_moe4(e, L, xrn_);
    enc_capture4(e, i, pos, 1);
  }
  // the final mix (its injection leaves the pre-mix state in x_: the MTP head's input)
  enc_gr_rows(e, mix_, 1, x_, 0, true, injM_, injA_, xrn_, uint32_t(D));
  if (!sample) return;
  enc_sample(e, s, xrn_, pos + 1, sp, step);
  ple_after_sample(e, s, pos + 1);
}

void Qwen4Exp::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                              Checkpoint* snap, int snap_row) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  if (snap && (!can_snapshot() || snap_row <= 0 || snap_row >= T)) fatal("prefill snapshot at row %d of %d", snap_row, T);
  const int D = cfg_.hidden;
  // the chunk's n-gram rows (the tokens are known): gathered while the GPU runs the previous chunk
  Buffer rows;
  if (c4_.ple_layer >= 0) {
    ple_pf_flip_ ^= 1;
    rows = ple_pf_[ple_pf_flip_];
    gather_rows(s.tokens(), pos0, T, rows.ptr(), true);
  }
  enc_embed(e, s.tok, pos0, T, yb_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0, T, yb_, D);  // image rows
  enc_expand(e, T);
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[size_t(i)];
    const Qwen4Layer& F = l4_[size_t(i)];
    if (i == dbg_layers_) break;
    if (i == c4_.ple_layer) {
      if (i > 0) enc_inject(e, T, x_, injM_);
      enc_ple_prefill(e, s, T, pos0, rows, snap, snap_row);
    }
    enc_gr_prefill(e, F.ga, T, x_, i > 0 && i != c4_.ple_layer, injM_, injA_);
    if (L.linear) {
      enc_proj(e, {{&L.qkv, qkv_, false}, {&L.z, zb_, false}, {&L.b, bb_, false}, {&L.a, ab_, false}}, xh_, T);
      enc_gdn_prefill_par(e, s, i, L, T, snap, snap_row);
      enc_qmm(e, L.out, hh_, yb_, T, false);
    } else {
      enc_attn4_prefill(e, s, i, L, F, pos0, T);
    }
    enc_gr_prefill(e, F.gm, T, x_, true, injA_, injM_);
    enc_moe_prefill(e, L, T, yb_, /*xh_ready=*/true);
    enc_capture4(e, i, pos0, T);
    // a command buffer per layer of a long chunk (~1/48 of it), so the desktop's own GPU work is not held up
    if (T >= 256 && i + 1 < cfg_.layers) e.split();
  }
  if (dbg_layers_ != 0) enc_inject(e, T, x_, injM_);  // the pre-mix state of every row (the MTP head's history, logits)
  if (sample) {
    enc_gr_rows(e, mix_, 1, x_, size_t(T - 1), false, injM_, injA_, xrn_, uint32_t(D));
    enc_sample(e, s, xrn_, pos0 + T, sp, step);
    ple_after_sample(e, s, pos0 + T);
  }
  if (snap) snap->pos = pos0 + snap_row;
}

void Qwen4Exp::encode_logits_rows(Encoder& e, int T, const Buffer& out) {
  enc_gr_prefill(e, mix_, T, x_, false, injM_, injA_);
  enc_qmm(e, lm_head_, xh_, out, T, false);
}

std::string Qwen4Exp::formats_summary() const {
  std::map<std::string, size_t> bytes;
  auto add = [&](const QLinear& q) {
    if (q && q.q20)  // (streamed: deferred tensors, no buffers; their size as stored)
      bytes["Q2_0"] += size_t(q.w->numel()) * dtype_size(q.w->dtype) + size_t(q.s->numel()) * dtype_size(q.s->dtype);
    else if (q) bytes[format_name(q.bits, q.gs)] += q.w->buf.size + (q.dense() ? 0 : q.s->buf.size + q.b->buf.size);
  };
  add(embed_);
  add(lm_head_);
  auto add_gr = [&](const GatedRes& g) {
    add(g.down);
    add(g.up);
    add(g.inj);
  };
  for (size_t i = 0; i < layers_.size(); ++i) {
    const Layer& L = layers_[i];
    for (const QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out, &L.q, &L.k, &L.v, &L.o, &L.router, &L.shared_gate, &L.eg,
                             &L.eu, &L.ed, &L.sg, &L.su, &L.sd})
      add(*q);
    add(l4_[i].iqk);
    add_gr(l4_[i].ga);
    add_gr(l4_[i].gm);
  }
  add_gr(mix_);
  add(ple_key_);
  add(ple_val_);
  size_t tot = 0;
  for (auto& [k, v] : bytes) tot += v;
  std::string r;
  for (auto& [k, v] : bytes) {
    char b[96];
    snprintf(b, sizeof b, "%s%s %.0f%%", r.empty() ? "" : ", ", k.c_str(), 100.0 * double(v) / double(std::max<size_t>(tot, 1)));
    r += b;
  }
  return r;
}

// ---- memory plan (headers only) ----

MemPlan Qwen4Exp::plan(const std::string& dir, int chunk) {
  MemPlan m;
  const json root = read_json(dir + "/config.json");
  Qwen35Config c = read_config(dir);
  Qwen4ExpConfig c4;
  parse4(root, c4);
  // tensors as stored: the trunk (vision tower and the n-gram table left out), the MTP head (mtp.*)
  std::vector<std::string> files;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0 && n != "ple.safetensors") files.push_back(n);
    }
    closedir(d);
  }
  const double share = c.experts ? double(c.top_k) / c.experts : 1.0;
  // Q2_0 routed experts are streamed (Qwen4Exp::init_store; NI_EXPERT_STORE=off keeps them resident): counted apart,
  // the weights holding the smallest store (3 layers' experts) in their place
  const char* es = getenv("NI_EXPERT_STORE");
  const bool stream = q20_experts_in(root) && !(es && (!strcmp(es, "off") || !strcmp(es, "0")));
  size_t lm_head = 0, table = 0;
  for (const std::string& f : files) {
    std::ifstream in(dir + "/" + f, std::ios::binary);
    uint64_t n = 0;
    if (!in.read(reinterpret_cast<char*>(&n), 8) || n > (256u << 20)) continue;
    std::string h(n, '\0');
    if (!in.read(h.data(), std::streamsize(n))) continue;
    json j;
    try {
      j = json::parse(h);
    } catch (...) {
      continue;
    }
    for (auto& [name, v] : j.items()) {
      if (name == "__metadata__") continue;
      if (name.find("vision_tower") != std::string::npos || name.find("visual.") != std::string::npos) continue;
      const size_t b = v["data_offsets"][1].get<size_t>() - v["data_offsets"][0].get<size_t>();
      if (name.find(".ngram_embedding.") != std::string::npos) {
        table += b;
        continue;
      }
      const size_t al = (b + 255) / 256 * 256;
      if (name.rfind("mtp.", 0) == 0) {
        if (!getenv("NI_NO_MTP")) m.mtp += al;
        continue;
      }
      if (stream && is_trunk_expert(name)) m.streamed += b;
      else m.weights += al;
      if (name.find("lm_head.") != std::string::npos) lm_head += b;
      if (name.find("embed_tokens") == std::string::npos)
        m.decode_bytes += name.find("switch_mlp") != std::string::npos ? size_t(double(b) * share) : b;
    }
  }
  (void)table;
  if (m.streamed)
    m.weights += size_t(3) * c.experts * q20::slot_layout(uint32_t(c.hidden), uint32_t(c.moe_I)).bytes;
  m.dense = false;
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = 0;
  m.head_row_bytes = lm_head / size_t(std::max(c.vocab, 1));
  {
    const json q = root.contains("quantization") ? root["quantization"] : json::object();
    if (q.empty()) {
      m.formats = "bf16";
    } else {
      m.formats = format_name(q.value("bits", 4), q.value("group_size", 64));
      std::map<std::string, int> other;
      for (auto& [k, v] : q.items())
        if (v.is_object() && k.find(".ngram_embedding.") == std::string::npos &&
            (v.value("bits", 4) != q.value("bits", 4) || v.value("group_size", 64) != q.value("group_size", 64)))
          ++other[v.value("mode", "") == "q2_0" ? "Q2_0" : format_name(v.value("bits", 4), v.value("group_size", 64))];
      for (auto& [f, n] : other) m.formats += " + " + std::to_string(n) + " modules " + f;
    }
  }
  {
    const double H = c.hidden, SD = double(c4.hc) * c.hidden, kd = double(c.lin_hk) * c.lin_dk,
                 vd = double(c.lin_hv) * c.lin_dv;
    const double qd = double(c.hq) * c.head_dim, kvd = double(c.hkv) * c.head_dim, gr = 2 * SD * c4.hc_rank + c4.hc * SD;
    double p = 0;
    for (int i = 0; i < c.layers; ++i) {
      p += c.is_linear[size_t(i)] ? H * (2 * kd + vd) + H * vd + 2 * H * c.lin_hv + vd * H
                                  : H * 2 * qd + 2 * H * kvd + qd * H + H * (c4.ix_heads + 1) * c4.ix_dim;
      p += double(c.top_k) * 3 * H * c.moe_I + 3 * H * c.shared_I + H * c.experts + H + 2 * gr;
    }
    m.decode_params = size_t(p + double(c.vocab) * H);
  }
  // buffers: Qwen35's for this config (x_ widened, the MTP rows) plus the hyper-connection / indexer / n-gram ones
  for (const ActBuf& a : act_layout(c, chunk, kMaxVerify, m.mtp > 0, size_t(kMaxVerify))) m.activations += a.bytes;
  m.activations += qwen4_act_bytes(c, c4, chunk, kMaxVerify, m.mtp > 0);
  const size_t C = 2 * size_t(c.lin_hk) * c.lin_dk + size_t(c.lin_hv) * c.lin_dv;
  int lin = 0, att = 0;
  for (int i = 0; i < c.layers; ++i) (c.is_linear[size_t(i)] ? lin : att) += 1;
  const size_t pconv = size_t(kRing) * c4.hc * c.hidden * 4;
  const size_t state = size_t(lin) * (3 * C * 4 + size_t(c.lin_hv) * c.lin_dv * c.lin_dk * 4) + pconv;
  const size_t tape = size_t(lin) * kMaxVerify * (C + size_t(c.lin_hv) * c.lin_dk + c.lin_hv + size_t(c.lin_hv) * c.lin_dv) * 4;
  m.slot_state = 3 * state + tape;
  m.kv_per_pos = size_t(att + (m.mtp ? 1 : 0)) *
                 (size_t(2) * c.hkv * c.head_dim * 2 + size_t(c4.ix_dim) * 2 + size_t(c4.ix_dim) * 4 / c4.ix_ratio);
  return m;
}

}  // namespace ni
