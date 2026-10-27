#include "model/qwen4_exp.h"
#include "loader/safetensors.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/json_file.h"
#include "util/log.h"
#include "vision/embed.h"

#include <nlohmann/json.hpp>

#include <CoreFoundation/CoreFoundation.h>
#include <dirent.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <unordered_map>

namespace ni {
extern const char* const kTensorKernelSource;

using json = nlohmann::json;

namespace {

Pipeline pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) {
    // The same trigger qwen35.cpp's pipe() carries: the tensor-library kernels (NI_HC_FUSE's qmt6_mix_m64_n64)
    // live in kTensorKernelSource and Gpu::pipeline() is required-by-default, so without this the first fused
    // dispatch would fatal("kernel ... not found") in a process that has not compiled the library yet.
    if (n.rfind("qmt", 0) == 0 || n.rfind("qattn_t", 0) == 0) Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
    it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  }
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

// the MTP head's tensors: mtp.* (Nace AI's packs) or language_model.mtp.* (Sushi's)
bool is_mtp(const std::string& n) { return n.rfind("mtp.", 0) == 0 || n.rfind("language_model.mtp.", 0) == 0; }

// a trunk layer's routed experts (the MTP head's, mtp.layers.*, stay with the head)
bool is_trunk_expert(const std::string& n) { return n.find(".mlp.switch_mlp.") != std::string::npos && !is_mtp(n); }

// The trunk layer a tensor belongs to ("[language_model.]model.layers.<i>.*"), -1 for the rest (embedding, head,
// final mix, the MTP head's mtp.layers.*)
int trunk_layer(const std::string& n) {
  const size_t at = n.find("model.layers.");
  return at == std::string::npos ? -1 : atoi(n.c_str() + at + 13);
}

// Whether a split's share [lo, hi) holds tensor n: its own layers' tensors, and on the head (lo 0) everything that is
// not another layer's (the embedding, the final mix, the LM head, the MTP head)
bool split_holds(const std::string& n, int lo, int hi) {
  const int l = trunk_layer(n);
  return l >= 0 ? l >= lo && l < hi : lo == 0;
}

Qwen4Exp::Split& process_split() {
  static Qwen4Exp::Split s;
  return s;
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
  const auto rp = json_file(dir + "/config.json");
  const json& r = *rp;
  if (!r.is_object()) return false;
  if (r.value("model_type", std::string()) == "qwen4_exp") return true;
  return r.contains("text_config") && r["text_config"].is_object() &&
         r["text_config"].value("model_type", std::string()) == "qwen4_exp_text";
}

float Qwen4Exp::norm_offset(double m) { return m <= 0.25 ? 1.0f : m >= 0.75 ? 0.0f : -1.0f; }

bool Qwen4Exp::q20_pack(const std::string& dir) {
  if (!is_model(dir)) return false;
  try {
    return q20_experts_in(*json_file(dir + "/config.json"));
  } catch (...) {
    return false;
  }
}

static bool tattn_env() {
  static const bool on = [] {
    const char* v = getenv("NI_QSA_TATTN");
    return v && atoi(v) != 0;
  }();
  return on;
}

bool Qwen4Exp::tattn(const Qwen35Config& c) {
  if (!tattn_env() || c.hkv <= 0 || c.hq / c.hkv != 12 || c.head_dim != 256) return false;
  if (__builtin_available(macOS 26.0, *)) return Gpu::get().apple_family() >= 10;
  return false;
}

bool Qwen4Exp::tattn(const std::string& dir) { return is_model(dir) && tattn(read_config(dir)); }

void Qwen4Exp::set_split(const Split& s) { process_split() = s; }
const Qwen4Exp::Split& Qwen4Exp::split() { return process_split(); }

// NI_HC_PRECOMB: the prefill MoE's moe_combine dispatch is skipped and its sum computed inside the next layer's
// gated residual (hc_pre_comb, qwen4.metal), bitwise identical by construction (the kernel's phase A replicates
// moe_combine's arithmetic verbatim). Default on for the Q2_0-expert pack only, where it was compared: Flash-Next
// Q2 on the M5 Pro, 25 Oct, logit_dump every row bitwise identical at chunk 2048 and 4096. The 4-bit pack takes
// the shared-expert fold (comb_pending_ on yd_'s tail rows), which no dump has covered, so there it needs =1.
// =0 turns it off everywhere.
static bool hc_precomb(bool q20_pack) {
  static const char* v = getenv("NI_HC_PRECOMB");
  return v ? strcmp(v, "0") != 0 : q20_pack;
}

// NI_PREFILL_GROUP: how many prefill chunks run as one layer-major group (1..8; the shipped default 4: groups
// on). =1 keeps the chunks one at a time, as before, for the merge gate's ABAB legs and debugging alone - there
// is no supported off posture: the gate losing closes the change unmerged rather than shipping it off. A value
// outside 1..8 is ignored and the default 4 stands (a bad value warns like NI_PRED_ALL_MIN's does); the group's
// per-chunk slots cost the plan (MemPlan::group_slots, kept out of `activations`) exactly what they cost the
// allocation (alloc4_group) - one formula, q20::group_slot_bytes - and the store-world choice (q20::group_world:
// Qwen4Exp::init_store runs it live, ni::planned_group_world predicts it before a load) charges them only while
// the streamed store is incomplete (an all-resident model never groups, q20::group_chunks). Defined below,
// after env_rows.
static int prefill_group_env();

Qwen4Exp::Qwen4Exp(const std::string& dir, int prefill_chunk, size_t mem_budget, const Split* split)
    : Qwen35(dir, prefill_chunk, NoLoad{}), mem_budget_arg_(mem_budget) {
  const Split& sp = split ? *split : process_split();
  lo_ = sp.lo;
  hi_ = sp.hi;  // (-1: load_config's, every layer)
  remote_ = sp.remote;
  // a split model's head hops mid-pass and a stage runs passes it does not own: neither groups. The all-resident
  // arm of the eligibility (a complete store) is answered at runtime, after init_store sizes the store
  // (prefill_group(), q20::group_chunks); the slot allocation waits for the same answer (alloc4_group).
  prefill_group_ = remote_ || lo_ != 0 ? 1 : prefill_group_env();
  load(dir);
}

std::unique_ptr<Qwen35> load_qwen(const std::string& dir, int prefill_chunk) {
  if (Qwen4Exp::is_model(dir)) return std::make_unique<Qwen4Exp>(dir, prefill_chunk);
  return std::make_unique<Qwen35>(dir, prefill_chunk);
}

Qwen4Exp::~Qwen4Exp() {
  if (ple_ev_) CFRelease(ple_ev_);
  if (hop_ev_) CFRelease(hop_ev_);
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
  g.norm = norm32(&ws_.get(tname(p + ".hc_norm.weight")), norm1_);
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
  // a split across Macs: the head (lo_ 0) sends the rest of every pass to its other Macs, the n-gram block (which reads
  // the tokens) among its own layers; a stage holds layers after that block
  const int L = cfg_.layers;
  const bool head = lo_ == 0, stage = lo_ > 0;
  if (head && hi_ < L && !remote_) fatal("qwen4_exp: layers %d:%d of %d with no Macs for the rest", lo_, hi_, L);
  if (stage && remote_) fatal("qwen4_exp: a stage (layers %d:%d) sends nothing on", lo_, hi_);
  if (c4_.ple_layer >= 0 && (head ? c4_.ple_layer >= hi_ : c4_.ple_layer >= lo_))
    fatal("qwen4_exp: the n-gram block is in layer %d, which the first Mac of a split must hold (here it holds 1-%d): a "
          "stage starts at layer %d or later", c4_.ple_layer + 1, head ? hi_ : lo_, c4_.ple_layer + 2);
  // the trunk and the MTP head (mtp.safetensors, read by load_mtp4): every tensor but the vision tower and the n-gram
  // table (mapped below); with NI_NO_MTP the head is left out too, as plan() leaves it out. A split's share: its own
  // layers (split_holds), the head's tensors on the head only.
  const bool no_mtp = getenv("NI_NO_MTP") != nullptr || stage;
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
      [no_mtp, lo = lo_, hi = hi_](const std::string& n) {
        return n.find("vision_tower") == std::string::npos && n.find("visual.") == std::string::npos &&
               n.find(".ngram_embedding.") == std::string::npos && !(no_mtp && is_mtp(n)) && split_holds(n, lo, hi);
      },
      [no_mtp](const std::string& f) { return f != "ple.safetensors" && !(no_mtp && f == "mtp.safetensors"); },
      defer);
  {
    // the naming from a tensor every share holds: its first layer's input mix
    const std::string probe = "model.layers." + std::to_string(lo_) + ".attn_hyper_connection.hc_norm.weight";
    if (ws_.has("language_model." + probe)) prefix_ = "language_model.";
    else if (ws_.has(probe)) prefix_ = "";
    else fatal("%s: unrecognized weight naming (expected the MLX qwen4_exp layout)", dir.c_str());
  }
  {
    // The RMSNorm weights are Qwen's zero-centred ones (applied as 1 + w), or a converter added the 1 (Sushi's packs:
    // the same weights + 1). The first attention layer's indexer q / k layernorms tell which, read from the files so
    // every Mac of a split decides alike whatever layers it holds: their means are -0.08 to -0.02 in Nace AI's 4-bit
    // pack and 0.92 to 0.98 in Sushi's, in every attention layer (the hyper-connection norms reach 0.77 in the 4-bit
    // pack: too close to tell).
    int a0 = 0;
    while (a0 < cfg_.layers - 1 && cfg_.is_linear[size_t(a0)]) ++a0;
    const std::string ix = tname("model.layers." + std::to_string(a0) + ".self_attn.indexer.");
    const SafetensorsIndex idx(dir, [&](const std::string& n) { return n.rfind(ix, 0) != 0; });
    double sum = 0;
    size_t cnt = 0;
    for (const char* w : {"q_layernorm.weight", "k_layernorm.weight"}) {
      const SafetensorsIndex::Entry* t = idx.find(ix + w);
      if (!t || (t->dtype != DType::BF16 && t->dtype != DType::F16)) continue;
      std::vector<uint16_t> v(size_t(t->numel()));
      idx.read(*t, 0, v.data(), v.size() * 2);
      for (uint16_t h : v) {
        _Float16 f;
        memcpy(&f, &h, 2);
        sum += t->dtype == DType::BF16 ? bf16_to_f32(h) : float(f);
      }
      cnt += v.size();
    }
    if (!cnt) fatal("qwen4_exp: no bf16 / f16 indexer layernorms in layer %d, which tell the RMSNorm convention", a0);
    const double m = sum / double(cnt);
    norm1_ = norm_offset(m);
    if (norm1_ < 0)
      fatal("qwen4_exp: layer %d's indexer layernorms average %.2f, between Qwen's zero-centred RMSNorm weights (near "
            "0) and weights with the 1 added at conversion (near 1): the convention is unclear", a0, m);
    logi("RMSNorm weights %s (indexer norm mean %.2f): read as %s",
         norm1_ ? "zero-centred" : "with the 1 added at conversion", m, norm1_ ? "1 + w" : "they are");
  }
  if (head) {
    embed_ = qlin("model.embed_tokens");
    lm_head_ = qlin("lm_head");
  }
  final_norm_ = nullptr;  // the final normalization is the hyper-connection mix
  layers_.assign(size_t(cfg_.layers), Layer{});
  l4_.assign(size_t(cfg_.layers), Qwen4Layer{});
  int lossy = 0;
  auto param = [&](const std::string& n) { return bf16_param(&ws_.get(tname(n)), lossy); };
  for (int i = lo_; i < hi_; ++i) {
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
      L.q_norm = bf16_copy(&ws_.get(tname(a + "q_norm.weight")), norm1_);
      L.k_norm = bf16_copy(&ws_.get(tname(a + "k_norm.weight")), norm1_);
      F.iqk = qlin(a + "indexer.index_qk_proj");
      F.iqn = norm32(&ws_.get(tname(a + "indexer.q_layernorm.weight")), norm1_);
      F.ikn = norm32(&ws_.get(tname(a + "indexer.k_layernorm.weight")), norm1_);
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
      ple_nk_ = norm32(&ws_.get(tname(q + "norm_key.weight")), norm1_);
      ple_nq_ = norm32(&ws_.get(tname(q + "norm_query.weight")), norm1_);
      ple_nc_ = norm32(&ws_.get(tname(q + "norm_conv.weight")), norm1_);
      ple_conv_ = param(q + "conv1d.weight");
      const Tensor& cw = ws_.get(tname(q + "conv1d.weight"));
      if (cw.shape.size() != 3 || cw.shape[1] != 4 || cw.shape[2] != 1) fatal("%s: conv1d is not [C, 4, 1]", q.c_str());
    }
  }
  if (head && c4_.ple_layer >= 0 && !ple_key_) fatal("qwen4_exp: no n-gram embedding at layer %d", c4_.ple_layer);
  if (head) mix_ = load_gr("model.hyper_connection_mixer", false);
  if (lossy) logi("qwen4_exp: %d parameter values not exactly representable in bf16", lossy);
  for (const QLinear* q : {&embed_, &lm_head_})
    if (*q && !q->dense() && (q->s->dtype != DType::BF16 || q->b->dtype != DType::BF16))
      fatal("quantized weights with non-bf16 scales/biases are not supported");
  check_shapes();
  check_formats();

  // the n-gram table: ple.safetensors' row shards, a GGUF's own table (ple.gguf: tools/qwen4_exp_q20.py --table
  // gguf) or Sushi's ngram_table.bin, mapped (never in GPU memory); the head's
  if (head && c4_.ple_layer >= 0) {
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
    const std::string file = dir + "/ple.safetensors", gfile = dir + "/ple.gguf", merged = dir + "/ngram_table.bin";
    if (std::ifstream(gfile)) {
      table_.open_gguf(gfile, "per_layer_token_embd.weight", cfg_.hidden / hash_.heads);
    } else if (!std::ifstream(file) && std::ifstream(merged)) {
      // Sushi's packs: the table as one [rows, dim] tensor in a safetensors file of its own: affine (weight / scales /
      // biases; mlx-serve's quantization of the released table) or bf16 (weight alone)
      parts.push_back({merged, "weight", "scales", "biases"});
      table_.open(parts, NgramTable::has_tensor(merged, "scales") ? NgramTable::Fmt::Affine : NgramTable::Fmt::BF16,
                  cfg_.hidden / hash_.heads);
    } else {
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
    }
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
    if (hc_precomb(q20_pack_)) v.push_back("hc_pre_comb_h");
    for (const std::string& n : v) Gpu::get().pipeline(n, false);
  }
  ple_ev_ = Gpu::get().new_shared_event();
  if (remote_) hop_ev_ = Gpu::get().new_shared_event();
  char share[96] = "";
  if (lo_ > 0 || hi_ < L)
    snprintf(share, sizeof share, " (this Mac: layers %d-%d%s)", lo_ + 1, hi_, remote_ ? ", the rest on other Macs" : "");
  logi("loaded %s: Qwen3.8-Flash-Next, %d layers (%d attention)%s, %.1f GB of weights in %.0f ms; %s; n-gram table "
       "%s%s%s", dir.c_str(), cfg_.layers, attn_layers(), share, ws_.total_bytes() / 1e9, now_ms() - t0,
       formats_summary().c_str(), table_.ok() ? table_.describe().c_str() : "none", store_ ? "; routed experts: " : "",
       store_ ? store_->describe().c_str() : "");
}

void Qwen4Exp::load_mtp4(const std::string& dir) {
  // the head's tensors: mtp.* in mtp.safetensors (Nace AI's packs), or language_model.mtp.* (Sushi's)
  const std::string mp = ws_.has("language_model.mtp.fc_embedding.weight") ? "language_model.mtp." : "mtp.";
  if (mp == "mtp." && !std::ifstream(dir + "/mtp.safetensors")) {
    logi("no MTP head in %s (mtp.safetensors); speculative decoding off", dir.c_str());
    return;
  }
  if (!ws_.has(mp + "fc_embedding.weight")) {
    logi("mtp.safetensors without mtp.fc_embedding; speculative decoding off");
    return;
  }
  const double t0 = now_ms();
  // the names are whole (prefix_ applies to the trunk's)
  const std::string keep = prefix_;
  prefix_ = "";
  int lossy = 0;
  (void)lossy;
  mtp_ne_ = norm32(&ws_.get(mp + "pre_fc_norm_embedding.weight"), norm1_);
  mtp_nh_ = norm32(&ws_.get(mp + "pre_fc_norm_hidden.weight"), norm1_);
  mtp_fce_ = qlin(mp + "fc_embedding");
  mtp_fch_ = qlin(mp + "fc_hidden");
  mtp_fce_.mtp = mtp_fch_.mtp = true;
  const std::string p = mp + "layers.0.";
  Layer& L = mtp_;
  L.linear = false;
  L.q = qlin(p + "self_attn.q_proj");
  L.k = qlin(p + "self_attn.k_proj");
  L.v = qlin(p + "self_attn.v_proj");
  L.o = qlin(p + "self_attn.o_proj");
  L.q_norm = bf16_copy(&ws_.get(p + "self_attn.q_norm.weight"), norm1_);
  L.k_norm = bf16_copy(&ws_.get(p + "self_attn.k_norm.weight"), norm1_);
  mtp4_.iqk = qlin(p + "self_attn.indexer.index_qk_proj");
  mtp4_.iqn = norm32(&ws_.get(p + "self_attn.indexer.q_layernorm.weight"), norm1_);
  mtp4_.ikn = norm32(&ws_.get(p + "self_attn.indexer.k_layernorm.weight"), norm1_);
  L.router = qlin(p + "mlp.gate");
  L.shared_gate = qlin(p + "mlp.shared_expert_gate");
  // EXL3 experts (Sushi's head) are read as the trunk's
  const bool tr = ws_.has(p + "mlp.switch_mlp.gate_proj.trellis");
  L.eg = qlin(p + "mlp.switch_mlp.gate_proj", true, tr);
  L.eu = qlin(p + "mlp.switch_mlp.up_proj", true, tr);
  L.ed = qlin(p + "mlp.switch_mlp.down_proj", true, tr);
  L.sg = qlin(p + "mlp.shared_expert.gate_proj");
  L.su = qlin(p + "mlp.shared_expert.up_proj");
  L.sd = qlin(p + "mlp.shared_expert.down_proj");
  for (QLinear* q : {&L.q, &L.k, &L.v, &L.o, &mtp4_.iqk, &L.router, &L.shared_gate, &L.eg, &L.eu, &L.ed, &L.sg, &L.su, &L.sd})
    q->mtp = true;
  mtp4_.ga = load_gr(p + "attn_hyper_connection", true);
  mtp4_.gm = load_gr(p + "mlp_hyper_connection", true);
  mtp_mix_ = load_gr(mp + "hyper_connection_mixer", false);
  prefix_ = keep;
  std::string why;
  if (L.eg.exl3) {
    if (!L.eg.same_format(L.eu) || !L.eg.same_format(L.ed) || !exl3::shapes_ok(exl3_shape(), exl3_experts(L), &why))
      fatal("MTP head: EXL3 experts: %s", why.empty() ? "gate/up/down differ in format" : why.c_str());
  } else if (!L.eg.same_format(L.eu) || !L.eg.same_format(L.ed) || moe_rows_kernel(true, L.eg, L.sg).empty() ||
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
static size_t qwen4_act_bytes(const Qwen35Config& c, const Qwen4ExpConfig& c4, int T, int maxv, bool mtp,
                              int group = 1) {
  const size_t D = size_t(c.hidden), SD = size_t(c4.hc) * D, Tz = size_t(T), V = size_t(maxv), R = size_t(c4.hc_rank);
  const size_t IQ = size_t(c4.ix_heads + 1) * c4.ix_dim;
  return Tz * SD * 4 - Tz * D * 4  // x_ grows from [T, D] to [T, S*D]
         + Tz * D * 4 + 2 * Tz * c4.hc * 4 + V * SD * 4 + Tz * SD * 2 + Tz * R * 4 + Tz * R * 2 + Tz * SD * 4 +
         Tz * IQ * 4 + Tz * c4.ix_heads * c4.ix_dim * 4 + Tz * SD * 4 + Tz * D * 4 + 2 * Tz * D * 2 + 4 * V * D * 4 +
         (mtp ? Tz * SD * 4 + V * SD * 4 - V * D * 4 : 0)  // the MTP head's state; mtp_post_ widened
         + Tz * size_t(c4.ix_budget + c4.ix_ratio + 3) / 4 * 4 * 4 + Tz * 4  // sparse attention's key lists
         + SD * 4 + 64 + (SD / 1024 + 1) * V * R * 4  // the one-row gated residual's row, hc_down_sk's partials
         + q20::group_slot_bytes(group, Tz, D, size_t(c4.hc));  // the chunks' slots, staging (NI_PREFILL_GROUP)
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

// The layer-major groups' slot buffers (NI_PREFILL_GROUP): the chunks' x_ / yb_ / injA_ / injM_ slots, swapped into
// the members per turn, and two sets of G n-gram stagings (the host gathers group g + 1's rows while the GPU runs
// g). init_store calls this only while the streamed store is incomplete - a group pays the store's reads alone,
// so an all-resident model never groups (q20::group_chunks) and never pays these bytes; the same conditional
// decides the plan's MemPlan::group_slots (plan() sizes it apart from `activations`, and
// ni::planned_group_world predicts which store world pays it). One formula with the
// plan's: q20::group_slot_bytes.
void Qwen4Exp::alloc4_group() {
  if (prefill_group_ < 2) return;
  Gpu& g = Gpu::get();
  const size_t T = size_t(tmax_), D = size_t(cfg_.hidden), SD = size_t(c4_.hc) * D, G = size_t(prefill_group_);
  xg_.resize(G), yg_.resize(G), injAg_.resize(G), injMg_.resize(G), ple_pfg_.resize(2 * G);
  for (size_t c = 0; c < G; ++c) {
    xg_[c] = g.alloc(T * SD * 4, "x4g");
    yg_[c] = g.alloc(T * D * 4, "block_outg");
    memset(yg_[c].ptr(), 0, yg_[c].size);  // accumulated into, as yb_
    injAg_[c] = g.alloc(T * c4_.hc * 4, "inject_ag");
    injMg_[c] = g.alloc(T * c4_.hc * 4, "inject_mg");
  }
  for (Buffer& b : ple_pfg_) b = g.alloc(T * D * 2, "ple_rowsg");
}

// ---- memory ----

std::unique_ptr<Seq> Qwen4Exp::new_seq(int limit, int initial) {
  auto s = Qwen35::new_seq(limit, 0);
  Gpu& g = Gpu::get();
  s->ixk.assign(size_t(cfg_.layers), Buffer{});
  s->ixb.assign(size_t(cfg_.layers), Buffer{});
  s->pconv = g.alloc(size_t(kRing) * c4_.hc * cfg_.hidden * 4, "ple_conv_state");
  memset(s->pconv.ptr(), 0, s->pconv.size);
  if (remote_) s->rid = remote_->new_seq(limit);
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && remote_ && !remote_reserve(*remote_, *s, cap))
    fatal("the other Macs of this split have no room for a %d-token conversation%s", cap,
          remote_->lost().empty() ? "" : (": " + remote_->lost()).c_str());
  if (cap > 0) grow_kv(*s, cap);
  return s;
}

bool Qwen4Exp::reserve(Seq& s, int need) {
  // the other Macs' layers first: they hold the same positions of their own caches, grown by their own budgets and
  // rounding, so what they hold (rcap) is tracked apart from this Mac's cap
  if (remote_ && !remote_reserve(*remote_, s, need)) return false;
  return Qwen35::reserve(s, need);
}

void Qwen4Exp::grow_kv(Seq& s, int cap) {
  const int old = s.cap;
  Qwen35::grow_kv(s, cap);
  // the indexer's caches grow alongside: rows by position, so the old buffer is the new one's prefix
  Gpu& g = Gpu::get();
  const size_t kb = size_t(cap) * c4_.ix_dim * 2, bb = (size_t(cap) / c4_.ix_ratio + 1) * c4_.ix_dim * 4;
  std::vector<std::pair<Buffer*, size_t>> bufs;
  for (int i = lo_; i < hi_; ++i)
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
  if (remote_ && s.rid >= 0) remote_->drop_kv(s.rid);
  s.rcap = 0;
}

size_t Qwen4Exp::kv_bytes_per_pos() const {
  int att = 0;
  for (int i = lo_; i < hi_; ++i) att += !cfg_.is_linear[size_t(i)];
  const size_t layers = size_t(att + (mtp_loaded_ ? 1 : 0));
  return layers * (size_t(2) * cfg_.hkv * cfg_.head_dim * 2 + size_t(c4_.ix_dim) * 2 + size_t(c4_.ix_dim) * 4 / c4_.ix_ratio);
}

void Qwen4Exp::reset(Seq& s) {
  Qwen35::reset(s);
  if (s.pconv) memset(s.pconv.ptr(), 0, s.pconv.size);
  if (pend_.s == &s) pend_ = Pending{};
  if (remote_ && s.rid >= 0) remote_->reset(s.rid);
}

Checkpoint Qwen4Exp::new_checkpoint() {
  Checkpoint c = Qwen35::new_checkpoint();
  c.pconv = Gpu::get().alloc(size_t(kRing) * c4_.hc * cfg_.hidden * 4, "ckpt_ple_conv");
  if (remote_) c.rid = remote_->new_checkpoint();
  return c;
}

void Qwen4Exp::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Qwen35::save(e, s, c, pos);
  enc_copy(e, s.pconv, c.pconv);
  if (remote_) remote_->save(s.rid, c.rid, pos);
}

void Qwen4Exp::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  Qwen35::restore(e, s, c);
  enc_copy(e, c.pconv, s.pconv);
  if (remote_) remote_->restore(s.rid, c.rid);
}

void Qwen4Exp::set_rope_positions(Seq& s, int n, const std::vector<vision::ImageSpan>& spans) {
  Qwen35::set_rope_positions(s, n, spans);
  if (remote_) remote_->rope(s.rid, s.rope_n, s.rope_delta, s.rope_n ? s.rope_pos.as<int32_t>() : nullptr);
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

namespace {
// A wait length in milliseconds from the environment: a whole number that is not negative, or `def` with one warning.
// A bare atoi would turn a typo ("10s", "abc") or an empty value into 0, which switches the wait off and brings back
// the early sample this knob exists to undo, with nothing in the log.
int env_ms(const char* name, int def) {
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = nullptr;
  const long x = strtol(v, &end, 10);
  if (end != v && !*end && x >= 0 && x <= 600000) return int(x);
  logi("%s=%s is not a wait of 0 or more milliseconds, ignored: using %d", name, v, def);
  return def;
}
}  // namespace

// The store over the deferred Q2_0 planes. Its size: NI_EXPERT_STORE=<GB> or all; by default what the budget leaves
// after the weights and buffers allocated so far (and the room set apart below).
void Qwen4Exp::init_store(const std::string& dir) {
  (void)dir;
  const uint32_t E = uint32_t(cfg_.experts);
  std::vector<std::string> files;
  std::map<std::string, int> fidx;
  // the layers held here (a split's share): the store's layer j is trunk layer lo_ + j
  std::vector<q20::LayerSrc> src(size_t(hi_ - lo_));
  static const char* const kMat[3] = {"gate", "up", "down"};
  for (int i = lo_; i < hi_; ++i)
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
        q20::PlaneSrc& p = src[size_t(i - lo_)].p[m * 2 + part];
        p.file = f;
        p.off = off;
        p.stride = len / E;
        p.len = uint32_t(len / E);
      }
  const q20::SlotLayout lay = q20::slot_layout(uint32_t(cfg_.hidden), uint32_t(cfg_.moe_I));
  const size_t all = size_t(hi_ - lo_) * E * lay.bytes;
  q20::ExpertStore::Options o;
  // The store-world decision is one function (q20::group_world) the live load and every pre-load prediction
  // (ni::planned_group_world) share, and the NI_EXPERT_STORE pin is one parse (q20::store_pin_bytes) both
  // sides share, so the load and the fit numbers never diverge on what the pin means or what the store costs.
  // A pin decides its own world: the store is the pin's bytes (the parser clamped to all), and a pin short of
  // every expert stays incomplete, so alloc4_group below allocates the slots on top (the same predicate,
  // !store_->complete()) - no settle wait, no budget/avail/cap sizing, the pin overrides them.
  size_t pin = 0;
  if (q20::store_pin_bytes(getenv("NI_EXPERT_STORE"), all, pin)) {
    const size_t gsb = q20::group_slot_bytes(prefill_group_, size_t(tmax_), size_t(cfg_.hidden), size_t(c4_.hc));
    q20::AutoStoreIn in;
    in.all = all, in.least = size_t(3) * E * lay.bytes;
    const q20::GroupWorld gw = q20::group_world(gsb, in, &pin);
    o.bytes = gw.st.bytes;
    if (gw.charge) logi("expert store: NI_PREFILL_GROUP=%d slots -%.2f GB", prefill_group_, double(gw.charge) / 1e9);
    logi("expert store: %s", q20::auto_store_reason(gw.st, gw.in).c_str());
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
    // the two explains the stall is not measured. NI_EXPERT_CAP=<share of the RAM, 0..1> sets it on any chip (0 = none;
    // q20::auto_store_cap). The available-memory guard (q20::auto_store_size) applies on every chip: the budget assumes
    // the Mac to itself. `kapitsa doctor` prints the same reason before a load (doctor.cpp).
    q20::AutoStoreIn in;
    in.all = all, in.least = size_t(3) * E * lay.bytes, in.left = left, in.used = used, in.keep = keep;
    in.ram = size_t(ram);
    // NI_EXPERT_SETTLE_MS: how long the guard may wait in all for memory a process that just exited still holds
    // (default 10000; 0 = one sample); it stops once 3 s pass without 0.5 GB more available
    const int settle = env_ms("NI_EXPERT_SETTLE_MS", 10000);
    const q20::SettledAvail sa = q20::settle_available(std::min(left, all), q20::auto_store_headroom(in.ram), settle, 250, 3000, size_t(500) << 20,
                                                       [](int ms) {
                                                         if (ms) usleep(useconds_t(ms) * 1000);
                                                         return q20::available_memory();
                                                       });
    in.avail = sa.avail;
    if (sa.waited_ms)
      logi("expert store: the system's available memory %.1f GB at first, %.1f GB after %.2f s", double(sa.first) / 1e9,
           double(sa.avail) / 1e9, sa.waited_ms / 1e3);
    in.cap = q20::auto_store_cap(Gpu::get().apple10());
    // The layer-major groups' store-world choice (q20::group_world - the one function every side asks;
    // ni::planned_group_world predicts it before a load): the store is sized WITHOUT the slot bytes first -
    // alloc4_group has not allocated them, so `used` above does not carry them - and only a store that stays
    // incomplete in that world pays them (the charged world re-sizes: left and avail shrink, used grows).
    // Monotone in one direction - more charge shrinks the store's budget, which only moves it further from
    // complete - so the one correction is the fixed point, no oscillation; and the allocation at the end of
    // this function follows the same decision (alloc4_group behind !store_->complete()), so the plan's
    // MemPlan::group_slots and the load's reality agree by construction. The all-resident world charges
    // nothing and allocates nothing.
    const size_t gsb =
        q20::group_slot_bytes(prefill_group_, size_t(tmax_), size_t(cfg_.hidden), size_t(c4_.hc));
    const q20::GroupWorld gw = q20::group_world(gsb, in);
    const q20::AutoStore& st = gw.st;
    in = gw.in;
    o.bytes = st.bytes;
    if (gw.charge) logi("expert store: NI_PREFILL_GROUP=%d slots -%.2f GB", prefill_group_, double(gw.charge) / 1e9);
    logi("expert store: %s", q20::auto_store_reason(st, in).c_str());
  }
  o.bytes = std::min(o.bytes, all);
  if (const char* r = getenv("NI_EXPERT_READERS"); r && *r) o.cache.readers = atoi(r);
  if (const char* io = getenv("NI_EXPERT_IO"); io && !strcmp(io, "direct")) o.cache.direct = true;
  if (const char* fl = getenv("NI_EXPERT_FILL"); fl && !strcmp(fl, "0")) o.cache.fill = false;
  if (const char* sv = getenv("NI_SCAN_VICTIM"); sv && !strcmp(sv, "0")) o.cache.scan_victim = false;
  store_ = std::make_unique<q20::ExpertStore>(files, src, E, lay, o);
  Gpu::get().make_resident(store_->buffers());
  if (!store_->complete()) {
    Gpu& g = Gpu::get();
    plog_ = g.alloc(size_t(kMaxVerify) * E * 4, "pred_logits");
    pidx_ = g.alloc(size_t(kMaxVerify) * cfg_.top_k * 4, "pred_idx");
    pwts_ = g.alloc(size_t(kMaxVerify) * (cfg_.top_k + 1) * 4, "pred_wts");
    // the layer-major groups' slots: only while the store is incomplete, the same predicate that makes
    // prefill_group() answer G to the engine (q20::group_chunks) - with every expert resident the chunk path
    // keeps its NI_HC_PRECOMB fold and these buffers stay unallocated
    alloc4_group();
  }
}

namespace {
// A row-count knob from the environment: a whole number that is not negative, or `def` with one warning. Negative,
// empty and non-numeric values would otherwise become 0 (or a negative number) and switch the gate on for every chunk.
int env_rows(const char* name, int def) {
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = nullptr;
  const long x = strtol(v, &end, 10);
  if (end != v && !*end && x >= 0 && x <= 1 << 20) return int(x);
  logi("%s=%s is not a row count of 0 or more, ignored: using %d", name, v, def);
  return def;
}
}  // namespace

static int prefill_group_env() {
  static const int g = [] {
    const int v = env_rows("NI_PREFILL_GROUP", 4);
    return v >= 1 && v <= 8 ? v : 4;
  }();
  return g;
}

// a prefill chunk's MoE: a chunk of 64+ rows routes most of the next layer's experts, so every one of them not in
// memory is read during this layer; shorter ones only read what they route
q20::Experts Qwen4Exp::q20_fetch(Encoder& e, const Layer& L, const Buffer& ids, uint32_t n, int T,
                                 const std::function<void()>& between) {
  const int li = layer_index(L);
  if (!store_ || li < 0) return Qwen35::q20_fetch(e, L, ids, n, T, between);
  // (the store's layers are this Mac's: trunk layer li is its li - lo_)
  // Seam wrap: a chunk's last layer has the store's layer 0 - the next chunk's first - read ahead instead of
  // nothing. The prediction never crosses a chunk boundary, and layer 0 is also the first victim of every sweep (a
  // scan admits its keys over the T1 keys of the layers below them), so without the wrap each seam turn re-reads
  // its whole routed set with the GPU blocked: on a 128K prompt the (chunks - 1) seam turns took ~40% of the
  // prompt's expert-store stall (measured with the wrap off). Only the whole-layer read can wrap - the next
  // chunk's rows are not these rows, so a routed prediction at the seam would name the wrong experts - and a
  // chunk below NI_PRED_ALL_MIN passes no prediction at all, where the wrap does nothing: on the engine's
  // automatic chunking (auto_chunk returns 256 rows or more) every chunk of a prompt but its last is above the
  // gate (prefill_chunk_rows gives min(chunk, left), or an even split's ceil(left / k) above chunk / 2; only the
  // last chunk can come out short). The prompt's last chunk does not
  // wrap (seam_more_, set by encode_prefill): decode follows it, and decode's first turn is one row routing
  // top-k experts, so a whole-layer sweep there would cost the first token for a read that turn barely uses.
  // group_inner_ (a layer-major group's turn with chunks remaining): its successor is the same layer over the next
  // chunk, which this turn leaves resident, so the turn reads nothing ahead - the visit's last turn admits the next
  // layer's whole set once for the visit, and only the group's final turn wraps to layer 0 (seam_more_)
  const int nl = group_inner_ ? -1 : q20::seam_next_layer(li, lo_, hi_, seam_more_);
  // The read-ahead of the whole next layer costs the store's size on an 85-token prompt (1232 ms of a 1613 ms TTFT),
  // and only a chunk of 64+ rows routes most of that layer anyway: NI_PRED_ALL_MIN is the shortest chunk that pays it
  // (default 64, the constant it replaces). A bad value is warned about and ignored, as NI_EXPERT_CAP's is: parsing
  // it to 0 would read the whole next layer on every chunk, the cost this gate exists to avoid.
  static const int pred_all_min = env_rows("NI_PRED_ALL_MIN", 64);
  // NI_SCAN_MIN: the shortest chunk whose sweep leaves the recency order alone (default 16, the constant it replaces).
  static const int scan_min = env_rows("NI_SCAN_MIN", 16);
  const bool all = T >= pred_all_min && nl >= 0;
  // group_floor_ (#624): inside a layer-major group a visit's sweep may only evict keys below its own layer's index,
  // so the keys the group's later turns still read stay; -1 (the chunk path, and a group's wrapping last visit)
  // leaves the sweep's victim rule as shipped
  store_->begin(e, li - lo_, ids, n, nullptr, 0, all ? nl : -1, all, /*scan=*/T > scan_min, /*warm=*/false,
                group_floor_);
  if (between) between();
  store_->end(e);
  return store_->experts(li - lo_);
}

q20::Experts Qwen4Exp::q20_rows_fetch(Encoder& e, const Layer& L, int R, const Buffer& idx, bool pred) {
  const int li = layer_index(L);
  if (!store_ || li < 0) return q20_experts(L);
  const uint32_t n = uint32_t(R * cfg_.top_k);
  store_->begin(e, li - lo_, idx, n, pred ? &pidx_ : nullptr, n, pred ? li + 1 - lo_ : -1, false, false, /*warm=*/true);
  store_->end(e);
  return store_->experts(li - lo_);
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

// Whether layer `layer`'s enc_capture4 skips — its early-returns exactly, so the NI_HC_PRECOMB fold
// gate never takes yb_ away from an armed capture.
bool Qwen4Exp::capture_noop(int layer, int pos0, int R) const {
  if (!cap_ || layer >= int(cap_->slot.size()) || cap_->slot[size_t(layer)] < 0) return true;
  return std::max(pos0, cap_->pos_base) >= std::min(pos0 + R, cap_->pos_base + cap_->rows);
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
      // verify rows (R >= 2): hc_up_mix4, which unpacks each run of up weights once for all rows, 4 blocks of 16
      // channels per threadgroup; bitwise equal to hc_up_mix3 / hc_up_mix2 (fn_check FN_ROWS, hc-time). At Flash-Next's
      // shape (D 2560, rank 320, 6-bit), R=3 down+up 65.0 -> 54.1 us (fn_check hc-time; NI_HC_MIX4=0: hc_up_mix3)
      static const bool mix4 = !(getenv("NI_HC_MIX4") && !strcmp(getenv("NI_HC_MIX4"), "0"));
      const bool m4 = mix4 && R >= 2 && D % 64 == 0;
      const int ucpt = m4 ? 4 : cpt;
      e.use(pipe((m4 ? "hc_up_mix4_c" : "hc_up_mix3_c") + std::to_string(ucpt) + b));
      e.buf(0, hpart_); e.buf(1, hx2_); e.buf(2, hstat_); e.buf(3, g.norm->buf);
      e.buf(4, g.up.w->buf); e.buf(5, g.up.s->buf); e.buf(6, g.up.b->buf); e.buf(7, u);
      e.buf(8, X, x_row0 * SD * 4); e.buf(9, yb_); e.buf(10, inj_out);
      e.val(11, fp);
      e.dispatch(D / (16 * ucpt), 1, 1, 256);
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
  if (comb_pending_.active && !inject) fatal("hc_precomb: pending combine met a non-injecting gated residual");
  if (comb_pending_.active) {
    // NI_HC_PRECOMB: hc_pre_comb_h computes the previous MoE's combine sum from yd_ / ys_ / rwts_ (its
    // phase A is moe_combine verbatim) and injects it from threadgroup memory; yb_ is neither read (it
    // stays zero from the last gated residual's zeroing) nor written.
    e.use(pipe("hc_pre_comb_h"));
    e.buf(0, X); e.buf(1, inj_prev); e.buf(2, g.norm->buf);
    e.buf(3, g.inj ? g.inj.w->buf : g.norm->buf); e.buf(4, xn4h_); e.buf(5, inj_out);
    e.val(6, p);
    e.buf(7, comb_pending_.yd); e.buf(8, comb_pending_.ys); e.buf(9, rwts_);
    const kp::Combine cp{comb_pending_.D, comb_pending_.k};  // the kernel's HcCombParams is this {D, k} layout
    e.val(10, cp);
    e.dispatch(uint32_t(T), 1, 1, 1024);
    comb_pending_ = CombPending{};  // consumed
  } else {
    e.use(pipe("hc_pre_h"));
    e.buf(0, X); e.buf(1, yb_); e.buf(2, inj_prev); e.buf(3, g.norm->buf);
    e.buf(4, g.inj ? g.inj.w->buf : g.norm->buf); e.buf(5, xn4h_); e.buf(6, inj_out);
    e.val(7, p);
    e.dispatch(uint32_t(T), 1, 1, 1024);
  }
  enc_qmm(e, g.down, xn4h_, hch_, T, false);
  const uint32_t n = uint32_t(T) * uint32_t(c4_.hc_rank);
  const float inv_s = 1.0f / float(S);
  e.use(pipe("hc_act_h"));
  e.buf(0, hch_); e.buf(1, hchh_); e.val(2, n); e.val(3, inv_s);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  // NI_HC_FUSE (default on; =0 off): the mix folded into the up projection's tensor-ops tile — qmt6_mix_m64_n64 runs
  // qmt6_m64_n64's k-loop once per stream and drains sigmoid(m) * Xn into a second destination tensor in
  // hc_mix_impl's order, so the [T, S*D] fp32 mix buffer (hcm_) is never written or read back (16K prefill on
  // Flash-Next q20: 84 MB written + 84 MB read and one dispatch per hyper-connection per chunk saved).
  // Bitwise identical to the two dispatches it replaces (Flash-Next Q2 logit_dump, 25 Oct: every row). The fold
  // touches only the 6-bit hyper-connection weights, which the Q2 pack takes unchanged from the 4-bit pack.
  // Engages only where the up GEMM it replaces is exactly enc_qmm_large's qmt6_m64_n64 branch: qmm_on() (its
  // q.on), qmt6_on(), T >= 64, and not the short-prefill enc_proj detour (!small_t(T), which enc_qmm takes when
  // shortpf() && T <= shortpf_max()). Against any other baseline (qrf, qmr, enc_proj) the fused tile would be a
  // reassociation rather than the bitwise claim, so the fold stays off there.
  static const bool fuse = !(getenv("NI_HC_FUSE") && !strcmp(getenv("NI_HC_FUSE"), "0"));  // default on: bitwise identical
  const uint32_t rank = uint32_t(c4_.hc_rank);
  if (fuse && qmm_on() && qmt6_on() && T >= 64 && !small_t(T) && g.down.same_format(g.up) && g.up.bits == 6 &&
      g.up.gs == 64 && rank % 64 == 0 && D % 64 == 0) {
    const struct {
      uint32_t M, N, K, x_ld, D, S, u_ld;
    } mp{uint32_t(T), S * D, rank, rank, D, S, D};
    e.use(pipe("qmt6_mix_m64_n64"));
    e.buf(0, hchh_); e.buf(1, g.up.w->buf); e.buf(2, g.up.s->buf); e.buf(3, g.up.b->buf);
    e.buf(4, xn4h_); e.buf(5, xh_);
    e.val(6, mp);
    e.dispatch(D / 64, (uint32_t(T) + 63) / 64, 1, 128);
    return;
  }
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

// NI_QSA_ROWFUSE=1: price the row-fused score kernel (qsa_score_rows) against the per-row one. It is a byte measure
// first — R rows read the same indexer key blocks, so four rows per threadgroup read them 4x fewer times — and a
// bitwise-expected one second: the per-score expression and its order are unchanged, so a logit md5 mismatch escalates
// rather than being accepted. Off by default; the kernel holds four rows of queries (NH <= 4 heads, HD <= 128) and one
// sub-chunk of keys in 16,384 B of threadgroup memory, the most the validation layer accepts.
bool qsa_rowfuse_on(uint32_t R, uint32_t NH, uint32_t HD, uint32_t nb) {
  static const bool on = [] { const char* v = getenv("NI_QSA_ROWFUSE"); return v && atoi(v) != 0; }();
  const uint32_t H4 = HD / 4;
  return on && R >= 2 && R <= 8 && HD % 4 == 0 && NH <= 4 && H4 <= 32 && nb > 8;
}
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
  if (qsa_rowfuse_on(uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), nb)) {
    e.use(pipe("qsa_score_rows"));
    e.buf(0, iq_, size_t(iq_row0) * c4_.ix_heads * c4_.ix_dim * 4); e.buf(1, bk); e.buf(2, qscore_);
    e.val(3, p);
    e.dispatch((nb + 127) / 128, (uint32_t(R) + 3) / 4, 1, 64);
  } else {
    e.use(pipe("qsa_score"));
    e.buf(0, iq_, size_t(iq_row0) * c4_.ix_heads * c4_.ix_dim * 4); e.buf(1, bk); e.buf(2, qscore_);
    e.val(3, p);
    e.dispatch((nb + 255) / 256, uint32_t(R), 1, 256);
  }
  e.use(pipe("qsa_select"));
  e.buf(0, qscore_); e.buf(1, qidx_); e.buf(2, qcnt_);
  e.val(3, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

// The shapes qsa_all's 1024-thread row threadgroups cover (see the kernel): eight head slices of HD = 128, and at
// most R / 2 <= 8 re-derived blocks per row (a block spans ratio >= 2 rows, so R <= 16 keeps bk4's nine rows free).
bool Qwen4Exp::qsa_fusable(int R) const {
  static const bool fuse = env_on("NI_QSA_FUSE");
  const int ratio = c4_.ix_ratio;
  return fuse && R >= 1 && R <= 16 && c4_.ix_heads >= 1 && c4_.ix_heads <= 8 && c4_.ix_dim == 128 && ratio >= 2 &&
         ratio <= 16 && (ratio & (ratio - 1)) == 0;
}

// enc_qsa + enc_qsa_select as one qsa_all dispatch (NI_QSA_FUSE): the four's computation, with block keys and scores a
// few ulp off theirs (see the kernel in qwen4.metal); the callers keep the sparse attention after enc_qsa_select
void Qwen4Exp::enc_qsa_fused(Encoder& e, const Seq& s, const Qwen4Layer& F, const Buffer& iqk, uint32_t ld, int R,
                             int pos0, const Buffer& ik, const Buffer& bk) {
  const QsaParams p{uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), uint32_t(cfg_.rot_dims), uint32_t(pos0),
                    uint32_t(c4_.ix_ratio), ld, cfg_.eps, cfg_.theta};
  const struct {
    uint32_t mn;
    int32_t delta;
    uint32_t sec1, sec2;
  } m{uint32_t(s.rope_n), s.rope_delta, uint32_t(3 * cfg_.mrope[1]), uint32_t(3 * cfg_.mrope[2])};
  const Buffer& mpos = s.rope_pos ? s.rope_pos : iq_;
  const uint32_t nb = uint32_t(pos0 + R) / uint32_t(c4_.ix_ratio);  // blocks of the last row
  const size_t need = size_t(R) * nb * 4;
  if (qscore_.size < need) regrow(e, qscore_, std::max(need, size_t(4) << 20), "qsa_scores");
  const QsaSelParams ps{uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), uint32_t(pos0),
                        uint32_t(c4_.ix_ratio), uint32_t(c4_.ix_budget / c4_.ix_ratio), nb, qidx_ld_};
  e.use(pipe("qsa_all"));
  e.buf(0, iqk); e.buf(1, F.iqn->buf); e.buf(2, F.ikn->buf); e.buf(3, ik); e.buf(4, bk);
  e.buf(5, qscore_); e.buf(6, qidx_); e.buf(7, qcnt_);
  e.val(8, p);
  e.buf(9, mpos); e.val(10, m); e.val(11, ps);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

void Qwen4Exp::enc_qsa_mask(Encoder& e, const Buffer& bk, int R, int pos0, int row0, uint32_t mwords) {
  const uint32_t nb = uint32_t(pos0 + R) / uint32_t(c4_.ix_ratio);
  const size_t need = size_t(R) * nb * 4, mneed = size_t(row0 + R) * mwords * 4;
  if (qscore_.size < need) regrow(e, qscore_, std::max(need, size_t(4) << 20), "qsa_scores");
  if (qmask_.size < mneed) regrow(e, qmask_, std::max(size_t(tmax_) * mwords * 4, mneed), "qsa_mask");
  const QsaSelParams p{uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), uint32_t(pos0), uint32_t(c4_.ix_ratio),
                       uint32_t(c4_.ix_budget / c4_.ix_ratio), nb, mwords};
  if (qsa_rowfuse_on(uint32_t(R), uint32_t(c4_.ix_heads), uint32_t(c4_.ix_dim), nb)) {
    e.use(pipe("qsa_score_rows"));
    e.buf(0, iq_, size_t(row0) * c4_.ix_heads * c4_.ix_dim * 4); e.buf(1, bk); e.buf(2, qscore_);
    e.val(3, p);
    e.dispatch((nb + 127) / 128, (uint32_t(R) + 3) / 4, 1, 64);
  } else {
    e.use(pipe("qsa_score"));
    e.buf(0, iq_, size_t(row0) * c4_.ix_heads * c4_.ix_dim * 4); e.buf(1, bk); e.buf(2, qscore_);
    e.val(3, p);
    e.dispatch((nb + 255) / 256, uint32_t(R), 1, 256);
  }
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
    // The lists on the matrix units (tensor/qsattn_t.metal, qattn_t_sparse_g12d2) where the kernel fits: 12 query
    // heads per KV head, head dim 256, Apple10 and macOS 26 (the Metal 4 library). Else attn_sparse_rows.
    // Opt-in (NI_QSA_TATTN=1): off by default since #585, a 64K decode that ends early and loops with it on.
    // tattn() is also engine.cpp auto_chunk's gate for the Q2_0 pack's 4096-row chunks.
    const bool tat = tattn(cfg_);
    if (tattn_env()) {
      static bool logged = false;
      if (!logged) {
        logged = true;
        fprintf(stderr, "[kapitsa] NI_QSA_TATTN: %s\n",
                tat ? "sparse list attention on the matrix units (qattn_t_sparse_g12d2)"
                    : "needs 12 query heads per KV head, head dim 256, an M5 and macOS 26; keeping attn_sparse_rows");
      }
    }
    if (tat) {
      const uint32_t rr = uint32_t(R);
      e.use(pipe("qattn_t_sparse_g12d2"));
      e.buf(0, qo_, qoff); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_, qoff * 2); e.buf(4, attn_out_, qoff);
      e.val(5, p);
      e.buf(6, qidx_); e.buf(7, qcnt_); e.val(8, rr);
      e.dispatch(rr, uint32_t(cfg_.hkv), 1, 64);  // a pair of simdgroups per row and KV head
      return;
    }
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
  if (qsa_fusable(1) && qsa_sparse(pos)) {  // the fused chain (NI_QSA_FUSE): the four dispatches as one
    enc_qsa_fused(e, s, F, iqk_, F.iqk.N, 1, pos, s.ixk[size_t(li)], s.ixb[size_t(li)]);
    enc_attn_sparse(e, s.kc[size_t(li)], s.vc[size_t(li)], s.cap, 1, true, 0);
    enc_qmv(e, L.o, attn_out_, yb_, nullptr, nullptr);
    return;
  }
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
// experts: q20::enc_decode); EXL3 experts and other formats take the per-row MoE (the verify path at one row; u must
// then be in xrn_).
void Qwen4Exp::enc_moe4(Encoder& e, const Layer& L, const Buffer& u) {
  if (L.eg.exl3) {
    if (u.mtl != xrn_.mtl) fatal("enc_moe4: the input must be xrn_");
    enc_moe_exl3_rows(e, L, 1);
    return;
  }
  const bool q20 = L.eg.q20;  // check_formats: all three Q2_0 and a 4-bit shared expert
  if (!q20 && !(L.eg.q4() && L.eu.q4() && L.ed.q4() && L.sg.q4() && L.su.q4() && L.sd.q4())) {
    if (u.mtl != xrn_.mtl) fatal("enc_moe4: the input must be xrn_");
    enc_moe_rows(e, L, 1, yb_, false, &xrn_);
    return;
  }
  constexpr uint32_t nsg = 8, gu_rows = 2, rows = 2;
  // streamed experts: the next layer's router on this input too (its top-k is read ahead during this layer)
  const int li = layer_index(L);
  const bool pred = q20 && store_ && !store_->complete() && li >= 0 && li + 1 < hi_;
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

void Qwen4Exp::enc_step_layers(Encoder& e, Seq& s, int pos, const PleWait& pw) {
  const int D = cfg_.hidden;
  for (int i = lo_; i < hi_; ++i) {
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
}

void Qwen4Exp::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  const int D = cfg_.hidden;
  const PleWait pw = c4_.ple_layer >= 0 ? ple_rows_for(s, pos) : PleWait{};
  enc_embed(e, s.tok, pos, 1, yb_);
  enc_expand(e, 1);
  enc_step_layers(e, s, pos, pw);
  enc_hop(e, Remote::Kind::step, s, pos, 1);
  // the final mix (its injection leaves the pre-mix state in x_: the MTP head's input)
  enc_gr_rows(e, mix_, 1, x_, 0, true, injM_, injA_, xrn_, uint32_t(D));
  if (!sample) return;
  enc_sample(e, s, xrn_, pos + 1, sp, step);
  ple_after_sample(e, s, pos + 1);
}

// One layer's prefill block over one chunk: the n-gram block on its layer, the gated residual, the token mixer, the
// MoE and the capture hook. The chunk loop below runs it per layer; the layer-major group path (encode_prefill_group)
// runs it per visit and chunk, with the chunk's state swapped into the members.
void Qwen4Exp::enc_prefill_layer(Encoder& e, Seq& s, int i, int pos0, int T, const Buffer& rows, Checkpoint* snap,
                                 int snap_row) {
  const Layer& L = layers_[size_t(i)];
  const Qwen4Layer& F = l4_[size_t(i)];
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
  // NI_HC_PRECOMB: fold this MoE's combine sum into layer i + 1's gated residual only when nothing
  // reads yb_ in between — no armed capture of this layer, and no PLE boundary, debug cut or final
  // layer after it (those consume yb_ through enc_inject / enc_hop) — and the kernel's threadgroup
  // sum fits: one float4 per thread of the row's 1024, so D % 4 == 0 and D / 4 <= 992 (D <= 3968).
  // A layer-major group never arms it: the next layer's residual there is another visit's turn, whose
  // chunk state has moved on (in_group_).
  precomb_arm_ = !in_group_ && hc_precomb(q20_pack_) && i + 1 < hi_ && i + 1 != dbg_layers_ && i + 1 != c4_.ple_layer &&
                 cfg_.hidden % 4 == 0 && cfg_.hidden <= 3968 && capture_noop(i, pos0, T);
  enc_moe_prefill(e, L, T, yb_, /*xh_ready=*/true);
  precomb_arm_ = false;
  enc_capture4(e, i, pos0, T);
  // a command buffer per layer of a long chunk (~1/48 of it), so the desktop's own GPU work is not held up
  if (T >= 256 && i + 1 < cfg_.layers) e.split();
}

void Qwen4Exp::enc_prefill_layers(Encoder& e, Seq& s, int pos0, int T, const Buffer& rows, Checkpoint* snap,
                                  int snap_row) {
  for (int i = lo_; i < hi_; ++i) {
    if (i == dbg_layers_) break;
    enc_prefill_layer(e, s, i, pos0, T, rows, snap, snap_row);
  }
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
  // the seam-wrap gate: every chunk but the prompt's last reads the store's layer 0 ahead at its last layer - the
  // engine sets `sample` on exactly the prompt's last chunk, and decode follows that one (q20_fetch's seam wrap)
  seam_more_ = q20::seam_wraps(sample);
  enc_prefill_layers(e, s, pos0, T, rows, snap, snap_row);
  seam_more_ = false;
  if (comb_pending_.active) fatal("hc_precomb: a pending combine outlived the prefill layers");
  // the pre-mix state of every row (the MTP head's history, logits); a split's last Mac injects its last block itself
  if (remote_ && dbg_layers_ < 0) enc_hop(e, Remote::Kind::prefill, s, pos0, T, snap, snap_row);
  else if (dbg_layers_ != 0) enc_inject(e, T, x_, injM_);
  if (sample) {
    enc_gr_rows(e, mix_, 1, x_, size_t(T - 1), false, injM_, injA_, xrn_, uint32_t(D));
    enc_sample(e, s, xrn_, pos0 + T, sp, step);
    ple_after_sample(e, s, pos0 + T);
  }
  if (snap) snap->pos = pos0 + snap_row;
}

// A layer-major prefill group (NI_PREFILL_GROUP): every layer visits every chunk of the group before the next layer
// runs, so the store reads a layer's experts once per group instead of once per chunk, and the chunk seams inside
// the group disappear. A turn's read-ahead follows the group's successor table, not the chunk path's: turn (i, c)'s
// successor is (i, c + 1) while chunks remain - the same layer, left resident, so an inner turn reads nothing ahead
// - a visit's last turn admits layer i + 1's whole set once for the visit, under pred_floor i (#624) so the sweep
// cannot evict keys the group's later turns still read, and only the group's final turn wraps to layer 0, with no
// floor, exactly like a chunk's seam wrap, for the chunk after the group. The chunks' state (x_ / yb_ /
// injA_ / injM_) lives in per-chunk slots swapped into the members per turn - Metal binds buffers at encode time -
// and the GDN recurrences and the KV advance in place because a visit's chunks run in position order. The engine
// only forms plain groups (no checkpoint inside a chunk, no split rerun, no image or grammar rows, never the pass's
// last chunk), so there is no snapshot, no sample and no precomb here.
void Qwen4Exp::encode_prefill_group(Encoder& e, Seq& s, int pos0, const int* Ts, int nc) {
  if (nc < 2 || nc > prefill_group_) fatal("prefill group of %d chunks (NI_PREFILL_GROUP=%d)", nc, prefill_group_);
  if (remote_ || lo_ != 0 || dbg_layers_ >= 0) fatal("prefill group on a split or debug model");
  if (s.embeds || s.mask_rows) fatal("prefill group over image or grammar rows");
  int total = 0;
  for (int c = 0; c < nc; ++c) {
    if (Ts[c] <= 0 || Ts[c] > tmax_) fatal("prefill group chunk %d: %d rows (max %d)", c, Ts[c], tmax_);
    total += Ts[c];
  }
  if (pos0 + total >= s.cap) fatal("context full (%d)", s.cap);
  // the two-sided evidence: a pass logs either the chunker's "prefill chunk of ..." or this line, so an engine log
  // shows which path ran every chunk of the prompt (grep both; an absent line is as loud as a present one)
  logd("prefill group of %d chunks (%d rows) over layers %d..%d", nc, total, lo_, hi_ - 1);
  const GroupSave saved = group_save();
  in_group_ = true;
  // prologue: gather every chunk's n-gram rows (two sets of staging alternate by group parity: the host gathers
  // group g + 1's rows while the GPU runs g, and the engine waits g - 1 before g + 1 starts) and encode each
  // chunk's embed + expand into its own slots
  group_parity_ ^= 1;
  const size_t set = size_t(group_parity_ & 1) * size_t(prefill_group_);
  Buffer grow[8];
  int pos = pos0;
  for (int c = 0; c < nc; ++c) {
    if (c4_.ple_layer >= 0) {
      grow[c] = ple_pfg_[set + size_t(c)];
      gather_rows(s.tokens(), pos, Ts[c], grow[c].ptr(), true);
    }
    group_swap(c);
    enc_embed(e, s.tok, pos, Ts[c], yb_);
    enc_expand(e, Ts[c]);
    pos += Ts[c];
  }
  // visits: layer i over the chunks in position order (the GDN recurrences and the KV advance in place), then
  // layer i + 1. A group is never a pass's last chunk, so the group's final turn wraps to layer 0 (seam_more_)
  // for the chunk after the group; every other turn keeps the successor table above.
  for (int i = lo_; i < hi_; ++i) {
    int cpos = pos0;
    for (int c = 0; c < nc; ++c) {
      group_inner_ = c + 1 < nc;
      seam_more_ = !group_inner_ && i + 1 == hi_;
      group_floor_ = seam_more_ ? -1 : i - lo_;  // #624: this visit's keys stay out of its sweep's victim set
      group_swap(c);
      enc_prefill_layer(e, s, i, cpos, Ts[c], grow[c], nullptr, 0);
      cpos += Ts[c];
    }
  }
  seam_more_ = false;
  group_floor_ = -1;
  group_inner_ = false;
  in_group_ = false;
  // the chunk path's tail: every row's pre-mix state, injected per chunk from its own slots (its last visit left
  // the final layer's block output in its own yb_), before the MTP fill reads them
  for (int c = 0; c < nc; ++c) {
    group_swap(c);
    enc_inject(e, Ts[c], x_, injM_);
  }
  group_restore(saved);
  if (comb_pending_.active) fatal("hc_precomb: a pending combine outlived a prefill group");
}

// ---- a model split across Macs by layers ----

// The head's hop: the pass so far is committed with a ready signal at its end; the remote waits for it on the CPU, sends
// the rows' state to the other Macs and writes their result back, and only then is the rest of the pass encoded. The
// GPU never waits for the network (a command buffer waiting on an event for ~5 s is killed: remote.h).
void Qwen4Exp::enc_hop(Encoder& e, Remote::Kind k, Seq& s, int pos, int R, const Checkpoint* snap, int snap_row) {
  if (!remote_) return;
  if (s.pend_tree) fatal("a split model verifies no trees");
  Remote::Pass p;
  p.kind = k;
  p.seq = s.rid;
  p.pos = pos;
  p.rows = R;
  p.n_pend = s.n_pend;
  p.snap = snap ? snap->rid : -1;
  p.snap_row = snap_row;
  Remote::Rows r;
  r.x = x_;
  r.y = yb_;
  r.inj = injM_;
  r.x_bytes = hidden_row_bytes();
  r.y_bytes = size_t(cfg_.hidden) * 4;
  r.inj_bytes = size_t(c4_.hc) * 4;
  r.event = hop_ev_;
  r.ready = ++hop_v_;
  e.signal_event(hop_ev_, r.ready);
  remote_->pass(p, r);  // (false: the Macs are lost, the rows garbage; the engine ends the request: Model::lost())
}

void Qwen4Exp::encode_stage(Encoder& e, Seq& s, const Remote::Pass& p, Checkpoint* snap) {
  if (lo_ == 0) fatal("encode_stage: the head runs whole passes");
  if (p.pos + p.rows >= s.cap) fatal("context full (%d)", s.cap);
  s.n_pend = p.n_pend;
  switch (p.kind) {
    case Remote::Kind::step:
      enc_step_layers(e, s, p.pos, PleWait{});
      break;
    case Remote::Kind::verify:
      enc_verify_layers(e, s, p.pos, p.rows, Buffer{}, 0);
      s.n_pend = 0;  // folded by this verify, as on the head; the head's next message says how many it kept
      break;
    case Remote::Kind::prefill:
      if (p.rows > tmax_) fatal("prefill chunk %d > %d", p.rows, tmax_);
      // seam_more_ stays false here: a Remote::Pass carries no chunk end, so a split model's tail never wraps a
      // seam (q20_fetch); its last chunk's layer 0 misses as it did before the wrap
      enc_prefill_layers(e, s, p.pos, p.rows, Buffer{}, snap, p.snap_row);
      if (comb_pending_.active) fatal("hc_precomb: a pending combine outlived the stage's prefill layers");
      if (hi_ == cfg_.layers) enc_inject(e, p.rows, x_, injM_);
      if (snap) snap->pos = p.pos + p.snap_row;
      break;
  }
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
    else if (q)
      bytes[q.exl3 ? q.name() : format_name(q.bits, q.gs)] +=
          q.w->buf.size + (q.dense() ? 0 : q.s->buf.size + q.b->buf.size);
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

std::vector<Qwen4Exp::LayerBytes> Qwen4Exp::layer_bytes(const std::string& dir, LayerBytes* head) {
  const Qwen35Config c = read_config(dir);
  const double share = c.experts ? double(c.top_k) / c.experts : 1.0;
  std::vector<LayerBytes> out(size_t(c.layers));
  for (int i = 0; i < c.layers; ++i) out[size_t(i)].linear = c.is_linear[size_t(i)];
  *head = LayerBytes{};
  const bool mtp = !getenv("NI_NO_MTP");
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      const std::string f = e->d_name;
      if (f.size() <= 12 || f.compare(f.size() - 12, 12, ".safetensors") != 0 || f == "ple.safetensors") continue;
      std::ifstream in(dir + "/" + f, std::ios::binary);
      uint64_t n = 0;
      if (!in.read(reinterpret_cast<char*>(&n), 8) || n > (256u << 20)) continue;
      std::string h(n, '\0');
      if (!in.read(h.data(), std::streamsize(n))) continue;
      const json j = json::parse(h, nullptr, false);
      if (!j.is_object()) continue;
      for (auto& [name, v] : j.items()) {
        if (name == "__metadata__" || name.find("vision_tower") != std::string::npos ||
            name.find("visual.") != std::string::npos || name.find(".ngram_embedding.") != std::string::npos ||
            (!mtp && is_mtp(name)))
          continue;
        const double b = double(v["data_offsets"][1].get<size_t>() - v["data_offsets"][0].get<size_t>());
        const int l = trunk_layer(name);
        LayerBytes& t = l >= 0 && l < c.layers ? out[size_t(l)] : *head;
        t.weights += b;
        if (is_trunk_expert(name)) t.experts += b;
        if (name.find("embed_tokens") == std::string::npos && !is_mtp(name))
          t.decode += name.find("switch_mlp") != std::string::npos ? b * share : b;
      }
    }
    closedir(d);
  }
  return out;
}

double Qwen4Exp::store_layer_bytes(const std::string& dir) {
  const Qwen35Config c = read_config(dir);
  return double(c.experts) * double(q20::slot_layout(uint32_t(c.hidden), uint32_t(c.moe_I)).bytes);
}

MemPlan Qwen4Exp::plan(const std::string& dir, int chunk, const Split* split) {
  MemPlan m;
  const json root = read_json(dir + "/config.json");
  Qwen35Config c = read_config(dir);
  Qwen4ExpConfig c4;
  parse4(root, c4);
  // the process's share of a split model: its layers' tensors, the head's own ones on the head
  const Split& sp = split ? *split : process_split();
  const int lo = sp.lo, hi = sp.hi < 0 ? c.layers : std::min(sp.hi, c.layers);
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
      if (!split_holds(name, lo, hi)) continue;
      const size_t al = (b + 255) / 256 * 256;
      if (is_mtp(name)) {
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
  if (m.streamed) {
    const size_t slot = q20::slot_layout(uint32_t(c.hidden), uint32_t(c.moe_I)).bytes;
    m.store_least = size_t(std::min(3, hi - lo)) * c.experts * slot;
    m.store_all = size_t(hi - lo) * c.experts * slot;
    m.weights += m.store_least;
  }
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
    for (int i = lo; i < hi; ++i) {
      p += c.is_linear[size_t(i)] ? H * (2 * kd + vd) + H * vd + 2 * H * c.lin_hv + vd * H
                                  : H * 2 * qd + 2 * H * kvd + qd * H + H * (c4.ix_heads + 1) * c4.ix_dim;
      p += double(c.top_k) * 3 * H * c.moe_I + 3 * H * c.shared_I + H * c.experts + H + 2 * gr;
    }
    m.decode_params = size_t(p + (lo == 0 ? double(c.vocab) * H : 0.0));
  }
  // buffers: Qwen35's for this config (x_ widened, the MTP rows) plus the hyper-connection / indexer / n-gram ones
  for (const ActBuf& a : act_layout(c, chunk, kMaxVerify, m.mtp > 0, size_t(kMaxVerify))) m.activations += a.bytes;
  m.activations += qwen4_act_bytes(c, c4, chunk, kMaxVerify, m.mtp > 0, 1);
  // the layer-major groups' slot term (q20::group_slot_bytes, the allocation's one formula): the plan sizes it
  // while a group can form at all - the experts stream (there is a store to pay for) and this process holds the
  // trunk from its first layer - but keeps it OUT of `activations`. Whether anything ever pays it is the
  // store-world choice (q20::group_world): the store is sized WITHOUT the term first, and a store complete
  // without it never groups and never allocates it (init_store), so no fit number may carry it there; an
  // incomplete store's world charges it, and the pre-load fit numbers that must carry the reservation add the
  // predicted charge (ni::planned_group_world: the engine's fit checks, start_need, doctor.cpp).
  m.group_slots = m.streamed && !(sp.remote || sp.lo != 0)
                      ? q20::group_slot_bytes(prefill_group_env(), size_t(chunk), size_t(c.hidden), size_t(c4.hc))
                      : 0;
  const size_t C = 2 * size_t(c.lin_hk) * c.lin_dk + size_t(c.lin_hv) * c.lin_dv;
  int lin = 0, att = 0;
  for (int i = lo; i < hi; ++i) (c.is_linear[size_t(i)] ? lin : att) += 1;
  const size_t pconv = size_t(kRing) * c4.hc * c.hidden * 4;
  const size_t state = size_t(lin) * (3 * C * 4 + size_t(c.lin_hv) * c.lin_dv * c.lin_dk * 4) + pconv;
  const size_t tape = size_t(lin) * kMaxVerify * (C + size_t(c.lin_hv) * c.lin_dk + c.lin_hv + size_t(c.lin_hv) * c.lin_dv) * 4;
  m.slot_state = 3 * state + tape;
  m.kv_per_pos = size_t(att + (m.mtp ? 1 : 0)) *
                 (size_t(2) * c.hkv * c.head_dim * 2 + size_t(c4.ix_dim) * 2 + size_t(c4.ix_dim) * 4 / c4.ix_ratio);
  return m;
}

}  // namespace ni
