#include "model/hy4_flash.h"

#include "model/hy4_engram.h"
#include "model/hy4_kernels.h"
#include "model/kparams.h"
#include "model/moe_a4.h"
#include "model/sampler.h"
#include "util/json_file.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>

namespace ni {

using json = nlohmann::json;

namespace {

constexpr uint32_t kS = hy4::kStreams;
constexpr int kHD = 256, kNope = 192, kRope = 64, kVD = 256, kKvb = kNope + kVD;

const json& text_cfg(const json& root) { return root.contains("text_config") ? root["text_config"] : root; }

std::string fname(int i) { return "model.layers." + std::to_string(i) + "."; }

// NI_HY4_DENSE=1: every full-context layer attends to every key, past kIndexDense too (a debugging switch)
bool dense_attention() {
  static const bool on = [] {
    const char* v = getenv("NI_HY4_DENSE");
    return v && v[0] == '1';
  }();
  return on;
}

}  // namespace

bool Hy4Flash::is_model(const std::string& dir) {
  const auto rp = json_file(dir + "/config.json");
  if (!rp) return false;
  const json& r = *rp;
  if (!r.is_object()) return false;
  if (r.value("model_type", std::string()) == "hy_v4_flash") return true;
  return r.contains("text_config") && r["text_config"].is_object() &&
         r["text_config"].value("model_type", std::string()) == "hy_v4_flash";
}

Hy4Flash::Hy4Flash(const std::string& dir, int prefill_chunk) : Qwen35(dir, prefill_chunk, NoLoad{}) { load(dir); }

Hy4Flash::~Hy4Flash() {
  if (eng_ev_) CFRelease(eng_ev_);
}

MemPlan Hy4Flash::plan(const std::string& dir, int chunk) {
  // the trunk's files as they are on disk, the routed experts and Engram tables left out (streamed / mapped)
  MemPlan m;
  const auto rp = json_file(dir + "/model.safetensors.index.json");
  if (rp && rp->contains("weight_map")) {
    std::map<std::string, bool> files;
    for (auto it = (*rp)["weight_map"].begin(); it != (*rp)["weight_map"].end(); ++it) {
      const std::string f = it.value().get<std::string>();
      files[f] = f.rfind("model-experts", 0) == 0 || f.rfind("model-engram", 0) == 0 || f == "model-vit.safetensors" ||
                 f == "model-dfly.safetensors";
    }
    for (const auto& [f, skip] : files) {
      std::ifstream in(dir + "/" + f, std::ios::binary | std::ios::ate);
      const size_t n = in ? size_t(in.tellg()) : 0;
      if (skip) {
        if (f.rfind("model-experts", 0) == 0) m.streamed += n;
      } else {
        m.weights += n;
      }
    }
  }
  // the store's slots: every routed expert in the largest layer's format (config.json's entries for the switch_mlp
  // modules, else its default; a4::Store reads the same formats from the shapes and checks them against these)
  std::map<int, std::pair<a4::Fmt, a4::Fmt>> fmts;  // MoE layer -> gate / up, down
  const auto cp = json_file(dir + "/config.json");
  const json q = cp && cp->contains("quantization") && (*cp)["quantization"].is_object() ? (*cp)["quantization"]
                                                                                           : json::object();
  auto fmt_of = [&](const std::string& module) {
    const json& e = q.contains(module) && q[module].is_object() ? q[module] : q;
    return a4::Fmt{e.value("bits", 4u), e.value("group_size", 64u)};
  };
  if (rp && rp->contains("weight_map"))
    for (auto it = (*rp)["weight_map"].begin(); it != (*rp)["weight_map"].end(); ++it) {
      const std::string& n = it.key();
      const size_t at = n.find(".mlp.switch_mlp.gate_proj.weight");
      if (n.rfind("model.layers.", 0) != 0 || at == std::string::npos) continue;
      const std::string base = n.substr(0, at) + ".mlp.switch_mlp.";
      fmts[std::atoi(n.c_str() + 13)] = {fmt_of(base + "gate_proj"), fmt_of(base + "down_proj")};
    }
  const a4::Shape sh{};
  size_t slot = 0;
  std::map<std::string, int> names;
  for (const auto& [l, f] : fmts) {
    slot = std::max<size_t>(slot, a4::slot_layout(sh, f.first, f.second).bytes);
    ++names[std::to_string(f.first.bits) + (f.first == f.second ? "" : "/" + std::to_string(f.second.bits))];
  }
  if (slot) {
    m.store_all = slot * sh.E * fmts.size();
    m.store_least = slot * sh.E * 3;
  } else {
    m.store_all = m.streamed;
    m.store_least = m.streamed / 41 * 3;
  }
  m.weights += m.store_least;  // as Qwen4Exp::plan: the store's least is loaded memory, the rest what the budget leaves
  // per slot: the full layers' latent and indexer rows grow with the context (plus the indexer's scratch, a score per
  // group of 4 positions for each of a chunk's rows); the sliding layers' latent rings are fixed (window + chunk rows)
  const json& t = cp && cp->is_object() ? text_cfg(*cp) : json::object();
  int full = 0, sliding = 0;
  if (t.contains("layer_types") && t["layer_types"].is_array())
    for (const auto& v : t["layer_types"]) ++(v.is_string() && v.get<std::string>() == "sliding_attention" ? sliding : full);
  m.kv_per_pos = size_t(full) * (hy4::kLatentRowBytes + kIxRowBytes) + size_t(chunk);
  m.slot_state = size_t(sliding) * size_t(t.value("sliding_window", 512) + chunk) * hy4::kLatentRowBytes;
  m.hidden = t.value("hidden_size", 3072);
  m.layers = t.value("num_hidden_layers", 42);
  m.activations = size_t(chunk) * (3072 * 4 * 12 + 16384 * 4 * 4 + 28672 * 4 + 13824 * 8 + 120832 * 4);
  std::string bits;
  for (const auto& [b, n] : names) bits += (bits.empty() ? "" : "+") + b;
  m.formats = (bits.empty() ? std::string("4") : bits) + "-bit experts + 8-bit trunk";
  return m;
}

void Hy4Flash::load(const std::string& dir) {
  const double t0 = now_ms();
  std::ifstream cf(dir + "/config.json");
  if (!cf) fatal("cannot read %s/config.json", dir.c_str());
  const json root = json::parse(cf);
  const json& t = text_cfg(root);
  load_quant_config(dir);
  cfg_.hidden = t.at("hidden_size").get<int>();
  cfg_.layers = t.at("num_hidden_layers").get<int>();
  // NI_HY4_LAYERS=n (debug): only the first n layers (the head after them); for runs before a pack is complete
  if (const char* v = getenv("NI_HY4_LAYERS"); v && atoi(v) > 0 && atoi(v) < cfg_.layers) {
    cfg_.layers = atoi(v);
    logi("NI_HY4_LAYERS: the first %d layers only (outputs are not the model's)", cfg_.layers);
  }
  cfg_.vocab = t.at("vocab_size").get<int>();
  cfg_.is_linear.assign(size_t(cfg_.layers), false);
  cfg_.hq = cfg_.hkv = t.value("swa_num_attention_heads", 64);
  cfg_.head_dim = kHD;
  cfg_.rot_dims = kRope;
  cfg_.eps = t.value("rms_norm_eps", 1e-5f);
  cfg_.moe = false;  // Qwen35's MoE buffers: the routed experts are this class's
  cfg_.dense_I = t.value("intermediate_size", 13824);
  cfg_.hc = int(kS);
  {
    // end of sequence: generation_config.json's, else the config's
    std::ifstream gf(dir + "/generation_config.json");
    json g = gf ? json::parse(gf) : json::object();
    const json& src = g.contains("eos_token_id") ? g["eos_token_id"] : t.contains("eos_token_id") ? t["eos_token_id"] : root.value("eos_token_id", json());
    if (src.is_array())
      for (const auto& v : src) cfg_.eos.push_back(v.get<int>());
    else if (src.is_number())
      cfg_.eos.push_back(src.get<int>());
  }
  lo_ = 0;
  hi_ = cfg_.layers;
  const int D = cfg_.hidden;
  if (D != 3072) fatal("hy4: hidden %d (expected 3072)", D);
  if (t.value("qk_nope_head_dim", 0) != kNope || t.value("qk_rope_head_dim", 0) != kRope || t.value("v_head_dim", 0) != kVD ||
      t.value("swa_qk_nope_head_dim", 0) != kNope || t.value("swa_v_head_dim", 0) != kVD)
    fatal("hy4: unexpected attention head dims");
  hcc_.D = uint32_t(D);
  hcc_.rms_eps = cfg_.eps;
  hcc_.hc_eps = t.value("hc_eps", 1e-6f);
  hcc_.mag = t.value("hc_magnitude", 2.0f);
  if (t.value("hc_mult", 4) != int(kS)) fatal("hy4: hc_mult %d", t.value("hc_mult", 4));

  // every tensor but the routed experts (moe_a4's store reads their files), the Engram tables (mapped by EngramCpu),
  // the vision tower and the DFly drafter
  ws_.open_dir(
      dir,
      [](const std::string& n) {
        return n.rfind("vit.", 0) != 0 && n.find("dfly") == std::string::npos && n.find(".switch_mlp.") == std::string::npos &&
               n.find(".multi_head_embedding.") == std::string::npos && n.find(".lookup_table") == std::string::npos;
      },
      [](const std::string& f) {
        return f.rfind("model-experts", 0) != 0 && f.rfind("model-engram", 0) != 0 && f != "model-vit.safetensors" &&
               f != "model-dfly.safetensors";
      });
  prefix_ = "";
  embed_ = qlin("model.embed_tokens");
  lm_head_ = qlin("lm_head");
  final_norm_ = &ws_.get("model.norm.weight");
  hc_head_ = hy4::Hc{&ws_.get("model.hc_head.hc_head_fn"), &ws_.get("model.hc_head.hc_head_base"),
                     &ws_.get("model.hc_head.hc_head_scale"), kS};

  const json& lt = t.at("layer_types");
  const std::vector<int> eng_ids = t.value("engram_layer_ids", std::vector<int>{});
  const float th_sl = t["rope_parameters"]["sliding_attention"].value("rope_theta", 1e4f);
  const json& rp = t["rope_parameters"];
  const json& rs = rp.contains("deepseek_sparse_attention") ? rp["deepseek_sparse_attention"] : rp["indexed_attention"];
  const float th_sp = rs.value("rope_theta", 1e7f);
  const uint32_t window = t.value("sliding_window", 512);
  hcc_window_ = int(window);
  const int dense_layers = t.value("first_k_dense_replace", 1);
  route_scale_ = t.value("routed_scaling_factor", 3.998f);

  // a row range of a matrix as its own QLinear (kv_a: the latent rows and the k_rope rows)
  auto rows = [&](const QLinear& L, uint32_t r0, uint32_t n) {
    QLinear q = L;
    auto view = [&](const Tensor* src, size_t row_bytes) {
      Tensor v = *src;
      v.buf = src->buf.view(size_t(r0) * row_bytes, size_t(n) * row_bytes);
      v.shape[0] = n;
      split_.push_back(v);
      return &split_.back();
    };
    const size_t wrow = L.row_bytes();
    q.w = view(L.w, wrow);
    if (!L.dense()) {
      q.s = view(L.s, L.group_row_bytes());
      q.b = view(L.b, L.group_row_bytes());
    } else {
      q.s = q.b = q.w;
    }
    q.N = n;
    return q;
  };

  hl_.assign(size_t(cfg_.layers), Hy4Layer{});
  for (int i = 0; i < cfg_.layers; ++i) {
    Hy4Layer& L = hl_[size_t(i)];
    const std::string p = fname(i);
    const std::string type = lt.at(size_t(i)).get<std::string>();
    L.sliding = type == "sliding_attention";
    L.in_norm = &ws_.get(p + "input_layernorm.weight");
    L.post_norm = &ws_.get(p + "post_attention_layernorm.weight");
    L.hca = hy4::Hc{&ws_.get(p + "hc_attn_layer.hc_pre.hc_fn"), &ws_.get(p + "hc_attn_layer.hc_pre.hc_base"),
                    &ws_.get(p + "hc_attn_layer.hc_pre.hc_scale"), 2 * kS};
    L.hcm = hy4::Hc{&ws_.get(p + "hc_mlp_layer.hc_pre.hc_fn"), &ws_.get(p + "hc_mlp_layer.hc_pre.hc_base"),
                    &ws_.get(p + "hc_mlp_layer.hc_pre.hc_scale"), 2 * kS};
    const std::string a = p + "self_attn.";
    L.q_a = qlin(a + "q_a_proj");
    L.q_b = qlin(a + "q_b_proj");
    const QLinear kva = qlin(a + "kv_a_proj_with_mqa");
    if (kva.N != 512 + kRope) fatal("%s: kv_a_proj_with_mqa has %u rows", a.c_str(), kva.N);
    L.kv_a_lat = rows(kva, 0, 512);
    L.kv_a_rope = rows(kva, 512, kRope);
    L.kv_b = qlin(a + "kv_b_proj");
    L.gate = qlin(a + "linear_gate");
    L.o = qlin(a + "o_proj");
    L.q_a_norm = &ws_.get(a + "q_a_layernorm.weight");
    L.kv_a_norm = &ws_.get(a + "kv_a_layernorm.weight");
    L.post_proj_norm = &ws_.get(a + "post_proj_norm.weight");
    L.at.H = L.q_b.N / kHD;
    L.at.theta = L.sliding ? th_sl : th_sp;
    L.at.window = L.sliding ? window : 0;
    L.at.sinks = &ws_.get(a + "learnable_sink_param");
    if (!L.sliding) {
      const std::string x = a + "compressed_indexer.";
      L.ix = true;
      L.ix_tk = qlin(x + "token_key_projection");
      L.ix_hw = qlin(x + "head_weight_projection");
      L.ix_q = qlin(x + "query_projection");
      L.ixw = hy4::IndexerWeights{&ws_.get(x + "token_key_norm.weight"), &ws_.get(x + "key_value_projection.weight"),
                                  &ws_.get(x + "key_gate_projection.weight"), &ws_.get(x + "intra_group_ape"),
                                  &ws_.get(x + "key_norm.weight")};
      if (L.ix_tk.N != 128 || L.ix_hw.N != 16 || L.ix_q.N != 16 * 128)
        fatal("%s: indexer shapes (token keys %u, head weights %u, queries %u)", x.c_str(), L.ix_tk.N, L.ix_hw.N,
              L.ix_q.N);
    }
    if (L.kv_b.N != L.at.H * kKvb || L.gate.N != L.at.H * kHD || L.o.K != L.at.H * kHD)
      fatal("%s: head shapes (q_b %u, kv_b %u, gate %u, o %u)", a.c_str(), L.q_b.N, L.kv_b.N, L.gate.N, L.o.K);
    const std::string m = p + "mlp.";
    if (i < dense_layers) {
      L.mg = qlin(m + "gate_proj");
      L.mu = qlin(m + "up_proj");
      L.md = qlin(m + "down_proj");
      L.post_fc2_norm = &ws_.get(m + "post_fc2_norm.weight");
    } else {
      L.moe = true;
      moe_layers_.push_back(i);
      L.router = qlin(m + "gate");
      L.router.gate = true;
      L.router_bias = &ws_.get(m + "gate.e_score_correction_bias");
      L.sg = qlin(m + "shared_experts.gate_proj");
      L.su = qlin(m + "shared_experts.up_proj");
      L.sd = qlin(m + "shared_experts.down_proj");
      L.lat_down = qlin(m + "routed_expert_down_proj");
      L.lat_up = qlin(m + "routed_expert_up_proj");
      L.combine_norm = &ws_.get(m + "combine_output_norm.weight");
    }
    if (std::find(eng_ids.begin(), eng_ids.end(), i + 1) != eng_ids.end()) {
      L.engram = int(eng_.size());
      eng_.emplace_back();
      Hy4Engram& E = eng_.back();
      E.layer = i;
      const std::string g = p + "engram_layer.";
      E.value = qlin(g + "value_proj");
      E.w.vb = &ws_.get(g + "value_proj.bias");
      E.w.pre_gamma = &ws_.get(g + "unified_pre_gamma");
      for (uint32_t s = 0; s < kS; ++s) {
        const std::string ss = std::to_string(s);
        E.key[s] = qlin(g + "key_projs." + ss);
        E.w.kb[s] = &ws_.get(g + "key_projs." + ss + ".bias");
        E.w.nk[s] = &ws_.get(g + "norm_key." + ss + ".weight");
        E.w.nq[s] = &ws_.get(g + "norm_query." + ss + ".weight");
        E.w.hc_scale[s] = &ws_.get(g + "unified_per_hc_scales." + ss);
      }
      E.cpu = std::make_unique<hy4::EngramCpu>(dir, t, i + 1);
    }
  }
  const size_t eng_want = size_t(std::count_if(eng_ids.begin(), eng_ids.end(), [&](int id) { return id - 1 < cfg_.layers; }));
  if (eng_.size() != eng_want) fatal("hy4: %zu Engram layers loaded, the config lists %zu", eng_.size(), eng_want);

  Gpu::get().make_resident(ws_.buffers());
  alloc_state();
  alloc_hy();
  {
    a4::Shape sh;
    sh.D = uint32_t(t.value("routed_expert_hidden_size", 1536));
    sh.I = uint32_t(t.value("moe_intermediate_size", 1536));
    sh.k = uint32_t(t.value("num_experts_per_tok", 16));
    sh.E = uint32_t(t.value("num_local_experts", t.value("n_routed_experts", 512)));
    sh.limit = t.value("swiglu_limit", 10.0f);
    store_ = std::make_unique<a4::Store>(dir, moe_layers_, sh, 0);
    Gpu::get().make_resident(store_->buffers());
  }
  eng_ev_ = Gpu::get().new_shared_event();
  prewarm_pipelines();
  for (const std::string& n : hy4::kernel_names()) hy4::pipeline(n, false);
  for (const std::string& n : store_->kernel_names()) hy4::pipeline(n, false);
  {  // the verify's rows (speculative decoding)
    Gpu& g = Gpu::get();
    const size_t D = size_t(cfg_.hidden), V = size_t(cfg_.vocab);
    htgt_ = g.alloc(D * 4, "hy_htgt");
    hn_ = g.alloc(size_t(kSpecRows) * D * 4, "hy_hn");
    vlog_ = g.alloc(size_t(kSpecRows) * V * 4, "hy_vlog");
    samp_ = g.alloc(16 * 4, "hy_samp");
    acc_ = g.alloc(16, "hy_acc");
    g.make_resident({htgt_, hn_, vlog_, samp_, acc_});
    hy4::pipeline("hy4_pick_row", false);
    hy4::pipeline("hy4_accept", false);
  }
  // the DFly2 drafter (model-dfly.safetensors): needs the whole trunk; NI_HY4_DFLY=0 leaves it out
  const char* dv = getenv("NI_HY4_DFLY");
  if (Hy4Dfly::present(dir) && !(dv && dv[0] == '0') && cfg_.layers == t.at("num_hidden_layers").get<int>()) {
    const Hy4DflyConfig dc = Hy4DflyConfig::from_json(dir + "/config.json");
    for (int L : dc.kv_layers)
      if (L < 0 || L >= cfg_.layers || hl_[size_t(L)].at.window != 0)
        fatal("hy4 dfly: target layer %d is not a full-context layer", L);
    dfly_kv_ = dc.kv_layers;
    for (size_t k = 0; k < dfly_kv_.size(); ++k) {
      Hy4Layer& T = hl_[size_t(dfly_kv_[k])];
      if (!T.ix) fatal("hy4 dfly: target layer %d has no indexer", dfly_kv_[k]);
      T.dfly = int(k);
      dsel_.push_back(Gpu::get().alloc(hy4::kIndexGroups * 4, "hy_dsel"));
      dsel_rows_.push_back(Gpu::get().alloc(size_t(kSpecRows) * hy4::kIndexGroups * 4, "hy_dsel_rows"));
    }
    Gpu::get().make_resident(dsel_);
    Gpu::get().make_resident(dsel_rows_);
    dfly_ = std::make_unique<Hy4Dfly>(dir, dc, embed_, lm_head_);
    Gpu::get().make_resident(dfly_->buffers());
    logi("DFly2 drafter: %d layers over target layers' caches, %.2f GB", int(dc.kv_layers.size()), dfly_->weight_bytes() / 1e9);
  }
  logi("loaded %s: Hy4-flash, %d layers, %.1f GB of resident weights in %.0f ms; %s; routed experts: %s", dir.c_str(),
       cfg_.layers, ws_.total_bytes() / 1e9, now_ms() - t0, formats_summary().c_str(), store_->describe().c_str());
}

std::string Hy4Flash::formats_summary() const {
  const Hy4Layer& L = hl_[1];
  return "trunk " + L.q_b.name() + ", kv_b " + L.kv_b.name() + ", router " + L.router.name() + ", head " +
         lm_head_.name();
}

void Hy4Flash::alloc_hy() {
  Gpu& g = Gpu::get();
  const size_t T = size_t(tmax_), D = size_t(cfg_.hidden);
  uint32_t Hmax = 0;
  for (const Hy4Layer& L : hl_) Hmax = std::max(Hmax, L.at.H);
  const size_t HD = size_t(Hmax) * kHD;
  const a4::Shape sh{};
  auto f32 = [&](size_t n, const char* label) { return g.alloc(std::max<size_t>(n, 1) * 4, label); };
  X_ = f32(T * kS * D, "hy_X");
  u_ = f32(T * D, "hy_u");
  xn_ = f32(T * D, "hy_xn");
  y_ = f32(T * D, "hy_y");
  post_a_ = f32(T * kS, "hy_post_a");
  post_m_ = f32(T * kS, "hy_post_m");
  xh_ = g.alloc(T * std::max<size_t>(HD, size_t(cfg_.dense_I)) * 2, "hy_xh");
  qa_ = f32(T * 1024, "hy_qa");
  kvl_ = f32(T * 512, "hy_kvl");
  kr_ = f32(T * kRope, "hy_kr");
  gt_ = f32(T * HD, "hy_gate");
  q_ = f32(T * HD, "hy_q");
  att_ = f32(T * HD, "hy_att");
  dg_ = f32(T * size_t(cfg_.dense_I), "hy_dg");
  du_ = f32(T * size_t(cfg_.dense_I), "hy_du");
  rl_ = f32(T * 512, "hy_router");
  idx_ = g.alloc(T * 16 * 4, "hy_idx");
  wts_ = f32(T * 16, "hy_wts");
  sgb_ = f32(T * 1536, "hy_sg");
  sub_ = f32(T * 1536, "hy_su");
  shb_ = f32(T * 1536, "hy_sh");
  ysb_ = f32(T * D, "hy_ys");
  lat_ = f32(T * 1536, "hy_lat");
  ylat_ = f32(T * 1536, "hy_ylat");
  moe_scratch_ = g.alloc(std::max<size_t>(a4::scratch_bytes(sh, int(T)), 16), "hy_moe_scratch");
  eng_v_ = f32(T * D, "hy_eng_v");
  eng_k_ = f32(T * kS * D, "hy_eng_k");
  const size_t ne = std::max<size_t>(eng_.size(), 1);
  for (Buffer& b : eng_pf_) b = f32(ne * T * 1536 + T, "hy_eng_pf");
  ix_tk_ = f32(T * 128, "hy_ix_tk");
  ix_hw_ = f32(T * 16, "hy_ix_hw");
  ix_q_ = f32(T * 16 * 128, "hy_ix_q");
  sel_ = g.alloc(T * hy4::kIndexGroups * 4, "hy_sel");
  for (Buffer& b : eng_dec_) b = f32(ne * 1536 + 1, "hy_eng_dec");
  owned_bufs_.insert(owned_bufs_.end(), {X_, u_, xn_, y_, post_a_, post_m_, xh_, qa_, kvl_, kr_, gt_, q_, att_, dg_,
                                         du_, rl_, idx_, wts_, sgb_, sub_, shb_, ysb_, lat_, ylat_, moe_scratch_, eng_v_,
                                         eng_k_, eng_pf_[0], eng_pf_[1], ix_tk_, ix_hw_, ix_q_, sel_});
  if (getenv("NI_HY4_DUMP_SEL")) {  // debug: room for every indexed layer's selection of one pass
    size_t nix = 0;
    for (const Hy4Layer& L : hl_) nix += L.ix ? 1 : 0;
    seldump_ = g.alloc(std::max<size_t>(nix, 1) * T * hy4::kIndexGroups * 4, "hy_seldump");
    owned_bufs_.push_back(seldump_);
  }
  Gpu::get().make_resident(owned_bufs_);
}

void Hy4Flash::write_selection() {
  if (!seldump_ || seldump_rec_.empty()) return;
  static FILE* f = [] {
    const char* path = getenv("NI_HY4_DUMP_SEL");
    FILE* fp = fopen(path, "wb");
    if (!fp) fatal("NI_HY4_DUMP_SEL: cannot write %s", path);
    return fp;
  }();
  const size_t row = hy4::kIndexGroups;
  for (size_t k = 0; k < seldump_rec_.size() / 3; ++k) {
    const int32_t hdr[3] = {seldump_rec_[3 * k], seldump_rec_[3 * k + 1], seldump_rec_[3 * k + 2]};
    fwrite(hdr, 4, 3, f);
    fwrite(seldump_.as<uint32_t>() + k * size_t(tmax_) * row, 4, size_t(hdr[2]) * row, f);
  }
  fflush(f);
  seldump_rec_.clear();
}

// ---- sequences ----

// Every layer caches the 576-wide MLA latent rows (hy4_ops.h, LatentCache) in s.kc[i]: the sliding layers' are rings of
// ring_cap() rows (allocated once), the full-context layers' grow linearly (the DFly2 drafter reads those).
uint32_t Hy4Flash::ring_cap() const { return uint32_t(hcc_window_ + tmax_); }

hy4::LatentCache Hy4Flash::lat_of(const Seq& s, int i) const {
  const bool ring = hl_[size_t(i)].at.window > 0;
  return hy4::LatentCache{s.kc[size_t(i)], ring ? lat_ring_cap_ : lat_cap_of(s), ring};
}

hy4::IndexerCache Hy4Flash::ix_of(const Seq& s, int i) const {
  if (size_t(i) >= s.ixk.size() || !s.ixk[size_t(i)]) return {};
  return hy4::IndexerCache{s.ixk[size_t(i)], s.ixb[size_t(i)], lat_cap_of(s)};
}

std::unique_ptr<Seq> Hy4Flash::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  s->kc.assign(size_t(cfg_.layers), Buffer{});
  s->vc.assign(size_t(cfg_.layers), Buffer{});
  s->ixk.assign(size_t(cfg_.layers), Buffer{});
  s->ixb.assign(size_t(cfg_.layers), Buffer{});
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0) grow_kv(*s, cap);
  reset(*s);
  return s;
}

size_t Hy4Flash::kv_bytes_per_pos() const {
  size_t n = 0;
  for (const Hy4Layer& L : hl_) n += L.at.window == 0;
  return n * (hy4::kLatentRowBytes + kIxRowBytes) + scratch_per_pos();
}

// The indexer's scratch holds a score per group of every position; the attention scratch grows with the context
// only under NI_HY4_DENSE (else a full layer's expand path attends to at most kIndexDense keys: grow_kv)
size_t Hy4Flash::scratch_per_pos() const {
  constexpr uint32_t a = 1u << 16, b = a + 4096;
  size_t n = hy4::index_scratch_bytes(tmax_, b) - hy4::index_scratch_bytes(tmax_, a);
  if (dense_attention()) {
    uint32_t Hmax = 0;
    for (const Hy4Layer& L : hl_) Hmax = std::max(Hmax, L.at.H);
    n += hy4::latent_scratch_bytes(Hmax, tmax_, b) - hy4::latent_scratch_bytes(Hmax, tmax_, a);
  }
  return (n + 4095) / 4096;
}

size_t Hy4Flash::kv_unit_bytes_per_pos() const { return hy4::kLatentRowBytes; }

uint32_t Hy4Flash::lat_cap_of(const Seq& s) const { return (uint32_t(s.cap) + 15) / 16 * 16; }

void Hy4Flash::grow_kv(Seq& s, int cap, size_t) {
  if (cap <= s.cap) return;
  Gpu& g = Gpu::get();
  std::vector<Buffer> res;
  const uint32_t newcap = (uint32_t(cap) + 15) / 16 * 16;
  for (int i = 0; i < cfg_.layers; ++i) {
    hy4::LatentCache c = lat_of(s, i);
    if (!c.rows) c = hy4::alloc_latent(c.ring ? ring_cap() : newcap, c.ring ? "hy_lat_ring" : "hy_lat", c.ring);
    else if (!c.ring) hy4::grow_latent(c, newcap, uint32_t(s.cap));
    if (c.ring) lat_ring_cap_ = c.cap;
    s.kc[size_t(i)] = c.rows;
    res.push_back(c.rows);
    if (hl_[size_t(i)].ix) {  // the indexer's token and group keys, linear as the layer's latent rows
      hy4::IndexerCache x = ix_of(s, i);
      if (!x.tok) x = hy4::alloc_indexer(newcap, "hy_ix");
      else hy4::grow_indexer(x, newcap, uint32_t(s.cap));
      s.ixk[size_t(i)] = x.tok;
      s.ixb[size_t(i)] = x.grp;
      res.push_back(x.tok);
      res.push_back(x.grp);
    }
  }
  s.cap = int(newcap);
  g.make_resident(res);
  // the attention scratch: a sliding layer's expand-from-cache prefill attends up to window + tmax keys, a full
  // layer's up to kIndexDense (past it the indexer's selection runs the absorbed path), or s.cap under NI_HY4_DENSE
  uint32_t Hmax = 0;
  for (const Hy4Layer& L : hl_) Hmax = std::max(Hmax, L.at.H);
  const uint32_t full = dense_attention() ? newcap : std::min<uint32_t>(newcap, hy4::kIndexDense);
  const size_t need = std::max<size_t>(
      hy4::latent_scratch_bytes(Hmax, tmax_, std::max<uint32_t>(full, uint32_t(hcc_window_ + tmax_))), 16);
  if (attn_scratch_.size < need) {
    if (attn_scratch_) g.release(attn_scratch_);
    attn_scratch_ = g.alloc(need, "hy_attn_scratch");
    g.make_resident({attn_scratch_});
  }
  const size_t ixneed = std::max<size_t>(hy4::index_scratch_bytes(tmax_, newcap), 16);
  if (ix_scratch_.size < ixneed) {
    if (ix_scratch_) g.release(ix_scratch_);
    ix_scratch_ = g.alloc(ixneed, "hy_ix_scratch");
    g.make_resident({ix_scratch_});
  }
}

void Hy4Flash::drop_kv(Seq& s) {
  for (int i = 0; i < int(s.kc.size()); ++i) {
    hy4::LatentCache c = lat_of(s, i);
    if (c.rows) hy4::free_latent(c);
    s.kc[size_t(i)] = Buffer{};
    if (size_t(i) < s.ixk.size() && s.ixk[size_t(i)]) {
      hy4::IndexerCache x = ix_of(s, i);
      hy4::free_indexer(x);
      s.ixk[size_t(i)] = s.ixb[size_t(i)] = Buffer{};
    }
  }
  s.cap = 0;
}

void Hy4Flash::reset(Seq& s) {
  if (pend_.s == &s) pend_ = Pending{};
}

// ---- encoders ----

void Hy4Flash::mm(Encoder& e, const QLinear& L, const Buffer& x, const Tensor* norm, const Buffer& y, int R) {
  if (R == 1) {
    enc_qmv(e, L, x, y, nullptr, norm);
    return;
  }
  enc_rows_half(e, x, norm, xh_, R, int(L.K));
  enc_qmm(e, L, xh_, y, R, false);
}

void Hy4Flash::gather_engram(const int* toks, int pos0, int R, const Buffer& dst) const {
  // per Engram layer k: rows fp32 [R, 1536] at dst + k * R * 1536 floats; then the mask, the same for every layer
  std::vector<uint8_t> mask(static_cast<size_t>(R));
  for (size_t k = 0; k < eng_.size(); ++k)
    eng_[k].cpu->rows(toks, pos0, R, dst.as<float>() + k * size_t(R) * 1536, mask.data());
  uint32_t* m = reinterpret_cast<uint32_t*>(dst.as<float>() + eng_.size() * size_t(R) * 1536);
  for (int r = 0; r < R; ++r) m[r] = mask[size_t(r)];
}

void Hy4Flash::engram_after_sample(Encoder& e, Seq& s, int pos) {
  if (eng_.empty()) return;
  const int slot = dec_next_;
  dec_next_ = (dec_next_ + 1) % 4;
  const uint64_t t = ++eng_ticket_;
  pend_ = Pending{&s, pos, t, slot};
  const Buffer dst = eng_dec_[slot];
  const Seq* sp = &s;
  void* ev = eng_ev_;
  e.on_completed([this, sp, pos, dst, t, ev] {
    gather_engram(const_cast<Seq*>(sp)->tokens(), pos, 1, dst);
    Gpu::signal_event_cpu(ev, t);
  });
}

void Hy4Flash::enc_engram_block(Encoder& e, int i, int R) {
  const Hy4Layer& L = hl_[size_t(i)];
  const Hy4Engram& E = eng_[size_t(L.engram)];
  const uint32_t D = uint32_t(cfg_.hidden);
  // the pending block output first: the Engram gate reads the streams
  if (pending_) {
    hy4::enc_hc_inject(e, D, R, X_, *pending_, *pending_post_);
    pending_ = pending_post_ = nullptr;
  }
  if (wait_ticket_) {
    e.wait_event(eng_ev_, wait_ticket_);
    wait_ticket_ = 0;
  }
  static const bool no_engram = [] {  // NI_HY4_NO_ENGRAM=1: skip the Engram memories (a debugging switch)
    const char* v = getenv("NI_HY4_NO_ENGRAM");
    return v && v[0] == '1';
  }();
  if (no_engram) return;
  const Buffer rows = eng_cur_.view(size_t(L.engram) * eng_stride_, size_t(R) * 1536 * 4);
  mm(e, E.value, rows, nullptr, eng_v_, R);
  for (uint32_t s = 0; s < kS; ++s)
    mm(e, E.key[s], rows, nullptr, eng_k_.view(size_t(s) * R * D * 4, size_t(R) * D * 4), R);
  const Buffer mask = eng_cur_.view(eng_.size() * eng_stride_, size_t(R) * 4);
  hy4::enc_engram(e, E.w, D, engram_eps_, R, eng_v_, eng_k_, X_, &mask);
}

void Hy4Flash::enc_attn_block(Encoder& e, Seq& s, int i, int pos0, int R, bool prefill) {
  const Hy4Layer& L = hl_[size_t(i)];
  const uint32_t D = uint32_t(cfg_.hidden);
  hy4::enc_hc_pre(e, L.hca, hcc_, R, X_, pending_, pending_post_, u_, false, &post_a_);
  pending_ = pending_post_ = nullptr;
  // projections of input_layernorm(u)
  mm(e, L.q_a, u_, L.in_norm, qa_, R);
  mm(e, L.kv_a_lat, u_, L.in_norm, kvl_, R);
  mm(e, L.kv_a_rope, u_, L.in_norm, kr_, R);
  mm(e, L.gate, u_, L.in_norm, gt_, R);
  mm(e, L.q_b, qa_, L.q_a_norm, q_, R);
  hy4::enc_rmsnorm(e, 512, R, kvl_, *L.kv_a_norm, cfg_.eps, kvl_);
  hy4::LatentCache c = lat_of(s, i);
  const uint32_t HD = L.at.H * kHD;
  // the compressed indexer: token keys every call; past kIndexDense keys, each row's selection of groups
  const Buffer* sel = nullptr;
  if (L.ix) {
    hy4::IndexerCache x = ix_of(s, i);
    mm(e, L.ix_tk, u_, L.in_norm, ix_tk_, R);
    hy4::enc_index_write(e, L.at, L.ixw, x, pos0, R, ix_tk_, 128, 0);
    if (pos0 + R > int(hy4::kIndexDense) && !dense_attention()) {
      mm(e, L.ix_hw, u_, L.in_norm, ix_hw_, R);
      mm(e, L.ix_q, qa_, L.q_a_norm, ix_q_, R);  // q_resid = q_a_layernorm(q_a)
      hy4::enc_index_select(e, L.at, x, pos0, R, ix_q_, 16 * 128, 0, ix_hw_, 16, 0, sel_, ix_scratch_);
      sel = &sel_;
      if (seldump_ && prefill) {  // NI_HY4_DUMP_SEL (debug): prefill passes (a 1-row tail too), not decode steps
        const size_t row = size_t(hy4::kIndexGroups) * 4, k = seldump_rec_.size() / 3;
        if ((k + 1) * size_t(tmax_) * row <= seldump_.size) {
          enc_copy(e, sel_.view(0, size_t(R) * row), seldump_.view(k * size_t(tmax_) * row, size_t(R) * row));
          seldump_rec_.insert(seldump_rec_.end(), {i, pos0, R});
        } else if (!seldump_full_) {  // nothing called write_selection() (kapitsa serve): stop recording
          seldump_full_ = true;
          logi("NI_HY4_DUMP_SEL: no write_selection() since the last pass; the selections are no longer recorded");
        }
      }
      // the drafter's: a verify's rows kept for encode_accept's pick; a prefill's or a step's last row now
      if (L.dfly >= 0) {
        const size_t k = size_t(L.dfly), n = size_t(R) * hy4::kIndexGroups * 4;
        if (!prefill && R > 1 && R <= kSpecRows) enc_copy(e, sel_.view(0, n), dsel_rows_[k].view(0, n));
        else hy4::enc_index_pick(e, sel_, R, nullptr, R - 1, dsel_[k]);
      }
    }
  }
  // prefill: the attention rows in half straight into o_proj's GEMM
  if (prefill && R > 1) {
    hy4::enc_attn_latent(e, L.at, c, pos0, R, q_, HD, kvl_, 512, kr_, kRope, 0, *L.kv_b.w, gt_, xh_, true, attn_scratch_,
                         sel);
    enc_qmm(e, L.o, xh_, y_, R, false);
  } else {
    hy4::enc_attn_latent(e, L.at, c, pos0, R, q_, HD, kvl_, 512, kr_, kRope, 0, *L.kv_b.w, gt_, att_, false, attn_scratch_,
                         sel);
    mm(e, L.o, att_, nullptr, y_, R);
  }
  hy4::enc_rmsnorm(e, D, R, y_, *L.post_proj_norm, cfg_.eps, y_);
  pending_ = &y_;
  pending_post_ = &post_a_;
}

void Hy4Flash::enc_mlp_block(Encoder& e, int i, int R, bool prefill) {
  const Hy4Layer& L = hl_[size_t(i)];
  const uint32_t D = uint32_t(cfg_.hidden);
  hy4::enc_hc_pre(e, L.hcm, hcc_, R, X_, pending_, pending_post_, u_, false, &post_m_);
  pending_ = pending_post_ = nullptr;
  if (!L.moe) {
    mm(e, L.mg, u_, L.post_norm, dg_, R);
    mm(e, L.mu, u_, L.post_norm, du_, R);
    hy4::enc_swiglu(e, L.mg.N, R, dg_, du_, dg_, 0.0f, false);
    mm(e, L.md, dg_, nullptr, y_, R);
    hy4::enc_rmsnorm(e, D, R, y_, *L.post_fc2_norm, cfg_.eps, y_);
  } else {
    hy4::enc_rmsnorm(e, D, R, u_, *L.post_norm, cfg_.eps, xn_);
    mm(e, L.router, xn_, nullptr, rl_, R);
    hy4::enc_router(e, L.router.N, 16, route_scale_, R, rl_, *L.router_bias, idx_, wts_);
    store_->begin(e, i, idx_, uint32_t(R) * 16, nullptr, 0, prefill);
    // work that does not need the experts overlaps the reads
    mm(e, L.sg, xn_, nullptr, sgb_, R);
    mm(e, L.su, xn_, nullptr, sub_, R);
    hy4::enc_swiglu(e, L.sg.N, R, sgb_, sub_, shb_, 0.0f, false);
    mm(e, L.sd, shb_, nullptr, ysb_, R);
    mm(e, L.lat_down, xn_, nullptr, lat_, R);
    store_->end(e);
    a4::Shape sh;
    sh.D = L.lat_down.N;
    sh.I = 1536;
    a4::enc_rows(e, store_->experts(i), sh, R, lat_, idx_, wts_, moe_scratch_, ylat_);
    mm(e, L.lat_up, ylat_, nullptr, y_, R);
    hy4::enc_add(e, D, R, ysb_, y_);
    hy4::enc_rmsnorm(e, D, R, y_, *L.combine_norm, cfg_.eps, y_);
  }
  pending_ = &y_;
  pending_post_ = &post_m_;
}

void Hy4Flash::enc_layers(Encoder& e, Seq& s, int pos0, int R, bool prefill) {
  pending_ = pending_post_ = nullptr;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (hl_[size_t(i)].engram >= 0) enc_engram_block(e, i, R);
    enc_attn_block(e, s, i, pos0, R, prefill);
    enc_mlp_block(e, i, R, prefill);
    if (prefill && R >= 256 && i + 1 < cfg_.layers) e.split();
  }
}

void Hy4Flash::enc_head(Encoder& e, int R, size_t row0) {
  const uint32_t D = uint32_t(cfg_.hidden);
  if (pending_) {
    hy4::enc_hc_inject(e, D, R, X_, *pending_, *pending_post_);
    pending_ = pending_post_ = nullptr;
  }
  const Buffer X = X_.view(row0 * kS * D * 4, size_t(R) * kS * D * 4);
  hy4::enc_hc_pre(e, hc_head_, hcc_, R, X, nullptr, nullptr, u_, false, nullptr);
}

void Hy4Flash::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  // the Engram rows of tokens[pos]: gathered by the previous pass's completion handler, or here
  if (pend_.s == &s && pend_.pos == pos && pend_.ticket) {
    eng_cur_ = eng_dec_[pend_.slot];
    wait_ticket_ = pend_.ticket;
    pend_ = Pending{};
  } else {
    eng_cur_ = eng_dec_[dec_next_];
    dec_next_ = (dec_next_ + 1) % 4;
    gather_engram(s.tokens(), pos, 1, eng_cur_);
    wait_ticket_ = 0;
  }
  eng_stride_ = 1536 * 4;
  enc_embed(e, s.tok, pos, 1, u_);
  hy4::enc_hc_expand(e, uint32_t(cfg_.hidden), 1, u_, X_);
  enc_layers(e, s, pos, 1, false);
  enc_head(e, 1, 0);
  if (!sample) return;
  if (dfly_) hy4::enc_rmsnorm(e, uint32_t(cfg_.hidden), 1, u_, *final_norm_, cfg_.eps, htgt_);
  enc_sample(e, s, u_, pos + 1, sp, step);
  engram_after_sample(e, s, pos + 1);
}

void Hy4Flash::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                              Checkpoint* snap, int) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  if (snap) fatal("hy4: in-pass snapshots are not supported (can_snapshot is false)");
  if (s.embeds) fatal("hy4: image inputs are not supported yet");
  eng_cur_ = eng_pf_[eng_pf_next_];
  eng_pf_next_ ^= 1;
  eng_stride_ = size_t(T) * 1536 * 4;
  wait_ticket_ = 0;
  gather_engram(s.tokens(), pos0, T, eng_cur_);
  enc_embed(e, s.tok, pos0, T, u_);
  hy4::enc_hc_expand(e, uint32_t(cfg_.hidden), T, u_, X_);
  enc_layers(e, s, pos0, T, true);
  // the streams of every row stay in X_ (encode_logits_rows); the pending output is folded in now
  if (pending_) {
    hy4::enc_hc_inject(e, uint32_t(cfg_.hidden), T, X_, *pending_, *pending_post_);
    pending_ = pending_post_ = nullptr;
  }
  if (sample) {
    enc_head(e, 1, size_t(T - 1));
    if (dfly_) hy4::enc_rmsnorm(e, uint32_t(cfg_.hidden), 1, u_, *final_norm_, cfg_.eps, htgt_);
    enc_sample(e, s, u_, pos0 + T, sp, step);
    engram_after_sample(e, s, pos0 + T);
  }
}

void Hy4Flash::encode_logits_rows(Encoder& e, int T, const Buffer& out) {
  enc_head(e, T, 0);
  enc_rows_half(e, u_, final_norm_, xh_, T, cfg_.hidden);
  enc_qmm(e, lm_head_, xh_, out, T, false);
}

// ---- checkpoints ----
Checkpoint Hy4Flash::new_checkpoint() {
  Checkpoint c;
  const size_t bytes = size_t((ring_cap() + 15) / 16 * 16) * hy4::kLatentRowBytes;
  for (const Hy4Layer& L : hl_)
    if (L.at.window > 0) c.conv.push_back(Gpu::get().alloc(bytes, "hy_ckpt_ring"));
  Gpu::get().make_resident(c.conv);
  return c;
}

void Hy4Flash::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  size_t k = 0;
  for (int i = 0; i < cfg_.layers; ++i)
    if (hl_[size_t(i)].at.window > 0) {
      if (s.kc[size_t(i)]) enc_copy(e, s.kc[size_t(i)], c.conv[k]);
      ++k;
    }
  c.pos = pos;
}

void Hy4Flash::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  size_t k = 0;
  for (int i = 0; i < cfg_.layers; ++i)
    if (hl_[size_t(i)].at.window > 0) {
      if (s.kc[size_t(i)]) enc_copy(e, c.conv[k], s.kc[size_t(i)]);
      ++k;
    }
}

bool Hy4Flash::experts_resident() const { return store_ && store_->complete(); }

// ---- DFly2 speculative decoding ----
void Hy4Flash::encode_draft(Encoder& e, Seq& s, int P) {
  if (!dfly_) fatal("hy4: no DFly2 drafter");
  if (P + kSpecRows >= s.cap) fatal("context full (%d)", s.cap);
  std::vector<hy4::LatentCache> cs;
  cs.reserve(dfly_kv_.size());
  for (int L : dfly_kv_) cs.push_back(lat_of(s, L));
  Hy4Dfly::In in;
  in.h = htgt_;
  in.t0 = s.tok.view(size_t(P) * 4, 4);
  in.a = P;
  for (hy4::LatentCache& c : cs) in.caches.push_back(&c);
  if (P > int(hy4::kIndexDense))
    for (const Buffer& b : dsel_) in.sel.push_back(&b);
  dfly_->draft(e, in, s.tok.view(size_t(P + 1) * 4, size_t(Hy4Dfly::kDraft) * 4));
}

void Hy4Flash::encode_verify(Encoder& e, Seq& s, int P, int R) {
  if (R < 1 || R > kSpecRows) fatal("hy4 verify of %d rows", R);
  if (P + R >= s.cap) fatal("context full (%d)", s.cap);
  eng_cur_ = eng_pf_[eng_pf_next_];
  eng_pf_next_ ^= 1;
  eng_stride_ = size_t(R) * 1536 * 4;
  wait_ticket_ = 0;
  pend_ = Pending{};
  gather_engram(s.tokens(), P, R, eng_cur_);
  enc_embed(e, s.tok, P, R, u_);
  hy4::enc_hc_expand(e, uint32_t(cfg_.hidden), R, u_, X_);
  enc_layers(e, s, P, R, false);
  enc_head(e, R, 0);
  hy4::enc_rmsnorm(e, uint32_t(cfg_.hidden), R, u_, *final_norm_, cfg_.eps, hn_);
  mm(e, lm_head_, hn_, nullptr, vlog_, R);
}

void Hy4Flash::enc_sample_row(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index,
                              const SamplingParams& sp, uint32_t step) {
  const uint32_t V = uint32_t(cfg_.vocab);
  if (full_vocab(sp)) {
    enc_sample_full(e, lrow, V, tokens, out_index, sprob_, sp, step);
    return;
  }
  const uint32_t K = topk_k(sp);
  const kp::TopkStage ts{V, K, topk_chunk(K)};
  const uint32_t ntg = topk_lists(V, K);
  e.use(Gpu::get().pipeline("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, topk_threads(K));
  const kp::Sample p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(Gpu::get().pipeline("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void Hy4Flash::encode_accept(Encoder& e, Seq& s, int P, int R, const SamplingParams& sp) {
  const size_t V = size_t(cfg_.vocab);
  // row r's token is position P+r+1's: plain decoding draws it with step P+r (encode_step's step = pos)
  for (int r = 0; r < R; ++r) enc_sample_row(e, vlog_.view(size_t(r) * V * 4, V * 4), samp_, r, sp, uint32_t(P + r));
  struct {
    uint32_t P, K;
  } ap{uint32_t(P), uint32_t(R - 1)};
  e.use(hy4::pipeline("hy4_accept"));
  e.buf(0, samp_);
  e.buf(1, s.tok);
  e.buf(2, acc_);
  e.val(3, ap);
  e.dispatch(1, 1, 1, 1);
  if (dfly_) {
    const uint32_t D = uint32_t(cfg_.hidden);
    if (P + R > int(hy4::kIndexDense))
      for (size_t k = 0; k < dsel_.size(); ++k) hy4::enc_index_pick(e, dsel_rows_[k], R, &acc_, 0, dsel_[k]);
    e.use(hy4::pipeline("hy4_pick_row"));
    e.buf(0, hn_);
    e.buf(1, acc_);
    e.buf(2, htgt_);
    e.val(3, D);
    e.dispatch((D + 255) / 256, 1, 1, 256);
  }
}

}  // namespace ni
