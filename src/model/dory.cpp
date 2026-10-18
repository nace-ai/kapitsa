#include "model/dory.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ni {

extern const char* const kNhKernelSource;
extern const char* const kDoryKernelSource;

namespace {

// host mirrors of the kernel parameter structs (src/kernels/dory/*.metal, src/kernels/nemotron/*.metal)
struct PrepP {
  uint32_t R, pos0, Hq, Hkv, rows, ring, base, q_ld, kv_ld;
};
struct AttnP {
  uint32_t R, pos0, Hq, Hkv, rows, ring, window, chunk, nchunks, k0;
  float scale;
};
struct AttnPfP {
  uint32_t T, pos0, Hq, Hkv, rows, ring, window;
  float scale;
};
struct RowP {
  uint32_t D, x_ld, y_ld, out_half;
};
struct SlerpP {
  uint32_t D, mode, write_half;
};
struct ScaleP {
  uint32_t V, ld;
};
struct StoreP {
  uint32_t T, r0, pos0, Hkv, rows, ring;
};
struct MvP {  // NhMvParams
  uint32_t N, K, x_stride, y_stride;
  float gs;
  uint32_t R;
};
struct GemmP {  // NhGemmP
  uint32_t M, N, K, x_ld, y_ld;
  float gs;
  uint32_t epi, q_estride, s_estride, k;
};
struct EmbedP {
  uint32_t D, tok_off;
};
struct TopkStageP {
  uint32_t V, K, chunk;
};
struct SampleP {
  uint32_t n, K;
  float temp, top_p;
  uint32_t seed, step, out_index;
};

Pipeline pipe(const std::string& n) {
  static std::mutex mu;
  static std::unordered_map<std::string, Pipeline> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) return {};
  try {
    return nlohmann::json::parse(f);
  } catch (...) {
    return {};
  }
}

uint32_t gpu_cores() {
  static const uint32_t c = uint32_t(std::max(1, Gpu::get().core_count()));
  return c;
}

const char* fmt_name(NhFmt f) {
  switch (f) {
    case NhFmt::Q8: return "Q8_0";
    case NhFmt::FP4: return "NVFP4";
    case NhFmt::IQ2: return "IQ2_XXS";
    case NhFmt::BF16: return "BF16";
    case NhFmt::F32: return "F32";
    case NhFmt::FP8: return "FP8";
  }
  return "?";
}

}  // namespace

namespace {
int serving_loops_ = 0;
LoopKv serving_kv_ = LoopKv::last_loop;
}  // namespace

void Dory::set_serving(int loops, LoopKv kv) {
  serving_loops_ = loops;
  serving_kv_ = kv;
}
int Dory::serving_loops() { return serving_loops_; }
LoopKv Dory::serving_kv() { return serving_kv_; }

const char* loop_kv_name(LoopKv m) { return m == LoopKv::per_loop ? "per-loop" : "last-loop"; }

bool parse_loop_kv(const std::string& s, LoopKv* out) {
  if (s == "last-loop" || s == "last_loop" || s == "last") return *out = LoopKv::last_loop, true;
  if (s == "per-loop" || s == "per_loop" || s == "per") return *out = LoopKv::per_loop, true;
  return false;
}

// ---- config ----
bool Dory::is_model(const std::string& dir) {
  struct stat st;
  if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
  const nlohmann::json j = read_json(dir + "/config.json");
  return j.is_object() && j.contains("model_type") && j["model_type"] == "dory";
}

DoryConfig Dory::read_config(const std::string& dir) {
  const nlohmann::json t = read_json(dir + "/config.json");
  if (!t.is_object() || !t.contains("model_type") || t["model_type"] != "dory")
    fatal("%s: config.json is not a Dory model", dir.c_str());
  auto num = [&](const char* k, double def) { return t.contains(k) && t[k].is_number() ? t[k].get<double>() : def; };
  auto flag = [&](const char* k, bool def) { return t.contains(k) && t[k].is_boolean() ? t[k].get<bool>() : def; };
  DoryConfig c;
  c.hidden = int(num("hidden_size", 0));
  c.vocab = int(num("vocab_size", 0));
  c.layers = int(num("num_hidden_layers", 0));
  c.hq = int(num("num_attention_heads", 0));
  c.hkv = int(num("num_key_value_heads", c.hq));
  c.hd = int(num("head_dim", c.hq ? c.hidden / c.hq : 0));
  c.ffn = int(num("intermediate_size", 0));
  c.n_in = int(num("n_input_layers", 0));
  c.n_out = int(num("n_output_layers", 0));
  c.n_rec = int(num("n_recurrent_layers", c.layers - c.n_in - c.n_out));
  c.loops = int(num("n_recurrent_loops", 1));
  c.window = int(num("sliding_window", 0));
  c.init_std = num("initializer_range", 0.02);
  c.alpha_init = num("ngpt_alpha_init", 0.05);
  c.gate_init = t.contains("attention_output_gate_logit_scale_init") && t["attention_output_gate_logit_scale_init"].is_number()
                    ? t["attention_output_gate_logit_scale_init"].get<double>()
                    : std::sqrt(double(c.hidden));
  if (t.contains("rope_parameters") && t["rope_parameters"].is_object()) {
    const auto& rp = t["rope_parameters"];
    for (const char* k : {"sliding_attention", "full_attention"}) {
      if (!rp.contains(k)) continue;
      const auto& r = rp[k];
      if (r.value("rope_type", std::string("default")) != "default" || r.value("partial_rotary_factor", 1.0) != 1.0)
        fatal("%s: Dory rope other than default with full rotary dims is not supported", dir.c_str());
      (std::string(k) == "sliding_attention" ? c.theta_swa : c.theta_full) = r.value("rope_theta", 10000.0);
    }
  }
  if (t.contains("layer_types") && t["layer_types"].is_array())
    for (auto& v : t["layer_types"]) {
      const std::string s = v.is_string() ? v.get<std::string>() : "";
      c.kind.push_back(s == "sliding_attention" ? 'S' : s == "full_attention" ? 'F' : s == "mlp" ? 'M' : '?');
    }
  const std::string act = t.value("hidden_act", std::string("silu"));
  if (int(c.kind.size()) != c.layers || std::count(c.kind.begin(), c.kind.end(), '?'))
    fatal("%s: Dory's layer_types must list %d attention / mlp layers", dir.c_str(), c.layers);
  if (c.n_in < 0 || c.n_rec <= 0 || c.n_out < 0 || c.n_in + c.n_rec + c.n_out != c.layers || c.loops < 1)
    fatal("%s: Dory's input / recurrent / output layers do not add up", dir.c_str());
  if (c.hd != 256 || !c.hkv || c.hq % c.hkv || c.hidden % 256 || c.hidden > 4096 || c.ffn % 128 || act != "silu" ||
      flag("attention_bias", false) || flag("mlp_bias", false) || flag("tie_word_embeddings", false))
    fatal("%s: unsupported Dory shape (head dim %d, %d / %d heads, hidden %d, FFN %d, %s)", dir.c_str(), c.hd, c.hq,
          c.hkv, c.hidden, c.ffn, act.c_str());
  if (std::count(c.kind.begin(), c.kind.end(), 'S') && c.window <= 0)
    fatal("%s: sliding-window layers without a sliding_window", dir.c_str());
  const std::string mode = t.value("recurrent_kv_cache_mode", std::string("per_loop"));
  (void)mode;  // (a serving choice here: --loop-kv; the config's default is vLLM's)
  auto add_ids = [&](const nlohmann::json& v, std::vector<int>& out) {
    if (v.is_number_integer()) out.push_back(v.get<int>());
    if (v.is_array())
      for (auto& x : v)
        if (x.is_number_integer() && std::find(out.begin(), out.end(), x.get<int>()) == out.end()) out.push_back(x.get<int>());
  };
  const nlohmann::json g = read_json(dir + "/generation_config.json");
  if (g.is_object() && g.contains("eos_token_id")) add_ids(g["eos_token_id"], c.eos);
  if (c.eos.empty() && t.contains("eos_token_id")) add_ids(t["eos_token_id"], c.eos);
  c.bos = int(num("bos_token_id", -1));
  return c;
}

size_t Dory::kv_per_pos(const DoryConfig& c, LoopKv kv, int loops) {
  if (loops <= 0) loops = c.loops;
  size_t n = 0;
  for (int l = 0; l < c.layers; ++l)
    if (c.kind[size_t(l)] == 'F') n += c.recurrent(l) && kv == LoopKv::per_loop ? size_t(loops) : 1;
  return n * 2 * size_t(c.hkv) * size_t(c.hd) * 2;
}

size_t Dory::ring_bytes(const DoryConfig& c, LoopKv kv, int loops) {
  if (loops <= 0) loops = c.loops;
  size_t n = 0;
  for (int l = 0; l < c.layers; ++l)
    if (c.kind[size_t(l)] == 'S') n += c.recurrent(l) && kv == LoopKv::per_loop ? size_t(loops) : 1;
  return n * 2 * size_t(c.hkv) * size_t(c.window + kRingSpare) * size_t(c.hd) * 2;
}

// ---- load ----
Dory::Dory(const std::string& path, int prefill_chunk, int loops, LoopKv kv) {
  path_ = path;
  while (path_.size() > 1 && path_.back() == '/') path_.pop_back();
  if (!is_model(path_)) fatal("%s: not a Dory checkpoint folder (config.json)", path.c_str());
  cfg_ = read_config(path_);
  loops_ = loops > 0 ? loops : cfg_.loops;
  if (loops_ > kMaxLoops) fatal("Dory runs 1 to %d loops (asked for %d)", kMaxLoops, loops_);
  kv_ = kv;
  Gpu::get().compile_extra("Nemotron-H", kNhKernelSource);
  Gpu::get().compile_extra("Dory", kDoryKernelSource);
  load(prefill_chunk);
}

Dory::~Dory() {
  if (embed_map_) munmap(embed_map_, embed_map_len_);
}

void Dory::load(int prefill_chunk) {
  const double t0 = now_ms();
  tmax_ = std::max(kMaxRows, prefill_chunk);
  Gpu& g = Gpu::get();
  st_ = std::make_unique<SafetensorsIndex>(path_);
  const SafetensorsIndex& idx = *st_;
  // matrices: one GPU buffer for the codes and one for the scales each, filled by parallel row jobs
  struct Job {
    const SafetensorsIndex::Entry* t = nullptr;
    NhMat* m = nullptr;
    int64_t r0 = 0, rows = 0;
    bool scales = false;  // NVFP4 block scales copied as they are
    size_t bytes = 0;
  };
  std::vector<Job> jobs;
  size_t src_bytes = 0;
  auto mat = [&](NhMat& m, const std::string& name) {
    const SafetensorsIndex::Entry& w = idx.get(name);
    const std::string base = name.substr(0, name.size() - 7);  // "x.weight" -> "x"
    const SafetensorsIndex::Entry* sc = idx.find(base + ".weight_scale");
    uint32_t N = 0, K = 0;
    if (!nh_describe(w, sc, m.fmt, N, K) || (m.fmt != NhFmt::BF16 && m.fmt != NhFmt::FP4))
      fatal("%s: tensor %s has a format Kapitsa does not run for Dory (BF16 or NVFP4)", path_.c_str(), name.c_str());
    m.N = N;
    m.K = K;
    m.E = 1;
    nh_row_bytes(m.fmt, K, m.q_row, m.s_row);
    m.q = g.alloc(std::max<size_t>(m.q_row * N, 16), "dory_weights", false);
    owned_.push_back(m.q);
    if (m.fmt == NhFmt::FP4) {
      if (!sc || sc->len != m.s_row * N) fatal("%s: %s has no NVFP4 scales of its size", path_.c_str(), name.c_str());
      m.s = g.alloc(std::max<size_t>(m.s_row * N, 16), "dory_scales", false);
      owned_.push_back(m.s);
      const SafetensorsIndex::Entry* s2 = idx.find(base + ".weight_scale_2");
      if (!s2 || s2->dtype != DType::F32 || s2->len != 4) fatal("%s: %s has no weight_scale_2", path_.c_str(), name.c_str());
      idx.read(*s2, 0, &m.gs, 4);
      jobs.push_back({sc, &m, 0, int64_t(N), true, sc->len});
      src_bytes += sc->len;
    } else {
      m.s = m.q;
      m.gs = 1.0f;
    }
    const size_t rb = nh_src_row_bytes(m.fmt, K);
    if (w.len != rb * N) fatal("%s: tensor %s's size disagrees with its shape", path_.c_str(), name.c_str());
    const int64_t per = std::max<int64_t>(1, int64_t((size_t(16) << 20) / rb));
    for (int64_t r0 = 0; r0 < int64_t(N); r0 += per) {
      const int64_t n = std::min<int64_t>(per, int64_t(N) - r0);
      jobs.push_back({&w, &m, r0, n, false, rb * size_t(n)});
    }
    src_bytes += w.len;
  };
  // fp32 vectors computed from BF16 ones as NVIDIA's code does in fp32 (f: x * scale, then squared or abs)
  auto vec = [&](Buffer& b, const std::string& name, size_t n, float scale, int op) {
    const SafetensorsIndex::Entry& t = idx.get(name);
    if (t.dtype != DType::BF16 || size_t(t.numel()) != n)
      fatal("%s: %s is not a BF16 vector of %zu", path_.c_str(), name.c_str(), n);
    std::vector<uint16_t> raw(n);
    idx.read(t, 0, raw.data(), n * 2);
    b = g.alloc(n * 4, "dory_vectors", false);
    owned_.push_back(b);
    float* d = b.as<float>();
    for (size_t i = 0; i < n; ++i) {
      const float v = bf16_to_f32(raw[i]) * scale;
      d[i] = op == 1 ? v * v : op == 2 ? std::fabs(v) : v;
    }
    src_bytes += t.len;
  };
  const DoryConfig& c = cfg_;
  const size_t H = size_t(c.hidden), QD = size_t(c.hq) * c.hd;
  const float alpha_scale = float(c.alpha_init / c.init_std);
  const float sqk_f = float(std::sqrt(double(c.hd)) / c.init_std);
  const float sg_f = float(std::sqrt(c.gate_init) / c.init_std);
  const float suv_f = float(std::sqrt(double(c.hidden)));
  layers_.resize(size_t(c.layers));
  int nc = 0;
  for (int l = 0; l < c.layers; ++l) {
    DoryLayer& L = layers_[size_t(l)];
    const std::string b = "backbone.layers." + std::to_string(l) + ".", mx = b + "mixer.";
    L.kind = c.kind[size_t(l)];
    L.rec = c.recurrent(l);
    if (L.kind == 'M') {
      mat(L.gate, mx + "gate_proj.weight");
      mat(L.up, mx + "up_proj.weight");
      mat(L.down, mx + "down_proj.weight");
      vec(L.sug, mx + "suv_gate", size_t(c.ffn), suv_f, 0);
      vec(L.suu, mx + "suv_up", size_t(c.ffn), suv_f, 0);
      vec(L.alpha, b + "mlp_alpha", H, alpha_scale, 2);
      if (int(L.gate.N) != c.ffn || int(L.gate.K) != c.hidden || L.up.N != L.gate.N || int(L.down.N) != c.hidden ||
          int(L.down.K) != c.ffn)
        fatal("%s: layer %d's MLP shapes disagree with config.json", path_.c_str(), l);
    } else {
      mat(L.q, mx + "q_proj.weight");
      mat(L.k, mx + "k_proj.weight");
      mat(L.v, mx + "v_proj.weight");
      mat(L.g, mx + "g_proj.weight");
      mat(L.o, mx + "o_proj.weight");
      vec(L.sqk2, mx + "sqk", QD, sqk_f, 1);
      vec(L.sg2, mx + "s_gate", QD, sg_f, 1);
      vec(L.alpha, b + "attn_alpha", H, alpha_scale, 2);
      if (L.q.N != QD || int(L.k.N) != c.hkv * c.hd || L.v.N != L.k.N || L.g.N != QD || int(L.o.N) != c.hidden ||
          L.o.K != QD || int(L.q.K) != c.hidden)
        fatal("%s: layer %d's attention shapes disagree with config.json", path_.c_str(), l);
      L.cache = nc;
      const int n = L.rec && kv_ == LoopKv::per_loop ? kMaxLoops : 1;
      for (int i = 0; i < n; ++i) {
        ring_.push_back(L.kind == 'S');
        cache_loop_.push_back(n > 1 ? i : 0);
      }
      nc += n;
    }
  }
  n_caches_ = nc;
  ring_rows_ = c.window + kRingSpare;
  mat(head_, "lm_head.weight");
  if (head_.fmt != NhFmt::BF16 || int(head_.N) != c.vocab || int(head_.K) != c.hidden)
    fatal("%s: the LM head is not a BF16 [%d][%d] matrix", path_.c_str(), c.vocab, c.hidden);
  {  // logits x logit_scale / initializer_range (fp32, as NVIDIA's compute_logits)
    const SafetensorsIndex::Entry& t = idx.get("lm_head.logit_scale");
    if (t.dtype != DType::BF16 || int(t.numel()) != c.vocab) fatal("%s: lm_head.logit_scale is not BF16 [vocab]", path_.c_str());
    std::vector<uint16_t> raw(size_t(c.vocab));
    idx.read(t, 0, raw.data(), raw.size() * 2);
    logit_scale_ = g.alloc(raw.size() * 4, "dory_logit_scale", false);
    owned_.push_back(logit_scale_);
    const float sd = float(c.init_std);
    for (size_t i = 0; i < raw.size(); ++i) logit_scale_.as<float>()[i] = bf16_to_f32(raw[i]) / sd;
  }
  {  // rope inverse frequencies (fp32, as vLLM's RotaryEmbedding: 1 / theta^(arange(0, D, 2) / D))
    const int n = c.hd / 2;
    inv_swa_ = g.alloc(size_t(n) * 4, "dory_rope", false);
    inv_full_ = g.alloc(size_t(n) * 4, "dory_rope", false);
    owned_.push_back(inv_swa_);
    owned_.push_back(inv_full_);
    for (int i = 0; i < n; ++i) {
      const float e = float(2 * i) / float(c.hd);
      inv_swa_.as<float>()[i] = 1.0f / std::pow(float(c.theta_swa), e);
      inv_full_.as<float>()[i] = 1.0f / std::pow(float(c.theta_full), e);
    }
  }
  {  // the BF16 embedding mapped from its file (no copy): only the rows a conversation uses become resident
    const SafetensorsIndex::Entry& t = idx.get("backbone.embeddings.weight");
    if (t.dtype != DType::BF16 || t.shape.size() != 2 || int(t.shape[0]) != c.vocab || int(t.shape[1]) != c.hidden)
      fatal("%s: the embedding is not a BF16 [%d][%d] matrix", path_.c_str(), c.vocab, c.hidden);
    embed_.fmt = NhFmt::BF16;
    embed_.N = uint32_t(c.vocab);
    embed_.K = uint32_t(c.hidden);
    if (t.off % 16 == 0) {
      const size_t page = size_t(getpagesize()), off0 = t.off / page * page;
      const size_t len = (t.off + t.len - off0 + page - 1) / page * page;
      const int fd = open(idx.path(t).c_str(), O_RDONLY);
      if (fd < 0) fatal("open %s failed", idx.path(t).c_str());
      void* p = mmap(nullptr, len, PROT_READ, MAP_SHARED, fd, off_t(off0));
      close(fd);
      if (p == MAP_FAILED) fatal("mapping the embedding of %s failed", idx.path(t).c_str());
      madvise(p, len, MADV_WILLNEED);
      embed_map_ = p;
      embed_map_len_ = len;
      Buffer bmap = g.wrap(p, len, "dory_embedding (mapped)", false);
      embed_.q = embed_.s = bmap.view(t.off - off0, t.len);
    } else {
      embed_.q = embed_.s = g.alloc(t.len, "dory_embedding", false);
      owned_.push_back(embed_.q);
      idx.read(t, 0, embed_.q.ptr(), t.len);
    }
  }
  if (mem_budget_ && g.allocated_bytes() > mem_budget_)
    fatal("insufficient memory: the weights of %s take %.1f GB", path_.c_str(), g.allocated_bytes() / 1e9);
  std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.bytes > b.bytes; });
  std::atomic<size_t> next{0};
  const unsigned nth = std::max(2u, std::min(10u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nth; ++i)
    th.emplace_back([&] {
      std::vector<uint8_t> raw;
      for (size_t k; (k = next.fetch_add(1)) < jobs.size();) {
        const Job& j = jobs[k];
        NhMat& m = *j.m;
        if (j.scales) {
          idx.read(*j.t, 0, m.s.ptr(), j.t->len);
          continue;
        }
        const size_t rb = nh_src_row_bytes(m.fmt, m.K);
        raw.resize(rb * size_t(j.rows));
        idx.read(*j.t, rb * size_t(j.r0), raw.data(), raw.size());
        nh_repack_rows(m.fmt, m.K, raw.data(), j.rows, m.q.as<uint8_t>() + m.q_row * size_t(j.r0), nullptr);
      }
    });
  for (auto& t : th) t.join();
  g.make_resident(owned_);
  dummy_ = g.alloc(4096, "dory_dummy");
  alloc_buffers();
  logi("Dory: %d layers (%d input, %d recurrent x %d loops, %d output), %.2f GB of weights read in %.1f s (%s); "
       "recurrent KV %s",
       c.layers, c.n_in, c.n_rec, loops_, c.n_out, src_bytes / 1e9, (now_ms() - t0) / 1000, formats_summary().c_str(),
       loop_kv_name(kv_));
}

std::string Dory::formats_summary() const {
  NhFmt at = NhFmt::F32, ml = NhFmt::F32;
  for (const DoryLayer& L : layers_) (L.kind == 'M' ? ml : at) = L.kind == 'M' ? L.gate.fmt : L.q.fmt;
  std::string r = at == ml ? std::string(fmt_name(at)) + " attention / MLP"
                           : std::string(fmt_name(at)) + " attention, " + fmt_name(ml) + " MLP";
  return r + ", " + fmt_name(head_.fmt) + " head, BF16 embedding";
}

void Dory::alloc_buffers() {
  Gpu& g = Gpu::get();
  const size_t R = kMaxRows, H = size_t(cfg_.hidden), QD = size_t(cfg_.hq) * cfg_.hd, KD = size_t(cfg_.hkv) * cfg_.hd,
               F = size_t(cfg_.ffn);
  x_ = g.alloc(R * H * 4, "dory_x");
  inp_ = g.alloc(R * H * 4, "dory_inp");
  y_ = g.alloc(R * H * 4, "dory_y");
  q_ = g.alloc(R * QD * 4, "dory_q");
  k_ = g.alloc(R * KD * 4, "dory_k");
  v_ = g.alloc(R * KD * 4, "dory_v");
  g_ = g.alloc(R * QD * 4, "dory_g");
  ao_ = g.alloc(R * QD * 4, "dory_ao");
  mg_ = g.alloc(R * F * 4, "dory_mg");
  mu_ = g.alloc(R * F * 4, "dory_mu");
  mh_ = g.alloc(R * F * 4, "dory_mh");
  part_o_ = g.alloc(R * size_t(cfg_.hq) * attn_max_chunks_ * size_t(cfg_.hd) * 4, "dory_part_o");
  part_ml_ = g.alloc(R * size_t(cfg_.hq) * attn_max_chunks_ * 2 * 4, "dory_part_ml");
  logits_ = g.alloc(R * size_t(cfg_.vocab) * 4, "dory_logits");
  const size_t nchunk1 = (size_t(cfg_.vocab) + 2047) / 2048;
  cand_v_ = g.alloc(nchunk1 * 64 * 4, "dory_cand_v");
  cand_i_ = g.alloc(nchunk1 * 64 * 4, "dory_cand_i");
  sprob_ = g.alloc(64 * 4, "dory_sprob");
  samp_ = g.alloc(64 * 4, "dory_samples");
}

void Dory::alloc_prefill() {
  if (pf_alloc_) return;
  Gpu& g = Gpu::get();
  const size_t T = size_t(tmax_), H = size_t(cfg_.hidden), QD = size_t(cfg_.hq) * cfg_.hd,
               KD = size_t(cfg_.hkv) * cfg_.hd, F = size_t(cfg_.ffn);
  px_ = g.alloc(T * H * 4, "dory_pf_x");
  pxh_ = g.alloc(T * H * 2, "dory_pf_xh");
  pinp_ = g.alloc(T * H * 4, "dory_pf_inp");
  py_ = g.alloc(T * H * 4, "dory_pf_y");
  pq_ = g.alloc(T * QD * 4, "dory_pf_q");
  pk_ = g.alloc(T * KD * 4, "dory_pf_k");
  pv_ = g.alloc(T * KD * 4, "dory_pf_v");
  pg_ = g.alloc(T * QD * 4, "dory_pf_g");
  pck_ = g.alloc(T * KD * 2, "dory_pf_ck");
  pcv_ = g.alloc(T * KD * 2, "dory_pf_cv");
  pao_ = g.alloc(T * QD * 4, "dory_pf_ao");
  paoh_ = g.alloc(T * QD * 2, "dory_pf_aoh");
  pmg_ = g.alloc(T * F * 4, "dory_pf_mg");
  pmu_ = g.alloc(T * F * 4, "dory_pf_mu");
  pmh_ = g.alloc(T * F * 2, "dory_pf_mh");
  pf_alloc_ = true;
}

MemPlan Dory::plan(const std::string& path, int prefill_chunk, LoopKv kv, int loops) {
  MemPlan m;
  if (!is_model(path)) return m;
  const DoryConfig c = read_config(path);
  if (loops <= 0) loops = c.loops;
  const SafetensorsIndex idx(path);
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = c.ffn;
  std::string fmts;
  for (const auto& [name, t] : idx.all()) {
    const bool embedding = name.find("embeddings.") != std::string::npos;
    const bool vector = t.shape.size() == 1;
    if (embedding) continue;  // mapped from its file: clean pages, the rows in use
    m.weights += vector && t.dtype == DType::BF16 ? 2 * t.len : t.len;
    const size_t lp = name.find("layers.");
    const int li = lp == std::string::npos ? -1 : std::atoi(name.c_str() + lp + 7);
    const double uses = li >= 0 && c.recurrent(li) ? double(loops) : 1.0;  // the recurrent block runs `loops` times
    m.decode_bytes += size_t(double(t.len) * uses);
    NhFmt f;
    uint32_t N = 0, K = 0;
    const size_t dot = name.rfind(".weight");
    const SafetensorsIndex::Entry* sc =
        dot != std::string::npos && dot + 7 == name.size() ? idx.find(name.substr(0, dot) + ".weight_scale") : nullptr;
    if (!vector && name.find("weight_scale") == std::string::npos && nh_describe(t, sc, f, N, K)) {
      m.decode_params += size_t(double(N) * K * uses);
      const std::string fn = fmt_name(f);
      if (fmts.find(fn) == std::string::npos) fmts += std::string(fmts.empty() ? "" : " / ") + fn;
      if (name.find("lm_head.") != std::string::npos) {
        size_t qb = 0, sb = 0;
        nh_row_bytes(f, K, qb, sb);
        m.head_row_bytes = qb + sb;
      }
    }
  }
  m.formats = fmts;
  m.dense = true;
  m.ffn_ane = false;
  m.kv_per_pos = kv_per_pos(c, kv, loops);
  // per slot: the sliding-window rings, live and in the slot's two checkpoints
  m.slot_state = 3 * ring_bytes(c, kv, loops);
  const size_t R = kMaxRows, H = size_t(c.hidden), QD = size_t(c.hq) * c.hd, KD = size_t(c.hkv) * c.hd,
               F = size_t(c.ffn), V = size_t(c.vocab);
  m.activations = R * (3 * H + 4 * QD + 2 * KD + 3 * F + V) * 4 + R * size_t(c.hq) * 64 * (size_t(c.hd) + 2) * 4 +
                  (size_t(16) << 20);
  const size_t T = size_t(std::max(kMaxRows, prefill_chunk));
  m.activations += T * (H * (4 + 2 + 4 + 4) + QD * (4 + 4 + 4 + 2) + KD * (4 + 4 + 2 + 2) + F * (4 + 4 + 2));
  return m;
}

// ---- sequences, KV caches, checkpoints ----
int Dory::cache_index(const DoryLayer& L, int loop) const {
  return L.cache + (L.rec && kv_ == LoopKv::per_loop ? loop : 0);
}

int Dory::cache_rows(const Seq& s, int cache) const { return is_ring(cache) ? ring_rows_ : s.cap; }

std::unique_ptr<Seq> Dory::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  s->kc.resize(size_t(n_caches_));
  s->vc.resize(size_t(n_caches_));
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0 && !reserve(*s, cap)) fatal("insufficient memory for a %d-token KV cache", cap);
  return s;
}

size_t Dory::kv_bytes_per_pos() const { return kv_per_pos(cfg_, kv_, loops_); }

// Every cache a conversation of `loops` loops uses (per-loop mode: the recurrent layers' caches of loops [0, loops)):
// the linear ones at s.cap rows, the rings at theirs. False (nothing allocated) when the budget does not allow it.
bool Dory::ensure_loop_caches(Seq& s, int loops) {
  if (s.cap <= 0) return true;
  Gpu& g = Gpu::get();
  const size_t row = size_t(cfg_.hkv) * cfg_.hd * 2;
  std::vector<int> missing;
  size_t need = 0;
  for (int c = 0; c < n_caches_; ++c)
    if (cache_loop_[size_t(c)] < loops && !s.kc[size_t(c)]) {
      missing.push_back(c);
      need += 2 * row * size_t(cache_rows(s, c));
    }
  if (missing.empty()) return true;
  if (mem_budget_ && g.allocated_bytes() + need > mem_budget_) return false;
  for (int c : missing) {
    const size_t nb = row * size_t(cache_rows(s, c));
    s.kc[size_t(c)] = g.alloc(nb, is_ring(c) ? "dory_kv_ring" : "dory_kv");
    s.vc[size_t(c)] = g.alloc(nb, is_ring(c) ? "dory_kv_ring" : "dory_kv");
  }
  return true;
}

bool Dory::reserve(Seq& s, int need) {
  const int loops = loops_of(s);
  if (need <= s.cap) return ensure_loop_caches(s, loops);
  if (need > s.limit) return false;
  Gpu& g = Gpu::get();
  const size_t per = kv_per_pos(cfg_, kv_, loops), row = size_t(cfg_.hkv) * cfg_.hd * 2;
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  const size_t rings = s.cap > 0 ? 0 : ring_bytes(cfg_, kv_, loops);  // the rings come with the first cache
  auto fits = [&](int c) {
    const size_t copy = std::min(size_t(256) << 20, size_t(s.cap) * per);
    return !mem_budget_ || g.allocated_bytes() + size_t(c - s.cap) * per + copy + rings <= mem_budget_;
  };
  int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
  if (!fits(target)) target = std::min(s.limit, round_up(need, 256));
  if (!fits(target)) return false;
  const int old = s.cap;
  // the linear caches grow (the ones in use: allocated, or of the loops in use when there were none yet)
  for (int c = 0; c < n_caches_; ++c) {
    if (is_ring(c)) continue;
    for (Buffer* b : {&s.kc[size_t(c)], &s.vc[size_t(c)]}) {
      if (!*b && !(old == 0 && cache_loop_[size_t(c)] < loops)) continue;
      Buffer nbuf = g.alloc(row * size_t(target), "dory_kv");
      if (old > 0 && *b) {
        const uint32_t n16 = uint32_t(cfg_.hd) * 2 / 16;
        const uint32_t p[4] = {uint32_t(old) * n16, uint32_t(target) * n16, uint32_t(cfg_.hkv), 0};
        auto e = g.begin();
        e->use(pipe("kv_regrow"));
        e->buf(0, *b);
        e->buf(1, nbuf);
        e->bytes(2, p, sizeof p);
        e->dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
        e->commit_and_wait();
      }
      g.release(*b);
      *b = nbuf;
    }
  }
  s.cap = target;
  return ensure_loop_caches(s, loops);  // the rings (first cache) and any loop's caches still missing
}

void Dory::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (auto& b : *v) {
      g.release(b);
      b = Buffer{};
    }
  s.cap = 0;
}

void Dory::reset(Seq& s) { (void)s; }  // no recurrent state: every position is written before it is read

// A checkpoint holds copies of the sliding-window rings (made on the first save): the linear caches keep every
// earlier position, the rings only the last window + 8.
Checkpoint Dory::new_checkpoint() { return Checkpoint{}; }

void Dory::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  Gpu& g = Gpu::get();
  c.pos = -1;
  if (c.rec.size() != size_t(2 * n_caches_)) c.rec.assign(size_t(2 * n_caches_), Buffer{});
  for (size_t i = 0; i < size_t(n_caches_); ++i) {
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = kv ? s.vc[i] : s.kc[i];
      Buffer& dst = c.rec[2 * i + size_t(kv)];
      if (!is_ring(int(i)) || !src) {
        if (dst) g.release(dst), dst = Buffer{};
        continue;
      }
      if (!dst) {
        if (mem_budget_ && g.allocated_bytes() + src.size > mem_budget_) return;  // no room: no checkpoint
        dst = g.alloc(src.size, "dory_ring_ckpt");
      }
      const uint32_t n = uint32_t(src.size / 16);
      e.use(pipe("dory_copy4"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
  }
  c.pos = pos;
}

void Dory::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  for (size_t i = 0; i < size_t(n_caches_) && 2 * i + 1 < c.rec.size(); ++i)
    for (int kv = 0; kv < 2; ++kv) {
      const Buffer& src = c.rec[2 * i + size_t(kv)];
      const Buffer& dst = kv ? s.vc[i] : s.kc[i];
      if (!src || !dst) continue;
      const uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
      e.use(pipe("dory_copy4"));
      e.buf(0, src);
      e.buf(1, dst);
      e.val(2, n);
      e.dispatch((n + 255) / 256, 1, 1, 256);
    }
}

// ---- encoders ----
void Dory::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int R, const Buffer& out) {
  const EmbedP ep{uint32_t(cfg_.hidden), uint32_t(tok_off)};
  e.use(pipe("nh_embed_bf16"));
  e.buf(0, embed_.q);
  e.buf(1, embed_.q);
  e.buf(2, tokens);
  e.buf(3, out);
  e.val(4, ep);
  e.dispatch(uint32_t(R), 1, 1, 256);
}

void Dory::enc_mv(Encoder& e, const NhMat& W, const Buffer& x, const Buffer& y, int R, int epi) {
  if (R == 1 && W.fmt == NhFmt::FP4 && W.K % 32 == 0 && epi <= 1) {
    const MvP p{W.N, W.K, W.K, W.N, W.gs, 1};
    e.use(pipe(epi ? "nh_x4_mv_l0_add" : "nh_x4_mv_l0"));
    e.buf(0, W.q);
    e.buf(1, W.s);
    e.buf(2, x);
    e.buf(3, y);
    e.val(4, p);
    e.dispatch((W.N + 7) / 8, 1, 1, 64);
    return;
  }
  const int rk = R <= 1 ? 1 : R <= 2 ? 2 : R <= 4 ? 4 : 8;
  const char* fm = W.fmt == NhFmt::FP4 ? "fp4" : W.fmt == NhFmt::Q8 ? "q8" : "bf16";
  const std::string name = std::string("nh_mv_") + fm + "_r" + std::to_string(rk) + (epi == 1 ? "_add" : "");
  const uint32_t nr = rk <= 2 ? 4 : rk == 4 ? 2 : 1, nsg = 4;
  const MvP p{W.N, W.K, W.K, W.N, W.gs, uint32_t(R)};
  e.use(pipe(name));
  e.buf(0, W.q);
  e.buf(1, W.s);
  e.buf(2, x);
  e.buf(3, y);
  e.val(4, p);
  e.dispatch((W.N + nr * nsg - 1) / (nr * nsg), 1, 1, 32 * nsg);
}

void Dory::enc_gemm(Encoder& e, const NhMat& W, const Buffer& xh, const Buffer& y, int M, int epi) {
  const int bm = M <= 16 ? 16 : M <= 32 ? 32 : 64;
  const std::string name = std::string(W.fmt == NhFmt::FP4 ? "nh_gemm_fp4" : W.fmt == NhFmt::Q8 ? "nh_gemm_q8" : "nh_gemm_bf16w") +
                           "_m" + std::to_string(bm);
  const GemmP p{uint32_t(M), W.N, W.K, W.K, W.N, W.gs, uint32_t(epi), 0, 0, 1};
  e.use(pipe(name));
  e.buf(0, xh);
  e.buf(1, W.q);
  e.buf(2, W.s);
  e.buf(3, y);
  e.val(4, p);
  e.buf(8, dummy_);
  e.dispatch((W.N + 63) / 64, (uint32_t(M) + uint32_t(bm) - 1) / uint32_t(bm), 1, 128);
}

void Dory::enc_slerp(Encoder& e, const Buffer& h, const Buffer& y, const Buffer& alpha, const Buffer& inp,
                     const Buffer* xh, int R, int mode) {
  const SlerpP p{uint32_t(cfg_.hidden), uint32_t(mode), xh ? 1u : 0u};
  e.use(pipe("dory_slerp"));
  e.buf(0, h);
  e.buf(1, y);
  e.buf(2, alpha);
  e.buf(3, inp);
  e.buf(4, xh ? *xh : dummy_);
  e.val(5, p);
  e.dispatch(uint32_t(R), 1, 1, 1024);
}

// One sublayer of R <= 8 rows (decode, verify, a short prompt): x_ in place. mode: what follows the residual
// (dory_slerp: 0 nothing, 1 the end of the last loop, 2 the end of a loop with another to follow).
void Dory::enc_layer_rows(Encoder& e, Seq& s, const DoryLayer& L, int loop, int pos0, int R, int mode) {
  const DoryConfig& c = cfg_;
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd);
  if (L.kind == 'M') {
    enc_mv(e, L.gate, x_, mg_, R, 0);
    enc_mv(e, L.up, x_, mu_, R, 0);
    const RowP rp{uint32_t(c.ffn), uint32_t(c.ffn), uint32_t(c.ffn), 0};
    e.use(pipe("dory_swiglu_norm"));
    e.buf(0, mg_);
    e.buf(1, mu_);
    e.buf(2, L.sug);
    e.buf(3, L.suu);
    e.buf(4, mh_);
    e.val(5, rp);
    e.dispatch(uint32_t(R), 1, 1, 1024);
    enc_mv(e, L.down, mh_, y_, R, 0);
    enc_slerp(e, x_, y_, L.alpha, inp_, nullptr, R, mode);
    return;
  }
  enc_mv(e, L.q, x_, q_, R, 0);
  enc_mv(e, L.k, x_, k_, R, 0);
  enc_mv(e, L.v, x_, v_, R, 0);
  enc_mv(e, L.g, x_, g_, R, 0);
  const int ci = cache_index(L, loop);
  const uint32_t rows = uint32_t(cache_rows(s, ci)), ring = is_ring(ci) ? 1u : 0u;
  if (!s.kc[size_t(ci)]) fatal("Dory: cache %d of loop %d not allocated", ci, loop);
  if (!ring && pos0 + R > s.cap) fatal("context full (%d)", s.cap);
  const PrepP pp{uint32_t(R), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring, 0, QD, KD};
  e.use(pipe("dory_prep"));
  e.buf(0, q_);
  e.buf(1, k_);
  e.buf(2, v_);
  e.buf(3, L.sqk2);
  e.buf(4, L.kind == 'S' ? inv_swa_ : inv_full_);
  e.buf(5, s.kc[size_t(ci)]);
  e.buf(6, s.vc[size_t(ci)]);
  e.val(7, pp);
  e.dispatch(uint32_t(c.hq + 2 * c.hkv), uint32_t(R), 1, 128);
  // keys [k0, pos0 + R) in chunks: one per GPU core and kv head, at least 64 keys (16-key tiles), at most 64 chunks
  const uint32_t window = L.kind == 'S' ? uint32_t(c.window) : 0u;
  const uint32_t k0 = window && uint32_t(pos0) + 1 > window ? uint32_t(pos0) + 1 - window : 0u;
  const uint32_t Lk = uint32_t(pos0 + R) - k0;
  uint32_t chunk = std::max<uint32_t>(64, (Lk + gpu_cores() - 1) / gpu_cores());
  chunk = std::max(chunk, (Lk + uint32_t(attn_max_chunks_) - 1) / uint32_t(attn_max_chunks_));
  chunk = (chunk + 15) / 16 * 16;
  const uint32_t nch = (Lk + chunk - 1) / chunk;
  const AttnP ap{uint32_t(R), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring, window, chunk, nch, k0,
                 1.0f / std::sqrt(float(c.hd))};
  e.use(pipe("dory_attn_partial"));
  e.buf(0, q_);
  e.buf(1, s.kc[size_t(ci)]);
  e.buf(2, s.vc[size_t(ci)]);
  e.buf(3, part_o_);
  e.buf(4, part_ml_);
  e.val(5, ap);
  e.dispatch(nch, uint32_t(c.hkv), uint32_t(R), 32 * uint32_t(c.hq / c.hkv));
  e.use(pipe("dory_attn_reduce"));
  e.buf(0, part_o_);
  e.buf(1, part_ml_);
  e.buf(2, ao_);
  e.val(3, ap);
  e.dispatch(uint32_t(c.hq), uint32_t(R), 1, uint32_t(c.hd));
  const RowP rp{QD, QD, QD, 0};
  e.use(pipe("dory_gate_norm"));
  e.buf(0, ao_);
  e.buf(1, g_);
  e.buf(2, L.sg2);
  e.buf(3, ao_);
  e.val(4, rp);
  e.dispatch(uint32_t(R), 1, 1, 1024);
  enc_mv(e, L.o, ao_, y_, R, 0);
  enc_slerp(e, x_, y_, L.alpha, inp_, nullptr, R, mode);
}

// One sublayer of a prefill chunk of T rows (GEMMs on half rows): px_ in place, its half copy in pxh_. The chunk's
// keys go into the layer's cache after its attention when `write` (in last-loop mode the recurrent layers keep their
// final loop's: earlier loops read the chunk's own keys from pck_ / pcv_ only).
void Dory::enc_layer_prefill(Encoder& e, Seq& s, const DoryLayer& L, int loop, int pos0, int T, int mode, bool write) {
  const DoryConfig& c = cfg_;
  const uint32_t QD = uint32_t(c.hq * c.hd), KD = uint32_t(c.hkv * c.hd);
  if (L.kind == 'M') {
    enc_gemm(e, L.gate, pxh_, pmg_, T, 0);
    enc_gemm(e, L.up, pxh_, pmu_, T, 0);
    const RowP rp{uint32_t(c.ffn), uint32_t(c.ffn), uint32_t(c.ffn), 1};
    e.use(pipe("dory_swiglu_norm"));
    e.buf(0, pmg_);
    e.buf(1, pmu_);
    e.buf(2, L.sug);
    e.buf(3, L.suu);
    e.buf(4, pmh_);
    e.val(5, rp);
    e.dispatch(uint32_t(T), 1, 1, 1024);
    enc_gemm(e, L.down, pmh_, py_, T, 0);
    enc_slerp(e, px_, py_, L.alpha, pinp_, &pxh_, T, mode);
    return;
  }
  enc_gemm(e, L.q, pxh_, pq_, T, 0);
  enc_gemm(e, L.k, pxh_, pk_, T, 0);
  enc_gemm(e, L.v, pxh_, pv_, T, 0);
  enc_gemm(e, L.g, pxh_, pg_, T, 0);
  const int ci = cache_index(L, loop);
  const uint32_t rows = uint32_t(cache_rows(s, ci)), ring = is_ring(ci) ? 1u : 0u;
  if (!s.kc[size_t(ci)]) fatal("Dory: cache %d of loop %d not allocated", ci, loop);
  if (!ring && pos0 + T > s.cap) fatal("context full (%d)", s.cap);
  // q in place; the chunk's own keys / values (half [Hkv][T][256]) into pck_ / pcv_
  const PrepP pp{uint32_t(T), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), uint32_t(T), 0, uint32_t(pos0), QD, KD};
  e.use(pipe("dory_prep"));
  e.buf(0, pq_);
  e.buf(1, pk_);
  e.buf(2, pv_);
  e.buf(3, L.sqk2);
  e.buf(4, L.kind == 'S' ? inv_swa_ : inv_full_);
  e.buf(5, pck_);
  e.buf(6, pcv_);
  e.val(7, pp);
  e.dispatch(uint32_t(c.hq + 2 * c.hkv), uint32_t(T), 1, 128);
  const uint32_t window = L.kind == 'S' ? uint32_t(c.window) : 0u;
  const AttnPfP ap{uint32_t(T), uint32_t(pos0), uint32_t(c.hq), uint32_t(c.hkv), rows, ring, window,
                   1.0f / std::sqrt(float(c.hd))};
  // (16-key tiles under GPU shader validation, whose instrumentation doubles the 32-key kernel's threadgroup memory
  // past the limit; NI_DORY_ATTN_K16=1 selects them too)
  static const bool k16 = getenv("MTL_SHADER_VALIDATION") || env_on("NI_DORY_ATTN_K16");
  e.use(pipe(k16 ? "dory_attn_prefill_k16" : "dory_attn_prefill"));
  e.buf(0, pq_);
  e.buf(1, s.kc[size_t(ci)]);
  e.buf(2, s.vc[size_t(ci)]);
  e.buf(3, pck_);
  e.buf(4, pcv_);
  e.buf(5, pao_);
  e.val(6, ap);
  e.dispatch((uint32_t(T) + 31) / 32, uint32_t(c.hq), 1, 256);
  if (write) {
    const uint32_t r0 = ring && uint32_t(T) > rows ? uint32_t(T) - rows : 0u;
    const StoreP sp{uint32_t(T), r0, uint32_t(pos0), uint32_t(c.hkv), rows, ring};
    e.use(pipe("dory_kv_store"));
    e.buf(0, pck_);
    e.buf(1, pcv_);
    e.buf(2, s.kc[size_t(ci)]);
    e.buf(3, s.vc[size_t(ci)]);
    e.val(4, sp);
    e.dispatch(uint32_t(T) - r0, 2 * uint32_t(c.hkv), 1, 32);
  }
  const RowP rp{QD, QD, QD, 1};
  e.use(pipe("dory_gate_norm"));
  e.buf(0, pao_);
  e.buf(1, pg_);
  e.buf(2, L.sg2);
  e.buf(3, paoh_);
  e.val(4, rp);
  e.dispatch(uint32_t(T), 1, 1, 1024);
  enc_gemm(e, L.o, paoh_, py_, T, 0);
  enc_slerp(e, px_, py_, L.alpha, pinp_, &pxh_, T, mode);
}

void Dory::enc_rows(Encoder& e, Seq& s, int pos0, int R) {
  if (R < 1 || R > kMaxRows) fatal("Dory: %d rows in one pass", R);
  const int loops = loops_of(s);
  const DoryConfig& c = cfg_;
  enc_embed(e, s.tok, pos0, R, x_);
  for (int l = 0; l < c.n_in; ++l) enc_layer_rows(e, s, layers_[size_t(l)], 0, pos0, R, 0);
  const SlerpP ls{uint32_t(c.hidden), 0, 0};
  e.use(pipe("dory_loop_start"));
  e.buf(0, x_);
  e.buf(1, inp_);
  e.buf(2, dummy_);
  e.val(3, ls);
  e.dispatch(uint32_t(R), 1, 1, 1024);
  for (int i = 0; i < loops; ++i)
    for (int l = c.n_in; l < c.n_in + c.n_rec; ++l) {
      const int mode = l + 1 < c.n_in + c.n_rec ? 0 : i + 1 < loops ? 2 : 1;
      enc_layer_rows(e, s, layers_[size_t(l)], i, pos0, R, mode);
    }
  for (int l = c.n_in + c.n_rec; l < c.layers; ++l) enc_layer_rows(e, s, layers_[size_t(l)], 0, pos0, R, 0);
}

void Dory::enc_prefill_rows(Encoder& e, Seq& s, int pos0, int T) {
  alloc_prefill();
  const int loops = loops_of(s);
  const DoryConfig& c = cfg_;
  enc_embed(e, s.tok, pos0, T, px_);
  const uint32_t H = uint32_t(c.hidden);
  e.use(pipe("dory_rows_half"));
  e.buf(0, px_);
  e.buf(1, pxh_);
  e.val(2, H);
  e.dispatch(uint32_t(T), 1, 1, 256);
  // a command buffer per layer of a long chunk, so the desktop's own GPU work is not held back behind one long one
  auto split = [&] {
    if (T >= 256) e.split();
  };
  for (int l = 0; l < c.n_in; ++l) {
    enc_layer_prefill(e, s, layers_[size_t(l)], 0, pos0, T, 0, true);
    split();
  }
  const SlerpP ls{H, 0, 1};
  e.use(pipe("dory_loop_start"));
  e.buf(0, px_);
  e.buf(1, pinp_);
  e.buf(2, pxh_);
  e.val(3, ls);
  e.dispatch(uint32_t(T), 1, 1, 1024);
  for (int i = 0; i < loops; ++i)
    for (int l = c.n_in; l < c.n_in + c.n_rec; ++l) {
      const int mode = l + 1 < c.n_in + c.n_rec ? 0 : i + 1 < loops ? 2 : 1;
      const bool write = kv_ == LoopKv::per_loop || i + 1 == loops;
      enc_layer_prefill(e, s, layers_[size_t(l)], i, pos0, T, mode, write);
      split();
    }
  for (int l = c.n_in + c.n_rec; l < c.layers; ++l) {
    enc_layer_prefill(e, s, layers_[size_t(l)], 0, pos0, T, 0, true);
    if (l + 1 < c.layers) split();
  }
}

void Dory::enc_head(Encoder& e, const Buffer& x, int R, const Buffer& out, size_t ld) {
  if (R <= kMaxRows) {
    enc_mv(e, head_, x, out, R, 0);
  } else {
    fatal("Dory: head over %d rows", R);
  }
  const ScaleP sp{uint32_t(cfg_.vocab), uint32_t(ld)};
  e.use(pipe("dory_logit_scale"));
  e.buf(0, out);
  e.buf(1, logit_scale_);
  e.val(2, sp);
  e.dispatch((uint32_t(cfg_.vocab) + 255) / 256, uint32_t(R), 1, 256);
}

void Dory::enc_sample(Encoder& e, const Buffer& lrow, const Buffer& tokens, int out_index, const SamplingParams& sp,
                      uint32_t step) {
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  const TopkStageP ts{uint32_t(cfg_.vocab), K, 2048};
  const uint32_t ntg = (uint32_t(cfg_.vocab) + 2047) / 2048;
  e.use(pipe("topk_stage1"));
  e.buf(0, lrow);
  e.buf(1, cand_v_);
  e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  const SampleP p2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(pipe("sample_stage2"));
  e.buf(0, cand_v_);
  e.buf(1, cand_i_);
  e.buf(2, tokens);
  e.buf(3, sprob_);
  e.val(4, p2);
  e.dispatch(1, 1, 1, 1024);
}

void Dory::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  enc_rows(e, s, pos, 1);
  if (!sample) return;
  enc_head(e, x_, 1, logits_, size_t(cfg_.vocab));
  enc_sample(e, logits_, s.tok, pos + 1, sp, step);
}

void Dory::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                          Checkpoint* snap, int snap_row) {
  (void)snap;
  (void)snap_row;
  if (T > tmax_) fatal("Dory: prefill of %d rows > chunk %d", T, tmax_);
  Buffer last;
  const size_t row = size_t(cfg_.hidden) * 4;
  if (T <= kMaxRows) {
    enc_rows(e, s, pos0, T);
    last = x_.view(size_t(T - 1) * row, row);
  } else {
    enc_prefill_rows(e, s, pos0, T);
    last = px_.view(size_t(T - 1) * row, row);
  }
  if (!sample) return;
  enc_head(e, last, 1, logits_, size_t(cfg_.vocab));
  enc_sample(e, logits_, s.tok, pos0 + T, sp, step);
}

void Dory::encode_logits_rows(Encoder& e, Seq& s, int pos0, int T, const Buffer& out) {
  const size_t row = size_t(cfg_.hidden) * 4, V = size_t(cfg_.vocab);
  if (T <= kMaxRows) {
    enc_rows(e, s, pos0, T);
    enc_head(e, x_, T, out, V);
    return;
  }
  enc_prefill_rows(e, s, pos0, T);
  for (int r0 = 0; r0 < T; r0 += kMaxRows) {
    const int R = std::min(kMaxRows, T - r0);
    enc_head(e, px_.view(size_t(r0) * row, size_t(R) * row), R, out.view(size_t(r0) * V * 4, size_t(R) * V * 4), V);
  }
}

}  // namespace ni
