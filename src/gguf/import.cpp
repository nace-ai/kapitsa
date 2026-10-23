// GGUF -> Kapitsa model folder (see import.h).
//
// Layout facts used here (llama.cpp's GGUF for Qwen3.5, arch "qwen35" / "qwen35moe"): tensor dims are stored
// innermost first, so a [N, K] matrix has ne = {K, N} and rows of K values; blk.N.attn_q holds the query and its
// output gate interleaved per head (as Hugging Face's q_proj); RMSNorm weights are stored as 1 + w (MLX's
// convention too) except the DeltaNet's gated norm (ssm_norm); ssm_a is -exp(A_log); ssm_dt.bias is dt_bias;
// ssm_conv1d is {4, C}; when the DeltaNet has fewer key than value heads, every value-head axis is stored in tiled
// order (value head v of key group k at v * n_k + k) instead of Hugging Face's grouped order (k * r + v); the MTP
// block sits after the trunk's layers (block_count includes it, nextn_predict_layers counts it), with eh_proj = fc,
// enorm / hnorm = pre_fc_norm_embedding / pre_fc_norm_hidden and nextn.shared_head_norm = the head's norm.
//
// Recursive's Flint (arch "flint"): the tensors are Hugging Face's as they are, checked against RecursiveSI/flint-int4's
// (unsloth's flint-UD-IQ2_XXS: norms, q / k norms, sinks (attn_sinks = softmax_offset) and router equal; q / k rows in
// Hugging Face's order, no RoPE permutation; expert e of a stacked tensor is Hugging Face's expert e); the layer kinds
// and RoPE come from attention.sliding_window_pattern / rope_pattern, the dense layers from leading_dense_block_count.
#include "gguf/import.h"

#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "hub/hub.h"
#include "tokenizer/tokenizer.h"
#include "util/term.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/clonefile.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <thread>

#ifndef KAPITSA_VERSION
#define KAPITSA_VERSION "0.2.10"
#endif

namespace ni::gguf {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;

namespace {

constexpr uint64_t kShardBytes = 5ull << 30;     // MLX's shard size
constexpr uint64_t kKeepFree = 6ull << 30;       // free disk an import must leave
constexpr uint64_t kChunkBytes = 64ull << 20;    // source bytes per conversion step

std::atomic<bool> g_stop{false};  // Ctrl+C during a conversion: stop, remove the unfinished folder

// SIGINT / SIGTERM set g_stop while it lives (the previous handlers come back after)
struct StopGuard {
  struct sigaction old_int {}, old_term {};
  StopGuard() {
    g_stop = false;
    struct sigaction sa {};
    sa.sa_handler = [](int) { g_stop = true; };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, &old_int);
    sigaction(SIGTERM, &sa, &old_term);
  }
  ~StopGuard() {
    sigaction(SIGINT, &old_int, nullptr);
    sigaction(SIGTERM, &old_term, nullptr);
  }
};

// ---------------------------------------------------------------- small helpers

std::string lower(std::string s) {
  for (auto& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool ends_with(const std::string& s, const std::string& t) {
  return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

std::string base_name(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

std::string dir_name(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? "." : s == 0 ? "/" : p.substr(0, s);
}

bool is_dir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool is_file(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// the device of a path, or of its closest existing parent
dev_t device_of(std::string p) {
  struct stat st;
  while (stat(p.c_str(), &st) != 0) {
    if (p == "/" || p.empty() || p == ".") return 0;
    p = dir_name(p);
  }
  return st.st_dev;
}

uint64_t free_bytes(std::string p) {
  struct statfs s;
  while (statfs(p.c_str(), &s) != 0) {
    if (p == "/" || p.empty()) return 0;
    p = dir_name(p);
  }
  return uint64_t(s.f_bavail) * uint64_t(s.f_bsize);
}

std::string gb(uint64_t b) { return hub::fmt_bytes(b); }

std::string fmt_secs(double s) {
  char b[32];
  if (!(s >= 0) || !std::isfinite(s)) return "?";
  if (s < 60) snprintf(b, sizeof b, "%.0fs", std::max(1.0, s));
  else if (s < 3600) snprintf(b, sizeof b, "%dm %02ds", int(s) / 60, int(s) % 60);
  else snprintf(b, sizeof b, "%dh %02dm", int(s) / 3600, int(s) % 3600 / 60);
  return b;
}

void write_text(const std::string& path, const std::string& data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f << data;
  if (!f) throw Error("Can't write " + hub::tilde(path) + ": " + strerror(errno) + ".");
}

// A file name for the models dir: lower case, [a-z0-9._-], no leading dot.
std::string clean_name(const std::string& s) {
  std::string o;
  for (char c : lower(s)) o += (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_' || c == '-') ? c : '-';
  while (!o.empty() && (o[0] == '.' || o[0] == '-')) o.erase(o.begin());
  while (!o.empty() && o.back() == '-') o.pop_back();
  return o;
}

int perf_cores() {
  int n = 0;
  size_t len = sizeof n;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) != 0 || n <= 0)
    n = int(std::max(1u, std::thread::hardware_concurrency()));
  return n;
}

// llama.cpp's file types (general.file_type), for names and messages
std::string ftype_name(int64_t t) {
  static const std::map<int64_t, const char*> m = {
      {0, "F32"},     {1, "F16"},     {2, "Q4_0"},     {3, "Q4_1"},     {7, "Q8_0"},     {8, "Q5_0"},
      {9, "Q5_1"},    {10, "Q2_K"},   {11, "Q3_K_S"},  {12, "Q3_K_M"},  {13, "Q3_K_L"},  {14, "Q4_K_S"},
      {15, "Q4_K_M"}, {16, "Q5_K_S"}, {17, "Q5_K_M"},  {18, "Q6_K"},    {19, "IQ2_XXS"}, {20, "IQ2_XS"},
      {21, "Q2_K_S"}, {22, "IQ3_XS"}, {23, "IQ3_XXS"}, {24, "IQ1_S"},   {25, "IQ4_NL"},  {26, "IQ3_S"},
      {27, "IQ3_M"},  {28, "IQ2_S"},  {29, "IQ2_M"},   {30, "IQ4_XS"},  {31, "IQ1_M"},   {32, "BF16"},
      {36, "TQ1_0"},  {37, "TQ2_0"},  {38, "MXFP4"}};
  auto it = m.find(t);
  return it == m.end() ? "" : it->second;
}

// ---------------------------------------------------------------- Kapitsa formats and the choice per tensor

struct Fmt {
  int bits = 16, gs = 0;  // bits 16: bf16
  uint32_t iq = 0;        // routed experts kept in llama.cpp's blocks: their type (iq_gpu_type)
  bool dense() const { return bits == 16 && !iq; }
  bool operator==(const Fmt& o) const { return bits == o.bits && gs == o.gs && iq == o.iq; }
  bool operator!=(const Fmt& o) const { return !(*this == o); }
};
const Fmt kBf16{16, 0}, k3g64{3, 64}, k4g32{4, 32}, k4g64{4, 64}, k5g64{5, 64}, k6g64{6, 64}, k8g64{8, 64};
Fmt iq_fmt(uint32_t t) { return Fmt{iq_bits(t), 256, t}; }

std::string fmt_name(Fmt f) {
  if (f.iq) return type_name(f.iq) + " (as stored)";
  return f.dense() ? "bf16" : std::to_string(f.bits) + "-bit group " + std::to_string(f.gs);
}

// the finer of two formats (more bits; a smaller group at equal bits)
Fmt finer(Fmt a, Fmt b) {
  if (a.bits != b.bits) return a.bits > b.bits ? a : b;
  return a.gs <= b.gs ? a : b;
}

bool refused(uint32_t t) { return !can_dequantize(t); }

// Precision classes of the GGUF types: 1-2-bit (IQ1_*, IQ2_*, Q2_K), 3-bit (IQ3_*, Q3_K), 4-bit (Q4_x, IQ4_x, MXFP4,
// NVFP4), 5-bit (Q5_x), 6-8-bit (Q6_K, Q8_0), float.
enum Cls { kC2, kC3, kC4, kC5, kC8, kCF };
Cls type_class(uint32_t t) {
  switch (t) {
    case kF32: case kF16: case kBF16: return kCF;
    case kQ8_0: case kQ6_K: return kC8;
    case kQ5_0: case kQ5_1: case kQ5_K: return kC5;
    case kQ3_K: case kIQ3_XXS: case kIQ3_S: return kC3;
    case kQ2_K: case kIQ2_XXS: case kIQ2_XS: case kIQ2_S: case kIQ1_S: case kIQ1_M: return kC2;
    default: return kC4;
  }
}

// A linear outside the MoE experts, by its own type: Q4_x keep their codes (4-bit group 32); 1-3-bit types take the
// smallest format that adds little to their error (measured: 3-bit types +9-21% at 4-bit group 32, 3-bit group 64
// would add 83-144%; 1-2-bit types +4-31% at 3-bit group 64).
Fmt own_format(uint32_t t) {
  switch (t) {
    case kF32: case kF16: case kBF16: return kBf16;
    case kQ4_0: case kQ4_1: case kQ4_K: return k4g32;
    case kQ8_0: case kQ6_K: return k8g64;
    case kQ5_0: case kQ5_1: case kQ5_K: return k6g64;
    default:
      switch (type_class(t)) {
        case kC3: return k4g32;
        case kC2: return k3g64;
        default: return k5g64;  // IQ4_NL, IQ4_XS, MXFP4, NVFP4
      }
  }
}

// A layer's routed experts, by the class of their least precise tensor. Kapitsa's MoE kernels run 3/4/5/6/8-bit group
// 64 and bf16 experts (4-bit group 32 only in the MTP head): 4-bit types take 5-bit group 64 (4-bit group 64 would
// double their error), 3-bit types 4-bit group 64, 1-2-bit types 3-bit group 64.
Fmt expert_format(Cls c) {
  switch (c) {
    case kC2: return k3g64;
    case kC3: return k4g64;
    case kC4: return k5g64;
    case kC5: return k6g64;
    case kC8: return k8g64;
    default: return kBf16;
  }
}

// ---------------------------------------------------------------- the model's shape (from the metadata)

struct Hp {
  std::string arch;
  bool moe = false, tied = false;
  int64_t H = 0, layers = 0, nextn = 0, hq = 0, hkv = 0, head_dim = 0, rot = 0, ctx = 0;
  int64_t conv = 0, dk = 0, hk = 0, hv = 0, dv = 0, interval = 4, ffn = 0;
  int64_t E = 0, topk = 0, moe_I = 0, shared_I = 0, vocab = 0;
  double eps = 1e-6, theta = 1e7;
  std::vector<int64_t> sections;
  std::vector<bool> recurrent;
  bool flint = false;                // Recursive's Flint (arch "flint")
  int64_t window = 0, dense_lead = 0;  // Flint: the sliding window, the leading dense-MLP layers
  std::vector<bool> swa, rope;         // Flint, per layer: sliding-window attention, RoPE
};

// an integer key, or the largest of an integer array (per-layer values)
int64_t ival(const File& f, const std::string& key, int64_t def = -1) {
  if (!f.has(key)) {
    if (def >= 0) return def;
    throw Error(f.parts()[0] + " has no " + key + " in its metadata; it is not a complete Qwen3.5 GGUF.");
  }
  const json& v = f.meta()[key];
  if (v.is_number_integer()) return v.get<int64_t>();
  if (v.is_array()) {
    int64_t m = 0;
    for (const auto& x : v)
      if (x.is_number_integer()) m = std::max(m, x.get<int64_t>());
    return m;
  }
  throw Error(f.parts()[0] + ": " + key + " is not a number.");
}

Hp read_hp(const File& f) {
  Hp h;
  h.arch = f.str("general.architecture");
  const std::string a = h.arch + ".";
  h.moe = h.arch == "qwen35moe";
  h.H = ival(f, a + "embedding_length");
  const int64_t blocks = ival(f, a + "block_count");
  h.nextn = ival(f, a + "nextn_predict_layers", 0);
  h.layers = blocks - h.nextn;
  h.hq = ival(f, a + "attention.head_count");
  h.hkv = ival(f, a + "attention.head_count_kv");
  h.head_dim = ival(f, a + "attention.key_length", h.hq ? h.H / h.hq : 0);
  h.rot = ival(f, a + "rope.dimension_count", h.head_dim / 4);
  h.ctx = ival(f, a + "context_length", 262144);
  h.eps = f.real(a + "attention.layer_norm_rms_epsilon", 1e-6);
  h.theta = f.real(a + "rope.freq_base", 1e7);
  for (const char* k : {"rope.dimension_sections", "rope.mrope_section", "mrope_sections"})
    if (f.has(a + k) && f.meta()[a + k].is_array()) {
      for (const auto& x : f.meta()[a + k])
        if (x.is_number_integer()) h.sections.push_back(x.get<int64_t>());
      break;
    }
  if (h.sections.empty()) h.sections = {11, 11, 10, 0};
  h.conv = ival(f, a + "ssm.conv_kernel");
  h.dk = ival(f, a + "ssm.state_size");
  h.hk = ival(f, a + "ssm.group_count");
  h.hv = ival(f, a + "ssm.time_step_rank");
  const int64_t inner = ival(f, a + "ssm.inner_size");
  h.dv = h.hv ? inner / h.hv : 0;
  h.interval = ival(f, a + "full_attention_interval", 4);
  if (f.has(a + "attention.recurrent_layers") && f.meta()[a + "attention.recurrent_layers"].is_array()) {
    for (const auto& x : f.meta()[a + "attention.recurrent_layers"]) h.recurrent.push_back(x.is_boolean() ? x.get<bool>() : x.get<int64_t>() != 0);
    h.recurrent.resize(size_t(std::max<int64_t>(h.layers, 0)), false);
  } else {
    for (int64_t i = 0; i < h.layers; ++i) h.recurrent.push_back(h.interval > 0 && (i + 1) % h.interval != 0);
  }
  if (h.moe) {
    h.E = ival(f, a + "expert_count");
    h.topk = ival(f, a + "expert_used_count");
    h.moe_I = ival(f, a + "expert_feed_forward_length");
    h.shared_I = ival(f, a + "expert_shared_feed_forward_length", 0);
  } else {
    h.ffn = ival(f, a + "feed_forward_length");
  }
  const Tensor* emb = f.find("token_embd.weight");
  if (!emb || emb->ne.size() != 2) throw Error(f.parts()[0] + " has no token_embd.weight tensor.");
  h.vocab = int64_t(emb->ne[1]);
  h.tied = !f.find("output.weight");
  if (h.H <= 0 || h.layers <= 0 || h.hq <= 0 || h.hkv <= 0 || h.hk <= 0 || h.hv <= 0 || h.hv % h.hk ||
      int64_t(emb->ne[0]) != h.H)
    throw Error(f.parts()[0] + ": inconsistent Qwen3.5 metadata (layers, heads or sizes).");
  return h;
}

// The checks Kapitsa's model loader makes (src/model/qwen35.cpp), before a long conversion.
void check_runs(const Hp& h, const std::string& file) {
  std::string why;
  if (h.dk != 128 || h.dv != 128) why = "DeltaNet head size " + std::to_string(h.dk) + "/" + std::to_string(h.dv) + " (128 needed)";
  else if (h.head_dim != 256) why = "attention head size " + std::to_string(h.head_dim) + " (256 needed)";
  else if (h.hq % h.hkv || h.hq / h.hkv > 8) why = "a grouped-query ratio above 8";
  else if (h.conv != 4) why = "a convolution of " + std::to_string(h.conv) + " (4 needed)";
  else if (h.moe && (h.moe_I % 512 || h.shared_I % 512)) why = "expert sizes that are not multiples of 512";
  else if (h.moe && (h.topk + 1) * std::max(h.moe_I, h.shared_I) > 4608) why = "experts too large for the decode kernels";
  else if (h.moe && (h.E != 256 || h.topk > 8)) why = std::to_string(h.E) + " experts, " + std::to_string(h.topk) + " per token (256 and at most 8 run)";
  else if (h.moe && (h.H > 4096 || h.hv * h.dv > 4096 || h.hq * h.head_dim > 4096)) why = "a hidden size above 4096 for a MoE model";
  else if (h.H % 64) why = "a hidden size that is not a multiple of 64";
  if (!why.empty()) throw Error(file + " is a Qwen3.5-family model with " + why + ", which Kapitsa does not run.");
}

// Flint's shape (the GGUF architecture "flint")
Hp read_hp_flint(const File& f) {
  Hp h;
  h.arch = "flint";
  h.flint = h.moe = true;
  const std::string a = "flint.", file = base_name(f.parts()[0]);
  h.H = ival(f, a + "embedding_length");
  h.layers = ival(f, a + "block_count");
  h.hq = ival(f, a + "attention.head_count");
  h.hkv = ival(f, a + "attention.head_count_kv");
  h.head_dim = ival(f, a + "attention.key_length", h.hq ? h.H / h.hq : 0);
  h.ctx = ival(f, a + "context_length", 262144);
  h.eps = f.real(a + "attention.layer_norm_rms_epsilon", 1e-6);
  h.theta = f.real(a + "rope.freq_base", 1e4);
  h.ffn = ival(f, a + "feed_forward_length");
  h.E = ival(f, a + "expert_count");
  h.topk = ival(f, a + "expert_used_count");
  h.moe_I = ival(f, a + "expert_feed_forward_length");
  h.shared_I = ival(f, a + "expert_shared_feed_forward_length");
  h.window = ival(f, a + "attention.sliding_window");
  h.dense_lead = ival(f, a + "leading_dense_block_count", 0);
  for (auto [key, out] : {std::pair{"attention.sliding_window_pattern", &h.swa}, {"attention.rope_pattern", &h.rope}}) {
    const json& v = f.meta().value(a + key, json());
    if (!v.is_array() || int64_t(v.size()) != h.layers) throw Error(file + " has no per-layer flint." + key + ".");
    for (const auto& x : v) out->push_back(x.is_boolean() ? x.get<bool>() : x.get<int64_t>() != 0);
  }
  // the routing Kapitsa's Flint runs: sigmoid scores, the top k renormalized, no scale
  if (ival(f, a + "expert_gating_func", 2) != 2 || !f.flag(a + "expert_weights_norm", true) ||
      f.real(a + "expert_weights_scale", 1.0) != 1.0 || ival(f, a + "attention.value_length", h.head_dim) != h.head_dim)
    throw Error(file + ": a Flint variant (routing or attention) Kapitsa does not run.");
  const Tensor* emb = f.find("token_embd.weight");
  if (!emb || emb->ne.size() != 2 || int64_t(emb->ne[0]) != h.H) throw Error(file + " has no token_embd.weight tensor.");
  h.vocab = int64_t(emb->ne[1]);
  h.tied = !f.find("output.weight");
  return h;
}

// The shapes Kapitsa's Flint runs (model/flint.cpp) with IQ experts (src/kernels/wfmt_iq.metal), before a conversion
void check_runs_flint(const Hp& h, const std::string& file) {
  if (h.H % 512 || h.H > 4096 || h.head_dim != 128 || h.hq <= 0 || h.hkv <= 0 || h.hq % h.hkv || h.hq / h.hkv > 8 ||
      h.E != 256 || h.topk < 1 || h.topk > 8 || h.window <= 0 || h.moe_I % 256 || h.shared_I % 64 || h.ffn % 64 ||
      (h.topk + 1) * std::max(h.moe_I, h.shared_I) > 7168 || h.tied || h.dense_lead >= h.layers)
    throw Error(file + " is a Flint shape Kapitsa does not run (hidden " + std::to_string(h.H) + ", heads " +
                std::to_string(h.hq) + " / " + std::to_string(h.hkv) + ", " + std::to_string(h.E) + " experts top-" +
                std::to_string(h.topk) + ").");
}

// ---------------------------------------------------------------- what to write

// output row -> source row: a block map (rows in blocks of `block` taken every `stride`, from `offset`), then the
// DeltaNet value-head order (rows from `vfrom` in heads of `hs` rows: grouped output head k * r + v <- tiled source
// head v * nk + k)
struct RowMap {
  uint64_t block = 0, stride = 0, offset = 0;
  int64_t vfrom = -1;
  uint64_t hs = 0, nk = 1, r = 1;
  uint64_t operator()(uint64_t row) const {
    if (vfrom >= 0 && row >= uint64_t(vfrom)) {
      const uint64_t l = row - uint64_t(vfrom), h = l / hs, d = l % hs;
      row = uint64_t(vfrom) + ((h % r) * nk + h / r) * hs + d;
    }
    return block ? row / block * stride + offset + row % block : offset + row;
  }
};

// One tensor written: a linear (its weight; with scales and biases when quantized) or a float parameter.
struct Lin {
  const File* f = nullptr;
  const Tensor* src = nullptr;
  std::vector<float> mult;        // NVFP4's .scale tensor: per expert (or one value)
  uint64_t mult_rows = 0;         // output rows per value of mult
  std::string name;               // a linear's module name (".weight" etc. appended), a parameter's tensor name
  bool param = false;
  std::vector<int64_t> shape;     // output shape (a linear: leading dims and K)
  uint64_t rows = 0, K = 0;
  RowMap map;
  int64_t cperm_hs = 0;           // > 0: columns in value-head order (out_proj), heads of this many columns
  uint64_t cperm_nk = 1;          // ... of this many key-head groups
  bool log_neg = false;           // ssm_a -> A_log
  Fmt fmt;
  bool mtp = false;               // goes to mtp.safetensors
  uint64_t srb = 0;               // bytes of a source row (K values)
  int file = 0;
  uint64_t off_w = 0, off_s = 0, off_b = 0;
  uint64_t row_w() const { return fmt.iq ? K / 256 * iq_block_bytes(fmt.iq) : fmt.dense() ? K * 2 : K * uint64_t(fmt.bits) / 8; }
  uint64_t row_s() const { return fmt.dense() || fmt.iq ? 0 : K / uint64_t(fmt.gs) * 2; }
  uint64_t src_bytes() const { return rows * srb; }
};

// bytes of K values of a tensor read as rows of K (a parameter may be read with another row length than ne[0])
uint64_t row_bytes_of(const Tensor& t, uint64_t K) {
  if (K == t.ne[0]) return t.row_bytes();
  if (t.type == kF32) return K * 4;
  if (t.type == kF16 || t.type == kBF16) return K * 2;
  throw Error(t.name + ": a " + type_name(t.type) + " tensor where a float parameter was expected.");
}

struct Plan {
  Hp hp;
  // A mostly 4-bit-or-better file (unsloth's UD mixes) with some 1-3-bit tensors: those take the file's majority
  // format, so a layer keeps one format (and no tensor drops below the rest of the model).
  bool follow = false;
  Cls major = kCF;      // the class holding most of the trunk's linear weights
  Fmt major_fmt;        // the most common format of the trunk's 4-bit-or-better dense linears
  std::map<std::string, int> followed;  // type -> tensors that took major_fmt
  std::vector<Lin> lins;
  std::vector<std::string> files;   // output safetensors names; the last is mtp.safetensors when has_mtp
  bool has_mtp = false;
  std::string mtp_source;
  std::map<std::string, int> conv_count;  // "Q4_K -> 4-bit group 32" -> tensors
  std::set<std::string> low, notes;
  bool iq = false;  // routed experts kept as stored (Fmt::iq): they go to experts*.safetensors
};

struct Builder {
  Plan& P;
  const File& F;      // the model
  const File* M;      // where the MTP block is (F, a separate file, or none)
  std::set<std::string> used;
  Builder(Plan& p, const File& f, const File* m) : P(p), F(f), M(m) {}

  const Tensor* get(const File& f, const std::string& n, bool required = true) {
    const Tensor* t = f.find(n);
    if (!t && required)
      throw Error(f.parts()[0] + " has no tensor " + n + "; it is not a complete Qwen3.5 GGUF (or a variant Kapitsa "
                  "does not know).");
    if (t) {
      used.insert((&f == &F ? "" : "mtp:") + n);
      if (refused(t->type))
        throw Error(base_name(f.parts()[0]) + " stores " + n + " as " + type_name(t->type) + ", which Kapitsa does not "
                    "convert: it reads F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K-Q6_K, IQ1_S/IQ1_M, IQ2_XXS/IQ2_XS/"
                    "IQ2_S, IQ3_XXS/IQ3_S, IQ4_NL/IQ4_XS, MXFP4 and NVFP4 (not the ternary TQ1_0/TQ2_0). Use another "
                    "quantization of the model, or Kapitsa's own model of this size (kapitsa --help).");
    }
    return t;
  }
  void check_ne(const Tensor* t, std::initializer_list<int64_t> ne) {
    std::vector<uint64_t> want(ne.begin(), ne.end());
    if (t->ne != want) {
      std::string a, b;
      for (auto x : t->ne) a += (a.empty() ? "" : ", ") + std::to_string(x);
      for (auto x : want) b += (b.empty() ? "" : ", ") + std::to_string(x);
      throw Error(t->name + " has dims {" + a + "}, expected {" + b + "} from the metadata.");
    }
  }
  // the NVFP4 scale beside a weight (one value, or one per expert): `experts` values expected, a stacked output's
  // rows per expert in `rows_per_expert` (0: one value for every row), `pick`: the value of one expert only
  void attach_scale(const File& f, Lin& L, uint64_t experts, uint64_t rows_per_expert, int64_t pick) {
    std::string n = L.src->name;
    if (!ends_with(n, ".weight")) return;
    n = n.substr(0, n.size() - 7) + ".scale";
    const Tensor* s = f.find(n);
    if (!s) return;
    used.insert((&f == &F ? "" : "mtp:") + n);
    if (s->type != kF32 || (s->elements() != 1 && s->elements() != experts))
      throw Error(n + ": a scale tensor Kapitsa does not read.");
    L.mult.resize(size_t(s->elements()));
    f.read(*s, 0, L.mult.data(), L.mult.size() * 4);
    L.mult_rows = 0;
    if (L.mult.size() > 1) {
      if (pick >= 0) L.mult = {L.mult[size_t(pick)]};
      else L.mult_rows = rows_per_expert;
    }
  }

  void count(const Lin& L) {
    std::string k = type_name(L.src->type) + " -> " + fmt_name(L.fmt);
    if (L.fmt == k4g32 && exact4(L.src->type) && L.mult.empty()) k += " (codes as stored)";
    else if (L.fmt.dense() && L.src->type == kBF16) k += " (as stored)";
    ++P.conv_count[(L.mtp ? "MTP head: " : "") + k];
    if (type_class(L.src->type) <= kC3 && !P.follow && !L.fmt.iq && !P.hp.flint) P.low.insert(type_name(L.src->type));
  }

  // a float parameter, bf16
  void param(const File& f, const std::string& src, const std::string& out, std::vector<int64_t> shape,
             uint64_t rows, uint64_t K, RowMap map = {}, bool log_neg = false, bool mtp = false) {
    Lin L;
    L.f = &f;
    L.src = get(f, src);
    if (L.src->elements() != rows * K) throw Error(src + ": unexpected size.");
    L.name = out;
    L.param = true;
    L.shape = std::move(shape);
    L.rows = rows, L.K = K, L.map = map, L.log_neg = log_neg;
    L.fmt = kBf16;
    L.mtp = mtp;
    L.srb = row_bytes_of(*L.src, K);
    P.lins.push_back(std::move(L));
  }

  // a linear [N, K] (or rows of a bigger tensor through `map`)
  // (scale_experts / scale_rows / pick: how the NVFP4 scale of a stacked expert tensor maps to these rows)
  Lin& linear(const File& f, const Tensor* t, const std::string& out, uint64_t N, uint64_t K, Fmt fmt,
              RowMap map = {}, int64_t cperm_hs = 0, bool mtp = false, uint64_t scale_experts = 1,
              uint64_t scale_rows = 0, int64_t pick = -1) {
    Lin L;
    L.f = &f;
    L.src = t;
    L.name = out;
    L.shape = {int64_t(N), int64_t(K)};
    L.rows = N, L.K = K, L.map = map, L.cperm_hs = cperm_hs;
    L.cperm_nk = uint64_t(P.hp.hk);
    L.fmt = fmt;
    if (!fmt.dense() && K % uint64_t(fmt.gs)) L.fmt = kBf16;  // no group fits: keep it exact
    L.mtp = mtp;
    if (K != t->ne[0]) throw Error("internal: " + t->name + " read with rows of " + std::to_string(K));
    L.srb = t->row_bytes();
    attach_scale(f, L, scale_experts, scale_rows, pick);
    count(L);
    P.lins.push_back(std::move(L));
    return P.lins.back();
  }

  // the format of a linear outside the experts (a 1-3-bit one takes the file's majority format in a mixed file)
  Fmt dense_format(uint32_t t) {
    if (P.follow && type_class(t) <= kC3) {
      ++P.followed[type_name(t)];
      return P.major_fmt;
    }
    return own_format(t);
  }
  Cls expert_class(uint32_t t) {
    const Cls c = type_class(t);
    if (P.follow && c <= kC3) {
      ++P.followed[type_name(t)];
      return P.major;
    }
    return c;
  }

  RowMap vmap(int64_t from, uint64_t hs) const {
    RowMap m;
    if (P.hp.hk != P.hp.hv) {
      m.vfrom = from;
      m.hs = hs, m.nk = uint64_t(P.hp.hk), m.r = uint64_t(P.hp.hv / P.hp.hk);
    }
    return m;
  }

  // attention block (trunk layer or the MTP layer): q (with its gate) / k / v / o and the q / k norms
  void attention(const File& f, const std::string& b, const std::string& out, bool mtp) {
    const Hp& h = P.hp;
    const uint64_t H = uint64_t(h.H), qd = uint64_t(h.hq * h.head_dim), kd = uint64_t(h.hkv * h.head_dim);
    auto fmt = [&](const Tensor* t) { return mtp ? k8g64 : dense_format(t->type); };
    if (const Tensor* q = get(f, b + "attn_q.weight", false)) {
      check_ne(q, {int64_t(H), int64_t(2 * qd)});
      linear(f, q, out + "self_attn.q_proj", 2 * qd, H, fmt(q), {}, 0, mtp);
      const Tensor* k = get(f, b + "attn_k.weight");
      check_ne(k, {int64_t(H), int64_t(kd)});
      linear(f, k, out + "self_attn.k_proj", kd, H, fmt(k), {}, 0, mtp);
      const Tensor* v = get(f, b + "attn_v.weight");
      check_ne(v, {int64_t(H), int64_t(kd)});
      linear(f, v, out + "self_attn.v_proj", kd, H, fmt(v), {}, 0, mtp);
    } else {  // fused q|k|v (convert_hf_to_gguf.py --fuse-qkv)
      const Tensor* t = get(f, b + "attn_qkv.weight");
      check_ne(t, {int64_t(H), int64_t(2 * qd + 2 * kd)});
      RowMap m;
      linear(f, t, out + "self_attn.q_proj", 2 * qd, H, fmt(t), m, 0, mtp);
      m.offset = 2 * qd;
      linear(f, t, out + "self_attn.k_proj", kd, H, fmt(t), m, 0, mtp);
      m.offset = 2 * qd + kd;
      linear(f, t, out + "self_attn.v_proj", kd, H, fmt(t), m, 0, mtp);
    }
    const Tensor* o = get(f, b + "attn_output.weight");
    check_ne(o, {int64_t(qd), int64_t(H)});
    linear(f, o, out + "self_attn.o_proj", H, qd, fmt(o), {}, 0, mtp);
    param(f, b + "attn_q_norm.weight", out + "self_attn.q_norm.weight", {h.head_dim}, 1, uint64_t(h.head_dim), {}, false, mtp);
    param(f, b + "attn_k_norm.weight", out + "self_attn.k_norm.weight", {h.head_dim}, 1, uint64_t(h.head_dim), {}, false, mtp);
  }

  void deltanet(const std::string& b, const std::string& out) {
    const Hp& h = P.hp;
    const uint64_t H = uint64_t(h.H), kd = uint64_t(h.hk * h.dk), vd = uint64_t(h.hv * h.dv), C = 2 * kd + vd;
    const Tensor* qkv = get(F, b + "attn_qkv.weight");
    check_ne(qkv, {int64_t(H), int64_t(C)});
    linear(F, qkv, out + "linear_attn.in_proj_qkv", C, H, dense_format(qkv->type), vmap(int64_t(2 * kd), uint64_t(h.dv)));
    const Tensor* z = get(F, b + "attn_gate.weight");
    check_ne(z, {int64_t(H), int64_t(vd)});
    linear(F, z, out + "linear_attn.in_proj_z", vd, H, dense_format(z->type), vmap(0, uint64_t(h.dv)));
    const Tensor* beta = get(F, b + "ssm_beta.weight");
    check_ne(beta, {int64_t(H), h.hv});
    linear(F, beta, out + "linear_attn.in_proj_b", uint64_t(h.hv), H, dense_format(beta->type), vmap(0, 1));
    const Tensor* alpha = get(F, b + "ssm_alpha.weight");
    check_ne(alpha, {int64_t(H), h.hv});
    linear(F, alpha, out + "linear_attn.in_proj_a", uint64_t(h.hv), H, dense_format(alpha->type), vmap(0, 1));
    const Tensor* o = get(F, b + "ssm_out.weight");
    check_ne(o, {int64_t(vd), int64_t(H)});
    linear(F, o, out + "linear_attn.out_proj", H, vd, dense_format(o->type), {}, h.hk != h.hv ? h.dv : 0);
    const Tensor* cw = get(F, b + "ssm_conv1d.weight");
    check_ne(cw, {h.conv, int64_t(C)});
    param(F, b + "ssm_conv1d.weight", out + "linear_attn.conv1d.weight", {int64_t(C), h.conv, 1}, C, uint64_t(h.conv),
          vmap(int64_t(2 * kd), uint64_t(h.dv)));
    param(F, b + "ssm_a", out + "linear_attn.A_log", {h.hv}, uint64_t(h.hv), 1, vmap(0, 1), true);
    const std::string dt = F.find(b + "ssm_dt.bias") ? b + "ssm_dt.bias" : b + "ssm_dt";  // Ollama's own name
    param(F, dt, out + "linear_attn.dt_bias", {h.hv}, uint64_t(h.hv), 1, vmap(0, 1));
    param(F, b + "ssm_norm.weight", out + "linear_attn.norm.weight", {h.dv}, 1, uint64_t(h.dv));
  }

  void mlp(const File& f, const std::string& b, const std::string& out, bool mtp) {
    const uint64_t H = uint64_t(P.hp.H), I = uint64_t(P.hp.ffn);
    const Tensor* g = get(f, b + "ffn_gate.weight");
    const Tensor* u = get(f, b + "ffn_up.weight");
    const Tensor* d = get(f, b + "ffn_down.weight");
    check_ne(g, {int64_t(H), int64_t(I)});
    check_ne(u, {int64_t(H), int64_t(I)});
    check_ne(d, {int64_t(I), int64_t(H)});
    // the decode kernel reads gate and up in one format: the finer of the two
    const Fmt gu = mtp ? k8g64 : finer(dense_format(g->type), dense_format(u->type));
    linear(f, g, out + "mlp.gate_proj", I, H, gu, {}, 0, mtp);
    linear(f, u, out + "mlp.up_proj", I, H, gu, {}, 0, mtp);
    linear(f, d, out + "mlp.down_proj", H, I, mtp ? k8g64 : dense_format(d->type), {}, 0, mtp);
  }

  // routed + shared experts, router and shared gate. Trunk: stacked switch_mlp tensors in one format per layer;
  // MTP head: numbered experts in 4-bit group 32 (what Kapitsa's MTP loader reads) and 8-bit group 64 for the rest.
  void moe(const File& f, const std::string& b, const std::string& out, bool mtp) {
    const Hp& h = P.hp;
    const uint64_t H = uint64_t(h.H), I = uint64_t(h.moe_I), S = uint64_t(h.shared_I), E = uint64_t(h.E);
    const Tensor* router = get(f, b + "ffn_gate_inp.weight");
    check_ne(router, {int64_t(H), int64_t(E)});
    const Tensor* gu = get(f, b + "ffn_gate_up_exps.weight", false);
    const Tensor* g = gu ? nullptr : get(f, b + "ffn_gate_exps.weight");
    const Tensor* u = gu ? nullptr : get(f, b + "ffn_up_exps.weight");
    const Tensor* d = get(f, b + "ffn_down_exps.weight");
    if (gu) check_ne(gu, {int64_t(H), int64_t(2 * I), int64_t(E)});
    else check_ne(g, {int64_t(H), int64_t(I), int64_t(E)}), check_ne(u, {int64_t(H), int64_t(I), int64_t(E)});
    check_ne(d, {int64_t(I), int64_t(H), int64_t(E)});
    const Tensor* sg = get(f, b + "ffn_gate_shexp.weight");
    const Tensor* su = get(f, b + "ffn_up_shexp.weight");
    const Tensor* sd = get(f, b + "ffn_down_shexp.weight");
    const Tensor* sgate = get(f, b + "ffn_gate_inp_shexp.weight");
    check_ne(sg, {int64_t(H), int64_t(S)});
    check_ne(su, {int64_t(H), int64_t(S)});
    check_ne(sd, {int64_t(S), int64_t(H)});
    if (sgate->elements() != H) throw Error(sgate->name + ": unexpected size.");
    const Tensor* gsrc = gu ? gu : g;
    const Tensor* usrc = gu ? gu : u;
    RowMap gmap, umap;  // fused gate|up: each expert's I gate rows, then its I up rows
    if (gu) gmap.block = umap.block = I, gmap.stride = umap.stride = 2 * I, umap.offset = I;
    if (mtp) {
      linear(f, router, out + "mlp.gate", E, H, k8g64, {}, 0, true);
      linear(f, sgate, out + "mlp.shared_expert_gate", 1, H, k8g64, {}, 0, true);
      linear(f, sg, out + "mlp.shared_expert.gate_proj", S, H, k8g64, {}, 0, true);
      linear(f, su, out + "mlp.shared_expert.up_proj", S, H, k8g64, {}, 0, true);
      linear(f, sd, out + "mlp.shared_expert.down_proj", H, S, k8g64, {}, 0, true);
      for (uint64_t e = 0; e < E; ++e) {
        const std::string x = out + "mlp.experts." + std::to_string(e) + ".";
        RowMap m = gmap;
        if (gu) m.offset = e * 2 * I;
        else m.offset = e * I;
        linear(f, gsrc, x + "gate_proj", I, H, k4g32, m, 0, true, E, 0, int64_t(e));
        if (gu) m.offset = e * 2 * I + I;
        linear(f, usrc, x + "up_proj", I, H, k4g32, m, 0, true, E, 0, int64_t(e));
        linear(f, d, x + "down_proj", H, I, k4g32, RowMap{0, 0, e * H}, 0, true, E, 0, int64_t(e));
      }
      return;
    }
    // trunk, routed experts in IQ2_XXS / IQ2_S / IQ3_S: kept as stored (src/kernels/wfmt_iq.metal reads the blocks; a
    // GGUF's IQ2_XXS experts converted to 3-bit group 64 took 1.7x their bytes, which put Qwen3.6-35B-A3B's unsloth
    // UD-IQ2_XXS at 15 GB instead of 11). Gate and up of one type, in tensors of their own; shapes the kernels take.
    // IQ3_XXS and IQ1_S experts too with KAPITSA_IMPORT_IQ_NEW=1; by default they are still converted, until a Qwen file
    // of them has been compared kept against converted (their kernels are checked on synthetic and Flint's blocks).
    auto kept = [](uint32_t t) {
      static const bool fresh = getenv("KAPITSA_IMPORT_IQ_NEW") && atoi(getenv("KAPITSA_IMPORT_IQ_NEW")) > 0;
      return iq_gpu_type(t) && (fresh || (t != kIQ3_XXS && t != kIQ1_S));
    };
    if (!gu && kept(g->type) && u->type == g->type && kept(d->type) && H % 512 == 0 && I % 256 == 0 &&
        !getenv("KAPITSA_IMPORT_IQ_CONVERT")) {
      auto stacked_iq = [&](const Tensor* t, const std::string& n, uint64_t N, uint64_t K) {
        Lin& L = linear(f, t, out + "mlp.switch_mlp." + n, E * N, K, iq_fmt(t->type));
        L.shape = {int64_t(E), int64_t(N), int64_t(K)};
      };
      stacked_iq(g, "gate_proj", I, H);
      stacked_iq(u, "up_proj", I, H);
      stacked_iq(d, "down_proj", H, I);
      P.iq = true;
      linear(f, sg, out + "mlp.shared_expert.gate_proj", S, H, k8g64);
      linear(f, su, out + "mlp.shared_expert.up_proj", S, H, k8g64);
      linear(f, sd, out + "mlp.shared_expert.down_proj", H, S, k8g64);
      linear(f, router, out + "mlp.gate", E, H, dense_format(router->type));
      linear(f, sgate, out + "mlp.shared_expert_gate", 1, H, dense_format(sgate->type));
      return;
    }
    // trunk: the layer's experts in one format, from their least precise tensor
    const Cls cg = expert_class(gsrc->type), cu = gu ? cg : expert_class(usrc->type), cd = expert_class(d->type);
    const Cls cls = std::min({cg, cu, cd});
    const Fmt ef = expert_format(cls);
    if (cls == kC4) {
      const uint32_t weakest = cg == cls ? gsrc->type : cu == cls ? usrc->type : d->type;
      P.notes.insert("MoE experts (" + type_name(weakest) + ") become " + fmt_name(ef) + ": Kapitsa's MoE kernels have no "
                     "group-32 form, and one bit fewer would about double the GGUF's error (this adds about a third)");
    }
    auto stacked = [&](const Tensor* t, const std::string& n, uint64_t N, uint64_t K, RowMap m) {
      Lin& L = linear(f, t, out + "mlp.switch_mlp." + n, E * N, K, ef, m, 0, false, E, N);
      L.shape = {int64_t(E), int64_t(N), int64_t(K)};
    };
    stacked(gsrc, "gate_proj", I, H, gmap);
    stacked(usrc, "up_proj", I, H, umap);
    stacked(d, "down_proj", H, I, {});
    // the shared expert: 8-bit group 64 (a kernel pair with every expert format), bf16 beside bf16 experts
    const Fmt sf = ef.dense() ? kBf16 : k8g64;
    linear(f, sg, out + "mlp.shared_expert.gate_proj", S, H, sf);
    linear(f, su, out + "mlp.shared_expert.up_proj", S, H, sf);
    linear(f, sd, out + "mlp.shared_expert.down_proj", H, S, sf);
    linear(f, router, out + "mlp.gate", E, H, dense_format(router->type));
    linear(f, sgate, out + "mlp.shared_expert_gate", 1, H, dense_format(sgate->type));
  }

  void build() {
    const Hp& h = P.hp;
    const std::string root = "language_model.model.";
    const Tensor* emb = get(F, "token_embd.weight");
    linear(F, emb, root + "embed_tokens", uint64_t(h.vocab), uint64_t(h.H), dense_format(emb->type));
    if (!h.tied) {
      const Tensor* o = get(F, "output.weight");
      check_ne(o, {h.H, h.vocab});
      linear(F, o, "language_model.lm_head", uint64_t(h.vocab), uint64_t(h.H), dense_format(o->type));
    }
    param(F, "output_norm.weight", root + "norm.weight", {h.H}, 1, uint64_t(h.H));
    for (int64_t i = 0; i < h.layers; ++i) {
      const std::string b = "blk." + std::to_string(i) + ".", out = root + "layers." + std::to_string(i) + ".";
      param(F, b + "attn_norm.weight", out + "input_layernorm.weight", {h.H}, 1, uint64_t(h.H));
      param(F, b + "post_attention_norm.weight", out + "post_attention_layernorm.weight", {h.H}, 1, uint64_t(h.H));
      if (h.recurrent[size_t(i)]) deltanet(b, out);
      else attention(F, b, out, false);
      if (h.moe) moe(F, b, out, false);
      else mlp(F, b, out, false);
    }
    if (!M) return;
    // the MTP block: block `layers` of the file that has it
    const std::string b = "blk." + std::to_string(h.layers) + ".", out = "mtp.layers.0.";
    const size_t first = P.lins.size();
    param(*M, b + "nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", {h.H}, 1, uint64_t(h.H), {}, false, true);
    param(*M, b + "nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", {h.H}, 1, uint64_t(h.H), {}, false, true);
    if (M->find(b + "nextn.shared_head_norm.weight"))
      param(*M, b + "nextn.shared_head_norm.weight", "mtp.norm.weight", {h.H}, 1, uint64_t(h.H), {}, false, true);
    else  // llama.cpp falls back to the trunk's final norm
      param(F, "output_norm.weight", "mtp.norm.weight", {h.H}, 1, uint64_t(h.H), {}, false, true);
    const Tensor* fc = get(*M, b + "nextn.eh_proj.weight");
    check_ne(fc, {2 * h.H, h.H});
    linear(*M, fc, "mtp.fc", uint64_t(h.H), uint64_t(2 * h.H), k8g64, {}, 0, true);
    param(*M, b + "attn_norm.weight", out + "input_layernorm.weight", {h.H}, 1, uint64_t(h.H), {}, false, true);
    param(*M, b + "post_attention_norm.weight", out + "post_attention_layernorm.weight", {h.H}, 1, uint64_t(h.H), {},
          false, true);
    attention(*M, b, out, true);
    if (h.moe) moe(*M, b, out, true);
    else mlp(*M, b, out, true);
    if (M->find(b + "nextn.embed_tokens.weight") || M->find(b + "nextn.shared_head_head.weight")) {
      // a head with its own embedding / output layer: Kapitsa's MTP decoding uses the trunk's
      P.lins.resize(first);
      P.notes.insert("MTP head left out: it has its own embedding or output layer, which Kapitsa's MTP decoding does not use");
      return;
    }
    P.has_mtp = true;
  }

  // Flint: Recursive's MLX names (model/flint.cpp). Every linear in 4-bit group 64, the format Flint's kernels run (its
  // shared expert too: the IQ kernels' _s4 forms; Recursive's export has it in 4 bits), but the LM head in 6 bits (its
  // qmvf_b6 matvec) and the embedding in 3 (the file's Q3_K), the router as stored (bf16), the routed experts as stored
  // (IQ blocks): 11.73 GB for unsloth's UD-IQ2_XXS, which a 16 GB Mac's 12.2 GB budget holds with one 8K-token
  // conversation (engine.cpp, fits_small). Against llama.cpp on the file itself (CPU, 192 decode rows), all in 4 bits
  // cost KL 0.29 and +0.15 nats a token; of that, the head kept as stored took back 0.15 and 0.09 (6 bits here: 0.23 and
  // +0.094), the attention 0.13 and 0.04, the shared expert 0.03; the 3-bit embedding adds 0.006 nats.
  void build_flint() {
    const Hp& h = P.hp;
    const uint64_t H = uint64_t(h.H), QD = uint64_t(h.hq * h.head_dim), KD = uint64_t(h.hkv * h.head_dim);
    const uint64_t E = uint64_t(h.E), I = uint64_t(h.moe_I), S = uint64_t(h.shared_I), FF = uint64_t(h.ffn);
    linear(F, get(F, "token_embd.weight"), "model.embed_tokens", uint64_t(h.vocab), H, k3g64);
    const Tensor* o = get(F, "output.weight");
    check_ne(o, {h.H, h.vocab});
    linear(F, o, "lm_head", uint64_t(h.vocab), H, k6g64);
    param(F, "output_norm.weight", "model.norm.weight", {h.H}, 1, H);
    auto lin = [&](const std::string& src, const std::string& out, uint64_t N, uint64_t K) {
      const Tensor* t = get(F, src);
      check_ne(t, {int64_t(K), int64_t(N)});
      linear(F, t, out, N, K, k4g64);
    };
    for (int64_t i = 0; i < h.layers; ++i) {
      const std::string b = "blk." + std::to_string(i) + ".", out = "model.layers." + std::to_string(i) + ".";
      param(F, b + "attn_norm.weight", out + "input_layernorm.weight", {h.H}, 1, H);
      param(F, b + "ffn_norm.weight", out + "post_attention_layernorm.weight", {h.H}, 1, H);
      param(F, b + "attn_q_norm.weight", out + "self_attn.q_norm.weight", {h.head_dim}, 1, uint64_t(h.head_dim));
      param(F, b + "attn_k_norm.weight", out + "self_attn.k_norm.weight", {h.head_dim}, 1, uint64_t(h.head_dim));
      param(F, b + "attn_sinks.weight", out + "self_attn.softmax_offset", {h.hq}, 1, uint64_t(h.hq));
      lin(b + "attn_q.weight", out + "self_attn.q_proj", QD, H);
      lin(b + "attn_k.weight", out + "self_attn.k_proj", KD, H);
      lin(b + "attn_v.weight", out + "self_attn.v_proj", KD, H);
      lin(b + "attn_output.weight", out + "self_attn.o_proj", H, QD);
      if (i < h.dense_lead) {
        lin(b + "ffn_gate.weight", out + "mlp.gate_proj", FF, H);
        lin(b + "ffn_up.weight", out + "mlp.up_proj", FF, H);
        lin(b + "ffn_down.weight", out + "mlp.down_proj", H, FF);
        continue;
      }
      const Tensor* router = get(F, b + "ffn_gate_inp.weight");
      check_ne(router, {h.H, h.E});
      linear(F, router, out + "mlp.gate", E, H, kBf16);
      const Tensor* g = get(F, b + "ffn_gate_exps.weight");
      const Tensor* u = get(F, b + "ffn_up_exps.weight");
      const Tensor* d = get(F, b + "ffn_down_exps.weight");
      check_ne(g, {h.H, int64_t(I), h.E});
      check_ne(u, {h.H, int64_t(I), h.E});
      check_ne(d, {int64_t(I), h.H, h.E});
      if (!iq_gpu_type(g->type) || u->type != g->type || !iq_gpu_type(d->type))
        throw Error(base_name(F.parts()[0]) + " stores layer " + std::to_string(i) + "'s routed experts as " +
                    type_name(g->type) + " / " + type_name(d->type) + ": Kapitsa imports Flint GGUFs whose experts are "
                    "in llama.cpp's IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS or IQ3_S blocks (unsloth's UD-IQ files); flint:4bit "
                    "is Recursive's 4-bit model.");
      for (auto [t, n, N, K] : {std::tuple{g, "gate_proj", I, H}, {u, "up_proj", I, H}, {d, "down_proj", H, I}}) {
        Lin& L = linear(F, t, out + "mlp.switch_mlp." + n, E * N, K, iq_fmt(t->type));
        L.shape = {int64_t(E), int64_t(N), int64_t(K)};
      }
      P.iq = true;
      lin(b + "ffn_gate_shexp.weight", out + "mlp.shared_expert.gate_proj", S, H);
      lin(b + "ffn_up_shexp.weight", out + "mlp.shared_expert.up_proj", S, H);
      lin(b + "ffn_down_shexp.weight", out + "mlp.shared_expert.down_proj", H, S);
    }
  }
};

// ---------------------------------------------------------------- tokenizer, config

std::string jstr(const std::string& s) { return json(s).dump(-1, ' ', false, json::error_handler_t::replace); }

// Qwen3.5's pre-tokenizer split (its tokenizer.json; Kapitsa's tokenizer implements the same)
const char* kQwen35Split =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|"
    "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

struct Tok {
  std::string tokenizer_json, tokenizer_config, template_;
  int eos = -1, endoftext = -1;
};

Tok make_tokenizer(const File& f, const Hp& h) {
  const std::string file = base_name(f.parts()[0]);
  const std::string model = h.flint ? "Flint" : "Qwen3.5", want = h.flint ? "flint" : "qwen35";
  if (f.str("tokenizer.ggml.model") != "gpt2")
    throw Error(file + ": tokenizer \"" + f.str("tokenizer.ggml.model") + "\" is not " + model + "'s byte-level BPE.");
  const std::string pre = f.str("tokenizer.ggml.pre");
  if (pre != want)
    throw Error(file + ": its pre-tokenizer is \"" + pre + "\", not " + model + "'s (" + want + "): not " + model +
                "'s tokenizer.");
  const json& toks = f.meta().value("tokenizer.ggml.tokens", json());
  const json& types = f.meta().value("tokenizer.ggml.token_type", json());
  const json& merges = f.meta().value("tokenizer.ggml.merges", json());
  if (!toks.is_array() || !types.is_array() || !merges.is_array() || toks.size() != types.size() || toks.empty() ||
      int64_t(toks.size()) > h.vocab)
    throw Error(file + ": the tokenizer's vocabulary is missing or inconsistent.");
  Tok t;
  t.template_ = f.str("tokenizer.chat_template");
  if (t.template_.empty()) throw Error(file + " has no chat template (tokenizer.chat_template).");
  // vocabulary: normal tokens; added tokens: control (special) and user-defined (not special); unused padding left out
  std::string vocab, added, decoder;
  for (size_t i = 0; i < toks.size(); ++i) {
    const int ty = types[i].is_number_integer() ? types[i].get<int>() : 1;
    const std::string& s = toks[i].get_ref<const std::string&>();
    if (ty == 1 || ty == 6) {
      vocab += (vocab.empty() ? "" : ",") + jstr(s) + ":" + std::to_string(i);
    } else if (ty == 3 || ty == 4) {
      const bool special = ty == 3;
      added += std::string(added.empty() ? "" : ",") + "{\"id\":" + std::to_string(i) + ",\"content\":" + jstr(s) +
               ",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,\"normalized\":false,\"special\":" +
               (special ? "true" : "false") + "}";
      decoder += std::string(decoder.empty() ? "" : ",") + "\"" + std::to_string(i) + "\":{\"content\":" + jstr(s) +
                 ",\"lstrip\":false,\"normalized\":false,\"rstrip\":false,\"single_word\":false,\"special\":" +
                 (special ? "true" : "false") + "}";
      if (s == "<|endoftext|>") t.endoftext = int(i);
    }
  }
  std::string mg;
  mg.reserve(merges.size() * 12);
  for (const auto& m : merges) mg += (mg.empty() ? "" : ",") + jstr(m.is_string() ? m.get<std::string>() : "");
  const std::string bl = "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":false,\"use_regex\":false}";
  auto split = [](const char* re) {
    return "{\"type\":\"Split\",\"pattern\":{\"Regex\":" + jstr(re) + "},\"behavior\":\"Isolated\",\"invert\":false},";
  };
  // Flint: its tokenizer.json's (RecursiveSI/flint-int4) normalizer, pre-tokenizer (the three splits Kapitsa's tokenizer
  // recognizes), post-processor and decoder
  const std::string pre_tok =
      h.flint ? "null,\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[" + split(Tokenizer::kFlintSplit[0]) +
                    split(Tokenizer::kFlintSplit[1]) + split(Tokenizer::kFlintSplit[2]) +
                    "{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"trim_offsets\":true,\"use_regex\":false}]},"
                    "\"post_processor\":null,\"decoder\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true,"
                    "\"trim_offsets\":true,\"use_regex\":true},\"model\":{\"type\":\"BPE\",\"dropout\":null,"
                    "\"unk_token\":null,\"continuing_subword_prefix\":null,\"end_of_word_suffix\":null,"
              : "{\"type\":\"NFC\"},\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[" + split(kQwen35Split) +
                    bl + "]},\"post_processor\":" + bl + ",\"decoder\":" + bl +
                    ",\"model\":{\"type\":\"BPE\",\"dropout\":null,\"unk_token\":null,\"continuing_subword_prefix\":\"\","
                    "\"end_of_word_suffix\":\"\",";
  t.tokenizer_json = "{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[" + added +
                     "],\"normalizer\":" + pre_tok +
                     "\"fuse_unk\":false,\"byte_fallback\":false,\"ignore_merges\":false,\"vocab\":{" + vocab +
                     "},\"merges\":[" + mg + "]}}";
  auto tok_at = [&](const char* key) -> json {
    const int64_t id = f.num(key, -1);
    if (id < 0 || id >= int64_t(toks.size())) return nullptr;
    return toks[size_t(id)];
  };
  t.eos = int(f.num("tokenizer.ggml.eos_token_id", -1));
  if (t.eos < 0 || t.eos >= int(toks.size())) throw Error(file + " names no end-of-sequence token.");
  ojson tc;
  tc["add_bos_token"] = f.flag("tokenizer.ggml.add_bos_token", false);
  if (h.flint) {  // as RecursiveSI/flint-int4's
    tc["add_eos_token"] = f.flag("tokenizer.ggml.add_eos_token", false);
    tc["bos_token"] = tok_at("tokenizer.ggml.bos_token_id");
    tc["clean_up_tokenization_spaces"] = false;
    tc["eos_token"] = toks[size_t(t.eos)];
    tc["model_max_length"] = h.ctx;
    tc["tokenizer_class"] = "PreTrainedTokenizerFast";
  } else {
    tc["add_prefix_space"] = false;
    tc["bos_token"] = tok_at("tokenizer.ggml.bos_token_id");
    tc["clean_up_tokenization_spaces"] = false;
    tc["eos_token"] = toks[size_t(t.eos)];
    tc["errors"] = "replace";
    tc["model_max_length"] = h.ctx;
    tc["pad_token"] = tok_at("tokenizer.ggml.padding_token_id");
    tc["split_special_tokens"] = false;
    tc["tokenizer_class"] = "Qwen2Tokenizer";
    tc["unk_token"] = nullptr;
  }
  tc["chat_template"] = t.template_;
  std::string s = tc.dump(2, ' ', false, json::error_handler_t::replace);
  // added_tokens_decoder spliced in as text (its keys are ids)
  s.insert(1, "\n  \"added_tokens_decoder\": {" + decoder + "},");
  t.tokenizer_config = s + "\n";
  return t;
}

// config.json's quantization: the most common format of the quantized trunk linears, the others per module (IQ blocks
// as stored: mode iq1_s / iq2_xxs / iq2_s / iq3_xxs / iq3_s); null when nothing is quantized
ojson quant_config(const Plan& P) {
  std::map<std::pair<int, int>, int> n;
  for (const Lin& L : P.lins)
    if (!L.param && !L.mtp && !L.fmt.dense() && !L.fmt.iq) ++n[{L.fmt.bits, L.fmt.gs}];
  if (n.empty()) return nullptr;
  auto top = std::max_element(n.begin(), n.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
  ojson q{{"group_size", top.second}, {"bits", top.first}, {"mode", "affine"}};
  for (const Lin& L : P.lins)
    if (L.fmt.iq)
      q[L.name] = ojson{{"group_size", 256}, {"bits", L.fmt.bits}, {"mode", lower(type_name(L.fmt.iq))}};
    else if (!L.param && !L.mtp && !L.fmt.dense() && (L.fmt.bits != top.first || L.fmt.gs != top.second))
      q[L.name] = ojson{{"group_size", L.fmt.gs}, {"bits", L.fmt.bits}};
  return q;
}

ojson source_config(const File& f) {
  ojson src{{"file", base_name(f.parts()[0])}};
  if (const std::string ft = ftype_name(f.num("general.file_type", -1)); !ft.empty()) src["file_type"] = ft;
  if (f.has("general.name")) src["name"] = f.str("general.name");
  return src;
}

// Flint's config.json: the fields of Recursive's export (RecursiveSI/flint-int4) that model/flint.cpp reads
ojson make_config_flint(const Plan& P, const Tok& tok, const File& f) {
  const Hp& h = P.hp;
  ojson c;
  c["architectures"] = json::array({"FlintForCausalLM"});
  c["attention_bias"] = false;
  c["attention_sink"] = true;
  c["bos_token_id"] = f.num("tokenizer.ggml.bos_token_id", -1);
  json eos = f.meta().value("tokenizer.ggml.eog_token_ids", json::array({tok.eos}));
  c["eos_token_id"] = eos;
  c["head_dim"] = h.head_dim;
  c["hidden_act"] = "silu";
  c["hidden_size"] = h.H;
  c["intermediate_size"] = h.ffn;
  json lt = json::array(), dense = json::array(), norope = json::array();
  for (int64_t i = 0; i < h.layers; ++i) {
    lt.push_back(h.swa[size_t(i)] ? "sliding_attention" : "full_attention");
    if (i < h.dense_lead) dense.push_back(i);
    if (!h.rope[size_t(i)]) norope.push_back(i);
  }
  c["layer_types"] = lt;
  c["max_position_embeddings"] = h.ctx;
  c["mlp_only_layers"] = dense;
  c["model_type"] = "flint";
  c["moe_intermediate_size"] = h.moe_I;
  c["no_rope_layer_indices"] = norope;
  c["norm_topk_prob"] = true;
  c["num_attention_heads"] = h.hq;
  c["num_experts"] = h.E;
  c["num_experts_per_tok"] = h.topk;
  c["num_hidden_layers"] = h.layers;
  c["num_key_value_heads"] = h.hkv;
  char eps[32];
  snprintf(eps, sizeof eps, "%g", float(h.eps));
  c["rms_norm_eps"] = std::stod(eps);
  c["rope_parameters"] = ojson{{"rope_theta", h.theta}, {"rope_type", "default"}};
  c["rope_theta"] = h.theta;
  c["router_expert_bias"] = false;
  c["scoring_func"] = "sigmoid";
  c["shared_expert_intermediate_size"] = h.shared_I;
  c["sliding_window"] = h.window;
  c["softmax_type"] = "learnable";
  c["tie_word_embeddings"] = false;
  c["vocab_size"] = h.vocab;
  c["pad_token_id"] = f.num("tokenizer.ggml.padding_token_id", tok.eos);
  if (const ojson q = quant_config(P); !q.is_null()) {
    c["quantization"] = q;
    c["quantization_config"] = q;
  }
  c["converted_from_gguf"] = source_config(f);
  return c;
}

ojson make_config(const Plan& P, const Tok& tok, const File& f) {
  if (P.hp.flint) return make_config_flint(P, tok, f);
  const Hp& h = P.hp;
  ojson t;
  t["attention_bias"] = false;
  t["attn_output_gate"] = true;
  t["dtype"] = "bfloat16";
  t["eos_token_id"] = tok.endoftext >= 0 ? tok.endoftext : tok.eos;
  t["full_attention_interval"] = h.interval;
  t["head_dim"] = h.head_dim;
  t["hidden_act"] = "silu";
  t["hidden_size"] = h.H;
  if (!h.moe) t["intermediate_size"] = h.ffn;
  json lt = json::array();
  for (bool r : h.recurrent) lt.push_back(r ? "linear_attention" : "full_attention");
  t["layer_types"] = lt;
  t["linear_conv_kernel_dim"] = h.conv;
  t["linear_key_head_dim"] = h.dk;
  t["linear_num_key_heads"] = h.hk;
  t["linear_num_value_heads"] = h.hv;
  t["linear_value_head_dim"] = h.dv;
  t["max_position_embeddings"] = h.ctx;
  t["model_type"] = h.moe ? "qwen3_5_moe_text" : "qwen3_5_text";
  if (h.moe) {
    t["moe_intermediate_size"] = h.moe_I;
    t["norm_topk_prob"] = true;
    t["num_experts"] = h.E;
    t["num_experts_per_tok"] = h.topk;
    t["shared_expert_intermediate_size"] = h.shared_I;
  }
  t["mtp_num_hidden_layers"] = P.has_mtp ? 1 : 0;
  t["num_attention_heads"] = h.hq;
  t["num_hidden_layers"] = h.layers;
  t["num_key_value_heads"] = h.hkv;
  char eps[32];
  snprintf(eps, sizeof eps, "%g", float(h.eps));
  t["rms_norm_eps"] = std::stod(eps);
  json sec = json::array();
  for (size_t i = 0; i < std::min<size_t>(3, h.sections.size()); ++i) sec.push_back(h.sections[i]);
  const double prf = double(h.rot) / double(h.head_dim);
  t["rope_parameters"] = ojson{{"mrope_interleaved", true}, {"mrope_section", sec}, {"partial_rotary_factor", prf},
                               {"rope_theta", h.theta}, {"rope_type", "default"}};
  t["partial_rotary_factor"] = prf;
  t["tie_word_embeddings"] = h.tied;
  t["vocab_size"] = h.vocab;
  ojson c;
  c["architectures"] = json::array({h.moe ? "Qwen3_5MoeForConditionalGeneration" : "Qwen3_5ForConditionalGeneration"});
  json eos = json::array({tok.eos});
  if (tok.endoftext >= 0 && tok.endoftext != tok.eos) eos.push_back(tok.endoftext);
  c["eos_token_id"] = eos;
  c["language_model_only"] = true;
  c["model_type"] = h.moe ? "qwen3_5_moe" : "qwen3_5";
  c["text_config"] = t;
  c["tie_word_embeddings"] = h.tied;
  if (const ojson q = quant_config(P); !q.is_null()) {
    c["quantization"] = q;
    c["quantization_config"] = q;
  }
  c["converted_from_gguf"] = source_config(f);
  return c;
}

// ---------------------------------------------------------------- conversion

struct OutFile {
  std::string name;
  int fd = -1;
  uint64_t data0 = 0;  // where the tensor data starts (after the header)
  uint64_t size = 0;
};

class Progress {
 public:
  Progress(uint64_t total, bool show, std::string what) : total_(total), show_(show), what_(std::move(what)) {
    t0_ = std::chrono::steady_clock::now();
    tty_ = isatty(2);
  }
  void add(uint64_t n) {
    done_ += n;
    if (!show_) return;
    const double s = secs();
    const int pct = total_ ? int(100.0 * double(done_) / double(total_)) : 100;
    if (term::fancy()) {  // a bar on the status line
      if (s - last_ < 0.1 && done_ < total_) return;
      last_ = s;
      const double left = done_ ? s * double(total_ - done_) / double(done_) : -1;
      char p[16];
      snprintf(p, sizeof p, "%3d%%", pct);
      std::string tail = std::string(p) + "  " + gb(done_) + term::dim(" of " + gb(total_));
      if (s > 0.5) {
        char r[32];
        snprintf(r, sizeof r, "%.0f MB/s", double(done_) / 1e6 / s);
        tail += "  " + term::dim(r);
      }
      if (done_ < total_ && s > 2) tail += "  " + term::dim(fmt_secs(left) + " left");
      const int bw = std::clamp(term::columns() - 6 - term::width(tail) - 3, 8, 40);
      term::status("    " + term::bar(total_ ? double(done_) / double(total_) : 1, bw) + "  " + tail);
      return;
    }
    if (tty_) {
      if (s - last_ < 0.25 && done_ < total_) return;
      last_ = s;
      const double left = done_ ? s * double(total_ - done_) / double(done_) : -1;
      fprintf(stderr, "\r%s: %3d%% (%s of %s)%s%s   ", what_.c_str(), pct, gb(done_).c_str(), gb(total_).c_str(),
              done_ < total_ && s > 2 ? ", " : "", done_ < total_ && s > 2 ? (fmt_secs(left) + " left").c_str() : "");
      fflush(stderr);
    } else if (pct >= next_) {
      fprintf(stderr, "%s: %d%% (%s of %s)\n", what_.c_str(), pct, gb(done_).c_str(), gb(total_).c_str());
      next_ = pct / 10 * 10 + 10;
    }
  }
  void end() {
    if (show_ && term::fancy()) term::status("");
    else if (show_ && tty_) fprintf(stderr, "\n");
  }
  double secs() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
  }

 private:
  uint64_t total_, done_ = 0;
  bool show_, tty_ = false;
  std::string what_;
  std::chrono::steady_clock::time_point t0_;
  double last_ = -1;
  int next_ = 10;
};

void pwrite_all(int fd, const void* p, size_t n, uint64_t off, const std::string& path) {
  const uint8_t* d = static_cast<const uint8_t*>(p);
  while (n) {
    const ssize_t w = pwrite(fd, d, n, off_t(off));
    if (w <= 0) {
      if (errno == ENOSPC) throw Error("The disk is full while writing " + hub::tilde(path) + ".");
      throw Error("Can't write " + hub::tilde(path) + ": " + strerror(errno) + ".");
    }
    d += w, n -= size_t(w), off += uint64_t(w);
  }
}

template <class F>
void parallel_rows(uint64_t n, int threads, F fn) {
  const int t = int(std::min<uint64_t>(uint64_t(std::max(1, threads)), n));
  if (t <= 1) {
    fn(0, n);
    return;
  }
  std::vector<std::thread> th;
  std::exception_ptr err;
  std::mutex mu;
  const uint64_t per = (n + uint64_t(t) - 1) / uint64_t(t);
  for (int i = 0; i < t; ++i) {
    const uint64_t a = uint64_t(i) * per, b = std::min(n, a + per);
    if (a >= b) break;
    th.emplace_back([&, a, b] {
      try {
        fn(a, b);
      } catch (...) {
        std::lock_guard<std::mutex> lk(mu);
        if (!err) err = std::current_exception();
      }
    });
  }
  for (auto& x : th) x.join();
  if (err) std::rethrow_exception(err);
}

// Converts output rows [r0, r1) of L from the source rows held in `src` (starting at source row s0).
void convert_rows(const Lin& L, const uint8_t* src, uint64_t s0, uint64_t r0, uint64_t r1, uint8_t* w, uint16_t* sc,
                  uint16_t* bi) {
  const uint64_t K = L.K, srb = L.srb, nb = L.fmt.dense() ? 0 : K / uint64_t(L.fmt.gs);
  const uint32_t type = L.src->type;
  std::vector<float> x(K), y(K);
  std::vector<uint8_t> codes(K), c2(K);
  std::vector<float> s32(K / 32 + 1), o32(K / 32 + 1);
  const bool direct = !L.param && L.fmt == k4g32 && exact4(type) && L.mult.empty();
  const uint64_t nk = L.cperm_nk;  // value-head columns (out_proj): key-head groups
  for (uint64_t row = r0; row < r1; ++row) {
    const uint64_t sr = L.map(row);
    const uint8_t* s = src + (sr - s0) * srb;
    uint8_t* wo = w + (row - r0) * L.row_w();
    if (direct) {
      decode4(type, s, codes.data(), s32.data(), o32.data(), K);
      if (L.cperm_hs > 0) {
        const uint64_t hs = uint64_t(L.cperm_hs), heads = K / hs;
        const uint64_t r = heads / nk;
        std::vector<float> s2(s32), o2(o32);
        c2 = codes;
        for (uint64_t h = 0; h < heads; ++h) {
          const uint64_t sh = (h % r) * nk + h / r;
          memcpy(codes.data() + h * hs, c2.data() + sh * hs, hs);
          for (uint64_t g = 0; g < hs / 32; ++g) {
            s32[h * hs / 32 + g] = s2[sh * hs / 32 + g];
            o32[h * hs / 32 + g] = o2[sh * hs / 32 + g];
          }
        }
      }
      pack_codes(codes.data(), K, 4, reinterpret_cast<uint32_t*>(wo));
      for (uint64_t g = 0; g < K / 32; ++g) {
        sc[(row - r0) * nb + g] = to_bf16(s32[g]);
        bi[(row - r0) * nb + g] = to_bf16(o32[g]);
      }
      continue;
    }
    dequantize(type, s, x.data(), K);
    if (!L.mult.empty()) {
      const float m = L.mult_rows ? L.mult[row / L.mult_rows] : L.mult[0];
      for (auto& v : x) v *= m;
    }
    if (L.log_neg)
      for (auto& v : x) v = std::log(-v);
    if (L.cperm_hs > 0) {
      const uint64_t hs = uint64_t(L.cperm_hs), heads = K / hs, r = heads / nk;
      y = x;
      for (uint64_t h = 0; h < heads; ++h) memcpy(x.data() + h * hs, y.data() + ((h % r) * nk + h / r) * hs, hs * 4);
    }
    if (L.fmt.dense()) {
      uint16_t* o = reinterpret_cast<uint16_t*>(wo);
      for (uint64_t i = 0; i < K; ++i) o[i] = to_bf16(x[i]);
      continue;
    }
    quantize(x.data(), K, L.fmt.bits, L.fmt.gs, codes.data(), sc + (row - r0) * nb, bi + (row - r0) * nb);
    pack_codes(codes.data(), K, L.fmt.bits, reinterpret_cast<uint32_t*>(wo));
  }
}

void convert(const Lin& L, const std::vector<OutFile>& files, int threads, Progress& prog) {
  const OutFile& of = files[size_t(L.file)];
  const uint64_t srb = L.srb;
  if (L.fmt.iq) {  // routed experts kept as stored: their bytes (an experts file without the GGUF's clone)
    if (of.fd < 0) return;  // in the clone, where they are already
    std::vector<uint8_t> buf;
    for (uint64_t o = 0; o < L.src->bytes; o += kChunkBytes) {
      if (g_stop) throw Error("Stopped; nothing was kept. The same command starts the conversion again.");
      const uint64_t n = std::min<uint64_t>(kChunkBytes, L.src->bytes - o);
      buf.resize(size_t(n));
      L.f->read(*L.src, o, buf.data(), buf.size());
      pwrite_all(of.fd, buf.data(), buf.size(), of.data0 + L.off_w + o, of.name);
      prog.add(n);
    }
    return;
  }
  // the whole tensor when its rows are reordered by value head (moderate sizes), else chunks of rows
  const bool whole = L.map.vfrom >= 0;
  const uint64_t step = whole ? L.rows : std::max<uint64_t>(1, kChunkBytes / std::max<uint64_t>(srb, 1));
  std::vector<uint8_t> src, w;
  std::vector<uint16_t> sc, bi;
  for (uint64_t r0 = 0; r0 < L.rows; r0 += step) {
    if (g_stop) throw Error("Stopped; nothing was kept. The same command starts the conversion again.");
    const uint64_t r1 = std::min(L.rows, r0 + step);
    uint64_t lo = UINT64_MAX, hi = 0;
    for (uint64_t r = r0; r < r1; ++r) {
      const uint64_t s = L.map(r);
      lo = std::min(lo, s), hi = std::max(hi, s);
    }
    if ((hi + 1) * srb > L.src->bytes) throw Error("internal: row map of " + L.name + " out of range");
    src.resize(size_t((hi - lo + 1) * srb));
    L.f->read(*L.src, lo * srb, src.data(), src.size());
    const uint64_t n = r1 - r0, nb = L.fmt.dense() ? 0 : L.K / uint64_t(L.fmt.gs);
    w.resize(size_t(n * L.row_w()));
    sc.resize(size_t(n * nb)), bi.resize(size_t(n * nb));
    parallel_rows(n, (L.K * n >= 65536) ? threads : 1, [&](uint64_t a, uint64_t b) {
      convert_rows(L, src.data(), lo, r0 + a, r0 + b, w.data() + a * L.row_w(), sc.data() + a * nb, bi.data() + a * nb);
    });
    pwrite_all(of.fd, w.data(), w.size(), of.data0 + L.off_w + r0 * L.row_w(), of.name);
    if (nb) {
      pwrite_all(of.fd, sc.data(), sc.size() * 2, of.data0 + L.off_s + r0 * nb * 2, of.name);
      pwrite_all(of.fd, bi.data(), bi.size() * 2, of.data0 + L.off_b + r0 * nb * 2, of.name);
    }
    prog.add(n * srb);
  }
}

// ---------------------------------------------------------------- sources, names, reuse

struct Source {
  std::string path;          // the file given (any part)
  std::string mtp;           // the separate MTP file, "" if none
  std::string ollama_name;   // "qwen3.5-0.8b" when the file is an Ollama blob with a manifest
};

// An Ollama blob (models/blobs/sha256-<hex>): the model:tag of a manifest that lists it, and the manifest's draft
// (MTP) layer when it has one.
void ollama_lookup(Source& s) {
  const std::string bn = base_name(s.path), blobs = dir_name(s.path);
  if (bn.rfind("sha256-", 0) != 0 || base_name(blobs) != "blobs") return;
  const std::string digest = "sha256:" + bn.substr(7), manifests = dir_name(blobs) + "/manifests";
  std::vector<std::string> stack{manifests};
  while (!stack.empty()) {
    const std::string d = stack.back();
    stack.pop_back();
    DIR* dd = opendir(d.c_str());
    if (!dd) continue;
    while (dirent* e = readdir(dd)) {
      const std::string n = e->d_name;
      if (n == "." || n == "..") continue;
      const std::string p = d + "/" + n;
      if (is_dir(p)) {
        stack.push_back(p);
        continue;
      }
      std::ifstream f(p);
      json m = json::parse(f, nullptr, false);
      if (m.is_discarded() || !m.contains("layers") || !m["layers"].is_array()) continue;
      bool ours = false;
      std::string draft;
      for (const auto& l : m["layers"]) {
        const std::string mt = l.value("mediaType", ""), dg = l.value("digest", "");
        if (mt == "application/vnd.ollama.image.model" && dg == digest) ours = true;
        if (mt == "application/vnd.ollama.image.draft" && dg.rfind("sha256:", 0) == 0) draft = blobs + "/sha256-" + dg.substr(7);
      }
      if (!ours) continue;
      s.ollama_name = clean_name(base_name(d) + "-" + n);  // .../library/<model>/<tag>
      if (s.mtp.empty() && !draft.empty() && is_file(draft)) s.mtp = draft;
      closedir(dd);
      return;
    }
    closedir(dd);
  }
}

std::string default_name(const Source& s, const File& f) {
  if (!s.ollama_name.empty()) return s.ollama_name;
  std::string bn = base_name(s.path);
  static const std::string ext = ".gguf";
  if (ends_with(lower(bn), ext)) {
    bn = bn.substr(0, bn.size() - ext.size());
    // a split file: drop -0000k-of-0000N
    if (bn.size() > 15 && bn[bn.size() - 15] == '-' && bn.compare(bn.size() - 9, 4, "-of-") == 0) bn = bn.substr(0, bn.size() - 15);
    return clean_name(bn);
  }
  std::string n = f.str("general.basename", f.str("general.name", "model"));
  if (const std::string ft = ftype_name(f.num("general.file_type", -1)); !ft.empty()) n += "-" + ft;
  return clean_name(n);
}

// the source files, for recognizing an earlier conversion
json source_id(const File& f, const File* mtp) {
  json a = json::array();
  auto add = [&](const File& x) {
    for (const auto& p : x.parts()) {
      struct stat st;
      stat(p.c_str(), &st);
      a.push_back(json{{"file", base_name(p)}, {"size", uint64_t(st.st_size)}, {"mtime", int64_t(st.st_mtime)}});
    }
  };
  add(f);
  if (mtp) add(*mtp);
  return a;
}

// mtp-*.gguf beside the file (or in an MTP/ folder beside it) of the same model: the one llama.cpp's converter
// writes with --mtp, as ggml-org, bartowski and unsloth publish it.
std::string find_mtp_file(const File& f, const Hp& h) {
  const std::string dir = dir_name(f.parts()[0]);
  std::vector<std::string> found;
  for (const std::string& d : {dir, dir + "/MTP", dir + "/mtp"}) {
    DIR* dd = opendir(d.c_str());
    if (!dd) continue;
    while (dirent* e = readdir(dd)) {
      const std::string n = e->d_name;
      if (lower(n).rfind("mtp-", 0) != 0 || !ends_with(lower(n), ".gguf")) continue;
      const std::vector<std::string> parts = split_parts(d + "/" + n);
      if (parts[0] != d + "/" + n) continue;  // count a split file once
      try {
        File m(d + "/" + n);
        const std::string a = m.str("general.architecture");
        if (a == h.arch && m.num(a + ".nextn_predict_layers", 0) > 0 && m.num(a + ".embedding_length", 0) == h.H &&
            m.num(a + ".block_count", 0) - m.num(a + ".nextn_predict_layers", 0) == h.layers)
          found.push_back(d + "/" + n);
      } catch (const Error&) {
      }
    }
    closedir(dd);
  }
  return found.size() == 1 ? found[0] : "";
}

// The class holding most of the trunk's linear weights and the most common format of its 4-bit-or-better dense
// linears: in a file that is mostly 4-bit or better, its 1-3-bit tensors take that format (Plan::follow).
void find_majority(Plan& P, const File& f) {
  std::map<int, double> by_class;
  std::map<std::pair<int, int>, double> by_fmt;
  bool low = false;
  for (const auto& t : f.tensors()) {
    if (t.name.rfind("blk.", 0) != 0 || t.ne.size() < 2 || !ends_with(t.name, ".weight") ||
        t.name.find("ssm_conv1d") != std::string::npos || atoi(t.name.c_str() + 4) >= P.hp.layers)
      continue;
    const Cls c = type_class(t.type);
    by_class[c] += double(t.elements());
    if (c <= kC3) low = true;
    else if (t.name.find("_exps") == std::string::npos) {
      const Fmt o = own_format(t.type);
      by_fmt[{o.bits, o.gs}] += double(t.elements());
    }
  }
  if (by_class.empty()) return;
  P.major = Cls(std::max_element(by_class.begin(), by_class.end(), [](auto& a, auto& b) { return a.second < b.second; })->first);
  P.major_fmt = k4g32;
  if (!by_fmt.empty()) {
    const auto m = std::max_element(by_fmt.begin(), by_fmt.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
    P.major_fmt = Fmt{m.first, m.second};
  }
  P.follow = low && P.major >= kC4;
}

void remove_tree(const std::string& dir) {
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

}  // namespace

bool is_gguf_file(const std::string& arg) {
  std::string p = arg;
  if (const char* home = getenv("HOME"); home && (p == "~" || p.rfind("~/", 0) == 0)) p = home + p.substr(1);
  if (!is_file(p)) return false;
  if (ends_with(lower(p), ".gguf")) return true;
  std::ifstream f(p, std::ios::binary);
  char m[4] = {};
  return f.read(m, 4) && memcmp(m, "GGUF", 4) == 0;
}

std::string import_model(const ImportOptions& opt) {
  Source s;
  s.path = opt.file;
  s.mtp = opt.mtp;
  ollama_lookup(s);
  File f(s.path);
  const std::string file = base_name(f.parts()[0]);
  const std::string arch = f.str("general.architecture");
  if (arch == "qwen3next")
    throw Error(file + " is Qwen3-Next, the model before Qwen3.5; Kapitsa runs the Qwen3.5 family (Qwen3.5, Qwen3.6, "
                "Qwen3.8).");
  if (arch != "qwen35" && arch != "qwen35moe" && arch != "flint")
    throw Error(file + " is a \"" + arch + "\" model; Kapitsa runs the Qwen3.5 family (GGUF architectures qwen35 and "
                "qwen35moe) and Recursive's Flint (flint).");
  const bool flint = arch == "flint";
  for (auto& [k, v] : f.meta().items())
    if (k.rfind("prism.hadamard.", 0) == 0)
      throw Error(file + " stores its weights rotated (a Hadamard transform its runtime undoes), which Kapitsa does "
                  "not read. Choose another quantization of the model.");
  Plan P;
  P.hp = flint ? read_hp_flint(f) : read_hp(f);
  if (flint) check_runs_flint(P.hp, file);
  else check_runs(P.hp, file);
  // the MTP head: the file's own block, else a separate file (Flint has none)
  std::unique_ptr<File> mf;
  const File* M = nullptr;
  if (P.hp.nextn > 0) {
    M = &f;
  } else if (!flint) {
    if (s.mtp.empty()) s.mtp = find_mtp_file(f, P.hp);
    if (!s.mtp.empty()) {
      mf = std::make_unique<File>(s.mtp);
      const std::string a = mf->str("general.architecture");
      if (a != arch || mf->num(a + ".embedding_length", 0) != P.hp.H || mf->num(a + ".nextn_predict_layers", 0) < 1 ||
          mf->num(a + ".block_count", 0) - mf->num(a + ".nextn_predict_layers", 0) != P.hp.layers)
        throw Error(base_name(s.mtp) + " is not an MTP head of " + file + ".");
      M = mf.get();
    }
  }
  const std::string name = opt.name.empty() ? default_name(s, f) : clean_name(opt.name);
  if (name.empty()) throw Error("--name needs letters or digits.");
  const std::string models = hub::models_dir(), dest = models + "/" + name;
  const json sid = source_id(f, mf.get());
  // an earlier conversion of the same files
  if (is_dir(dest) && !opt.force) {
    std::ifstream mfs(dest + "/.kapitsa/manifest.json");
    const json m = json::parse(mfs, nullptr, false);
    if (!m.is_discarded() && m.contains("gguf") && m["gguf"].value("files", json()) == sid && hub::is_complete(dest)) {
      if (opt.progress && term::fancy())
        term::line("  " + term::accent(term::kOk) + " Using " + term::bold(hub::tilde(dest)) + term::dim(", converted from " + file + " earlier"),
                   stderr);
      else if (opt.progress)
        fprintf(stderr, "Using %s, converted from %s earlier.\n", hub::tilde(dest).c_str(), file.c_str());
      return dest;
    }
    throw Error(hub::tilde(dest) + " exists already and is not a conversion of " + file + ". Name this one with "
                "--name, or replace that folder with --force.");
  }
  if (!flint) find_majority(P, f);
  Builder B(P, f, M);
  if (flint) B.build_flint();
  else B.build();
  if (M && !P.has_mtp) M = nullptr;
  for (const auto& t : f.tensors())  // tensors not read: the vision tower (none in llama.cpp's text GGUFs) is fine
    if (!B.used.count(t.name) && !ends_with(t.name, ".input_scale") && t.name.rfind("v.", 0) != 0 &&
        t.name.rfind("mm.", 0) != 0 && t.name.rfind("mtp.", 0) != 0 && !(P.hp.nextn && t.name.rfind("blk." + std::to_string(P.hp.layers) + ".", 0) == 0))
      P.notes.insert("not converted: " + t.name + " (a tensor Kapitsa does not use)");
  Tok tok = make_tokenizer(f, P.hp);

  // output files: trunk shards of at most 5 GB, the routed experts kept as stored (experts*.safetensors, one per GGUF
  // part that holds them), then mtp.safetensors
  uint64_t cur = 0, total = 0, src_total = 0;
  int shards = 0;
  for (Lin& L : P.lins) {
    if (L.mtp || L.fmt.iq) continue;
    const uint64_t b = L.rows * L.row_w() + 2 * L.rows * L.row_s();
    if (shards == 0 || (cur + b > kShardBytes && cur > 0)) ++shards, cur = 0;
    L.file = shards - 1;
    cur += b;
  }
  std::vector<int> xparts;  // the GGUF part of each experts file
  for (const Lin& L : P.lins)
    if (L.fmt.iq && std::find(xparts.begin(), xparts.end(), L.src->part) == xparts.end()) xparts.push_back(L.src->part);
  const int nx = int(xparts.size());
  std::vector<std::vector<Lin*>> per(size_t(shards + nx + (P.has_mtp ? 1 : 0)));
  for (Lin& L : P.lins) {
    if (L.mtp) L.file = shards + nx;
    else if (L.fmt.iq) L.file = shards + int(std::find(xparts.begin(), xparts.end(), L.src->part) - xparts.begin());
    per[size_t(L.file)].push_back(&L);
  }
  for (int i = 0; i < shards; ++i) {
    char b[64];
    snprintf(b, sizeof b, "model-%05d-of-%05d.safetensors", i + 1, shards);
    P.files.push_back(shards == 1 ? "model.safetensors" : b);
  }
  for (int i = 0; i < nx; ++i) {
    char b[64];
    snprintf(b, sizeof b, "experts-%05d-of-%05d.safetensors", i + 1, nx);
    P.files.push_back(nx == 1 ? "experts.safetensors" : b);
  }
  if (P.has_mtp) P.files.push_back("mtp.safetensors");
  // An experts file is, where it can be, an APFS clone of its GGUF part with a safetensors header written over the
  // GGUF's (its tensor data then sits where the GGUF has it: no second copy of the experts on the disk, and the folder
  // keeps working when the GGUF is deleted); else (another disk, KAPITSA_IMPORT_COPY=1) a file of the experts' bytes.
  const bool copy_env = getenv("KAPITSA_IMPORT_COPY") && atoi(getenv("KAPITSA_IMPORT_COPY")) > 0;
  std::vector<bool> xclone(size_t(nx), false);
  std::vector<uint64_t> xdata0(size_t(nx), 0);  // the GGUF part's tensor data start
  std::vector<std::string> xheader(static_cast<size_t>(nx));  // the clone's header
  // headers and offsets
  std::vector<std::string> headers(P.files.size());
  std::vector<uint64_t> sizes(P.files.size());
  json weight_map = json::object();
  uint64_t trunk_bytes = 0;
  for (size_t fi = 0; fi < P.files.size(); ++fi) {
    const bool xf = int(fi) >= shards && int(fi) < shards + nx;
    json h = json::object();
    uint64_t off = 0;
    for (Lin* L : per[fi]) {
      auto put = [&](const std::string& n, const char* dt, std::vector<int64_t> shape, uint64_t bytes) {
        h[n] = json{{"dtype", dt}, {"shape", shape}, {"data_offsets", {off, off + bytes}}};
        if (fi < size_t(shards + nx)) weight_map[n] = P.files[fi];
        const uint64_t at = off;
        off += bytes;
        return at;
      };
      if (L->param) {
        L->off_w = put(L->name, "BF16", L->shape, L->rows * L->row_w());
        continue;
      }
      std::vector<int64_t> ws = L->shape, ss = L->shape;
      if (L->fmt.iq) {  // the blocks as bytes: [E, N, K / 256 * block bytes]
        ws.back() = int64_t(L->row_w());
        L->off_w = put(L->name + ".weight", "U8", ws, L->rows * L->row_w());
        continue;
      }
      if (L->fmt.dense()) {
        L->off_w = put(L->name + ".weight", "BF16", ws, L->rows * L->row_w());
        continue;
      }
      ws.back() = int64_t(L->K * uint64_t(L->fmt.bits) / 32);
      ss.back() = int64_t(L->K / uint64_t(L->fmt.gs));
      L->off_w = put(L->name + ".weight", "U32", ws, L->rows * L->row_w());
      L->off_s = put(L->name + ".scales", "BF16", ss, L->rows * L->row_s());
      L->off_b = put(L->name + ".biases", "BF16", ss, L->rows * L->row_s());
    }
    json meta{{"format", "mlx"}};
    if (P.has_mtp && fi + 1 == P.files.size())
      meta = json{{"format", "mlx"}, {"source", base_name(M->parts()[0])}, {"norms", "mlx (+1)"},
                  {"base_hidden_variant", "post_norm"}};
    h["__metadata__"] = meta;
    std::string hs = h.dump();
    hs.append((8 - hs.size() % 8) % 8, ' ');
    headers[fi] = hs;
    sizes[fi] = 8 + hs.size() + off;
    if (xf) {
      // the clone's header: the tensors at their places in the GGUF part, its data start the safetensors' (the JSON
      // padded with spaces over the GGUF's header)
      const size_t xi = size_t(fi) - size_t(shards);
      const int part = xparts[xi];
      uint64_t d0 = UINT64_MAX;
      for (const auto& t : f.tensors())
        if (t.part == part) d0 = std::min(d0, t.offset);
      json c = json::object();
      for (Lin* L : per[fi]) {
        std::vector<int64_t> ws = L->shape;
        ws.back() = int64_t(L->row_w());
        c[L->name + ".weight"] =
            json{{"dtype", "U8"}, {"shape", ws}, {"data_offsets", {L->src->offset - d0, L->src->offset - d0 + L->src->bytes}}};
      }
      c["__metadata__"] = json{{"format", "gguf-iq"}, {"source", base_name(f.parts()[size_t(part)])}};
      std::string cs = c.dump();
      xdata0[xi] = d0;
      if (!copy_env && d0 != UINT64_MAX && cs.size() + 8 <= d0 &&
          device_of(f.parts()[size_t(part)]) == device_of(models)) {
        cs.append(size_t(d0 - 8 - cs.size()), ' ');
        xheader[xi] = cs;
        xclone[xi] = true;
      }
      for (Lin* L : per[fi])
        if (!xclone[xi]) src_total += L->src_bytes();
      total += xclone[xi] ? d0 : sizes[fi];  // a clone writes its header's pages
      trunk_bytes += off;
      continue;
    }
    for (Lin* L : per[fi]) src_total += L->src_bytes();
    total += sizes[fi];
    if (fi < size_t(shards)) trunk_bytes += off;
  }
  if (nx) {
    const bool all = std::find(xclone.begin(), xclone.end(), false) == xclone.end();
    P.notes.insert(all ? "MoE experts kept as stored (IQ1_S / IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S): read in place "
                         "through an APFS clone of the GGUF (no second copy on the disk)"
                       : "MoE experts kept as stored (IQ1_S / IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S), copied into "
                         "experts.safetensors");
  }
  total += tok.tokenizer_json.size() + tok.tokenizer_config.size() + tok.template_.size() + (1u << 20);
  if (!opt.plan_dir.empty()) {  // hub::plan_dir: the headers a memory plan reads, and the sizes
    std::error_code ec;
    std::filesystem::create_directories(opt.plan_dir, ec);
    write_text(opt.plan_dir + "/config.json", make_config(P, tok, f).dump(2, ' ', false, json::error_handler_t::replace) + "\n");
    uint64_t disk = total;  // a clone holds all of its GGUF part once the GGUF is gone
    for (size_t i = 0; i < P.files.size(); ++i) {
      const uint64_t hl = headers[i].size();
      write_text(opt.plan_dir + "/" + P.files[i], std::string(reinterpret_cast<const char*>(&hl), 8) + headers[i]);
      const size_t xi = i - size_t(shards);
      struct stat st;
      if (int(i) >= shards && int(i) < shards + nx && xclone[xi] && stat(f.parts()[size_t(xparts[xi])].c_str(), &st) == 0)
        disk += uint64_t(st.st_size) - xdata0[xi];
    }
    write_text(opt.plan_dir + "/import.json", json{{"write", total}, {"disk", disk}}.dump() + "\n");
    return opt.plan_dir;
  }

  // what will be written, and whether it fits
  if (opt.progress && term::fancy()) {
    // in a terminal: the file and where it goes, its formats as a small table, notes dim, warnings yellow
    const std::string dot = std::string(" ") + term::kDot + " ";
    term::line("  " + term::accent("⇄") + " Converting " + term::bold(base_name(file)) + "  " +
                   term::dim(std::string(P.hp.moe ? "MoE" : "dense") + dot + gb(f.file_bytes()) +
                             (mf ? dot + "MTP head from " + base_name(s.mtp) : "")),
               stderr);
    term::line("    " + term::dim(term::clip("into " + hub::tilde(dest), term::columns() - 5)), stderr);
    int nw = 0;
    for (auto& [k, n] : P.conv_count) nw = std::max(nw, int(std::to_string(n).size()));
    for (auto& [k, n] : P.conv_count) {
      std::string what = k;
      if (const size_t a = what.find(" -> "); a != std::string::npos) what = what.substr(0, a) + term::dim(" → ") + what.substr(a + 4);
      term::line("    " + term::right(std::to_string(n), nw) + term::dim(" × ") + what, stderr);
    }
    const int w = std::max(30, std::min(term::columns(), 100) - 6);
    auto say = [&](const std::string& mark, const std::string& text, bool dim) {
      const auto ls = term::wrap(text, w);
      for (size_t i = 0; i < ls.size(); ++i)
        term::line("    " + (i ? std::string("  ") : mark + " ") + (dim ? term::dim(ls[i]) : ls[i]), stderr);
    };
    for (const auto& n : P.notes) say(term::dim(term::kArrow), n, true);
    if (!P.followed.empty()) {
      std::string l;
      for (const auto& [t, n] : P.followed) l += (l.empty() ? "" : ", ") + t + " x" + std::to_string(n);
      say(term::dim(term::kArrow), "the 1-3-bit tensors (" + l + ") take the file's majority format, " + fmt_name(P.major_fmt) +
                                       (P.hp.moe ? " (MoE experts: " + fmt_name(expert_format(P.major)) + ")" : ""),
          true);
    }
    if (!P.low.empty()) {
      std::string l;
      for (const auto& t : P.low) l += (l.empty() ? "" : "/") + t;
      say(term::warn(term::kWarn),
          "This file is mostly " + l + ", 1-3-bit formats Kapitsa has no kernels for: they become 3-bit group 64 (1-2-bit "
          "types) or 4-bit group 32 (3-bit types), at about the file's quality and larger than the file. A Q4_K_M or better "
          "GGUF gives a better model at about the size of the conversion.",
          false);
    }
    if (!P.has_mtp && !P.hp.flint) say(term::dim(term::kArrow), "no MTP head (neither in the file nor as an mtp-*.gguf beside it): decoding without MTP drafts", true);
  } else if (opt.progress) {
    fprintf(stderr, "Converting %s (%s, %s%s) into %s:\n", file.c_str(), P.hp.moe ? "MoE" : "dense",
            gb(f.file_bytes()).c_str(), mf ? (" + MTP head from " + base_name(s.mtp)).c_str() : "",
            hub::tilde(dest).c_str());
    for (auto& [k, n] : P.conv_count) fprintf(stderr, "  %4d x %s\n", n, k.c_str());
    for (const auto& n : P.notes) fprintf(stderr, "  note: %s\n", n.c_str());
    if (!P.followed.empty()) {
      std::string l;
      for (const auto& [t, n] : P.followed) l += (l.empty() ? "" : ", ") + t + " x" + std::to_string(n);
      fprintf(stderr, "  note: the 1-3-bit tensors (%s) take the file's majority format, %s%s\n", l.c_str(),
              fmt_name(P.major_fmt).c_str(), P.hp.moe ? (" (MoE experts: " + fmt_name(expert_format(P.major)) + ")").c_str() : "");
    }
    if (!P.low.empty()) {
      std::string l;
      for (const auto& t : P.low) l += (l.empty() ? "" : "/") + t;
      fprintf(stderr, "  warning: this file is mostly %s, 1-3-bit formats Kapitsa has no kernels for: they become 3-bit "
                      "group 64 (1-2-bit types) or 4-bit group 32 (3-bit types), at about the file's quality and larger "
                      "than the file. A Q4_K_M or better GGUF gives a better model at about the size of the conversion.\n",
              l.c_str());
    }
    if (!P.has_mtp && !P.hp.flint)
      fprintf(stderr, "  no MTP head (neither in the file nor as an mtp-*.gguf beside it): decoding without MTP drafts\n");
  }
  if (opt.dry_run && term::fancy()) {
    const uint64_t avail = free_bytes(models);
    const std::string text = "Dry run: the conversion would write " + gb(total) + " (" + gb(avail) + " free on that disk" +
                             (avail < total + kKeepFree ? "; not enough, 6 GB must stay free" : "") + "); nothing was written.";
    const auto ls = term::wrap(text, std::max(30, std::min(term::columns(), 100) - 6));
    for (size_t i = 0; i < ls.size(); ++i)
      term::line("  " + (i ? std::string("  ") : (avail < total + kKeepFree ? term::warn(term::kWarn) : term::dim(term::kArrow)) + " ") +
                     term::dim(ls[i]),
                 stderr);
    return dest;
  }
  if (opt.dry_run) {
    const uint64_t avail = free_bytes(models);
    fprintf(stderr, "Dry run: the conversion would write %s into %s (%s free on that disk%s).\n", gb(total).c_str(),
            hub::tilde(dest).c_str(), gb(avail).c_str(), avail < total + kKeepFree ? "; not enough, 6 GB must stay free" : "");
    return dest;
  }
  std::error_code ec;
  std::filesystem::create_directories(models, ec);
  const uint64_t avail = free_bytes(models);
  if (avail < total + kKeepFree)
    throw Error("Not enough disk space: the converted model takes " + gb(total) + " and " + gb(avail) + " is free on "
                "the disk of " + hub::tilde(models) + ", which must keep 6 GB free. Free some space, or set KAPITSA_HOME "
                "to a folder on another disk.");

  // write into a hidden folder, renamed when complete
  StopGuard stop_guard;
  const std::string tmp = models + "/." + name + ".importing";
  remove_tree(tmp);
  std::filesystem::create_directories(tmp + "/.kapitsa", ec);
  if (ec) throw Error("Can't create " + hub::tilde(tmp) + ": " + ec.message() + ".");
  try {
    write_text(tmp + "/config.json", make_config(P, tok, f).dump(2, ' ', false, json::error_handler_t::replace) + "\n");
    write_text(tmp + "/tokenizer.json", tok.tokenizer_json);
    write_text(tmp + "/tokenizer_config.json", tok.tokenizer_config);
    write_text(tmp + "/chat_template.jinja", tok.template_);
    {
      ojson g{{"eos_token_id", make_config(P, tok, f)["eos_token_id"]}};
      for (const char* k : {"temp", "top_k", "top_p", "min_p"})
        if (f.has(std::string("general.sampling.") + k)) {
          // a float32 value as the shortest decimal that reads back to it (top_p 0.95, not 0.949999988079071)
          json v = f.meta()[std::string("general.sampling.") + k];
          for (int p = 1; v.is_number_float() && p <= 9; ++p) {
            char b[32];
            snprintf(b, sizeof b, "%.*g", p, v.get<double>());
            if (float(strtod(b, nullptr)) == float(v.get<double>())) {
              v = strtod(b, nullptr);
              break;
            }
          }
          g[std::string(k) == "temp" ? "temperature" : k] = v;
        }
      write_text(tmp + "/generation_config.json", g.dump(2) + "\n");
    }
    std::vector<OutFile> outs(P.files.size());
    for (size_t i = 0; i < P.files.size(); ++i) {
      OutFile& o = outs[i];
      o.name = tmp + "/" + P.files[i];
      const bool xf = int(i) >= shards && int(i) < shards + nx;
      if (xf && xclone[i - size_t(shards)]) {
        // the GGUF part's clone with the safetensors header over its own (falls back to a copy)
        const size_t xi = i - size_t(shards);
        char real[PATH_MAX];
        const std::string src = f.parts()[size_t(xparts[xi])];
        if (clonefile(realpath(src.c_str(), real) ? real : src.c_str(), o.name.c_str(), 0) == 0) {
          const int fd = open(o.name.c_str(), O_WRONLY);
          if (fd < 0) throw Error("Can't write " + hub::tilde(o.name) + ": " + strerror(errno) + ".");
          const uint64_t hl = xheader[xi].size();
          pwrite_all(fd, &hl, 8, 0, o.name);
          pwrite_all(fd, xheader[xi].data(), xheader[xi].size(), 8, o.name);
          if (close(fd) != 0) throw Error("Can't write " + hub::tilde(o.name) + ": " + strerror(errno) + ".");
          continue;  // fd stays -1: convert() leaves its tensors alone
        }
        xclone[xi] = false;
        for (Lin* L : per[i]) src_total += L->src_bytes();
        if (free_bytes(models) < sizes[i] + kKeepFree)
          throw Error("Not enough disk space: the GGUF can't be cloned into " + hub::tilde(models) + " (" +
                      strerror(errno) + "), and a copy of its experts takes " + gb(sizes[i]) + ", which would leave "
                      "less than 6 GB free.");
      }
      o.fd = open(o.name.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
      if (o.fd < 0) throw Error("Can't create " + hub::tilde(o.name) + ": " + strerror(errno) + ".");
      fcntl(o.fd, F_NOCACHE, 1);
      if (ftruncate(o.fd, off_t(sizes[i])) != 0) throw Error("Can't write " + hub::tilde(o.name) + ": " + strerror(errno) + ".");
      const uint64_t hl = headers[i].size();
      pwrite_all(o.fd, &hl, 8, 0, o.name);
      pwrite_all(o.fd, headers[i].data(), headers[i].size(), 8, o.name);
      o.data0 = 8 + hl;
      o.size = sizes[i];
    }
    const int threads = opt.threads > 0 ? opt.threads : std::min(8, perf_cores());
    Progress prog(src_total, opt.progress, "Converting " + file);
    for (const Lin& L : P.lins) convert(L, outs, threads, prog);
    prog.end();
    for (auto& o : outs) {
      if (o.fd >= 0 && close(o.fd) != 0) throw Error("Can't write " + hub::tilde(o.name) + ": " + strerror(errno) + ".");
      o.fd = -1;
    }
    if (shards > 0) {
      json idx{{"metadata", {{"total_size", trunk_bytes}}}, {"weight_map", weight_map}};
      write_text(tmp + "/model.safetensors.index.json", idx.dump(2) + "\n");
    }
    // the manifest, last: what `kapitsa list` shows and what makes the folder complete
    ojson files = ojson::array();
    uint64_t bytes = 0;
    for (const auto& e : std::filesystem::directory_iterator(tmp)) {
      if (!e.is_regular_file()) continue;
      files.push_back(ojson{{"path", e.path().filename().string()}, {"size", uint64_t(e.file_size())}});
      bytes += uint64_t(e.file_size());
    }
    ojson conv = ojson::object();
    for (auto& [k, n] : P.conv_count) conv[k] = n;
    const std::string ft = ftype_name(f.num("general.file_type", -1));
    ojson man{{"name", name},
              {"repo", ""},
              {"revision", ""},
              {"commit", ""},
              {"variant", ft.empty() ? "gguf" : "gguf " + ft},
              {"quant", "converted from GGUF"},
              {"gguf", {{"files", sid}, {"conversions", conv}}},
              {"files", files},
              {"companions", ojson::array()},
              {"total_bytes", bytes},
              {"kapitsa", KAPITSA_VERSION}};
    write_text(tmp + "/.kapitsa/manifest.json", man.dump(2) + "\n");
    if (is_dir(dest)) {  // --force
      if (dest.compare(0, models.size() + 1, models + "/") != 0) throw Error("refusing to replace " + dest);
      remove_tree(dest);
    }
    if (rename(tmp.c_str(), dest.c_str()) != 0) throw Error("Can't create " + hub::tilde(dest) + ": " + strerror(errno) + ".");
    if (opt.progress && term::fancy())
      term::line("  " + term::accent(term::kOk) + " Converted " + term::bold(base_name(file)) +
                     term::dim(" in " + fmt_secs(prog.secs()) + ", " + gb(bytes)),
                 stderr);
    else if (opt.progress)
      fprintf(stderr, "Converted %s in %s: %s in %s.\n", file.c_str(), fmt_secs(prog.secs()).c_str(), gb(bytes).c_str(),
              hub::tilde(dest).c_str());
  } catch (...) {
    remove_tree(tmp);
    throw;
  }
  return dest;
}

int run_import(int argc, char** argv) {
  ImportOptions o;
  const char* usage =
      "kapitsa import FILE.gguf [--name NAME] [--mtp MTP.gguf] [--threads N] [--force] [--dry-run]\n\n"
      "Converts a GGUF of a Qwen3.5-family model (from LM Studio, Ollama, llama.cpp or a Hugging Face GGUF repo) once\n"
      "into a model folder in %s, which `kapitsa serve --model NAME` runs (`kapitsa serve --model FILE.gguf`\n"
      "converts on first use). A split file is given by any of its parts.\n\n"
      "  --name NAME    the folder's name (default: from the file name)\n"
      "  --mtp FILE     the model's MTP head as a separate GGUF (default: the file's own, or an mtp-*.gguf beside it)\n"
      "  --threads N    CPU threads (default: the performance cores, at most 8)\n"
      "  --force        convert again over an existing folder of that name\n"
      "  --dry-run      show the formats, size and folder of the conversion, and write nothing\n\n"
      "Formats: Q4_0/Q4_1/Q4_K keep their codes (4-bit group 32); Q8_0/Q6_K become 8-bit, Q5_x 6-bit, IQ4_x/MXFP4/NVFP4\n"
      "5-bit (group 64), IQ3_x/Q3_K 4-bit group 32, IQ1/IQ2/Q2_K 3-bit group 64, F16/BF16/F32 bf16; in a mostly\n"
      "4-bit-or-better file its 1-3-bit tensors take the file's majority format. MoE experts take 5-bit group 64 from\n"
      "4-bit types; in IQ1_S / IQ2_XXS / IQ2_S / IQ3_XXS / IQ3_S they stay as stored, read in place through an APFS\n"
      "clone of the GGUF (no second copy on the disk; KAPITSA_IMPORT_COPY=1 copies them).\n";
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        if (term::fancy()) term::error(a + " needs a value", "kapitsa import --help shows how to use it.");
        else fprintf(stderr, "kapitsa import: %s needs a value\n", a.c_str());
        exit(2);
      }
      return argv[++i];
    };
    if (a == "--help" || a == "-h") {
      if (term::fancy()) {
        char b[4096];
        snprintf(b, sizeof b, usage, hub::tilde(hub::models_dir()).c_str());
        term::help(b);
      } else {
        printf(usage, hub::tilde(hub::models_dir()).c_str());
      }
      return 0;
    } else if (a == "--name") o.name = next();
    else if (a == "--mtp") o.mtp = next();
    else if (a == "--threads") o.threads = atoi(next().c_str());
    else if (a == "--force") o.force = true;
    else if (a == "--dry-run") o.dry_run = true;
    else if (!a.empty() && a[0] != '-' && o.file.empty()) o.file = a;
    else {
      if (term::fancy()) term::error("unexpected " + a, "kapitsa import --help shows how to use it.");
      else fprintf(stderr, "kapitsa import: unexpected %s\n(kapitsa import --help shows how to use it.)\n", a.c_str());
      return 2;
    }
  }
  if (o.file.empty()) {
    if (term::fancy()) term::error("Which file?", "e.g. kapitsa import ~/Downloads/Qwen3.5-4B-Q4_K_M.gguf");
    else fprintf(stderr, "kapitsa import: which file? e.g. kapitsa import ~/Downloads/Qwen3.5-4B-Q4_K_M.gguf\n");
    return 2;
  }
  if (const char* home = getenv("HOME"); home && o.file.rfind("~/", 0) == 0) o.file = home + o.file.substr(1);
  if (const char* home = getenv("HOME"); home && o.mtp.rfind("~/", 0) == 0) o.mtp = home + o.mtp.substr(1);
  try {
    const std::string dir = import_model(o);
    if (o.dry_run) return 0;
    if (term::fancy()) {
      term::line("  " + term::accent(term::kOk) + " " +
                 term::clip(term::bold(base_name(dir)) + " is ready" + term::dim("  " + hub::tilde(dir)), term::columns() - 5));
      term::line("  " + term::dim(std::string(term::kArrow) + " kapitsa serve --model " + base_name(dir) + " starts it"));
    } else {
      printf("%s is ready in %s. Start it: kapitsa serve --model %s\n", base_name(o.file).c_str(), hub::tilde(dir).c_str(),
             base_name(dir).c_str());
    }
  } catch (const Error& e) {
    if (term::fancy()) term::status(""), term::error(e.what());
    else fprintf(stderr, "kapitsa: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    if (term::fancy()) term::status(""), term::error(e.what());
    else fprintf(stderr, "kapitsa: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace ni::gguf
