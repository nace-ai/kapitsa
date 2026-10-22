// 8-row target verify (DFlash2 block verification: anchor/pending token + up to 7 drafts).
// Every projection runs on the qmm8 MMA GEMM (src/kernels/verify8.metal): the weights are read once
// for all rows and the 8 rows fill one 8x8 simdgroup_matrix dimension. Inputs reach the GEMMs as
// "X8 tables" (half rows in the MMA operand order plus group sums) written by x8_prep, or directly by
// the producer (the SwiGLU gate/up kernel writes the down projection's table). The MoE layers route
// each row's top-8, group the (row, expert) pairs by distinct expert (moe8_plan) and read every
// touched expert once for all its rows; the shared expert is one more 8-row tile.
// Taken by encode_verify for R >= 5 (NI_V8=1: R >= 2); MTP verifies with R <= 4 keep their kernels.
#include "kernels/kflash.h"
#include "model/kparams.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <set>
#include <cmath>
#include <unordered_map>

namespace ni {

extern const char* const kTensorKernelSource;

// The GPU's matrix units through Metal 4 tensor_ops (qmm8t): Apple10 (M5) on macOS 26. (An M4 Pro runs tensor_ops too,
// but qmm8t's operand layout, probed on the M5, gives it wrong logits: an M4 cannot stand in for an M5 in its tests.)
static bool tw_supported() {
  static const bool ok = [] {
    if (Gpu::get().apple_family() < 10) return false;
    if (__builtin_available(macOS 26.0, *)) return true;
    return false;
  }();
  return ok;
}

static Pipeline pipe8(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

static int env_int(const char* n, int def) {
  const char* v = getenv(n);
  return v && *v ? atoi(v) : def;
}

// gate/up kernel (NI_V8_GU): "1" = 8 simdgroups x (1 gate + 1 up fragment), "2" = 4 x (2 + 2) (NI_V8_GU_PF=1:
// their software-pipelined loops), "<np>k<ks>" = np fragments per matrix with a ks-way in-threadgroup K split
// (prefetching loop): 1k2, 1k4, 2k2, 2k4.
struct GuCfg {
  std::string kernel;
  uint32_t threads;
};
static const GuCfg& gu_cfg() {
  static const GuCfg c = [] {
    const char* v = getenv("NI_V8_GU");
    const std::string s = v && *v ? v : "1k2";  // measured best on both models
    const bool pf = env_int("NI_V8_GU_PF", 0) != 0;
    if (s == "1") return GuCfg{pf ? "qmm8_gu1_pf_b4" : "qmm8_gu1_b4", 256u};
    if (s == "2") return GuCfg{pf ? "qmm8_gu_pf_b4" : "qmm8_gu_b4", 128u};
    if (s.size() == 3 && (s[0] == '1' || s[0] == '2') && s[1] == 'k' && (s[2] == '2' || s[2] == '4')) {
      const uint32_t np = uint32_t(s[0] - '0'), ks = uint32_t(s[2] - '0');
      return GuCfg{"qmm8_gu" + s + "_b4", 32u * (8u / np) * ks};
    }
    fatal("NI_V8_GU=%s: expected 1, 2, 1k2, 1k4, 2k2 or 2k4", s.c_str());
    return GuCfg{};
  }();
  return c;
}
V8Knobs& v8_knobs() {
  static V8Knobs k = [] {
    V8Knobs d;
    d.cm = env_int("NI_V8_CM", d.cm);
    d.pair = env_int("NI_V8_PAIR", d.pair);
    d.kplan = env_int("NI_V8_KPLAN", Gpu::get().apple_family() <= 7 ? 3 : d.kplan);
    d.gusplit = env_int("NI_V8_GUSPLIT", d.gusplit);
    d.gdnsplit = env_int("NI_V8_GDNSPLIT", d.gdnsplit);
    d.prepu = env_int("NI_V8_PREPU", d.prepu);
    d.tw = env_int("NI_V8_TW", tw_supported() ? d.tw : 0);
    d.twm = env_int("NI_V8_TWM", d.twm);
    d.fuse = env_int("NI_V8_FUSE", d.fuse);
    return d;
  }();
  return k;
}

V8Knobs v8_knobs_parse(const char* spec, V8Knobs k) {
  std::string s = spec ? spec : "";
  size_t a = 0;
  while (a < s.size()) {
    size_t b = s.find(',', a);
    if (b == std::string::npos) b = s.size();
    const std::string kv = s.substr(a, b - a);
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) fatal("knobs: '%s' is not key=value", kv.c_str());
    const std::string key = kv.substr(0, eq);
    const int v = atoi(kv.c_str() + eq + 1);
    if (key == "cm") k.cm = v;
    else if (key == "pair") k.pair = v;
    else if (key == "kplan") k.kplan = v;
    else if (key == "gusplit") k.gusplit = v;
    else if (key == "gdnsplit") k.gdnsplit = v;
    else if (key == "prepu") k.prepu = v;
    else if (key == "skip") k.skip = v;
    else if (key == "pad") k.pad = v;
    else if (key == "tw") k.tw = v;
    else if (key == "twm") k.twm = v;
    else if (key == "fuse") k.fuse = v;
    else if (key == "emu") k.emu = v;
    else if (key == "split") k.split = v;
    else if (key == "shs") k.shs = v;
    else if (key == "tkp") k.tkp = v;
    else if (key == "dfl") k.dfl = v;
    else fatal("knobs: unknown key '%s'", key.c_str());
    a = b + 1;
  }
  return k;
}

// GEMM operand mode (V8Knobs::cm, NI_V8_CM): 0 = codes as 1024 + q, 1 = codes as exact half subnormals
// q * 2^-24 (no 1024 * sum(x) cancellation, one integer op fewer per MMA; bench_v8: error 2-5e-7 instead
// of 2-3e-5; teacher-forced 27B verify: max |logit diff| vs plain decode 0.025 instead of 0.051).
// V8Knobs::pair: two groups per inner step.
static std::string gu_kernel(uint32_t K) {
  const std::string& k = gu_cfg().kernel;
  const V8Knobs& kn = v8_knobs();
  const bool pair = kn.pair && (K / 64) % 2 == 0;  // scale / bias pairs need an even group count per row
  if (k == "qmm8_gu1k2_b4" && (kn.cm == 1 || pair))
    return std::string("qmm8_gu1k2") + (kn.cm == 1 ? "sn" : "") + (pair ? "b" : "") + "_b4";
  return k;
}
static uint32_t gu_threads() { return gu_cfg().threads; }

void Qwen35::v8_alloc() {
  if (v8_ready_) return;
  Gpu& g = Gpu::get();
  const int D = cfg_.hidden, VD = cfg_.lin_hv * cfg_.lin_dv;
  const int W = std::max({D, VD, cfg_.hq * cfg_.head_dim, cfg_.dense_I, cfg_.moe_I, cfg_.shared_I, 64});
  if (W % 64) fatal("8-row verify: widths must be multiples of 64");
  // row tiles: a wide tree verify's tables follow each other (set_tree_rows; MoE targets stay at one tile)
  const size_t rt = cfg_.moe ? 1 : size_t(std::max(1, tree_rows_ / 8));
  v8t_ = g.alloc(size_t(W) * 8 * 2 * rt, "x8_table");
  v8s_ = g.alloc(size_t(W) / 64 * 8 * 4 * rt, "x8_sums");
  v8tl_ = g.alloc(size_t(W) * 8 * 2 * rt, "x8_table_lo");
  if (tw_supported()) {
    v8tt_ = g.alloc(size_t(W) * 8 * 2 * rt, "x8_table_t");
    g.compile_extra("tensor", kTensorKernelSource, true);
  }
  v8sl_ = g.alloc(size_t(W) / 64 * 8 * 4 * rt, "x8_sums_lo");
  const int tiles = cfg_.moe ? kMaxVerify * cfg_.top_k + 1 : int(rt);
  const int Ih = cfg_.moe ? std::max(cfg_.moe_I, cfg_.shared_I) : std::max(cfg_.dense_I, 64);
  v8h_ = g.alloc(size_t(tiles) * Ih * 8 * 2, "x8_h_tables");
  // per-head arrival counters of gdn_step_verify8s, one set per sequence of a batched pass (encode_verify_multi runs
  // their steps side by side)
  v8cnt_ = g.alloc((size_t(std::max(cfg_.lin_hv, 1)) * 4 + 16) * 8, "gdn8_counters");
  memset(v8cnt_.ptr(), 0, v8cnt_.size);
  if (!cfg_.moe) {
    v8g_ = g.alloc(size_t(8) * rt * cfg_.dense_I * 4, "x8_gate_rows");
    v8u_ = g.alloc(size_t(8) * rt * cfg_.dense_I * 4, "x8_up_rows");
  }
  v8hs_ = g.alloc(size_t(tiles) * Ih / 64 * 8 * 4, "x8_h_sums");
  if (cfg_.moe) {
    v8tiles_ = g.alloc(size_t(tiles) * 4, "moe8_tiles");
    v8nt_ = g.alloc(16, "moe8_ntiles");
    v8slot_ = g.alloc(size_t(tiles) * 8 * 4, "moe8_slots");
    v8args_ = g.alloc(128, "moe8_args");
    v8out_ = g.alloc(size_t(kMaxVerify) * cfg_.top_k * D * 4, "moe8_out");
    v8sh_ = g.alloc(size_t(kMaxVerify) * D * 4, "moe8_shared_out");
    v8ustat_ = g.alloc(size_t(cfg_.layers) * 4, "moe8_stats");
    v8sm_ = g.alloc(5 * 72 * 4, "moe8_scalar_tiles");  // verify8.metal kV8SM tiles: experts, 2 rows, 2 slots each
    v8hsc_ = g.alloc(size_t(2 * 72) * Ih * 4, "moe8_scalar_h");
    v8hse_ = g.alloc(size_t(2 * 72) * Ih / 64 * 4, "moe8_scalar_hsums");
    v8sm2_ = g.alloc(5 * 72 * 4, "moe8_scalar_tiles_b");
    v8hsc2_ = g.alloc(size_t(2 * 72) * Ih * 4, "moe8_scalar_h_b");
    v8hse2_ = g.alloc(size_t(2 * 72) * Ih / 64 * 4, "moe8_scalar_hsums_b");
    v8xn_ = g.alloc(size_t(8) * D * 4, "x8_rows");
  }
  v8_ready_ = true;
}

bool Qwen35::v8_tw() const { return v8_knobs().tw && tw_supported() && v8tt_.size; }

// ---- KFlash tree verifies wider than 8 rows (16 / 32 rows) ----
// verify_tree_ok / verify_wide_ok on a config alone (the memory plan has no model: it reads the same knobs)
static bool v8_tree_ok_c(const Qwen35Config& c) {
  static const bool gdn8 = env_int("NI_V8_GDN", 1) != 0;
  static const int attn_mma = env_int("NI_V8_ATTN", 1);
  const int split_thr = v8_knobs().gdnsplit;
  const bool split = gdn8 && c.lin_dv == 128 && c.lin_dk == 128 && split_thr >= 32 && split_thr <= 1024 &&
                     split_thr % 32 == 0 && 128 % (split_thr / 8) == 0;
  return split && attn_mma && c.hq / c.hkv <= 16 && c.head_dim == 256;
}

static bool v8_wide_ok_c(const Qwen35Config& c) {
  if (c.moe || !v8_tree_ok_c(c) || c.dense_I % 256 || !v8_knobs().gusplit) return false;
  // every X8 table of the pass on x8_prep2 (or the gated norm's), i.e. widths multiple of 256
  for (const int K : {c.hidden, c.hq * c.head_dim})
    if (K % 256) return false;
  return env_int("NI_V8_PREP", 2) != 1;
}

bool Qwen35::verify_wide_ok() const { return v8_wide_ok_c(cfg_); }

uint32_t Qwen35::kf_dense_rows(const Qwen35Config& c) {
  // one 8-row tile: the default, and all a MoE target or a target without the wide kernels can run
  const uint32_t one = 1u << (kMaxVerify - 1);
  const char* kf = getenv("NI_KFLASH");
  const char* v = getenv("NI_DFLASH_TREE");
  const bool tree = v && *v ? atoi(v) != 0 : (!(kf && *kf) || atoi(kf) != 0);
  const char* r = getenv("NI_KFLASH_ROWS");
  // wider rows are asked for (NI_KFLASH_ROWS names 16 or 32) on a target that runs them: the default stays one tile
  // until a Mac's A/B says a wider count pays there (docs/kflash.md, docs/usage.md)
  if (!tree || !v8_wide_ok_c(c) || !r || !*r) return one;
  uint32_t allowed = one | (1u << 15) | (1u << 31);
  uint32_t want = 0;
  for (const char* p = r; *p;) {
    const int n = atoi(p);
    if (n == 8 || n == 16 || n == 32) want |= 1u << (n - 1);
    else fatal("NI_KFLASH_ROWS=%s: row counts 8, 16 or 32", r);
    while (*p && *p != ',') ++p;
    if (*p == ',') ++p;
  }
  const uint32_t w = want & allowed;
  return w ? w : one;  // nothing allowed left: the 8-row tile
}

void Qwen35::set_tree_rows(int rows) {
  // 8, 16 or 32 rows, a power of two: the tables hold whole 8-row tiles and the tile-pair MLP reads two at a time, so a
  // capacity of three tiles (24 rows) would read past them (a live verify of up to 24 rows still fits in 32)
  rows = rows <= 8 ? 8 : rows <= 16 ? 16 : kMaxTreeRows;
  if (rows <= tree_rows_) return;
  tree_rows_ = rows;
  Gpu& g = Gpu::get();
  auto regrow = [&](Buffer& b, size_t bytes, const char* label) {
    if (b.size >= bytes) return;
    g.release(b);
    b = g.alloc(bytes, label);
  };
  const size_t V = size_t(cfg_.vocab), R = size_t(rows);
  regrow(logits_, V * R * 4, "logits");
  regrow(cand_v_, R * (V / 2048 + 2) * 64 * 4, "cand_v");
  regrow(cand_i_, R * (V / 2048 + 2) * 64 * 4, "cand_i");
  regrow(samp_, std::max<size_t>(64, R * 4), "samples");
  if (v8_ready_) {  // the 8-row verify's buffers again, sized for the tiles
    for (Buffer* b : {&v8g_, &v8u_, &v8cnt_, &v8tt_, &v8t_, &v8s_, &v8tl_, &v8sl_, &v8h_, &v8hs_, &v8tiles_, &v8nt_,
                      &v8slot_, &v8args_, &v8out_, &v8sh_, &v8ustat_, &v8sm_, &v8hsc_, &v8xn_, &v8hse_, &v8sm2_,
                      &v8hsc2_, &v8hse2_})
      if (*b) g.release(*b);
    v8_ready_ = false;
    v8_alloc();
  }
}

// a ring of descriptor regions (one per wide verify in flight): word 0 the count, then {x, y, z, flags} per grid
static constexpr int kV8wRegions = 4, kV8wDesc = 64;

void Qwen35::grow_tape(Encoder& e, Seq& s, int rows) {
  Gpu& g = Gpu::get();
  const size_t C = 2 * size_t(cfg_.lin_hk) * cfg_.lin_dk + size_t(cfg_.lin_hv) * cfg_.lin_dv;
  const size_t Hv = size_t(cfg_.lin_hv);
  auto grow = [&](Buffer& b, size_t per_row, const char* label) {
    const size_t bytes = size_t(rows) * per_row;
    if (b.size >= bytes) return;
    Buffer nb = g.alloc(bytes, label);
    enc_copy(e, b, nb);  // the pending rows the next pass folds (in order with the work before it)
    e.on_completed([old = b]() mutable { Gpu::get().release(old); });  // passes encoded before it may still read it
    b = nb;
  };
  for (int i = 0; i < cfg_.layers; ++i) {
    if (!cfg_.is_linear[i]) continue;
    grow(s.pend_qkv[i], C * 4, "pend_qkv");
    grow(s.pend_k[i], Hv * cfg_.lin_dk * 4, "pend_k");
    grow(s.pend_g[i], Hv * 4, "pend_g");
    grow(s.pend_d[i], Hv * cfg_.lin_dv * 4, "pend_d");
  }
}

void Qwen35::v8_wide_begin(Encoder& e, Seq& s, int R) {
  if (!verify_wide_ok()) fatal("tree verify of %d rows: this model's verify runs 8 rows (verify_wide_ok)", R);
  if (R > tree_rows_) fatal("tree verify of %d rows: the verify buffers hold %d (set_tree_rows)", R, tree_rows_);
  Gpu& g = Gpu::get();
  if (!v8w_desc_) {
    v8w_desc_ = g.alloc(size_t(kV8wRegions) * kV8wDesc * 16, "v8w_desc");
    v8w_args_ = g.alloc(size_t(kV8wRegions) * kV8wDesc * 16, "v8w_args");
    memset(v8w_desc_.ptr(), 0, v8w_desc_.size);
  }
  // the sequence's tape: the rows this target's verifies run (set_tree_rows, which new_seq allocated), so a wider
  // verify than the tape was grown for only grows it once, and a 16-row block's tape stays at 16 rows
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.is_linear[i]) {
      if (s.pend_g[i].size < size_t(tree_rows_) * cfg_.lin_hv * 4) grow_tape(e, s, tree_rows_);
      break;
    }
  v8w_R_ = R;
  // NI_V8W_DIRECT=<tiles> (timing only, wrong unless the tree has that many live tiles): direct grids of that many
  // tiles, the indirect dispatches' own cost
  static const int direct = env_int("NI_V8W_DIRECT", 0);
  if (direct > 0) {
    v8w_host_nt_ = std::min(direct, R / 8);
    return;
  }
  if (Encoder::profiling()) {
    // NI_PROFILE commits every dispatch on its own: the tree is final once the work before this verify is done, so the
    // host reads its live rows and the grids go out directly
    g.begin()->commit_and_wait();
    const int n = std::max(1, v8_tree_->as<int32_t>()[kf::kN]);
    v8w_host_nt_ = std::min(R, (n + 7) / 8 * 8) / 8;
    return;
  }
  v8w_host_nt_ = -1;
  v8w_region_ = (v8w_region_ + 1) % kV8wRegions;
  v8w_n_ = 0;
  const uint32_t Ru = uint32_t(R);
  e.use(pipe8("v8_tree_plan"));
  e.buf(0, *v8_tree_);
  e.buf(1, v8w_desc_, size_t(v8w_region_) * kV8wDesc * 16);
  e.buf(2, v8w_args_, size_t(v8w_region_) * kV8wDesc * 16);
  e.val(3, Ru);
  e.dispatch(1, 1, 1, 64);
}

void Qwen35::v8_wide_end() {
  if (v8w_R_ && v8w_host_nt_ < 0) v8w_desc_.as<uint32_t>()[size_t(v8w_region_) * kV8wDesc * 4] = uint32_t(v8w_n_);
  v8w_R_ = 0;
  v8w_host_nt_ = -1;
}

void Qwen35::v8_disp(Encoder& e, uint32_t x, uint32_t y, uint32_t z, uint32_t flags, uint32_t tx) {
  if (!v8w_R_ || v8w_host_nt_ >= 0) {  // the grid itself (one tile, or the live tiles NI_PROFILE read)
    const uint32_t nt = v8w_R_ ? uint32_t(v8w_host_nt_) : 1u, rows = v8w_R_ ? 8 * nt : 8u;
    if (flags & kV8wX) x *= nt;
    if (flags & kV8wXpair) x *= (nt + 1) / 2;
    if (flags & kV8wY) y *= nt;
    if (flags & kV8wYrows) y = rows;
    if (flags & kV8wXrows) x = rows;
    e.dispatch(x, y, z, tx);
    return;
  }
  uint32_t* d = v8w_desc_.as<uint32_t>() + size_t(v8w_region_) * kV8wDesc * 4;
  int idx = -1;
  for (int i = 0; i < v8w_n_ && idx < 0; ++i) {
    const uint32_t* q = d + 4 * (1 + i);
    if (q[0] == x && q[1] == y && q[2] == z && q[3] == flags) idx = i;
  }
  if (idx < 0) {
    if (v8w_n_ + 1 >= kV8wDesc) fatal("wide tree verify: more than %d grid shapes", kV8wDesc - 1);
    idx = v8w_n_++;
    uint32_t* q = d + 4 * (1 + idx);
    q[0] = x;
    q[1] = y;
    q[2] = z;
    q[3] = flags;
  }
  e.dispatch_indirect(v8w_args_, (size_t(v8w_region_) * kV8wDesc + size_t(idx)) * 16, tx);
}

bool Qwen35::enc_x8(Encoder& e, const Buffer& x, int R, int K, const Tensor* norm, bool lo, int x_ld,
                    const Buffer* xn_out, bool tw) {
  if (K % 64 || R < 1 || R > (v8w_R_ ? v8w_R_ : 8)) fatal("x8_prep: K=%d R=%d", K, R);
  // x8_prep2 (default): K/256 threadgroups, coalesced table stores; NI_V8_PREP=1: x8_prep (8 threadgroups)
  static const bool prep1 = env_int("NI_V8_PREP", 2) == 1;
  const bool v2 = !prep1 && K % 256 == 0;
  if (v8w_R_ && !v2) fatal("x8_prep: a wide tree verify needs x8_prep2 (K=%d)", K);
  const bool wt = tw && v2 && v8_tw() && size_t(K) * 16 <= v8tt_.size;
  if (norm && (v8_knobs().skip & 256)) return wt;
  kp::X8Prep p{uint32_t(K), uint32_t(R), uint32_t(x_ld ? x_ld : K), norm ? 1u : 0u, xn_out ? 1u : 0u, cfg_.eps,
               uint32_t(K), lo ? 1u : 0u, v8_knobs().prepu ? 1u : 0u, wt ? 1u : 0u};
  e.use(pipe8(v2 ? "x8_prep2" : "x8_prep"));
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, v8t_);
  e.buf(3, v8s_);
  e.buf(4, xn_out ? *xn_out : x);
  e.val(5, p);
  e.buf(6, v8tl_);
  e.buf(7, v8sl_);
  e.buf(8, wt ? v8tt_ : v8tl_);  // (unused unless write_t)
  if (v8w_R_) v8_disp(e, uint32_t(K / 256), 1, 1, kV8wY, 256);  // a wide tree verify: the live row tiles
  else e.dispatch(v2 ? uint32_t(K / 256) : 8u, 1, 1, 256);
  return wt;
}

// Tiling of the qmm8 grid: nf fragments (8 rows each) per simdgroup, cs column blocks per threadgroup,
// ks k-slices per column block (fixed-order sum in threadgroup memory), cs * ks <= 8 simdgroups.
// k-slices are added until the grid has ~64 simdgroups per GPU core (1280 here) or a slice would drop below
// 4 groups of 64 inputs. NI_V8_NF / NI_V8_CS / NI_V8_KS override (tuning).
struct V8Plan {
  int nf, cs, ks;
};
// Default (bench_v8 tiling sweep, 27B shapes: gate+up 34816x5120 457 vs 504 us at cs 4 / ks 1, LM head
// 3141-3199 vs 3354-3380): 8 simdgroups per threadgroup as 2 column blocks x 4 k-slices whenever a slice
// keeps >= 4 groups (K >= 1024), 8 k-slices for grids under ~1280 simdgroups. NI_V8_PLAN=1: the first
// plan (cs 4, k-slices only until the grid has 1280 simdgroups).
static V8Plan v8_plan(uint32_t N, uint32_t K, int nf_def) {
  static const int nf_env = env_int("NI_V8_NF", 0), cs_env = env_int("NI_V8_CS", 0), ks_env = env_int("NI_V8_KS", 0);
  static const bool plan1 = env_int("NI_V8_PLAN", 2) == 1;
  // 64 simdgroups per GPU core: 1280 on the 20-core M4 Pro the plans were tuned on (Gpu::core_count)
  static const uint32_t full = 64u * uint32_t(std::max(1, Gpu::get().core_count()));
  V8Plan p{nf_env ? nf_env : nf_def, cs_env ? cs_env : 4, 1};
  const uint32_t blocks = (N + 8 * p.nf - 1) / (8 * p.nf), gpr = K / 64;
  if (ks_env) p.ks = ks_env;
  else if (plan1)
    while (p.ks < 8 && blocks * p.ks < full && gpr / (p.ks * 2) >= 4) p.ks *= 2;
  else {
    p.ks = gpr >= 16 ? 4 : gpr >= 8 ? 2 : 1;
    if (p.ks == 4 && gpr >= 32 && blocks * 4 < full) p.ks = 8;
    if (!cs_env) p.cs = 8;
    // kplan (V8Knobs): with >= 8 groups per slice, 8 k-slices x 1 column block per threadgroup (kplan 1), and 16
    // k-slices (512 threads) for K >= 16384 (kplan 2). bench_v8 cmp, CM 1 + pairs, 27B shapes (min of 12 interleaved
    // runs, cs1/ks8 vs the cs2/ks4 plan): gate+up 427 vs 446 us, GDN in_proj 209 vs 223, o 84 vs 88, LM head
    // 3038 vs 3103; down 240 (ks8) / 232 (ks16) vs 256
    const int kplan = v8_knobs().kplan;
    if (kplan == 3 && nf_def == 2 && !ks_env) {
      // kplan 3, the M1's (14-core M1 Pro, bench_v8 cmp of every cs x ks on the 4B's and 9B's shapes, CM 1 + pairs,
      // median of 9 interleaved runs against kplan 2): k-slices of >= 16 groups (8 at most) with one column block
      // per threadgroup; grids that would exceed 16384 simdgroups (the LM heads) without k-slices, 2 column blocks.
      // 9B: gate+up 24576x4096 521 vs 565 us, GDN in_proj 268 vs 290, q|k|v 222 vs 240, o / GDN out 97 vs 104,
      // LM head 4926 vs 5597, down (ks 8 either way) 284; 4B: gate+up 18432x2560 254 vs 270, GDN in_proj 174 vs 184,
      // q|k|v 146 vs 154, o / GDN out 65 vs 68, LM head 3207 vs 3489, down 139 either way. On each of the 12 shapes
      // this is the fastest of the 26 variants measured (nf 1 / 2, cs x ks, the other inner loops).
      p.ks = 1;
      while (p.ks < 8 && gpr / (p.ks * 2) >= 16) p.ks *= 2;
      while (p.ks > 1 && blocks * p.ks > 16384) p.ks /= 2;
      if (!cs_env) p.cs = p.ks == 1 ? 2 : 1;
    } else if (kplan >= 1 && !ks_env && gpr >= 64) {
      p.ks = kplan >= 2 && gpr >= 256 ? 16 : 8;
      if (!cs_env) p.cs = 1;
    }
  }
  p.cs = std::max(1, std::min(p.cs, (p.ks > 8 ? 16 : 8) / p.ks));
  return p;
}

// qmm8t's tiling: NT weight rows per simdgroup (16; 32 for grids beyond 8192 threadgroups of 16, the LM heads), ks
// k-slices per threadgroup (4 for K 2049-4096, else 8), PF prefetching the next group (all but K <= 2048 at NT 16).
// bench_v8 cmp / bench_tq (median of 8-15 interleaved runs, us; vs qmm8snb with the M5's v8_plan and with its best
// tiling nf1 cs1 ks8): 35B GDN in_proj 12352x2048 31.0 vs 39.1 / 34.9; q|k|v 9216x2048 24.5 vs 28.9 / 27.2; GDN out /
// o 2048x4096 12.8 vs 16.4 / 15.4; LM head 248320x2048 530 vs 638 / 655; 27B GDN in_proj 16480x5120 105.6 vs 114.5,
// o 5120x6144 43.7 vs 47.6, LM head 248320x5120 1503 vs 1577.
// NI_V8T_NT / NI_V8T_KS / NI_V8T_PF override (tuning).
struct V8tPlan {
  int nt, ks, pf;
};
static V8tPlan v8t_plan(uint32_t Ntot, uint32_t K) {
  static const int nt_env = env_int("NI_V8T_NT", 0), ks_env = env_int("NI_V8T_KS", 0), pf_env = env_int("NI_V8T_PF", -1);
  const uint32_t gpr = K / 64;
  V8tPlan p{(Ntot + 15) / 16 > 8192 ? 32 : 16, gpr > 32 && gpr <= 64 ? 4 : 8, 0};
  if (nt_env == 16 || nt_env == 32) p.nt = nt_env;
  if (ks_env >= 1 && ks_env <= 8) p.ks = ks_env;
  p.pf = pf_env >= 0 ? (pf_env ? 1 : 0) : (p.nt == 32 || gpr > 32 ? 1 : 0);
  while (p.ks > 1 && gpr < uint32_t(p.ks)) p.ks /= 2;
  return p;
}

void Qwen35::enc_qmm8(Encoder& e, const std::vector<Seg8>& segs, int R, const Buffer& table, const Buffer& sums,
                      bool x2, const Buffer* tt) {
  if (segs.empty() || segs.size() > 4) fatal("qmm8: %zu segments", segs.size());
  if (R < 1 || R > (v8w_R_ ? v8w_R_ : 8)) fatal("qmm8: R=%d", R);
  const uint32_t K = segs[0].L->K;
  const int bits = segs[0].L->bits;
  // segments of different formats: one GEMM per format
  for (size_t i = 1; i < segs.size(); ++i)
    if (!segs[i].L->same_format(*segs[0].L)) {
      std::vector<Seg8> same, rest;
      for (auto& sg : segs) (sg.L->same_format(*segs[0].L) ? same : rest).push_back(sg);
      enc_qmm8(e, same, R, table, sums, x2, tt);
      enc_qmm8(e, rest, R, table, sums, x2, tt);
      return;
    }
  uint32_t Ntot = 0;
  for (auto& sg : segs) {
    if (sg.L->K != K) fatal("qmm8: segments must share K");
    Ntot += sg.L->N;
  }
  if (!segs[0].L->q4() && !segs[0].L->q8()) {  // wfmt.metal's qmm8g_<fmt>: bf16, 3/5/6-bit, group 32
    V8Plan pl = v8_plan(Ntot, K, 1);
    pl.nf = std::min(pl.nf, 2);
    if (pl.cs * pl.ks > 8) pl.ks = 8, pl.cs = 1;
    kp::Qmm8 p{};
    p.K = K;
    p.R = uint32_t(R);
    p.nseg = uint32_t(segs.size());
    p.ks = uint32_t(pl.ks);
    uint32_t tg = 0;
    e.use(pipe8("qmm8g_" + segs[0].L->fmt() + "_nf" + std::to_string(pl.nf)));
    for (size_t i = 0; i < 4; ++i) {
      const Seg8& sg = segs[std::min(i, segs.size() - 1)];
      if (i < segs.size()) {
        p.N[i] = sg.L->N;
        p.y_ld[i] = sg.y_ld ? sg.y_ld : sg.L->N;
        p.add[i] = sg.add ? 1u : 0u;
        p.tg_begin[i] = tg;
        const uint32_t blocks = (sg.L->N + 8 * pl.nf - 1) / (8 * pl.nf);
        tg += (blocks + pl.cs - 1) / pl.cs;
      }
      e.buf(int(i * 4 + 0), sg.L->w->buf);
      e.buf(int(i * 4 + 1), sg.L->s->buf);
      e.buf(int(i * 4 + 2), sg.L->b->buf);
      e.buf(int(i * 4 + 3), sg.y);
    }
    p.tg_begin[segs.size()] = tg;
    e.buf(16, table);
    e.buf(17, sums);
    e.val(18, p);
    if (v8w_R_) v8_disp(e, tg, 1, 1, kV8wX, 32 * pl.cs * pl.ks);  // a wide tree verify: x the live row tiles
    else e.dispatch(tg, 1, 1, 32 * pl.cs * pl.ks);
    return;
  }
  if (tt && !x2 && bits == 4 && segs[0].L->q4() && segs[0].L->gs == 64 && K % 64 == 0 && v8_tw() &&
      size_t(K) * 16 <= tt->size) {
    const V8tPlan pl = v8t_plan(Ntot, K);
    kp::Qmm8 p{};
    p.K = K;
    p.R = uint32_t(R);
    p.nseg = uint32_t(segs.size());
    p.ks = uint32_t(pl.ks);
    uint32_t tg = 0;
    // a wide tree verify: the tile-pair kernel (one 16-row matmul per weight load) unless NI_V8T_PAIR=0
    static const bool pair_env = env_int("NI_V8T_PAIR", 1) != 0;
    const bool pair = v8w_R_ && pair_env;
    e.use(pipe8(std::string(pl.pf ? (pair ? "qmm8t2p" : "qmm8tp") : (pair ? "qmm8t2" : "qmm8t")) + "_b4_nt" + std::to_string(pl.nt)));
    for (size_t i = 0; i < 4; ++i) {
      const Seg8& sg = segs[std::min(i, segs.size() - 1)];
      if (i < segs.size()) {
        p.N[i] = sg.L->N;
        p.y_ld[i] = sg.y_ld ? sg.y_ld : sg.L->N;
        p.add[i] = sg.add ? 1u : 0u;
        p.tg_begin[i] = tg;
        tg += (sg.L->N + pl.nt - 1) / pl.nt;
      }
      e.buf(int(i * 4 + 0), sg.L->w->buf);
      e.buf(int(i * 4 + 1), sg.L->s->buf);
      e.buf(int(i * 4 + 2), sg.L->b->buf);
      e.buf(int(i * 4 + 3), sg.y);
    }
    p.tg_begin[segs.size()] = tg;
    e.buf(16, *tt);
    e.buf(17, sums);
    e.val(18, p);
    if (v8w_R_) v8_disp(e, tg, 1, 1, pair ? kV8wXpair : kV8wX, 32 * pl.ks);
    else e.dispatch(tg, 1, 1, 32 * pl.ks);
    return;
  }
  if (segs[0].L->gs != 64) fatal("qmm8: group %d", segs[0].L->gs);
  V8Plan pl = v8_plan(Ntot, K, bits == 8 ? 1 : 2);
  if (x2) pl.nf = 1;
  else if (bits == 8) pl.nf = std::min(pl.nf, 2);
  // NI_V8_GEMM picks the 4-bit inner loop: pf (default: software-pipelined), base, s (split chains), pfs
  static const std::string var = getenv("NI_V8_GEMM") && *getenv("NI_V8_GEMM") ? getenv("NI_V8_GEMM") : "pf";
  std::string pre = x2 ? "qmm8x2" : bits == 4 && var != "base" ? "qmm8" + var : "qmm8";
  const V8Knobs& kn = v8_knobs();
  const bool pair = kn.pair && (K / 64) % 2 == 0;  // scale / bias pairs need an even group count per row
  if (!x2 && bits == 4 && var == "pf" && (kn.cm == 1 || pair))
    pre = std::string("qmm8") + (kn.cm == 1 ? "sn" : "") + (pair ? "b" : (kn.cm == 1 ? "" : "pf"));
  if (pl.cs * pl.ks > 8) {  // only the CM 1 + pair loop has a 16-simdgroup instantiation
    if (pre == "qmm8snb" && (pl.nf == 1 || pl.nf == 2)) pre += "16";
    else pl.ks = 8, pl.cs = 1;
  }
  const std::string name = pre + "_b" + std::to_string(bits) + "_nf" + std::to_string(pl.nf);
  kp::Qmm8 p{};
  p.K = K;
  p.R = uint32_t(R);
  p.nseg = uint32_t(segs.size());
  p.ks = uint32_t(pl.ks);
  uint32_t tg = 0;
  e.use(pipe8(name));
  for (size_t i = 0; i < 4; ++i) {
    const Seg8& sg = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = sg.L->N;
      p.y_ld[i] = sg.y_ld ? sg.y_ld : sg.L->N;
      p.add[i] = sg.add ? 1u : 0u;
      p.tg_begin[i] = tg;
      const uint32_t blocks = (sg.L->N + 8 * pl.nf - 1) / (8 * pl.nf);
      tg += (blocks + pl.cs - 1) / pl.cs;
    }
    e.buf(int(i * 4 + 0), sg.L->w->buf);
    e.buf(int(i * 4 + 1), sg.L->s->buf);
    e.buf(int(i * 4 + 2), sg.L->b->buf);
    e.buf(int(i * 4 + 3), sg.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.buf(16, table);
  e.buf(17, sums);
  e.val(18, p);
  e.buf(19, x2 ? v8tl_ : table);
  e.buf(20, x2 ? v8sl_ : sums);
  if (v8w_R_ && x2) fatal("qmm8: a wide tree verify has no two-term GEMM");
  if (v8w_R_) v8_disp(e, tg, 1, 1, kV8wX, 32 * pl.cs * pl.ks);
  else e.dispatch(tg, 1, 1, 32 * pl.cs * pl.ks);
  for (int i = 0; i < kn.pad; ++i) {  // timing only (V8Knobs::pad): empty dispatches
    const uint32_t zero = 0;
    e.use(pipe8("copy_u32"));
    e.buf(0, table); e.buf(1, v8cnt_);
    e.val(2, zero);
    e.dispatch(1, 1, 1, 32);
  }
}

// Attention block for R rows at positions pos..pos+R-1: x_ += o_proj(attn(norm(x_))).
void Qwen35::enc_attn8(Encoder& e, const Seq& s, const Layer& L, const Buffer& kc, const Buffer& vc, int cap, int pos,
                       int R, bool table_ready) {
  const int D = cfg_.hidden, HD = cfg_.hq * cfg_.head_dim;
  const int skip = v8_knobs().skip;
  const bool tq = table_ready ? v8_tw() : enc_x8(e, x_, R, D, L.in_norm, false, 0, nullptr, true);
  if (!(skip & 512)) enc_qmm8(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, R, v8t_, v8s_, false, tq ? &v8tt_ : nullptr);
  if (skip & 2) {
    enc_qmm8(e, {{&L.o, x_, uint32_t(D), true}}, R, v8t_, v8s_);
    return;
  }
  kp::AttnPrep ap{uint32_t(R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                  uint32_t(cfg_.rot_dims), uint32_t(pos), uint32_t(cap), cfg_.eps, cfg_.theta};
  const bool tree = v8_tree_ != nullptr;  // DFlash2 tree drafting (set_verify_tree)
  e.use(pipe8(tree ? "attn_prep_tree" : mrope(s) ? "attn_prep_m" : "attn_prep"));
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, kc); e.buf(7, vc);
  e.val(8, ap);
  if (mrope(s) || tree) enc_mrope_args(e, s);
  if (tree) e.buf(11, *v8_tree_);
  if (v8w_R_) v8_disp(e, uint32_t(cfg_.hq + 2 * cfg_.hkv), 1, 1, kV8wYrows, uint32_t(cfg_.head_dim));  // the live rows
  else e.dispatch(cfg_.hq + 2 * cfg_.hkv, R, 1, cfg_.head_dim);
  // NI_V8_ATTN: 1 = attn8m_partial (MMA, all rows and heads of a kv head per threadgroup),
  // 0 = attn8_partial (scalar, RB rows per threadgroup)
  static const int attn_mma = env_int("NI_V8_ATTN", 1);
  const int G = cfg_.hq / cfg_.hkv;
  const bool mma = attn_mma && G <= 16 && cfg_.head_dim == 256;
  if (tree && !mma) fatal("tree verify: the 8-row attention must run attn8m_partial");
  // history chunks (flash decoding): ~8 threadgroups per core over the kv heads, >= 64 keys each,
  // at most 1024 / R (partials buffer)
  static const int tg_target = env_int("NI_V8_ATTN_TGS", 160);
  // NI_V8_TREE_ATTN=0: the tree block's keys inside the MMA tiles (attn8m_partial_tree), whose rounding follows the
  // rows' slots; else the MMA tiles take the history alone and attn8_tree_block the block, one more chunk
  // NI_V8_TREE_ATTN=2 (default): tree_zone -- the MMA tiles take each row's keys below a point that follows its logical
  // position alone, in chunks of one fixed size, and attn8_tree_zone the keys from there to the row (kernels/verify8.metal)
  static const int tree_attn = env_int("NI_V8_TREE_ATTN", 2);
  // A wide tree verify (v8w_R_: 16 / 32 rows, dense targets) always takes the first: its row tiles run attn8m_partial_tree
  // with the tree block's keys inside the MMA tiles; the zone and block kernels are 8-row kernels
  const bool zone = tree && tree_attn >= 2 && !v8w_R_;
  const bool block = tree && tree_attn == 1 && !v8w_R_;
  const uint32_t Lmax = uint32_t(pos + R), gran = mma ? 32 : 16, extra = (block || zone) ? 1u : 0u;
  const uint32_t want = std::max<uint32_t>(1, (uint32_t(tg_target) + cfg_.hkv - 1) / cfg_.hkv);
  uint32_t chunk = std::max<uint32_t>(64, (Lmax + want - 1) / want);
  if (zone) {
    // a fixed size: it follows the sequence's capacity, never the anchor, so a position's chunk is the same in every
    // tree and cycle
    static const uint32_t zc = uint32_t(env_int("NI_V8_TREE_CHUNK", 128));
    chunk = std::max<uint32_t>(zc, 32);
    while (((uint32_t(cap) + chunk - 1) / chunk + extra) * uint32_t(R) > 1024) chunk += 32;
  }
  chunk = (chunk + gran - 1) / gran * gran;
  while (!zone && ((Lmax + chunk - 1) / chunk + extra) * uint32_t(R) > 1024) chunk += gran;
  // the zone's MMA keys end at or before pos: no chunk past it holds one
  const uint32_t nch = zone ? std::max<uint32_t>(1, (uint32_t(pos) + chunk - 1) / chunk) : (Lmax + chunk - 1) / chunk;
  kp::AttnRows rp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos), chunk, uint32_t(cap),
                  1.0f / std::sqrt(float(cfg_.head_dim)), nch + extra};
  const uint32_t Ru = uint32_t(R);
  if (mma) {
    e.use(pipe8(zone ? "attn8m_partial_zone" : block ? "attn8m_partial_hist" : tree ? "attn8m_partial_tree" : "attn8m_partial"));
    e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
    e.val(5, rp);
    e.val(6, Ru);
    if (zone || (tree && !block)) e.buf(7, *v8_tree_);
    // a wide tree verify: the chunks x the live row tiles along x (a chunk's tiles side by side)
    if (v8w_R_) v8_disp(e, nch, uint32_t(cfg_.hkv), 1, kV8wX, 64 * G);
    else e.dispatch(nch, cfg_.hkv, 1, 64 * G);
    if (block || zone) {
      e.use(pipe8(zone ? "attn8_tree_zone" : "attn8_tree_block"));
      e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
      e.val(5, rp);
      e.buf(6, *v8_tree_);
      e.dispatch(cfg_.hq, R, 1, 32);
    }
  } else {
    static const int rb_env = env_int("NI_V8_ATTN_RB", 0);
    const int RB = std::max(1, std::min(rb_env ? rb_env : 32 / G, R));
    e.use(pipe8("attn8_partial"));
    e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, part_o_); e.buf(4, part_ml_);
    e.val(5, rp);
    e.val(6, Ru);
    e.dispatch(nch, cfg_.hkv, (R + RB - 1) / RB, 32 * G * RB);
  }
  e.use(pipe8("attn_rows_reduce"));
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, rp);
  if (v8w_R_) v8_disp(e, uint32_t(cfg_.hq), 1, 1, kV8wYrows, uint32_t(cfg_.head_dim));
  else e.dispatch(cfg_.hq, R, 1, cfg_.head_dim);
  const bool to = enc_x8(e, attn_out_, R, HD, nullptr, false, 0, nullptr, true);
  if (!(skip & 64)) enc_qmm8(e, {{&L.o, x_, uint32_t(D), true}}, R, v8t_, v8s_, false, to ? &v8tt_ : nullptr);
}

bool Qwen35::enc_gu8(Encoder& e, const QLinear& gate, const QLinear& up, const Buffer& table, const Buffer& sums,
                     const Buffer& h, const Buffer& hs, const Buffer* tt, const Buffer* ht, int R) {
  v8_alloc();  // the split path's scratch (a drafter may encode before the first verify)
  const uint32_t I = gate.N, K = gate.K;
  const bool q4 = gate.q4() && up.q4();  // the fused qmm8_gu kernels read 4-bit group 64
  if (I % 64 || up.N != I || up.K != K || !gate.same_format(up)) fatal("gu8: shapes");
  if ((v8_knobs().gusplit || !q4) && v8g_ && I <= uint32_t(cfg_.dense_I) && I % 256 == 0) {
    // gate and up as two segments of one GEMM (v8_plan's tiling, fp32 rows) + SwiGLU into the down GEMM's table:
    // the fused qmm8_gu1k2 kernel (8 column slices x 2 k-slices of one 64-column group) streamed ~10% slower.
    // A wide tree verify: R rows (its capacity), the live row tiles' tables (v8_disp)
    const uint32_t Rg = v8w_R_ ? uint32_t(R) : 8u;
    enc_qmm8(e, {{&gate, v8g_, I}, {&up, v8u_, I}}, int(Rg), table, sums, false, tt);
    const bool wt = ht && v8_tw() && size_t(I) * 16 <= ht->size;
    kp::SwigluX8 sp{I, Rg, wt ? 1u : 0u};
    e.use(pipe8("swiglu_x8"));
    e.buf(0, v8g_); e.buf(1, v8u_); e.buf(2, h); e.buf(3, hs);
    e.val(4, sp);
    e.buf(5, wt ? *ht : h);  // (unused unless write_t)
    if (v8w_R_) v8_disp(e, I / 256, 1, 1, kV8wY, 256);
    else e.dispatch(I / 256, 1, 1, 256);
    return wt;
  }
  if (v8w_R_) fatal("gu8: a wide tree verify needs the split gate / up path");
  if (!q4) fatal("gu8: %s gate/up need the split path (an intermediate size multiple of 256)", gate.fmt().c_str());
  kp::Qmm8Gu p{K, I, 0, 0, 0, 0, 0};
  e.use(pipe8(gu_kernel(K)));
  e.buf(0, gate.w->buf); e.buf(1, gate.s->buf); e.buf(2, gate.b->buf);
  e.buf(3, up.w->buf); e.buf(4, up.s->buf); e.buf(5, up.b->buf);
  e.buf(6, gate.w->buf); e.buf(7, gate.s->buf); e.buf(8, gate.b->buf);
  e.buf(9, up.w->buf); e.buf(10, up.s->buf); e.buf(11, up.b->buf);
  e.buf(12, table); e.buf(13, sums); e.buf(14, h); e.buf(15, hs);
  e.val(16, p);
  e.buf(17, sums); e.buf(18, sums);
  e.dispatch(I / 64, 1, 1, gu_threads());
  return false;
}

// Dense SwiGLU MLP for R rows: x_ += down(silu(gate(n)) * up(n)), n = norm(x_).
void Qwen35::enc_mlp8(Encoder& e, const Layer& L, int R) {
  const int D = cfg_.hidden;
  const uint32_t I = L.mg.N;
  if (I % 64 || L.mu.N != I || L.mg.K != uint32_t(D)) fatal("mlp8: shapes");
  // V8Knobs::twm (with tw): gate|up and down on qmm8t too; the T tables share v8tt_ (gate|up reads it before
  // swiglu_x8 overwrites it with h)
  const bool tm = v8_knobs().twm != 0;
  const bool tx = enc_x8(e, x_, R, D, L.post_norm, false, 0, nullptr, tm);
  const int skip = v8_knobs().skip;
  bool th = false;
  if (!(skip & 16))
    th = enc_gu8(e, L.mg, L.mu, v8t_, v8s_, v8h_, v8hs_, tx ? &v8tt_ : nullptr, tm ? &v8tt_ : nullptr, R);
  if (!(skip & 8)) enc_qmm8(e, {{&L.md, x_, uint32_t(D), true}}, R, v8h_, v8hs_, false, th ? &v8tt_ : nullptr);
}

// The moe8e passes (MMA-exact small experts) rely on the MMA adding its products one at a time in k order with an fp32
// rounding after each; mma_chain_check tests that on this GPU once (a fresh command buffer, before the first verify's
// is committed). NI_V8_MOE_EMU: 0 = never use them (sampled verifies run every routed expert as an MMA tile),
// 1 (default on the M5) = sampled verifies, 2 = greedy verifies too (instead of moe8s2's fp32 passes).
static int moe_emu_mode() {
  static const bool ok = [] {
    Gpu& g = Gpu::get();
    Buffer bad = g.alloc(16, "mma_check");
    memset(bad.ptr(), 0, 16);
    auto e = g.begin();
    e->use(pipe8("mma_chain_check"));
    e->buf(0, bad);
    e->dispatch(64, 1, 1, 32);
    e->commit_and_wait();
    const uint32_t n = bad.as<uint32_t>()[0];
    g.release(bad);
    if (n) logi("moe8e: this GPU's MMA rounds differently (%u of 4096 check values); sampled verifies keep MMA tiles", n);
    return n == 0;
  }();
  // default on the M5 (Apple10), where it was measured; other Macs keep the MMA tiles unless NI_V8_MOE_EMU is set
  static const int env = std::clamp(env_int("NI_V8_MOE_EMU", Gpu::get().apple10() ? 1 : 0), 0, 2);
  const int m = v8_knobs().emu >= 0 ? std::min(v8_knobs().emu, 2) : env;
  return m && ok ? m : 0;
}

// MoE block for R rows: x_ += sum_j w_j expert_j(n) + w_s shared(n), n = norm(x_).
bool Qwen35::enc_moe8(Encoder& e, const Layer& L, int R, int li, const Tensor* next_norm) {
  if (L.eg.iq) {  // llama.cpp's IQ blocks (qwen35_iq.cpp)
    enc_moe8_iq(e, L, R, li);
    return false;
  }
  const int D = cfg_.hidden, k = cfg_.top_k, E = cfg_.experts;
  const uint32_t I = uint32_t(cfg_.moe_I);
  // other expert formats: wfmt.metal's moe8g_* passes for the format pairs they have (the experts de-duplicated over
  // the rows, as the 4-bit passes; no scalar small-expert pass), else the per-row MoE (router, top-k and expert
  // matvecs per row; row-independent)
  const bool q4 = L.eg.q4() && L.eu.q4() && L.ed.q4() && L.sg.q4() && L.su.q4() && L.sd.q4();
  std::string combo;
  if (!q4) {
    static const std::set<std::string> have = {"f16_f16",     "f8g64_f8g64", "f6g64_f6g64", "f6g64_f8g64",
                                               "f5g64_f5g64", "f5g64_f8g64", "f4g64_f8g64", "f3g64_f8g64"};
    combo = L.eg.fmt() + "_" + L.sg.fmt();
    if (!have.count(combo) || !L.eg.same_format(L.eu) || !L.eg.same_format(L.ed) || !L.sg.same_format(L.su) ||
        !L.sg.same_format(L.sd)) {
      enc_moe_rows(e, L, R, x_, false);
      return false;
    }
  }
  if (cfg_.shared_I != cfg_.moe_I || I % 64 || E > 1024 || k > 8 || D % 64) fatal("moe8: unsupported expert layout");
  // two-term (~22-bit) inputs and exact codes for the router: keeps the top-8 choice at fp32 level
  static const bool precise = env_int("NI_V8_ROUTER_X2", 1) != 0;
  // experts used by <= small rows run as register-dequant matvecs over just those rows instead of an
  // 8-row MMA tile (35B p700: 24.0 vs 24.9 ms per verify)
  static const int small_env = std::clamp(env_int("NI_V8_MOE_SMALL", 2), 0, 2);
  // MMA-exact passes (moe8e: bitwise what the MMA tiles give the rows) for 2-row tiles: list A (experts used by
  // <= small rows) takes them on sampled verifies (row-independent numerics; NI_V8_MOE_EMU=2: greedy ones too), else
  // moe8s2's fp32 passes; experts with small < rows <= split become ceil(rows / 2) such tiles (in list A, or list B when
  // A runs moe8s2), and with shs (split 8) the shared expert too: no MMA tile pass left. 35B, 64 tokens teacher-forced,
  // sampled-verify numerics (tools/bench_verify8, interleaved): every expert on MMA tiles 11.78 ms per verify; 1-2-row
  // experts on moe8e 11.05; + split 4 10.81; split 8 + shs 10.15.
  // the moe8e passes reproduce the default MMA tiles (gate/up qmm8_gu1k2snb_b4: subnormal codes, scale / bias pairs,
  // its K slices; down moe8_down_nf2): other NI_V8_GU / NI_V8_CM / NI_V8_PAIR / NI_V8_MOE_NF settings keep the MMA tiles
  static const bool dn_default = env_int("NI_V8_MOE_NF", 2) == 2;
  const int emode =
      q4 && D == 2048 && I == 512 && dn_default && gu_kernel(uint32_t(D)) == "qmm8_gu1k2snb_b4" ? moe_emu_mode() : 0;
  const bool emu = emode >= (v8_moe_exact_ ? 1 : 2);
  const int small = !q4 ? 0 : v8_moe_exact_ && !emu ? 0 : small_env;
  static const int split_env = std::clamp(env_int("NI_V8_MOE_SPLIT", 8), 0, 8);
  static const int shs_env = env_int("NI_V8_MOE_SHS", 1);
  // NI_V8_MOE_SEP=1: greedy verifies (list A on moe8s2) split too, into list B (same numerics: moe8e = MMA tiles), but
  // slower: 11.28 vs 10.97 ms per verify (moe8s2 + MMA tiles); NI_V8_MOE_EMU=2 (everything on moe8e) 10.21 ms, but
  // greedy logits then move from moe8s2's fp32 numerics to the MMA tiles' (ni_v8check: mean |diff| vs plain decode
  // 0.031 instead of 0.020)
  static const bool sep_env = env_int("NI_V8_MOE_SEP", 0) != 0;
  const bool sep_on = emode && !emu && sep_env;
  const int split = emu || sep_on ? std::max(v8_knobs().split >= 0 ? std::clamp(v8_knobs().split, 0, 8) : split_env, small)
                                  : small;
  const bool shs = (emu || sep_on) && split >= 8 && (v8_knobs().shs >= 0 ? v8_knobs().shs : shs_env) != 0;
  const bool sep = sep_on && (split > small || shs);  // list B: moe8e beside list A's moe8s2
  enc_x8(e, x_, R, D, L.post_norm, precise, 0, small && !emu ? &v8xn_ : nullptr);  // moe8s2 reads fp32 rows
  enc_qmm8(e, {{&L.router, rlog_, uint32_t(E)}, {&L.shared_gate, slog_, 1u}}, R, v8t_, v8s_, precise);
  kp::Topk tp{uint32_t(E), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  // the router top-k inside moe8_plan (E == 256: its 8 simdgroups are the rows; the same topk256_row code)
  static const bool topk_env = env_int("NI_V8_MOE_TOPKPLAN", 1) != 0;
  const bool topk_in_plan = E == 256 && k <= 8 && (v8_knobs().tkp >= 0 ? v8_knobs().tkp != 0 : topk_env);
  if (!topk_in_plan) {
    e.use(pipe8(E == 256 && k <= 8 ? "moe_topk256_rows" : "moe_topk_rows"));
    e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
    e.val(4, tp);
    e.dispatch(R, 1, 1, 32);
  }
  static const int dn_nf = env_int("NI_V8_MOE_NF", 2), dn_cs = 4;
  if (D % (8 * dn_nf * dn_cs)) fatal("moe8: hidden %d", D);
  // scalar passes: 8 simdgroups x 4 columns per threadgroup, inputs staged (hidden <= 2048)
  constexpr uint32_t s_nsg = 8, SR = 4;
  if (small && D > 2048) fatal("moe8s2: hidden %d > 2048", D);
  kp::Moe8Plan pp{uint32_t(R), uint32_t(k), uint32_t(E), I / 64, uint32_t(D / (8 * dn_nf * dn_cs)), uint32_t(li),
                  uint32_t(small), emu ? I / 64 : I / (SR * s_nsg), emu ? uint32_t(D) / 128 : uint32_t(D) / (SR * s_nsg),
                  v8_rlim_ ? 1u : 0u, uint32_t(split), shs ? 1u : 0u, sep ? 1u : 0u, uint32_t(D) / 128,
                  topk_in_plan ? 1u : 0u};
  e.use(pipe8("moe8_plan"));
  e.buf(0, ridx_); e.buf(1, v8tiles_); e.buf(2, v8nt_); e.buf(3, v8slot_); e.buf(4, v8args_);
  e.val(5, pp);
  e.buf(6, v8ustat_); e.buf(7, v8sm_);
  e.buf(8, v8_rlim_ ? *v8_rlim_ : v8nt_);
  e.buf(9, v8sm2_);
  e.buf(10, rlog_); e.buf(11, slog_); e.buf(12, rwts_);
  e.val(13, tp);
  e.dispatch(1, 1, 1, uint32_t(std::max(32, (E + 31) / 32 * 32)));
  // gate/up + SwiGLU per touched expert (+ the shared expert), one h table per tile; strides in the routed format's
  // words / scale groups
  const uint32_t wstride = uint32_t(size_t(I) * L.eg.row_bytes() / 4), sstride = L.eg.dense() ? 0u : I * (uint32_t(D) / L.eg.gs);
  const uint32_t dw = uint32_t(size_t(D) * L.ed.row_bytes() / 4), ds = L.ed.dense() ? 0u : uint32_t(D) * (I / L.ed.gs);
  if (!(shs && split >= 8)) {  // else no expert takes an MMA tile
    kp::Qmm8Gu gp{uint32_t(D), I, wstride, sstride, 1u, I * 8, I / 64 * 8};
    e.use(pipe8(q4 ? gu_kernel(uint32_t(D)) : "moe8g_gu_" + combo));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
    e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
    e.buf(12, v8t_); e.buf(13, v8s_); e.buf(14, v8h_); e.buf(15, v8hs_);
    e.val(16, gp);
    e.buf(17, v8tiles_); e.buf(18, v8nt_);
    e.dispatch_indirect(v8args_, 0, q4 ? gu_threads() : 512u);
    // down per tile -> per (row, slot) outputs, then the fixed-order combine
    kp::Moe8Down dp{I, uint32_t(D), dw, ds, I * 8, I / 64 * 8, uint32_t(k)};
    if (!q4 && dn_nf != 2) fatal("moe8g: NI_V8_MOE_NF must be 2 for %s experts", L.eg.fmt().c_str());
    e.use(pipe8(q4 ? "moe8_down_nf" + std::to_string(dn_nf) : "moe8g_down_" + combo));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
    e.buf(6, v8h_); e.buf(7, v8hs_); e.buf(8, v8out_); e.buf(9, v8sh_);
    e.val(10, dp);
    e.buf(11, v8tiles_); e.buf(12, v8nt_); e.buf(13, v8slot_);
    e.dispatch_indirect(v8args_, 12, 32 * dn_cs);
  }
  // moe8e passes over a tile list (sm, h, h sums, tile count ntiles[1] at byte offset nt_off, grids at args_off)
  auto moe8e = [&](const Buffer& sm, const Buffer& h, const Buffer& hsum, size_t nt_off, size_t args_off) {
    kp::Moe8e ep{uint32_t(D), I, wstride, sstride, uint32_t(k)};
    e.use(pipe8("moe8e_gu_r2"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, v8t_); e.buf(7, v8s_); e.buf(8, h); e.buf(9, hsum);
    e.val(10, ep);
    e.buf(11, sm); e.buf(12, v8nt_, nt_off);
    e.buf(13, L.sg.w->buf); e.buf(14, L.sg.s->buf); e.buf(15, L.sg.b->buf);
    e.buf(16, L.su.w->buf); e.buf(17, L.su.s->buf); e.buf(18, L.su.b->buf);
    e.dispatch_indirect(v8args_, args_off, 256);
    kp::Moe8e dq{I, uint32_t(D), dw, ds, uint32_t(k)};
    e.use(pipe8("moe8e_down_r2"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, h); e.buf(4, hsum); e.buf(5, v8out_);
    e.val(6, dq);
    e.buf(7, sm); e.buf(8, v8nt_, nt_off);
    e.buf(9, L.sd.w->buf); e.buf(10, L.sd.s->buf); e.buf(11, L.sd.b->buf); e.buf(12, v8sh_);
    e.dispatch_indirect(v8args_, args_off + 12, 128);
  };
  if (emu && (small || split > small || shs)) {
    moe8e(v8sm_, v8hsc_, v8hse_, 0, 24);
  } else if (small) {
    kp::Moe8s sp{uint32_t(D), I, wstride, sstride, uint32_t(k), uint32_t(D)};
    e.use(pipe8("moe8s2_gu"));
    e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
    e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
    e.buf(6, v8xn_); e.buf(7, v8hsc_);
    e.val(8, sp);
    e.buf(9, v8sm_); e.buf(10, v8nt_);
    e.dispatch_indirect(v8args_, 24, 32 * s_nsg);
    kp::Moe8s dq{I, uint32_t(D), dw, ds, uint32_t(k), I};
    e.use(pipe8("moe8s2_down"));
    e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
    e.buf(3, v8hsc_); e.buf(4, v8out_);
    e.val(5, dq);
    e.buf(6, v8sm_); e.buf(7, v8nt_);
    e.dispatch_indirect(v8args_, 36, 32 * s_nsg);
  }
  if (sep) moe8e(v8sm2_, v8hsc2_, v8hse2_, 4, 48);  // list B (its count in ntiles[2])
  // with the next norm: the combine also writes that norm's X8 table as x8_prep2 would (moe8_combine_x8: one
  // threadgroup per row, N / 4 threads; bitwise moe8_combine + x8_prep2)
  static const bool prep1 = env_int("NI_V8_PREP", 2) == 1;
  const V8Knobs& kn = v8_knobs();
  // with qmm8t on (v8_tw()), also its T table, so the next GEMMs find both tables written
  const bool wt = v8_tw();
  if (next_norm && (kn.fuse & 1) && !(kn.skip & 256) && !prep1 && D % 256 == 0 && D <= 2048 &&
      (!wt || size_t(D) * 16 <= v8tt_.size)) {
    kp::Moe8CombineX8 cp{uint32_t(D), uint32_t(k), uint32_t(R), cfg_.eps, kn.prepu ? 1u : 0u, wt ? 1u : 0u};
    e.use(pipe8("moe8_combine_x8"));
    e.buf(0, v8out_); e.buf(1, v8sh_); e.buf(2, rwts_); e.buf(3, x_);
    e.buf(4, next_norm->buf); e.buf(5, v8t_); e.buf(6, v8s_);
    e.val(7, cp);
    e.buf(8, wt ? v8tt_ : v8t_);  // (unused unless write_t)
    e.dispatch(8, 1, 1, uint32_t(D / 4));
    return true;
  }
  kp::Moe8Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe8("moe8_combine"));
  e.buf(0, v8out_); e.buf(1, v8sh_); e.buf(2, rwts_); e.buf(3, x_);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, R, 1, 64);
  return false;
}

// dev (tools/bench_verify8 BENCH_REPLAY): the last verify's expert passes again (its plan: the last MoE layer's tiles
// in list A,
// the final input table) with layer li's weights; mask: 1 tile gate/up, 2 tile down, 4 small gate/up, 8 small down
// (V8Knobs::emu 0: the moe8s2 passes, else the moe8e ones). Timing only.
void Qwen35::moe8_replay(Encoder& e, int li, int mask) {
  const Layer& L = layers_[li];
  const int D = cfg_.hidden, k = cfg_.top_k;
  const uint32_t I = uint32_t(cfg_.moe_I);
  const uint32_t U1 = v8nt_.as<uint32_t>()[0], S = std::max(1u, v8nt_.as<uint32_t>()[1]);
  static bool said = false;
  if (!said) said = true, logi("moe8_replay: %u MMA tiles (+ shared), %u small tiles", U1 - 1, v8nt_.as<uint32_t>()[1]);
  const uint32_t wstride = uint32_t(size_t(I) * L.eg.row_bytes() / 4), sstride = I * (uint32_t(D) / L.eg.gs);
  const uint32_t dw = uint32_t(size_t(D) * L.ed.row_bytes() / 4), ds = uint32_t(D) * (I / L.ed.gs);
  const bool s2 = v8_knobs().emu == 0;
  // BENCH_GROUPS with a grouped encoder: the tile and small passes of a phase as one group (run side by side)
  const bool grp = e.grouped() && getenv("BENCH_GROUPS");
  for (int phase = 0; phase < 2; ++phase) {
    const int pm = mask & (phase ? 10 : 5);
    if (grp) e.group_begin();
    if (pm & 1) {
      kp::Qmm8Gu gp{uint32_t(D), I, wstride, sstride, 1u, I * 8, I / 64 * 8};
      e.use(pipe8(gu_kernel(uint32_t(D))));
      e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
      e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
      e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
      e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
      e.buf(12, v8t_); e.buf(13, v8s_); e.buf(14, v8h_); e.buf(15, v8hs_);
      e.val(16, gp);
      e.buf(17, v8tiles_); e.buf(18, v8nt_);
      e.dispatch(I / 64, U1, 1, gu_threads());
    }
    if (pm & 2) {
      kp::Moe8Down dp{I, uint32_t(D), dw, ds, I * 8, I / 64 * 8, uint32_t(k)};
      e.use(pipe8("moe8_down_nf2"));
      e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
      e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
      e.buf(6, v8h_); e.buf(7, v8hs_); e.buf(8, v8out_); e.buf(9, v8sh_);
      e.val(10, dp);
      e.buf(11, v8tiles_); e.buf(12, v8nt_); e.buf(13, v8slot_);
      e.dispatch(uint32_t(D) / 64, U1, 1, 128);
    }
    if (pm & 4) {
      e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
      e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
      if (s2) {
        kp::Moe8s sp{uint32_t(D), I, wstride, sstride, uint32_t(k), uint32_t(D)};
        e.use(pipe8("moe8s2_gu"));
        e.buf(6, v8xn_); e.buf(7, v8hsc_);
        e.val(8, sp);
        e.buf(9, v8sm_); e.buf(10, v8nt_);
        e.dispatch(I / 32, S, 1, 256);
      } else {
        kp::Moe8e ep{uint32_t(D), I, wstride, sstride, uint32_t(k)};
        e.use(pipe8("moe8e_gu_r2"));
        e.buf(6, v8t_); e.buf(7, v8s_); e.buf(8, v8hsc_); e.buf(9, v8hse_);
        e.val(10, ep);
        e.buf(11, v8sm_); e.buf(12, v8nt_);
        e.buf(13, L.sg.w->buf); e.buf(14, L.sg.s->buf); e.buf(15, L.sg.b->buf);
        e.buf(16, L.su.w->buf); e.buf(17, L.su.s->buf); e.buf(18, L.su.b->buf);
        e.dispatch(I / 64, S, 1, 256);
      }
    }
    if (pm & 8) {
      e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
      if (s2) {
        kp::Moe8s dq{I, uint32_t(D), dw, ds, uint32_t(k), I};
        e.use(pipe8("moe8s2_down"));
        e.buf(3, v8hsc_); e.buf(4, v8out_);
        e.val(5, dq);
        e.buf(6, v8sm_); e.buf(7, v8nt_);
        e.dispatch(uint32_t(D) / 32, S, 1, 256);
      } else {
        kp::Moe8e dq{I, uint32_t(D), dw, ds, uint32_t(k)};
        e.use(pipe8("moe8e_down_r2"));
        e.buf(3, v8hsc_); e.buf(4, v8hse_); e.buf(5, v8out_);
        e.val(6, dq);
        e.buf(7, v8sm_); e.buf(8, v8nt_);
        e.buf(9, L.sd.w->buf); e.buf(10, L.sd.s->buf); e.buf(11, L.sd.b->buf); e.buf(12, v8sh_);
        e.dispatch(uint32_t(D) / 128, S, 1, 128);
      }
    }
    if (grp) e.group_end();
  }
}

void Qwen35::encode_verify8(Encoder& e, Seq& s, int pos, int R, const SamplingParams& sp, uint32_t step) {
  // a tree of more than 8 rows: a wide tree verify (16 or 32 rows: GPU-driven row tiles, v8_wide_begin)
  const bool wide = v8_tree_ != nullptr && R > 8;
  // R whole 8-row tiles of the capacity set_tree_rows sized (8, 16 or 32 rows): the tile-pair MLP reads two tiles per
  // pair, so an odd tile count's last pair reaches one tile past R but stays inside the capacity's tables
  if (R < 1 || R > (wide ? tree_rows_ : 8) || (wide && R % 8))
    fatal("encode_verify8: R=%d: the verify tables hold %d rows (set_tree_rows)", R, wide ? tree_rows_ : 8);
  if (pos + R + 1 >= s.cap) fatal("context full (%d)", s.cap);
  v8_alloc();
  const int D = cfg_.hidden;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  if (wide) v8_wide_begin(e, s, R);
  if (wide) {  // the live rows' embeddings
    const QLinear& E = embed_;
    kp::Embed ep{uint32_t(D), uint32_t(E.gs), uint32_t(pos)};
    e.use(pipe8(E.q4() ? std::string("embed_q4") : "embed_" + E.fmt()));
    e.buf(0, E.w->buf); e.buf(1, E.s->buf); e.buf(2, E.b->buf); e.buf(3, s.tok); e.buf(4, x_);
    e.val(5, ep);
    v8_disp(e, uint32_t((D / 8 + 63) / 64), 1, 1, kV8wYrows, 64);
  } else {
    enc_embed(e, s.tok, pos, R, x_);
  }
  // DFlash2 tree drafting: the rows are a tree (v8_tree_) and / or the pending rows a tree cycle's accepted path
  // (s.pend_tree): the *_tree kernels (the KFlash tree descriptor, the path in s.pend_path)
  const bool tree = v8_tree_ != nullptr, ptree = s.pend_tree && s.n_pend > 0;
  if ((tree || ptree) && !s.pend_path) fatal("encode_verify8: a tree verify without the sequence's path buffer");
  const Buffer& tree_buf = tree ? *v8_tree_ : s.pend_path;
  // NI_V8_TREE_PIN=0: the GDN step's update left to the compiler's contraction (gdn_update)
  static const bool tree_pin = env_int("NI_V8_TREE_PIN", 1) != 0;
  kp::GdnVerify gv{uint32_t(R), uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps,
                   tree ? 1u : 0u, ptree ? 1u : 0u, (tree || ptree) && tree_pin ? 1u : 0u};
  bool table_ready = false;  // the previous layer's MoE combine wrote this layer's input-norm table
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[i];
    const int skip = v8_knobs().skip;
    if (L.linear) {
      const bool ti = table_ready ? v8_tw() : enc_x8(e, x_, R, D, L.in_norm, false, 0, nullptr, true);
      if (!(skip & 32))
        enc_qmm8(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, R, v8t_, v8s_, false, ti ? &v8tt_ : nullptr);
      if (!(skip & 4)) {
        // fuse & 4: gdn_prep_verify8 (the rows' q / k norms reduced together; bitwise gdn_prep_verify); a wide tree
        // verify: gdn_prep_verify8w_tree (its live rows, windows from the ancestors)
        const bool fused = R <= 8 && cfg_.lin_dk == 128 && (v8_knobs().fuse & 4);
        if (wide && cfg_.lin_dk != 128) fatal("wide tree verify: GDN key heads of 128");
        e.use(pipe8(wide ? std::string("gdn_prep_verify8w_tree")
                         : std::string(fused ? "gdn_prep_verify8" : "gdn_prep_verify") + (tree || ptree ? "_tree" : "")));
        e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
        e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
        e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
        e.buf(12, s.pend_qkv[i]);
        e.val(13, gv);
        if (tree || ptree) {
          e.buf(14, tree_buf);
          e.buf(15, s.pend_path);
        }
        e.dispatch(C / 128, 1, 1, 128);
      }
      static const bool gdn8 = env_int("NI_V8_GDN", 1) != 0;  // gdn_step_verify8: one norm pass
      // V8Knobs::gdnsplit (default): the step over value-row blocks (gdn_step_verify8s, split_thr threads each),
      // then gdn_gnorm_x8 writes the out_proj's table (bitwise what gdn_step_verify8 + x8_prep2 give)
      const int split_thr = v8_knobs().gdnsplit;
      const bool split = gdn8 && cfg_.lin_dv == 128 && cfg_.lin_dk == 128 && split_thr >= 32 && split_thr <= 1024 &&
                         split_thr % 32 == 0 && 128 % (split_thr / 8) == 0;
      const bool tgn = v8_tw() && size_t(VD) * 16 <= v8tt_.size;
      if ((tree || ptree) && !split) fatal("tree verify: the GDN step must run split (V8Knobs::gdnsplit)");
      if (!(skip & 1) && split) {
        // value-row blocks of split_thr / 32 simdgroups x 4 rows, then the gated norm into the table
        e.use(pipe8(tree || ptree ? "gdn_step_verify8s_tree" : "gdn_step_verify8s"));
        e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
        e.buf(5, s.rec[i]); e.buf(6, gout_);
        e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
        e.val(12, gv);
        e.buf(13, v8cnt_);
        if (tree || ptree) {
          e.buf(14, tree_buf);
          e.buf(15, s.pend_path);
        }
        e.dispatch(cfg_.lin_hv, 128 / (split_thr / 8), 1, uint32_t(split_thr));
        // gdn_gnorm_x8t: also the T table of qmm8t
        e.use(pipe8(tgn ? "gdn_gnorm_x8t" : "gdn_gnorm_x8"));
        e.buf(0, gout_); e.buf(1, zb_); e.buf(2, L.gnorm->buf); e.buf(3, v8t_); e.buf(4, v8s_);
        e.val(5, gv);
        if (tgn) e.buf(6, v8tt_);
        if (wide) v8_disp(e, uint32_t(cfg_.lin_hv), 1, 1, kV8wY, 256);  // the live row tiles
        else e.dispatch(cfg_.lin_hv, 1, 1, 256);
      } else if (!(skip & 1)) {
        e.use(pipe8(gdn8 ? "gdn_step_verify8" : "gdn_step_verify"));
        e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
        e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
        e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
        e.val(12, gv);
        e.dispatch(cfg_.lin_hv, 1, 1, 1024);
      }
      bool to = tgn && split && !(skip & 1);
      if (!split) to = enc_x8(e, gout_, R, VD, nullptr, false, 0, nullptr, true);
      if (!(skip & 64)) enc_qmm8(e, {{&L.out, x_, uint32_t(D), true}}, R, v8t_, v8s_, false, to ? &v8tt_ : nullptr);
    } else {
      enc_attn8(e, s, L, s.kc[i], s.vc[i], s.cap, pos, R, table_ready);
    }
    table_ready = false;
    if (cfg_.moe)
      table_ready = enc_moe8(e, L, R, i, i + 1 < cfg_.layers ? layers_[i + 1].in_norm : final_norm_);
    else enc_mlp8(e, L, R);
    if (cap_) enc_capture(e, i, pos, R);  // block-drafter capture (DFlash2), off unless armed (reads x_ only)
  }
  s.n_pend = 0;  // folded by this verify; commit_rows() sets the new count
  s.pend_tree = false;
  const bool tl = table_ready ? v8_tw() : enc_x8(e, x_, R, D, final_norm_, false, 0, nullptr, true);
  if (v8_knobs().skip & 128) {
    if (wide) v8_wide_end();
    return;
  }
  enc_qmm8(e, {{&lm_head_, logits_}}, R, v8t_, v8s_, false, tl ? &v8tt_ : nullptr);
  const uint32_t V = uint32_t(cfg_.vocab), chunk = 2048, ntg = (V + chunk - 1) / chunk;
  const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  // structured output: each row's mask (a tree's row r: node r's, after its path), nothing without masks
  enc_logit_mask(e, s, logits_, R, int(V), int(V));
  kp::TopkStage ts{V, K, chunk};
  // a tree: its live rows alone (tree[kKfN])
  e.use(pipe8(tree ? "topk_stage1_rows_tree" : "topk_stage1_rows"));
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  if (tree) e.buf(4, *v8_tree_);
  if (wide) v8_disp(e, ntg, 1, 1, kV8wYrows, 256);
  else e.dispatch(ntg, R, 1, 256);
  kp::SampleRows sr{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, 0, 0, 0};
  // a tree's row r draws with the uniform of its token's position, step + depth(r)
  e.use(pipe8(tree ? "sample_stage2_rows_tree" : "sample_stage2_rows"));
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, samp_);
  e.val(3, sr);
  e.buf(4, cand_i_);  // (no remap table)
  if (tree) e.buf(5, *v8_tree_);
  if (wide) v8_disp(e, 1, 1, 1, kV8wXrows, 1024);
  else e.dispatch(R, 1, 1, 1024);
  if (wide) v8_wide_end();
}

bool Qwen35::verify_tree_ok() const { return v8_tree_ok_c(cfg_); }

void Qwen35::encode_kv_tree_compact(Encoder& e, Seq& s, int pos, const Buffer& tree) {
  const kp::KvTree p{uint32_t(pos), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(s.cap)};
  for (int i = 0; i < cfg_.layers; ++i) {
    if (cfg_.is_linear[i]) continue;
    e.use(pipe8("kv_tree_compact"));
    e.buf(0, s.kc[i]); e.buf(1, s.vc[i]); e.buf(2, tree);
    e.val(3, p);
    e.dispatch(1, 2 * cfg_.hkv, 1, cfg_.head_dim);
  }
}

// ---- batched decode: one 8-row verify over the rows of several sequences ----
// Every row-independent part (norms into the X8 tables, projections, MLP / MoE, LM head, the top-k stage) runs once
// over all the rows; the parts that read or write a sequence's own state run per group on views of the shared row
// buffers: the GDN conv window, recurrence and tape (gdn_prep_verify, gdn_step_verify8s), the attention's KV cache
// write and its attention over the cache (attn_prep, attn8m_partial, attn_rows_reduce), the sampler's second stage
// (its parameters and RNG positions). Each group folds its sequence's pending rows first, as encode_verify8 does.
void Qwen35::encode_verify_multi(Encoder& e, const std::vector<VerifyGroup>& groups, int split_after) {
  int R = 0;
  for (const VerifyGroup& gr : groups) {
    if (!gr.s || gr.R < 1) fatal("encode_verify_multi: an empty group");
    if (gr.pos + gr.R + 1 >= gr.s->cap) fatal("context full (%d)", gr.s->cap);
    R += gr.R;
  }
  if (groups.empty() || R > 8) fatal("encode_verify_multi: %d rows", R);
  if (v8_rlim_) fatal("encode_verify_multi: a verify row limit is set");
  if (v8_tree_) fatal("encode_verify_multi: a verify tree is set");
  v8_alloc();
  // a sequence whose pending rows are a tree cycle's accepted path (it drafted alone with DFlash2 trees, then joined
  // the batch) folds them on its own first
  for (const VerifyGroup& gr : groups)
    if (gr.s->pend_tree) encode_flush(e, *gr.s);
  const int D = cfg_.hidden, HD = cfg_.hq * cfg_.head_dim, VD = cfg_.lin_hv * cfg_.lin_dv;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const size_t qk = size_t(cfg_.lin_hk) * cfg_.lin_dk, hv = size_t(cfg_.lin_hv), kvw = size_t(cfg_.hkv) * cfg_.head_dim;
  std::vector<size_t> r0(groups.size());
  for (size_t g = 0, r = 0; g < groups.size(); ++g) {
    r0[g] = r;
    r += size_t(groups[g].R);
  }
  // rows [r0, r0 + R) of a row-major buffer of `w` floats per row
  auto rows = [&](const Buffer& b, size_t g, size_t w) {
    return b.view(r0[g] * w * 4, size_t(groups[g].R) * w * 4);
  };
  // (with a grouped encoder the per-sequence dispatches of one stage run side by side: they read and write
  // disjoint rows of the shared buffers, their own sequence's state, their own counters and attention partials)
  const size_t cnt_stride = (size_t(std::max(cfg_.lin_hv, 1)) * 4 + 16);
  e.group_begin();
  for (size_t g = 0; g < groups.size(); ++g)
    enc_embed(e, groups[g].s->tok, groups[g].pos, groups[g].R, rows(x_, g, size_t(D)));
  e.group_end();
  static const bool gdn8 = env_int("NI_V8_GDN", 1) != 0;
  const int split_thr = v8_knobs().gdnsplit;
  const bool split = gdn8 && cfg_.lin_dv == 128 && cfg_.lin_dk == 128 && split_thr >= 32 && split_thr <= 1024 &&
                     split_thr % 32 == 0 && 128 % (split_thr / 8) == 0;
  static const int attn_mma = env_int("NI_V8_ATTN", 1);
  static const int tg_target = env_int("NI_V8_ATTN_TGS", 160);
  const int G = cfg_.hq / cfg_.hkv;
  const bool mma = attn_mma && G <= 16 && cfg_.head_dim == 256;
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[i];
    if (L.linear) {
      const bool ti = enc_x8(e, x_, R, D, L.in_norm, false, 0, nullptr, true);
      enc_qmm8(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, R, v8t_, v8s_, false, ti ? &v8tt_ : nullptr);
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        Seq& s = *groups[g].s;
        const kp::GdnVerify gv{uint32_t(groups[g].R), uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv),
                               C, cfg_.eps};
        e.use(pipe8("gdn_prep_verify"));
        e.buf(0, rows(qkv_, g, C)); e.buf(1, rows(ab_, g, hv)); e.buf(2, rows(bb_, g, hv)); e.buf(3, s.conv[i]);
        e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
        e.buf(7, rows(gq_, g, qk)); e.buf(8, rows(gk_, g, qk)); e.buf(9, rows(gv_, g, size_t(VD)));
        e.buf(10, rows(gg_, g, hv)); e.buf(11, rows(gbeta_, g, hv));
        e.buf(12, s.pend_qkv[i]);
        e.val(13, gv);
        e.dispatch(C / 128, 1, 1, 128);
      }
      e.group_end();
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        Seq& s = *groups[g].s;
        const kp::GdnVerify gv{uint32_t(groups[g].R), uint32_t(s.n_pend), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv),
                               C, cfg_.eps};
        if (split) {
          e.use(pipe8("gdn_step_verify8s"));
          e.buf(0, rows(gq_, g, qk)); e.buf(1, rows(gk_, g, qk)); e.buf(2, rows(gv_, g, size_t(VD)));
          e.buf(3, rows(gg_, g, hv)); e.buf(4, rows(gbeta_, g, hv));
          e.buf(5, s.rec[i]); e.buf(6, rows(gout_, g, size_t(VD)));
          e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
          e.val(12, gv);
          e.buf(13, v8cnt_.view(g * cnt_stride, cnt_stride));
          e.dispatch(cfg_.lin_hv, 128 / (split_thr / 8), 1, uint32_t(split_thr));
        } else {
          e.use(pipe8(gdn8 ? "gdn_step_verify8" : "gdn_step_verify"));
          e.buf(0, rows(gq_, g, qk)); e.buf(1, rows(gk_, g, qk)); e.buf(2, rows(gv_, g, size_t(VD)));
          e.buf(3, rows(gg_, g, hv)); e.buf(4, rows(gbeta_, g, hv));
          e.buf(5, s.rec[i]); e.buf(6, rows(zb_, g, size_t(VD))); e.buf(7, L.gnorm->buf);
          e.buf(8, rows(gout_, g, size_t(VD)));
          e.buf(9, s.pend_k[i]); e.buf(10, s.pend_g[i]); e.buf(11, s.pend_d[i]);
          e.val(12, gv);
          e.dispatch(cfg_.lin_hv, 1, 1, 1024);
        }
      }
      e.group_end();
      bool to = false;
      if (split) {  // the gated norm of all rows into the out_proj's table (gdn_gnorm_x8t: and the T table)
        const kp::GdnVerify ga{uint32_t(R), 0u, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C, cfg_.eps};
        to = v8_tw() && size_t(VD) * 16 <= v8tt_.size;
        e.use(pipe8(to ? "gdn_gnorm_x8t" : "gdn_gnorm_x8"));
        e.buf(0, gout_); e.buf(1, zb_); e.buf(2, L.gnorm->buf); e.buf(3, v8t_); e.buf(4, v8s_);
        e.val(5, ga);
        if (to) e.buf(6, v8tt_);
        e.dispatch(cfg_.lin_hv, 1, 1, 256);
      } else {
        to = enc_x8(e, gout_, R, VD, nullptr, false, 0, nullptr, true);
      }
      enc_qmm8(e, {{&L.out, x_, uint32_t(D), true}}, R, v8t_, v8s_, false, to ? &v8tt_ : nullptr);
    } else {
      const bool tq = enc_x8(e, x_, R, D, L.in_norm, false, 0, nullptr, true);
      enc_qmm8(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, R, v8t_, v8s_, false, tq ? &v8tt_ : nullptr);
      // each sequence's history chunks as enc_attn8 cuts them, its partials in a region of their own (the
      // sequences' chunks x rows share the buffer's 1024)
      std::vector<kp::AttnRows> rp(groups.size());
      std::vector<size_t> pbase(groups.size());
      const uint32_t share = std::max<uint32_t>(1, 1024 / uint32_t(groups.size()));
      for (size_t g = 0, base = 0; g < groups.size(); ++g) {
        const int pos = groups[g].pos, Rg = groups[g].R;
        const uint32_t Lmax = uint32_t(pos + Rg), gran = mma ? 32 : 16;
        const uint32_t want = std::max<uint32_t>(1, (uint32_t(tg_target) + cfg_.hkv - 1) / cfg_.hkv);
        uint32_t chunk = std::max<uint32_t>(64, (Lmax + want - 1) / want);
        chunk = (chunk + gran - 1) / gran * gran;
        while ((Lmax + chunk - 1) / chunk * uint32_t(Rg) > share) chunk += gran;
        const uint32_t nch = (Lmax + chunk - 1) / chunk;
        rp[g] = kp::AttnRows{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos), chunk, uint32_t(groups[g].s->cap),
                             1.0f / std::sqrt(float(cfg_.head_dim)), nch};
        pbase[g] = base;
        base += size_t(Rg) * size_t(cfg_.hq) * nch;  // partial rows (r * Hq + h) * nchunks + c
      }
      auto part = [&](const Buffer& b, size_t g, size_t w) {
        const size_t n = size_t(groups[g].R) * size_t(cfg_.hq) * rp[g].nchunks;
        return b.view(pbase[g] * w * 4, n * w * 4);
      };
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        const Seq& s = *groups[g].s;
        const kp::AttnPrep ap{uint32_t(groups[g].R), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                              uint32_t(cfg_.rot_dims), uint32_t(groups[g].pos), uint32_t(s.cap), cfg_.eps, cfg_.theta};
        e.use(pipe8(mrope(s) ? "attn_prep_m" : "attn_prep"));
        e.buf(0, rows(qg_, g, 2 * size_t(HD))); e.buf(1, rows(kr_, g, kvw)); e.buf(2, rows(vr_, g, kvw));
        e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
        e.buf(5, rows(qo_, g, size_t(HD))); e.buf(6, s.kc[i]); e.buf(7, s.vc[i]);
        e.val(8, ap);
        if (mrope(s)) enc_mrope_args(e, s);
        e.dispatch(cfg_.hq + 2 * cfg_.hkv, groups[g].R, 1, cfg_.head_dim);
      }
      e.group_end();
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        const Seq& s = *groups[g].s;
        const int Rg = groups[g].R;
        const uint32_t Ru = uint32_t(Rg);
        if (mma) {
          e.use(pipe8("attn8m_partial"));
          e.buf(0, rows(qo_, g, size_t(HD))); e.buf(1, s.kc[i]); e.buf(2, s.vc[i]);
          e.buf(3, part(part_o_, g, size_t(cfg_.head_dim))); e.buf(4, part(part_ml_, g, 2));
          e.val(5, rp[g]);
          e.val(6, Ru);
          e.dispatch(rp[g].nchunks, cfg_.hkv, 1, 64 * G);
        } else {
          static const int rb_env = env_int("NI_V8_ATTN_RB", 0);
          const int RB = std::max(1, std::min(rb_env ? rb_env : 32 / G, Rg));
          e.use(pipe8("attn8_partial"));
          e.buf(0, rows(qo_, g, size_t(HD))); e.buf(1, s.kc[i]); e.buf(2, s.vc[i]);
          e.buf(3, part(part_o_, g, size_t(cfg_.head_dim))); e.buf(4, part(part_ml_, g, 2));
          e.val(5, rp[g]);
          e.val(6, Ru);
          e.dispatch(rp[g].nchunks, cfg_.hkv, (Rg + RB - 1) / RB, 32 * G * RB);
        }
      }
      e.group_end();
      e.group_begin();
      for (size_t g = 0; g < groups.size(); ++g) {
        e.use(pipe8("attn_rows_reduce"));
        e.buf(0, part(part_o_, g, size_t(cfg_.head_dim))); e.buf(1, part(part_ml_, g, 2));
        e.buf(2, rows(qg_, g, 2 * size_t(HD))); e.buf(3, rows(attn_out_, g, size_t(HD)));
        e.val(4, rp[g]);
        e.dispatch(cfg_.hq, groups[g].R, 1, cfg_.head_dim);
      }
      e.group_end();
      const bool to = enc_x8(e, attn_out_, R, HD, nullptr, false, 0, nullptr, true);
      enc_qmm8(e, {{&L.o, x_, uint32_t(D), true}}, R, v8t_, v8s_, false, to ? &v8tt_ : nullptr);
    }
    if (cfg_.moe) enc_moe8(e, L, R, i);
    else enc_mlp8(e, L, R);
    e.group_begin();
    for (size_t g = 0; g < groups.size(); ++g) {  // the drafters' captures, by row
      const CaptureHook* cap = groups[g].cap;
      if (!cap || i >= int(cap->slot.size()) || cap->slot[size_t(i)] < 0) continue;
      const kp::DflCapture cp{uint32_t(D), cap->ld, uint32_t(cap->slot[size_t(i)]) * uint32_t(D),
                              uint32_t(groups[g].cap_row0)};
      e.use(pipe8("dfl_capture"));
      e.buf(0, rows(x_, g, size_t(D)));
      e.buf(1, cap->dst);
      e.val(2, cp);
      e.dispatch((uint32_t(D) / 4 + 255) / 256, uint32_t(groups[g].R), 1, 256);
    }
    e.group_end();
    if (i + 1 == split_after) e.split();  // (the GPU starts on the first layers)
  }
  for (const VerifyGroup& gr : groups) gr.s->n_pend = 0;  // folded by this pass; commit_rows() sets the new count
  const bool tl = enc_x8(e, x_, R, D, final_norm_, false, 0, nullptr, true);
  enc_qmm8(e, {{&lm_head_, logits_}}, R, v8t_, v8s_, false, tl ? &v8tt_ : nullptr);
  // sampling: the top-k stage once over every row (the largest candidate count any group needs), the second stage
  // per group with its parameters
  uint32_t Kmax = 1;
  for (const VerifyGroup& gr : groups)
    if (gr.sp.temperature > 0) Kmax = std::max(Kmax, uint32_t(std::clamp(gr.sp.top_k <= 0 ? 32 : gr.sp.top_k, 1, 32)));
  const uint32_t V = uint32_t(cfg_.vocab), chunk = 2048, ntg = (V + chunk - 1) / chunk;
  // structured output: the masked members' rows, after one GPU wait for the host's masks (it signals every member's at
  // once: the latest value covers them all); nothing without masks
  {
    void* ev = nullptr;
    uint64_t val = 0;
    for (const VerifyGroup& gr : groups)
      if (gr.s->mask_rows > 0 && gr.s->mask_event) ev = gr.s->mask_event, val = std::max(val, gr.s->mask_value);
    if (ev) e.wait_event(ev, val);
    for (size_t g = 0; g < groups.size(); ++g) {
      Seq& sq = *groups[g].s;
      if (sq.mask_rows <= 0) continue;
      void* const keep = sq.mask_event;
      sq.mask_event = nullptr;  // (waited for above)
      enc_logit_mask(e, sq, rows(logits_, g, V), groups[g].R, int(V), int(V));
      sq.mask_event = keep;
    }
  }
  const kp::TopkStage ts{V, Kmax, chunk};
  e.use(pipe8("topk_stage1_rows"));
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, R, 1, 256);
  e.group_begin();
  for (size_t g = 0; g < groups.size(); ++g) {
    const SamplingParams& sp = groups[g].sp;
    const uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
    const bool own = groups[g].tok_out;
    const kp::SampleRows sr{ntg * Kmax, K, sp.temperature, sp.top_p, sp.seed, uint32_t(groups[g].pos),
                            own ? uint32_t(groups[g].pos + 1) : uint32_t(r0[g]), 0, 0};
    e.use(pipe8("sample_stage2_rows"));
    e.buf(0, rows(cand_v_, g, size_t(ntg) * Kmax)); e.buf(1, rows(cand_i_, g, size_t(ntg) * Kmax));
    e.buf(2, own ? groups[g].s->tok : samp_);
    e.val(3, sr);
    e.buf(4, cand_i_);  // (no remap table)
    e.dispatch(uint32_t(groups[g].R), 1, 1, 1024);
  }
  e.group_end();
}

} // namespace ni
