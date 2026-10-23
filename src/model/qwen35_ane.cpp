// GPU + Neural Engine split of the dense prefill FFN (NI_ANE). See src/ane/ffn_split.h.
//
//   NI_ANE=1            enable (dense models; the MoE path is unchanged)
//   NI_ANE_UNITS=<u>    ANE share in 512-channel units (default: calibration, else per-chip default)
//   NI_ANE_MIN_ROWS=<r> smallest prefill chunk that splits (default: calibration, else 512)
//   NI_ANE_FNS=a,b,..   ANE program functions (rows); default every 128 rows from min_rows to 2048
//   NI_ANE_VERIFY=0     skip the startup check (on by default)
//   NI_ANE_FAULT=<l>    test: the ANE job of layer l of the first split chunk never starts
//   NI_ANE_WAVE=0       the plain split (default: the two-half wavefront, for chunks >= 2 x the
//                       smallest function; smaller chunks always run the plain split)
#include "ane/ffn_split.h"
#include "model/kparams.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

namespace ni {
namespace {

Pipeline kpipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

std::vector<uint32_t> parse_rows(const char* v) {
  std::vector<uint32_t> r;
  if (!v) return r;
  std::string s = v;
  for (size_t p = 0; p < s.size();) {
    size_t q = s.find(',', p);
    r.push_back(uint32_t(std::stoul(s.substr(p, q - p))));
    p = q == std::string::npos ? s.size() : q + 1;
  }
  return r;
}

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace

bool Qwen35::ane_splits(int T) const { return ane_ && ane_on_ && !cfg_.moe && ane_->splits(T); }

void Qwen35::ane_begin_chunk(int T) {
  ane_->begin_chunk(T);
  ++ane_stats_.chunks;
}

bool Qwen35::ane_finish_chunk() {
  if (!ane_chunk_) return true;
  ane_chunk_ = false;
  double ms = 0;
  const bool ok = ane_->finish_chunk(&ms);
  if (ok) {
    ane_stats_.ane_ms += ms;
    ane_stats_.evals += ane_->last_jobs();
  } else {
    ++ane_stats_.reruns;
    logi("NI_ANE: Neural Engine FFN split stopped (%s); this chunk reruns on the GPU, which runs the prefill FFN "
         "alone from now on", ane_->reason().c_str());
  }
  return ok;
}

void Qwen35::ane_abort(const std::string& why) {
  if (ane_chunk_ && ane_) ane_->retire(why);
}

void Qwen35::init_ane() {
  if (cfg_.moe) {
    logi("NI_ANE: the Neural Engine split is for dense FFNs; this MoE model runs its prefill on the GPU");
    return;
  }
  if (Encoder::profiling()) {
    logi("NI_ANE: off under NI_PROFILE");
    return;
  }
  // the split stages its int8 weights from any format (ane_ffn.metal, wfmt.metal); the GPU's share of a layer runs
  // gate and up as one fused GEMM, so the two need one format
  for (const Layer& L : layers_) {
    for (const QLinear* q : {&L.mg, &L.mu, &L.md})
      if (!format_supported(q->bits, q->gs)) {
        logi("NI_ANE: no split for %s FFN weights; GPU only", format_name(q->bits, q->gs).c_str());
        return;
      }
    if (!L.mg.same_format(L.mu)) {
      logi("NI_ANE: FFN gate and up in different formats (%s, %s); GPU only", format_name(L.mg.bits, L.mg.gs).c_str(),
           format_name(L.mu.bits, L.mu.gs).c_str());
      return;
    }
  }
  const std::string chip = Gpu::get().name();
  const std::string mkey = ane::model_key(dir_);
  // the two-half wavefront is the default schedule (NI_ANE_WAVE=0: the plain split)
  const bool wave = !getenv("NI_ANE_WAVE") || env_on("NI_ANE_WAVE");
  ane::Calibration cal;
  const bool have_cal = ane::read_calibration(ane::machine_key(chip, mkey) + (wave ? "|wave" : ""), cal);
  // the GPU's share on the matrix units (enc_mlp_gpu_part_qmt) for 4-bit group 64 FFN weights: the M5 default share on
  // the models it was measured on (mma_share_measured)
  bool mma = qmt_ready() && mma_share_measured(cfg_);
  auto g64 = [](const QLinear& q) { return q.bits == 4 && q.gs == 64; };
  for (const Layer& L : layers_) mma = mma && g64(L.mg) && g64(L.mu) && g64(L.md);  // as plan()'s ffn_mma
  const uint32_t total = uint32_t(cfg_.dense_I / 512);
  uint32_t units = have_cal ? cal.units : ane::default_units(chip, Gpu::get().core_count(), total, wave, mma);
  uint32_t min_rows = have_cal && cal.min_rows ? cal.min_rows : 512;
  if (getenv("NI_ANE_UNITS")) units = uint32_t(atoi(getenv("NI_ANE_UNITS")));
  if (getenv("NI_ANE_MIN_ROWS")) min_rows = uint32_t(atoi(getenv("NI_ANE_MIN_ROWS")));
  if (!units) {
    logi("NI_ANE: no calibration for %s (run ane_calib) and no default share; GPU only", chip.c_str());
    return;
  }
  if (units * 512 > uint32_t(cfg_.dense_I)) {
    logi("NI_ANE: share of %u units exceeds the FFN's %d; GPU only", units, cfg_.dense_I / 512);
    return;
  }
  if (int(std::clamp<uint32_t>(min_rows, 1, ane::FfnSplit::kMaxRows)) > tmax_) {
    logi("NI_ANE: prefill chunks of %d rows never reach the split's %u; GPU only", tmax_, min_rows);
    return;
  }
  const int fault = getenv("NI_ANE_FAULT") ? atoi(getenv("NI_ANE_FAULT")) : -1;
  const int lanes = wave ? 2 : 1;
  ane_init_ = {units, min_rows, parse_rows(getenv("NI_ANE_FNS")), fault, lanes,
               getenv("NI_ANE_UNITS") ? "NI_ANE_UNITS" : have_cal ? "calibration" : "default for this chip"};
  if (env_on("NI_ANE_ASYNC")) {
    // The server path: the ANE daemon compiles the program in the background (~40 s when it has
    // evicted it, ~0.1 s when it still holds it); prefill runs on the GPU alone until ane_poll()
    // sees the compilation done and builds the split from the daemon's cache.
    ane::FfnConfig c = ane_cfg(units, min_rows, ane_init_.fns, fault, lanes);
    c.compile_only = true;
    auto st = std::make_shared<std::atomic<int>>(1);
    ane_async_st_ = st;
    ane_thread_ = std::thread([st, c, ls = ane_layers()]() mutable {
      const double t0 = now_ms();
      try {
        ane::FfnSplit warm(c, std::move(ls));
        logi("NI_ANE: Neural Engine program ready in the background (%s, %.1f s)", warm.startup_note().c_str(),
             (now_ms() - t0) / 1000);
        st->store(2);
      } catch (const std::exception& e) {
        logi("NI_ANE: Neural Engine split unavailable (%s); the GPU runs the prefill FFN alone", e.what());
        st->store(3);
      }
    });
    logi("NI_ANE: compiling the Neural Engine program in the background; prefill uses the GPU alone until it is ready");
    return;
  }
  ane_finish_init();
}

Qwen35::AnePlan Qwen35::ane_plan(const std::string& dir, const MemPlan& m) {
  AnePlan a;
  a.total = m.dense ? uint32_t(m.ffn / 512) : 0;
  if (!m.dense) return a.source = "MoE: GPU", a;
  if (!m.ffn_ane) return a.source = "FFN weight format not staged by the split; GPU", a;
  if (getenv("NI_ANE") && !strcmp(getenv("NI_ANE"), "0")) return a.source = "NI_ANE=0", a;
  if (!ane::FfnSplit::rotate_simdgroups(uint32_t(m.hidden))) return a.source = "hidden size not supported", a;
  if (const std::string& why = ane::unavailable(); !why.empty()) return a.source = "Neural Engine unavailable", a;
  const std::string chip = Gpu::get().name();
  const bool wave = !getenv("NI_ANE_WAVE") || env_on("NI_ANE_WAVE");
  ane::Calibration cal;
  if (ane::read_calibration(ane::machine_key(chip, ane::model_key(dir)) + (wave ? "|wave" : ""), cal))
    a.units = cal.units, a.source = "calibration";
  else
    a.units = ane::default_units(chip, Gpu::get().core_count(), a.total, wave, m.ffn_mma),
    a.source = "default for this chip";
  if (getenv("NI_ANE_UNITS")) a.units = uint32_t(atoi(getenv("NI_ANE_UNITS"))), a.source = "NI_ANE_UNITS";
  if (!a.units || a.units > a.total) {
    a.units = 0;
    a.source = "no share for this chip (ane_calib --wave --write)";
    return a;
  }
  a.bytes = ane::split_bytes(uint32_t(m.hidden), uint32_t(m.ffn), a.units, m.layers, wave ? 2 : 1) + m.slot_state / 3;
  return a;
}

bool Qwen35::ane_release() {
  if (ane_async_st_) {  // a compile still running in the background: never build the split
    if (ane_thread_.joinable()) ane_thread_.detach();
    ane_async_st_.reset();
  }
  if (!ane_) return false;
  logi("NI_ANE: releasing the Neural Engine split (%.0f MB) for KV cache; prefill runs on the GPU alone", ane_->metal_bytes() / 1e6);
  ane_.reset();
  return true;
}

void Qwen35::ane_poll() {
  if (!ane_async_st_) return;
  const int st = ane_async_st_->load();
  if (st == 1) return;  // still compiling
  if (ane_thread_.joinable()) ane_thread_.join();
  ane_async_st_.reset();
  if (st == 2) ane_finish_init();
}

void Qwen35::ane_finish_init() {
  const uint32_t units = ane_init_.units, min_rows = ane_init_.min_rows;
  const int fault = ane_init_.fault, lanes = ane_init_.lanes;
  if (!ane_rebuild(units, min_rows, ane_init_.fns, fault, lanes)) return;
  const char* source = ane_init_.source;
  logi("NI_ANE: Neural Engine FFN split%s: %u/%d units (%u channels; GPU %u) from the %s, chunks >= %u rows; %s; "
       "%.1f MB Metal; ready in %.1f s",
       lanes > 1 ? " (wavefront)" : "", units, cfg_.dense_I / 512, ane_->ane_channels(), ane_->gpu_channels(), source,
       ane_->min_rows(),
       ane_->startup_note().c_str(), ane_->metal_bytes() / 1e6, ane_->startup_ms() / 1000);
  if (fault >= 0) logi("NI_ANE_FAULT: the ANE job of layer %d of the first split chunk will not start", fault);
  if (tmax_ > int(ane::FfnSplit::kMaxRows))
    logi("NI_ANE: prefill chunks of more than %u rows run on the GPU alone (prefill chunk is %d)", ane::FfnSplit::kMaxRows,
         tmax_);
  if (env_on("NI_ANE_VERIFY") || !getenv("NI_ANE_VERIFY")) {
    const double t0 = now_ms();
    // the check runs whole chunks through the model's activation buffers: at most tmax_ rows
    std::vector<int> rows = {int(ane_->fn_rows().front()), std::min(int(ane::FfnSplit::kMaxRows), tmax_)};
    if (env_on("NI_ANE_VERIFY_ALL")) rows.assign(ane_->fn_rows().begin(), ane_->fn_rows().end());
    double err = -1;
    try {
      err = ane_verify(2, rows);
    } catch (const std::exception& e) {
      logi("NI_ANE: startup check failed (%s); GPU only", e.what());
      ane_.reset();
      return;
    }
    if (!(err <= 0.10)) {
      logi("NI_ANE: startup check: the split differs from the GPU alone by %.1f%% of the ANE's part (> 10%%); GPU only",
           100 * err);
      ane_.reset();
      return;
    }
    logi("NI_ANE: startup check %.2f%% of the ANE's part from the GPU alone (%zu functions, %.0f ms)", 100 * err,
         rows.size(), now_ms() - t0);
  }
}

ane::FfnConfig Qwen35::ane_cfg(uint32_t units, uint32_t min_rows, const std::vector<uint32_t>& fn_rows, int fault_layer,
                               int lanes) const {
  ane::FfnConfig c;
  c.lanes = lanes;
  c.units = units;
  c.min_rows = min_rows;
  c.fn_rows = fn_rows;
  c.fault_layer = fault_layer;
  c.cache_dir = ane::cache_dir();
  c.model_key = ane::model_key(dir_);
  // the test transforms (NI_REQUANT / NI_DEQUANT) change the weights the row scales come from: their own cache file
  for (const char* v : {"NI_REQUANT", "NI_DEQUANT"})
    if (const char* t = getenv(v); t && *t) {
      c.model_key += std::string("-") + (v[3] == 'R' ? "rq" : "dq");
      for (const char* q = t; *q; ++q) c.model_key += isalnum((unsigned char)*q) ? *q : '_';
    }
  return c;
}

std::vector<ane::FfnLayer> Qwen35::ane_layers() const {
  std::vector<ane::FfnLayer> ls;
  for (const Layer& L : layers_)
    ls.push_back({{L.mg.w->buf, L.mg.s->buf, L.mg.b->buf, L.mg.N, L.mg.K, L.mg.bits, L.mg.gs},
                  {L.mu.w->buf, L.mu.s->buf, L.mu.b->buf, L.mu.N, L.mu.K, L.mu.bits, L.mu.gs},
                  {L.md.w->buf, L.md.s->buf, L.md.b->buf, L.md.N, L.md.K, L.md.bits, L.md.gs}});
  return ls;
}

bool Qwen35::ane_rebuild(uint32_t units, uint32_t min_rows, const std::vector<uint32_t>& fn_rows, int fault_layer,
                         int lanes) {
  ane_.reset();
  try {
    ane_ = std::make_unique<ane::FfnSplit>(ane_cfg(units, min_rows, fn_rows, fault_layer, lanes), ane_layers());
  } catch (const std::exception& e) {
    logi("NI_ANE: Neural Engine split unavailable (%s); the GPU runs the prefill FFN alone", e.what());
    ane_.reset();
    return false;
  }
  // the split's GPU part writes its channels' fp32 gate/up rows of a whole chunk into hg_/hu_
  ensure_hgu(size_t(std::min(int(ane::FfnSplit::kMaxRows), tmax_)) * std::max<uint32_t>(ane_->gpu_channels(), 1) * 4);
  if (mem_budget_ && Gpu::get().allocated_bytes() > mem_budget_) {
    logi("NI_ANE: the Neural Engine split's %.0f MB would exceed the memory budget (%.1f GB); GPU only",
         ane_->metal_bytes() / 1e6, mem_budget_ / 1e9);
    ane_.reset();
    return false;
  }
  return true;
}

// The GPU's channels [0, g) of the FFN of T rows of xh_: gate/up over their leading g rows,
// SwiGLU, down over the leading g inputs of its rows (row stride I), added into x (fp32).
void Qwen35::enc_mlp_gpu_part(Encoder& e, const Layer& L, int T, uint32_t g, const Buffer& x) {
  if (!g) return;  // the ANE takes every channel
  if (enc_mlp_gpu_part_qmt(e, L, T, g, x)) return;
  const uint32_t D = uint32_t(cfg_.hidden), I = L.mg.N;
  const bool q4 = L.mg.q4() && L.mu.q4() && L.md.q4();
  // gate/up of the leading g rows and SwiGLU into hhalf_ [T, g]: 4-bit as two GEMMs into fp32 hg_ / hu_; the other
  // formats fused, on the prefill path's GEMMs for the format
  bool fused = false;
  if (!q4) {
    QLinear G = L.mg, U = L.mu;
    G.N = U.N = g;
    fused = enc_qmm_gu(e, G, U, xh_, hhalf_, T);
    if (!fused && !(L.mg.q4() && L.mu.q4())) fatal("NI_ANE: no fused gate/up GEMM for %s", L.mg.fmt().c_str());
  }
  if (!fused) {
    for (int k = 0; k < 2; ++k) {
      const QLinear& Q = k ? L.mu : L.mg;
      e.use(kpipe("qmm_b4_m64_n64"));
      kp::Qmm p{uint32_t(T), g, Q.K, Q.K, g, 0, uint32_t(Q.gs), 0, 0};
      e.buf(0, xh_); e.buf(1, Q.w->buf); e.buf(2, Q.s->buf); e.buf(3, Q.b->buf); e.buf(4, k ? hu_ : hg_);
      e.val(5, p);
      e.dispatch((g + 63) / 64, (uint32_t(T) + 63) / 64, 1, 128);
    }
    const uint32_t n = uint32_t(T) * g;
    e.use(kpipe("swiglu_half"));
    e.buf(0, hg_); e.buf(1, hu_); e.buf(2, hhalf_); e.val(3, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
  // down over the leading g inputs of its rows (row stride I), added into x
  const QLinear& Dn = L.md;
  const uint32_t wld = Dn.dense() ? I / 2 : I * uint32_t(Dn.bits) / 32, sld = Dn.dense() ? 1u : I / uint32_t(Dn.gs);
  kp::QmmLd p{{uint32_t(T), D, g, g, D, 1u, uint32_t(Dn.gs), 0, 0}, wld, sld};
  static const bool mf = !getenv("NI_QRF16_MF") || atoi(getenv("NI_QRF16_MF")) != 0;  // as enc_qrf
  const bool mfast = Dn.dense() && mf;
  e.use(kpipe(Dn.q4() ? std::string("qmm_b4_m64_n64_ld")
                      : "qrf_" + Dn.fmt() + (mfast ? "_m32_n128_mf_ld" : "_m32_n128_ld")));
  e.buf(0, hhalf_); e.buf(1, Dn.w->buf); e.buf(2, Dn.s->buf); e.buf(3, Dn.b->buf); e.buf(4, x);
  e.val(5, p);
  if (Dn.q4()) e.dispatch((D + 63) / 64, (uint32_t(T) + 63) / 64, 1, 128);
  else if (mfast) e.dispatch((uint32_t(T) + 31) / 32, (D + 127) / 128, 1, 128);
  else e.dispatch((D + 127) / 128, (uint32_t(T) + 31) / 32, 1, 128);
}

// NI_ANE_WAVE=1: the chunk runs as two row halves A and B in a skewed schedule, so the ANE's
// FFN of one half overlaps the GPU's work on the other (the plain split leaves the ANE idle while
// the GPU runs each layer's mixer). Per layer i, in GPU order:
//   mixer_i(A); norm; pack(A); launch FFN_i(A) on the ANE; GPU part of FFN_i(A)
//   join FFN_{i-1}(B) (waits for the ANE); mixer_i(B); norm; pack(B); launch FFN_i(B)
//   stage layer i+1's weights; join FFN_i(A); GPU part of FFN_i(B)
// so the ANE runs FFN_i(A) under mixer_i(B) and FFN_i(B) under mixer_{i+1}(A). B's mixer follows
// A's within a layer (GDN state and causal KV flow from A to B, as between two chunks). Each half
// has its own activation surfaces (lane); both use layer i's weight set i & 1.
// A block drafter's capture (DFlash2, Qwen35::set_capture) takes each half's rows when they hold the
// layer's final output: A's after join FFN_i(A), B's after join FFN_i(B), one step of the skew later.
int Qwen35::ane_wave_rows(int T) const {
  if (!ane_ || ane_->lanes() < 2) return 0;
  const int h = (T / 2 + 127) / 128 * 128;
  if (h >= T || h < int(ane_->fn_rows().front()) || T - h < int(ane_->min_rows()) / 2) return 0;
  return h;
}

void Qwen35::enc_prefill_wave(Encoder& e, Seq& s, int pos0, int T, int h) {
  const int D = cfg_.hidden, TA = h, TB = T - h;
  const Buffer xA = x_.view(0, size_t(TA) * D * 4), xB = x_.view(size_t(TA) * D * 4, size_t(TB) * D * 4);
  const uint32_t g = ane_->gpu_channels();
  ane::FfnSplit& A = *ane_;
  for (int i = 0; i < cfg_.layers; ++i) {
    const Layer& L = layers_[i];
    enc_mixer_prefill(e, s, i, L, pos0, TA, xA);
    enc_rows_half(e, xA, L.post_norm, xh_, TA, D);
    if (i == 0) A.stage_layer(e, 0, 0);
    A.pack(e, 0, xh_, TA);
    A.launch(e, 0, i, TA);
    enc_mlp_gpu_part(e, L, TA, g, xA);
    if (i > 0) {
      A.join(e, 1, xB, TB);
      if (cap_) enc_capture(e, i - 1, pos0 + TA, TB, &xB);  // layer i-1 is complete in B's rows
    }
    enc_mixer_prefill(e, s, i, L, pos0 + TA, TB, xB);
    enc_rows_half(e, xB, L.post_norm, xh_, TB, D);
    A.pack(e, 1, xh_, TB);
    A.launch(e, 1, i, TB);
    if (i + 1 < cfg_.layers) A.stage_layer(e, i + 1, (i + 1) & 1);
    A.join(e, 0, xA, TA);
    if (cap_) enc_capture(e, i, pos0, TA, &xA);  // layer i is complete in A's rows
    enc_mlp_gpu_part(e, L, TB, g, xB);
  }
  A.join(e, 1, xB, TB);
  if (cap_) enc_capture(e, cfg_.layers - 1, pos0 + TA, TB, &xB);
}

void Qwen35::enc_mlp_prefill_split(Encoder& e, int li, const Layer& L, int T, const Buffer& x) {
  enc_rows_half(e, x, L.post_norm, xh_, T, cfg_.hidden);
  const uint32_t g = ane_->gpu_channels();
  ane_->encode_layer(e, li, xh_, x, [&](Encoder& en) { enc_mlp_gpu_part(en, L, T, g, x); });
}

// ---- startup check and benchmarks (random post-norm input rows in xh_) ----
namespace {
// Scratch buffers released however the scope ends (a failed check throws).
struct ScratchBufs {
  std::vector<Buffer> v;
  Buffer& add(size_t bytes, const char* label) { return v.emplace_back(Gpu::get().alloc(bytes, label)); }
  ~ScratchBufs() {
    for (auto& b : v) Gpu::get().release(b);
  }
};
}  // namespace

static void fill_random_half(Buffer& b, size_t n, uint32_t seed) {
  std::mt19937 r(seed);
  std::uniform_real_distribution<float> u(-2.f, 2.f);
  _Float16* p = b.as<_Float16>();
  for (size_t i = 0; i < n; ++i) p[i] = _Float16(u(r));
}

double Qwen35::ane_verify(int n_layers, const std::vector<int>& rows) {
  Gpu& g = Gpu::get();
  // Tm rows through the model's activation buffers (hg_, hu_, hhalf_ hold tmax_ rows)
  const int D = cfg_.hidden, Tm = std::min(int(ane::FfnSplit::kMaxRows), tmax_);
  n_layers = std::min(n_layers, cfg_.layers);
  fill_random_half(xh_, size_t(Tm) * D, 20261016);
  ScratchBufs bufs;
  bufs.v.reserve(5);
  Buffer out[3];
  for (auto& o : out) o = bufs.add(size_t(Tm) * D * 4, "ane verify");
  // the GPU-alone reference's fp32 gate/up rows (hg_/hu_ hold only the split's GPU channels)
  Buffer sg = bufs.add(size_t(Tm) * cfg_.dense_I * 4, "ane verify g"), su = bufs.add(size_t(Tm) * cfg_.dense_I * 4, "ane verify u");
  auto gpu_only = [&](Buffer& o) {
    memset(o.ptr(), 0, o.size);
    auto e = g.begin();
    for (int l = 0; l < n_layers; ++l) {
      const Layer& L = layers_[l];
      enc_qmm(*e, L.mg, xh_, sg, Tm, false);
      enc_qmm(*e, L.mu, xh_, su, Tm, false);
      const uint32_t n = uint32_t(Tm) * L.mg.N;
      e->use(kpipe("swiglu_half"));
      e->buf(0, sg); e->buf(1, su); e->buf(2, hhalf_); e->val(3, n);
      e->dispatch((n + 255) / 256, 1, 1, 256);
      enc_qmm(*e, L.md, hhalf_, o, Tm, true);
    }
    e->commit_and_wait();
  };
  auto split = [&](Buffer& o, int T, bool ane) {
    memset(o.ptr(), 0, o.size);
    ane_->fill_partial(ane ? 0x7e00 : 0);  // NaN where the ANE leaves the output unwritten
    ane_->begin_chunk(T);
    auto e = g.begin();
    const uint32_t gc = ane_->gpu_channels();
    for (int l = 0; l < n_layers; ++l) {
      const Layer& L = layers_[l];
      ane_->encode_layer(*e, l, xh_, o, [&](Encoder& en) { enc_mlp_gpu_part(en, L, T, gc, o); }, ane);
    }
    e->commit_and_wait();
    if (!ane_->finish_chunk()) throw std::runtime_error("split of " + std::to_string(T) + " rows failed: " + ane_->reason());
  };
  gpu_only(out[0]);
  split(out[1], Tm, false);
  double worst = 0;
  for (int T : rows) {
    if (T > Tm) continue;
    split(out[2], T, true);
    const float *a = out[0].as<float>(), *p = out[1].as<float>(), *s = out[2].as<float>();
    double diff = 0, contrib = 0;
    for (size_t i = 0; i < size_t(T) * D; ++i) {
      if (!std::isfinite(s[i])) throw std::runtime_error("split output of " + std::to_string(T) + " rows is not finite");
      diff += (double(s[i]) - a[i]) * (double(s[i]) - a[i]);
      contrib += (double(a[i]) - p[i]) * (double(a[i]) - p[i]);
    }
    const double err = std::sqrt(diff / std::max(contrib, 1e-30));
    logd("NI_ANE: check %d rows: %.3f%%", T, 100 * err);
    worst = std::max(worst, err);
  }
  return worst;
}

AneLayerTimes Qwen35::ane_bench(int l0, int n, int T, int reps, bool stage) {
  Gpu& g = Gpu::get();
  const int D = cfg_.hidden;
  if (T > tmax_) throw std::runtime_error("ane_bench: " + std::to_string(T) + " rows exceed the prefill chunk");
  fill_random_half(xh_, size_t(T) * D, 7);
  ScratchBufs bufs;
  Buffer o = bufs.add(size_t(T) * D * 4, "ane bench");
  memset(o.ptr(), 0, o.size);
  Buffer sg = bufs.add(size_t(T) * cfg_.dense_I * 4, "ane bench g"), su = bufs.add(size_t(T) * cfg_.dense_I * 4, "ane bench u");
  AneLayerTimes r;
  std::vector<double> go, gp, an, sp, ai;
  const uint32_t gc = ane_->gpu_channels();
  for (int rep = 0; rep < reps + 1; ++rep) {  // first round warms up
    {
      auto e = g.begin();
      for (int l = l0; l < l0 + n; ++l) {
        const Layer& L = layers_[l];
        enc_qmm(*e, L.mg, xh_, sg, T, false);
        enc_qmm(*e, L.mu, xh_, su, T, false);
        const uint32_t nn = uint32_t(T) * L.mg.N;
        e->use(kpipe("swiglu_half"));
        e->buf(0, sg); e->buf(1, su); e->buf(2, hhalf_); e->val(3, nn);
        e->dispatch((nn + 255) / 256, 1, 1, 256);
        enc_qmm(*e, L.md, hhalf_, o, T, true);
      }
      e->commit_and_wait();
      if (rep) go.push_back(e->gpu_ms() / n);
    }
    {
      ane_->begin_chunk(T);
      auto e = g.begin();
      for (int l = l0; l < l0 + n; ++l) {
        const Layer& L = layers_[l];
        ane_->encode_layer(*e, l, xh_, o, [&](Encoder& en) { enc_mlp_gpu_part(en, L, T, gc, o); }, false, stage);
      }
      e->commit_and_wait();
      ane_->finish_chunk();
      if (rep) gp.push_back(e->gpu_ms() / n);
    }
    {
      ane_->begin_chunk(T);
      double t = 0;
      for (int l = 0; l < n; ++l) t += ane_->eval_sync(l & 1);
      if (rep) an.push_back(t / n);
      ane_->finish_chunk();
    }
    {
      ane_->begin_chunk(T);
      auto e = g.begin();
      const double w0 = now_ms();
      for (int l = l0; l < l0 + n; ++l) {
        const Layer& L = layers_[l];
        ane_->encode_layer(*e, l, xh_, o, [&](Encoder& en) { enc_mlp_gpu_part(en, L, T, gc, o); }, true, stage);
      }
      e->commit_and_wait();
      const double wall = now_ms() - w0;
      double ams = 0;
      const bool ok = ane_->finish_chunk(&ams);
      if (!ok) throw std::runtime_error("split failed: " + ane_->reason());
      if (rep) {
        sp.push_back(std::max(wall, e->gpu_ms()) / n);
        ai.push_back(ams / n);
      }
    }
  }
  r.gpu_only = median(go);
  r.gpu_part = median(gp);
  r.ane = median(an);
  r.split = median(sp);
  r.ane_in_split = median(ai);
  return r;
}

double Qwen35::ane_bench_chunk(int T, int reps, bool split) {
  Gpu& g = Gpu::get();
  // A Seq does not release its buffers (the engine's live as long as the process): this one,
  // made per call (ane_calib times several shares), is released when the call ends.
  struct SeqGuard {
    std::unique_ptr<Seq> s;
    ~SeqGuard() {
      Gpu& gg = Gpu::get();
      gg.release(s->tok);
      gg.release(s->mtp_kc);
      gg.release(s->mtp_vc);
      for (auto* v : {&s->conv, &s->rec, &s->kc, &s->vc, &s->pend_qkv, &s->pend_k, &s->pend_g, &s->pend_d})
        for (auto& b : *v) gg.release(b);
    }
  } guard{new_seq(T + 64)};
  Seq* seq = guard.s.get();
  std::mt19937 r(11);
  for (int i = 0; i < T; ++i) seq->tokens()[i] = int(r() % 150000u);
  SamplingParams sp;
  std::vector<double> ms;
  ane_on_ = split;
  for (int rep = 0; rep < reps + 1; ++rep) {
    reset(*seq);
    auto e = g.begin();
    const double t0 = now_ms();
    encode_prefill(*e, *seq, 0, T, sp, 0, false);
    e->commit_and_wait();
    const double t = now_ms() - t0;
    const bool ok = ane_finish_chunk();
    if (!ok) {
      ane_on_ = true;
      throw std::runtime_error("split failed: " + ane_->reason());
    }
    if (rep) ms.push_back(t);
  }
  ane_on_ = true;
  return median(ms);
}

}  // namespace ni
