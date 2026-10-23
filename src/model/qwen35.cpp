#include "model/qwen35.h"
#include "ane/ffn_split.h"
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "model/kparams.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <dirent.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>

namespace ni {
extern const char* const kTensorKernelSource;

using json = nlohmann::json;

static json read_json(const std::string& path) {
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  return json::parse(f);
}

static json quant_cfg_;
static Pipeline pipe(const std::string& n);

// Formats with kernels on every path: the specialized 4-bit and 8-bit group-64 kernels, and the generic ones
// (src/kernels/wfmt.metal) for bf16 and the other MLX affine widths MLX packs use.
bool format_supported(int bits, int gs) {
  if (bits == 16) return true;
  if (gs == 64) return bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8;
  return gs == 32 && bits == 4;
}
std::string format_name(int bits, int gs) {
  return bits == 16 ? std::string("bf16") : std::to_string(bits) + "-bit group " + std::to_string(gs);
}
std::string QLinear::fmt() const {
  if (iq) return gguf::iq_suffix(iq);
  if (q20) return "q20";
  return bits == 16 ? std::string("f16") : "f" + std::to_string(bits) + "g" + std::to_string(gs);
}
std::string QLinear::name() const { return iq ? gguf::type_name(iq) : q20 ? std::string("Q2_0") : format_name(bits, gs); }
size_t QLinear::row_bytes() const {
  if (iq) return size_t(K) / 256 * gguf::iq_block_bytes(iq);
  return size_t(K) * size_t(bits) / 8;
}
static bool gdn_par();
static bool qmm8_on();
static int env_int(const char* n, int def) {
  const char* v = getenv(n);
  return v && *v ? atoi(v) : def;
}

// The large prefill tiles with M tiles along grid x for the formats other than bf16 (enc_qrf: 3/5/6-bit and 4-bit
// group 32; enc_qmm_large: 8-bit), the M1's default (NI_QRF_MF=0/1 overrides): the threadgroups running together
// share the weight tile through the cache. Each output keeps its K order, so the results are the same bit for bit.
// M1 Pro, 2048 x 8192 x 2048 alone: 6-bit group 64 25.3 -> 23.8 ms, 5-bit 24.9 -> 23.8, 3-bit 27.3 -> 25.8, 4-bit
// group 32 59.5 -> 21.4, 8-bit 26.6 -> 24.9; a 40-layer stand-in of Qwen3.6-35B-A3B UD-IQ2_XXS (6-bit projections from
// Q5_K, 8-bit from Q6_K) reads 4096 tokens in 8.15 instead of 9.07 s (6-bit alone: 8.28). The other chips keep the
// N-fast order until measured there.
static bool prefill_mfast() {
  static const bool v = env_int("NI_QRF_MF", Gpu::get().apple_family() <= 7 ? 1 : 0) != 0;
  return v;
}

// GPU cores (Gpu::core_count; NI_GPU_CORES overrides). The dispatch heuristics below that size grids to fill the
// GPU were tuned on the 20-core M4 Pro and scale with it, so they are unchanged there.
static uint32_t gpu_cores() {
  static const uint32_t n = uint32_t(std::max(1, Gpu::get().core_count()));
  return n;
}


// NI_MOE_FUSED (read once): MoE prefill through the fused grouped kernels in qmm_moe.metal.
// 1: gate/up/SwiGLU in one GEMM, down GEMM, shared expert folded in as one more expert;
// 2 (default): same, shared expert on the dense path (35B 4k: 785 -> 862 tok/s); 0: the original path.
static int moe_fused() {
  static const int v = getenv("NI_MOE_FUSED") && *getenv("NI_MOE_FUSED") ? atoi(getenv("NI_MOE_FUSED")) : 2;
  return v;
}

// NI_MOE_QMR (read once, needs NI_MOE_FUSED): the fused MoE prefill's expert GEMMs on the qmr design
// (qmm_moe3.metal: weights dequantized straight into MMA registers, X staged in threadgroup memory,
// row fragments past a tile's rows skipped). Value: tile rows (32, the default, or 16; 1 = 32),
// 0 = off (the qmm_moe_gu / qmm_moe_dn kernels). 35B 4k prefill -5.9%, 16k -5%; logits KL 2e-5.
static int moe_qmr() {
  static const int v = [] {
    int b = getenv("NI_MOE_QMR") && *getenv("NI_MOE_QMR") ? atoi(getenv("NI_MOE_QMR")) : 32;
    if (b == 1) b = 32;
    if (b != 0 && b != 16 && b != 32) fatal("NI_MOE_QMR=%d: expected 0, 1, 16 or 32", b);
    if (b) logi("NI_MOE_QMR: MoE prefill expert GEMMs qrm_gu/qrm_dn, %d-row tiles", b);
    return b;
  }();
  return v;
}

// NI_MOE_QTM: the qmt tile (tensor_ops) on the routed experts' grouped GEMMs (qmtm_gu_* / qmtm_dn_*) instead of the
// simdgroup qrm kernels. On where the tensor units are usable (the M5 family on macOS 26, the same condition as the
// dense NI_QMT path) and the model keeps the 35B's shapes (below): 4K prefill 2,135 -> 1,574 ms and 16K 10,810 ->
// 8,570 ms on the M5 Pro. `NI_MOE_QTM=0` keeps the qrm path; every other Mac defaults to qrm.
static bool qmt_on();
static bool moe_qtm() {
  // The tile sums each output's K reduction in a different association order, so it does not reproduce the
  // simdgroup path's logits to the bit: it moves the 35B's logits by ~1.1e-3 mean KL, 1.3x the 8.1e-4 the shipped
  // qrm path already moves against the order before it, and its cold 4K and 16K temperature-0 continuations came out
  // identical. That is not noise, so it is gated by `tools/logit_compare.py --reorder` (the
  // order-change bounds, docs in AGENTS.md) instead of the model's noise floor, and the user accepted the trade on
  // 2026-10-22 for a 1.25x 4K prefill rate (2,272 -> 1,702 ms, interleaved A/B on the M5 Pro). Speed and drift were
  // measured there; a tile's rounding is shape-dependent rather than Mac-dependent. Only passes above 128 rows reach
  // this path -- shorter ones take the short-prefill MoE path above (8-row tiles), so a chunk's tail rows do not -- and
  // a dump at --chunk 160, whose 96-row tails go short, shows the drift and passes the gate. Off wherever the dense
  // path's tensor units are off.
  static const bool v = env_int("NI_MOE_QTM", qmt_on() ? 1 : 0) != 0;
  return v;
}

// The 35B's grouped prefill on the 32-core M1 Max benefits from staging a full
// 64-input group per step. Other measured devices keep their 32-input tiles.
static int moe_qmr_k(int hidden, int intermediate) {
  static const int override_k = [] {
    const int v = getenv("NI_MOE_QMR_K") ? atoi(getenv("NI_MOE_QMR_K")) : 0;
    if (v != 0 && v != 32 && v != 64) fatal("NI_MOE_QMR_K=%d: expected 0, 32 or 64", v);
    return v;
  }();
  if (override_k) return override_k;
  return hidden == 2048 && intermediate == 512 && Gpu::get().name() == "Apple M1 Max" &&
                 Gpu::get().core_count() == 32 ? 64 : 32;
}

// NI_QMM (read once): dense prefill GEMM family for 4-bit, group-64 weights.
//   unset/empty/1 (default): qmr (qmm3.metal: weights dequantized straight into MMA registers, X
//      staged in threadgroup memory) with the tile picked from M and N; 27B prefill 4k +19-21%,
//      35B +6-8% on macOS 27, greedy output identical
//   0: qmm_b4_m64_n64 (M >= 64) / qmm_b4_m32_n32 (the original kernels)
//   <kernel name>, e.g. qmr_k64_w12_m32_n64: that kernel for M >= 64 and N > 64 (tile and threads
//      parsed from the name), the automatic choice otherwise
struct QmrSel {
  bool on = false;
  std::string name;  // large-M kernel
  uint32_t bm = 32, bn = 128, threads = 128;
};
static const QmrSel& qmr_sel() {
  static const QmrSel s = [] {
    QmrSel q;
    const char* v = getenv("NI_QMM");
    if (v && !strcmp(v, "0")) return q;
    q.on = true;
    q.name = "qmr_k32_w14_m32_n128";
    // The M1 (Apple7): 64-deep K steps on 32x64 tiles of 64 threads. bench_qmm on the M1 Pro, 2048-row chunks of
    // Qwen3.5-4B's and 9B's projections: 3.07-3.17 TFLOP/s against 2.53-2.69 for qmr_k32_w14_m32_n128, on all 9
    // shapes (4B: 8192x2560 28.7 vs 33.9 ms, 2560x9216 30.6 vs 36.2; 9B: 8192x4096 43.4 vs 51.5, 4096x12288 65.2
    // vs 77.2), the fastest of the 7 qmr tiles on each
    if (Gpu::get().apple_family() <= 7) q.name = "qmr_k64_w12_m32_n64", q.bn = 64, q.threads = 64;
    if (v && *v && strcmp(v, "1") != 0) {
      unsigned bm = 0, bn = 0, wm = 0, wn = 0;
      const char* w = strstr(v, "_w");
      const char* m = strstr(v, "_m");
      if (!w || sscanf(w, "_w%1u%1u", &wm, &wn) != 2 || !m || sscanf(m, "_m%u_n%u", &bm, &bn) != 2)
        fatal("NI_QMM=%s: expected 1 or a qmr kernel name like qmr_k32_w14_m32_n128", v);
      q.name = v;
      q.bm = bm, q.bn = bn, q.threads = 32 * wm * wn;
    }
    logi("NI_QMM: dense prefill GEMM %s (M >= 64), qmr_k64_w14_m16_n64 / qmr_k64_w12_m8_n64 below", q.name.c_str());
    return q;
  }();
  return s;
}

// NI_MOE_BUCKET3 (read once; default on, 0 = moe_bucket2): moe_bucket3 (offsets and tiles by parallel
// prefix sums; identical tiles) instead of moe_bucket2 (thread 0 loops over the experts).
static const char* moe_bucket_name() {
  static const bool b3 = env_int("NI_MOE_BUCKET3", 1) != 0;
  return b3 ? "moe_bucket3" : "moe_bucket2";
}

// NI_SHORTPF (read once; default on, NI_SHORTPF=0 restores the previous behaviour): short-prefill
// paths for cached-turn suffixes and short prompts.
//  * the engine captures the turn-boundary checkpoint inside the prefill pass (gdn_*_snap kernels)
//    instead of splitting the suffix into two passes that each read every weight;
//  * passes of at most NI_SHORTPF_MAX (default and cap 128) rows run their projections on the
//    short-prefill GEMMs (NI_SHORTPF_GEMM below), with the projections that share an input in one
//    dispatch; MoE experts in 8-row tiles (NI_SHORTPF_MOE_MAX); attention over a long cached prefix
//    as (query, head) rows with the keys split over threadgroups (NI_SHORTPF_ATTN_MAX).
static bool shortpf() {
  static const bool on = [] {
    const char* v = getenv("NI_SHORTPF");
    return !(v && !strcmp(v, "0"));
  }();
  return on;
}
static int shortpf_max() {
  static const int v = getenv("NI_SHORTPF_MAX") && *getenv("NI_SHORTPF_MAX") ? atoi(getenv("NI_SHORTPF_MAX")) : 128;
  return std::clamp(v, 0, 128);
}
// Row limits of the other short-prefill paths (read once; 0 turns a path off):
//   NI_SHORTPF_ATTN_MAX (default 128): attention as (query, head) rows of one KV head with the keys
//     split over threadgroups (attn_prefill11g / 11gs + attn11_reduce). This branch runs before the
//     qattn_t one, so on an M5 a model whose heads it fits (4B qg4, 35B qg8, 27B g6) takes 11g for a chunk
//     between the old bound and this one instead of qattn_t: 192 was measured at 141 rows (9.74
//     against 11.16 ms) and 162 (11.27 against 12.71) on the 27B before it had a qattn_t form, and is
//     left off the default for that reason (the 27B's qattn_t_g6d2s8 runs 141 rows in 4.16 ms).
//   NI_SHORTPF_MOE_MAX (default 128): MoE experts in 8-row tiles (qmms_moe_gu / qmms_moe_dn)
static int shortpf_attn_max() {
  static const int v = std::clamp(env_int("NI_SHORTPF_ATTN_MAX", 128), 0, 512);
  return v;
}
static int shortpf_moe_max() {
  static const int v = std::clamp(env_int("NI_SHORTPF_MOE_MAX", 128), 0, 128);
  return v;
}
// NI_SHORTPF_GEMM (read once): kernels for the projections of passes of at most NI_SHORTPF_MAX rows
//   auto (default): per projection group, the faster of the two short-prefill families below as
//         measured with bench_qmms on M4 Pro over the 35B / 27B shapes (short_qmms())
//   qmms: register-streaming qmms (qmms.metal): weights dequantized straight into simdgroup-matrix
//         A operands, no threadgroup tiles, K split over simdgroups; any T <= 64 in 8-row blocks
//   tile: qmr (qmm3.metal) with one row tile holding all rows (two from 49 rows), 8-row steps, so each
//         weight is read and dequantized once per pass; groups sharing the input in one dispatch
//         (qmrs_*), gate/up fused (qmr_gu_*)
//   qmr:  the large-prefill choice (qmr m8 / m16 tiles, one dispatch per projection)
// 8-bit weights (the 35B's router / shared-expert gate) always take qmms in auto/tile/qmms.
enum class SpGemm { Qmr, Auto, Tile, Qmms };
static SpGemm sp_gemm() {
  static const SpGemm v = [] {
    const char* s = getenv("NI_SHORTPF_GEMM");
    if (!s || !*s || !strcmp(s, "auto")) return SpGemm::Auto;
    if (!strcmp(s, "tile")) return SpGemm::Tile;
    if (!strcmp(s, "qmms")) return SpGemm::Qmms;
    if (!strcmp(s, "qmr")) return SpGemm::Qmr;
    fatal("NI_SHORTPF_GEMM=%s: expected auto, tile, qmms or qmr", s);
  }();
  return v;
}

static bool qmr_ok(const QLinear& L) { return L.bits == 4 && L.gs == 64 && L.K % 64 == 0; }
static double qbytes(const QLinear& L) { return double(L.N) * L.K * L.bits / 8 + double(L.N) * (L.K / L.gs) * 4; }

// Auto rule: qmms wins wherever the weights are small or the rows few (bandwidth/latency bound:
// 1.2-2.5x the qmr tiles at <= 8 rows, every 35B shape up to 56 rows, fused gate/up up to 56 rows);
// the qmr tiles win on big projection groups past 16 rows (compute bound: the 27B GDN in-proj and
// attention q/k/v) and on the 27B MLP down projection (K = 17408) from 9 rows.
static bool short_qmms(int M, double bytes, bool gu, bool multi, uint32_t K) {
  if (M <= 8) return true;
  if (!gu && !multi && K >= 12288) return false;
  if (M <= 16) return true;
  if (gu) return bytes < 24e6 || M <= 56;
  return bytes < 24e6 && M <= 56;
}

// A chunk of M >= 64 new rows whose GEMM this matrix's tensor_ops tile can serve (qmtg_m64_n64 / qmt_m64_n64: qmt_on, 4-bit
// group 64, K a multiple of 64, M >= 64) is faster there than in the short-prefill families, which are sized for the
// chunks under 64 rows (NI_SHORTPF's own comment). qmr_sel().on is checked too: the qmt tile is only reached behind it,
// so NI_QMM=0 keeps the short-prefill path the chunk had. Measured on the M5 Pro, 27B, 16K cached prefix, one chunk of new
// rows, short-prefill against tensor_ops: 73 rows 679 -> 399 ms, 124 rows 1005 -> 483 ms. The short-prefill staging
// bound (one weight read, fewer blocks) only wins below 64 rows, where qmt does not run; 8-bit and unaligned
// projections, and every MoE expert GEMM, keep the short path they have.
bool Qwen35::sp_qmt_pref(const QLinear& L, int M) const {
  static const bool on = env_int("NI_SP_QMT", 1) != 0;  // 0 restores the pre-change short-prefill preference
  if (!on || cfg_.moe) return false;  // a MoE model's chunk keeps the short-prefill / qmtm split it had
  return M >= 64 && qmt_on() && qmr_sel().on && L.bits == 4 && L.gs == 64 && L.K % 64 == 0;
}

// Short-prefill qmr tile for M <= 128 rows: R row tiles (1 up to 48 rows, 2 above; fused gate/up
// tiles hold at most 32 rows), BM = 8 ceil(M / R / 8), 64-wide N tiles (32-wide measured slower on
// every 35B / 27B shape). `seg`: the multi-projection form (qmrs_*).
struct ShortTile {
  std::string name;
  uint32_t bm, bn, th;
};
static ShortTile short_tile(int M, bool gu, bool seg) {
  const std::string pre = gu ? "qmr_gu_k64" : seg ? "qmrs_k64" : "qmr_k64";
  if (M <= 8) {
    if (gu) return {"qmr_gu_k64_w12_m8_n32", 8, 32, 64};
    return {pre + "_w12_m8_n64", 8, 64, 64};
  }
  const int R = gu ? (M + 31) / 32 : M <= 48 ? 1 : 2;
  const uint32_t bm = uint32_t(((M + R - 1) / R + 7) / 8 * 8);
  return {pre + "_w14_m" + std::to_string(bm) + "_n64", bm, 64, 128};
}

// Projections sharing the half input xh (same K): one short-prefill dispatch where the kernels
// allow it, else one enc_qmm per projection.
void Qwen35::enc_proj(Encoder& e, const std::vector<SegQ>& segs, const Buffer& xh, int M) {
  bool qmt = M >= 64 && qmt_on();
  for (auto& s : segs) qmt = qmt && sp_qmt_pref(*s.L, M);
  if (!qmt && small_t(M) && sp_gemm() != SpGemm::Qmr && segs.size() <= 4) {
    bool all_qmr = true, all_qmms = true;
    double bytes = 0;
    for (auto& s : segs) {
      all_qmr = all_qmr && qmr_ok(*s.L) && s.L->K == segs[0].L->K;
      all_qmms = all_qmms && qmms_ok(*s.L);
      bytes += qbytes(*s.L);
    }
    bool qm = !all_qmr || sp_gemm() == SpGemm::Qmms ||
              (sp_gemm() == SpGemm::Auto && short_qmms(M, bytes, false, segs.size() > 1, segs[0].L->K));
    if (qm && all_qmms && M <= 64) {  // qmms: at most 8 blocks of 8 rows
      enc_qmms(e, segs, xh, M);
      return;
    }
    if (all_qmr) {
      enc_qmr_short(e, segs, xh, M);
      return;
    }
  }
  // one projection: the large-prefill kernels (enc_qmm would come back here)
  for (auto& s : segs) segs.size() > 1 ? enc_qmm(e, *s.L, xh, s.y, M, s.add) : enc_qmm_large(e, *s.L, xh, s.y, M, s.add);
}

// Short-prefill qmr: one projection on qmr_k64_*, several on the qmrs_k64_* form (short_tile()).
void Qwen35::enc_qmr_short(Encoder& e, const std::vector<SegQ>& segs, const Buffer& xh, int M) {
  const bool seg = segs.size() > 1;
  const ShortTile t = short_tile(M, false, seg);
  e.use(pipe(t.name));
  if (!seg) {
    const QLinear& L = *segs[0].L;
    kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, segs[0].add ? 1u : 0u, uint32_t(L.gs), 0, 0};
    e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, segs[0].y);
    e.val(5, p);
    e.dispatch((L.N + t.bn - 1) / t.bn, (M + t.bm - 1) / t.bm, 1, t.th);
    return;
  }
  kp::Qmms p{};
  p.K = segs[0].L->K;
  p.T = uint32_t(M);
  p.x_ld = p.K;
  p.nseg = uint32_t(segs.size());
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const SegQ& sq = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = sq.L->N;
      p.y_ld[i] = sq.L->N;
      p.add[i] = sq.add ? 1u : 0u;
      p.tg_begin[i] = tg;
      tg += (sq.L->N + t.bn - 1) / t.bn;
    }
    e.buf(int(i * 4 + 0), sq.L->w->buf);
    e.buf(int(i * 4 + 1), sq.L->s->buf);
    e.buf(int(i * 4 + 2), sq.L->b->buf);
    e.buf(int(i * 4 + 3), sq.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.buf(16, xh);
  e.val(17, p);
  e.dispatch(tg, (M + t.bm - 1) / t.bm, 1, t.th);
}

// Creates the compute pipelines the prefill paths can pick (every short-prefill variant the row
// count can select, and the large-prefill GEMM / attention / MoE / GDN kernels) at load, in
// parallel, instead of at first use: on a binary whose kernels are not in the system shader cache
// yet each first use compiles (the first cached turns of a new build took 250-330 ms instead of
// 90-120 ms). With a warm cache this takes ~0.1 s. NI_PREWARM=0 skips it.
void Qwen35::prewarm_pipelines() {
  if (getenv("NI_PREWARM") && !strcmp(getenv("NI_PREWARM"), "0")) return;
  const double t0 = now_ms();
  std::set<std::string> names = {"embed_q4", "rows_to_half", "attn_prep", "attn_prefill11", "attn_prefill11r", "swiglu_half",
                                 "moe_topk_rows", "moe_topk_rows8", "moe_bucket", "moe_bucket2", "moe_bucket3", "moe_combine", "copy_u32", "kv_copy_prefix",
                                 "gdn_prep_par", "gdn_step_par_l16", "gdn_step_parr4_l16", "gdn_gnorm", "topk_stage1",
                                 "sample_stage2",
                                 "qmr_k32_w14_m32_n128", "qmr_k64_w14_m16_n64", "qmr_k64_w12_m8_n64",
                                 "qmr_gu_k32_w14_m32_n64", "qmr_gu_k64_w14_m16_n64", "qmr_gu_k64_w12_m8_n32",
                                 "qmm_b8_m32_n32", "qmm_moe_gu_b4_m64_n32", "qmm_moe_dn_b4_m64_n64"};
  if (qmm8_on())
    for (const char* n : {"qmr8_k32_w14_m32_n128", "qmr8_k32_w14_m16_n64", "qmr8_gu_k32_w14_m32_n64", "qmr8_gu_k32_w14_m16_n64"})
      names.insert(n);
  if (qmm8_on() && prefill_mfast()) names.insert("qmr8_k32_w14_m32_n128_mf");
  if (Gpu::get().apple_family() <= 7)  // the M1's prefill attention and tiles (enc_attn_prefill, qmr_sel, enc_qmm_gu)
    for (const char* n : {"attn_prefill11_k16xr", "qmr_k64_w12_m32_n64", "qmr_gu_k64_w14_m32_n64"}) names.insert(n);
  if (cfg_.moe && moe_qmr()) {
    const std::string k = std::to_string(moe_qmr_k(cfg_.hidden, cfg_.moe_I));
    names.insert("qrm_gu_k" + k + "_m" + std::to_string(moe_qmr()) + "_n64");
    names.insert("qrm_dn_k" + k + "_m" + std::to_string(moe_qmr()) + "_n128");
  }
  if (shortpf()) {
    for (int M = 1; M <= std::max(1, shortpf_max()); ++M) {
      names.insert(short_tile(M, false, false).name);
      names.insert(short_tile(M, false, true).name);
      names.insert(short_tile(M, true, false).name);
    }
    for (int b : {4, 8})
      for (int t = 1; t <= 8; ++t) {
        names.insert("qmms_b" + std::to_string(b) + "_t" + std::to_string(t));
        names.insert("qmms_gu_b" + std::to_string(b) + "_t" + std::to_string(t));
      }
    for (const char* n : {"qmms_moe_gu_b4", "qmms_moe_dn_b4", "attn_prefill11g", "attn_prefill11gs", "attn11_reduce",
                          "gdn_prep_par_snap", "gdn_step_par_snap_l16", "gdn_step_parr4_snap_l16", "qmms_moe2_gu_b4",
                          "qmms_moe2_dn_b4"})
      names.insert(n);
  }
  // kernels of the weight formats without specialized ones (wfmt.metal), decode paths included
  {
    std::set<std::string> fmts;  // generic dense formats
    bool moe_gen = false;
    auto see = [&](const QLinear& q) {
      if (q && !q.q4() && !q.q8()) fmts.insert(q.fmt());
    };
    for (const Layer& L : layers_) {
      for (const QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out, &L.q, &L.k, &L.v, &L.o, &L.router, &L.shared_gate, &L.sg,
                               &L.su, &L.sd, &L.mg, &L.mu, &L.md})
        see(*q);
      if (L.mg && L.mg.q8()) names.insert("mlp_gate_up_f8g64");
      if (L.eg.q20) {  // Q2_0 routed experts (src/model/moe_q20.cpp)
        for (const std::string& n : q20::kernel_names(q20_arena())) names.insert(n);
      } else if (L.eg.iq) {  // llama.cpp's IQ blocks (qwen35_iq.cpp)
        const bool t16 = cfg_.moe_I == 512 && cfg_.shared_I == 512;
        for (const std::string& n : {"moei_gu_rows_" + L.eg.fmt() + "_r2", "moe8i_gu_" + L.eg.fmt(),
                                     "qrmi_gu_" + L.eg.fmt() + "_m32_n64_k32", "moei_down_rows_" + L.ed.fmt() + "_r8" +
                                     (t16 ? "_t16" : ""), "moe8i_down_" + L.ed.fmt() + "_nf2",
                                     "qrmi_dn_" + L.ed.fmt() + "_m32_n64", std::string("moe_bucket3"),
                                     std::string("moe_topk256_rows"), std::string("moe8_plan"),
                                     std::string("moe8_combine"), std::string("moe_combine")})
          names.insert(n);
      } else if (L.eg && !L.eg.q4() && L.eg.bits != 4) {
        moe_gen = true;
        names.insert("qrmf_gu_" + L.eg.fmt() + "_m32_n64");
        names.insert("qrmf_dn_" + L.eg.fmt() + "_m32_n128");
        names.insert("qmmsf_moe_gu_" + L.eg.fmt());
        names.insert("qmmsf_moe_dn_" + L.eg.fmt());
      }
      if (L.eg && !moe_rows_kernel(true, L.eg, L.sg).empty() && !(L.eg.q4() && L.sg.q4())) {
        names.insert(moe_rows_kernel(true, L.eg, L.sg));
        names.insert(moe_rows_kernel(false, L.ed, L.sd, moe_wide()));
        names.insert("moe8g_gu_" + L.eg.fmt() + "_" + L.sg.fmt());  // when instantiated (else not created)
      }
    }
    see(lm_head_);
    if (!embed_.q4()) names.insert("embed_" + embed_.fmt());
    for (const std::string& f : fmts) {
      for (const char* t : {"_m32_n128", "_m16_n64", "_m8_n64"}) names.insert("qrf_" + f + t);
      for (const char* t : {"_m32_n64", "_m16_n64", "_m8_n32"}) names.insert("qrf_gu_" + f + t);
      if (f == "f16") {
        names.insert("qrf_f16_m32_n128_mf");
        names.insert("qrf_gu_f16_m32_n64_mf");
      } else if (prefill_mfast()) {
        names.insert("qrf_" + f + "_m32_n128_mf");
      }
      for (int t = 1; t <= 8; ++t) {
        names.insert("qmmsf_" + f + "_t" + std::to_string(t));
        names.insert("qmmsf_gu_" + f + "_t" + std::to_string(t));
      }
      for (const char* t : {"_r2", "_r4", "_r2_add", "_r2_norm", "_r4_norm"}) names.insert("qmvg_" + f + t);
      names.insert("qmvg_multi_" + f);
      names.insert("mlp_gate_up_" + f);
      names.insert("qmm8g_" + f + "_nf1");
      names.insert("qmm8g_" + f + "_nf2");
    }
    if (!fmts.empty() || moe_gen) names.insert("moe_topk256_rows");
  }
  std::vector<std::string> v(names.begin(), names.end());
  Gpu& g = Gpu::get();
  g.pipeline(v[0]);  // compiles the library once, before the threads
  const unsigned nt = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned i = 0; i < nt; ++i)
    th.emplace_back([&, i] {
      for (size_t j = i; j < v.size(); j += nt) g.pipeline(v[j], false);
    });
  for (auto& t : th) t.join();
  logi("prewarmed %zu prefill pipelines in %.0f ms", v.size(), now_ms() - t0);
}

static void parse_config(const json& root, Qwen35Config& cfg) {
  json t = root.contains("text_config") ? root["text_config"] : root;
  json quant = root.contains("quantization") ? root["quantization"] : json::object();
  cfg.group_size = quant.value("group_size", 64);
  cfg.hidden = t["hidden_size"];
  cfg.layers = t["num_hidden_layers"];
  cfg.vocab = t["vocab_size"];
  cfg.eps = t.value("rms_norm_eps", 1e-6f);
  cfg.lin_hk = t["linear_num_key_heads"];
  cfg.lin_hv = t["linear_num_value_heads"];
  cfg.lin_dk = t["linear_key_head_dim"];
  cfg.lin_dv = t["linear_value_head_dim"];
  cfg.conv_k = t.value("linear_conv_kernel_dim", 4);
  cfg.hq = t["num_attention_heads"];
  cfg.hkv = t["num_key_value_heads"];
  cfg.head_dim = t.value("head_dim", cfg.hidden / cfg.hq);
  float prf = 0.25f;
  if (t.contains("rope_parameters")) {
    auto& rp = t["rope_parameters"];
    cfg.theta = rp.value("rope_theta", 1e7f);
    prf = rp.value("partial_rotary_factor", 0.25f);
    if (rp.contains("mrope_section") && rp["mrope_section"].is_array() && rp["mrope_section"].size() == 3)
      for (int i = 0; i < 3; ++i) cfg.mrope[i] = rp["mrope_section"][i];
  } else {
    cfg.theta = t.value("rope_theta", 1e7f);
    prf = t.value("partial_rotary_factor", 0.25f);
  }
  cfg.rot_dims = int(cfg.head_dim * prf);
  int interval = t.value("full_attention_interval", 4);
  cfg.is_linear.resize(cfg.layers);
  if (t.contains("layer_types") && t["layer_types"].is_array()) {
    for (int i = 0; i < cfg.layers; ++i) cfg.is_linear[i] = t["layer_types"][i] == "linear_attention";
  } else {
    for (int i = 0; i < cfg.layers; ++i) cfg.is_linear[i] = (i + 1) % interval != 0;
  }
  cfg.experts = t.value("num_experts", 0);
  cfg.moe = cfg.experts > 0;
  cfg.top_k = t.value("num_experts_per_tok", 0);
  cfg.moe_I = t.value("moe_intermediate_size", 0);
  cfg.shared_I = t.value("shared_expert_intermediate_size", 0);
  cfg.dense_I = t.value("intermediate_size", 0);
  cfg.norm_topk = t.value("norm_topk_prob", true);
  cfg.hc = t.value("hc_count", 1);
  for (auto& [k, v] : quant.items())
    if (v.is_object() && v.value("mode", std::string()).rfind("iq", 0) == 0) cfg.iq = true;
  auto eos = root.contains("eos_token_id") ? root["eos_token_id"] : t["eos_token_id"];
  if (eos.is_array())
    for (auto& e : eos) cfg.eos.push_back(e);
  else
    cfg.eos.push_back(eos);
}

Qwen35Config Qwen35::read_config(const std::string& dir) {
  Qwen35Config c;
  parse_config(read_json(dir + "/config.json"), c);
  return c;
}

void Qwen35::load_config(const std::string& dir) {
  json root = read_json(dir + "/config.json");
  quant_cfg_ = root.contains("quantization") ? root["quantization"] : json::object();
  parse_config(root, cfg_);
  if (hi_ <= 0) hi_ = cfg_.layers;
  if (lo_ < 0 || lo_ >= hi_ || hi_ > cfg_.layers) fatal("layers %d:%d of a %d-layer model", lo_, hi_, cfg_.layers);
}

// Tensor name in the checkpoint for a canonical (MLX) name "model.layers.3.mlp.gate_proj.weight", "lm_head.weight":
// MLX conversions prefix "language_model." (vision-language packs) or nothing; the official Hugging Face
// checkpoints name the text model "model.language_model.*" and keep "lm_head.*" at the root.
std::string Qwen35::tname(const std::string& canonical) const {
  if (!hf_names_) return prefix_ + canonical;
  if (canonical.rfind("model.", 0) == 0) return "model.language_model." + canonical.substr(6);
  return canonical;
}

// A float tensor as bf16 (t itself when it is bf16, else an owned bf16 copy); `offset` is added first (the
// zero-centred RMSNorm weights of Hugging Face checkpoints are applied as 1 + w).
const Tensor* Qwen35::bf16_copy(const Tensor* t, float offset) {
  if (!t) return t;
  if (t->dtype == DType::BF16 && offset == 0.0f) return t;
  if (t->dtype != DType::BF16 && t->dtype != DType::F32 && t->dtype != DType::F16) fatal("float tensor of unsupported dtype");
  const size_t n = size_t(t->numel());
  Buffer b = Gpu::get().alloc(n * 2 + 16, "bf16_copy", /*tracked=*/false);
  uint16_t* d = b.as<uint16_t>();
  for (size_t i = 0; i < n; ++i) {
    const float v = t->dtype == DType::F32 ? t->buf.as<float>()[i]
                    : t->dtype == DType::F16 ? float(t->buf.as<_Float16>()[i])
                                             : bf16_to_f32(t->buf.as<uint16_t>()[i]);
    d[i] = f32_to_bf16(v + offset);
  }
  owned_bufs_.push_back(b);
  Tensor c = *t;
  c.buf = b.view(0, n * 2);
  c.dtype = DType::BF16;
  owned_.push_back(c);
  return &owned_.back();
}

// The GGUF type of a module kept in llama.cpp's IQ blocks (quantization mode iq1_s / iq2_xxs / iq2_s / iq3_xxs /
// iq3_s), else 0
static uint32_t iq_mode(const std::string& full, const std::string& name) {
  for (const auto& key : {full, name})
    if (quant_cfg_.contains(key) && quant_cfg_[key].is_object())
      if (const uint32_t t = gguf::iq_mode_type(quant_cfg_[key].value("mode", std::string()))) return t;
  return 0;
}

QLinear Qwen35::qlin(const std::string& name, bool required, bool q20_ok) {
  QLinear L;
  const std::string full = tname(name);
  L.w = ws_.find(full + ".weight");
  if (!L.w) {
    if (required) fatal("missing %s.weight", full.c_str());
    return L;
  }
  if (const uint32_t t = iq_mode(full, name)) {
    // routed experts in llama.cpp's blocks, [E, N, K / 256 * block bytes] as bytes: read by load_iq_experts
    const size_t bb = gguf::iq_block_bytes(t);
    if (L.w->dtype != DType::U8 || L.w->shape.size() != 3 || L.w->shape[2] % int64_t(bb))
      fatal("%s.weight: not the %s blocks of stacked experts", full.c_str(), gguf::type_name(t).c_str());
    L.iq = t;
    L.bits = gguf::iq_bits(t);
    L.gs = 256;
    L.N = uint32_t(L.w->shape[1]);
    L.K = uint32_t(L.w->shape[2] / int64_t(bb) * 256);
    L.s = L.b = L.w;
    iq_src_[L.w] = full + ".weight";
    return L;
  }
  L.s = ws_.find(full + ".scales");
  L.b = ws_.find(full + ".biases");
  if (!L.s && !L.b) {
    // unquantized: bf16 weights (f16 / f32 ones get a bf16 copy)
    if (L.w->shape.size() != 2) fatal("%s.weight: expected a 2-D matrix", full.c_str());
    if (L.w->dtype != DType::BF16) {
      logi("%s: %s weights converted to bf16 at load", full.c_str(), L.w->dtype == DType::F16 ? "f16" : "f32");
      L.w = bf16_copy(L.w);
    }
    L.bits = 16;
    L.gs = 64;
    L.N = uint32_t(L.w->shape[0]);
    L.K = uint32_t(L.w->shape[1]);
    L.s = L.b = L.w;
    if (L.K % 64) fatal("%s: input width %u is not a multiple of 64", full.c_str(), L.K);
    return L;
  }
  int bits = quant_cfg_.value("bits", 4), gs = quant_cfg_.value("group_size", 64);
  std::string mode = quant_cfg_.value("mode", std::string("affine"));
  // per-module overrides are keyed by the module path without ".weight"
  for (auto key : {full, name}) {
    if (quant_cfg_.contains(key) && quant_cfg_[key].is_object()) {
      bits = quant_cfg_[key].value("bits", bits);
      gs = quant_cfg_[key].value("group_size", gs);
      mode = quant_cfg_[key].value("mode", mode);
    }
  }
  if (mode == "q2_0") {
    // ggml Q2_0 (src/model/moe_q20.h): codes [.., N, K/16] uint32 in MLX's 2-bit order, fp16 d [.., N, K/64], no biases
    if (!q20_ok) fatal("%s: Q2_0 weights are supported for Qwen3.8-Flash-Next's routed experts only", full.c_str());
    if (bits != 2 || gs != 64 || !L.s || L.b || L.s->dtype != DType::F16 || L.w->dtype != DType::U32)
      fatal("%s: Q2_0 needs uint32 2-bit codes, group 64 and fp16 scales without biases", full.c_str());
    L.bits = 2;
    L.gs = 64;
    L.q20 = true;
    L.K = uint32_t(L.w->shape.back() * 16);
    L.N = uint32_t(L.w->shape[L.w->shape.size() - 2]);
    if (L.K % 64 || int64_t(L.K / 64) != L.s->shape.back() || L.s->shape.size() != L.w->shape.size())
      fatal("%s: Q2_0 scales do not match the codes", full.c_str());
    L.b = L.s;  // bindings stay valid; no kernel reads it
    return L;
  }
  if (mode != "affine")
    fatal("%s: MLX \"%s\" quantization is not supported; Kapitsa runs MLX affine quantizations (2-8 bits) and bf16",
          full.c_str(), mode.c_str());
  if (!L.s || !L.b) fatal("%s: scales without biases (not an MLX affine quantization)", full.c_str());
  L.bits = bits;
  L.gs = gs;
  int64_t words = L.w->shape.back();
  L.K = uint32_t(words * 32 / bits);
  L.N = uint32_t(L.w->shape[L.w->shape.size() - 2]);
  if (int64_t(L.K / gs) != L.s->shape.back() || L.K % gs) fatal("%s: bits/group mismatch", full.c_str());
  if (!format_supported(bits, gs))
    fatal("%s: %s weights are not supported (bf16, 3/4/5/6/8-bit group 64 and 4-bit group 32 are)", full.c_str(),
          format_name(bits, gs).c_str());
  return L;
}

q20::Experts Qwen35::q20_experts(const Layer& L) const {
  q20::Experts x;
  x.gc = L.eg.w->buf; x.gd = L.eg.s->buf;
  x.uc = L.eu.w->buf; x.ud = L.eu.s->buf;
  x.dc = L.ed.w->buf; x.dd = L.ed.s->buf;
  return x;
}

q20::Shared Qwen35::q20_shared(const Layer& L) const {
  return q20::Shared{L.sg.w->buf, L.sg.s->buf, L.sg.b->buf, L.su.w->buf, L.su.s->buf,
                     L.su.b->buf, L.sd.w->buf, L.sd.s->buf, L.sd.b->buf};
}

q20::Shape Qwen35::q20_shape() const {
  return q20::Shape{uint32_t(cfg_.hidden), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(cfg_.top_k),
                    uint32_t(cfg_.experts)};
}

q20::Experts Qwen35::q20_fetch(Encoder& e, const Layer& L, const Buffer& ids, uint32_t n, int T,
                               const std::function<void()>& between) {
  (void)e, (void)ids, (void)n, (void)T;
  if (between) between();
  return q20_experts(L);
}

Qwen35::Qwen35(const std::string& dir, int prefill_chunk) : dir_(dir), tmax_(prefill_chunk) {
  double t0 = now_ms();
  load_config(dir);
  // Text-only engine for now: skip the vision tower and (until speculative decoding lands) MTP. Under the test
  // transforms (NI_DEQUANT / NI_REQUANT) the linears they select are read only when converted.
  // routed experts in llama.cpp's IQ blocks (GGUF imports) are read by load_iq_experts, into their GPU layout
  WeightStore::Filter defer = [td = transform_defer()](const std::string& n) {
    if (n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0 && iq_mode(n.substr(0, n.size() - 7), "")) return true;
    return td && td(n);
  };
  ws_.open_dir(
      dir,
      [](const std::string& n) {
        return n.find("vision_tower") == std::string::npos && n.find("visual.") == std::string::npos &&
               n.rfind("mtp.", 0) != 0;
      },
      nullptr, defer);
  if (ws_.has("language_model.model.embed_tokens.weight")) prefix_ = "language_model.";
  else if (ws_.has("model.language_model.embed_tokens.weight")) hf_names_ = true;  // official Hugging Face layout
  else if (ws_.has("model.embed_tokens.weight")) prefix_ = "";
  else fatal("unrecognized weight naming");
  // Hugging Face checkpoints store the RMSNorm weights zero-centred (applied as 1 + w; MLX conversions add the
  // 1) and conv1d as [C, 1, 4] (MLX: [C, 4, 1]); mlx-lm tells them apart by the conv1d shape, and so does this.
  for (int i = 0; i < cfg_.layers && !hf_norms_; ++i)
    if (cfg_.is_linear[i]) {
      const Tensor* cw = ws_.find(tname("model.layers." + std::to_string(i) + ".linear_attn.conv1d.weight"));
      hf_norms_ = cw && cw->shape.size() == 3 && cw->shape.back() != 1;
      break;
    }
  if (hf_names_ || hf_norms_)
    logi("Hugging Face checkpoint layout (%s%s)", hf_names_ ? "model.language_model.* names" : "conv1d [C, 1, 4]",
         hf_norms_ ? ", RMSNorm weights read as 1 + w" : "");

  embed_ = qlin("model.embed_tokens");
  // tied embeddings (the small Qwen3.5 models): the output layer is the embedding table, stored in the
  // same [vocab, hidden] layout
  lm_head_ = ws_.has(tname("lm_head.weight")) ? qlin("lm_head") : embed_;
  final_norm_ = &ws_.get(tname("model.norm.weight"));
  layers_.resize(cfg_.layers);
  for (int i = 0; i < cfg_.layers; ++i) {
    Layer& L = layers_[i];
    std::string p = "model.layers." + std::to_string(i) + ".";
    L.linear = cfg_.is_linear[i];
    L.in_norm = &ws_.get(tname(p + "input_layernorm.weight"));
    L.post_norm = &ws_.get(tname(p + "post_attention_layernorm.weight"));
    if (L.linear) {
      std::string a = p + "linear_attn.";
      L.qkv = qlin(a + "in_proj_qkv");
      L.z = qlin(a + "in_proj_z");
      L.a = qlin(a + "in_proj_a");
      L.b = qlin(a + "in_proj_b");
      L.out = qlin(a + "out_proj");
      L.conv_w = &ws_.get(tname(a + "conv1d.weight"));
      L.A_log = &ws_.get(tname(a + "A_log"));
      L.dt_bias = &ws_.get(tname(a + "dt_bias"));
      L.gnorm = &ws_.get(tname(a + "norm.weight"));
    } else {
      std::string a = p + "self_attn.";
      L.q = qlin(a + "q_proj");
      L.k = qlin(a + "k_proj");
      L.v = qlin(a + "v_proj");
      L.o = qlin(a + "o_proj");
      L.q_norm = &ws_.get(tname(a + "q_norm.weight"));
      L.k_norm = &ws_.get(tname(a + "k_norm.weight"));
    }
    if (cfg_.moe) {
      std::string m = p + "mlp.";
      L.router = qlin(m + "gate");
      L.shared_gate = qlin(m + "shared_expert_gate");
      L.router.gate = L.shared_gate.gate = true;
      if (hf_names_ && ws_.has(tname(m + "experts.gate_up_proj")))
        fatal("%s: Hugging Face MoE checkpoints (fused experts.gate_up_proj) are not supported; use an MLX conversion "
              "(e.g. mlx-community/<model>-bf16)", dir.c_str());
      L.eg = qlin(m + "switch_mlp.gate_proj");
      L.eu = qlin(m + "switch_mlp.up_proj");
      L.ed = qlin(m + "switch_mlp.down_proj");
      L.sg = qlin(m + "shared_expert.gate_proj");
      L.su = qlin(m + "shared_expert.up_proj");
      L.sd = qlin(m + "shared_expert.down_proj");
    } else {
      std::string m = p + "mlp.";
      L.mg = qlin(m + "gate_proj");
      L.mu = qlin(m + "up_proj");
      L.md = qlin(m + "down_proj");
    }
  }
  // The kernels read every float parameter as bf16. Checkpoints differ: the Qwen3.5-4B MLX conversion
  // stores A_log as f32 (bf16 values widened), the 27B / 35B as bf16; read as bf16, f32 data gave wrong
  // decay rates in every DeltaNet layer (logits KL 6e-2 vs MLX, greedy text off after a dozen tokens).
  {
    int lossy = 0, converted = 0;
    auto fix = [&](const Tensor*& t) {
      if (t && t->dtype != DType::BF16) {
        t = bf16_param(t, lossy);
        ++converted;
      }
    };
    if (hf_norms_) {  // 1 + w, rounded once to bf16 (as mlx-lm's sanitize); the DeltaNet gated norm is plain w
      for (Layer& L : layers_)
        for (const Tensor** t : {&L.in_norm, &L.post_norm, &L.q_norm, &L.k_norm})
          if (*t) *t = bf16_copy(*t, 1.0f);
      final_norm_ = bf16_copy(final_norm_, 1.0f);
    }
    for (Layer& L : layers_)
      for (const Tensor** t : {&L.in_norm, &L.post_norm, &L.conv_w, &L.A_log, &L.dt_bias, &L.gnorm, &L.q_norm, &L.k_norm})
        fix(*t);
    fix(final_norm_);
    if (converted)
      logi("converted %d parameter tensors to bf16 at load (%d values not exactly representable)", converted, lossy);
    auto check = [](const QLinear& q) {
      if (q && !q.dense() && !q.iq && (q.s->dtype != DType::BF16 || q.b->dtype != DType::BF16))
        fatal("quantized weights with non-bf16 scales/biases are not supported");
    };
    check(embed_);
    check(lm_head_);
    for (const Layer& L : layers_)
      for (const QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out, &L.q, &L.k, &L.v, &L.o, &L.router, &L.shared_gate, &L.eg,
                               &L.eu, &L.ed, &L.sg, &L.su, &L.sd, &L.mg, &L.mu, &L.md})
        check(*q);
  }
  load_iq_experts();
  // Checks the kernels' assumptions.
  check_shapes();
  test_transforms();
  check_formats();

  if (!getenv("NI_NO_MTP")) load_mtp(dir);
  Gpu::get().make_resident(ws_.buffers());
  Gpu::get().make_resident(owned_bufs_);  // converted / copied tensors (bf16 copies, test transforms, MTP head)
  alloc_state();
  // the chain passes of a cached suffix size the verify buffers for their own rows (vpf_pass_rows), which is wider
  // than the drafter's tree on a target whose block drafts 16 rows: raising the capacity here, before anything is in
  // flight, only resizes this model's buffers (the drafter sizes its own drafting width from kf_dense_rows)
  pass_rows_ = vpf_pass_rows(cfg_);
  if (pass_rows_ > tree_rows_) set_tree_rows(pass_rows_);
  prewarm_pipelines();
  logi("loaded %s: %d layers, %s, %.1f GB mapped in %.0f ms; weights %s", dir.c_str(), cfg_.layers,
       cfg_.moe ? "MoE" : "dense", ws_.total_bytes() / 1e9, now_ms() - t0, formats_summary().c_str());
  // GPU + Neural Engine split of the dense prefill FFN: on by default for dense models with a
  // calibration or a default share for this chip (27B 4k prefill ~2.8x); NI_ANE=0 keeps the GPU alone.
  if (!cfg_.moe && !(getenv("NI_ANE") && !strcmp(getenv("NI_ANE"), "0"))) init_ane();
}

Qwen35::Qwen35(const std::string& dir, int prefill_chunk, NoLoad) : dir_(dir), tmax_(prefill_chunk) {}

// The kernels' assumptions on the architecture.
void Qwen35::check_shapes() const {
  if (cfg_.lin_dk != 128 || cfg_.lin_dv != 128) fatal("GDN head dims must be 128");
  if (cfg_.head_dim != 256) fatal("attention head_dim must be 256");
  if (cfg_.hq / cfg_.hkv > 16) fatal("GQA ratio > 16 unsupported");
  if (cfg_.conv_k != 4) fatal("conv kernel must be 4");
  if (cfg_.moe && (cfg_.moe_I % 64 || cfg_.shared_I % 64)) fatal("expert sizes must be multiples of 64");
  // the decode / row MoE kernels stage every slot's h: (k + 1) * I floats, 7168 at most (the _w forms past 4608)
  if (cfg_.moe && (cfg_.top_k + 1) * std::max(cfg_.moe_I, cfg_.shared_I) > 7168) fatal("MoE decode: (k+1)*I > 7168");
  if (cfg_.moe && (cfg_.experts % 32 || cfg_.experts > 512 || cfg_.top_k > 16)) fatal("MoE: unsupported expert count");
  if (cfg_.moe && cfg_.hidden > 4096) fatal("MoE decode kernels stage at most 4096 inputs");
}

// Staged h floats of the decode / row MoE kernels past the 4608 of the original forms: their wide (_w) instantiations
bool Qwen35::moe_wide() const { return cfg_.moe && (cfg_.top_k + 1) * std::max(cfg_.moe_I, cfg_.shared_I) > 4608; }

Qwen35::~Qwen35() {
  // a background ANE compile only touches its own copies (no Metal objects): let it finish on its own
  if (ane_thread_.joinable()) ane_thread_.detach();
}

// NI_QMM_GU without its log line (enc_qmm_gu logs it): the dense gate/up/SwiGLU fused on qmr
static bool qmm_gu_enabled() {
  static const bool v = [] {
    const char* g = getenv("NI_QMM_GU");
    return !(g && !strcmp(g, "0")) && qmr_sel().on;
  }();
  return v;
}

// The activation buffers and their sizes for prefill chunks of T rows: alloc_state allocates exactly
// these and plan() adds them up, so a fit check counts what loading will allocate. hgu_rows: rows of the
// dense FFN's fp32 gate/up outputs (hg_/hu_; see alloc_state).
std::vector<Qwen35::ActBuf> Qwen35::act_layout(const Qwen35Config& c, int T, int maxv, bool mtp, size_t hgu_rows) {
  const size_t D = size_t(c.hidden), V = size_t(c.vocab), Tz = size_t(T);
  const size_t C = 2 * size_t(c.lin_hk) * c.lin_dk + size_t(c.lin_hv) * c.lin_dv, VD = size_t(c.lin_hv) * c.lin_dv;
  const size_t HQD = size_t(c.hq) * c.head_dim, KVD = size_t(c.hkv) * c.head_dim;
  const size_t mtp_rows = mtp ? Tz : size_t(maxv);  // MTP history passes take up to a chunk of rows
  const size_t max_chunks = 1024;
  std::vector<ActBuf> v = {
      {&Qwen35::x_, Tz * D * 4, "x"},
      {&Qwen35::xn_, D * 4, "xn"},
      {&Qwen35::logits_, V * maxv * 4, "logits"},
      {&Qwen35::qkv_, Tz * C * 4, "qkv"},
      {&Qwen35::zb_, Tz * VD * 4, "zb"},
      {&Qwen35::ab_, Tz * c.lin_hv * 4, "ab"},
      {&Qwen35::bb_, Tz * c.lin_hv * 4, "bb"},
      {&Qwen35::gq_, Tz * c.lin_hk * c.lin_dk * 4, "gq"},
      {&Qwen35::gk_, Tz * c.lin_hk * c.lin_dk * 4, "gk"},
      {&Qwen35::gv_, Tz * VD * 4, "gv"},
      {&Qwen35::gg_, Tz * c.lin_hv * 4, "gg"},
      {&Qwen35::gbeta_, Tz * c.lin_hv * 4, "gbeta"},
      {&Qwen35::gout_, Tz * VD * 4, "gout"},
      {&Qwen35::qg_, Tz * HQD * 2 * 4, "qg"},
      {&Qwen35::kr_, Tz * KVD * 4, "kr"},
      {&Qwen35::vr_, Tz * KVD * 4, "vr"},
      {&Qwen35::qo_, Tz * HQD * 4, "qo"},
      {&Qwen35::attn_out_, Tz * HQD * 4, "attn_out"},
      {&Qwen35::part_o_, size_t(c.hq) * max_chunks * c.head_dim * 4, "part_o"},
      {&Qwen35::part_ml_, size_t(c.hq) * max_chunks * 2 * 4, "part_ml"},
      {&Qwen35::eidx_, 64 * 4, "eidx"},
      {&Qwen35::ewts_, 64 * 4, "ewts"},
      {&Qwen35::eh_, size_t(maxv * c.top_k + 1) * maxv * std::max(c.moe_I, c.shared_I) * 4 + 16, "eh"},
      {&Qwen35::ue_, size_t(maxv) * 64 * 4, "uniq_experts"},
      {&Qwen35::uw_, size_t(maxv) * 64 * maxv * 4, "uniq_weights"},
      {&Qwen35::un_, 16, "uniq_count"},
      {&Qwen35::cand_v_, size_t(maxv) * (V / 2048 + 2) * 64 * 4, "cand_v"},
      {&Qwen35::cand_i_, size_t(maxv) * (V / 2048 + 2) * 64 * 4, "cand_i"},
      {&Qwen35::mtp_e_, mtp_rows * D * 4, "mtp_e"},
      {&Qwen35::mtp_u_, mtp_rows * 2 * D * 4, "mtp_u"},
      {&Qwen35::mtp_x_, mtp_rows * D * 4, "mtp_x"},
      {&Qwen35::mtp_post_, size_t(maxv) * D * 4, "mtp_post"},
      {&Qwen35::xrn_, size_t(maxv) * std::max(D, VD) * 4, "xrn"},
      {&Qwen35::samp_, 64, "samples"},
      {&Qwen35::dlog_, V * 4, "draft_logits"},
      {&Qwen35::mh_, size_t(maxv) * std::max(c.dense_I, 1) * 4, "mlp_h"},
      {&Qwen35::sprob_, 64 * 4, "sprob"},
  };
  const size_t P = Tz * std::max(c.top_k, 1);
  const size_t I = size_t(std::max({c.moe_I, c.shared_I, c.dense_I, 1}));
  // fused MoE prefill: T shared-expert rows after the P pair rows (hg/hu stay: a layer the fused kernels
  // do not support falls back to the original path). IQ experts (GGUF imports) run their own prefill: neither (the
  // 35B's 2048-row chunk: 86 MB less), unless an MTP head's experts (4-bit) may take the other paths.
  const bool iq_only = c.iq && !mtp;
  const size_t S = c.moe && moe_fused() && !iq_only ? Tz : 0;
  const size_t W = size_t(std::max({c.hidden, c.lin_hv * c.lin_dv, c.hq * c.head_dim, c.dense_I}));  // widest half row
  // MoE-only buffers (routing, sorted pairs, expert outputs, the shared expert's fallback rows) take one
  // row on dense models
  const size_t Pm = c.moe ? P : 1, Tm = c.moe ? Tz : 1;
  const std::vector<ActBuf> w = {
      {&Qwen35::xh_, Tz * W * 2, "xh"},
      {&Qwen35::hh_, Tz * W * 2, "hh"},
      {&Qwen35::ridx_, Pm * 4, "ridx"},
      {&Qwen35::rwts_, Tm * (c.top_k + 1) * 4 + 16, "rwts"},
      {&Qwen35::spair_, (Pm + S) * 4, "spair"},
      {&Qwen35::stok_, (Pm + S) * 4, "stok"},
      {&Qwen35::tiles_, (Pm + S + size_t(std::max(c.experts, 1)) + 1) * 3 * 4, "tiles"},
      {&Qwen35::ntiles_, 16, "ntiles"},
      {&Qwen35::hg_, (c.moe ? (iq_only ? 1 : P) : hgu_rows) * I * 4, "hg"},
      {&Qwen35::hu_, (c.moe ? (iq_only ? 1 : P) : hgu_rows) * I * 4, "hu"},
      {&Qwen35::hhalf_, (P + S) * I * 2, "hhalf"},
      {&Qwen35::yd_, (Pm + S) * D * 4, "yd"},
      {&Qwen35::sgb_, Tm * I * 4, "sgb"},
      {&Qwen35::sub_, Tm * I * 4, "sub"},
      {&Qwen35::ys_, Tm * D * 4, "ys"},
      {&Qwen35::rlog_, Tm * std::max(c.experts, 1) * 4, "rlog"},
      {&Qwen35::slog_, Tm * 4 + 16, "slog"},
  };
  v.insert(v.end(), w.begin(), w.end());
  return v;
}

// Rows of the dense FFN's fp32 gate/up buffers (hg_/hu_): the whole chunk for the unfused fallback
// (NI_QMM_GU=0); the Neural Engine split's GPU part takes chunk x its own channels (ane_rebuild grows
// them; the split's startup check and benchmarks use their own scratch); else only the MTP head's
// small-batch rows (enc_mlp_rows). 27B: 2 x 143 MB -> 2 x 0.3 MB without the split.
static bool qmm8_for(const QLinear& L);
// enc_qmm_gu's large-prefill condition: when it holds, enc_qmm_gu takes the pair at every row count
static bool gu_fused(const QLinear& G, const QLinear& U) {
  if (!qmm_gu_enabled() || G.N != U.N || G.K != U.K) return false;
  auto ok4 = [](const QLinear& q) { return q.bits == 4 && q.gs == 64 && q.K % 64 == 0; };
  auto ok8 = [](const QLinear& q) { return q.bits == 8 && q.gs == 64 && q.K % 64 == 0 && qmm8_for(q); };
  // the other formats: wfmt.metal's qrf_gu (enc_qmm_gu)
  auto gen = [](const QLinear& q) { return !q.q4() && !q.q8() && q.K % 64 == 0; };
  return (ok4(G) && ok4(U)) || (ok8(G) && ok8(U)) || (gen(G) && G.same_format(U));
}
size_t Qwen35::dense_hgu_rows() const {
  bool fused = true;
  for (const Layer& L : layers_)
    if (L.mg) fused = fused && gu_fused(L.mg, L.mu);
  if (mtp_loaded_ && mtp_.mg) fused = fused && gu_fused(mtp_.mg, mtp_.mu);  // MTP history passes: up to a chunk
  return fused ? size_t(kMaxVerify) : size_t(tmax_);
}

// Sums the tensors the loader would keep from one safetensors file (256-byte aligned, as it stores them);
// `mtp`: only mtp.* tensors, with 2-D bf16 linears as the 8-bit copies load_mtp makes.
struct TensorMeta {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t bytes = 0;
};
static size_t st_bytes(const std::string& path, bool mtp, double expert_share = -1, size_t* decode = nullptr,
                       size_t* lm_head = nullptr, std::map<std::string, TensorMeta>* metas = nullptr,
                       int mtp_bits = 4) {
  std::ifstream f(path, std::ios::binary);
  uint64_t n = 0;
  if (!f.read(reinterpret_cast<char*>(&n), 8) || n > (256u << 20)) return 0;
  std::string h(n, '\0');
  if (!f.read(h.data(), std::streamsize(n))) return 0;
  json j;
  try {
    j = json::parse(h);
  } catch (...) {
    return 0;
  }
  size_t total = 0;
  for (auto& [name, v] : j.items()) {
    if (name == "__metadata__") continue;
    const bool is_mtp = name.rfind("mtp.", 0) == 0;
    if (mtp != is_mtp || name.find("vision_tower") != std::string::npos || name.find("visual.") != std::string::npos)
      continue;
    size_t b = v["data_offsets"][1].get<size_t>() - v["data_offsets"][0].get<size_t>();
    if (mtp && v["dtype"] == "BF16" && v["shape"].size() == 2 && name.size() > 7 &&
        name.compare(name.size() - 7, 7, ".weight") == 0 && name.find("norm") == std::string::npos) {
      const size_t N = v["shape"][0], K = v["shape"][1];
      b = N * K * size_t(mtp_bits) / 8 + 2 * (N * K / 64) * 2;  // quantized at load: codes + bf16 scales and biases
    }
    total += (b + 255) / 256 * 256;
    if (metas) {  // keyed by the name's tail: embed_tokens.weight, lm_head.scales, mlp.gate_proj.weight (layer 0) ...
      for (const char* k : {"embed_tokens.weight", "embed_tokens.scales", "lm_head.weight", "lm_head.scales"})
        if (name.size() >= strlen(k) && name.compare(name.size() - strlen(k), strlen(k), k) == 0)
          (*metas)[k] = TensorMeta{v["dtype"], v["shape"].get<std::vector<int64_t>>(), b};
      // every layer's dense FFN tensors, keyed from "layers.": layers.<i>.mlp.{gate,up,down}_proj.{weight,scales}
      const size_t lp = name.find("layers.");
      for (const char* k : {".mlp.gate_proj.weight", ".mlp.gate_proj.scales", ".mlp.up_proj.weight",
                            ".mlp.up_proj.scales", ".mlp.down_proj.weight", ".mlp.down_proj.scales"}) {
        const size_t kl = strlen(k);
        if (lp != std::string::npos && name.size() >= kl && name.compare(name.size() - kl, kl, k) == 0)
          (*metas)[name.substr(lp)] = TensorMeta{v["dtype"], v["shape"].get<std::vector<int64_t>>(), b};
      }
      if (name.find("embed_tokens.") != std::string::npos) (*metas)["embed_bytes"].bytes += b;
    }
    if (lm_head && name.find("lm_head.") != std::string::npos) *lm_head += b;
    if (decode && name.find("embed_tokens") == std::string::npos)
      *decode += name.find("switch_mlp") != std::string::npos ? size_t(double(b) * expert_share) : b;
  }
  return total;
}

Qwen35::MemPlan Qwen35::plan(const std::string& dir, int chunk, bool rows16) {
  MemPlan m;
  Qwen35Config c;
  parse_config(read_json(dir + "/config.json"), c);
  std::vector<std::string> files;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      const std::string n = e->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) files.push_back(n);
    }
    closedir(d);
  }
  bool has_mtp = false;
  const double share = c.moe && c.experts ? double(c.top_k) / c.experts : 1.0;
  size_t embed = 0;  // tied models read the embedding table as their output layer
  size_t lm_head = 0;
  std::map<std::string, TensorMeta> metas;
  for (const std::string& f : files) {
    m.weights += st_bytes(dir + "/" + f, false, share, &m.decode_bytes, &lm_head, &metas);
    if (f == "mtp.safetensors" && !getenv("NI_NO_MTP")) {
      m.mtp = st_bytes(dir + "/" + f, true, -1, nullptr, nullptr, nullptr, mtp_head_bits(c.moe));
      has_mtp = m.mtp > 0;
    }
  }
  m.dense = !c.moe;
  m.hidden = c.hidden;
  m.layers = c.layers;
  m.ffn = c.moe ? 0 : c.dense_I;
  // format (bits, group; 16 = bf16) of a linear over the hidden size (embedding, LM head, dense gate projection)
  auto fmt_of = [&](const char* w, const char* sc) -> std::pair<int, int> {
    auto iw = metas.find(w), is = metas.find(sc);
    if (iw == metas.end()) return {0, 0};
    if (is == metas.end()) return {16, 0};  // unquantized
    const int64_t words = iw->second.shape.back(), groups = is->second.shape.back();
    if (iw->second.dtype != "U32" || groups <= 0) return {0, 0};
    return {int(words * 32 / c.hidden), int(c.hidden / groups)};
  };
  const json r = read_json(dir + "/config.json");
  const bool tied = [&] {
    const json t = r.contains("text_config") ? r["text_config"] : r;
    return t.value("tie_word_embeddings", r.value("tie_word_embeddings", false)) || !lm_head;
  }();
  if (tied) embed = metas["embed_bytes"].bytes;  // the output layer reads the whole table every token
  m.decode_bytes += embed;
  // the draft head's rows (DFlash2): the output layer's own, or an 8-bit group-64 copy for formats without the
  // drafter's 4/8-bit kernels (DFlash2::build_head)
  const auto hf = tied ? fmt_of("embed_tokens.weight", "embed_tokens.scales") : fmt_of("lm_head.weight", "lm_head.scales");
  const bool head_native = (hf.first == 4 || hf.first == 8) && hf.second == 64;
  m.head_row_bytes = head_native ? (lm_head ? lm_head : embed) / std::max(c.vocab, 1)
                                 : size_t(c.hidden) + size_t(c.hidden / 64) * 4;
  // what the weights are, for messages: the default format and how many modules the config sets otherwise
  {
    const json q = r.contains("quantization") ? r["quantization"] : json::object();
    if (q.empty()) {
      m.formats = "bf16";
    } else {
      m.formats = format_name(q.value("bits", 4), q.value("group_size", 64));
      std::map<std::string, int> other;
      for (auto& [k, v] : q.items()) {
        if (!v.is_object()) continue;
        const std::string mode = v.value("mode", std::string());
        if (mode.rfind("iq", 0) == 0) {  // llama.cpp's IQ blocks: IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S
          std::string u = mode;
          for (auto& c : u) c = char(toupper(static_cast<unsigned char>(c)));
          ++other[u];
        } else if (v.value("bits", 4) != q.value("bits", 4) || v.value("group_size", 64) != q.value("group_size", 64)) {
          ++other[format_name(v.value("bits", 4), v.value("group_size", 64))];
        }
      }
      for (auto& [f, n] : other) m.formats += " + " + std::to_string(n) + " modules " + f;
    }
  }
  // dense FFN in a format the Neural Engine split stages its int8 weights from (init_ane checks every layer)
  if (!c.moe) {
    const auto ff = fmt_of("layers.0.mlp.gate_proj.weight", "layers.0.mlp.gate_proj.scales");
    m.ffn_ane = format_supported(ff.first, ff.second);
    // the split's GPU share on the matrix units (enc_mlp_gpu_part_qmt): every layer's gate, up and down 4-bit g64, on
    // a model the M5 share was measured on, as init_ane decides it, so the planned and the loaded default share agree
    m.ffn_mma = qmt_on() && mma_share_measured(c);
    for (int i = 0; i < c.layers && m.ffn_mma; ++i)
      for (const char* t : {"gate_proj", "up_proj", "down_proj"}) {
        const std::string pre = "layers." + std::to_string(i) + ".mlp." + t;
        auto iw = metas.find(pre + ".weight"), is = metas.find(pre + ".scales");
        const int64_t K = t[0] == 'd' ? c.dense_I : c.hidden;  // input width: 4 bits = K / 8 words, g64 = K / 64
        m.ffn_mma = m.ffn_mma && iw != metas.end() && is != metas.end() && iw->second.dtype == "U32" &&
                    iw->second.shape.back() == K / 8 && is->second.shape.back() == K / 64;
      }
  }
  // weights one plain decode step multiplies (the prefill estimate's FLOPs): from the architecture
  {
    const double H = c.hidden, kd = double(c.lin_hk) * c.lin_dk, vd = double(c.lin_hv) * c.lin_dv;
    const double qd = double(c.hq) * c.head_dim, kvd = double(c.hkv) * c.head_dim;
    double p = 0;
    for (int i = 0; i < c.layers; ++i) {
      p += c.is_linear[i] ? H * (2 * kd + vd) + H * vd + 2 * H * c.lin_hv + vd * H : H * 2 * qd + 2 * H * kvd + qd * H;
      p += c.moe ? double(c.top_k) * 3 * H * c.moe_I + 3 * H * c.shared_I + H * c.experts + H : 3 * H * c.dense_I;
    }
    m.decode_params = size_t(p + double(c.vocab) * H);
  }
  // hg_/hu_: the verify rows when the dense FFN takes the fused path, as dense_hgu_rows() (every format fuses
  // gate/up on large prefills while NI_QMM_GU is on; 8-bit ones through qmr8, unless NI_QMM8=0)
  const json q = r.contains("quantization") ? r["quantization"] : json::object();
  const bool fused = qmm_gu_enabled() && (q.value("bits", 4) != 8 || qmm8_on());
  // the verify rows the plan reserves: 8; 16 where the engine drafts a 16-row block (rows16, and the block pins the
  // verify to its width); or 32 where KFlash's wide tree verifies run on this target (kf_dense_rows): the buffers and
  // tape are that tall (set_tree_rows, new_seq), and the drafting pass picks its rows from the same call
  // and a cached suffix's chain pass may run wider than the drafter drafts (vpf_pass_rows): its tape is the tallest
  const uint32_t wide = kf_dense_rows(c);
  const size_t vrows = std::max(rows16 ? size_t(16) : wide > (1u << (kMaxVerify - 1)) ? size_t(kMaxTreeRows) : size_t(kMaxVerify),
                                size_t(vpf_pass_rows(c)));
  for (const ActBuf& a : act_layout(c, chunk, int(vrows), has_mtp, fused ? vrows : size_t(chunk)))
    m.activations += a.bytes;
  const size_t C = 2 * size_t(c.lin_hk) * c.lin_dk + size_t(c.lin_hv) * c.lin_dv;
  int lin = 0, att = 0;
  for (int i = 0; i < c.layers; ++i) (c.is_linear[i] ? lin : att) += 1;
  const size_t state = size_t(lin) * (3 * C * 4 + size_t(c.lin_hv) * c.lin_dv * c.lin_dk * 4);
  const size_t tape = size_t(lin) * vrows * (C + size_t(c.lin_hv) * c.lin_dk + c.lin_hv + size_t(c.lin_hv) * c.lin_dv) * 4;
  m.slot_state = 3 * state + tape;  // live state + the engine's two checkpoints + the verify tape
  m.kv_per_pos = size_t(att + (has_mtp ? 1 : 0)) * 2 * c.hkv * c.head_dim * 2;
  return m;
}

void Qwen35::alloc_state() {
  Gpu& g = Gpu::get();
  for (const ActBuf& a : act_layout(cfg_, tmax_, kMaxVerify, mtp_loaded_, dense_hgu_rows()))
    this->*(a.member) = g.alloc(a.bytes, a.label);
}

// Grows hg_/hu_ to hold `bytes` each (between command buffers: nothing in flight may use them).
void Qwen35::ensure_hgu(size_t bytes) {
  if (hg_.size >= bytes) return;
  Gpu& g = Gpu::get();
  g.release(hg_);
  g.release(hu_);
  hg_ = g.alloc(bytes, "hg");
  hu_ = g.alloc(bytes, "hu");
}

std::unique_ptr<Seq> Qwen35::new_seq(int limit, int initial) {
  Gpu& g = Gpu::get();
  auto s = std::make_unique<Seq>();
  s->limit = limit;
  s->tok = g.alloc(size_t(limit + 8) * 4, "tokens");
  const int C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  size_t bytes = 0;
  for (int i = 0; i < cfg_.layers; ++i) {
    // attention caches: allocated by grow_kv below (or later, by reserve)
    s->kc.push_back(Buffer{});
    s->vc.push_back(Buffer{});
    if (cfg_.is_linear[i] && local(i)) {
      s->conv.push_back(g.alloc(size_t(3) * C * 4, "conv_state"));
      s->rec.push_back(g.alloc(size_t(cfg_.lin_hv) * cfg_.lin_dv * cfg_.lin_dk * 4, "gdn_state"));
      bytes += s->conv.back().size + s->rec.back().size;
      // the verify tape at the rows this model's verifies run (plan() reserves the same): no growth on a request
      const size_t vr = size_t(std::max(kMaxVerify, verify_rows_max()));
      s->pend_qkv.push_back(g.alloc(vr * C * 4, "pend_qkv"));
      s->pend_k.push_back(g.alloc(vr * cfg_.lin_hv * cfg_.lin_dk * 4, "pend_k"));
      s->pend_g.push_back(g.alloc(vr * cfg_.lin_hv * 4, "pend_g"));
      s->pend_d.push_back(g.alloc(vr * cfg_.lin_hv * cfg_.lin_dv * 4, "pend_d"));
    } else {
      s->conv.push_back(Buffer{});
      s->rec.push_back(Buffer{});
      s->pend_qkv.push_back(Buffer{});
      s->pend_k.push_back(Buffer{});
      s->pend_g.push_back(Buffer{});
      s->pend_d.push_back(Buffer{});
    }
  }
  const int cap = initial < 0 ? limit : std::min(initial, limit);
  if (cap > 0) grow_kv(*s, cap);
  logd("sequence: recurrent state %.0f MB, KV for %d of %d tokens (%.0f MB)", bytes / 1e6, s->cap, limit,
       double(kv_bytes_per_pos()) * s->cap / 1e6);
  reset(*s);
  return s;
}

size_t Qwen35::kv_bytes_per_pos() const {
  int att = 0;
  for (int i = lo_; i < hi_; ++i) att += !cfg_.is_linear[i];
  return size_t(att + (mtp_loaded_ ? 1 : 0)) * 2 * cfg_.hkv * cfg_.head_dim * 2;
}

// Copies positions [0, old_cap) of every kv head of a [Hkv, old_cap, D] cache into a [Hkv, new_cap, D] one.
void Qwen35::enc_kv_regrow(Encoder& e, const Buffer& src, const Buffer& dst, int old_cap, int new_cap) {
  const uint32_t n16 = uint32_t(cfg_.head_dim) * 2 / 16;  // 16-byte words per position
  const uint32_t p[4] = {uint32_t(old_cap) * n16, uint32_t(new_cap) * n16, uint32_t(cfg_.hkv), 0};  // uint3: 16 bytes
  e.use(pipe("kv_regrow"));
  e.buf(0, src);
  e.buf(1, dst);
  e.bytes(2, p, sizeof p);
  e.dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
}

void Qwen35::grow_kv(Seq& s, int cap) {
  Gpu& g = Gpu::get();
  const int old = s.cap;
  std::vector<std::pair<Buffer*, const char*>> bufs;  // every attention cache of the sequence
  for (int i = lo_; i < hi_; ++i)
    if (!cfg_.is_linear[i]) {
      bufs.push_back({&s.kc[i], "k_cache"});
      bufs.push_back({&s.vc[i], "v_cache"});
    }
  if (mtp_loaded_) {
    bufs.push_back({&s.mtp_kc, "mtp_k_cache"});
    bufs.push_back({&s.mtp_vc, "mtp_v_cache"});
  }
  const size_t nb = size_t(cfg_.hkv) * cap * cfg_.head_dim * 2;
  // In batches of ~256 MB: a batch's new buffers are filled from the old ones, then the old ones are
  // released, so growing never holds two copies of the whole cache.
  for (size_t i = 0; i < bufs.size();) {
    std::vector<Buffer> olds;
    std::unique_ptr<Encoder> e;
    for (size_t batch = 0; i < bufs.size() && (batch == 0 || batch + nb <= (size_t(256) << 20)); ++i, batch += nb) {
      Buffer& b = *bufs[i].first;
      Buffer nbuf = g.alloc(nb, bufs[i].second);
      if (old > 0 && b) {
        if (!e) e = g.begin();
        enc_kv_regrow(*e, b, nbuf, old, cap);
      }
      olds.push_back(b);
      b = nbuf;
    }
    if (e) e->commit_and_wait();
    for (auto& b : olds) g.release(b);
  }
  s.cap = cap;
}

bool Qwen35::reserve(Seq& s, int need) {
  if (need <= s.cap) return true;
  if (need > s.limit) return false;
  const size_t per = kv_bytes_per_pos();
  auto round_up = [](int v, int m) { return (v + m - 1) / m * m; };
  auto fits = [&](int c) {
    // final size; the batched copy holds a batch's old buffers a moment longer (at most ~256 MB, and at most the old
    // cache: nothing when the cache starts empty)
    const size_t copy = std::min(size_t(256) << 20, size_t(s.cap) * per);
    return !mem_budget_ || Gpu::get().allocated_bytes() + size_t(c - s.cap) * per + copy <= mem_budget_;
  };
  // at least a quarter more (fewer copies while a conversation fills), in 1024-position steps
  int target = std::min(s.limit, round_up(std::max(need, s.cap + s.cap / 4), 1024));
  if (!fits(target)) target = std::min(s.limit, round_up(need, 256));
  if (!fits(target)) return false;
  const double t0 = now_ms();
  const int old = s.cap;
  grow_kv(s, target);
  logd("KV cache %d -> %d positions (%.0f MB) in %.1f ms", old, target, double(per) * target / 1e6, now_ms() - t0);
  return true;
}

void Qwen35::drop_kv(Seq& s) {
  Gpu& g = Gpu::get();
  for (auto* v : {&s.kc, &s.vc})
    for (auto& b : *v) g.release(b);
  g.release(s.mtp_kc);
  g.release(s.mtp_vc);
  s.cap = 0;
}

int Qwen35::rope_positions(const std::vector<vision::ImageSpan>& spans, std::vector<int32_t>& P) {
  P.clear();
  int row = 0, cur = 0;  // next row, its position
  for (const vision::ImageSpan& sp : spans) {
    if (sp.pos < row || sp.n != sp.grid_t * sp.grid_h * sp.grid_w) fatal("rope_positions: bad image span");
    for (; row < sp.pos; ++row, ++cur) P.insert(P.end(), {cur, cur, cur});
    for (int f = 0; f < sp.grid_t; ++f)
      for (int y = 0; y < sp.grid_h; ++y)
        for (int x = 0; x < sp.grid_w; ++x, ++row) P.insert(P.end(), {cur + f, cur + y, cur + x});
    cur += std::max(sp.grid_h, sp.grid_w);
  }
  return cur - row;
}

void Qwen35::set_rope_positions(Seq& s, int n, const std::vector<vision::ImageSpan>& spans) {
  s.rope_n = 0;
  s.rope_delta = 0;
  if (spans.empty()) return;
  std::vector<int32_t> table;
  const int delta = rope_positions(spans, table);
  const int end = int(table.size() / 3);  // rows before the last image's end get explicit positions
  if (end > n || end > s.limit) fatal("set_rope_positions: images past the prompt");
  if (!s.rope_pos || s.rope_pos.size < size_t(end) * 12) {
    Gpu::get().release(s.rope_pos);
    s.rope_pos = Gpu::get().alloc(size_t((end + 4095) / 4096 * 4096) * 12, "rope_positions");
  }
  memcpy(s.rope_pos.ptr(), table.data(), table.size() * 4);
  s.rope_n = end;
  s.rope_delta = delta;
}

// attn_prep_m's buffers 9 and 10 (after attn_prep's): the sequence's M-RoPE positions
void Qwen35::enc_mrope_args(Encoder& e, const Seq& s) {
  struct {
    uint32_t mn;
    int32_t delta;
    uint32_t sec1, sec2;
  } m{uint32_t(s.rope_n), s.rope_delta, uint32_t(3 * cfg_.mrope[1]), uint32_t(3 * cfg_.mrope[2])};
  e.buf(9, s.rope_pos ? s.rope_pos : qg_);
  e.val(10, m);
}

void Qwen35::reset(Seq& s) {
  s.n_pend = 0;
  s.pend_tree = false;
  for (auto& b : s.conv)
    if (b) memset(b.ptr(), 0, b.size);
  for (auto& b : s.rec)
    if (b) memset(b.ptr(), 0, b.size);
}

std::vector<float> Qwen35::final_norm_weights() const {
  if (!final_norm_) fatal("final norm weights are missing");
  const size_t n = size_t(cfg_.hidden);
  std::vector<float> out(n);
  if (final_norm_->dtype == DType::BF16) {
    const uint16_t* w = final_norm_->buf.as<uint16_t>();
    for (size_t i = 0; i < n; ++i) out[i] = bf16_to_f32(w[i]);
  } else if (final_norm_->dtype == DType::F32) {
    const float* w = final_norm_->buf.as<float>();
    for (size_t i = 0; i < n; ++i) out[i] = w[i];
  } else if (final_norm_->dtype == DType::F16) {
    const _Float16* w = final_norm_->buf.as<_Float16>();
    for (size_t i = 0; i < n; ++i) out[i] = float(w[i]);
  } else {
    fatal("final norm weights have an unsupported dtype");
  }
  return out;
}

Checkpoint Qwen35::new_checkpoint() {
  Gpu& g = Gpu::get();
  Checkpoint c;
  const int C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  for (int i = 0; i < cfg_.layers; ++i) {
    bool lin = cfg_.is_linear[i] && local(i);
    c.conv.push_back(lin ? g.alloc(size_t(3) * C * 4, "ckpt_conv") : Buffer{});
    c.rec.push_back(lin ? g.alloc(size_t(cfg_.lin_hv) * cfg_.lin_dv * cfg_.lin_dk * 4, "ckpt_rec") : Buffer{});
  }
  return c;
}

void Qwen35::enc_copy(Encoder& e, const Buffer& src, const Buffer& dst) {
  static Pipeline cp = Gpu::get().pipeline("copy_u32");
  uint32_t n = uint32_t(std::min(src.size, dst.size) / 16);
  e.use(cp);
  e.buf(0, src);
  e.buf(1, dst);
  e.val(2, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
}

void Qwen35::save(Encoder& e, const Seq& s, Checkpoint& c, int pos) {
  for (int i = lo_; i < hi_; ++i)
    if (cfg_.is_linear[i]) {
      enc_copy(e, s.conv[i], c.conv[i]);
      enc_copy(e, s.rec[i], c.rec[i]);
    }
  c.pos = pos;
}

void Qwen35::restore(Encoder& e, Seq& s, const Checkpoint& c) {
  for (int i = lo_; i < hi_; ++i)
    if (cfg_.is_linear[i]) {
      enc_copy(e, c.conv[i], s.conv[i]);
      enc_copy(e, c.rec[i], s.rec[i]);
    }
}

std::string Qwen35::disk_format() const {
  std::string kinds;
  for (int i = 0; i < cfg_.layers; ++i) kinds += cfg_.is_linear[i] ? 'G' : 'A';
  char b[256];
  snprintf(b, sizeof b, "qwen35 %s; attention: K, V f16 [%d][cap][%d]%s; GDN: conv f32 [3][%d], state f32 [%d][%d][%d]",
           kinds.c_str(), cfg_.hkv, cfg_.head_dim, mtp_loaded_ ? " (MTP head too)" : "",
           2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv, cfg_.lin_hv, cfg_.lin_dv, cfg_.lin_dk);
  return b;
}

void Qwen35::disk_rows(const Seq& s, bool mtp, std::vector<KvRows>& out) const {
  const size_t row = size_t(cfg_.head_dim) * 2;
  for (int i = 0; i < cfg_.layers; ++i)
    if (!cfg_.is_linear[i]) {
      out.push_back({s.kc[size_t(i)], cfg_.hkv, row});
      out.push_back({s.vc[size_t(i)], cfg_.hkv, row});
    }
  if (mtp && mtp_loaded_) {
    out.push_back({s.mtp_kc, cfg_.hkv, row});
    out.push_back({s.mtp_vc, cfg_.hkv, row});
  }
}

void Qwen35::disk_state(const Seq& s, const Checkpoint* c, std::vector<Buffer>& out) const {
  for (int i = 0; i < cfg_.layers; ++i)
    if (cfg_.is_linear[i])
      out.push_back(c ? c->conv[size_t(i)] : s.conv[size_t(i)]), out.push_back(c ? c->rec[size_t(i)] : s.rec[size_t(i)]);
}

bool Qwen35::copy_prefix(Encoder& e, const Seq& src, Seq& dst, int n) {
  if (n <= 0 || n > src.cap || n > dst.cap) return false;
  const uint32_t n16 = uint32_t(cfg_.head_dim) * 2 / 16;  // 16-byte words per position
  const uint32_t p[4] = {uint32_t(n) * n16, uint32_t(src.cap) * n16, uint32_t(dst.cap) * n16, 0};  // uint3: 16 bytes
  auto copy = [&](const Buffer& a, const Buffer& b) {
    e.use(pipe("kv_copy_prefix"));
    e.buf(0, a);
    e.buf(1, b);
    e.bytes(2, p, sizeof p);
    e.dispatch((p[0] + 255) / 256, uint32_t(cfg_.hkv), 1, 256);
  };
  for (int i = 0; i < cfg_.layers; ++i)
    if (!cfg_.is_linear[i]) {
      copy(src.kc[i], dst.kc[i]);
      copy(src.vc[i], dst.vc[i]);
    }
  if (mtp_loaded_) {
    copy(src.mtp_kc, dst.mtp_kc);
    copy(src.mtp_vc, dst.mtp_vc);
  }
  return true;
}

// ---- weight formats ----
// Per-row MoE kernels (decode, verify, MTP rows) for routed experts e and shared expert s: the specialized
// 4-bit ones, else the generic instantiation of wfmt.metal for the pair, or "" if there is none.
std::string moe_rows_kernel(bool gate_up, const QLinear& e, const QLinear& s, bool wide) {
  // wide: the down kernel's staged h exceeds 4608 floats (moe_wide): the _w forms, which exist for these pairs
  if (wide && !gate_up) {
    if (e.q4() && s.q4()) return "moe_down_rows_g64_w";
    const std::string pair = e.fmt() + "_" + s.fmt();
    return pair == "f4g64_f8g64" || pair == "f3g64_f8g64" ? "moeg_down_rows_" + pair + "_w" : "";
  }
  if (e.q4() && s.q4()) return gate_up ? "moe_gate_up_rows_g64" : "moe_down_rows_g64";
  if (e.bits == 4 && e.gs == 32 && s.q8()) return gate_up ? "moe_gate_up_rows_g32_s8" : "moe_down_rows_g32_s8";
  static const std::set<std::string> have = {"f16_f16",     "f8g64_f8g64", "f4g64_f8g64", "f6g64_f6g64", "f6g64_f8g64",
                                             "f5g64_f5g64", "f5g64_f8g64", "f3g64_f8g64", "f4g64_f4g64"};
  const std::string pair = e.fmt() + "_" + s.fmt();
  if (!have.count(pair)) return "";
  return (gate_up ? "moeg_gu_rows_" : "moeg_down_rows_") + pair;
}
// The routed experts' prefill GEMMs: qrm (4-bit group 64) or the generic qrmf / qmmsf_moe forms.
static bool moe_prefill_format(const QLinear& q) {
  if (q.bits == 4) return q.gs % 32 == 0;
  return q.gs == 64 && (q.bits == 3 || q.bits == 5 || q.bits == 6 || q.bits == 8) || q.dense();
}

// ---- test transforms (NI_DEQUANT, NI_REQUANT): other formats from a 4-bit model, same function ----
// NI_DEQUANT=<spec>: the selected 4/8-bit (and 4-bit group 32) linears become bf16 (dequantized: w = s q + b rounded
// to bf16), so the bf16 kernels run on real shapes and the outputs must match the quantized model's to rounding.
// NI_REQUANT=<bits>[:<spec>] (bits 5, 6 or 8): the selected 4-bit group-64 linears re-encoded exactly as <bits>-bit
// group 64 (codes q << (bits - 4), scales / 2^(bits - 4)): the same weights, so outputs match the 4-bit model's to
// rounding, and the kernels run at the speed of a real <bits>-bit model. Both keep the original buffers too
// (memory: the 4-bit file plus the new copies), so whole large models do not fit.
// spec: comma list of all | embed | head | attn | gdn | router | experts | shared | mlp, optionally after a layer
// range "a-b:" (e.g. "0-7:mlp,attn"); default all.
static bool spec_match(const std::string& spec, const char* role, int layer) {
  std::string body = spec;
  const size_t colon = spec.find(':');
  if (colon != std::string::npos) {
    int a = 0, b = 1 << 30;
    if (sscanf(spec.c_str(), "%d-%d", &a, &b) < 1) fatal("bad layer range in %s", spec.c_str());
    if (layer >= 0 && (layer < a || layer > b)) return false;
    if (layer < 0) return false;
    body = spec.substr(colon + 1);
  }
  size_t i = 0;
  while (i <= body.size()) {
    const size_t j = std::min(body.find(',', i), body.size());
    const std::string t = body.substr(i, j - i);
    if (t == "all" || t == role) return true;
    i = j + 1;
  }
  return false;
}

// role ("embed", "head", "gdn", "attn", "router", "experts", "shared", "mlp") and layer (-1 outside the layers) of a
// projection tensor (.weight / .scales / .biases of a linear); false for every other tensor
static bool transform_role(const std::string& n, std::string& role, int& layer) {
  const auto ends = [&](const char* s) { return n.size() >= strlen(s) && n.compare(n.size() - strlen(s), strlen(s), s) == 0; };
  if (!ends(".weight") && !ends(".scales") && !ends(".biases")) return false;
  const auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
  layer = -1;
  if (const size_t p = n.find(".layers."); p != std::string::npos) layer = atoi(n.c_str() + p + 8);
  if (has("embed_tokens.")) role = "embed";
  else if (has("lm_head.")) role = "head";
  else if (has(".linear_attn.in_proj_") || has(".linear_attn.out_proj.")) role = "gdn";
  else if (has(".self_attn.q_proj.") || has(".self_attn.k_proj.") || has(".self_attn.v_proj.") || has(".self_attn.o_proj."))
    role = "attn";
  else if (has(".mlp.gate.") || has(".mlp.shared_expert_gate.")) role = "router";
  else if (has(".mlp.switch_mlp.")) role = "experts";
  else if (has(".mlp.shared_expert.")) role = "shared";
  else if (has(".mlp.gate_proj.") || has(".mlp.up_proj.") || has(".mlp.down_proj.")) role = "mlp";
  else return false;
  return true;
}

WeightStore::Filter Qwen35::transform_defer() const {
  const char* dq = getenv("NI_DEQUANT");
  const char* rq = getenv("NI_REQUANT");
  if (!(dq && *dq) && !(rq && *rq)) return nullptr;
  std::string dspec = dq && *dq ? dq : "", rspec;
  if (rq && *rq) rspec = strchr(rq, ':') ? strchr(rq, ':') + 1 : "all";
  return [dspec, rspec](const std::string& n) {
    std::string role;
    int layer = -1;
    if (!transform_role(n, role, layer)) return false;
    return (!dspec.empty() && spec_match(dspec, role.c_str(), layer)) ||
           (!rspec.empty() && spec_match(rspec, role.c_str(), layer));
  };
}

void Qwen35::test_transforms() {
  const char* dq = getenv("NI_DEQUANT");
  const char* rq = getenv("NI_REQUANT");
  if (!(dq && *dq) && !(rq && *rq)) return;
  int rbits = 0, rgs = 64;
  std::string rspec = "all";
  if (rq && *rq) {
    rbits = atoi(rq);
    if (!strncmp(rq, "4g32", 4)) rgs = 32;
    if (rbits != 5 && rbits != 6 && rbits != 8 && !(rbits == 4 && rgs == 32))
      fatal("NI_REQUANT=%s: 5, 6, 8 or 4g32", rq);
    if (const char* c = strchr(rq, ':')) rspec = c + 1;
  }
  const std::string dspec = dq && *dq ? dq : "";
  Gpu& g = Gpu::get();
  size_t nd = 0, nr = 0, bytes = 0;
  // tensors the loader deferred (transform_defer): their names, and their bytes on demand in temporary buffers
  std::map<const Tensor*, std::string> names;
  for (auto& [n, t] : ws_.all())
    if (ws_.deferred(n)) names[&t] = n;
  std::vector<Buffer> temps;
  auto src = [&](const Tensor* t) -> Buffer {
    auto it = names.find(t);
    if (it == names.end()) return t->buf;
    Buffer b = g.alloc(size_t(t->numel()) * dtype_size(t->dtype) + 16, "deferred", false);
    ws_.read_deferred(it->second, b.ptr());
    temps.push_back(b);
    return b;
  };
  auto drop_temps = [&] {
    for (Buffer& b : temps) g.release(b);
    temps.clear();
  };
  auto dequant = [&](QLinear& q) {
    if (!q || q.dense()) return;
    if (!(q.q4() || q.q8() || (q.bits == 4 && q.gs == 32))) fatal("NI_DEQUANT: %s weights", format_name(q.bits, q.gs).c_str());
    const size_t rows = q.w->numel() / q.w->shape.back();  // N, or E * N for stacked experts
    Buffer out = g.alloc(rows * q.K * 2 + 16, "dequant_bf16", false);
    const Buffer sw = src(q.w), ss = src(q.s), sb = src(q.b);
    auto e = g.begin();
    e->use(pipe("wf_dequant_" + q.fmt()));
    e->buf(0, sw); e->buf(1, ss); e->buf(2, sb); e->buf(3, out);
    const uint32_t nk[2] = {uint32_t(rows), q.K};
    e->bytes(4, nk, sizeof nk);
    const uint32_t n = uint32_t(rows) * (q.K / 16);
    e->dispatch((n + 255) / 256, 1, 1, 256);
    e->commit_and_wait();
    drop_temps();
    owned_bufs_.push_back(out);
    Tensor t;
    t.buf = out.view(0, rows * q.K * 2);
    t.dtype = DType::BF16;
    t.shape = q.w->shape;
    t.shape.back() = q.K;
    owned_.push_back(t);
    q.w = q.s = q.b = &owned_.back();
    q.bits = 16;
    q.gs = 64;
    ++nd;
    bytes += out.size;
  };
  auto requant = [&](QLinear& q) {
    if (!q || !q.q4()) return;
    const size_t rows = q.w->numel() / q.w->shape.back();
    if (rgs == 32) {  // same codes, each 64-group's scale and bias written for its two 32-groups
      const uint32_t g64 = q.K / 64;
      Buffer sc = g.alloc(rows * g64 * 4 + 16, "requant_s32", false), bi = g.alloc(rows * g64 * 4 + 16, "requant_b32", false);
      const Buffer ss = src(q.s), sb = src(q.b);
      for (size_t i = 0; i < rows * g64; ++i)
        for (int h = 0; h < 2; ++h) {
          sc.as<uint16_t>()[2 * i + h] = ss.as<uint16_t>()[i];
          bi.as<uint16_t>()[2 * i + h] = sb.as<uint16_t>()[i];
        }
      drop_temps();
      if (names.count(q.w)) {  // the codes are kept: an owned copy of the deferred weight
        Buffer w = src(q.w);
        temps.clear();  // kept, not dropped
        owned_bufs_.push_back(w);
        Tensor tw = *q.w;
        tw.buf = w.view(0, size_t(q.w->numel()) * 4);
        owned_.push_back(tw);
        q.w = &owned_.back();
      }
      owned_bufs_.push_back(sc);
      owned_bufs_.push_back(bi);
      Tensor ts = *q.s, tb = *q.b;
      ts.buf = sc.view(0, rows * g64 * 4);
      tb.buf = bi.view(0, rows * g64 * 4);
      ts.shape.back() *= 2;
      tb.shape.back() *= 2;
      owned_.push_back(ts);
      q.s = &owned_.back();
      owned_.push_back(tb);
      q.b = &owned_.back();
      q.gs = 32;
      ++nr;
      bytes += sc.size + bi.size;
      return;
    }
    const uint32_t K = q.K, gpr = K / 64, sh = uint32_t(rbits - 4);
    const size_t wpr = size_t(K) * rbits / 32;
    Buffer w = g.alloc(rows * wpr * 4 + 16, "requant_w", false), sc = g.alloc(rows * gpr * 2, "requant_s", false);
    const Buffer sw_buf = src(q.w), ss_buf = src(q.s), sb_buf = src(q.b);
    const uint32_t* src = sw_buf.as<uint32_t>();
    const uint16_t* ss = ss_buf.as<uint16_t>();
    uint32_t* dst = w.as<uint32_t>();
    uint16_t* ds = sc.as<uint16_t>();
    auto work = [&](size_t r0, size_t r1) {
      for (size_t r = r0; r < r1; ++r) {
        uint32_t* d = dst + r * wpr;
        std::fill(d, d + wpr, 0u);
        for (uint32_t k = 0; k < K; ++k) {
          const uint32_t c = ((src[r * (K / 8) + k / 8] >> (4 * (k % 8))) & 15u) << sh;
          const uint64_t bit = uint64_t(k) * rbits;
          d[bit / 32] |= c << (bit % 32);
          if (bit % 32 + rbits > 32) d[bit / 32 + 1] |= c >> (32 - bit % 32);
        }
        for (uint32_t gi = 0; gi < gpr; ++gi) ds[r * gpr + gi] = f32_to_bf16(bf16_to_f32(ss[r * gpr + gi]) / float(1u << sh));
      }
    };
    const unsigned nth = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    const size_t per = (rows + nth - 1) / nth;
    for (unsigned t = 0; t < nth; ++t)
      if (t * per < rows) th.emplace_back(work, t * per, std::min(rows, (t + 1) * per));
    for (auto& t : th) t.join();
    owned_bufs_.push_back(w);
    owned_bufs_.push_back(sc);
    if (names.count(q.b)) {  // the biases are kept: an owned copy of the deferred one
      temps.erase(std::find_if(temps.begin(), temps.end(), [&](const Buffer& b) { return b.mtl == sb_buf.mtl; }));
      owned_bufs_.push_back(sb_buf);
      Tensor tb = *q.b;
      tb.buf = sb_buf.view(0, size_t(q.b->numel()) * 2);
      owned_.push_back(tb);
      q.b = &owned_.back();
    }
    drop_temps();
    Tensor tw = *q.w, ts = *q.s;
    tw.buf = w.view(0, rows * wpr * 4);
    tw.shape.back() = int64_t(wpr);
    ts.buf = sc;
    owned_.push_back(tw);
    q.w = &owned_.back();
    owned_.push_back(ts);
    q.s = &owned_.back();
    q.bits = rbits;
    ++nr;
    bytes += w.size + sc.size;
  };
  auto apply = [&](QLinear& q, const char* role, int layer) {
    if (!q) return;  // a layer another Mac holds (a split model)
    if (!dspec.empty() && spec_match(dspec, role, layer)) dequant(q);
    else if (rbits && spec_match(rspec, role, layer)) requant(q);
  };
  const bool tied = lm_head_.w == embed_.w;
  apply(embed_, "embed", -1);
  if (tied) lm_head_ = embed_;
  else apply(lm_head_, "head", -1);
  for (int i = 0; i < cfg_.layers; ++i) {
    Layer& L = layers_[i];
    for (QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out}) apply(*q, "gdn", i);
    for (QLinear* q : {&L.q, &L.k, &L.v, &L.o}) apply(*q, "attn", i);
    for (QLinear* q : {&L.router, &L.shared_gate}) apply(*q, "router", i);
    for (QLinear* q : {&L.eg, &L.eu, &L.ed})
      if (!q->iq) apply(*q, "experts", i);
    for (QLinear* q : {&L.sg, &L.su, &L.sd}) apply(*q, "shared", i);
    for (QLinear* q : {&L.mg, &L.mu, &L.md}) apply(*q, "mlp", i);
  }
  // deferred tensors no transform converted (a format it leaves alone): read as they are
  size_t kept = 0;
  auto keep = [&](const Tensor*& t) {
    if (!t || !names.count(t)) return;
    Buffer b = src(t);
    temps.clear();
    owned_bufs_.push_back(b);
    Tensor c = *t;
    c.buf = b.view(0, size_t(t->numel()) * dtype_size(t->dtype));
    owned_.push_back(c);
    t = &owned_.back();
    ++kept;
  };
  auto keep_q = [&](QLinear& q) {
    if (!q) return;
    const bool d = q.dense();
    keep(q.w);
    if (d) q.s = q.b = q.w;
    else keep(q.s), keep(q.b);
  };
  keep_q(embed_);
  if (tied) lm_head_ = embed_;
  else keep_q(lm_head_);
  for (Layer& L : layers_)
    for (QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out, &L.q, &L.k, &L.v, &L.o, &L.router, &L.shared_gate, &L.eg, &L.eu,
                       &L.ed, &L.sg, &L.su, &L.sd, &L.mg, &L.mu, &L.md})
      keep_q(*q);
  logi("test transforms: %zu linears dequantized to bf16 (NI_DEQUANT=%s), %zu re-encoded as %d-bit (NI_REQUANT=%s), "
       "%zu deferred tensors kept as they were: %.2f GB in new buffers", nd, dq ? dq : "", nr, rbits, rq ? rq : "", kept,
       bytes / 1e9);
}

void Qwen35::check_formats() const {
  if (!cfg_.moe) return;
  auto all = [&](const Layer& L, const char* what) {
    if (!L.eg) return;
    if (L.eg.q20 || L.eu.q20 || L.ed.q20) {  // Q2_0 (src/model/moe_q20.h)
      if (!(L.eg.q20 && L.eu.q20 && L.ed.q20)) fatal("%s: experts' gate/up/down differ in format", what);
      if (!(L.sg.q4() && L.su.q4() && L.sd.q4())) fatal("%s: Q2_0 experts need a 4-bit group-64 shared expert", what);
      std::string why;
      if (!q20::shapes_ok(q20_shape(), &why)) fatal("%s: Q2_0 experts: %s", what, why.c_str());
      return;
    }
    if (L.eg.iq || L.eu.iq || L.ed.iq) {  // llama.cpp's IQ blocks (qwen35_iq.cpp)
      const std::string why = iq_layer_check(L);
      if (!why.empty()) fatal("%s: IQ experts (%s) need %s", what, L.eg.name().c_str(), why.c_str());
      return;
    }
    if (!L.eg.same_format(L.eu) || !L.eg.same_format(L.ed) || !L.sg.same_format(L.su))
      fatal("%s: experts' gate/up/down (or the shared expert's gate/up) differ in format", what);
    if (!moe_prefill_format(L.eg))
      fatal("%s: %s routed experts are not supported", what, format_name(L.eg.bits, L.eg.gs).c_str());
    if (moe_rows_kernel(true, L.eg, L.sg).empty() || moe_rows_kernel(false, L.ed, L.sd, moe_wide()).empty())
      fatal("%s: no MoE kernels for %s experts with a %s shared expert", what, format_name(L.eg.bits, L.eg.gs).c_str(),
            format_name(L.sg.bits, L.sg.gs).c_str());
  };
  for (size_t i = 0; i < layers_.size(); ++i) all(layers_[i], ("layer " + std::to_string(i)).c_str());
}

std::string Qwen35::formats_summary() const {
  std::map<std::string, size_t> bytes;  // format -> weight bytes
  auto add = [&](const QLinear& q) {
    if (q) bytes[q.name()] += q.w->buf.size + (q.dense() || q.iq ? 0 : q.q20 ? q.s->buf.size : q.s->buf.size + q.b->buf.size);
  };
  add(embed_);
  if (lm_head_.w != embed_.w) add(lm_head_);
  for (const Layer& L : layers_)
    for (const QLinear* q : {&L.qkv, &L.z, &L.a, &L.b, &L.out, &L.q, &L.k, &L.v, &L.o, &L.router, &L.shared_gate,
                             &L.eg, &L.eu, &L.ed, &L.sg, &L.su, &L.sd, &L.mg, &L.mu, &L.md})
      add(*q);
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

void Qwen35::encode_logits_rows(Encoder& e, int T, const Buffer& out) {
  enc_rows_half(e, x_, final_norm_, xh_, T, cfg_.hidden);
  enc_qmm(e, lm_head_, xh_, out, T, false);
}

void Qwen35::enc_embed(Encoder& e, const Buffer& tokens, int tok_off, int T, const Buffer& out) {
  const uint32_t D = uint32_t(cfg_.hidden);
  kp::Embed ep{D, uint32_t(embed_.gs), uint32_t(tok_off)};
  e.use(pipe(embed_.q4() ? std::string("embed_q4") : "embed_" + embed_.fmt()));
  e.buf(0, embed_.w->buf); e.buf(1, embed_.s->buf); e.buf(2, embed_.b->buf); e.buf(3, tokens); e.buf(4, out);
  e.val(5, ep);
  e.dispatch((D / 8 + 63) / 64, uint32_t(T), 1, 64);
}

// ---- encoders ----
static constexpr uint32_t kRows = 4, kSg = 2; // qmv: rows per simdgroup, simdgroups per threadgroup

void Qwen35::enc_qmv(Encoder& e, const QLinear& L, const Buffer& x, const Buffer& y, const Buffer* resid,
                     const Tensor* norm) {
  // a folded norm on qmvf (the LM head after the final norm); plain single matrices keep qmv2 / qmv3 (bench_qmvf:
  // as fast or faster without a norm to fold)
  if (norm && !resid && enc_qmvf(e, {{&L, y}}, x, 1, norm)) return;
  // Tuned on M4 Pro: huge matrices (lm_head) like 4 rows x 4 simdgroups, the rest 2 rows x 8.
  const bool huge = L.N >= 65536;
  uint32_t rows = huge ? 4 : 2, nsg = huge ? 4 : 8;
  std::string name;
  if (L.q4() || (L.q8() && norm && L.K <= 4096)) {
    name = "qmv2_b" + std::to_string(L.bits) + "_g64_r" + std::to_string(rows);
    if (L.K > 4096) {  // wide inputs: x staged in 4096-wide tiles
      if (L.bits == 8 || resid) rows = 2;
      name = "qmv3_b" + std::to_string(L.bits) + "_r" + std::to_string(rows);
    }
  } else if (L.q8()) {  // the 8-bit forms the tiled kernel has for any K: plain, _add, _norm
    rows = 2;
    name = "qmv3_b8_r2";
  } else {  // wfmt.metal: any format, any K; rows 2 (also with a residual) or 4 (huge, no residual)
    if (resid || (!huge && !norm)) rows = 2;
    name = "qmvg_" + L.fmt() + "_r" + std::to_string(rows);
  }
  if (resid) name += "_add";
  if (norm) name += "_norm";
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, Gpu::get().pipeline(name)).first;
  e.use(it->second);
  kp::Qmv p{L.N, L.K, L.K, L.N, cfg_.eps};
  e.buf(0, L.w->buf);
  e.buf(1, L.s->buf);
  e.buf(2, L.b->buf);
  e.buf(3, x);
  e.buf(4, y);
  e.val(5, p);
  e.buf(6, resid ? *resid : y);
  e.buf(7, norm ? norm->buf : y);
  e.dispatch((L.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

void Qwen35::enc_qmv_multi(Encoder& e, std::vector<Seg> segs, const Buffer& x, const Tensor* norm,
                           const Buffer* xn_out) {
  if (enc_qmvf(e, segs, x, 1, norm, xn_out)) return;
  // segments in other formats than 4/8-bit group 64: one generic dispatch per format (the first writes xn_out)
  bool special = true;
  for (auto& sg : segs) special = special && (sg.L->q4() || sg.L->q8());
  if (!special) {
    std::vector<Seg> rest = segs;
    bool first = true;
    while (!rest.empty()) {
      std::vector<Seg> same, other;
      const QLinear& f0 = *rest[0].L;
      const bool sp0 = f0.q4() || f0.q8();
      for (auto& sg : rest)
        ((sp0 ? (sg.L->q4() || sg.L->q8()) : sg.L->same_format(f0)) ? same : other).push_back(sg);
      if (sp0) {
        enc_qmv_multi(e, same, x, norm, first ? xn_out : nullptr);
      } else if (!enc_qmvf(e, same, x, 1, norm, first ? xn_out : nullptr, {}, 0, 0, false, true)) {  // qmvf_b6
        constexpr uint32_t rows = 2, nsg = 8;
        e.use(pipe("qmvg_multi_" + f0.fmt()));
        kp::QmvMulti2 p{};
        p.K = same[0].L->K;
        p.nseg = uint32_t(same.size());
        p.eps = cfg_.eps;
        p.norm = norm ? 1u : 0u;
        p.write_xn = first && xn_out ? 1u : 0u;
        uint32_t tg = 0;
        for (size_t i = 0; i < 4; ++i) {
          const Seg& sg = same[std::min(i, same.size() - 1)];
          if (i < same.size()) {
            if (sg.L->K != p.K) fatal("qmv_multi: segments must share K");
            p.N[i] = sg.L->N;
            p.bits[i] = uint32_t(sg.L->bits);
            p.tg_begin[i] = tg;
            tg += (sg.L->N + rows * nsg - 1) / (rows * nsg);
          }
          e.buf(int(i * 4 + 0), sg.L->w->buf);
          e.buf(int(i * 4 + 1), sg.L->s->buf);
          e.buf(int(i * 4 + 2), sg.L->b->buf);
          e.buf(int(i * 4 + 3), sg.y);
        }
        p.tg_begin[same.size()] = tg;
        e.buf(16, x);
        e.val(17, p);
        e.buf(18, norm ? norm->buf : x);
        e.buf(19, xn_out ? *xn_out : x);
        e.dispatch(tg, 1, 1, 32 * nsg);
      }
      first = false;
      rest = other;
    }
    return;
  }
  static Pipeline pp = Gpu::get().pipeline("qmv2_multi_r2");
  static Pipeline pw = Gpu::get().pipeline("qmv3_multi_r2");
  constexpr uint32_t rows = 2, nsg = 8;
  e.use(segs[0].L->K > 4096 ? pw : pp);
  kp::QmvMulti2 p{};
  p.K = segs[0].L->K;
  p.nseg = uint32_t(segs.size());
  p.eps = cfg_.eps;
  p.norm = norm ? 1u : 0u;
  p.write_xn = xn_out ? 1u : 0u;
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const Seg& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      if (s.L->K != p.K || s.L->gs != 64) fatal("qmv_multi: segments must share K and group 64");
      p.N[i] = s.L->N;
      p.bits[i] = uint32_t(s.L->bits);
      p.tg_begin[i] = tg;
      tg += (s.L->N + rows * nsg - 1) / (rows * nsg);
    }
    e.buf(int(i * 4 + 0), s.L->w->buf);
    e.buf(int(i * 4 + 1), s.L->s->buf);
    e.buf(int(i * 4 + 2), s.L->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  p.tg_begin[segs.size()] = tg;
  e.buf(16, x);
  e.val(17, p);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, xn_out ? *xn_out : x);
  e.dispatch(tg, 1, 1, 32 * nsg);
}

// NI_QMVF (read once; default 1): the decode and 2-row (MTP verify / draft) matvecs that share an input or fold an
// RMSNorm run qmvf.metal (4- and 8-bit group-64 weights); 0 restores the previous kernels (qmv2_multi / qmv3_multi,
// mlp_gate_up, rows_norm + one qmvrc1 / qmvr_multi per matrix, rows_norm + swiglu_f32).
static bool qmvf_on() {
  static const bool v = env_int("NI_QMVF", 1) != 0;
  return v;
}

bool Qwen35::enc_qmvf(Encoder& e, const std::vector<Seg>& segs, const Buffer& x, int R, const Tensor* norm,
                      const Buffer* xn_out, const std::vector<bool>& add, uint32_t x_ld, uint32_t y_ld, bool gu,
                      bool b6_ok) {
  if (!qmvf_on() || R < 1 || R > 2 || segs.empty() || segs.size() > 4 || (gu && segs.size() != 2)) return false;
  const int bits = segs[0].L->bits;
  const uint32_t K = segs[0].L->K;
  // 6-bit group 64 (GGUF imports' Q5_K projections) in IQ models on the M1 (Apple7), the one-row decode only (b6_ok:
  // enc_qmv_multi): the GDN and attention input projections with their norm folded, Qwen3.6-35B-A3B UD-IQ2_XXS on the
  // M1 Pro: 12352 x 2048 (qkv | z) 144 instead of 172 us (qmvg_multi_f6g64, which normalizes x again in every
  // threadgroup), 8192 x 2048 100 / 118. Verify / batched rows keep the X8 GEMMs (two rows here cost 1.75x one: the
  // dequantization is per row), so a single pass stays bit-identical to a batched one; the other 6-bit models keep
  // qmvg_multi (sums reassociated: not the same bits), the other chips until measured there. NI_QMVF6=0 turns it off.
  static const bool b6_on = env_int("NI_QMVF6", 1) != 0;
  const bool b6 = b6_ok && b6_on && bits == 6 && R == 1 && !gu && cfg_.iq && Gpu::get().apple_family() <= 7;
  if (bits != 4 && bits != 8 && !b6) return false;
  // 4-bit group 32 (GGUF imports' Q4_K LM heads), one matrix, on the M1 (Apple7): its 248K-row head on
  // qmvf_b4g32 2.6 instead of 3.1 ms (qmvg_f4g32_r4_norm, Qwen3.6-35B-A3B UD-IQ2_XXS on the M1 Pro); not yet
  // measured on the other chips
  const bool g32 = segs.size() == 1 && !gu && bits == 4 && segs[0].L->gs == 32 && !segs[0].L->iq &&
                   Gpu::get().apple_family() <= 7;
  for (const Seg& s : segs)
    if (!(s.L->q4() || s.L->q8() || g32 || (b6 && s.L->gs == 64 && !s.L->iq)) || s.L->bits != bits || s.L->K != K)
      return false;
  if (K % 64 || (xn_out && !norm)) return false;
  // one threadgroup per token row and 8 weight rows (2 simdgroups x 4 rows: the qmvrc1 layout, its token rows
  // adjacent in the grid so they share each weight block through the cache)
  constexpr int nt = 1, rows = 4, nsg = 2;
  char name[96];
  snprintf(name, sizeof name, "qmvf%s_b%d%s_t%d_r%d%s", gu ? "_gu" : "", bits, g32 ? "g32" : "", nt, rows,
           norm ? "_norm" : "");
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(name);
  if (it == cache.end()) it = cache.emplace(name, Gpu::get().pipeline(name)).first;
  e.use(it->second);
  kp::Qmvf p{};
  p.K = K;
  p.R = uint32_t(R);
  p.x_ld = x_ld ? x_ld : K;
  p.nseg = uint32_t(segs.size());
  p.eps = cfg_.eps;
  p.write_xn = xn_out ? 1u : 0u;
  const uint32_t per_tg = uint32_t(rows * nsg);
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const Seg& s = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = s.L->N;
      p.y_ld[i] = y_ld ? y_ld : s.L->N;
      p.add[i] = i < add.size() && add[i] ? 1u : 0u;
      p.tg_begin[i] = tg;
      if (!gu || i == 0) tg += (s.L->N + per_tg - 1) / per_tg;
    }
    e.buf(int(i * 4 + 0), s.L->w->buf);
    e.buf(int(i * 4 + 1), s.L->s->buf);
    e.buf(int(i * 4 + 2), s.L->b->buf);
    e.buf(int(i * 4 + 3), s.y);
  }
  if (gu) tg = p.tg_begin[1];
  p.tg_begin[segs.size()] = tg;
  e.buf(16, x);
  e.val(17, p);
  e.buf(18, norm ? norm->buf : x);
  e.buf(19, xn_out ? *xn_out : x);
  e.dispatch(uint32_t((R + nt - 1) / nt), tg, 1, 32 * uint32_t(nsg));
  return true;
}

void Qwen35::enc_gdn(Encoder& e, Seq& s, int li, const Layer& L) {
  static Pipeline prep = Gpu::get().pipeline("gdn_prep");
  static Pipeline step = Gpu::get().pipeline("gdn_step");
  enc_qmv_multi(e, {{&L.qkv, qkv_}, {&L.z, zb_}, {&L.b, bb_}, {&L.a, ab_}}, x_, L.in_norm);
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  kp::GdnPrep pp{1, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C};
  e.use(prep);
  e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[li]);
  e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
  e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
  e.val(12, pp);
  e.dispatch(C / 128, 1, 1, 128);
  kp::GdnStep sp{1, uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), cfg_.eps};
  e.use(step);
  e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
  e.buf(5, s.rec[li]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
  e.val(9, sp);
  e.dispatch(cfg_.lin_hv, 1, 1, 1024);
  enc_qmv(e, L.out, gout_, x_, &x_, nullptr);
}

void Qwen35::enc_attn(Encoder& e, Seq& s, int li, const Layer& L, int pos) {
  static Pipeline prep = Gpu::get().pipeline("attn_prep");
  static Pipeline part = Gpu::get().pipeline("attn_decode_partial");
  static Pipeline red = Gpu::get().pipeline("attn_decode_reduce");
  enc_qmv_multi(e, {{&L.q, qg_}, {&L.k, kr_}, {&L.v, vr_}}, x_, L.in_norm);
  kp::AttnPrep ap{1, uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim), uint32_t(cfg_.rot_dims),
                  uint32_t(pos), uint32_t(s.cap), cfg_.eps, cfg_.theta};
  e.use(mrope(s) ? pipe("attn_prep_m") : prep);
  e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
  e.buf(5, qo_); e.buf(6, s.kc[li]); e.buf(7, s.vc[li]);
  e.val(8, ap);
  if (mrope(s)) enc_mrope_args(e, s);
  e.dispatch(cfg_.hq + 2 * cfg_.hkv, 1, 1, cfg_.head_dim);

  const uint32_t Lk = uint32_t(pos + 1);
  // split-K: one chunk per GPU core and kv head (35B: 2 x 20 threadgroups on the M4 Pro), chunks of at
  // least 64 positions
  const uint32_t cores = gpu_cores();
  uint32_t chunk = std::max<uint32_t>(64, (Lk + cores - 1) / cores);
  chunk = (chunk + 15) / 16 * 16;
  uint32_t nch = (Lk + chunk - 1) / chunk;
  kp::AttnDec dp{uint32_t(cfg_.hq), uint32_t(cfg_.hkv), Lk, chunk, uint32_t(s.cap),
                 1.0f / std::sqrt(float(cfg_.head_dim)), nch};
  e.use(part);
  e.buf(0, qo_); e.buf(1, s.kc[li]); e.buf(2, s.vc[li]); e.buf(3, part_o_); e.buf(4, part_ml_);
  e.val(5, dp);
  e.dispatch(nch, cfg_.hkv, 1, 32 * (cfg_.hq / cfg_.hkv));
  e.use(red);
  e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
  e.val(4, dp);
  e.dispatch(cfg_.hq, 1, 1, cfg_.head_dim);
  enc_qmv(e, L.o, attn_out_, x_, &x_, nullptr);
}

void Qwen35::enc_moe(Encoder& e, const Layer& L) {
  // the fused decode kernels read 4-bit group-64 experts and shared expert; other formats run the per-row MoE
  // (router, top-k, gate/up, down: the verify path at one row)
  if (!(L.eg.q4() && L.eu.q4() && L.ed.q4() && L.sg.q4() && L.su.q4() && L.sd.q4())) {
    enc_moe_rows(e, L, 1, x_, false);
    return;
  }
  // NI_GU: 2 = fused top-k (v2), 3 = separate top-k dispatch + v3 rows 2, 4 = v3 rows 4
  static int variant = getenv("NI_GU") ? atoi(getenv("NI_GU")) : 3;
  static Pipeline gu = Gpu::get().pipeline(variant == 2 ? "moe_gate_up2" : variant == 3 ? "moe_gate_up3_r2" : "moe_gate_up3_r4");
  const Pipeline dn = pipe(moe_wide() ? "moe_down2_w" : "moe_down2");
  static Pipeline topk = Gpu::get().pipeline(cfg_.experts == 256 && cfg_.top_k <= 8 ? "moe_topk256" : "moe_topk");
  constexpr uint32_t nsg = 8;
  const uint32_t gu_rows = variant == 4 ? 4 : 2, rows = 2;
  enc_qmv_multi(e, {{&L.router, rlog_}, {&L.shared_gate, slog_}}, x_, L.post_norm, &xn_);
  kp::Moe mp{uint32_t(cfg_.hidden), uint32_t(cfg_.moe_I), uint32_t(cfg_.shared_I), uint32_t(cfg_.top_k), cfg_.eps};
  uint32_t E = uint32_t(cfg_.experts);
  if (variant != 2) {
    kp::Topk tp{E, uint32_t(cfg_.top_k), cfg_.norm_topk ? 1u : 0u};
    e.use(topk);
    e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, eidx_); e.buf(3, ewts_);
    e.val(4, tp);
    e.dispatch(1, 1, 1, 32);
  }
  e.use(gu);
  e.buf(0, L.eg.w->buf); e.buf(1, L.eg.s->buf); e.buf(2, L.eg.b->buf);
  e.buf(3, L.eu.w->buf); e.buf(4, L.eu.s->buf); e.buf(5, L.eu.b->buf);
  e.buf(6, L.sg.w->buf); e.buf(7, L.sg.s->buf); e.buf(8, L.sg.b->buf);
  e.buf(9, L.su.w->buf); e.buf(10, L.su.s->buf); e.buf(11, L.su.b->buf);
  e.buf(12, xn_); e.buf(13, rlog_); e.buf(14, slog_); e.buf(15, eh_);
  e.val(16, mp);
  e.buf(17, eidx_); e.buf(18, ewts_); e.val(19, E);
  uint32_t I = uint32_t(std::max(cfg_.moe_I, cfg_.shared_I));
  e.dispatch((I + gu_rows * nsg - 1) / (gu_rows * nsg), cfg_.top_k + 1, 1, 32 * nsg);
  e.use(dn);
  e.buf(0, L.ed.w->buf); e.buf(1, L.ed.s->buf); e.buf(2, L.ed.b->buf);
  e.buf(3, L.sd.w->buf); e.buf(4, L.sd.s->buf); e.buf(5, L.sd.b->buf);
  e.buf(6, eh_); e.buf(7, eidx_); e.buf(8, ewts_); e.buf(9, x_);
  e.val(10, mp);
  e.dispatch((cfg_.hidden + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
}

void Qwen35::enc_mlp(Encoder& e, const Layer& L) {
  if (enc_qmvf(e, {{&L.mg, mh_}, {&L.mu, mh_}}, x_, 1, L.post_norm, nullptr, {}, 0, 0, true)) {
    enc_qmv(e, L.md, mh_, x_, &x_, nullptr);
    return;
  }
  static Pipeline gu = Gpu::get().pipeline("mlp_gate_up");
  constexpr uint32_t rows = 2, nsg = 8;
  kp::Qmv p{L.mg.N, L.mg.K, L.mg.K, L.mg.N, cfg_.eps};
  if (!L.mg.same_format(L.mu)) fatal("dense MLP: gate and up differ in format");
  e.use(L.mg.q4() ? gu : pipe("mlp_gate_up_" + L.mg.fmt()));
  e.buf(0, L.mg.w->buf); e.buf(1, L.mg.s->buf); e.buf(2, L.mg.b->buf);
  e.buf(3, L.mu.w->buf); e.buf(4, L.mu.s->buf); e.buf(5, L.mu.b->buf);
  e.buf(6, x_); e.buf(7, L.post_norm->buf); e.buf(8, mh_);
  e.val(9, p);
  e.dispatch((L.mg.N + rows * nsg - 1) / (rows * nsg), 1, 1, 32 * nsg);
  enc_qmv(e, L.md, mh_, x_, &x_, nullptr);
}

void Qwen35::enc_mlp_prefill(Encoder& e, const Layer& L, int T, const Buffer& xbuf) {
  const int D = cfg_.hidden;
  enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  if (!enc_qmm_gu(e, L.mg, L.mu, xh_, hhalf_, T)) {
    enc_qmm(e, L.mg, xh_, hg_, T, false);
    enc_qmm(e, L.mu, xh_, hu_, T, false);
    uint32_t n = uint32_t(T) * L.mg.N;
    e.use(pipe("swiglu_half"));
    e.buf(0, hg_); e.buf(1, hu_); e.buf(2, hhalf_); e.val(3, n);
    e.dispatch((n + 255) / 256, 1, 1, 256);
  }
  enc_qmm(e, L.md, hhalf_, xbuf, T, true);
}

void Qwen35::encode_step(Encoder& e, Seq& s, int pos, const SamplingParams& sp, uint32_t step, bool sample) {
  if (pos + 1 >= s.cap) fatal("context full (%d)", s.cap);
  enc_embed(e, s.tok, pos, 1, x_);
  static int dbg_layers = getenv("NI_DEBUG_LAYERS") ? atoi(getenv("NI_DEBUG_LAYERS")) : -1;
  // profiling aid: NI_SKIP bitmask 1=moe 2=gdn 4=attn 8=head (output becomes garbage)
  static int skip = getenv("NI_SKIP") ? atoi(getenv("NI_SKIP")) : 0;
  for (int i = 0; i < cfg_.layers; ++i) {
    if (i == dbg_layers) return;
    const Layer& L = layers_[i];
    if (L.linear) {
      if (!(skip & 2)) enc_gdn(e, s, i, L);
    } else if (!(skip & 4)) {
      enc_attn(e, s, i, L, pos);
    }
    if (i + 1 == dbg_layers && getenv("NI_DEBUG_HALF")) return;
    if (skip & 1) continue;
    if (cfg_.moe) enc_moe(e, L);
    else enc_mlp(e, L);
    if (cap_) enc_capture(e, i, pos, 1);
  }
  if (!sample || (skip & 8)) return;
  enc_sample(e, s, x_, pos + 1, sp, step);
}

void Qwen35::enc_sample(Encoder& e, Seq& s, const Buffer& xrow, int out_index, const SamplingParams& sp, uint32_t step) {
  static Pipeline st1 = Gpu::get().pipeline("topk_stage1");
  static Pipeline st2 = Gpu::get().pipeline("sample_stage2");
  enc_qmv(e, lm_head_, xrow, logits_, nullptr, final_norm_);
  enc_logit_mask(e, s, logits_, 1, cfg_.vocab, cfg_.vocab);  // structured output (nothing without masks)
  // stage 2 holds 4 candidates per thread (4096): 122 threadgroups x 32 fits
  uint32_t K = sp.temperature > 0 ? uint32_t(std::clamp(sp.top_k <= 0 ? 32 : sp.top_k, 1, 32)) : 1u;
  kp::TopkStage ts{uint32_t(cfg_.vocab), K, 2048};
  uint32_t ntg = (cfg_.vocab + 2047) / 2048;
  e.use(st1);
  e.buf(0, logits_); e.buf(1, cand_v_); e.buf(2, cand_i_);
  e.val(3, ts);
  e.dispatch(ntg, 1, 1, 256);
  kp::Sample sp2{ntg * K, K, sp.temperature, sp.top_p, sp.seed, step, uint32_t(out_index)};
  e.use(st2);
  e.buf(0, cand_v_); e.buf(1, cand_i_); e.buf(2, s.tok); e.buf(3, sprob_);
  e.val(4, sp2);
  e.dispatch(1, 1, 1, 1024);
}

// ---- prefill ----
static Pipeline pipe(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) {
    if (n.rfind("qmt", 0) == 0 || n.rfind("qattn_t", 0) == 0) Gpu::get().compile_extra("tensor", kTensorKernelSource, true);
    it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  }
  return it->second;
}

// NI_QMT (read once; default on, 0 = off): the dense 4-bit prefill GEMMs (M >= 64, group 64) on Metal 4's tensor_ops
// (qmt.metal: the M5's matrix units; bench_qmm on the M5 Pro, 2048-row chunks of Qwen3.5-4B's projections: 21-23
// TFLOP/s against qmr's 7.4-7.6, the fused gate/up 22.8 against 7.5). Needs macOS 26 and an Apple10 GPU.
static bool qmt_on() {
  static const bool on = [] {
    if (env_int("NI_QMT", 1) == 0) return false;
    if (Gpu::get().apple_family() < 10) return false;
    bool ok = false;
    if (__builtin_available(macOS 26.0, *)) ok = true;
    if (ok) logi("NI_QMT: dense prefill GEMMs on tensor_ops (qmt_m64_n64, qmtg_m64_n64), M >= 64");
    return ok;
  }();
  return on;
}

void Qwen35::enc_rows_half(Encoder& e, const Buffer& x, const Tensor* norm, const Buffer& y, int T, int D) {
  e.use(pipe("rows_to_half"));
  kp::Rows p{uint32_t(D), norm ? 1u : 0u, cfg_.eps};
  e.buf(0, x);
  e.buf(1, norm ? norm->buf : x);
  e.buf(2, y);
  e.val(3, p);
  e.dispatch(T, 1, 1, 256);
}

void Qwen35::enc_qmm(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add) {
  if (!sp_qmt_pref(L, M) && small_t(M) && sp_gemm() != SpGemm::Qmr && (qmr_ok(L) || qmms_ok(L))) {
    enc_proj(e, {{&L, y, add}}, xh, M);
    return;
  }
  enc_qmm_large(e, L, xh, y, M, add);
}

// NI_QMM8 (read once): 8-bit g64 large-prefill GEMMs on qmr8_* (qmm3.metal: the qmr design with 8-bit
// codes, 1.4-1.5x qmm_b8_m32_n32) instead of qmm_b8_m32_n32. 1 (default): every 8-bit linear but the trunk's
// MoE router and shared-expert gate (the MTP head's projections, an 8-bit trunk's attention, DeltaNet and MLP
// matrices); 2: also the router and shared-expert gate (changes routing near-ties: p4000 logits KL 2e-4 for
// ~0.3%); 0 = off.
static int qmm8_mode() {
  static const int v = [] {
    const int m = std::clamp(env_int("NI_QMM8", 1), 0, 2);
    if (m) logi("NI_QMM8=%d: 8-bit prefill GEMMs on qmr8 (%s)", m, m == 1 ? "all but the MoE router" : "all");
    return m;
  }();
  return v;
}
static bool qmm8_on() { return qmm8_mode() > 0; }
static bool qmm8_for(const QLinear& L) { return qmm8_mode() == 2 || (qmm8_mode() == 1 && !L.gate); }

// Prefill GEMM for the formats of wfmt.metal (bf16, 3/5/6-bit, 4-bit group 32): qmr's tiles (m32 x n128 from 64
// rows, m16 x n64 below, m8 x n64 up to 8). NI_QRF16_MF (read once; default on): bf16 large tiles with the M
// tiles along grid x, so the threadgroups running together share the weight tile through the cache.
// NI_QRF_MF (read once; default on for the M1 (Apple7) only): the other formats' large tiles too (prefill_mfast).
void Qwen35::enc_qrf(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add) {
  static const bool mf = env_int("NI_QRF16_MF", 1) != 0;
  std::string n = "qrf_" + L.fmt();
  uint32_t bm = 32, bn = 128, th = 128;
  bool mfast = false;
  if (M <= 8) n += "_m8_n64", bm = 8, bn = 64, th = 64;
  else if (M < 64 || L.N <= 64) n += "_m16_n64", bm = 16, bn = 64;
  else if (L.dense() ? mf : prefill_mfast()) n += "_m32_n128_mf", mfast = true;
  else n += "_m32_n128";
  e.use(pipe(n));
  kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, add ? 1u : 0u, uint32_t(L.gs), 0, 0};
  e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
  e.val(5, p);
  const uint32_t nt = (L.N + bn - 1) / bn, mt = (uint32_t(M) + bm - 1) / bm;
  if (mfast) e.dispatch(mt, nt, 1, th);
  else e.dispatch(nt, mt, 1, th);
}

void Qwen35::enc_qmm_large(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, int M, bool add) {
  const QmrSel& q = qmr_sel();
  if (q.on && qmm8_for(L) && L.bits == 8 && L.gs == 64 && L.K % 64 == 0) {
    const bool small = M < 64 || L.N <= 64, mfast = !small && prefill_mfast();
    const char* n = small ? "qmr8_k32_w14_m16_n64" : mfast ? "qmr8_k32_w14_m32_n128_mf" : "qmr8_k32_w14_m32_n128";
    const uint32_t bm = small ? 16 : 32, bn = small ? 64 : 128;
    e.use(pipe(n));
    kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, add ? 1u : 0u, uint32_t(L.gs), 0, 0};
    e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
    e.val(5, p);
    if (mfast) e.dispatch((M + bm - 1) / bm, (L.N + bn - 1) / bn, 1, 128);
    else e.dispatch((L.N + bn - 1) / bn, (M + bm - 1) / bm, 1, 128);
    return;
  }
  if (q.on && qmt_on() && M >= 64 && L.bits == 4 && L.gs == 64 && L.K % 64 == 0) {
    e.use(pipe("qmt_m64_n64"));
    kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, add ? 1u : 0u, uint32_t(L.gs), 0, 0};
    e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
    e.val(5, p);
    e.dispatch((L.N + 63) / 64, (uint32_t(M) + 63) / 64, 1, 128);
    return;
  }
  if (q.on && L.bits == 4 && L.gs == 64 && L.K % 64 == 0) {
    const std::string* n = &q.name;
    uint32_t bm = q.bm, bn = q.bn, th = q.threads;
    static const std::string small = "qmr_k64_w14_m16_n64", tiny = "qmr_k64_w12_m8_n64";
    if (M <= 8) n = &tiny, bm = 8, bn = 64, th = 64;
    else if (M < 64 || L.N <= 64) n = &small, bm = 16, bn = 64, th = 128;
    e.use(pipe(*n));
    kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, add ? 1u : 0u, uint32_t(L.gs), 0, 0};
    e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
    e.val(5, p);
    e.dispatch((L.N + bn - 1) / bn, (M + bm - 1) / bm, 1, th);
    return;
  }
  if (!L.q4() && !L.q8()) {  // wfmt.metal's qrf_* (the qmr design for any format)
    enc_qrf(e, L, xh, y, M, add);
    return;
  }
  const bool big = M >= 64 && L.bits == 4;
  const int BM = big ? 64 : 32, BN = big ? 64 : 32;
  std::string n = "qmm_b" + std::to_string(L.bits) + (big ? "_m64_n64" : "_m32_n32");
  e.use(pipe(n));
  kp::Qmm p{uint32_t(M), L.N, L.K, L.K, L.N, add ? 1u : 0u, uint32_t(L.gs), 0, 0};
  e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
  e.val(5, p);
  e.dispatch((L.N + BN - 1) / BN, (M + BM - 1) / BM, 1, 128);
}

bool Qwen35::small_t(int T) const { return shortpf() && T >= 1 && T <= shortpf_max(); }

bool Qwen35::can_snapshot() const { return shortpf() && gdn_par(); }

bool Qwen35::qmms_ok(const QLinear& L) const {
  return (L.q4() || L.q8() || format_supported(L.bits, L.gs)) && L.K % 64 == 0 && L.N > 0;
}
// short-prefill kernel family of a format: qmms_b4 / qmms_b8 (specialized) or wfmt.metal's qmmsf_<fmt>
static std::string qmms_family(const QLinear& L, bool gu) {
  if (L.q4() || L.q8()) return std::string(gu ? "qmms_gu_b" : "qmms_b") + std::to_string(L.bits);
  return (gu ? "qmmsf_gu_" : "qmmsf_") + L.fmt();
}

// NI_QMMS_TY (read once; default 1, 0 = off): a qmms dispatch with fewer than 96 threadgroups (the 35B's router
// + shared gate: 17, its shared expert gate/up: 32) splits its 8-row token blocks over grid.y, at
// least 128 threadgroups in all; each output keeps its arithmetic (K split and order): bitwise equal.
// NI_QMMS_TY=2: also splits every dispatch of more than 4 token blocks into groups of at most 4
// (t4 kernels: fewer registers), each group reading the weights again.
static uint32_t qmms_row_groups(uint32_t tg, int T) {
  static const int mode = env_int("NI_QMMS_TY", 1);
  const uint32_t tb = uint32_t((T + 7) / 8);
  if (!mode || tb <= 1) return 1;
  // thresholds per GPU core: 96 and 128 threadgroups on the 20-core M4 Pro
  const uint32_t enough = 24 * gpu_cores() / 5, want = 32 * gpu_cores() / 5;
  uint32_t gy = tg >= enough ? 1u : std::min(tb, (want + tg - 1) / tg);
  if (mode == 2) gy = std::max(gy, (tb + 3) / 4);
  return gy;
}

// Y_i[T, N_i] (+)= X[T, K] W_i^T for up to 4 projections sharing the half input rows xh (row
// stride K); one dispatch per weight width. Threadgroups of 8 simdgroups; K is split over ks of
// them (bench_qmms on M4 Pro: ks 4 for K < 4096, 8 from 4096; NI_QMMS_KS overrides).
void Qwen35::enc_qmms(Encoder& e, std::vector<SegQ> segs, const Buffer& xh, int T) {
  if (segs.empty()) return;
  if (T < 1 || T > 64) fatal("qmms: T=%d", T);
  for (size_t i = 1; i < segs.size(); ++i)
    if (!segs[i].L->same_format(*segs[0].L) || segs[i].L->K != segs[0].L->K) {
      // mixed formats: split into runs of one format
      std::vector<SegQ> same, rest;
      for (auto& sq : segs) (sq.L->same_format(*segs[0].L) && sq.L->K == segs[0].L->K ? same : rest).push_back(sq);
      enc_qmms(e, same, xh, T);
      enc_qmms(e, rest, xh, T);
      return;
    }
  if (segs.size() > 4) {
    enc_qmms(e, std::vector<SegQ>(segs.begin(), segs.begin() + 4), xh, T);
    enc_qmms(e, std::vector<SegQ>(segs.begin() + 4, segs.end()), xh, T);
    return;
  }
  const QLinear& L0 = *segs[0].L;
  const uint32_t K = L0.K, gpr = K / 64, nsg = 8;
  static const int ks_env = getenv("NI_QMMS_KS") && *getenv("NI_QMMS_KS") ? atoi(getenv("NI_QMMS_KS")) : 0;
  uint32_t ks = ks_env > 0 ? uint32_t(ks_env) : (K >= 4096 ? 8u : 4u);
  while (ks > 1 && (gpr / ks < 2 || nsg % ks)) ks /= 2;
  kp::Qmms p{};
  p.K = K;
  p.T = uint32_t(T);
  p.x_ld = K;
  p.nseg = uint32_t(segs.size());
  p.ks = ks;
  const uint32_t rbs = nsg / ks;
  uint32_t tg = 0;
  for (size_t i = 0; i < 4; ++i) {
    const SegQ& sq = segs[std::min(i, segs.size() - 1)];
    if (i < segs.size()) {
      p.N[i] = sq.L->N;
      p.y_ld[i] = sq.L->N;
      p.add[i] = sq.add ? 1u : 0u;
      p.tg_begin[i] = tg;
      tg += ((sq.L->N + 7) / 8 + rbs - 1) / rbs;
    }
    e.buf(int(i * 4 + 0), sq.L->w->buf);
    e.buf(int(i * 4 + 1), sq.L->s->buf);
    e.buf(int(i * 4 + 2), sq.L->b->buf);
    e.buf(int(i * 4 + 3), sq.y);
  }
  p.tg_begin[segs.size()] = tg;
  const uint32_t gy = qmms_row_groups(tg, T), TB = uint32_t((T + 7) / 8 + gy - 1) / gy;
  e.use(pipe(qmms_family(L0, false) + "_t" + std::to_string(TB)));
  e.buf(16, xh);
  e.val(17, p);
  e.dispatch(tg, uint32_t((T + 8 * TB - 1) / (8 * TB)), 1, 32 * nsg);
}

// H[T, N] = half(silu(X G^T) * (X U^T)) (row stride N) in one pass over both matrices.
bool Qwen35::enc_qmms_gu(Encoder& e, const QLinear& G, const QLinear& U, const Buffer& xh, const Buffer& h, int T) {
  if (!qmms_ok(G) || !qmms_ok(U) || !G.same_format(U) || G.N != U.N || G.K != U.K || T < 1 || T > 64) return false;
  const uint32_t K = G.K, gpr = K / 64, nsg = 8;
  static const int ks_env = getenv("NI_QMMS_KS") && *getenv("NI_QMMS_KS") ? atoi(getenv("NI_QMMS_KS")) : 0;
  uint32_t ks = ks_env > 0 ? uint32_t(ks_env) : (K >= 4096 ? 8u : 4u);
  while (ks > 1 && (gpr / ks < 2 || nsg % ks)) ks /= 2;
  kp::Qmms p{};
  p.K = K;
  p.T = uint32_t(T);
  p.x_ld = K;
  p.nseg = 2;
  p.ks = ks;
  p.N[0] = p.N[1] = G.N;
  p.y_ld[0] = p.y_ld[1] = G.N;
  const uint32_t rbs = nsg / ks, tg = ((G.N + 7) / 8 + rbs - 1) / rbs;
  p.tg_begin[1] = p.tg_begin[2] = tg;
  const uint32_t gy = qmms_row_groups(tg, T), TB = uint32_t((T + 7) / 8 + gy - 1) / gy;
  e.use(pipe(qmms_family(G, true) + "_t" + std::to_string(TB)));
  e.buf(0, G.w->buf); e.buf(1, G.s->buf); e.buf(2, G.b->buf); e.buf(3, h);
  e.buf(4, U.w->buf); e.buf(5, U.s->buf); e.buf(6, U.b->buf);
  e.buf(16, xh);
  e.val(17, p);
  e.dispatch(tg, uint32_t((T + 8 * TB - 1) / (8 * TB)), 1, 32 * nsg);
  return true;
}

// Router top-k of T token rows (rlog_, slog_ -> ridx_, rwts_). NI_TOPK8 (read once; default on, 0 off):
// moe_topk_rows8 (E = 256: unrolled, 8 tokens per 256-thread threadgroup; bitwise equal) instead of
// moe_topk_rows (a 32-thread threadgroup per token).
void Qwen35::enc_topk_rows(Encoder& e, uint32_t E, uint32_t k, bool norm, int T) {
  static const bool t8 = env_int("NI_TOPK8", 1) != 0;
  const kp::Topk tp{E, k, norm ? 1u : 0u};
  e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
  if (t8 && tp.E == 256 && tp.k <= 16) {
    e.use(pipe("moe_topk_rows8"));
    e.val(4, tp);
    const uint32_t Tu = uint32_t(T);
    e.val(5, Tu);
    e.dispatch((Tu + 7) / 8, 1, 1, 256);
    return;
  }
  e.use(pipe("moe_topk_rows"));
  e.val(4, tp);
  e.dispatch(T, 1, 1, 32);
}

// MoE block at small T (NI_SHORTPF): router + shared gate in one qmms dispatch, experts in tiles
// of <= 8 sorted pair rows (qmms_moe_gu / qmms_moe_dn: each simdgroup streams 8 weight rows of
// one expert once for all of the tile's rows), shared expert as dense gate/up + down.
// NI_SHORTPF_MOE_FMA (read once; default on, 0 = the 8-row MMA for every tile): short-prefill experts of at
// most 2 rows as FMA matvecs, the others by the 8-row MMA (qmms_moe2_*); also takes group-32 experts (the
// MTP head's), which otherwise fall back to the large-prefill tiles. 35B cached turn (4k + 53) -5.5 ms.
static bool moe_fma() {
  static const bool on = env_int("NI_SHORTPF_MOE_FMA", 1) != 0;
  return on;
}

bool Qwen35::enc_moe_prefill_small(Encoder& e, const Layer& L, int T, const Buffer& xbuf, bool xh_ready) {
  const int D = cfg_.hidden, k = cfg_.top_k, P = T * k, E = cfg_.experts;
  const QLinear &G = L.eg, &U = L.eu, &Dn = L.ed;
  // Q2_0 experts: q20::enc_prefill's 8-row tiles (no shared rows)
  const bool q20 = G.q20 && U.q20 && Dn.q20;
  // generic expert formats (wfmt.metal qmmsf_moe_*: bf16, 8/6/5/3-bit group 64): 8-row MMA tiles, no FMA rows
  const bool gen = !G.q4() && G.bits != 4 && moe_prefill_format(G) && G.same_format(U) && G.same_format(Dn);
  auto fits = [gen, q20](const QLinear& q) {
    if (q20) return q.q20 && q.K % 64 == 0 && q.N % 8 == 0;
    if (gen) return q.K % 64 == 0 && q.N % 8 == 0;
    return q.bits == 4 && (q.gs == 64 || (moe_fma() && q.gs == 32)) && q.K % 64 == 0 && q.N % 8 == 0;
  };
  if (!fits(G) || !fits(U) || !fits(Dn) || U.N != G.N || U.K != G.K || G.K != uint32_t(D) || Dn.K != G.N ||
      Dn.N != uint32_t(D) || !qmms_ok(L.router) || !qmms_ok(L.shared_gate) || !qmms_ok(L.sd) ||
      L.sg.N != L.su.N || L.sg.K != L.su.K || L.sg.bits != L.su.bits || !qmms_ok(L.sg))
    return false;
  constexpr int bm = 8;
  constexpr uint32_t nsg = 4;
  if ((G.N / 8) % nsg || (Dn.N / 8) % nsg) return false;
  // NI_SHORTPF_FOLD=1: the shared expert rides along as expert E (its T rows in 8-row tiles after the
  // routed ones) when it is 4-bit with the routed experts' shapes; else it runs on the dense path
  static const bool fold_on = env_int("NI_SHORTPF_FOLD", 0) != 0;
  auto same = [&](const QLinear& a, const QLinear& b) {
    return (gen ? a.same_format(b) : a.bits == 4 && a.gs == b.gs) && a.N == b.N && a.K == b.K;
  };
  const bool fold = !q20 && fold_on && moe_fused() && same(L.sg, G) && same(L.su, U) && same(L.sd, Dn);
  const int S = fold ? T : 0;
  if (!xh_ready) enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_proj(e, {{&L.router, rlog_, false}, {&L.shared_gate, slog_, false}}, xh_, T);
  kp::Topk tp{uint32_t(E), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  enc_topk_rows(e, tp.E, tp.k, tp.norm != 0, T);
  kp::Bucket2 bp{uint32_t(P), uint32_t(k), uint32_t(E), uint32_t(bm), uint32_t(S)};
  e.use(pipe(moe_bucket_name()));
  e.buf(0, ridx_); e.buf(1, spair_); e.buf(2, stok_); e.buf(3, tiles_); e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  const uint32_t max_tiles = uint32_t((P + bm - 1) / bm + E + (S + bm - 1) / bm);
  const uint32_t shared_e = fold ? uint32_t(E) : uint32_t(E) + 1;
  // shared expert on all rows: half h in hh_ (row stride shared_I), down into ys_
  auto enc_shared = [&] {
    if (!enc_qmm_gu(e, L.sg, L.su, xh_, hh_, T)) {
      enc_qmm(e, L.sg, xh_, sgb_, T, false);
      enc_qmm(e, L.su, xh_, sub_, T, false);
      uint32_t ns = uint32_t(T * cfg_.shared_I);
      e.use(pipe("swiglu_half"));
      e.buf(0, sgb_); e.buf(1, sub_); e.buf(2, hh_); e.val(3, ns);
      e.dispatch((ns + 255) / 256, 1, 1, 256);
    }
    enc_qmm(e, L.sd, hh_, ys_, T, false);
  };
  if (q20) {
    // Q2_0 (never folded): the shared expert first, while a streamed-expert store loads the routed ones
    const q20::Experts X = q20_fetch(e, L, ridx_, uint32_t(P), T, enc_shared);
    q20::enc_prefill(e, X, q20_shape(), bm, max_tiles, xh_, hhalf_, yd_, tiles_, ntiles_, stok_, spair_);
  } else {
    // expert strides in words / scale groups (generic formats: N rows of row_bytes; bf16 has no groups)
    const uint32_t gws = uint32_t(size_t(G.N) * G.row_bytes() / 4), gss = G.dense() ? 0u : G.N * (G.K / G.gs);
    const uint32_t dws = uint32_t(size_t(Dn.N) * Dn.row_bytes() / 4), dss = Dn.dense() ? 0u : Dn.N * (Dn.K / Dn.gs);
    kp::Qmm2 p{G.N, G.K, G.K, G.N, uint32_t(G.gs), gws, gss, shared_e};
    const bool fma = moe_fma();
    const QLinear &SG = fold ? L.sg : G, &SU = fold ? L.su : U, &SD = fold ? L.sd : Dn;
    {
      e.use(pipe(gen ? "qmmsf_moe_gu_" + G.fmt() : fma ? "qmms_moe2_gu_b4" : "qmms_moe_gu_b4"));
      e.buf(0, xh_);
      e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf);
      e.buf(4, U.w->buf); e.buf(5, U.s->buf); e.buf(6, U.b->buf);
      e.buf(7, hhalf_);
      e.val(8, p);
      e.buf(9, tiles_); e.buf(10, ntiles_); e.buf(11, stok_);
      e.buf(12, SG.w->buf); e.buf(13, SG.s->buf); e.buf(14, SG.b->buf);
      e.buf(15, SU.w->buf); e.buf(16, SU.s->buf); e.buf(17, SU.b->buf);
      e.dispatch(G.N / (8 * nsg), max_tiles, 1, 32 * nsg);
    }
    kp::Qmm2 q{Dn.N, Dn.K, Dn.K, Dn.N, uint32_t(Dn.gs), dws, dss, shared_e};
    {
      e.use(pipe(gen ? "qmmsf_moe_dn_" + Dn.fmt() : fma ? "qmms_moe2_dn_b4" : "qmms_moe_dn_b4"));
      e.buf(0, hhalf_);
      e.buf(1, Dn.w->buf); e.buf(2, Dn.s->buf); e.buf(3, Dn.b->buf);
      e.buf(4, yd_);
      e.val(5, q);
      e.buf(6, tiles_); e.buf(7, ntiles_); e.buf(8, spair_);
      e.buf(9, SD.w->buf); e.buf(10, SD.s->buf); e.buf(11, SD.b->buf);
      e.dispatch(Dn.N / (8 * nsg), max_tiles, 1, 32 * nsg);
    }
    if (!fold) enc_shared();
  }
  kp::Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe("moe_combine"));
  e.buf(0, yd_); e.buf(1, fold ? yd_.view(size_t(P) * D * 4, size_t(T) * D * 4) : ys_); e.buf(2, rwts_);
  e.buf(3, xbuf);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, T, 1, 64);
  return true;
}

bool Qwen35::qmt_ready() { return qmt_on(); }

// The M5 default share of the Neural Engine split (default_units' gpu_mma, #318) only on the models it was measured
// and output-checked on: Qwen3.5 4B (hidden 2560, FFN 9216, 32 layers) and Qwen3.8 27B (5120, 17408, 64). Another
// 4-bit group-64 dense model (the 9B) keeps the old default share, the whole FFN on the Neural Engine at 20 GPU cores,
// until it is measured. NI_ANE_MMA=1 takes the M5 share on any such model (to measure one), 0 the old share on these.
bool Qwen35::mma_share_measured(const Qwen35Config& c) {
  if (const char* v = getenv("NI_ANE_MMA")) return atoi(v) != 0;
  return (c.hidden == 2560 && c.dense_I == 9216 && c.layers == 32) ||
         (c.hidden == 5120 && c.dense_I == 17408 && c.layers == 64);
}

// The GPU's channels [0, g) of the Neural Engine split's FFN (enc_mlp_gpu_part) on the M5's matrix units: gate/up and
// SwiGLU of the leading g rows fused (qmtg_m64_n64) into hhalf_ [T, g], down over the leading g inputs of its rows
// (qmt_m64_n64_ld) added into x. The qmm_b4_m64_n64 GEMMs the split otherwise runs are ~3x slower per channel on the
// M5 Pro (ane_calib, 27B, 2048 rows: 5.4 ms per GPU unit against 1.6 for the GPU's whole FFN), which pushed the share
// the GPU keeps to 0. The gate/up products are the GPU-alone FFN's (same kernel, tile and K order).
bool Qwen35::enc_mlp_gpu_part_qmt(Encoder& e, const Layer& L, int T, uint32_t g, const Buffer& x) {
  const QLinear &G = L.mg, &U = L.mu, &Dn = L.md;
  auto ok = [](const QLinear& q) { return q.bits == 4 && q.gs == 64 && q.K % 64 == 0; };
  if (!qmt_on() || T < 64 || g % 64 || !ok(G) || !ok(U) || !ok(Dn) || G.N != U.N || G.K != U.K) return false;
  const uint32_t D = uint32_t(cfg_.hidden), I = G.N;
  e.use(pipe("qmtg_m64_n64"));
  kp::Qmm pg{uint32_t(T), g, G.K, G.K, g, 0, uint32_t(G.gs), 0, 0};
  e.buf(0, xh_); e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf); e.buf(4, hhalf_);
  e.val(5, pg);
  e.buf(6, U.w->buf); e.buf(7, U.s->buf); e.buf(8, U.b->buf);
  e.dispatch((g + 31) / 32, (uint32_t(T) + 63) / 64, 1, 128);
  e.use(pipe("qmt_m64_n64_ld"));
  kp::QmmLd pd{{uint32_t(T), D, g, g, D, 1u, uint32_t(Dn.gs), 0, 0}, I / 8, I / 64};
  e.buf(0, hhalf_); e.buf(1, Dn.w->buf); e.buf(2, Dn.s->buf); e.buf(3, Dn.b->buf); e.buf(4, x);
  e.val(5, pd);
  e.dispatch((D + 63) / 64, (uint32_t(T) + 63) / 64, 1, 128);
  return true;
}

// NI_QMM_GU (read once, needs qmr; default on, NI_QMM_GU=0 off): dense MLP / shared-expert gate and up projections
// and SwiGLU in one qmr_gu dispatch writing the half h rows (no fp32 hg/hu, no swiglu pass).
bool Qwen35::enc_qmm_gu(Encoder& e, const QLinear& G, const QLinear& U, const Buffer& xh, const Buffer& h, int M) {
  const bool qmt_gu = sp_qmt_pref(G, M) && sp_qmt_pref(U, M);  // qmtg_m64_n64 fuses the pair
  if (!qmt_gu && small_t(M) && sp_gemm() != SpGemm::Qmr) {
    const bool tiles = qmr_ok(G) && qmr_ok(U) && G.N == U.N && G.K == U.K;
    const bool qm = !tiles || sp_gemm() == SpGemm::Qmms ||
                    (sp_gemm() == SpGemm::Auto && short_qmms(M, qbytes(G) + qbytes(U), true, false, G.K));
    if (qm && M <= 64 && enc_qmms_gu(e, G, U, xh, h, M)) return true;
  }
  if (!qmt_gu && small_t(M) && sp_gemm() != SpGemm::Qmr && qmr_ok(G) && qmr_ok(U) && G.N == U.N && G.K == U.K) {
    const ShortTile t = short_tile(M, true, false);
    e.use(pipe(t.name));
    kp::Qmm p{uint32_t(M), G.N, G.K, G.K, G.N, 0, uint32_t(G.gs), 0, 0};
    e.buf(0, xh); e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf); e.buf(4, h);
    e.val(5, p);
    e.buf(6, U.w->buf); e.buf(7, U.s->buf); e.buf(8, U.b->buf);
    e.dispatch((G.N + t.bn - 1) / t.bn, (M + t.bm - 1) / t.bm, 1, t.th);
    return true;
  }
  // the M1 (Apple7): 64-deep K steps (bench_qmm on the M1 Pro, 2048 rows: 4B 9216x2560 3.25 vs 3.02 TFLOP/s, 9B
  // 12288x4096 3.27 vs 3.03, the faster of the 3 qmr_gu tiles on both)
  static const char* const gu = Gpu::get().apple_family() <= 7 ? "qmr_gu_k64_w14_m32_n64" : "qmr_gu_k32_w14_m32_n64";
  static const bool on = [] {
    const char* g = getenv("NI_QMM_GU");
    const bool v = !(g && !strcmp(g, "0")) && qmr_sel().on;
    if (v) logi("NI_QMM_GU: dense gate/up/SwiGLU fused (%s, m16_n64 / m8_n32 below 64 rows)", gu);
    return v;
  }();
  auto ok8 = [](const QLinear& q) { return q.bits == 8 && q.gs == 64 && q.K % 64 == 0; };
  const bool b8 = on && qmm8_for(G) && qmm8_for(U) && ok8(G) && ok8(U) && G.N == U.N && G.K == U.K;
  auto ok = [](const QLinear& q) { return q.bits == 4 && q.gs == 64 && q.K % 64 == 0; };
  if (on && !G.q4() && !G.q8() && G.same_format(U) && G.N == U.N && G.K == U.K) {
    // wfmt.metal's fused gate/up for the other formats (qrf_gu_*; bf16 large tiles with M along grid x)
    static const bool mf = env_int("NI_QRF16_MF", 1) != 0;
    std::string n = "qrf_gu_" + G.fmt();
    uint32_t bm = 32, bn = 64, th = 128;
    bool mfast = false;
    if (M <= 8) n += "_m8_n32", bm = 8, bn = 32, th = 64;
    else if (M < 64) n += "_m16_n64", bm = 16;
    else if (G.dense() && mf) n += "_m32_n64_mf", mfast = true;
    else n += "_m32_n64";
    e.use(pipe(n));
    kp::Qmm p{uint32_t(M), G.N, G.K, G.K, G.N, 0, uint32_t(G.gs), 0, 0};
    e.buf(0, xh); e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf); e.buf(4, h);
    e.val(5, p);
    e.buf(6, U.w->buf); e.buf(7, U.s->buf); e.buf(8, U.b->buf);
    const uint32_t nt = (G.N + bn - 1) / bn, mt = (uint32_t(M) + bm - 1) / bm;
    if (mfast) e.dispatch(mt, nt, 1, th);
    else e.dispatch(nt, mt, 1, th);
    return true;
  }
  if (!b8 && (!on || !ok(G) || !ok(U) || G.N != U.N || G.K != U.K)) return false;
  if (!b8 && qmt_on() && M >= 64) {
    e.use(pipe("qmtg_m64_n64"));
    kp::Qmm p{uint32_t(M), G.N, G.K, G.K, G.N, 0, uint32_t(G.gs), 0, 0};
    e.buf(0, xh); e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf); e.buf(4, h);
    e.val(5, p);
    e.buf(6, U.w->buf); e.buf(7, U.s->buf); e.buf(8, U.b->buf);
    e.dispatch((G.N + 31) / 32, (uint32_t(M) + 63) / 64, 1, 128);
    return true;
  }
  const char* n = gu;
  uint32_t bm = 32, bn = 64, th = 128;
  if (b8) n = M < 64 ? "qmr8_gu_k32_w14_m16_n64" : "qmr8_gu_k32_w14_m32_n64", bm = M < 64 ? 16 : 32;
  else if (M <= 8) n = "qmr_gu_k64_w12_m8_n32", bm = 8, bn = 32, th = 64;
  else if (M < 64) n = "qmr_gu_k64_w14_m16_n64", bm = 16, bn = 64, th = 128;
  e.use(pipe(n));
  kp::Qmm p{uint32_t(M), G.N, G.K, G.K, G.N, 0, uint32_t(G.gs), 0, 0};
  e.buf(0, xh); e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf); e.buf(4, h);
  e.val(5, p);
  e.buf(6, U.w->buf); e.buf(7, U.s->buf); e.buf(8, U.b->buf);
  e.dispatch((G.N + bn - 1) / bn, (M + bm - 1) / bm, 1, th);
  return true;
}

void Qwen35::enc_qmm_moe(Encoder& e, const QLinear& L, const Buffer& xh, const Buffer& y, const Buffer& xrows,
                         const Buffer& yrows, int P, int bm) {
  std::string n = "qmm_moe_b4_m" + std::to_string(bm) + "_n64";
  e.use(pipe(n));
  // expert matrices are stacked [E, N, K]: strides in words / groups
  uint32_t wstride = L.N * (L.K * L.bits / 32), sstride = L.N * (L.K / L.gs);
  kp::Qmm p{uint32_t(P), L.N, L.K, L.K, L.N, 0, uint32_t(L.gs), wstride, sstride};
  e.buf(0, xh); e.buf(1, L.w->buf); e.buf(2, L.s->buf); e.buf(3, L.b->buf); e.buf(4, y);
  e.val(5, p);
  e.buf(6, tiles_); e.buf(7, ntiles_); e.buf(8, xrows); e.buf(9, yrows);
  uint32_t max_tiles = uint32_t((P + bm - 1) / bm + cfg_.experts);
  e.dispatch((L.N + 63) / 64, max_tiles, 1, 128);
}

void Qwen35::enc_moe_prefill(Encoder& e, const Layer& L, int T, const Buffer& xbuf, bool xh_ready) {
  if (L.eg.iq) {
    enc_moe_prefill_iq(e, L, T, xbuf, xh_ready);
    return;
  }
  if (shortpf() && T <= shortpf_moe_max() && enc_moe_prefill_small(e, L, T, xbuf, xh_ready)) return;
  if ((moe_fused() || L.eg.bits != 4) && enc_moe_prefill_fused(e, L, T, xbuf, xh_ready)) return;
  if (L.eg.bits != 4) fatal("MoE prefill: no kernels for %s experts", format_name(L.eg.bits, L.eg.gs).c_str());
  const int D = cfg_.hidden, k = cfg_.top_k, P = T * k;
  if (!xh_ready) enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_proj(e, {{&L.router, rlog_, false}, {&L.shared_gate, slog_, false}}, xh_, T);
  kp::Topk tp{uint32_t(cfg_.experts), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  enc_topk_rows(e, tp.E, tp.k, tp.norm != 0, T);
  const int per_expert = P / cfg_.experts;
  // 16-row tiles measured fastest for expert GEMMs at every chunk size (32-row tiles hit a
  // 2-3 TFLOP/s pathology on M4 Pro; 64-row tiles waste more rows in partial tiles).
  int bm = 16;
  (void)per_expert;
  if (getenv("NI_BM")) bm = atoi(getenv("NI_BM"));
  kp::Bucket bp{uint32_t(P), uint32_t(k), uint32_t(cfg_.experts), uint32_t(bm)};
  e.use(pipe("moe_bucket"));
  e.buf(0, ridx_); e.buf(1, spair_); e.buf(2, stok_); e.buf(3, tiles_); e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  enc_qmm_moe(e, L.eg, xh_, hg_, stok_, spair_, P, bm);
  enc_qmm_moe(e, L.eu, xh_, hu_, stok_, spair_, P, bm);
  uint32_t n = uint32_t(P * cfg_.moe_I);
  e.use(pipe("swiglu_half"));
  e.buf(0, hg_); e.buf(1, hu_); e.buf(2, hhalf_); e.val(3, n);
  e.dispatch((n + 255) / 256, 1, 1, 256);
  enc_qmm_moe(e, L.ed, hhalf_, yd_, spair_, spair_, P, bm);
  // shared expert on all rows
  enc_qmm(e, L.sg, xh_, sgb_, T, false);
  enc_qmm(e, L.su, xh_, sub_, T, false);
  uint32_t ns = uint32_t(T * cfg_.shared_I);
  e.use(pipe("swiglu_half"));
  e.buf(0, sgb_); e.buf(1, sub_); e.buf(2, hh_); e.val(3, ns);
  e.dispatch((ns + 255) / 256, 1, 1, 256);
  enc_qmm(e, L.sd, hh_, ys_, T, false);
  kp::Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe("moe_combine"));
  e.buf(0, yd_); e.buf(1, ys_); e.buf(2, rwts_); e.buf(3, xbuf);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, T, 1, 64);
}

// Opt-in MoE prefill (NI_MOE_FUSED): router and top-k as above, then
//   moe_bucket2 -> qmm_moe_gu (gate+up+SwiGLU -> half h, sorted rows) -> qmm_moe_dn (-> yd by pair)
// with the shared expert as expert index E over the T token rows (mode 1, when its shapes and
// quantization match the routed experts') or on the dense path (mode 2). Per row the arithmetic
// matches the original path (same dequantized halfs, same MMA order over K). Returns false when
// the layer does not fit the kernels' assumptions (the caller then runs the original path).
bool Qwen35::enc_moe_prefill_fused(Encoder& e, const Layer& L, int T, const Buffer& xbuf, bool xh_ready) {
  const int D = cfg_.hidden, k = cfg_.top_k, P = T * k, E = cfg_.experts;
  const QLinear &G = L.eg, &U = L.eu, &Dn = L.ed;
  // Q2_0 experts: q20::enc_prefill's qrm-design kernels on 32-row tiles (no shared rows)
  const bool q20 = G.q20 && U.q20 && Dn.q20;
  // NI_MOE_QMR: qrm_gu (64 + 64 weight rows) / qrm_dn (128 rows) on 32- or 16-row tiles (4-bit g64)
  // generic expert formats: wfmt.metal's qrmf_* (the qrm design, 32-row tiles)
  const bool gen = G.bits != 4 && moe_prefill_format(G) && G.same_format(U) && G.same_format(Dn) && G.K % 64 == 0 &&
                   Dn.K % 64 == 0 && G.N % 64 == 0 && Dn.N % 128 == 0;
  // qmtm_gu_m64_n64 / qmtm_dn_m64_n64: tile width BN 64 (gate/up: 32 gate + 32 up columns per threadgroup) and K
  // step 32, the measured best; the tensor library builds only these two.
  constexpr int qtm_gu_n = 64, qtm_dn_n = 64;
  // qmtm_gu_m64_n64 / qmtm_dn_m64_n64 on 64-row tiles: the qmt tile (tensor_ops) on the routed experts' grouped
  // GEMMs (4-bit g64). 64 rows is the whole tile of an expert's bucket at 4K-16K rows of a 256-expert layer.
  const bool qtm = !q20 && !gen && moe_qtm() && qmt_on() && G.bits == 4 && G.gs == 64 && U.bits == 4 &&
                   U.gs == 64 && Dn.bits == 4 && Dn.gs == 64 && G.K % 64 == 0 && Dn.K % 64 == 0 &&
                   G.N % qtm_gu_n == 0 && Dn.N % qtm_dn_n == 0;
  const bool qrm = !qtm && !gen && moe_qmr() && G.gs == 64 && U.gs == 64 && Dn.gs == 64 && G.K % 64 == 0 &&
                   Dn.K % 64 == 0 && G.N % 64 == 0 && Dn.N % 128 == 0;
  // NI_MOE_BM: rows per tile (16, 32 or 64); partial tiles skip their empty 8-row fragments
  static const int bm_env = getenv("NI_MOE_BM") ? atoi(getenv("NI_MOE_BM")) : 64;
  const int bm = q20 || gen ? 32 : qtm ? 64 : qrm ? moe_qmr() : bm_env;
  const int kBnGu = q20 || qrm || gen ? 64 : qtm ? qtm_gu_n / 2 : 32, kBnDn = q20 || qrm || gen ? 128 : qtm ? qtm_dn_n : 64;
  auto fits = [gen, q20](const QLinear& q) {
    if (q20) return q.q20 && q.K % 64 == 0;
    return gen ? q.K % 32 == 0 : q.bits == 4 && q.gs % 32 == 0 && q.K % 32 == 0;
  };
  // gate and up share K, strides and the xh_ row stride (D) in one dispatch
  if (!fits(G) || !fits(U) || !fits(Dn) || G.N % kBnGu || Dn.N % kBnDn || U.N != G.N || U.K != G.K ||
      U.gs != G.gs || G.K != uint32_t(D) || Dn.K != G.N || Dn.N != uint32_t(D) || (bm != 16 && bm != 32 && bm != 64)) {
    static bool warned = false;
    if (!warned) logi("NI_MOE_FUSED: layer shapes/quantization not supported, using the original MoE path");
    warned = true;
    return false;
  }
  const bool fold = !q20 && moe_fused() == 1 && fits(L.sg) && fits(L.su) && fits(L.sd) && L.sg.N == G.N && L.su.N == G.N &&
                    L.sg.K == G.K && L.su.K == G.K && L.sd.K == Dn.K && L.sd.N == Dn.N && L.sg.same_format(G) &&
                    L.su.same_format(G) && L.sd.same_format(Dn) && L.sg.gs == G.gs && L.su.gs == G.gs &&
                    L.sd.gs == Dn.gs;
  static bool qtm_logged = false;
  if (qtm && !qtm_logged) {
    qtm_logged = true;
    logi("NI_MOE_QTM: routed expert GEMMs on tensor_ops (qmtm_gu_m64_n64, qmtm_dn_m64_n64, 64-row tiles)");
  }
  const int S = fold ? T : 0;
  if (!xh_ready) enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_proj(e, {{&L.router, rlog_, false}, {&L.shared_gate, slog_, false}}, xh_, T);
  kp::Topk tp{uint32_t(E), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  enc_topk_rows(e, tp.E, tp.k, tp.norm != 0, T);
  kp::Bucket2 bp{uint32_t(P), uint32_t(k), uint32_t(E), uint32_t(bm), uint32_t(S)};
  e.use(pipe(moe_bucket_name()));
  e.buf(0, ridx_); e.buf(1, spair_); e.buf(2, stok_); e.buf(3, tiles_); e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  const uint32_t max_tiles = uint32_t((P + bm - 1) / bm + E + (S + bm - 1) / bm);
  const std::string m = "_b4_m" + std::to_string(bm);
  const std::string qm = "_k" + std::to_string(moe_qmr_k(D, cfg_.moe_I)) + "_m" + std::to_string(bm);
  // shared expert on all rows (as in enc_moe_prefill)
  auto enc_shared = [&] {
    if (!enc_qmm_gu(e, L.sg, L.su, xh_, hh_, T)) {
      enc_qmm(e, L.sg, xh_, sgb_, T, false);
      enc_qmm(e, L.su, xh_, sub_, T, false);
      uint32_t ns = uint32_t(T * cfg_.shared_I);
      e.use(pipe("swiglu_half"));
      e.buf(0, sgb_); e.buf(1, sub_); e.buf(2, hh_); e.val(3, ns);
      e.dispatch((ns + 255) / 256, 1, 1, 256);
    }
    enc_qmm(e, L.sd, hh_, ys_, T, false);
  };
  if (q20) {
    // Q2_0 (never folded): the shared expert first, while a streamed-expert store loads the routed ones
    const q20::Experts X = q20_fetch(e, L, ridx_, uint32_t(P), T, enc_shared);
    q20::enc_prefill(e, X, q20_shape(), bm, max_tiles, xh_, hhalf_, yd_, tiles_, ntiles_, stok_, spair_);
  }
  // gate + up + SwiGLU: hhalf_ row r = sorted row r (pairs grouped by expert, then shared rows)
  if (!q20) {
    // expert matrices are stacked [E, N, K]: strides in words / groups
    kp::Qmm2 p{G.N, G.K, G.K, G.N, uint32_t(G.gs), uint32_t(size_t(G.N) * G.row_bytes() / 4),
               G.dense() ? 0u : G.N * (G.K / G.gs), uint32_t(E)};
    const QLinear &SG = fold ? L.sg : G, &SU = fold ? L.su : U;
    e.use(pipe(qtm ? "qmtm_gu_m64_n64"
               : gen ? "qrmf_gu_" + G.fmt() + "_m32_n64"
               : qrm ? "qrm_gu" + qm + "_n" + std::to_string(kBnGu) : "qmm_moe_gu" + m + "_n" + std::to_string(kBnGu)));
    e.buf(0, xh_);
    e.buf(1, G.w->buf); e.buf(2, G.s->buf); e.buf(3, G.b->buf);
    e.buf(4, U.w->buf); e.buf(5, U.s->buf); e.buf(6, U.b->buf);
    e.buf(7, hhalf_);
    e.val(8, p);
    e.buf(9, tiles_); e.buf(10, ntiles_); e.buf(11, stok_);
    e.buf(12, SG.w->buf); e.buf(13, SG.s->buf); e.buf(14, SG.b->buf);
    e.buf(15, SU.w->buf); e.buf(16, SU.s->buf); e.buf(17, SU.b->buf);
    e.dispatch(G.N / kBnGu, max_tiles, 1, 128);
  }
  // down: yd_ row = pair id (shared rows: P + token)
  if (!q20) {
    kp::Qmm2 p{Dn.N, Dn.K, Dn.K, Dn.N, uint32_t(Dn.gs), uint32_t(size_t(Dn.N) * Dn.row_bytes() / 4),
               Dn.dense() ? 0u : Dn.N * (Dn.K / Dn.gs), uint32_t(E)};
    const QLinear& SD = fold ? L.sd : Dn;
    e.use(pipe(qtm ? "qmtm_dn_m64_n64"
               : gen ? "qrmf_dn_" + Dn.fmt() + "_m32_n128"
               : qrm ? "qrm_dn" + qm + "_n" + std::to_string(kBnDn) : "qmm_moe_dn" + m + "_n" + std::to_string(kBnDn)));
    e.buf(0, hhalf_);
    e.buf(1, Dn.w->buf); e.buf(2, Dn.s->buf); e.buf(3, Dn.b->buf);
    e.buf(4, yd_);
    e.val(5, p);
    e.buf(6, tiles_); e.buf(7, ntiles_); e.buf(8, spair_);
    e.buf(9, SD.w->buf); e.buf(10, SD.s->buf); e.buf(11, SD.b->buf);
    e.dispatch(Dn.N / kBnDn, max_tiles, 1, 128);
  }
  if (!fold && !q20) enc_shared();
  kp::Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe("moe_combine"));
  e.buf(0, yd_);
  e.buf(1, fold ? yd_.view(size_t(P) * D * 4, size_t(T) * D * 4) : ys_);
  e.buf(2, rwts_); e.buf(3, xbuf);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, T, 1, 64);
  return true;
}

// NI_GDN_PAR=1 (read once): token-parallel GDN prefill. gdn_prep_par runs the conv/norm with
// a simdgroup per 8 tokens x 128 channels instead of a T-long loop per channel; the recurrence
// spreads each head's value rows over simdgroups/threadgroups with no barriers (gdn_step_par_l*),
// and gdn_gnorm applies the gated RMSNorm, writing out_proj's half input directly (replaces
// rows_to_half). Same arithmetic order as the default kernels. NI_GDN_STEP picks the recurrence
// layout (gdn_step_par_l{4,8,16,32}: lanes per value row), NI_GDN_SGS its simdgroups per threadgroup.
// On by default (35B 4k prefill +5.5%); NI_GDN_PAR=0 runs the per-channel gdn_prep/gdn_step path.
static bool gdn_par() {
  static const bool on = !getenv("NI_GDN_PAR") || !*getenv("NI_GDN_PAR") || atoi(getenv("NI_GDN_PAR")) > 0;
  return on;
}

void Qwen35::enc_gdn_prefill_par(Encoder& e, Seq& s, int li, const Layer& L, int T, Checkpoint* snap, int snap_row) {
  // default gdn_step_parr4_l16: four value rows per lane (independent recurrences interleaved), 1.6x
  // gdn_step_par_l16 and bitwise equal to it
  static const std::string step_name = getenv("NI_GDN_STEP") ? getenv("NI_GDN_STEP") : "gdn_step_parr4_l16";
  static const uint32_t sgs = getenv("NI_GDN_SGS") ? uint32_t(atoi(getenv("NI_GDN_SGS"))) : 2u;
  static const uint32_t lanes = uint32_t(atoi(step_name.c_str() + step_name.rfind("_l") + 2));
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const uint32_t Hv = uint32_t(cfg_.lin_hv);
  // rows per threadgroup = sgs * 32/lanes (gdn_step_parr{R}_*: R rows per lane, R times that) must split
  // the 128 value rows of a head evenly
  static const uint32_t rpl = step_name.rfind("gdn_step_parr", 0) == 0 ? uint32_t(atoi(step_name.c_str() + 13)) : 1u;
  const uint32_t rows_tg = lanes ? sgs * (32 / lanes) * rpl : 0;
  if (!rows_tg || 32 % lanes || 128 % rows_tg || sgs > 32)
    fatal("NI_GDN_STEP=%s with NI_GDN_SGS=%u: bad layout", step_name.c_str(), sgs);
  kp::GdnPrep pp{uint32_t(T), uint32_t(cfg_.lin_hk), Hv, C};
  const uint32_t srow = uint32_t(snap_row);
  e.use(pipe(snap ? "gdn_prep_par_snap" : "gdn_prep_par"));
  e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[li]);
  e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
  e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
  e.val(12, pp);
  if (snap) {
    e.buf(13, snap->conv[li]);
    e.val(14, srow);
  }
  e.dispatch((T + 31) / 32, C / 128 + 1, 1, 128);
  kp::GdnStep sp{uint32_t(T), uint32_t(cfg_.lin_hk), Hv, cfg_.eps};
  if (snap) {
    // gdn_step_<form>_l{L} -> gdn_step_<form>_snap_l{L}
    static const std::string snap_name = step_name.substr(0, step_name.rfind('_')) + "_snap" +
                                         step_name.substr(step_name.rfind('_'));
    e.use(pipe(snap_name));
  } else {
    e.use(pipe(step_name));
  }
  e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
  e.buf(5, s.rec[li]); e.buf(6, gout_);
  e.val(7, sp);
  if (snap) {
    e.buf(8, snap->rec[li]);
    e.val(9, srow);
  }
  e.dispatch(Hv * (128 / rows_tg), 1, 1, 32 * sgs);
  e.use(pipe(gdn_sigmoid_ ? "gdn_gnorm_sg" : "gdn_gnorm"));
  e.buf(0, gout_); e.buf(1, zb_); e.buf(2, L.gnorm->buf); e.buf(3, hh_);
  e.val(4, sp);
  e.dispatch((uint32_t(T) * Hv + 7) / 8, 1, 1, 256);
}

// Causal attention of T new query rows (qo_, gate qg_) against the cache (kc, vc) -> attn_out_.
// attn_prefill11 (default): MLX-style layout, 32 queries of one query head per threadgroup,
// fragments in registers (2.1x v10 on the 35B shape, 3.1x on the 27B's G=6). NI_ATTN overrides.
void Qwen35::enc_attn_prefill(Encoder& e, const Buffer& kc, const Buffer& vc, int T, int pos0, int cap) {
  kp::AttnPref fp{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(pos0), uint32_t(cap),
                  1.0f / std::sqrt(float(cfg_.head_dim))};
  // default attn_prefill11r: v11 skipping the O rescale when no row's max moved (bitwise equal)
  static const std::string an = getenv("NI_ATTN") && *getenv("NI_ATTN") ? getenv("NI_ATTN") : "attn_prefill11r";
  // NI_SHORTPF: few queries over a long cached prefix. Rows are (query, head) pairs of one KV
  // head (attn_prefill11g*: a 32-row block holds 32/G queries x G heads instead of padding T to 32
  // per head), and the keys are split over threadgroups when the blocks alone cannot fill the GPU
  // (aim ~100 in all: within ~5% of the best split of a 2-16 sweep at 8-128 queries over 4k/16k
  // keys on both models; >= 8 key tiles of 32 each; NI_ATTN_SPLIT overrides). Partials in
  // part_o_/part_ml_ (room for nsplit * T <= 1024 rows), combined by attn11_reduce.
  if (shortpf() && T <= shortpf_attn_max() && (an == "attn_prefill11" || an == "attn_prefill11r")) {
    const uint32_t G = uint32_t(cfg_.hq / cfg_.hkv);
    const uint32_t nqb = (uint32_t(T) * G + 31) / 32, gy = uint32_t(cfg_.hkv), tiles = (uint32_t(pos0 + T) + 31) / 32;
    static const int ns_env = getenv("NI_ATTN_SPLIT") && *getenv("NI_ATTN_SPLIT") ? atoi(getenv("NI_ATTN_SPLIT")) : 0;
    const uint32_t target = 5 * gpu_cores();  // ~5 threadgroups per core (100 on the 20-core M4 Pro)
    uint32_t ns = ns_env > 0 ? uint32_t(ns_env) : (target + nqb * gy - 1) / (nqb * gy);
    ns = std::min({ns, tiles / 8, 1024u / uint32_t(T), 16u});
    if (ns < 2) {
      e.use(pipe("attn_prefill11g"));
      e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_); e.buf(4, attn_out_);
      e.val(5, fp);
      e.dispatch(nqb, gy, 1, 256);
      return;
    }
    if (ns >= 2) {
      e.use(pipe("attn_prefill11gs"));
      e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_); e.buf(4, attn_out_);
      e.val(5, fp);
      e.buf(6, part_o_); e.buf(7, part_ml_);
      e.val(8, ns);
      e.dispatch(nqb, gy, ns, 256);
      e.use(pipe("attn11_reduce"));
      e.buf(0, part_o_); e.buf(1, part_ml_); e.buf(2, qg_); e.buf(3, attn_out_);
      e.val(4, fp);
      e.val(5, ns);
      e.dispatch(T, cfg_.hq, 1, cfg_.head_dim);
      return;
    }
  }
  // The M1 (Apple7) runs 16-key tiles (attn_prefill11_k16xr: 11r's skipped unit rescales over attn_prefill11_k16x's
  // 16-key tiles, fewer registers and the exchange in its own buffer). bench_attn on the M1 Pro, a 2048-query chunk
  // of Qwen3.5-4B/9B's heads (16 q, 4 kv, D 256) against 4K / 8K / 16K keys: 39.3 / 90.8 / 203.7 ms vs 60.2 / 140.1 /
  // 309.2 for attn_prefill11r. The 16-key tiles' online softmax rounds differently: outputs are not bit-identical to
  // 11r's, error against an fp64 reference the same (2.1-2.4e-4).
  static const std::string full = getenv("NI_ATTN") && *getenv("NI_ATTN") ? an
                                  : Gpu::get().apple_family() <= 7 ? "attn_prefill11_k16xr" : an;
  uint32_t qb = 4, th = 256, gy = cfg_.hkv;
  // NI_QATTN (default on, 0 = off): D 256, 4 or 8 query heads per KV head, on tensor_ops (qattn_t.metal) on the M5.
  // G = 8 (the 35B) runs qattn_t_g8s4k16: 2 tokens x the KV head's 8 heads per 16-row tile, 8 tokens per
  // threadgroup. bench_attn, T 2112 against 2112 / 4224 / 8448 / 16493 keys: 4.08 / 11.49 / 28.38 / 56.63 ms vs
  // attn_prefill11r's 5.41 / 15.96 / 39.97 / 79.64 (1.32-1.41x), error 2.1-3.2e-4 against the fp64 reference (11r:
  // 2.1-2.3e-4). End to end on the 35B through bench/harness on 487ea80 (the experts already on the tensor units),
  // 3 interleaved sessions, median: 16K prefill 1,801 -> 1,999 tok/s (+11%), 4K flat (2,424 -> 2,476); on 330b216,
  // before the experts moved, 4K prefill 1,693 -> 1,636 ms and 16K 8,819 -> 7,929 ms (median of three).
  // It reassociates the K reductions (the products are 11r's: half x half in fp32), so its logits move: decode rows
  // 5.6e-4 mean KL, 1.10x the 35B's noise-floor gate, and it departs from that gate by the user's call (AGENTS.md,
  // "Changing speed or numerics"; tools/logit_floors/ is still the M4 Pro measurement of the old kernels).
  // NI_QATTN_G8=0 keeps the 35B on 11r; NI_ATTN overrides both.
  // NI_QATTN_D2 (default on, 0 = off): G = 8 runs qattn_t_g8d2s8 instead, the head dim split over a pair of
  // simdgroups (half the accumulator registers; qattn_t.metal), 8 tokens per threadgroup as before. bench_attn on the
  // M5 Pro, T 4224 against 8448 keys / T 3328 against 16000: 24.22 / 44.42 ms vs g8s4k16's 44.08 / 78.23 (1.82x /
  // 1.76x), error against the fp64 reference the same (2.58e-4 / 3.09e-4 vs 2.60e-4 / 3.08e-4). The scores add the
  // two halves of the head dim separately, so logits are not bit-identical to g8s4k16's.
  static const bool qattn = !(getenv("NI_ATTN") && *getenv("NI_ATTN")) && env_int("NI_QATTN", 1) != 0 && qmt_on();
  // Only the environment is read once: the head ratios depend on cfg_, which changes when the server swaps models
  // (--allow-switch), and function-local statics survive the swap.
  static const bool g8_on = env_int("NI_QATTN_G8", 1) != 0;
  static const bool d2_on = env_int("NI_QATTN_D2", 1) != 0;
  // NI_QATTN_G6 (default on, 0 = off): G = 6 (the 27B, 24 query heads over 4 KV heads) runs qattn_t_g6d2s8, the d2
  // kernel over (token, head) rows, 64 rows (10 2/3 tokens) per threadgroup; off, it stays on attn_prefill11r.
  // bench_attn on the M5 Pro, the 27B's heads: T 141 / 512 against a 16K prefix 4.16 / 12.76 ms vs 11r's 10.93 /
  // 30.77; T 2048 against 4096 / 16384 keys 8.85 / 44.23 ms vs 22.21 / 111.64 (2.4-2.6x). Error against the fp64
  // reference 2.6-3.1e-4 (11r 1.9-2.6e-4; the shipped G = 4 and G = 8 qattn kernels sit the same distance above it).
  static const bool g6_on = env_int("NI_QATTN_G6", 1) != 0;
  const bool qg4 = cfg_.hq == 4 * cfg_.hkv;
  const bool qg8 = g8_on && cfg_.hq == 8 * cfg_.hkv;
  const bool qg6 = g6_on && cfg_.hq == 6 * cfg_.hkv;
  if (qattn && cfg_.head_dim == 256 && qg6) {
    e.use(pipe("qattn_t_g6d2s8"));
    e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_); e.buf(4, attn_out_);
    e.val(5, fp);
    e.dispatch((uint32_t(T) * 6 + 63) / 64, cfg_.hkv, 1, 256);   // 4 pairs x 16 rows
    return;
  }
  if (qattn && cfg_.head_dim == 256 && (qg4 || qg8)) {
    e.use(pipe(qg8 ? (d2_on ? "qattn_t_g8d2s8" : "qattn_t_g8s4k16") : "qattn_t_s8k16"));
    e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_); e.buf(4, attn_out_);
    e.val(5, fp);
    if (qg8) {
      // g8s4k16: 4 simdgroups x 2 tokens of one KV head; g8d2s8: 4 pairs x 2 tokens
      e.dispatch((uint32_t(T) + 7) / 8, cfg_.hkv, 1, d2_on ? 256 : 128);
    } else {
      e.dispatch((uint32_t(T) + 31) / 32, cfg_.hkv, 1, 256);
    }
    return;
  }
  if (full.rfind("attn_prefill11", 0) == 0) { qb = 32; gy = cfg_.hq; }  // 32 queries of one q head
  else if (full == "attn_prefill") th = 128;
  else if (full == "attn_prefill2" || full == "attn_prefill3" || full == "attn_prefill4") qb = 8;
  else if (full.find("_q8") != std::string::npos) { qb = 8; th = 512; }
  e.use(pipe(full));
  e.buf(0, qo_); e.buf(1, kc); e.buf(2, vc); e.buf(3, qg_); e.buf(4, attn_out_);
  e.val(5, fp);
  e.dispatch((T + qb - 1) / qb, gy, 1, th);
}

// The token mixer (GDN or attention) of layer i for T rows at positions pos0.., residual rows x.
// snap / snap_row: capture the recurrent state after row snap_row (NI_SHORTPF in-pass checkpoint).
void Qwen35::enc_mixer_prefill(Encoder& e, Seq& s, int i, const Layer& L, int pos0, int T, const Buffer& xbuf,
                               Checkpoint* snap, int snap_row) {
  const int D = cfg_.hidden;
  const uint32_t C = 2 * cfg_.lin_hk * cfg_.lin_dk + cfg_.lin_hv * cfg_.lin_dv;
  const int VD = cfg_.lin_hv * cfg_.lin_dv;
  enc_rows_half(e, xbuf, L.in_norm, xh_, T, D);
  if (L.linear) {
    enc_proj(e, {{&L.qkv, qkv_, false}, {&L.z, zb_, false}, {&L.b, bb_, false}, {&L.a, ab_, false}}, xh_, T);
    if (gdn_par()) {
      enc_gdn_prefill_par(e, s, i, L, T, snap, snap_row);
    } else {
      kp::GdnPrep pp{uint32_t(T), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), C};
      e.use(pipe("gdn_prep"));
      e.buf(0, qkv_); e.buf(1, ab_); e.buf(2, bb_); e.buf(3, s.conv[i]);
      e.buf(4, L.conv_w->buf); e.buf(5, L.A_log->buf); e.buf(6, L.dt_bias->buf);
      e.buf(7, gq_); e.buf(8, gk_); e.buf(9, gv_); e.buf(10, gg_); e.buf(11, gbeta_);
      e.val(12, pp);
      e.dispatch(C / 128, 1, 1, 128);
      kp::GdnStep sp2{uint32_t(T), uint32_t(cfg_.lin_hk), uint32_t(cfg_.lin_hv), cfg_.eps};
      e.use(pipe("gdn_step"));
      e.buf(0, gq_); e.buf(1, gk_); e.buf(2, gv_); e.buf(3, gg_); e.buf(4, gbeta_);
      e.buf(5, s.rec[i]); e.buf(6, zb_); e.buf(7, L.gnorm->buf); e.buf(8, gout_);
      e.val(9, sp2);
      e.dispatch(cfg_.lin_hv, 1, 1, 1024);
      enc_rows_half(e, gout_, nullptr, hh_, T, VD);
    }
    enc_qmm(e, L.out, hh_, xbuf, T, true);
  } else {
    enc_proj(e, {{&L.q, qg_, false}, {&L.k, kr_, false}, {&L.v, vr_, false}}, xh_, T);
    kp::AttnPrep ap{uint32_t(T), uint32_t(cfg_.hq), uint32_t(cfg_.hkv), uint32_t(cfg_.head_dim),
                    uint32_t(cfg_.rot_dims), uint32_t(pos0), uint32_t(s.cap), cfg_.eps, cfg_.theta};
    e.use(pipe(mrope(s) ? "attn_prep_m" : "attn_prep"));
    e.buf(0, qg_); e.buf(1, kr_); e.buf(2, vr_); e.buf(3, L.q_norm->buf); e.buf(4, L.k_norm->buf);
    e.buf(5, qo_); e.buf(6, s.kc[i]); e.buf(7, s.vc[i]);
    e.val(8, ap);
    if (mrope(s)) enc_mrope_args(e, s);
    e.dispatch(cfg_.hq + 2 * cfg_.hkv, T, 1, cfg_.head_dim);
    enc_attn_prefill(e, s.kc[i], s.vc[i], T, pos0, s.cap);
    enc_rows_half(e, attn_out_, nullptr, hh_, T, cfg_.hq * cfg_.head_dim);
    enc_qmm(e, L.o, hh_, xbuf, T, true);
  }
}

void Qwen35::encode_prefill(Encoder& e, Seq& s, int pos0, int T, const SamplingParams& sp, uint32_t step, bool sample,
                            Checkpoint* snap, int snap_row) {
  if (T > tmax_) fatal("prefill chunk %d > %d", T, tmax_);
  if (pos0 + T >= s.cap) fatal("context full (%d)", s.cap);
  if (snap && (!can_snapshot() || snap_row <= 0 || snap_row >= T)) fatal("prefill snapshot at row %d of %d", snap_row, T);
  draft_vocab_rewind(s, pos0);  // tokens from pos0 on may be new
  const int D = cfg_.hidden;
  enc_embed(e, s.tok, pos0, T, x_);
  if (s.embeds) vision::enc_embed_runs(e, *s.embeds, pos0, T, x_, D);  // image rows
  // NI_ANE: dense FFN split with the Neural Engine. A chunk carrying an in-pass checkpoint stays on the
  // GPU (the engine cuts such chunks at the boundary instead when the split would apply).
  ane_chunk_ = !snap && ane_splits(T);
  if (ane_chunk_) ane_begin_chunk(T);
  if (ane_chunk_ && ane_wave_rows(T)) {
    enc_prefill_wave(e, s, pos0, T, ane_wave_rows(T));
  } else {
    for (int i = 0; i < cfg_.layers; ++i) {
      const Layer& L = layers_[i];
      enc_mixer_prefill(e, s, i, L, pos0, T, x_, snap, snap_row);
      if (cfg_.moe) enc_moe_prefill(e, L, T, x_);
      else if (ane_chunk_) enc_mlp_prefill_split(e, i, L, T, x_);  // joins the ANE's part into x_ in the pass
      else enc_mlp_prefill(e, L, T, x_);
      if (cap_) enc_capture(e, i, pos0, T);  // block-drafter capture (DFlash2), off unless armed
    }
  }
  if (sample) enc_sample(e, s, x_.view(size_t(T - 1) * D * 4, size_t(D) * 4), pos0 + T, sp, step);
  if (snap) snap->pos = pos0 + snap_row;
}

void Qwen35::enc_capture(Encoder& e, int layer, int pos0, int T, const Buffer* x) {
  if (!cap_ || layer >= int(cap_->slot.size()) || cap_->slot[layer] < 0) return;
  const int lo = std::max(pos0, cap_->pos_base), hi = std::min(pos0 + T, cap_->pos_base + cap_->rows);
  if (lo >= hi) return;
  const uint32_t D = uint32_t(cfg_.hidden);
  kp::DflCapture p{D, cap_->ld, uint32_t(cap_->slot[layer]) * D, uint32_t(lo - cap_->pos_base)};
  e.use(pipe("dfl_capture"));
  e.buf(0, x ? *x : x_, size_t(lo - pos0) * D * 4);
  e.buf(1, cap_->dst);
  e.val(2, p);
  if (v8w_R_) {  // a wide tree verify: its live rows (the hook covers all its rows)
    if (lo != pos0 || hi != pos0 + T) fatal("capture: a wide tree verify's hook must cover its %d rows", T);
    v8_disp(e, (D / 4 + 255) / 256, 1, 1, kV8wYrows, 256);
  } else {
    e.dispatch((D / 4 + 255) / 256, uint32_t(hi - lo), 1, 256);
  }
}

Buffer Qwen35::debug_buf(const std::string& n) {
  if (n == "rlog") return rlog_;
  if (n == "slog") return slog_;
  if (n == "eidx") return eidx_;
  if (n == "ewts") return ewts_;
  if (n == "eh") return eh_;
  if (n == "gout") return gout_;
  if (n == "qkv") return qkv_;
  if (n == "gq") return gq_;
  if (n == "gv") return gv_;
  if (n == "gg") return gg_;
  if (n == "gbeta") return gbeta_;
  if (n == "zb") return zb_;
  return x_;
}

} // namespace ni
