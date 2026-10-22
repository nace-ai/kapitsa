// Routed experts in llama.cpp's IQ blocks (IQ1_S, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S), as `kapitsa import` keeps them from
// a GGUF (src/gguf/import.cpp): loading (the blocks' bits moved into the layout of src/kernels/wfmt_iq.metal) and the
// MoE passes of such layers on wfmt_iq.metal's kernels: one or a few token rows (decode, the small-batch verify), the
// 8-row verify (moe8) and prefill. The shared expert is the importer's 8-bit group 64; router and top-k are the other
// formats' passes.
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "model/kparams.h"
#include "model/qwen35.h"
#include "util/log.h"

#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <thread>
#include <unordered_map>

namespace ni {

namespace {

Pipeline pipe_iq(const std::string& n) {
  static std::unordered_map<std::string, Pipeline> cache;
  auto it = cache.find(n);
  if (it == cache.end()) it = cache.emplace(n, Gpu::get().pipeline(n)).first;
  return it->second;
}

kp::IqSec sec(const QLinear& q) { return kp::IqSec{q.iq_x1, q.iq_x2, q.iq_x3, q.iq_d}; }

int env_int(const char* n, int def) {
  const char* v = getenv(n);
  return v && *v ? atoi(v) : def;
}

}  // namespace

// The routed experts' blocks are deferred by the WeightStore: read here, one matrix at a time, and moved into the GPU
// layout (same bytes) by parallel threads.
// The weights are locked in memory (mlock) as they are written: IQ experts are how a model this size fits a 16 GB Mac,
// and there the VM compressed and swapped the freshly written buffers while the rest loaded (they look idle until the
// GPU first reads them), which the first prompt then paid back. M1 Pro 16 GB, 40 layers of the 35B-A3B (11.2 GB of
// weights), three runs each: first 269-token prompt 15-20 s (3-5 GB swapped out, up to 10 GB in the compressor) ->
// 0.9 s with nothing swapped out; decode unchanged (49 tok/s). NI_MLOCK=0 leaves them pageable; a lock refused
// (past vm.user_wire_limit, 13 GB on 16 GB Macs) is logged and the load goes on without one.
void Qwen35::load_iq_experts() {
  Gpu& g = Gpu::get();
  bool any = false;
  for (const Layer& L : layers_) any |= L.eg.iq || L.eu.iq || L.ed.iq;
  if (!any) return;
  bool lock = !(getenv("NI_MLOCK") && !strcmp(getenv("NI_MLOCK"), "0"));
  size_t locked = 0;
  auto wire = [&](const Buffer& b) {
    if (!lock) return;
    if (mlock(b.ptr(), b.size)) {
      logi("could not lock weights in memory (%s) past %.2f GB: the rest stays pageable", strerror(errno), locked / 1e9);
      lock = false;
      return;
    }
    locked += b.size;
  };
  for (const Buffer& b : ws_.buffers()) wire(b);  // read already: locked before the experts' load can page it out
  std::vector<uint8_t> raw;
  size_t total = 0;
  const double t0 = now_ms();
  const unsigned nth = std::max(2u, std::min(8u, std::thread::hardware_concurrency()));
  for (Layer& L : layers_)
    for (QLinear* q : {&L.eg, &L.eu, &L.ed}) {
      if (!q->iq) continue;
      const auto it = iq_src_.find(q->w);
      if (it == iq_src_.end()) fatal("internal: IQ experts without their tensor");
      const size_t rows = size_t(q->w->shape[0]) * q->N, rb = q->row_bytes();
      const gguf::IqLayout lay = gguf::iq_layout(q->iq, rows, q->K);
      raw.resize(rows * rb);
      ws_.read_deferred(it->second, raw.data());
      Buffer b = g.alloc(lay.bytes, "iq_experts", /*tracked=*/false);
      std::vector<std::thread> th;
      const size_t per = (rows + nth - 1) / nth;
      for (unsigned i = 0; i < nth; ++i) {
        const size_t r0 = std::min(rows, i * per), r1 = std::min(rows, r0 + per);
        if (r0 < r1)
          th.emplace_back([&, r0, r1] { gguf::iq_repack(q->iq, raw.data() + r0 * rb, rows, q->K, r0, r1, b.as<uint8_t>()); });
      }
      for (auto& t : th) t.join();
      wire(b);
      owned_bufs_.push_back(b);
      Tensor t = *q->w;
      t.buf = b.view(0, lay.bytes);
      owned_.push_back(t);
      q->w = q->s = q->b = &owned_.back();
      q->iq_x1 = uint32_t(lay.x1), q->iq_x2 = uint32_t(lay.x2), q->iq_x3 = uint32_t(lay.x3), q->iq_d = uint32_t(lay.d);
      total += lay.bytes;
    }
  iq_src_.clear();
  logi("read %.2f GB of IQ experts in %.0f ms; %.2f GB of weights locked in memory", total / 1e9, now_ms() - t0,
       locked / 1e9);
}

// What the IQ MoE kernels need of a layer (src/kernels/wfmt_iq.metal); "" when it has it
std::string Qwen35::iq_layer_check(const Layer& L) const {
  const int D = cfg_.hidden, I = cfg_.moe_I;
  if (!L.eg.iq || L.eu.iq != L.eg.iq || !L.ed.iq) return "routed gate, up and down in IQ blocks (gate and up of one type)";
  if (!L.sg.q8() || !L.su.q8() || !L.sd.q8()) return "an 8-bit group-64 shared expert";
  if (cfg_.shared_I != I) return "a shared expert as wide as the routed ones";
  if (D % 512 || I % 256) return "a hidden size multiple of 512 and an expert size multiple of 256";
  if (cfg_.experts > 512 || cfg_.top_k > 8) return "at most 512 experts and 8 per token";
  if (L.eg.N != uint32_t(I) || L.eg.K != uint32_t(D) || L.ed.N != uint32_t(D) || L.ed.K != uint32_t(I))
    return "expert shapes that match the config";
  return "";
}

// One or a few token rows (decode, the small-batch verify), after enc_moe_rows' router and top-k: xrn_ the rows'
// normalized inputs, ridx_ / rwts_ their experts and weights (the shared expert's last). x += moe.
// bench_moe_iq on the M1 Pro, Qwen3.6-35B-A3B's layer 0 (IQ2_XXS gate/up, IQ2_S down), one row: gate/up 2 rows x 2
// simdgroups 68 us (4 x 2: 84, 8 x 2: 133), down 8 rows per threadgroup with the lanes of a row sharing its h 47 us
// (16 rows: 63; without the sharing: 97); 0.12 ms per layer, 86 GB/s of experts read (the first version: 0.16).
void Qwen35::enc_moe_iq_rows(Encoder& e, const Layer& L, int R, const Buffer& x) {
  const uint32_t D = uint32_t(cfg_.hidden), I = uint32_t(cfg_.moe_I), SI = uint32_t(cfg_.shared_I);
  const uint32_t k = uint32_t(cfg_.top_k), Imax = std::max(I, SI);
  const kp::MoeIq p{D, I, SI, k, sec(L.eg), sec(L.ed)};
  constexpr uint32_t gnr = 2, gsg = 2, dnr = 8;
  e.use(pipe_iq("moei_gu_rows_" + L.eg.fmt() + "_r2"));
  e.buf(0, L.eg.w->buf); e.buf(1, L.eu.w->buf);
  e.buf(2, L.sg.w->buf); e.buf(3, L.sg.s->buf); e.buf(4, L.sg.b->buf);
  e.buf(5, L.su.w->buf); e.buf(6, L.su.s->buf); e.buf(7, L.su.b->buf);
  e.buf(8, xrn_); e.buf(9, eh_);
  e.val(10, p);
  e.buf(11, ridx_);
  e.dispatch((Imax + gnr * gsg - 1) / (gnr * gsg), k + 1, uint32_t(R), 32 * gsg);
  const bool t16 = I == 512 && SI == 512;  // the lanes of a row share its h (the kernel's T = 16)
  e.use(pipe_iq("moei_down_rows_" + L.ed.fmt() + "_r8" + (t16 ? "_t16" : "")));
  e.buf(0, L.ed.w->buf);
  e.buf(1, L.sd.w->buf); e.buf(2, L.sd.s->buf); e.buf(3, L.sd.b->buf);
  e.buf(4, eh_); e.buf(5, ridx_); e.buf(6, rwts_); e.buf(7, x);
  e.val(8, p);
  e.dispatch((D + dnr - 1) / dnr, uint32_t(R), 1, 32 * (k + 1));
}

// The 8-row verify's MoE block (enc_moe8's passes): x_ += sum_j w_j expert_j(n) + w_s shared(n), n = norm(x_), with
// moe8_plan's tiles (every touched expert once for all its rows, the shared expert one more tile) on moe8i_gu /
// moe8i_down, then moe8_combine. Row-independent (no scalar passes for lightly used experts).
void Qwen35::enc_moe8_iq(Encoder& e, const Layer& L, int R, int li) {
  const int D = cfg_.hidden, k = cfg_.top_k, E = cfg_.experts;
  const uint32_t I = uint32_t(cfg_.moe_I);
  static const bool precise = env_int("NI_V8_ROUTER_X2", 1) != 0;
  enc_x8(e, x_, R, D, L.post_norm, precise);
  enc_qmm8(e, {{&L.router, rlog_, uint32_t(E)}, {&L.shared_gate, slog_, 1u}}, R, v8t_, v8s_, precise);
  kp::Topk tp{uint32_t(E), uint32_t(k), cfg_.norm_topk ? 1u : 0u};
  // the router top-k inside moe8_plan, as enc_moe8 (E == 256; NI_V8_MOE_TOPKPLAN / V8Knobs::tkp)
  static const bool topk_env = env_int("NI_V8_MOE_TOPKPLAN", 1) != 0;
  const bool topk_in_plan = E == 256 && k <= 8 && (v8_knobs().tkp >= 0 ? v8_knobs().tkp != 0 : topk_env);
  if (!topk_in_plan) {
    e.use(pipe_iq(E == 256 && k <= 8 ? "moe_topk256_rows" : "moe_topk_rows"));
    e.buf(0, rlog_); e.buf(1, slog_); e.buf(2, ridx_); e.buf(3, rwts_);
    e.val(4, tp);
    e.dispatch(R, 1, 1, 32);
  }
  constexpr uint32_t nf = 2, cs = 4;
  // every touched expert on one MMA-layout tile (no small / split tiles, no list B)
  kp::Moe8Plan pp{uint32_t(R), uint32_t(k), uint32_t(E), I / 64, uint32_t(D) / (8 * nf * cs), uint32_t(li), 0u, 1u, 1u,
                  v8_rlim_ ? 1u : 0u, 0u, 0u, 0u, 1u, topk_in_plan ? 1u : 0u};
  e.use(pipe_iq("moe8_plan"));
  e.buf(0, ridx_); e.buf(1, v8tiles_); e.buf(2, v8nt_); e.buf(3, v8slot_); e.buf(4, v8args_);
  e.val(5, pp);
  e.buf(6, v8ustat_); e.buf(7, v8sm_);
  e.buf(8, v8_rlim_ ? *v8_rlim_ : v8nt_);
  e.buf(9, v8sm2_);
  e.buf(10, rlog_); e.buf(11, slog_); e.buf(12, rwts_);
  e.val(13, tp);
  e.dispatch(1, 1, 1, uint32_t(std::max(32, (E + 31) / 32 * 32)));
  const kp::Moe8Iq gp{uint32_t(D), I, I * 8, I / 64 * 8, uint32_t(k), sec(L.eg)};
  e.use(pipe_iq("moe8i_gu_" + L.eg.fmt()));
  e.buf(0, L.eg.w->buf); e.buf(1, L.eu.w->buf);
  e.buf(2, L.sg.w->buf); e.buf(3, L.sg.s->buf); e.buf(4, L.sg.b->buf);
  e.buf(5, L.su.w->buf); e.buf(6, L.su.s->buf); e.buf(7, L.su.b->buf);
  e.buf(8, v8t_); e.buf(9, v8s_); e.buf(10, v8h_); e.buf(11, v8hs_);
  e.val(12, gp);
  e.buf(13, v8tiles_); e.buf(14, v8nt_);
  e.dispatch_indirect(v8args_, 0, 512);
  const kp::Moe8Iq dp{I, uint32_t(D), I * 8, I / 64 * 8, uint32_t(k), sec(L.ed)};
  e.use(pipe_iq("moe8i_down_" + L.ed.fmt() + "_nf2"));
  e.buf(0, L.ed.w->buf);
  e.buf(1, L.sd.w->buf); e.buf(2, L.sd.s->buf); e.buf(3, L.sd.b->buf);
  e.buf(4, v8h_); e.buf(5, v8hs_); e.buf(6, v8out_); e.buf(7, v8sh_);
  e.val(8, dp);
  e.buf(9, v8tiles_); e.buf(10, v8nt_); e.buf(11, v8slot_);
  e.dispatch_indirect(v8args_, 12, 32 * cs);
  kp::Moe8Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe_iq("moe8_combine"));
  e.buf(0, v8out_); e.buf(1, v8sh_); e.buf(2, rwts_); e.buf(3, x_);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, R, 1, 64);
}

// Prefill of T rows: router and top-k on the prefill GEMMs, the (row, expert) pairs bucketed by expert into tiles of
// 32 rows (moe_bucket3), gate/up + SwiGLU and down on qrmi_* (the qrm design), the shared expert on the dense path
// (8-bit), moe_combine. xh_ready: xh_ holds the layer's normalized half rows (Qwen4Exp).
void Qwen35::enc_moe_prefill_iq(Encoder& e, const Layer& L, int T, const Buffer& xbuf, bool xh_ready) {
  const int D = cfg_.hidden, k = cfg_.top_k, P = T * k, E = cfg_.experts;
  const uint32_t I = uint32_t(cfg_.moe_I);
  if (!xh_ready) enc_rows_half(e, xbuf, L.post_norm, xh_, T, D);
  enc_proj(e, {{&L.router, rlog_, false}, {&L.shared_gate, slog_, false}}, xh_, T);
  enc_topk_rows(e, uint32_t(E), uint32_t(k), cfg_.norm_topk, T);
  constexpr int bm = 32;
  kp::Bucket2 bp{uint32_t(P), uint32_t(k), uint32_t(E), uint32_t(bm), 0u};
  e.use(pipe_iq("moe_bucket3"));
  e.buf(0, ridx_); e.buf(1, spair_); e.buf(2, stok_); e.buf(3, tiles_); e.buf(4, ntiles_);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  const uint32_t max_tiles = uint32_t((P + bm - 1) / bm + E);
  // gate/up on 32-deep K steps (104 GPRs against 124), down 64 output columns per threadgroup: a 4-layer model of
  // Qwen3.6-35B-A3B UD-IQ2_XXS reads a 4096-token prompt in 919 instead of 940 ms on the M1 Pro (2048: 446 / 455;
  // 64: 31.2 / 30.7)
  const kp::QrmIq gp{I, uint32_t(D), uint32_t(D), I, sec(L.eg)};
  e.use(pipe_iq("qrmi_gu_" + L.eg.fmt() + "_m32_n64_k32"));
  e.buf(0, xh_); e.buf(1, L.eg.w->buf); e.buf(2, L.eu.w->buf); e.buf(3, hhalf_);
  e.val(4, gp);
  e.buf(5, tiles_); e.buf(6, ntiles_); e.buf(7, stok_);
  e.dispatch(I / 64, max_tiles, 1, 128);
  const kp::QrmIq dp{uint32_t(D), I, I, uint32_t(D), sec(L.ed)};
  e.use(pipe_iq("qrmi_dn_" + L.ed.fmt() + "_m32_n64"));
  e.buf(0, hhalf_); e.buf(1, L.ed.w->buf); e.buf(2, yd_);
  e.val(3, dp);
  e.buf(4, tiles_); e.buf(5, ntiles_); e.buf(6, spair_);
  e.dispatch(uint32_t(D) / 64, max_tiles, 1, 128);
  // the shared expert on all rows: half h in hh_ (row stride shared_I), down into ys_
  if (!enc_qmm_gu(e, L.sg, L.su, xh_, hh_, T)) {
    enc_qmm(e, L.sg, xh_, sgb_, T, false);
    enc_qmm(e, L.su, xh_, sub_, T, false);
    uint32_t ns = uint32_t(T * cfg_.shared_I);
    e.use(pipe_iq("swiglu_half"));
    e.buf(0, sgb_); e.buf(1, sub_); e.buf(2, hh_); e.val(3, ns);
    e.dispatch((ns + 255) / 256, 1, 1, 256);
  }
  enc_qmm(e, L.sd, hh_, ys_, T, false);
  kp::Combine cp{uint32_t(D), uint32_t(k)};
  e.use(pipe_iq("moe_combine"));
  e.buf(0, yd_); e.buf(1, ys_); e.buf(2, rwts_); e.buf(3, xbuf);
  e.val(4, cp);
  e.dispatch((D / 4 + 63) / 64, T, 1, 64);
}

}  // namespace ni
