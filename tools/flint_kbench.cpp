// Flint's decode kernels on synthetic weights of its shapes (hidden 2048; experts 256 x 512, top-6, shared expert 1024;
// q | k | v 4096 + 512 + 512; o 2048 x 4096), timed in-pipeline: NL copies of a layer's weights (so the cache cannot
// hold them; NL x 14 MB) cycled through by back-to-back dispatches in one command buffer, the GPU time per call and the
// weight bytes it read per second. Trials interleaved over the kernels; medians printed.
//   flint_kbench [kernel ...] [--layers NL] [--calls N] [--trials T]
// kernels: gu3 (moe_gate_up3_r2, routed + shared), gu3r (its routed slots alone), gu1 (moe_gate_up), dn1 / dn1_128 /
// dn1_64 (moe_down, 8 / 4 / 2 simdgroups), dn2w (moe_down2_w), qkvf (qmvf over q | k | v), o (qmvf o + residual), o2
// (qmv2_b4_g64_r2_add), router (flint_router_topk), rshared (flint_router_shared), headN (the LM head on qmvf, N
// simdgroups), and attention at position KB_POS (default 1024): prep, partial, reduce, d2 (flint_attn_d2), a trailing F
// for a full-attention layer (a sliding-window one otherwise). "A+B" times A then B per call, "A|B" side by side.
// router, rshared and d2 time the fused kernels (split 0): on macOS 14, which has no device-scope fences, they run without
// their merges (flint.metal, FLINT_FENCES), so time only the rest there.
// Takes the GPU lock ($BENCH_GPU_LOCK, default /tmp/kapitsa-gpu.lock) unless NI_LOCK_HELD=1 or NACE_GPU_LOCK_HELD=1.
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/kparams.h"
#include "util/log.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace ni {
extern const char* const kFlintKernelSource;
}
using namespace ni;

namespace {

constexpr uint32_t D = 2048, E = 256, I = 512, SI = 1024, K6 = 6, QN = 4096, KN = 512;
constexpr uint32_t EA = 8;  // experts allocated per layer copy (the routed ids are 0 .. 5)

struct Mat {
  Buffer w, s, b;
  uint32_t N = 0, K = 0, E = 1;
  size_t bytes() const { return w.size + s.size + b.size; }
};

Mat rand_mat(uint32_t N, uint32_t K, uint32_t Ecount, std::mt19937& rng) {
  Gpu& g = Gpu::get();
  Mat m;
  m.N = N, m.K = K, m.E = Ecount;
  const size_t rows = size_t(N) * Ecount;
  m.w = g.alloc(rows * K / 2, "kb_w", false);
  m.s = g.alloc(rows * (K / 64) * 2, "kb_s", false);
  m.b = g.alloc(rows * (K / 64) * 2, "kb_b", false);
  uint32_t* w = m.w.as<uint32_t>();
  for (size_t i = 0; i < m.w.size / 4; ++i) w[i] = rng();
  uint16_t* s = m.s.as<uint16_t>();
  uint16_t* b = m.b.as<uint16_t>();
  for (size_t i = 0; i < m.s.size / 2; ++i) s[i] = 0x3c00 /* bf16 1/128-ish */ - 0x0700, b[i] = 0xbc00 - 0x0700;
  return m;
}

struct Layer {
  Mat eg, eu, ed, sg, su, sd, q, k, v, o;
  Buffer router, norm;
};

Pipeline pipe(const std::string& n) { return Gpu::get().pipeline(n); }

}  // namespace

int main(int argc, char** argv) {
  std::vector<std::string> names;
  int NL = 16, calls = 256, trials = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--layers" && i + 1 < argc) NL = atoi(argv[++i]);
    else if (a == "--calls" && i + 1 < argc) calls = atoi(argv[++i]);
    else if (a == "--trials" && i + 1 < argc) trials = atoi(argv[++i]);
    else names.push_back(a);
  }
  if (names.empty()) names = {"qkvf", "d2", "d2F", "o", "rshared", "gu3r", "dn1_64", "head2"};
  bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD") && !lock.acquire([](const std::string&) {}))
    fatal("cannot take the GPU lock: %s", strerror(errno));
  Gpu& g = Gpu::get();
  g.compile_extra("Flint", kFlintKernelSource);
  std::mt19937 rng(7);
  std::vector<Layer> L;
  L.resize(size_t(NL));
  for (auto& l : L) {
    l.eg = rand_mat(I, D, EA, rng), l.eu = rand_mat(I, D, EA, rng), l.ed = rand_mat(D, I, EA, rng);
    l.sg = rand_mat(SI, D, 1, rng), l.su = rand_mat(SI, D, 1, rng), l.sd = rand_mat(D, SI, 1, rng);
    l.q = rand_mat(QN, D, 1, rng), l.k = rand_mat(KN, D, 1, rng), l.v = rand_mat(KN, D, 1, rng);
    l.o = rand_mat(D, QN, 1, rng);
    l.router = g.alloc(size_t(E) * D * 2, "kb_router", false);
    uint16_t* r = l.router.as<uint16_t>();
    for (size_t i = 0; i < size_t(E) * D; ++i) r[i] = uint16_t(0x3c00 + (rng() & 0xff));
    l.norm = g.alloc(D * 2, "kb_norm", false);
    uint16_t* nw = l.norm.as<uint16_t>();
    for (uint32_t i = 0; i < D; ++i) nw[i] = 0x3f80;  // bf16 1.0
  }
  Buffer x = g.alloc(8 * QN * 4, "kb_x"), xn = g.alloc(8 * D * 4, "kb_xn"), y = g.alloc(8 * 8192 * 4, "kb_y");
  Buffer h = g.alloc((K6 + 1) * SI * 4 * 8, "kb_h"), idx = g.alloc(64 * 4, "kb_idx"), wts = g.alloc(64 * 4, "kb_wts");
  Buffer logits = g.alloc(8 * E * 4, "kb_logits"), cnt = g.alloc(16, "kb_cnt");
  memset(cnt.ptr(), 0, 16);
  for (uint32_t i = 0; i < 8 * QN; ++i) x.as<float>()[i] = float(int(rng() % 2001) - 1000) / 1000.0f;
  for (uint32_t i = 0; i < 8 * D; ++i) xn.as<float>()[i] = x.as<float>()[i];
  for (uint32_t i = 0; i < (K6 + 1) * SI * 8; ++i) h.as<float>()[i] = 0.01f;
  for (int j = 0; j < 6; ++j) idx.as<uint32_t>()[j] = uint32_t(j), wts.as<float>()[j] = 0.1f;
  wts.as<float>()[6] = 1.0f;
  for (uint32_t i = 0; i < E; ++i) logits.as<float>()[i] = i < 6 ? 5.0f - float(i) * 0.1f : -5.0f;
  const kp::Moe mp{D, I, SI, K6, 1e-6f};
  // attention state: a ring (528 rows) and a linear cache (32768 rows) per KV head, half, random; q, partials, counters
  const uint32_t kPos = getenv("KB_POS") ? uint32_t(atoi(getenv("KB_POS"))) : 1024;
  const uint32_t kSmax = getenv("KB_SMAX") ? uint32_t(atoi(getenv("KB_SMAX"))) : 12;  // flint_attn_d2's splits
  std::vector<float> cs(128);
  for (int i = 0; i < 64; ++i) {
    const float ang = float(kPos) * float(1.0 / std::pow(1e4, 2.0 * i / 128.0));
    cs[size_t(2 * i)] = float(std::cos(double(ang))), cs[size_t(2 * i + 1)] = float(std::sin(double(ang)));
  }
  Buffer kcr = g.alloc(4 * 528 * 128 * 2, "kb_kcr"), vcr = g.alloc(4 * 528 * 128 * 2, "kb_vcr");
  Buffer kcf = g.alloc(size_t(4) * 32768 * 128 * 2, "kb_kcf"), vcf = g.alloc(size_t(4) * 32768 * 128 * 2, "kb_vcf");
  for (Buffer* b : {&kcr, &vcr, &kcf, &vcf}) {
    _Float16* hp = b->as<_Float16>();
    for (size_t i = 0; i < b->size / 2; ++i) hp[i] = _Float16(float(int(rng() % 2001) - 1000) / 1000.0f);
  }
  Buffer q = g.alloc(32 * 128 * 4, "kb_q"), ao = g.alloc(32 * 128 * 4, "kb_ao");
  Buffer po = g.alloc(size_t(32) * 64 * 128 * 4 * 8, "kb_po"), pml = g.alloc(size_t(32) * 64 * 2 * 4 * 8, "kb_pml");
  Buffer acnt = g.alloc(64, "kb_acnt"), inv = g.alloc(64 * 4, "kb_inv");
  memset(acnt.ptr(), 0, 64);
  for (int i = 0; i < 64; ++i) inv.as<float>()[i] = float(1.0 / std::pow(1e4, 2.0 * i / 128.0));
  for (int i = 0; i < 32 * 128; ++i) q.as<float>()[i] = 0.01f * float(i % 97);

  // one dispatch of kernel n on layer l; returns the weight bytes it reads
  auto enc = [&](Encoder& e, const std::string& n, Layer& l) -> size_t {
    const size_t expert_gu = 2 * size_t(K6) * l.eg.bytes() / EA + l.sg.bytes() + l.su.bytes();
    const size_t expert_dn = size_t(K6) * l.ed.bytes() / EA + l.sd.bytes();
    if (n == "rshared") {
      const struct {
        uint32_t D, E, k, SI, hstride, nr;
        float eps;
        uint32_t split;
      } rp{D, E, K6, SI, SI, E / 8, 1e-6f, 0};
      e.use(pipe("flint_router_shared"));
      e.buf(0, x); e.buf(1, l.norm); e.buf(2, l.router); e.buf(3, logits); e.buf(4, xn);
      e.bytes(5, &rp, sizeof rp);
      e.buf(6, idx); e.buf(7, wts); e.buf(8, cnt);
      e.buf(9, l.sg.w); e.buf(10, l.sg.s); e.buf(11, l.sg.b);
      e.buf(12, l.su.w); e.buf(13, l.su.s); e.buf(14, l.su.b); e.buf(15, h);
      e.dispatch(E / 8 + (SI + 15) / 16, 1, 1, 256);
      return size_t(E) * D * 2 + l.sg.bytes() + l.su.bytes();
    }
    if (n == "gu3r") {  // the routed slots alone
      e.use(pipe("moe_gate_up3_r2"));
      e.buf(0, l.eg.w); e.buf(1, l.eg.s); e.buf(2, l.eg.b);
      e.buf(3, l.eu.w); e.buf(4, l.eu.s); e.buf(5, l.eu.b);
      e.buf(6, l.sg.w); e.buf(7, l.sg.s); e.buf(8, l.sg.b);
      e.buf(9, l.su.w); e.buf(10, l.su.s); e.buf(11, l.su.b);
      e.buf(12, xn); e.buf(13, logits); e.buf(14, logits); e.buf(15, h);
      e.val(16, mp);
      e.buf(17, idx); e.buf(18, wts); e.val(19, E);
      e.dispatch((I + 15) / 16, K6, 1, 256);
      return 2 * size_t(K6) * l.eg.bytes() / EA;
    }
    if (n == "gu3" || n == "gu1") {
      const bool v3 = n == "gu3";
      e.use(pipe(v3 ? "moe_gate_up3_r2" : "moe_gate_up"));
      e.buf(0, l.eg.w); e.buf(1, l.eg.s); e.buf(2, l.eg.b);
      e.buf(3, l.eu.w); e.buf(4, l.eu.s); e.buf(5, l.eu.b);
      e.buf(6, l.sg.w); e.buf(7, l.sg.s); e.buf(8, l.sg.b);
      e.buf(9, l.su.w); e.buf(10, l.su.s); e.buf(11, l.su.b);
      if (v3) {
        e.buf(12, xn); e.buf(13, logits); e.buf(14, logits); e.buf(15, h);
        e.val(16, mp);
        e.buf(17, idx); e.buf(18, wts); e.val(19, E);
        e.dispatch((SI + 15) / 16, K6 + 1, 1, 256);
      } else {
        e.buf(12, x); e.buf(13, l.norm); e.buf(14, idx); e.buf(15, h);
        e.val(16, mp);
        e.dispatch((SI + 31) / 32, K6 + 1, 1, 256);
      }
      return expert_gu;
    }
    if (n == "dn1" || n == "dn2w" || n == "dn1_128" || n == "dn1_64") {
      const bool v1 = n != "dn2w";
      const uint32_t nsg = n == "dn1_128" ? 4 : n == "dn1_64" ? 2 : 8;
      e.use(pipe(v1 ? "moe_down" : "moe_down2_w"));
      e.buf(0, l.ed.w); e.buf(1, l.ed.s); e.buf(2, l.ed.b);
      e.buf(3, l.sd.w); e.buf(4, l.sd.s); e.buf(5, l.sd.b);
      e.buf(6, h); e.buf(7, idx); e.buf(8, wts); e.buf(9, y);
      e.val(10, mp);
      e.dispatch(v1 ? (D + 4 * nsg - 1) / (4 * nsg) : (D + 15) / 16, 1, 1, 32 * nsg);
      return expert_dn;
    }
    if (n.rfind("head", 0) == 0) {  // headN: the LM head (150144 x 2048, the final norm folded), N simdgroups
      static Mat hm = [&] {
        std::mt19937 r2(11);
        return rand_mat(150144, D, 1, r2);
      }();
      static Buffer hy = Gpu::get().alloc(size_t(150144) * 4, "kb_head_y");
      const uint32_t nsg = uint32_t(atoi(n.c_str() + 4));
      kp::Qmvf p{};
      p.K = D, p.R = 1, p.x_ld = D, p.nseg = 1, p.eps = 1e-6f, p.write_xn = 0;
      const uint32_t tg = (hm.N + 4 * nsg - 1) / (4 * nsg);
      p.N[0] = hm.N, p.y_ld[0] = hm.N, p.add[0] = 0, p.tg_begin[0] = 0, p.tg_begin[1] = tg;
      for (int i = 0; i < 4; ++i) {
        e.buf(i * 4 + 0, hm.w); e.buf(i * 4 + 1, hm.s); e.buf(i * 4 + 2, hm.b); e.buf(i * 4 + 3, hy);
      }
      e.use(pipe("qmvf_b4_t1_r4_norm"));
      e.buf(16, x); e.val(17, p); e.buf(18, l.norm); e.buf(19, x);
      e.dispatch(1, tg, 1, 32 * nsg);
      return hm.bytes();
    }
    if (n == "qkvf") {
      kp::Qmvf p{};
      p.K = D, p.R = 1, p.x_ld = D, p.nseg = 3, p.eps = 1e-6f, p.write_xn = 0;
      const Mat* m[3] = {&l.q, &l.k, &l.v};
      uint32_t tg = 0, off = 0;
      for (int i = 0; i < 4; ++i) {
        const Mat& mm = *m[std::min(i, 2)];
        if (i < 3) {
          p.N[i] = mm.N, p.y_ld[i] = 5120, p.add[i] = 0, p.tg_begin[i] = tg;
          tg += (mm.N + 7) / 8;
        }
        e.buf(i * 4 + 0, mm.w); e.buf(i * 4 + 1, mm.s); e.buf(i * 4 + 2, mm.b);
        e.buf(i * 4 + 3, y, size_t(i < 3 ? off : 0) * 4);
        if (i < 3) off += mm.N;
      }
      p.tg_begin[3] = tg;
      e.use(pipe("qmvf_b4_t1_r4_norm"));
      e.buf(16, x); e.val(17, p); e.buf(18, l.norm); e.buf(19, x);
      e.dispatch(1, tg, 1, 64);
      return l.q.bytes() + l.k.bytes() + l.v.bytes();
    }
    if (n == "o") {
      kp::Qmvf p{};
      p.K = QN, p.R = 1, p.x_ld = QN, p.nseg = 1, p.eps = 1e-6f, p.write_xn = 0;
      p.N[0] = D, p.y_ld[0] = D, p.add[0] = 1, p.tg_begin[0] = 0, p.tg_begin[1] = D / 8;
      for (int i = 0; i < 4; ++i) {
        e.buf(i * 4 + 0, l.o.w); e.buf(i * 4 + 1, l.o.s); e.buf(i * 4 + 2, l.o.b); e.buf(i * 4 + 3, y);
      }
      e.use(pipe("qmvf_b4_t1_r4"));
      e.buf(16, x); e.val(17, p); e.buf(18, x); e.buf(19, x);
      e.dispatch(1, D / 8, 1, 64);
      return l.o.bytes();
    }
    if (n == "o2") {
      const kp::Qmv qp{D, QN, QN, D, 1e-6f};
      e.use(pipe("qmv2_b4_g64_r2_add"));
      e.buf(0, l.o.w); e.buf(1, l.o.s); e.buf(2, l.o.b);
      e.buf(3, x); e.buf(4, y); e.val(5, qp); e.buf(6, y); e.buf(7, y);
      e.dispatch((D + 15) / 16, 1, 1, 256);
      return l.o.bytes();
    }
    if (n == "router") {
      const struct {
        uint32_t R, D, E, k;
        float eps;
        uint32_t write_xn, split;
      } rp{1, D, E, K6, 1e-6f, 1, 0};
      e.use(pipe("flint_router_topk"));
      e.buf(0, x); e.buf(1, l.norm); e.buf(2, l.router); e.buf(3, logits); e.buf(4, xn);
      e.bytes(5, &rp, sizeof rp);
      e.buf(6, idx); e.buf(7, wts); e.buf(8, cnt);
      e.dispatch(E / 4, 1, 1, 128);
      return size_t(E) * D * 2;
    }
    if (n == "prep" || n == "partial" || n == "reduce" || n == "prepF" || n == "partialF" || n == "reduceF") {
      // decode attention of one row at position kPos: a sliding-window layer (ring of 528 rows, window 513), or with
      // a trailing F a full-attention layer (a linear cache)
      const bool full = n.back() == 'F';
      const std::string b = full ? n.substr(0, n.size() - 1) : n;
      const uint32_t rows = full ? 32768u : 528u, window = full ? 0u : 513u, pos = kPos;
      Buffer& kc = full ? kcf : kcr;
      Buffer& vc = full ? vcf : vcr;
      const uint32_t k0 = window && pos + 1 > window ? pos + 1 - window : 0u;
      if (b == "prep") {
        const struct { uint32_t R, pos0, Hq, Hkv, rows, ring, rope, chunk, qkv_ld; float eps; uint32_t tree; } pp{
            1, pos, 32, 4, rows, full ? 0u : 1u, full ? 0u : 1u, 0, 5120, 1e-6f, 0};
        e.use(pipe("flint_prep"));
        e.buf(0, y); e.buf(1, q); e.buf(2, l.norm); e.buf(3, l.norm); e.buf(4, inv); e.buf(5, kc); e.buf(6, vc);
        e.bytes(7, &pp, sizeof pp); e.buf(8, inv);
        e.dispatch(40, 1, 1, 128);
        return 0;
      }
      const uint32_t Lk = pos + 1 - k0;
      uint32_t chunk = std::max<uint32_t>(64, (Lk + 19) / 20);
      chunk = std::max(chunk, (Lk + 63) / 64);
      chunk = (chunk + 31) / 32 * 32;
      const uint32_t nch = (Lk + chunk - 1) / chunk;
      const struct { uint32_t R, pos0, Hq, Hkv, rows, ring, window, k0, chunk, nchunks; float scale; } ap{
          1, pos, 32, 4, rows, full ? 0u : 1u, window, k0, chunk, nch, 0.0884f};
      if (b == "partial") {
        e.use(pipe("flint_attn_partial"));
        e.buf(0, q); e.buf(1, kc); e.buf(2, vc); e.buf(3, po); e.buf(4, pml); e.bytes(5, &ap, sizeof ap); e.buf(6, pml);
        e.dispatch(nch, 4, 1, 256);
        return 0;
      }
      if (b == "reduce") {
        e.use(pipe("flint_attn_reduce"));
        e.buf(0, po); e.buf(1, pml); e.buf(2, l.norm); e.buf(3, ao); e.bytes(4, &ap, sizeof ap);
        e.dispatch(32, 1, 1, 128);
        return 0;
      }
      fatal("unknown kernel %s", n.c_str());
    }
    if (n == "d2" || n == "d2F") {
      // one decode row's attention at position kPos: a sliding-window layer, or (d2F) a full-attention one
      const bool full = n == "d2F";
      const uint32_t rows = full ? 32768u : 528u, window = full ? 0u : 513u, pos = kPos;
      Buffer& kc = full ? kcf : kcr;
      Buffer& vc = full ? vcf : vcr;
      const uint32_t k0 = window && pos + 1 > window ? pos + 1 - window : 0u;
      const uint32_t T = (pos + 1 - k0 + 31) / 32;
      uint32_t S = std::min(T, kSmax);
      const uint32_t tps = (T + S - 1) / S;
      S = (T + tps - 1) / tps;
      const struct { uint32_t pos, Hq, Hkv, rows, ring, rope, k0, tps, S; float eps, scale; uint32_t split; } dp{
          pos, 32, 4, rows, full ? 0u : 1u, full ? 0u : 1u, k0, tps, S, 1e-6f, 0.0884f, 0};
      e.use(pipe("flint_attn_d2"));
      e.buf(0, y); e.buf(1, l.norm); e.buf(2, l.norm); e.bytes(3, cs.data(), cs.size() * 4); e.buf(4, kc); e.buf(5, vc);
      e.buf(6, l.norm); e.buf(7, ao); e.buf(8, po); e.buf(9, pml); e.buf(10, acnt); e.bytes(11, &dp, sizeof dp);
      e.dispatch(S, 4, 1, 128);
      return 0;
    }
    fatal("unknown kernel %s", n.c_str());
  };
  // "A+B": A then B on each call; "A|B": A and B side by side (a group of a concurrent encoder)
  auto enc_seq = [&](Encoder& e, const std::string& n, Layer& l) -> size_t {
    size_t b = 0, at = 0;
    while (at <= n.size()) {
      size_t plus = n.find('+', at);
      if (plus == std::string::npos) plus = n.size();
      const std::string part = n.substr(at, plus - at);
      const bool grp = part.find('|') != std::string::npos;
      if (grp) e.group_begin();
      size_t a2 = 0;
      while (a2 <= part.size()) {
        size_t bar = part.find('|', a2);
        if (bar == std::string::npos) bar = part.size();
        b += enc(e, part.substr(a2, bar - a2), l);
        a2 = bar + 1;
      }
      if (grp) e.group_end();
      at = plus + 1;
    }
    return b;
  };
  std::vector<std::vector<double>> us(names.size());
  std::vector<size_t> bytes(names.size());
  for (int t = 0; t < trials; ++t)
    for (size_t k = 0; k < names.size(); ++k) {
      auto e = names[k].find('|') != std::string::npos || getenv("KB_GROUPED") ? g.begin_grouped() : g.begin();
      size_t b = 0;
      for (int c = 0; c < calls; ++c) b = enc_seq(*e, names[k], L[size_t(c % NL)]);
      e->commit_and_wait();
      us[k].push_back(e->gpu_ms() * 1000.0 / calls);
      bytes[k] = b;
      if (names[k] == "router" || names[k] == "rshared") {  // (ids valid again)
        for (uint32_t i = 0; i < E; ++i) logits.as<float>()[i] = i < 6 ? 5.0f - float(i) * 0.1f : -5.0f;
        for (int j = 0; j < 6; ++j) idx.as<uint32_t>()[j] = uint32_t(j);
      }
    }
  for (size_t k = 0; k < names.size(); ++k) {
    std::sort(us[k].begin(), us[k].end());
    const double med = us[k][us[k].size() / 2];
    printf("%-10s %8.2f us/call  %6.1f GB/s  (%.2f MB; min %.2f, max %.2f us)\n", names[k].c_str(), med,
           double(bytes[k]) / med / 1e3, double(bytes[k]) / 1e6, us[k].front(), us[k].back());
  }
  return 0;
}
