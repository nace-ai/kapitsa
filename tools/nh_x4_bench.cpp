// NVFP4 kernels of the x4 set (src/kernels/nemotron/nh_x_fp4.metal) against today's (nh_mv1 / nh_mv / nh_v8 /
// nh_gemm) on the Mamba-2 projection shapes: decode (1 row), verify (8 rows), short blocks (up to 32 rows).
// Each kernel runs once for a check against an fp64 product of the dequantized weights (rel.err = max |y - ref| /
// max |ref|), then is timed: 10 back-to-back calls per command buffer, median of --reps, on the same weights
// ("same", nh_bench's method) and rotating over copies of the weights (~250 MB per shape, so no call finds the
// previous one's weights in the system cache: "rot", what a model's 40 different layers see).
//   nh_x4_bench [--reps n] [--copies n] [--gpr] [--norot] [--same] [filter ...]
// (filter: substrings of the case label, "a+b" needs both; --same also times on the same weights, --norot only so)
#include "metal/gpu.h"
#include "model/nemotron_h.h"
#include "util/log.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace ni {
extern const char* const kNhKernelSource;
}
using namespace ni;

namespace {

uint64_t g_rng = 0x9E3779B97F4A7C15ull;
uint64_t rnd() {  // splitmix64
  uint64_t z = (g_rng += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
float nrand() {  // ~N(0, 1)
  float s = 0;
  for (int i = 0; i < 4; ++i) s += float(rnd() & 0xFFFF) / 65535.0f;
  return (s - 2.0f) * 1.7320508f;
}

double e4m3(uint8_t x) {
  if (x == 0x7F) return 0.0;
  const int e = (x >> 3) & 0xF, m = x & 7;
  return e == 0 ? std::ldexp(double(m), -9) : std::ldexp(1.0 + double(m) / 8.0, e - 7);
}
const double kE2M1[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

// Logical NVFP4 matrix: code[n][k] (E2M1 nibble, s e1 e0 m), scale[n][k / 16] (UE4M3), tensor scale gs
struct Fp4 {
  uint32_t N = 0, K = 0;
  float gs = 1.0f;
  std::vector<uint8_t> code, sc;
  std::vector<float> w;  // dequantized (exact in fp32)
};

Fp4 make_fp4(uint32_t N, uint32_t K, float gs) {
  Fp4 m;
  m.N = N;
  m.K = K;
  m.gs = gs;
  m.code.resize(size_t(N) * K);
  m.sc.resize(size_t(N) * K / 16);
  for (size_t i = 0; i < m.code.size(); i += 16) {
    uint64_t r = rnd();
    for (int j = 0; j < 16; ++j) m.code[i + j] = uint8_t((r >> (4 * j)) & 15);
  }
  for (size_t i = 0; i < m.sc.size(); ++i) {
    // mostly realistic block scales (e 4..14), some subnormal and extreme ones, a few zero blocks
    const uint64_t r = rnd();
    const int sel = int(r & 63);
    uint8_t b;
    if (sel == 0) b = uint8_t((r >> 8) & 7);            // subnormal / zero
    else if (sel == 1) b = uint8_t(0x70 + ((r >> 8) % 15));  // the largest (0x7E max)
    else b = uint8_t(0x20 + ((r >> 8) % 0x50));           // e 4..13
    if (b == 0x7F) b = 0x7E;
    m.sc[i] = b;
  }
  m.w.resize(size_t(N) * K);
  for (size_t i = 0; i < m.w.size(); ++i) m.w[i] = float(kE2M1[m.code[i]] * e4m3(m.sc[i / 16]) * gs);
  return m;
}

// layout L0: byte i of a 16-block = element i | element i + 8 << 4 (Kapitsa's repack)
// layout L1: word h (h = 0, 1) of a block: nibble j = element 4h + j, nibble 4 + j = element 8 + 4h + j
// layout L2: L1 with codes stored as (m2 m1 m0 s)
void pack(const Fp4& m, int layout, uint8_t* q, uint8_t* s) {
  const size_t nb = size_t(m.N) * m.K / 16;
  for (size_t b = 0; b < nb; ++b) {
    const uint8_t* c = m.code.data() + b * 16;
    uint8_t* d = q + b * 8;
    if (layout == 0) {
      for (int i = 0; i < 8; ++i) d[i] = uint8_t(c[i] | (c[i + 8] << 4));
    } else {
      auto enc = [&](uint8_t v) -> uint32_t { return layout == 2 ? uint32_t(((v & 7) << 1) | (v >> 3)) : v; };
      for (int h = 0; h < 2; ++h) {
        uint32_t w = 0;
        for (int j = 0; j < 4; ++j) w |= enc(c[4 * h + j]) << (4 * j) | enc(c[8 + 4 * h + j]) << (16 + 4 * j);
        memcpy(d + 4 * h, &w, 4);
      }
    }
  }
  memcpy(s, m.sc.data(), nb);
}

struct GpuMat {
  Buffer q, s;            // copies x (codes | scales)
  size_t qb = 0, sb = 0;  // bytes of one copy
  int copies = 0;
};

GpuMat upload(const Fp4& m, int layout, int copies) {
  Gpu& g = Gpu::get();
  GpuMat r;
  r.qb = size_t(m.N) * m.K / 2;
  r.sb = size_t(m.N) * m.K / 16;
  r.copies = copies;
  r.q = g.alloc(r.qb * copies, "x4q", false);
  r.s = g.alloc(r.sb * copies, "x4s", false);
  pack(m, layout, r.q.as<uint8_t>(), r.s.as<uint8_t>());
  for (int c = 1; c < copies; ++c) {
    memcpy(r.q.as<uint8_t>() + c * r.qb, r.q.as<uint8_t>(), r.qb);
    memcpy(r.s.as<uint8_t>() + c * r.sb, r.s.as<uint8_t>(), r.sb);
  }
  return r;
}

void parallel_for(size_t n, const std::function<void(size_t, size_t)>& f) {
  const size_t nt = std::max(1u, std::min(12u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (size_t t = 0; t < nt; ++t) th.emplace_back([&, t] { f(n * t / nt, n * (t + 1) / nt); });
  for (auto& x : th) x.join();
}

// ref[t][n] = sum_k w[n][k] x[t][k] in fp64
std::vector<double> reference(const Fp4& m, const std::vector<float>& x, uint32_t M) {
  std::vector<double> ref(size_t(M) * m.N);
  parallel_for(m.N, [&](size_t n0, size_t n1) {
    for (size_t n = n0; n < n1; ++n) {
      const float* wr = m.w.data() + n * m.K;
      for (uint32_t t = 0; t < M; ++t) {
        const float* xr = x.data() + size_t(t) * m.K;
        double a = 0;
        for (uint32_t k = 0; k < m.K; ++k) a += double(wr[k]) * double(xr[k]);
        ref[size_t(t) * m.N + n] = a;
      }
    }
  });
  return ref;
}

// ---- GPR / stack from a binary archive (ni_compile's parser) ----
uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint32_t le32(const uint8_t* p) { uint32_t v; memcpy(&v, p, 4); return v; }
uint64_t le64(const uint8_t* p) { uint64_t v; memcpy(&v, p, 8); return v; }
bool macho_section(const uint8_t* m, size_t n, const char* seg, const char* sect, const uint8_t** out, size_t* len) {
  if (n < 32) return false;
  uint32_t ncmds = le32(m + 16);
  size_t off = 32;
  for (uint32_t i = 0; i < ncmds && off + 8 <= n; ++i) {
    uint32_t cmd = le32(m + off), csz = le32(m + off + 4);
    if (cmd == 0x19 && off + 72 <= n) {
      uint32_t nsect = le32(m + off + 64);
      for (uint32_t s = 0; s < nsect; ++s) {
        const uint8_t* sd = m + off + 72 + 80 * s;
        if (sd + 80 > m + n) return false;
        if (strncmp((const char*)sd, sect, 16) == 0 && strncmp((const char*)sd + 16, seg, 16) == 0) {
          uint64_t size = le64(sd + 40);
          uint32_t soff = le32(sd + 48);
          if (soff + size > n) return false;
          *out = m + soff;
          *len = size;
          return true;
        }
      }
    }
    off += csz;
  }
  return false;
}
void kernel_stats(const std::vector<uint8_t>& b, long* stack, long* gprs) {
  *stack = *gprs = -1;
  if (b.size() < 8) return;
  uint32_t nfat = be32(b.data() + 4);
  for (uint32_t i = 0; i < nfat && 8 + 20 * (i + 1) <= b.size(); ++i) {
    const uint8_t* e = b.data() + 8 + 20 * i;
    if (be32(e) != 0x01000013) continue;
    uint32_t off = be32(e + 8), size = be32(e + 12);
    if (off + size > b.size()) return;
    const uint8_t *inner, *md;
    size_t ninner, nmd;
    if (!macho_section(b.data() + off, size, "__TEXT", "__compute", &inner, &ninner)) return;
    if (!macho_section(inner, ninner, "__GPU_METADATA", "__compute", &md, &nmd)) return;
    std::vector<uint32_t> w(nmd / 4);
    for (size_t k = 0; k < w.size(); ++k) w[k] = le32(md + 4 * k);
    for (size_t k = 4; k + 2 < w.size(); ++k) {
      uint32_t f = w[k];
      if (f >> 24 != 1 || (((f >> 16) & 0xff) != 1 && ((f >> 16) & 0xff) != 3) || f == 0x01000000 || (f & 0xff) > 3)
        continue;
      uint32_t c = w[k + 1];
      bool seen = c > 0 && (w[k - 1] == c || w[k - 2] == c || w[k - 3] == c || w[k - 4] == c);
      if (seen && w[k + 2] >= 4 && w[k + 2] <= 256) { *stack = c; *gprs = w[k + 2]; return; }
      if (c <= 256) { *stack = 0; *gprs = c; return; }
    }
    return;
  }
}
std::string gpr_str(const std::string& fn) {
  const char* tmp = getenv("TMPDIR");
  const std::string path = std::string(tmp ? tmp : "/tmp") + "/x4_" + fn + ".bin";
  if (!Gpu::get().archive(fn, path)) return "gpr ?";
  std::ifstream f(path, std::ios::binary);
  std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  remove(path.c_str());
  long st, gp;
  kernel_stats(b, &st, &gp);
  char buf[64];
  snprintf(buf, sizeof buf, "gpr %ld%s", gp, st > 0 ? (" stack " + std::to_string(st)).c_str() : "");
  return buf;
}

// ---- kernels ----
enum Kind { MV, V8, GEMM };
struct KV {
  std::string name;
  Kind kind;
  int layout;
  uint32_t rows_tg, threads;
  uint32_t sk = 1;  // split-K threadgroups per row tile (buffers 5, 6)
  std::string label() const {
    char b[96];
    if (sk > 1 || name.rfind("nh_x4_mm", 0) == 0) snprintf(b, sizeof b, "%s/%u/s%u", name.c_str(), threads / 32, sk);
    else snprintf(b, sizeof b, "%s/%u", name.c_str(), threads / 32);
    return b;
  }
};

struct NhMvP {
  uint32_t N, K, xs, ys;
  float gs;
  uint32_t R;
};
struct NhV8P_ {
  uint32_t N, K, x_ld, y_ld, R;
  float gs;
  uint32_t epi;
};
struct NhGemmP_ {
  uint32_t M, N, K, x_ld, y_ld;
  float gs;
  uint32_t epi, qe, se, k;
};

// split-K scratch (part) and counters (cnt) of an nh_x4_mm kernel: exactly the documented sizes, so shader
// validation (MTL_SHADER_VALIDATION=1) catches any write past them
size_t mm_part_bytes(const KV& kv, uint32_t N, uint32_t M) {
  const uint32_t tiles = (N + kv.rows_tg - 1) / kv.rows_tg, tb = (M + 7) / 8;
  return size_t(kv.sk) * tiles * (kv.threads / 32) * 512 * tb;
}
size_t mm_cnt_bytes(const KV& kv, uint32_t N) { return size_t((N + kv.rows_tg - 1) / kv.rows_tg) * 4; }

void encode(Encoder& e, const KV& kv, const GpuMat& gm, int copy, const Buffer& x, const Buffer& y, uint32_t N,
            uint32_t K, uint32_t M, uint32_t R, float gs, uint32_t epi, const Buffer& dummy, uint32_t x_ld = 0,
            uint32_t y_ld = 0, const Buffer* part = nullptr, const Buffer* cnt = nullptr) {
  if (!x_ld) x_ld = K;
  if (!y_ld) y_ld = N;
  Gpu& g = Gpu::get();
  e.use(g.pipeline(kv.name));
  if (kv.kind == MV) {
    const NhMvP p{N, K, K, N, gs, 1};
    e.buf(0, gm.q, size_t(copy) * gm.qb);
    e.buf(1, gm.s, size_t(copy) * gm.sb);
    e.buf(2, x);
    e.buf(3, y);
    e.bytes(4, &p, sizeof p);
    e.dispatch((N + kv.rows_tg - 1) / kv.rows_tg, 1, 1, kv.threads);
  } else if (kv.kind == V8) {
    const NhV8P_ p{N, K, x_ld, y_ld, R, gs, epi};
    e.buf(0, x);
    e.buf(1, gm.q, size_t(copy) * gm.qb);
    e.buf(2, gm.s, size_t(copy) * gm.sb);
    e.buf(3, y);
    e.bytes(4, &p, sizeof p);
    e.buf(8, dummy);
    if (kv.name.rfind("nh_x4_mm", 0) == 0) {  // split-K grid: x parts, y row tiles; buffers 5, 6
      if (!part || !cnt) fatal("%s: no split-K buffers", kv.name.c_str());
      e.buf(5, *part);
      e.buf(6, *cnt);
      e.dispatch(kv.sk, (N + kv.rows_tg - 1) / kv.rows_tg, 1, kv.threads);
    } else {
      e.dispatch((N + kv.rows_tg - 1) / kv.rows_tg, 1, 1, kv.threads);
    }
  } else {
    const uint32_t bm = kv.rows_tg;  // GEMM: rows_tg holds the tile's token rows
    const NhGemmP_ p{R, N, K, K, N, gs, epi, 0, 0, 1};
    e.buf(0, x);
    e.buf(1, gm.q, size_t(copy) * gm.qb);
    e.buf(2, gm.s, size_t(copy) * gm.sb);
    e.buf(3, y);
    e.bytes(4, &p, sizeof p);
    e.buf(5, dummy);
    e.buf(6, dummy);
    e.buf(7, dummy);
    e.buf(8, dummy);
    e.dispatch((N + 63) / 64, (R + bm - 1) / bm, 1, 128);
  }
  (void)M;
}

// The layout-L2 versions of the FP4 branches of nh_gemm.metal's nh_dq16 and nh_ane.metal's nh_w4 (what the port
// would paste), checked against the logical weights
const char* kL2PatchSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
#define NI_INLINE inline __attribute__((always_inline))
constant float nh_fp4_lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
NI_INLINE float nh_ue4m3(uint b) { return (b == 0x7Fu) ? 0.0f : float(as_type<half>(ushort(b << 7))) * 128.0f; }
// nh_dq16, FMT 1 (16 consecutive weights of row n at k, k % 16 == 0, x their block scale)
NI_INLINE void dq16_l2(const device uchar* q, const device uchar* s, uint K, uint n, uint k, thread half* dst) {
    // FP4, layout L2: word h of a block holds element 4h + j in nibble j and 8 + 4h + j in nibble 4 + j (j = 0..3),
    // codes as (m2 m1 m0 s)
    const uint2 c = *(const device uint2*)(q + size_t(n) * (K / 2) + k / 2);
    const float sc = nh_ue4m3(s[size_t(n) * (K / 16) + k / 16]);
    for (int e = 0; e < 16; ++e) {
      const uint v = (((e >> 2) & 1) ? c.y : c.x) >> (4 * ((e & 3) + ((e >> 3) << 2))) & 15u;
      dst[e] = half(nh_fp4_lut[(v >> 1) | ((v & 1u) << 3)] * sc);
    }
}
// nh_w4, FMT 1 (4 weights of row n at [k, k + 4), k % 4 == 0, x block scale x gs)
NI_INLINE float4 w4_l2(const device uchar* q, const device uchar* s, float gs, uint K, uint n, uint k) {
    // FP4, layout L2: elements k .. k + 3 are nibbles 0-3 (k % 16 < 8) or 4-7 of the block's word (k / 4) & 1
    const uint blk = k / 16, i0 = k % 16;
    const uint w = *(const device uint*)(q + size_t(n) * (K / 2) + blk * 8 + 4 * ((i0 >> 2) & 1u)) >> (i0 >= 8 ? 16u : 0u);
    const float sc = nh_ue4m3(s[size_t(n) * (K / 16) + blk]) * gs;
    float4 r;
    for (int e = 0; e < 4; ++e) {
      const uint v = (w >> (4 * e)) & 15u;
      r[e] = nh_fp4_lut[(v >> 1) | ((v & 1u) << 3)];
    }
    return r * sc;
}
kernel void x4l2_check(const device uchar* q [[buffer(0)]], const device uchar* s [[buffer(1)]],
                       device float* o16 [[buffer(2)]], device float* o4 [[buffer(3)]], constant uint2& nk [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
  const uint K = nk.y, n = i / (K / 16), k = (i % (K / 16)) * 16;
  if (n >= nk.x) return;
  half d[16];
  dq16_l2(q, s, K, n, k, d);
  for (int e = 0; e < 16; ++e) o16[size_t(n) * K + k + e] = float(d[e]);
  for (int e = 0; e < 16; e += 4) {
    const float4 v = w4_l2(q, s, 1.0f, K, n, k + e);
    for (int u = 0; u < 4; ++u) o4[size_t(n) * K + k + e + u] = v[u];
  }
}
)MSL";

// Hardware facts the decode variants rely on (fast-math library, like the engine's)
const char* kProbeSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
// Streaming patterns over a [N][RB] byte matrix (RB = row bytes): a simdgroup reads G groups of RPI rows per step,
// each row by 32 / RPI lanes x 16 B (one uint4 per lane and row: RPI rows x (512 / RPI) B per load instruction);
// KS simdgroups of a threadgroup split the steps (interleaved), threadgroups split the rows.
template <int RPI, int G, int KS>
void x4p_stream_impl(const device uchar* w, device uint* out, uint N, uint RB, uint tg, uint sg, uint lane) {
  constexpr uint LPR = 32 / RPI, SB = LPR * 16;  // lanes per row, bytes per row and step
  const uint ks = sg % KS, cs = sg / KS;
  const uint r0 = (tg * 1 + cs) * RPI * G;  // one column slice per threadgroup (nsg = KS)
  const uint lr = lane / LPR, lc = lane % LPR;
  uint acc = 0;
  const device uchar* rp[G];
  for (int g = 0; g < G; ++g) rp[g] = w + size_t(min(r0 + uint(g) * RPI + lr, N - 1)) * RB + lc * 16;
  for (uint st = ks; st * SB < RB; st += KS) {
    _Pragma("clang loop unroll(full)")
    for (int g = 0; g < G; ++g) {
      const uint4 v = *(const device uint4*)(rp[g] + st * SB);
      acc ^= v.x + v.y + v.z + v.w;
    }
  }
  if (acc == 0x12345678u) out[tg] = acc;
}
#define X4P_STREAM(RPI, G, KS)                                                                                   \
  kernel void x4p_stream_r##RPI##_g##G##_k##KS(const device uchar* w [[buffer(0)]], device uint* out [[buffer(1)]], \
                                               constant uint2& nk [[buffer(2)]], uint tg [[threadgroup_position_in_grid]], \
                                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    x4p_stream_impl<RPI, G, KS>(w, out, nk.x, nk.y, tg, sg, lane);                                               \
  }
X4P_STREAM(1, 4, 4)
X4P_STREAM(1, 8, 4)
X4P_STREAM(2, 2, 4)
X4P_STREAM(2, 4, 4)
X4P_STREAM(4, 1, 4)
X4P_STREAM(4, 2, 4)
X4P_STREAM(8, 1, 4)
X4P_STREAM(8, 2, 4)
X4P_STREAM(8, 4, 4)
X4P_STREAM(8, 2, 8)
X4P_STREAM(8, 4, 2)
X4P_STREAM(8, 4, 1)
X4P_STREAM(4, 4, 2)
X4P_STREAM(4, 8, 1)
X4P_STREAM(1, 4, 1)
kernel void x4p_cvt(device float* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  // 0..127: UE4M3 via a half (b << 7) to float; 128..143: E2M1 bit trick (half bits) to float; 144..159: the same
  // value times a normal half in half arithmetic, then to float
  if (i < 128) out[i] = float(as_type<half>(ushort(i << 7)));
  else if (i < 144) {
    const uint n = i - 128;
    out[i] = float(as_type<half>(ushort(((n & 7u) << 9) | ((n & 8u) << 12))));
  } else if (i < 160) {
    const uint n = i - 144;
    const half h = as_type<half>(ushort(((n & 7u) << 9) | ((n & 8u) << 12)));
    out[i] = float(h * half(out[200]));
  }
}
)MSL";

}  // namespace

int main(int argc, char** argv) {
  int reps = 9, copies = 6, limit = 240;
  bool gpr = false, rot = true, same = false, small = false;
  std::vector<std::string> only;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
    else if (a == "--copies" && i + 1 < argc) copies = atoi(argv[++i]);
    else if (a == "--gpr") gpr = true;
    else if (a == "--norot") rot = false;
    else if (a == "--same") same = true;
    else if (a == "--limit" && i + 1 < argc) limit = atoi(argv[++i]);
    else if (a == "--small") small = true;
    else only.push_back(a);
  }
  auto want = [&](const std::string& n) {
    if (only.empty()) return true;
    for (auto& o : only) {
      // a filter "a+b" needs every part
      size_t p = 0;
      bool all = true;
      while (p <= o.size()) {
        size_t q = o.find('+', p);
        if (q == std::string::npos) q = o.size();
        if (n.find(o.substr(p, q - p)) == std::string::npos) all = false;
        p = q + 1;
      }
      if (all) return true;
    }
    return false;
  };
  // a hard wall-clock limit for the whole process (SIGALRM ends it): a hung or runaway GPU job cannot outlive it
  alarm(unsigned(limit));
  if (small) {  // validation runs: tiny shapes, one weight copy, few rounds
    rot = false;
    reps = std::min(reps, 2);
  }
  auto want_shape = [&](const char* n) {
    for (auto& o : only)
      if (o.rfind(n, 0) == 0) return true;
    return false;
  };
  Gpu& g = Gpu::get();
  g.compile_extra("Nemotron-H", kNhKernelSource);
  Buffer dummy = g.alloc(4096, "dummy");

  if (want("probe")) {
    g.compile_extra("x4probe", kProbeSrc);
    Buffer o = g.alloc(1024, "probe");
    o.as<float>()[200] = 1.0f / 256.0f;  // a normal half factor: subnormal x normal -> deeper subnormal
    auto e = g.begin();
    e->use(g.pipeline("x4p_cvt"));
    e->buf(0, o);
    e->dispatch(1, 1, 1, 160);
    e->commit_and_wait();
    int bad = 0;
    for (int b = 0; b < 127; ++b) {
      const double want_v = e4m3(uint8_t(b)) / 256.0;
      if (o.as<float>()[b] != float(want_v)) {
        if (bad++ < 4) printf("probe: e4m3 %02x via half -> %g, want %g\n", b, o.as<float>()[b], want_v);
      }
    }
    for (int n = 0; n < 16; ++n) {
      const double v = kE2M1[n] * std::ldexp(1.0, -14);
      if (o.as<float>()[128 + n] != float(v) && !(v == 0 && o.as<float>()[128 + n] == 0)) {
        if (bad++ < 8) printf("probe: e2m1 %x via half bits -> %g, want %g\n", n, o.as<float>()[128 + n], v);
      }
      const double v2 = v / 256.0;
      if (o.as<float>()[144 + n] != float(v2) && !(v2 == 0 && o.as<float>()[144 + n] == 0)) {
        if (bad++ < 12) printf("probe: e2m1 %x x 2^-8 in half -> %g, want %g\n", n, o.as<float>()[144 + n], v2);
      }
    }
    printf("probe: half subnormals %s (%d mismatches)\n", bad ? "NOT preserved" : "preserved", bad);
  }
  if (want("l2patch")) {
    g.compile_extra("x4l2", kL2PatchSrc);
    const uint32_t N = 61, K = 4096;
    Fp4 m = make_fp4(N, K, 1.0f);
    GpuMat gm = upload(m, 2, 1);
    Buffer o16 = g.alloc(size_t(N) * K * 4, "o16"), o4 = g.alloc(size_t(N) * K * 4, "o4");
    const uint32_t nk[2] = {N, K};
    auto e = g.begin();
    e->use(g.pipeline("x4l2_check"));
    e->buf(0, gm.q);
    e->buf(1, gm.s);
    e->buf(2, o16);
    e->buf(3, o4);
    e->bytes(4, nk, 8);
    e->dispatch((N * K / 16 + 255) / 256, 1, 1, 256);
    e->commit_and_wait();
    size_t bad16 = 0, bad4 = 0;
    for (size_t i = 0; i < size_t(N) * K; ++i) {
      if (o16.as<float>()[i] != float(__fp16(m.w[i]))) ++bad16;  // nh_dq16 rounds to half
      if (o4.as<float>()[i] != m.w[i]) ++bad4;
    }
    printf("l2patch: nh_dq16 L2 %zu / nh_w4 L2 %zu mismatches of %zu weights\n", bad16, bad4, size_t(N) * K);
    // the CPU repack for nh_repack_rows (ModelOpt bytes: byte i of a block = element 2i | element 2i + 1 << 4)
    std::vector<uint8_t> mo(size_t(N) * K / 2), l2(size_t(N) * K / 2), want_l2(size_t(N) * K / 2), sc(size_t(N) * K / 16);
    for (size_t i = 0; i < mo.size(); ++i) mo[i] = uint8_t(m.code[2 * i] | (m.code[2 * i + 1] << 4));
    for (size_t b = 0; b < mo.size() / 8; ++b) {
      const uint8_t* src = mo.data() + b * 8;
      // ---- the snippet ----
      uint32_t w[2] = {0, 0};
      for (int e = 0; e < 16; ++e) {
        const uint32_t c = (src[e >> 1] >> (4 * (e & 1))) & 15u;  // ModelOpt: byte e / 2, the low nibble the even element
        w[(e >> 2) & 1] |= (((c & 7u) << 1) | (c >> 3)) << (4 * ((e & 3) + ((e >> 3) << 2)));
      }
      memcpy(l2.data() + b * 8, w, 8);
      // ----
    }
    pack(m, 2, want_l2.data(), sc.data());
    printf("l2patch: repack snippet %s\n", l2 == want_l2 ? "matches layout L2" : "DIFFERS from layout L2");
    g.release(o16);
    g.release(o4);
    g.release(gm.q);
    g.release(gm.s);
  }
  if (want("stream")) {
    g.compile_extra("x4probe", kProbeSrc);
    // in_proj-sized codes (18560 x 2048 B), 6 rotating copies
    const uint32_t N = 18560, RB = 2048;
    const int NC = 6;
    Buffer w = g.alloc(size_t(N) * RB * NC, "stream", false), o = g.alloc(size_t(N) * 4, "o");  // out[tg], tg < N
    memset(w.ptr(), 0x5A, size_t(N) * RB * NC);
    struct SV { int rpi, gg, ks; };
    std::vector<SV> svs = {{1, 4, 4}, {1, 8, 4}, {2, 2, 4}, {2, 4, 4}, {4, 1, 4}, {4, 2, 4}, {8, 1, 4}, {8, 2, 4},
                           {8, 4, 4}, {8, 2, 8}, {8, 4, 2}, {8, 4, 1}, {4, 4, 2}, {4, 8, 1}, {1, 4, 1}};
    std::vector<std::vector<double>> ts(svs.size());
    int cyc = 0;
    for (int rd = 0; rd < reps + 2; ++rd)
      for (size_t i = 0; i < svs.size(); ++i) {
        const SV& v = svs[i];
        char nm[64];
        snprintf(nm, sizeof nm, "x4p_stream_r%d_g%d_k%d", v.rpi, v.gg, v.ks);
        const uint32_t rows_tg = uint32_t(v.rpi * v.gg);
        const uint32_t nk[2] = {N, RB};
        auto e = g.begin();
        for (int c = 0; c < 10; ++c) {
          cyc = (cyc + 1) % NC;
          e->use(g.pipeline(nm));
          e->buf(0, w, size_t(cyc) * N * RB);
          e->buf(1, o);
          e->bytes(2, nk, 8);
          e->dispatch((N + rows_tg - 1) / rows_tg, 1, 1, 32 * v.ks);
        }
        e->commit_and_wait();
        if (rd >= 2) ts[i].push_back(e->gpu_ms() / 10);
      }
    for (size_t i = 0; i < svs.size(); ++i) {
      std::sort(ts[i].begin(), ts[i].end());
      const double ms = ts[i][ts[i].size() / 2];
      printf("stream rows/instr %d, groups %d, k-slices %d: %7.4f ms %6.1f GB/s\n", svs[i].rpi, svs[i].gg, svs[i].ks, ms,
             double(N) * RB / ms / 1e6);
    }
    g.release(w);
    g.release(o);
  }

  struct Shape {
    const char* name;
    uint32_t N, K;
    float gs;
  };
  // in / out / odd always; w8 (diagnostic) and the LiLiCorr drafter's fc / gate-up / down only when named
  const Shape big[] = {{"in", 18560, 4096, 0.0123f}, {"out", 4096, 8192, 0.37f}, {"odd", 997, 4096, 1.0f},
                       {"w8", 8192, 8192, 0.5f},      {"dfc", 4096, 24576, 0.2f}, {"dup", 12288, 4096, 0.2f},
                       {"ddn", 4096, 12288, 0.2f}};
  // --small: every grid edge (rows not a multiple of 8 / 16 / 64, K not a multiple of 1024, uneven split-K parts)
  const Shape tiny[] = {{"in", 200, 1024, 0.5f}, {"out", 72, 2048, 0.37f}, {"odd", 77, 1152, 1.0f}};
  std::vector<Shape> shapes(small ? std::begin(tiny) : std::begin(big), small ? std::end(tiny) : std::end(big));

  for (const Shape& sh : shapes) {
    const bool named_only = !strcmp(sh.name, "w8") || sh.name[0] == 'd';
    if (named_only && !want_shape(sh.name)) continue;
    // the variants of this shape
    struct Run {
      KV kv;
      uint32_t M, R;
    };
    std::vector<Run> runs;
    // today's kernels (the engine's choices: nh_mv1_fp4_n8 for in_proj, nh_mv_fp4_r1(_add) for out_proj, nh_v8_fp4_s4,
    // nh_gemm_fp4_m16 / _m32) and the x4 set: L2 (layout 2) and L0 (today's layout)
    auto add = [&](const char* n, Kind k, int layout, uint32_t rows_tg, uint32_t threads, uint32_t M,
                   std::initializer_list<uint32_t> Rs) {
      for (uint32_t R : Rs) runs.push_back({{n, k, layout, rows_tg, threads}, M, R});
    };
    add("nh_mv1_fp4_n8", MV, 0, 32, 128, 1, {1});
    add("nh_mv_fp4_r1", MV, 0, 16, 128, 1, {1});
    add("nh_mv_fp4_r1_add", MV, 0, 16, 128, 1, {1});
    add("nh_x4_mv", MV, 2, 8, 64, 1, {1});
    add("nh_x4_mv_add", MV, 2, 8, 64, 1, {1});
    add("nh_x4_mv_l0", MV, 0, 8, 64, 1, {1});
    add("nh_x4_mv_l0_add", MV, 0, 8, 64, 1, {1});
    add("nh_v8_fp4_s4", V8, 0, 32, 128, 8, {8, 5});
    add("nh_x4_v8", V8, 2, 16, 256, 8, {8, 5});
    add("nh_x4_v8_l0", V8, 0, 16, 256, 8, {8, 5});
    auto addm = [&](const char* n, int layout, uint32_t sk, uint32_t M, std::initializer_list<uint32_t> Rs) {
      for (uint32_t R : Rs) {
        Run r{{n, V8, layout, 64, 128}, M, R};
        r.kv.sk = sk;
        runs.push_back(r);
      }
    };
    add("nh_gemm_fp4_m16", GEMM, 0, 16, 128, 16, {16, 13});
    add("nh_gemm_fp4_m32", GEMM, 0, 32, 128, 24, {24, 21});
    add("nh_gemm_fp4_m32", GEMM, 0, 32, 128, 32, {32, 29});
    for (uint32_t sk : {1u, 2u, 4u, 8u}) {
      addm("nh_x4_mm16", 2, sk, 16, {16, 13});
      addm("nh_x4_mm24", 2, sk, 24, {24, 21});
      addm("nh_x4_mm32", 2, sk, 32, {32, 29});
      addm("nh_x4_mm16_l0", 0, sk, 16, {16, 13});
      addm("nh_x4_mm24_l0", 0, sk, 24, {24, 21});
      addm("nh_x4_mm32_l0", 0, sk, 32, {32, 29});
    }
    // filter
    std::vector<Run> sel;
    for (auto& r : runs) {
      char lab[160];
      snprintf(lab, sizeof lab, "%s M=%u R=%u %s", sh.name, r.M, r.R, r.kv.label().c_str());
      if (want(lab) && g.pipeline(r.kv.name, false).pso) sel.push_back(r);
    }
    if (sel.empty()) continue;
    printf("== %s %ux%u (gs %g): %zu runs\n", sh.name, sh.N, sh.K, sh.gs, sel.size());
    fflush(stdout);
    Fp4 m = make_fp4(sh.N, sh.K, sh.gs);
    const int ncopy = rot ? std::max(2, int(std::ceil(250e6 / (double(sh.N) * sh.K * 0.5625)))) : 1;
    const int nc = rot ? std::max(copies, ncopy) : 1;
    std::map<int, GpuMat> mats;
    std::map<uint32_t, std::vector<float>> xs;          // per M (half-rounded for M > 1)
    std::map<uint32_t, std::vector<double>> refs;
    const double bytes = double(sh.N) * sh.K * 0.5625;
    struct Live {
      Buffer xb, yb, part, cnt;
      double err = 0, err_add = -1, err_half = -1;
      bool bad_pad = false;
      std::vector<double> t_rot, t_same;
    };
    std::vector<Live> live(sel.size());
    // phase 1: one call per kernel, checked against the fp64 reference
    for (size_t i = 0; i < sel.size(); ++i) {
      const Run& r = sel[i];
      if (!mats.count(r.kv.layout)) mats[r.kv.layout] = upload(m, r.kv.layout, nc);
      const uint32_t M = r.M;
      if (!xs.count(M)) {
        std::vector<float> x(size_t(M) * sh.K);
        for (auto& v : x) v = nrand();
        if (M > 1)
          for (auto& v : x) v = float(__fp16(v));
        xs[M] = x;
        refs[M] = reference(m, x, M);
      }
      const std::vector<float>& x = xs[M];
      const std::vector<double>& ref = refs[M];
      const GpuMat& gm = mats[r.kv.layout];
      Live& L = live[i];
      L.xb = g.alloc(size_t(M) * sh.K * 4, "x");
      L.yb = g.alloc(size_t(M) * sh.N * 4, "y");
      const bool is_mm = r.kv.name.rfind("nh_x4_mm", 0) == 0;
      if (is_mm) {
        L.part = g.alloc(std::max<size_t>(16, r.kv.sk > 1 ? mm_part_bytes(r.kv, sh.N, M) : 16), "x4part");
        L.cnt = g.alloc(std::max<size_t>(16, mm_cnt_bytes(r.kv, sh.N)), "x4cnt");
        memset(L.cnt.ptr(), 0, L.cnt.size);
      }
      const Buffer* lp = is_mm ? &L.part : nullptr;
      const Buffer* lc = is_mm ? &L.cnt : nullptr;
      if (r.kv.kind == MV) memcpy(L.xb.ptr(), x.data(), size_t(sh.K) * 4);
      else
        for (size_t k = 0; k < size_t(M) * sh.K; ++k) L.xb.as<__fp16>()[k] = __fp16(x[k]);
      for (size_t k = 0; k < size_t(M) * sh.N; ++k) L.yb.as<float>()[k] = -777.0f;
      {
        auto e = g.begin();
        encode(*e, r.kv, gm, 0, L.xb, L.yb, sh.N, sh.K, M, r.R, sh.gs, 0, dummy, 0, 0, lp, lc);
        e->commit_and_wait();
      }
      double maxerr = 0, maxref = 0;
      for (uint32_t t = 0; t < M; ++t)
        for (uint32_t n = 0; n < sh.N; ++n) {
          const float v = L.yb.as<float>()[size_t(t) * sh.N + n];
          if (t >= r.R) {
            if (v != -777.0f) L.bad_pad = true;
            continue;
          }
          const double rf = ref[size_t(t) * sh.N + n];
          maxerr = std::max(maxerr, std::isfinite(v) ? std::fabs(rf - v) : 1e30);
          maxref = std::max(maxref, std::fabs(rf));
        }
      L.err = maxerr / std::max(maxref, 1e-30);
      if (is_mm)
        for (size_t k = 0; k < L.cnt.size / 4; ++k)
          if (L.cnt.as<uint32_t>()[k]) {
            printf("%s: split-K counter %zu left at %u\n", r.kv.name.c_str(), k, L.cnt.as<uint32_t>()[k]);
            break;
          }
      // the add epilogue on a prefilled y (MV *_add; V8 kinds with X / Y rows 64 halves / 40 floats apart, padding
      // untouched), and V8's half store (epi 3)
      const bool add_mv = r.kv.kind == MV && r.kv.name.size() > 4 && r.kv.name.compare(r.kv.name.size() - 4, 4, "_add") == 0;
      if (add_mv || (r.kv.kind == V8)) {
        const uint32_t xl = r.kv.kind == V8 ? sh.K + 64 : sh.K, yl = r.kv.kind == V8 ? sh.N + 40 : sh.N;
        Buffer x2 = g.alloc(size_t(M) * xl * 4, "x2"), y2 = g.alloc(size_t(M) * yl * 4, "y2");
        if (r.kv.kind == MV) memcpy(x2.ptr(), x.data(), size_t(sh.K) * 4);
        else
          for (uint32_t t = 0; t < M; ++t)
            for (uint32_t k = 0; k < xl; ++k) x2.as<__fp16>()[size_t(t) * xl + k] = k < sh.K ? __fp16(x[size_t(t) * sh.K + k]) : __fp16(1000.0f);
        std::vector<float> y0(size_t(M) * yl);
        for (auto& v : y0) v = nrand();
        memcpy(y2.ptr(), y0.data(), y0.size() * 4);
        {
          auto e = g.begin();
          encode(*e, r.kv, gm, 0, x2, y2, sh.N, sh.K, M, r.R, sh.gs, 1, dummy, xl, yl, lp, lc);
          e->commit_and_wait();
        }
        double me = 0, mr = 0;
        bool bad = false;
        for (uint32_t t = 0; t < M; ++t)
          for (uint32_t n = 0; n < yl; ++n) {
            const float v = y2.as<float>()[size_t(t) * yl + n], v0 = y0[size_t(t) * yl + n];
            if (t >= r.R || n >= sh.N) {
              if (v != v0) bad = true;
              continue;
            }
            const double rf = ref[size_t(t) * sh.N + n];
            me = std::max(me, std::isfinite(v) ? std::fabs(rf - (double(v) - double(v0))) : 1e30);
            mr = std::max(mr, std::fabs(rf));
          }
        L.err_add = me / std::max(mr, 1e-30);
        L.bad_pad |= bad;
        if (r.kv.kind == V8 && r.kv.name.rfind("nh_x4", 0) == 0) {
          for (size_t k = 0; k < size_t(M) * sh.N; ++k) L.yb.as<float>()[k] = -777.0f;
          auto e = g.begin();
          encode(*e, r.kv, gm, 0, L.xb, L.yb, sh.N, sh.K, M, r.R, sh.gs, 3, dummy, 0, 0, lp, lc);
          e->commit_and_wait();
          double mh = 0;
          for (uint32_t t = 0; t < r.R; ++t)
            for (uint32_t n = 0; n < sh.N; ++n) {
              const double v = double(float(L.yb.as<__fp16>()[size_t(t) * sh.N + n]));
              mh = std::max(mh, std::isfinite(v) ? std::fabs(ref[size_t(t) * sh.N + n] - v) : 1e30);
            }
          L.err_half = mh / std::max(maxref, 1e-30);
        }
        g.release(x2);
        g.release(y2);
      }
    }
    // phase 2: timing rounds, every kernel once per round (10 back-to-back calls per command buffer), so drift in
    // the GPU's clock hits all of them alike
    int cyc = 0;
    auto timed = [&](size_t i, bool rot_mode) {
      const Run& r = sel[i];
      const GpuMat& gm = mats[r.kv.layout];
      auto e = g.begin();
      for (int c = 0; c < 10; ++c) {
        const int copy = rot_mode ? (cyc = (cyc + 1) % gm.copies) : 0;
        encode(*e, r.kv, gm, copy, live[i].xb, live[i].yb, sh.N, sh.K, r.M, r.R, sh.gs, 0, dummy, 0, 0,
               live[i].part.mtl ? &live[i].part : nullptr, live[i].cnt.mtl ? &live[i].cnt : nullptr);
      }
      e->commit_and_wait();
      return e->gpu_ms() / 10;
    };
    for (int w = 0; w < 2; ++w)
      for (size_t i = 0; i < sel.size(); ++i) timed(i, rot);
    for (int rd = 0; rd < reps; ++rd)
      for (size_t i = 0; i < sel.size(); ++i) {
        if (rot) live[i].t_rot.push_back(timed(i, true));
        if (same || !rot) live[i].t_same.push_back(timed(i, false));
      }
    auto med = [](std::vector<double> v) {
      if (v.empty()) return 0.0;
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    for (size_t i = 0; i < sel.size(); ++i) {
      const Run& r = sel[i];
      const double ms_rot = med(live[i].t_rot), ms_same = med(live[i].t_same);
      const double ms = ms_rot > 0 ? ms_rot : ms_same;
      const double flops = 2.0 * double(sh.N) * sh.K * r.M;
      const double mn = *std::min_element(ms_rot > 0 ? live[i].t_rot.begin() : live[i].t_same.begin(),
                                          ms_rot > 0 ? live[i].t_rot.end() : live[i].t_same.end());
      printf("%-4s M=%-2u R=%-2u %-26s %s %7.4f ms %6.1f GB/s (min %7.4f)", sh.name, r.M, r.R, r.kv.label().c_str(),
             ms_rot > 0 ? "rot" : "same", ms, bytes / ms / 1e6, mn);
      if (ms_rot > 0 && ms_same > 0) printf(" (same %7.4f ms %6.1f)", ms_same, bytes / ms_same / 1e6);
      if (r.M > 1) printf(" %5.2f TF", flops / ms / 1e9);
      printf("  err %.1e", live[i].err);
      if (live[i].err_add >= 0) printf(" add %.1e", live[i].err_add);
      if (live[i].err_half >= 0) printf(" half %.1e", live[i].err_half);
      if (live[i].bad_pad) printf(" PAD-WRITTEN");
      if (gpr) printf("  %s", gpr_str(r.kv.name).c_str());
      printf("\n");
      g.release(live[i].xb);
      g.release(live[i].yb);
      if (live[i].part.mtl) g.release(live[i].part);
      if (live[i].cnt.mtl) g.release(live[i].cnt);
    }
    fflush(stdout);
    for (auto& [k, v] : mats) {
      g.release(v.q);
      g.release(v.s);
    }
  }
  return 0;
}
