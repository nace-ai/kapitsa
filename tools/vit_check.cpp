// Checks of image input (src/vision) against the reference implementations; tests/vision/*.py drive it.
//   vit_check decode IMAGE OUT.rgb                    decoded RGB (PIL's convert("RGB") values); prints "w h digest"
//   vit_check resize IN.rgb W H OW OH OUT.rgb         Pillow's bicubic resize of raw RGB
//   vit_check smart H W FACTOR MIN MAX                smart_resize; prints "h w"
//   vit_check prep MODEL IMAGE OUT.f32                the processor's pixel_values (float32, the reference's layout:
//                                                     Qwen [patches, C*T*P*P], RADIO [3, H, W]); prints
//                                                     "grid_h grid_w tokens width height"
//   vit_check aaresize IN.rgb W H OW OH OUT.f32       torch's antialiased bicubic interpolate of raw RGB (float32
//                                                     channel planes, values 0-255)
//   vit_check patches W H BUDGET MIN                  the RADIO processor's patch grid; prints "cols rows"
//   vit_check patches FILE                            the same for each line "W H BUDGET MIN" of FILE
//   vit_check rope SPANS OUT.i32                      M-RoPE positions of a prompt: SPANS = "n pos:gh:gw,..." rows;
//                                                     writes int32 [n, 3]
//   vit_check tower MODEL IN.rgb W H OUT [LAYERS]     GPU: the encoder on raw RGB of W x H (a size its processor
//                                                     keeps: multiples of 32): bf16 embeddings, or with LAYERS the fp32
//                                                     hidden state of every row after that many blocks
//   vit_check bench MODEL W H [REPS]                  GPU: encode time of a W x H image (random pixels)
//   vit_check kernels                                 GPU, tiny synthetic tensors: attn_prep_m vs attn_prep (bitwise
//                                                     on text rows), vs a CPU M-RoPE on image rows; embed_rows_bf16;
//                                                     vit_ln_f32 vs a CPU LayerNorm
#include "model/kparams.h"
#include "model/qwen35.h"
#include "util/bf16.h"
#include "util/log.h"
#include "vision/qwen_vit.h"
#include "vision/radio_vit.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;
using namespace ni::vision;

static std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) fatal("cannot read %s", p.c_str());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static void dump(const std::string& p, const void* data, size_t n) {
  FILE* f = fopen(p.c_str(), "wb");
  if (!f || fwrite(data, 1, n, f) != n) fatal("cannot write %s", p.c_str());
  fclose(f);
}

static std::unique_ptr<ImageEncoder> open_vit(const std::string& dir) {
  std::unique_ptr<ImageEncoder> e = open_encoder(dir);  // the model's own limits (as the reference processors)
  if (!e) fatal("%s: no vision tower Kapitsa runs", dir.c_str());
  return e;
}

// attn_prep / attn_prep_m on random rows: text rows must match bitwise, image rows a CPU rotation by (t, h, w)
static int kernel_checks() {
  Gpu& g = Gpu::get();
  const uint32_t T = 7, Hq = 4, Hkv = 2, D = 256, rot = 64, S = 16, pos0 = 3;
  const float eps = 1e-6f, theta = 1e7f;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto fill = [&](Buffer& b, size_t n) {
    for (size_t i = 0; i < n; ++i) b.as<float>()[i] = nd(rng);
  };
  Buffer qg = g.alloc(size_t(T) * Hq * 2 * D * 4), kr = g.alloc(size_t(T) * Hkv * D * 4), vr = g.alloc(size_t(T) * Hkv * D * 4);
  fill(qg, size_t(T) * Hq * 2 * D);
  fill(kr, size_t(T) * Hkv * D);
  fill(vr, size_t(T) * Hkv * D);
  Buffer qn = g.alloc(D * 2), kn = g.alloc(D * 2);
  for (uint32_t i = 0; i < D; ++i) qn.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.1f * nd(rng)), kn.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.1f * nd(rng));
  Buffer qa = g.alloc(size_t(T) * Hq * D * 4), qb = g.alloc(size_t(T) * Hq * D * 4);
  Buffer ka = g.alloc(size_t(Hkv) * S * D * 2), kb = g.alloc(size_t(Hkv) * S * D * 2);
  Buffer va = g.alloc(size_t(Hkv) * S * D * 2), vb = g.alloc(size_t(Hkv) * S * D * 2);
  // positions: rows pos0.. of the sequence; rows < mn from the table (an "image" of 2 x 3 at rows 4..9)
  const uint32_t mn = 10;
  Buffer mp = g.alloc(size_t(S) * 12);
  std::vector<int32_t> P(size_t(S) * 3);
  for (uint32_t r = 0; r < S; ++r) P[3 * r] = P[3 * r + 1] = P[3 * r + 2] = int(r);
  for (int k = 0; k < 6; ++k) {  // image rows 4..9: t = 4, h = 4 + k / 3, w = 4 + k % 3
    P[3 * (4 + k)] = 4;
    P[3 * (4 + k) + 1] = 4 + k / 3;
    P[3 * (4 + k) + 2] = 4 + k % 3;
  }
  memcpy(mp.ptr(), P.data(), P.size() * 4);
  const int delta = 7 - 10;  // after the image: max position 6 + 1 = 7 at row 10
  struct {
    uint32_t mn;
    int32_t delta;
    uint32_t sec1, sec2;
  } mr{mn, delta, 33, 30};
  const kp::AttnPrep ap{T, Hq, Hkv, D, rot, pos0, S, eps, theta};
  auto run = [&](bool m, const Buffer& q, const Buffer& kc, const Buffer& vc, uint32_t mnv, int dv) {
    auto e = g.begin();
    e->use(g.pipeline(m ? "attn_prep_m" : "attn_prep"));
    e->buf(0, qg); e->buf(1, kr); e->buf(2, vr); e->buf(3, qn); e->buf(4, kn);
    e->buf(5, q); e->buf(6, kc); e->buf(7, vc);
    e->val(8, ap);
    if (m) {
      auto mm = mr;
      mm.mn = mnv;
      mm.delta = dv;
      e->buf(9, mp);
      e->val(10, mm);
    }
    e->dispatch(Hq + 2 * Hkv, T, 1, D);
    e->commit_and_wait();
  };
  int fails = 0;
  // 1. no image positions (mn 0, delta 0): bitwise equal to attn_prep
  run(false, qa, ka, va, 0, 0);
  run(true, qb, kb, vb, 0, 0);
  const bool same = !memcmp(qa.ptr(), qb.ptr(), qa.size) && !memcmp(ka.ptr(), kb.ptr(), ka.size) && !memcmp(va.ptr(), vb.ptr(), va.size);
  printf("%s attn_prep_m without image positions == attn_prep bitwise\n", same ? "ok  " : "FAIL");
  fails += !same;
  // 2. with the table: CPU reference (q/k RMSNorm, then rotate pairs (i, i + rot/2) by the axis M-RoPE picks)
  run(true, qb, kb, vb, mn, delta);
  double maxerr = 0;
  for (uint32_t t = 0; t < T; ++t) {
    const uint32_t row = pos0 + t;
    int tp[3];
    for (int a = 0; a < 3; ++a) tp[a] = row < mn ? P[3 * row + a] : int(row) + delta;
    for (uint32_t h = 0; h < Hq + Hkv; ++h) {
      const bool isq = h < Hq;
      const float* x = isq ? qg.as<float>() + (size_t(t) * Hq + h) * 2 * D : kr.as<float>() + (size_t(t) * Hkv + (h - Hq)) * D;
      double ss = 0;
      for (uint32_t i = 0; i < D; ++i) ss += double(x[i]) * x[i];
      const double inv = 1.0 / std::sqrt(ss / D + eps);
      std::vector<double> v(D);
      for (uint32_t i = 0; i < D; ++i) v[i] = x[i] * inv * bf16_to_f32((isq ? qn : kn).as<uint16_t>()[i]);
      std::vector<double> o = v;
      for (uint32_t i = 0; i < rot / 2; ++i) {
        const int a = (i % 3 == 1 && i < 33) ? 1 : (i % 3 == 2 && i < 30) ? 2 : 0;
        const double ang = tp[a] * std::pow(double(theta), -2.0 * i / rot);
        o[i] = v[i] * std::cos(ang) - v[i + rot / 2] * std::sin(ang);
        o[i + rot / 2] = v[i + rot / 2] * std::cos(ang) + v[i] * std::sin(ang);
      }
      for (uint32_t i = 0; i < D; ++i) {
        const double got = isq ? qb.as<float>()[(size_t(t) * Hq + h) * D + i]
                               : double(float(kb.as<_Float16>()[(size_t(h - Hq) * S + row) * D + i]));
        maxerr = std::max(maxerr, std::abs(got - o[i]) / (isq ? 1.0 : std::max(1.0, std::abs(o[i]))));
      }
    }
  }
  printf("%s attn_prep_m with (t, h, w) positions vs a CPU M-RoPE: max error %.2e (q fp32; k half, relative)\n",
         maxerr < 2e-3 ? "ok  " : "FAIL", maxerr);
  fails += maxerr >= 2e-3;
  // 3. embed_rows_bf16 through enc_embed_runs: rows [2, 5) of a 6-row chunk at prompt position 10 from an image whose
  // run starts at 11 (its rows 1..3), rows outside untouched
  const int Dm = 64;
  Buffer src = g.alloc(size_t(8) * Dm * 2), x = g.alloc(size_t(6) * Dm * 4);
  for (int i = 0; i < 8 * Dm; ++i) src.as<uint16_t>()[i] = f32_to_bf16(nd(rng));
  for (int i = 0; i < 6 * Dm; ++i) x.as<float>()[i] = -1.0f;
  std::vector<EmbedRun> runs{EmbedRun{11, 3, src, 1, Dm}};
  {
    auto e = g.begin();
    enc_embed_runs(*e, runs, 10, 6, x, Dm);
    e->commit_and_wait();
  }
  bool ok = true;
  for (int r = 0; r < 6; ++r)
    for (int i = 0; i < Dm; ++i) {
      const float want = (r >= 1 && r < 4) ? bf16_to_f32(src.as<uint16_t>()[(1 + r - 1) * Dm + i]) : -1.0f;
      ok = ok && x.as<float>()[r * Dm + i] == want;
    }
  printf("%s embed_rows_bf16: image rows copied exactly, others untouched\n", ok ? "ok  " : "FAIL");
  fails += !ok;
  // 4. vit_ln_f32 (RADIO's final LayerNorm, in place) on rows [1, 4) of 5 (a pending bias first) vs a CPU LayerNorm
  // in double; rows outside untouched
  {
    const int Dl = 80, R = 5;
    const float leps = 1e-6f;
    Buffer xl = g.alloc(size_t(R) * Dl * 4), pb = g.alloc(Dl * 2), lw = g.alloc(Dl * 2), lb = g.alloc(Dl * 2);
    for (int i = 0; i < R * Dl; ++i) xl.as<float>()[i] = 3.0f + 2.0f * nd(rng);
    for (int i = 0; i < Dl; ++i) {
      pb.as<uint16_t>()[i] = f32_to_bf16(0.5f * nd(rng));
      lw.as<uint16_t>()[i] = f32_to_bf16(1.0f + 0.2f * nd(rng));
      lb.as<uint16_t>()[i] = f32_to_bf16(0.1f * nd(rng));
    }
    const std::vector<float> x0(xl.as<float>(), xl.as<float>() + R * Dl);
    {
      auto e = g.begin();
      vit_ln_inplace(*e, xl.view(size_t(Dl) * 4, size_t(3) * Dl * 4), uint32_t(Dl), &pb, lw, lb, 3, Dl, leps);
      e->commit_and_wait();
    }
    double err = 0;
    bool untouched = true;
    for (int r = 0; r < R; ++r) {
      const float* got = xl.as<float>() + size_t(r) * Dl;
      const float* in = x0.data() + size_t(r) * Dl;
      if (r < 1 || r >= 4) {
        untouched = untouched && !memcmp(got, in, Dl * 4);
        continue;
      }
      double m = 0, v = 0;
      for (int i = 0; i < Dl; ++i) m += double(in[i]) + bf16_to_f32(pb.as<uint16_t>()[i]);
      m /= Dl;
      for (int i = 0; i < Dl; ++i) {
        const double d = double(in[i]) + bf16_to_f32(pb.as<uint16_t>()[i]) - m;
        v += d * d;
      }
      const double rs = 1.0 / std::sqrt(v / Dl + leps);
      for (int i = 0; i < Dl; ++i) {
        const double want = (double(in[i]) + bf16_to_f32(pb.as<uint16_t>()[i]) - m) * rs * bf16_to_f32(lw.as<uint16_t>()[i]) +
                            bf16_to_f32(lb.as<uint16_t>()[i]);
        err = std::max(err, std::abs(got[i] - want));
      }
    }
    const bool lok = err < 1e-5 && untouched;
    printf("%s vit_ln_f32 vs a CPU LayerNorm: max error %.2e, rows outside %s\n", lok ? "ok  " : "FAIL", err,
           untouched ? "untouched" : "CHANGED");
    fails += !lok;
  }
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: see tools/vit_check.cpp\n");
    return 2;
  }
  const std::string cmd = argv[1];
  if (cmd == "kernels") return kernel_checks();
  try {
    if (cmd == "decode" && argc == 4) {
      const Image im = decode(slurp(argv[2]));
      dump(argv[3], im.rgb.data(), im.rgb.size());
      printf("%d %d %s\n", im.width, im.height, hex(im.digest, 32).c_str());
      return 0;
    }
    if (cmd == "resize" && argc == 8) {
      const std::string raw = slurp(argv[2]);
      const int w = atoi(argv[3]), h = atoi(argv[4]), ow = atoi(argv[5]), oh = atoi(argv[6]);
      if (raw.size() != size_t(w) * h * 3) fatal("resize: %zu bytes for %d x %d", raw.size(), w, h);
      const std::vector<uint8_t> in(raw.begin(), raw.end());
      const auto out = pil_resize_bicubic(in, w, h, ow, oh);
      dump(argv[7], out.data(), out.size());
      return 0;
    }
    if (cmd == "smart" && argc == 7) {
      const auto [h, w] = qwen_smart_resize(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atol(argv[5]), atol(argv[6]));
      printf("%d %d\n", h, w);
      return 0;
    }
    if (cmd == "prep" && argc == 5) {
      auto vit = open_vit(argv[2]);
      const Prepared p = vit->prepare(decode(slurp(argv[3])));
      const std::vector<float> pv = vit->pixel_values(p);
      dump(argv[4], pv.data(), pv.size() * 4);
      printf("%d %d %d %d %d\n", p.grid_h, p.grid_w, p.tokens, p.width, p.height);
      return 0;
    }
    if (cmd == "aaresize" && argc == 8) {
      const std::string raw = slurp(argv[2]);
      const int w = atoi(argv[3]), h = atoi(argv[4]), ow = atoi(argv[5]), oh = atoi(argv[6]);
      if (raw.size() != size_t(w) * h * 3) fatal("aaresize: %zu bytes for %d x %d", raw.size(), w, h);
      const std::vector<float> out = torch_resize_bicubic_aa(std::vector<uint8_t>(raw.begin(), raw.end()), w, h, ow, oh);
      dump(argv[7], out.data(), out.size() * 4);
      return 0;
    }
    if (cmd == "patches" && argc == 3) {
      std::ifstream in(argv[2]);
      int w, h, budget, mn;
      while (in >> w >> h >> budget >> mn) {
        const auto [cols, rows] = radio_target_patches(w, h, 16, budget, mn);
        printf("%d %d\n", cols, rows);
      }
      return 0;
    }
    if (cmd == "patches" && argc == 6) {
      const auto [cols, rows] = radio_target_patches(atoi(argv[2]), atoi(argv[3]), 16, atoi(argv[4]), atoi(argv[5]));
      printf("%d %d\n", cols, rows);
      return 0;
    }
    if (cmd == "rope" && argc == 4) {
      // "n pos:gh:gw,pos:gh:gw,..."
      std::istringstream in(argv[2]);
      int n = 0;
      std::string list;
      in >> n >> list;
      std::vector<ImageSpan> spans;
      std::istringstream ls(list);
      std::string item;
      while (std::getline(ls, item, ',')) {
        ImageSpan s;
        if (sscanf(item.c_str(), "%d:%d:%d", &s.pos, &s.grid_h, &s.grid_w) != 3) fatal("rope: bad span %s", item.c_str());
        s.grid_t = 1;
        s.n = s.grid_h * s.grid_w;
        spans.push_back(s);
      }
      std::vector<int32_t> table;
      const int delta = Qwen35::rope_positions(spans, table);
      const int end = int(table.size() / 3);
      for (int r = end; r < n; ++r) table.insert(table.end(), {r + delta, r + delta, r + delta});
      dump(argv[3], table.data(), table.size() * 4);
      printf("%d\n", delta);
      return 0;
    }
    if (cmd == "tower" && (argc == 7 || argc == 8)) {
      auto vit = open_vit(argv[2]);
      const std::string raw = slurp(argv[3]);
      Image im;
      im.width = atoi(argv[4]);
      im.height = atoi(argv[5]);
      if (raw.size() != size_t(im.width) * im.height * 3) fatal("tower: %zu bytes for %d x %d", raw.size(), im.width, im.height);
      im.rgb.assign(raw.begin(), raw.end());
      const Prepared p = vit->prepare(im);
      if (p.width != im.width || p.height != im.height)
        fatal("tower: the processor resizes %d x %d to %d x %d; give it a size it keeps", im.width, im.height, p.width,
              p.height);
      const int layers = argc == 8 ? atoi(argv[7]) : -1;
      vit->load();
      const size_t bytes = layers >= 0 ? size_t(vit->debug_rows(p)) * vit->debug_width() * 4
                                       : size_t(p.tokens) * vit->out_dim() * 2;
      Buffer out = Gpu::get().alloc(bytes + 16, "out");
      if (layers >= 0) vit->set_debug_stop(layers);
      const double t0 = now_ms();
      vit->encode(p, out);
      fprintf(stderr, "encoded %d x %d (%d tokens) in %.1f ms\n", p.width, p.height, p.tokens, now_ms() - t0);
      dump(argv[6], out.ptr(), bytes);
      return 0;
    }
    if (cmd == "bench" && (argc == 5 || argc == 6)) {
      auto vit = open_vit(argv[2]);
      const int w = atoi(argv[3]), h = atoi(argv[4]), reps = argc == 6 ? atoi(argv[5]) : 3;
      Image im;
      im.width = w;
      im.height = h;
      im.rgb.resize(size_t(w) * h * 3);
      std::mt19937 rng(1);
      for (auto& v : im.rgb) v = uint8_t(rng());
      double t0 = now_ms();
      const Prepared p = vit->prepare(im);
      const double prep = now_ms() - t0;
      t0 = now_ms();
      vit->load();
      const double load = now_ms() - t0;
      Buffer out = Gpu::get().alloc(size_t(p.tokens) * vit->out_dim() * 2 + 16, "out");
      printf("%dx%d -> %dx%d, %d patches, %d tokens: prepare %.0f ms, load %.0f ms, work %.0f MB\n", w, h, p.width,
             p.height, p.tokens * 4, p.tokens, prep, load, vit->work_bytes(p) / 1048576.0);
      for (int i = 0; i < reps; ++i) {
        t0 = now_ms();
        vit->encode(p, out);
        printf("  encode %.1f ms\n", now_ms() - t0);
      }
      if (Encoder::profiling()) Encoder::profile_report(reps);
      return 0;
    }
  } catch (const Error& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  fprintf(stderr, "usage: see tools/vit_check.cpp\n");
  return 2;
}
