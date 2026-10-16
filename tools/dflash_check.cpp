// DFlash2 drafter check against the Python/MLX oracle dumps (research/inference/dflash2/dumps).
//   dflash_check <target_dir> <drafter_dir> <dump_dir> [--bits 4|8] [--time N] [--live] [--long P]
//                [--no-h0] [--out dir] [--sim N]
// 1. Context commit of the dump's exact prompt captures (ctx_x_bf16) -> ring; compares c, K, V.
// 2. Per dumped cycle: the drafting pass at the dump's P/anchor (starting from the dump's block
//    embedding h_layers[0] unless --no-h0) -> per-layer intermediates (cycle 0), h per layer, final
//    hidden, logits, top-16, selector hidden, edges, walk, chain; then commits the cycle's kept
//    verify captures (verify_capt_bf16 rows 0..m-1) and compares commit_c/k/v.
// 3. --time N: GPU time of the drafting pass (median of N single-pass command buffers) at the
//    dump's context, and at --long P (default 4096: full 2047-row window; ring content is stale,
//    timing only).
// 4. --live: our own target prefills the prompt with the capture hook; compares the captures with
//    ctx_x_bf16 and drafts cycle 0 from our own state.
// 5. --sim N: our target decodes N greedy tokens one step at a time with the capture hook on every
//    step; then the DFlash loop is replayed against that sequence with all 7 proposals verified
//    (accept = prefix match with the greedy tokens, commit m = a + 1 rows from the captures): tokens
//    per cycle at full verify depth with plain-decode numerics (the oracle's `serial` semantics).
// --out dir writes our tensors in the dump layout (dflash2_ref.py compare <dump> <dir>).
#include "metal/gpu.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "util/bf16.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace ni;
using json = nlohmann::json;

// ---- minimal .npy / .npz (stored) reader & .npy writer ----
struct Npy {
  std::string descr;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  template <class T> const T* as() const { return reinterpret_cast<const T*>(data.data()); }
  std::vector<float> f32() const {  // any numeric dtype -> float
    std::vector<float> o(numel());
    for (int64_t i = 0; i < numel(); ++i) {
      if (descr == "<f4") o[i] = as<float>()[i];
      else if (descr == "<f8") o[i] = float(as<double>()[i]);
      else if (descr == "<i4") o[i] = float(as<int32_t>()[i]);
      else if (descr == "<i8") o[i] = float(as<int64_t>()[i]);
      else if (descr == "<u2") o[i] = bf16_to_f32(as<uint16_t>()[i]);
      else fatal("npy dtype %s", descr.c_str());
    }
    return o;
  }
  std::vector<int64_t> ints() const {
    std::vector<int64_t> o(numel());
    for (int64_t i = 0; i < numel(); ++i) {
      if (descr == "<i4") o[i] = as<int32_t>()[i];
      else if (descr == "<i8") o[i] = as<int64_t>()[i];
      else fatal("npy int dtype %s", descr.c_str());
    }
    return o;
  }
};

static size_t dsize(const std::string& d) {
  if (d == "<f4" || d == "<i4" || d == "<u4") return 4;
  if (d == "<f8" || d == "<i8") return 8;
  if (d == "<u2" || d == "<f2" || d == "<i2") return 2;
  if (d == "|b1" || d == "|u1") return 1;
  fatal("npy dtype %s", d.c_str());
}

// Parses an .npy image at p; returns bytes consumed.
static size_t parse_npy(const uint8_t* p, size_t n, Npy& o) {
  if (n < 10 || memcmp(p, "\x93NUMPY", 6)) fatal("not an npy image");
  const int major = p[6];
  size_t hl, off;
  if (major == 1) {
    hl = p[8] | (p[9] << 8);
    off = 10;
  } else {
    hl = p[8] | (p[9] << 8) | (size_t(p[10]) << 16) | (size_t(p[11]) << 24);
    off = 12;
  }
  std::string h((const char*)p + off, hl);
  auto get = [&](const std::string& key) {
    size_t k = h.find("'" + key + "'");
    if (k == std::string::npos) fatal("npy header lacks %s", key.c_str());
    return h.substr(h.find(':', k) + 1);
  };
  std::string d = get("descr");
  d = d.substr(d.find('\'') + 1);
  o.descr = d.substr(0, d.find('\''));
  if (get("fortran_order").find("True") < 6) fatal("fortran-order npy");
  std::string s = get("shape");
  s = s.substr(s.find('(') + 1, s.find(')') - s.find('(') - 1);
  o.shape.clear();
  for (size_t i = 0; i < s.size();) {
    while (i < s.size() && (s[i] == ' ' || s[i] == ',')) ++i;
    if (i >= s.size()) break;
    size_t j = i;
    while (j < s.size() && isdigit(s[j])) ++j;
    o.shape.push_back(std::stoll(s.substr(i, j - i)));
    i = j;
  }
  const size_t bytes = size_t(o.numel()) * dsize(o.descr);
  if (off + hl + bytes > n) fatal("truncated npy");
  o.data.assign(p + off + hl, p + off + hl + bytes);
  return off + hl + bytes;
}

static std::vector<uint8_t> slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fatal("cannot read %s", path.c_str());
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}
static bool exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}
static Npy load_npy(const std::string& path) {
  auto b = slurp(path);
  Npy o;
  parse_npy(b.data(), b.size(), o);
  return o;
}
// np.savez archives are ZIP_STORED; members are found by their local headers.
static std::map<std::string, Npy> load_npz(const std::string& path) {
  auto b = slurp(path);
  std::map<std::string, Npy> m;
  size_t i = 0;
  while (i + 30 <= b.size()) {
    uint32_t sig;
    memcpy(&sig, &b[i], 4);
    if (sig != 0x04034b50) break;
    uint16_t method, nlen, elen;
    memcpy(&method, &b[i + 8], 2);
    memcpy(&nlen, &b[i + 26], 2);
    memcpy(&elen, &b[i + 28], 2);
    if (method != 0) fatal("%s: compressed npz", path.c_str());
    std::string name((const char*)&b[i + 30], nlen);
    size_t d = i + 30 + nlen + elen;
    Npy o;
    size_t used = parse_npy(&b[d], b.size() - d, o);
    if (name.size() > 4 && name.substr(name.size() - 4) == ".npy") name = name.substr(0, name.size() - 4);
    m[name] = std::move(o);
    i = d + used;
    // optional data descriptor
    if (i + 4 <= b.size()) {
      memcpy(&sig, &b[i], 4);
      if (sig == 0x08074b50) i += 16;  // may be 24 with zip64; resync below
    }
    while (i + 4 <= b.size()) {
      memcpy(&sig, &b[i], 4);
      if (sig == 0x04034b50 || sig == 0x02014b50) break;
      ++i;
    }
  }
  return m;
}
static void save_npy(const std::string& path, const std::string& descr, const std::vector<int64_t>& shape,
                     const void* data, size_t bytes) {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); ++i) s += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? "," : "");
  s += ")";
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': " + s + ", }";
  while ((10 + h.size() + 1) % 64) h += ' ';
  h += '\n';
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t hl = uint16_t(h.size());
  f.write((const char*)&hl, 2);
  f.write(h.data(), h.size());
  f.write((const char*)data, bytes);
}

// ---- comparison helpers ----
struct Err {
  double rel = 0, max_abs = 0, max_ref = 0;
  int64_t n = 0;
};
static Err cmp(const float* a, const float* ref, int64_t n) {
  double num = 0, den = 0;
  Err e;
  for (int64_t i = 0; i < n; ++i) {
    if (!std::isfinite(ref[i]) && !std::isfinite(a[i])) continue;
    const double d = double(a[i]) - ref[i];
    num += d * d;
    den += double(ref[i]) * ref[i];
    e.max_abs = std::max(e.max_abs, std::fabs(d));
    e.max_ref = std::max(e.max_ref, std::fabs(double(ref[i])));
    ++e.n;
  }
  e.rel = std::sqrt(num / std::max(den, 1e-300));
  return e;
}
static double g_worst_h = 0;
static void report(const std::string& name, const Err& e) {
  printf("  %-26s rel %.3e  max_abs %.3e  (max|ref| %.3e, max_abs/max|ref| %.3e)\n", name.c_str(), e.rel, e.max_abs,
         e.max_ref, e.max_abs / std::max(e.max_ref, 1e-30));
}
static std::vector<float> half_to_f32(const Buffer& b, size_t n) {
  std::vector<float> o(n);
  const __fp16* h = b.as<__fp16>();
  for (size_t i = 0; i < n; ++i) o[i] = float(h[i]);
  return o;
}
static std::vector<float> f32_of(const Buffer& b, size_t n) {
  std::vector<float> o(n);
  memcpy(o.data(), b.ptr(), n * 4);
  return o;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: dflash_check <target_dir> <drafter_dir> <dump_dir> [--bits 4|8] [--time N] [--live] "
                    "[--long P] [--no-h0] [--out dir]\n");
    return 1;
  }
  const std::string tdir = argv[1], ddir = argv[2], dump = argv[3];
  int bits = 0, ntime = 0, plong = 4096, sim = 0;
  bool live = false, use_h0 = true;
  std::string out;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--bits" && i + 1 < argc) bits = atoi(argv[++i]);
    else if (a == "--time" && i + 1 < argc) ntime = atoi(argv[++i]);
    else if (a == "--long" && i + 1 < argc) plong = atoi(argv[++i]);
    else if (a == "--live") live = true;
    else if (a == "--sim" && i + 1 < argc) sim = atoi(argv[++i]);
    else if (a == "--no-h0") use_h0 = false;
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else fatal("unknown option %s", a.c_str());
  }
  setvbuf(stdout, nullptr, _IOLBF, 0);
  // the oracle dumps hold full-vocabulary draft logits and top-16 sets: always the full draft head here
  setenv("NI_DFLASH_HEAD", "0", 1);
  Gpu& g = Gpu::get();
  Qwen35 tgt(tdir, 2048);
  DFlash2 dr(tgt, ddir, bits);
  const auto& C = dr.config();
  const int H = C.H, NL = C.NL, V = C.vocab, nH = int(C.target_layers.size()) * H;
  json meta = json::parse(std::ifstream(dump + "/meta.json"));
  printf("== dflash_check %s (drafter %d-bit, dump draft_q4=%s)\n", dump.c_str(), dr.bits(),
         meta.value("draft_q4", false) ? "true" : "false");
  const auto prompt = load_npy(dump + "/prompt_ids.npy").ints();
  const auto ctx_pos = load_npy(dump + "/ctx_positions.npy").ints();
  const Npy ctx_x = load_npy(dump + "/ctx_x_bf16.npy");
  const int n = int(prompt.size()), Nc = int(ctx_pos.size());
  if (ctx_x.shape[0] != Nc || ctx_x.shape[1] != nH) fatal("ctx_x shape");
  auto seq = tgt.new_seq(std::max(8192, plong + 64));
  Seq& s = *seq;
  for (int i = 0; i < n; ++i) s.tokens()[i] = int(prompt[i]);
  auto dseq = dr.new_seq();
  DFlashSeq& d = *dseq;
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  dr.set_debug(true);
  if (!out.empty()) mkdir(out.c_str(), 0755);

  // ---- 1. prompt context commit from the dump's captures ----
  dr.begin_prefill(n);
  dr.end_prefill();  // the hook is not needed: the captures come from the dump
  dr.set_prompt_captures(ctx_x.as<uint16_t>(), Nc);
  {  // both commit paths, interleaved (the last one run, the default, stays in the ring)
    std::vector<double> t8, tq;
    for (int i = 0; i < 4; ++i)
      for (int v = 0; v < 2; ++v) {
        dr.set_pf8(v == 0 ? 0 : 1 << 30);
        auto e = g.begin();
        dr.encode_prefill_commit(*e, d);
        e->commit_and_wait();
        if (i) (v ? t8 : tq).push_back(e->gpu_ms());
      }
    dr.set_pf8(-1);
    auto e = g.begin();
    dr.encode_prefill_commit(*e, d);
    e->commit_and_wait();
    std::sort(t8.begin(), t8.end());
    std::sort(tq.begin(), tq.end());
    printf("context commit of %d prompt rows: %.2f ms GPU (default path); 8-row chunks median %.2f ms, "
           "prefill GEMM median %.2f ms\n", Nc, e->gpu_ms(), t8[t8.size() / 2], tq[tq.size() / 2]);
  }
  // ring readback: [l][h][slot][d] -> [l][j][h][d]
  auto ring_rows = [&](const Buffer& ring, const std::vector<int64_t>& pos) {
    std::vector<float> o(size_t(NL) * pos.size() * 8 * 128);
    const uint16_t* r = ring.as<uint16_t>();
    for (int l = 0; l < NL; ++l)
      for (size_t j = 0; j < pos.size(); ++j)
        for (int h = 0; h < 8; ++h)
          for (int dd = 0; dd < 128; ++dd)
            o[((size_t(l) * pos.size() + j) * 8 + h) * 128 + dd] =
                bf16_to_f32(r[((size_t(l) * 8 + h) * 2048 + (pos[j] & 2047)) * 128 + dd]);
    return o;
  };
  // the fp32 c rows of the prefill commit are kept in the drafter's debug buffer (first Nc rows)
  printf("[ctx] prompt context (positions %lld..%lld)\n", (long long)ctx_pos.front(), (long long)ctx_pos.back());
  {
    auto ref_c = load_npy(dump + "/ctx_c.npy").f32();
    auto ref_k = load_npy(dump + "/ctx_k.npy").f32();
    auto ref_v = load_npy(dump + "/ctx_v.npy").f32();
    report("ctx_c (fp32)", cmp(dr.ctx_c().as<float>(), ref_c.data(), int64_t(ref_c.size())));
    auto k = ring_rows(d.ring_k, ctx_pos), v = ring_rows(d.ring_v, ctx_pos);
    report("ctx_k (ring, bf16)", cmp(k.data(), ref_k.data(), int64_t(ref_k.size())));
    report("ctx_v (ring, bf16)", cmp(v.data(), ref_v.data(), int64_t(ref_v.size())));
    if (!out.empty()) {
      save_npy(out + "/ctx_k.npy", "<f4", {NL, Nc, 8, 128}, k.data(), k.size() * 4);
      save_npy(out + "/ctx_v.npy", "<f4", {NL, Nc, 8, 128}, v.data(), v.size() * 4);
    }
  }

  // ---- 2. cycles ----
  int ci = 0;
  int chains_equal = 0, chains_total = 0;
  for (;; ++ci) {
    char cdn[32];
    snprintf(cdn, sizeof cdn, "/cycle_%02d", ci);
    const std::string cd = dump + cdn;
    if (!exists(cd + "/meta.json")) break;
    json cm = json::parse(std::ifstream(cd + "/meta.json"));
    const int P = cm["P"], anchor = cm["anchor"], m = cm["committed_rows"];
    printf("[cycle %d] P=%d anchor=%d committed=%d\n", ci, P, anchor, m);
    if (P != d.ctx) fatal("cycle %d: P %d but ring holds %d", ci, P, d.ctx);
    s.tokens()[P] = anchor;
    const Npy hl = load_npy(cd + "/h_layers.npy");  // [NL+1][8][H]
    auto ref_h = hl.f32();
    if (use_h0) dr.set_h0(ref_h.data());
    else dr.clear_h0();
    double ms;
    {
      auto e = g.begin();
      dr.encode_draft(*e, s, d, P, greedy);
      e->commit_and_wait();
      ms = e->gpu_ms();
    }
    printf("  drafting pass (with %zu debug copies): %.3f ms GPU\n", dr.debug_bufs().size(), ms);
    const auto& db = dr.debug_bufs();
    auto dbf = [&](const std::string& k, size_t cnt) { return f32_of(db.at(k), cnt); };
    // embedding vs the dump's block rows (MLX's bf16 dequantization)
    if (!use_h0) report("h_0 (embedding)", cmp(dbf("h_0", 8 * H).data(), ref_h.data(), 8 * H));
    // per-layer intermediates (cycle 0)
    if (exists(cd + "/layer0.npz")) {
      for (int l = 0; l < NL; ++l) {
        auto z = load_npz(cd + "/layer" + std::to_string(l) + ".npz");
        const std::string sf = "_" + std::to_string(l);
        printf("  layer %d\n", l);
        auto rep = [&](const char* key, const std::vector<float>& ours) {
          auto r = z.at(key).f32();
          report(std::string("  ") + key, cmp(ours.data(), r.data(), int64_t(r.size())));
        };
        rep("n", dbf("n" + sf, 8 * H));
        rep("D", dbf("D" + sf, 8 * 4 * (H / 16)));
        rep("a", half_to_f32(db.at("a" + sf), 8 * H));
        {  // Q [32][8][128] -> [8][32][128]
          auto q = dbf("q" + sf, 32 * 8 * 128), qr = q;
          for (int h = 0; h < 32; ++h)
            for (int r = 0; r < 8; ++r)
              for (int x = 0; x < 128; ++x) qr[(r * 32 + h) * 128 + x] = q[(h * 8 + r) * 128 + x];
          rep("q", qr);
          auto kb = dbf("kb" + sf, 8 * 8 * 128), vb = dbf("vb" + sf, 8 * 8 * 128), kr = kb, vr = vb;
          for (int h = 0; h < 8; ++h)
            for (int r = 0; r < 8; ++r)
              for (int x = 0; x < 128; ++x) {
                kr[(r * 8 + h) * 128 + x] = kb[(h * 8 + r) * 128 + x];
                vr[(r * 8 + h) * 128 + x] = vb[(h * 8 + r) * 128 + x];
              }
          rep("kb", kr);
          rep("vb", vr);
        }
        rep("o", half_to_f32(db.at("o" + sf), 8 * 4096));
        rep("u", dbf("u" + sf, 8 * H));
        rep("h_mid", dbf("h_mid" + sf, 8 * H));
        rep("n2", dbf("n2" + sf, 8 * H));
        rep("D2", dbf("D2" + sf, 8 * 4 * (H / 16)));
        rep("mm", half_to_f32(db.at("mm" + sf), 8 * H));
        rep("hgu", half_to_f32(db.at("hgu" + sf), size_t(8) * C.I));
        rep("f", dbf("f" + sf, 8 * H));
        rep("h_out", dbf("h_" + std::to_string(l + 1), 8 * H));
      }
    }
    std::vector<float> our_h((NL + 1) * 8 * H);
    for (int l = 0; l <= NL; ++l) {
      auto x = dbf("h_" + std::to_string(l), 8 * H);
      memcpy(&our_h[size_t(l) * 8 * H], x.data(), 8 * H * 4);
      Err e = cmp(x.data(), &ref_h[size_t(l) * 8 * H], 8 * H);
      if (l > 0) g_worst_h = std::max(g_worst_h, e.rel);
      report("h_layers[" + std::to_string(l) + "]", e);
    }
    auto hn = dbf("hn", 8 * H);
    auto ref_hn = load_npy(cd + "/hn.npy").f32();
    report("hn (final norm)", cmp(hn.data(), ref_hn.data(), 8 * H));
    auto ref_lg = load_npy(cd + "/draft_logits.npy").f32();  // [7][V]
    report("draft_logits rows 1..7", cmp(dr.logits() + V, ref_lg.data(), int64_t(7) * V));
    auto ref_sel = load_npy(cd + "/sel_hidden.npy").f32();
    report("sel_hidden", cmp(dr.sel_hidden() + 256, ref_sel.data(), 7 * 256));
    auto ref_ti = load_npy(cd + "/top16_ids.npy").ints();
    auto ref_tv = load_npy(cd + "/top16_vals.npy").f32();
    int ov_total = 0, ordered_same = 0;
    printf("  top16 overlap per position:");
    for (int p = 0; p < 7; ++p) {
      std::set<int64_t> a(ref_ti.begin() + p * 16, ref_ti.begin() + p * 16 + 16), b(dr.cand_ids() + p * 16,
                                                                                  dr.cand_ids() + p * 16 + 16);
      int ov = 0;
      for (auto x : b) ov += int(a.count(x));
      ov_total += ov;
      bool same = true;
      for (int k = 0; k < 16; ++k) same = same && ref_ti[p * 16 + k] == dr.cand_ids()[p * 16 + k];
      ordered_same += same;
      printf(" %d", ov);
    }
    printf("  (%d/112, %d/7 rows identical in order)\n", ov_total, ordered_same);
    report("top16_vals", cmp(dr.cand_vals(), ref_tv.data(), 7 * 16));
    {
      // index-aligned (differs wherever two near-tied candidates swapped order) and matched by
      // (predecessor id, candidate id) over the pairs both lattices contain
      auto ref_e = load_npy(cd + "/edges.npy").f32();
      report("edges (index-aligned)", cmp(dr.edges(), ref_e.data(), 7 * 256));
      std::vector<float> a, b;
      for (int p = 0; p < 7; ++p)
        for (int j = 0; j < (p == 0 ? 1 : 16); ++j)
          for (int k = 0; k < 16; ++k) {
            const int64_t rp = p == 0 ? anchor : ref_ti[(p - 1) * 16 + j], rs = ref_ti[p * 16 + k];
            for (int j2 = 0; j2 < (p == 0 ? 1 : 16); ++j2)
              for (int k2 = 0; k2 < 16; ++k2) {
                const int64_t op = p == 0 ? anchor : dr.cand_ids()[(p - 1) * 16 + j2], os = dr.cand_ids()[p * 16 + k2];
                if (op == rp && os == rs) {
                  a.push_back(dr.edges()[p * 256 + j2 * 16 + k2]);
                  b.push_back(ref_e[p * 256 + j * 16 + k]);
                }
              }
          }
      report("edges (id-matched, " + std::to_string(a.size()) + "/1552)", cmp(a.data(), b.data(), int64_t(a.size())));
      auto ref_ws = load_npy(cd + "/walk_scores.npy").f32();
      report("walk_scores (index-aligned)", cmp(dr.walk_scores(), ref_ws.data(), 7 * 16));
    }
    auto ref_chain = load_npy(cd + "/chain.npy").ints();
    bool eq = true;
    printf("  chain ours :");
    for (int p = 0; p < 7; ++p) {
      printf(" %d", s.tokens()[P + 1 + p]);
      eq = eq && s.tokens()[P + 1 + p] == ref_chain[p];
    }
    printf("\n  chain dump :");
    for (int p = 0; p < 7; ++p) printf(" %lld", (long long)ref_chain[p]);
    printf("  -> %s\n", eq ? "IDENTICAL" : "DIFFERENT");
    if (!eq) {  // near-tie diagnostics at the first differing position
      for (int p = 0; p < 7; ++p) {
        if (s.tokens()[P + 1 + p] == ref_chain[p]) continue;
        auto ref_ws = load_npy(cd + "/walk_scores.npy").f32();
        const float* ws = dr.walk_scores() + p * 16;
        float b1 = -INFINITY, b2 = -INFINITY;
        for (int k = 0; k < 16; ++k) {
          float v = ref_ws[p * 16 + k];
          if (v > b1) { b2 = b1; b1 = v; } else if (v > b2) b2 = v;
        }
        printf("  first difference at p=%d: dump walk top-2 gap %.4f; ours S[k*]=%.4f\n", p, b1 - b2,
               ws[dr.walk_k()[p]]);
        break;
      }
    }
    chains_equal += eq;
    ++chains_total;
    if (!out.empty()) {
      const std::string od = out + cdn;
      mkdir(od.c_str(), 0755);
      save_npy(od + "/h_layers.npy", "<f4", {NL + 1, 8, H}, our_h.data(), our_h.size() * 4);
      save_npy(od + "/hn.npy", "<f4", {8, H}, hn.data(), hn.size() * 4);
      save_npy(od + "/draft_logits.npy", "<f4", {7, V}, dr.logits() + V, size_t(7) * V * 4);
      save_npy(od + "/sel_hidden.npy", "<f4", {7, 256}, dr.sel_hidden() + 256, 7 * 256 * 4);
      save_npy(od + "/top16_ids.npy", "<i4", {7, 16}, dr.cand_ids(), 7 * 16 * 4);
      save_npy(od + "/top16_vals.npy", "<f4", {7, 16}, dr.cand_vals(), 7 * 16 * 4);
      save_npy(od + "/edges.npy", "<f4", {7, 16, 16}, dr.edges(), 7 * 256 * 4);
      save_npy(od + "/walk_scores.npy", "<f4", {7, 16}, dr.walk_scores(), 7 * 16 * 4);
      save_npy(od + "/walk_k.npy", "<i4", {7}, dr.walk_k(), 7 * 4);
      save_npy(od + "/chain.npy", "<i4", {7}, s.tokens() + P + 1, 7 * 4);
    }
    // commit the kept verify rows (positions P..P+m-1) from the dump's captures
    if (exists(cd + "/commit_positions.npy")) {
      const Npy vc = load_npy(cd + "/verify_capt_bf16.npy");
      memcpy(dr.capt().ptr(), vc.as<uint16_t>(), size_t(8) * nH * 2);
      std::vector<int64_t> pos;
      for (int j = 0; j < m; ++j) pos.push_back(P + j);
      {
        auto e = g.begin();
        dr.encode_commit(*e, d, m);
        e->commit_and_wait();
        printf("  commit of %d rows: %.3f ms GPU\n", m, e->gpu_ms());
      }
      auto k = ring_rows(d.ring_k, pos), v = ring_rows(d.ring_v, pos);
      auto rk = load_npy(cd + "/commit_k.npy").f32(), rv = load_npy(cd + "/commit_v.npy").f32();
      auto rc = load_npy(cd + "/commit_c.npy").f32();
      report("commit_c (fp32)", cmp(dr.ctx_c().as<float>(), rc.data(), int64_t(rc.size())));
      report("commit_k (ring)", cmp(k.data(), rk.data(), int64_t(rk.size())));
      report("commit_v (ring)", cmp(v.data(), rv.data(), int64_t(rv.size())));
    } else {
      break;
    }
  }
  printf("== chains identical: %d/%d cycles; worst per-layer h rel err %.3e (D1 bound 1e-2)\n", chains_equal,
         chains_total, g_worst_h);

  // ---- sampled walk (T > 0): recompute q, the uniforms and the chosen candidates on the CPU ----
  {
    dr.set_debug(false);
    dr.clear_h0();
    const int P = d.ctx;
    s.tokens()[P] = int(prompt.back());
    double qerr = 0;
    int kbad = 0, sbad = 0, near = 0, total = 0;
    for (float T : {0.6f, 1.0f})
      for (uint32_t seed : {1u, 2u, 3u}) {
        SamplingParams sp;
        sp.temperature = T;
        sp.seed = seed;
        auto e = g.begin();
        dr.encode_draft(*e, s, d, P, sp);
        e->commit_and_wait();
        auto pcg = [](uint32_t v) {
          uint32_t st = v * 747796405u + 2891336453u;
          uint32_t w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u;
          return (w >> 22u) ^ w;
        };
        uint32_t jp = 0;
        for (int p = 0; p < 7; ++p) {
          const float* S = dr.walk_scores() + p * 16;
          // S_p[k] must be U_p[k] + E[p][j_p][k] for the walked predecessor j_p
          for (int k = 0; k < 16; ++k)
            if (std::fabs(S[k] - (dr.cand_vals()[p * 16 + k] + dr.edges()[p * 256 + jp * 16 + k])) > 1e-4f) ++sbad;
          double mx = -1e30, z = 0, q[16];
          for (int k = 0; k < 16; ++k) mx = std::max(mx, double(S[k]));
          for (int k = 0; k < 16; ++k) z += q[k] = std::exp((S[k] - mx) / T);
          for (int k = 0; k < 16; ++k) qerr = std::max(qerr, std::fabs(q[k] / z - dr.qprob()[p * 16 + k]));
          const float u = float(pcg(seed ^ pcg(uint32_t(P + 1 + p)) ^ 0x57A1C3E5u) & 0xffffffu) / 16777216.0f;
          int kexp = 15;
          double cum = 0;
          for (int k = 0; k < 16; ++k) {
            cum += q[k] / z;
            if (std::fabs(cum - u) < 1e-5) ++near;
            if (cum > u) { kexp = k; break; }
          }
          if (kexp != dr.walk_k()[p]) ++kbad;
          if (s.tokens()[P + 1 + p] != dr.cand_ids()[p * 16 + dr.walk_k()[p]]) ++kbad;
          jp = uint32_t(dr.walk_k()[p]);
          ++total;
        }
      }
    printf("[sampled walk] T 0.6/1.0 x 3 seeds, %d positions: max |q - softmax(S/T)| %.2e, chosen index != "
           "inverse CDF of u: %d (u within 1e-5 of a boundary: %d), walk scores inconsistent: %d\n",
           total, qerr, kbad, near, sbad);
  }

  // ---- 3. timing ----
  if (ntime > 0) {
    dr.set_debug(false);
    dr.clear_h0();
    auto time_at = [&](int P, const char* what) {
      d.ctx = P;
      s.tokens()[P] = int(prompt.back());
      // interleaved A/B of the two attention kernels (default = mma) when not profiling
      const std::vector<int> vars = Encoder::profiling() ? std::vector<int>{0} : std::vector<int>{2, 1};
      std::vector<std::vector<double>> t(vars.size());
      for (int i = 0; i < ntime + 2; ++i)
        for (size_t v = 0; v < vars.size(); ++v) {
          dr.set_attn_variant(vars[v]);
          auto e = g.begin();
          dr.encode_draft(*e, s, d, P, greedy);
          e->commit_and_wait();
          if (i >= 2) t[v].push_back(e->gpu_ms());
        }
      dr.set_attn_variant(0);
      for (size_t v = 0; v < vars.size(); ++v) {
        std::sort(t[v].begin(), t[v].end());
        printf("[time] drafting pass at P=%d (%s), attention %s: median %.3f ms, min %.3f ms, max %.3f ms over %d runs\n",
               P, what, vars[v] == 1 ? "scalar" : "mma", t[v][t[v].size() / 2], t[v].front(), t[v].back(), ntime);
      }
      if (Encoder::profiling()) Encoder::profile_report(ntime + 2);
    };
    const int P0 = d.ctx;
    if (Encoder::profiling()) {
      printf("(profile of the check phase above, discarded below)\n");
      Encoder::profile_report(1);
    }
    time_at(P0, "dump context");
    time_at(plong, "full window");
    d.ctx = P0;
  }

  // ---- 4. live: our target's captures ----
  if (live) {
    auto seq2 = tgt.new_seq(8192);
    Seq& s2 = *seq2;
    for (int i = 0; i < n; ++i) s2.tokens()[i] = int(prompt[i]);
    auto d2 = dr.new_seq();
    dr.begin_prefill(n);
    for (int p = 0; p < n; p += tgt.prefill_chunk()) {
      const int T = std::min(tgt.prefill_chunk(), n - p);
      auto e = g.begin();
      tgt.encode_prefill(*e, s2, p, T, greedy, 0, p + T == n);
      e->commit_and_wait();
    }
    dr.end_prefill();
    // compare our target's captures with MLX's, per slot (target layer)
    {
      const uint16_t* ours = dr.prompt_captures().as<uint16_t>();
      const uint16_t* ref = ctx_x.as<uint16_t>();
      for (size_t sl = 0; sl < C.target_layers.size(); ++sl) {
        std::vector<float> a(size_t(Nc) * H), b(size_t(Nc) * H);
        for (int j = 0; j < Nc; ++j)
          for (int x = 0; x < H; ++x) {
            a[size_t(j) * H + x] = bf16_to_f32(ours[size_t(j) * nH + sl * H + x]);
            b[size_t(j) * H + x] = bf16_to_f32(ref[size_t(j) * nH + sl * H + x]);
          }
        report("live capture L" + std::to_string(C.target_layers[sl]), cmp(a.data(), b.data(), int64_t(a.size())));
      }
    }
    {
      auto e = g.begin();
      dr.encode_prefill_commit(*e, *d2);
      e->commit_and_wait();
    }
    const int first = s2.tokens()[n];
    printf("[live] our target's first token %d, dump %d -> %s\n", first, int(meta["first_token"]),
           first == int(meta["first_token"]) ? "same" : "DIFFERENT");
    auto k = ring_rows(d2->ring_k, ctx_pos), v = ring_rows(d2->ring_v, ctx_pos);
    auto ref_k = load_npy(dump + "/ctx_k.npy").f32(), ref_v = load_npy(dump + "/ctx_v.npy").f32();
    report("live ctx_k vs dump", cmp(k.data(), ref_k.data(), int64_t(ref_k.size())));
    report("live ctx_v vs dump", cmp(v.data(), ref_v.data(), int64_t(ref_v.size())));
    dr.set_debug(false);
    dr.clear_h0();
    {
      auto e = g.begin();
      dr.encode_draft(*e, s2, *d2, n, greedy);
      e->commit_and_wait();
    }
    auto ref_chain = load_npy(dump + "/cycle_00/chain.npy").ints();
    auto ref_ti = load_npy(dump + "/cycle_00/top16_ids.npy").ints();
    int ov = 0;
    for (int p = 0; p < 7; ++p) {
      std::set<int64_t> a(ref_ti.begin() + p * 16, ref_ti.begin() + p * 16 + 16);
      for (int kq = 0; kq < 16; ++kq) ov += int(a.count(dr.cand_ids()[p * 16 + kq]));
    }
    printf("[live] cycle 0 chain ours:");
    for (int p = 0; p < 7; ++p) printf(" %d", s2.tokens()[n + 1 + p]);
    printf("  dump:");
    for (int p = 0; p < 7; ++p) printf(" %lld", (long long)ref_chain[p]);
    printf("  top16 overlap %d/112\n", ov);
  }
  // ---- 5. full-depth replay against our own greedy sequence ----
  if (sim > 0) {
    dr.set_debug(false);
    dr.clear_h0();
    const int cap = n + sim + 64;
    auto sq = tgt.new_seq(cap), sdr = tgt.new_seq(cap);
    for (int i = 0; i < n; ++i) sq->tokens()[i] = int(prompt[i]);
    auto d3 = dr.new_seq();
    dr.begin_prefill(n);
    for (int p = 0; p < n; p += tgt.prefill_chunk()) {
      const int T = std::min(tgt.prefill_chunk(), n - p);
      auto e = g.begin();
      tgt.encode_prefill(*e, *sq, p, T, greedy, 0, p + T == n);
      e->commit_and_wait();
    }
    dr.end_prefill();
    {
      auto e = g.begin();
      dr.encode_prefill_commit(*e, *d3);
      e->commit_and_wait();
    }
    CaptureHook hk;
    hk.slot.assign(tgt.config().layers, -1);
    for (size_t i = 0; i < C.target_layers.size(); ++i) hk.slot[C.target_layers[i]] = int(i);
    hk.ld = uint32_t(nH);
    hk.pos_base = n;
    hk.rows = sim;
    hk.dst = g.alloc(size_t(sim) * nH * 2, "sim_capt");
    tgt.set_capture(&hk);
    double td = now_ms();
    for (int p = n; p < n + sim; p += 32) {
      auto e = g.begin();
      for (int q = p; q < std::min(n + sim, p + 32); ++q) tgt.encode_step(*e, *sq, q, greedy, uint32_t(q), true);
      e->commit_and_wait();
    }
    tgt.set_capture(nullptr);
    td = now_ms() - td;
    const int* gt = sq->tokens();
    printf("[sim] %d greedy tokens decoded with capture in %.0f ms; first token %d (dump %d)\n", sim, td, gt[n],
           int(meta["first_token"]));
    int P = n, cycles = 0, mprev = 0, total = 0;
    std::vector<int> hist(8, 0), reached(7, 0), acc(7, 0);
    double gms = 0;
    while (P + 8 <= n + sim) {
      sdr->tokens()[P] = gt[P];
      auto e = g.begin();
      if (mprev) dr.encode_commit(*e, *d3, mprev);
      dr.encode_draft(*e, *sdr, *d3, P, greedy);
      e->commit_and_wait();
      gms += e->gpu_ms();
      int a = 0;
      while (a < 7 && sdr->tokens()[P + 1 + a] == gt[P + 1 + a]) ++a;
      for (int i = 0; i <= std::min(a, 6); ++i) {
        reached[i]++;
        if (i < a) acc[i]++;
      }
      const int m = a + 1;
      memcpy(dr.capt().ptr(), hk.dst.as<uint16_t>() + size_t(P - n) * nH, size_t(m) * nH * 2);
      mprev = m;
      hist[a]++;
      total += m;
      ++cycles;
      P += m;
    }
    printf("[sim] full-depth DFlash replay (%d-bit drafter): %d cycles, %.3f tok/cycle (accepted + bonus), "
           "%.3f ms GPU per cycle (commit + draft)\n", dr.bits(), cycles, double(total) / cycles, gms / cycles);
    printf("[sim] accepted-length histogram a=0..7:");
    for (int a = 0; a < 8; ++a) printf(" %d", hist[a]);
    printf("\n[sim] acceptance rate by position:");
    for (int i = 0; i < 7; ++i) printf(" %.2f", reached[i] ? double(acc[i]) / reached[i] : 0.0);
    printf("\n");
  }
  return 0;
}
