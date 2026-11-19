#include "model/moe_a4.h"

#include "model/expert_store.h"
#include "model/hy4_kernels.h"
#include "model/kparams.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>

namespace ni::a4 {

namespace {

constexpr uint32_t kPage = 16384;
constexpr uint32_t kBM = 32;        // prefill tile rows (moe_bucket3)
constexpr uint32_t kGuBN = 32, kDnBN = 64;
constexpr uint32_t kRowsTg = 16;    // rows of a rows-path threadgroup: 8 simdgroups x 2 rows

struct Params {  // A4Params
  uint32_t D, I, k, E;
  float limit;
  uint32_t R;
};

struct SlotParams {  // A4Slot
  uint32_t o[9];
};

size_t al(size_t n) { return (n + 255) / 256 * 256; }

// the scratch's views for R rows
struct Scratch {
  size_t h = 0, xh = 0, yrows = 0, xrows = 0, tiles = 0, ntiles = 0, H = 0, Y = 0, end = 0;
  uint32_t max_tiles = 0;
};

Scratch scratch_of(const Shape& sh, size_t R) {
  Scratch s;
  const size_t P = R * sh.k;
  s.max_tiles = uint32_t((P + kBM - 1) / kBM + std::min<size_t>(sh.E, P));
  size_t o = 0;
  s.xh = o, o += al(R * sh.D * 2);
  s.yrows = o, o += al(P * 4);
  s.xrows = o, o += al(P * 4);
  s.tiles = o, o += al(size_t(s.max_tiles) * 12);
  s.ntiles = o, o += al(16);
  s.H = o, o += al(P * sh.I * 2);
  s.Y = o, o += al(P * sh.D * 4);
  s.h = 0;  // the rows path: h fp32 [P, I] from the start
  s.end = std::max(o, al(P * sh.I * 4));
  return s;
}

std::string kname(const std::string& base, const Fmt& f, bool slot) {
  return base + (f.q20 ? "_q20" : "_b" + std::to_string(f.bits) + "g" + std::to_string(f.gs)) + (slot ? "_slot" : "");
}
Pipeline pipe(const std::string& base, const Fmt& f, const Experts& x) {
  return hy4::pipeline(kname(base, f, x.arena));
}
bool fmt_ok(const Fmt& f) {
  return f.q20 ? f.bits == 2 && f.gs == 64 : (f.bits == 3 || f.bits == 4) && (f.gs == 32 || f.gs == 64 || f.gs == 128);
}

SlotParams slot_params(const Experts& x) {
  SlotParams L{};
  for (int i = 0; i < 9; ++i) L.o[i] = x.off[i];
  return L;
}

// matrix m's three planes (or the arena) at buffers i .. i + 2
void bind_mat(Encoder& e, const Experts& x, int m, int i) {
  if (x.arena) {
    e.buf(i, x.a0); e.buf(i + 1, x.a1); e.buf(i + 2, x.slots);
    return;
  }
  const Buffer* b[3][3] = {{&x.gw, &x.gs, &x.gb}, {&x.uw, &x.us, &x.ub}, {&x.dw, &x.ds, &x.db}};
  const bool q20 = (m == 2 ? x.dn : x.gu).q20;
  // Q2_0 has no bias allocation; bind its scale buffer to the unused argument.
  for (int j = 0; j < 3; ++j) e.buf(i + j, *b[m][q20 && j == 2 ? 1 : j]);
}

void enc_rows_path(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xin, const Buffer& idx,
                   const Buffer& wts, const Buffer& scratch, const Buffer& y) {
  const Params p{sh.D, sh.I, sh.k, sh.E, sh.limit, uint32_t(R)};
  const SlotParams L = slot_params(x);
  const Buffer h = scratch.view(0, size_t(R) * sh.k * sh.I * 4);
  e.use(pipe("a4_gu_rows", x.gu, x));
  bind_mat(e, x, 0, 0);
  bind_mat(e, x, 1, 3);
  e.buf(6, xin); e.buf(7, idx); e.buf(8, h);
  e.val(9, p);
  e.val(10, L);
  e.dispatch((sh.I + kRowsTg - 1) / kRowsTg, uint32_t(R) * sh.k, 1, 256);
  e.use(pipe("a4_dn_rows", x.dn, x));
  bind_mat(e, x, 2, 0);
  e.buf(3, h); e.buf(4, idx); e.buf(5, wts); e.buf(6, y);
  e.val(7, p);
  e.val(8, L);
  e.dispatch((sh.D + kRowsTg - 1) / kRowsTg, uint32_t(R), 1, 256);
}

void enc_tiles_path(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xin, const Buffer& idx,
                    const Buffer& wts, const Buffer& scratch, const Buffer& y) {
  const Scratch s = scratch_of(sh, size_t(R));
  const uint32_t P = uint32_t(R) * sh.k;
  const Params p{sh.D, sh.I, sh.k, sh.E, sh.limit, uint32_t(R)};
  const SlotParams L = slot_params(x);
  const Buffer xh = scratch.view(s.xh, size_t(R) * sh.D * 2), yrows = scratch.view(s.yrows, size_t(P) * 4),
               xrows = scratch.view(s.xrows, size_t(P) * 4), tiles = scratch.view(s.tiles, size_t(s.max_tiles) * 12),
               ntiles = scratch.view(s.ntiles, 16), H = scratch.view(s.H, size_t(P) * sh.I * 2),
               Y = scratch.view(s.Y, size_t(P) * sh.D * 4);
  // x to half, and the pairs bucketed by expert into tiles of kBM rows (independent of each other)
  const uint32_t n4 = uint32_t(R) * sh.D / 4;
  e.use(hy4::pipeline("a4_x_half"));
  e.buf(0, xin); e.buf(1, xh);
  e.val(2, n4);
  e.dispatch((n4 + 255) / 256, 1, 1, 256);
  const kp::Bucket2 bp{P, sh.k, sh.E, kBM, 0u};
  e.use(hy4::pipeline("moe_bucket3"));
  e.buf(0, idx); e.buf(1, yrows); e.buf(2, xrows); e.buf(3, tiles); e.buf(4, ntiles);
  e.val(5, bp);
  e.dispatch(1, 1, 1, 1024);
  e.use(pipe("a4_gu_mm", x.gu, x));
  e.buf(0, xh);
  bind_mat(e, x, 0, 1);
  bind_mat(e, x, 1, 4);
  e.buf(7, H);
  e.val(8, p);
  e.val(9, L);
  e.buf(10, tiles); e.buf(11, ntiles); e.buf(12, xrows);
  e.dispatch(sh.I / kGuBN, s.max_tiles, 1, 128);
  e.use(pipe("a4_dn_mm", x.dn, x));
  e.buf(0, H);
  bind_mat(e, x, 2, 1);
  e.buf(4, Y);
  e.val(5, p);
  e.val(6, L);
  e.buf(7, tiles); e.buf(8, ntiles); e.buf(9, yrows);
  e.dispatch(sh.D / kDnBN, s.max_tiles, 1, 128);
  e.use(hy4::pipeline("a4_combine"));
  e.buf(0, Y); e.buf(1, wts); e.buf(2, y);
  e.val(3, p);
  e.dispatch((sh.D / 4 + 63) / 64, uint32_t(R), 1, 64);
}

}  // namespace

std::string fmt_name(const Fmt& f) {
  return (f.q20 ? "Q2_0 " : "") + std::to_string(f.bits) + "-bit group " + std::to_string(f.gs);
}

std::vector<std::string> kernel_names(const Fmt& gu, const Fmt& dn) {
  std::vector<std::string> v = {"a4_x_half", "moe_bucket3", "a4_combine"};
  for (bool slot : {false, true}) {
    v.push_back(kname("a4_gu_rows", gu, slot));
    v.push_back(kname("a4_gu_mm", gu, slot));
    v.push_back(kname("a4_dn_rows", dn, slot));
    v.push_back(kname("a4_dn_mm", dn, slot));
  }
  return v;
}

std::vector<std::string> kernel_names() {
  std::set<std::string> all;
  for (uint32_t bits : {3u, 4u})
    for (uint32_t gs : {32u, 64u, 128u})
      for (const std::string& n : kernel_names(Fmt{bits, gs}, Fmt{bits, gs})) all.insert(n);
  for (const std::string& n : kernel_names(Fmt{2, 64, true}, Fmt{2, 64, true})) all.insert(n);
  return {all.begin(), all.end()};
}

bool shapes_ok(const Shape& sh, std::string* why) {
  auto no = [why](const char* m) {
    if (why) *why = m;
    return false;
  };
  if (!sh.D || !sh.I || sh.D % 64 || sh.I % 64) return no("D and I must be multiples of 64");
  if (sh.D > 2048) return no("D over 2048 (the rows path stages x in threadgroup memory)");
  if (sh.k == 0 || sh.k > 16) return no("top-k must be 1-16");
  if (sh.E == 0 || sh.E > 512) return no("1-512 experts (moe_bucket3)");
  return true;
}

bool formats_ok(const Shape& sh, const Fmt& gu, const Fmt& dn, std::string* why) {
  if (!shapes_ok(sh, why)) return false;
  for (const Fmt* f : {&gu, &dn}) {
    if (!fmt_ok(*f)) {
      if (why) *why = fmt_name(*f) + " (expected affine 3/4-bit group 32/64/128 or Q2_0 2-bit group 64)";
      return false;
    }
  }
  if (sh.D % gu.gs || sh.I % dn.gs) {
    if (why) *why = "a group size does not divide its matrices' rows";
    return false;
  }
  return true;
}

SlotLayout slot_layout(const Shape& sh, const Fmt& gu, const Fmt& dn) {
  // gate / up [I, D], down [D, I]: I * D * bits / 8 bytes of codes, I * D / gs * 2 of scales and of biases each
  SlotLayout L;
  uint32_t at = 0;
  for (uint32_t m = 0; m < 3; ++m) {
    const Fmt& f = m == 2 ? dn : gu;
    L.off[3 * m] = at;
    L.len[3 * m] = sh.I * sh.D / 8 * f.bits;
    at += (L.len[3 * m] + kPage - 1) / kPage * kPage;
  }
  for (uint32_t m = 0; m < 3; ++m) {
    const Fmt& f = m == 2 ? dn : gu;
    for (uint32_t j = 0; j < (f.q20 ? 1u : 2u); ++j) {
      L.off[3 * m + 1 + j] = at;
      L.len[3 * m + 1 + j] = sh.I * sh.D / f.gs * 2;
      at += L.len[3 * m + 1 + j];
    }
  }
  L.bytes = (at + kPage - 1) / kPage * kPage;
  return L;
}

size_t scratch_bytes(const Shape& sh, int tmax) { return scratch_of(sh, size_t(std::max(tmax, 1))).end; }

void enc_rows(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xin, const Buffer& idx,
              const Buffer& wts, const Buffer& scratch, const Buffer& y, Path path) {
  if (R <= 0) return;
  if (path == Path::Auto) path = R <= kRowsMax ? Path::Rows : Path::Tiles;
  // the kernels write the scratch without bounds checks: a buffer sized for fewer rows is refused here
  const size_t need = path == Path::Rows ? size_t(R) * sh.k * sh.I * 4 : scratch_of(sh, size_t(R)).end;
  if (scratch.size < need) fatal("a4: %d rows need %zu bytes of scratch, the buffer has %zu", R, need, scratch.size);
  if (path == Path::Rows) enc_rows_path(e, x, sh, R, xin, idx, wts, scratch, y);
  else enc_tiles_path(e, x, sh, R, xin, idx, wts, scratch, y);
}

void enc_rows(Encoder& e, const Experts& x, const Shape& sh, int R, const Buffer& xin, const Buffer& idx,
              const Buffer& wts, const Buffer& scratch, const Buffer& y) {
  enc_rows(e, x, sh, R, xin, idx, wts, scratch, y, Path::Auto);
}

// ---- Store ----

namespace {

struct StTensor {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t off = 0, len = 0;  // absolute in the file
};

// a safetensors file's header: name -> dtype, shape, absolute extent
std::map<std::string, StTensor> read_header(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) fatal("a4 store: cannot open %s", path.c_str());
  uint64_t n = 0;
  if (pread(fd, &n, 8, 0) != 8 || n == 0 || n > (uint64_t(1) << 28)) {
    close(fd);
    fatal("a4 store: %s is not a safetensors file", path.c_str());
  }
  std::string h(size_t(n), '\0');
  size_t done = 0;
  while (done < n) {
    const ssize_t r = pread(fd, h.data() + done, n - done, off_t(8 + done));
    if (r <= 0) break;
    done += size_t(r);
  }
  close(fd);
  if (done < n) fatal("a4 store: short header in %s", path.c_str());
  std::map<std::string, StTensor> out;
  const nlohmann::json j = nlohmann::json::parse(h, nullptr, false);
  if (j.is_discarded() || !j.is_object()) fatal("a4 store: bad header in %s", path.c_str());
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& v = it.value();
    StTensor t;
    t.dtype = v.value("dtype", "");
    for (const auto& d : v.at("shape")) t.shape.push_back(d.get<int64_t>());
    const uint64_t b = v.at("data_offsets")[0].get<uint64_t>(), e = v.at("data_offsets")[1].get<uint64_t>();
    t.off = 8 + n + b;
    t.len = e - b;
    out[it.key()] = std::move(t);
  }
  return out;
}

size_t ram_bytes() {
  uint64_t ram = 0;
  size_t len = sizeof ram;
  sysctlbyname("hw.memsize", &ram, &len, nullptr, 0);
  return size_t(ram);
}

}  // namespace

struct Store::Impl {
  Shape sh;
  std::vector<SlotLayout> lay;  // per store layer
  std::vector<Fmt> gu, dn;      // per store layer
  std::vector<int> layers;
  std::map<int, int> index;  // trunk layer -> store layer
  std::unique_ptr<q20::ExpertStore> st;
  int at(int layer) const {
    auto it = index.find(layer);
    if (it == index.end()) fatal("a4 store: layer %d has no streamed experts", layer);
    return it->second;
  }
};

Store::Store(const std::string& dir, const std::vector<int>& layers, const Shape& sh, size_t arena_bytes)
    : impl_(std::make_unique<Impl>()) {
  Impl& m = *impl_;
  m.sh = sh;
  m.layers = layers;
  std::string why;
  if (!shapes_ok(sh, &why)) fatal("a4 store: %s", why.c_str());
  if (layers.empty()) fatal("a4 store: no layers");
  for (size_t i = 0; i < layers.size(); ++i) m.index[layers[i]] = int(i);
  // config.json's quantization (when the folder has one): the default and the entries keyed by module path
  nlohmann::json quant;
  {
    std::ifstream f(dir + "/config.json");
    if (f) {
      const nlohmann::json c = nlohmann::json::parse(f, nullptr, false);
      if (!c.is_discarded() && c.is_object()) {
        if (c.contains("quantization") && c["quantization"].is_object()) quant = c["quantization"];
        else if (c.contains("quantization_config") && c["quantization_config"].is_object()) quant = c["quantization_config"];
      }
    }
  }
  auto config_fmt = [&](const std::string& module, Fmt* f) {
    if (!quant.is_object()) return false;
    const nlohmann::json& q = quant.contains(module) && quant[module].is_object() ? quant[module] : quant;
    const std::string mode = q.value("mode", "affine");
    if (mode != "affine" && mode != "q2_0") fatal("a4 store: unsupported mode %s for %s", mode.c_str(), module.c_str());
    f->q20 = mode == "q2_0";
    if (!f->q20 && (!q.contains("bits") || !q.contains("group_size"))) return false;
    f->bits = q.value("bits", 2u);
    f->gs = q.value("group_size", 64u);
    return true;
  };
  // where each tensor is: the index's weight map, else every *.safetensors of the folder
  std::map<std::string, std::string> where;  // tensor -> file name
  {
    std::ifstream f(dir + "/model.safetensors.index.json");
    if (f) {
      const nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
      if (j.is_discarded() || !j.contains("weight_map")) fatal("a4 store: bad %s/model.safetensors.index.json", dir.c_str());
      for (auto it = j["weight_map"].begin(); it != j["weight_map"].end(); ++it) where[it.key()] = it.value().get<std::string>();
    }
  }
  std::map<std::string, std::map<std::string, StTensor>> headers;  // file name -> header
  if (where.empty()) {
    DIR* d = opendir(dir.c_str());
    if (!d) fatal("a4 store: cannot read %s", dir.c_str());
    while (dirent* de = readdir(d)) {
      const std::string n = de->d_name;
      if (n.size() > 12 && n.compare(n.size() - 12, 12, ".safetensors") == 0) {
        headers[n] = read_header(dir + "/" + n);
        for (const auto& kv : headers[n]) where[kv.first] = n;
      }
    }
    closedir(d);
  }
  std::vector<std::string> files;
  std::map<std::string, int> fidx;
  std::vector<q20::LayerSrc> src(layers.size());
  m.lay.resize(layers.size());
  m.gu.resize(layers.size());
  m.dn.resize(layers.size());
  static const char* const kMat[3] = {"gate", "up", "down"};
  static const char* const kPart[3] = {"weight", "scales", "biases"};
  auto tensor = [&](const std::string& n) -> const StTensor& {
    auto w = where.find(n);
    if (w == where.end()) fatal("a4 store: %s is not in %s", n.c_str(), dir.c_str());
    auto h = headers.find(w->second);
    if (h == headers.end()) h = headers.emplace(w->second, read_header(dir + "/" + w->second)).first;
    auto t = h->second.find(n);
    if (t == h->second.end()) fatal("a4 store: %s is not in %s", n.c_str(), w->second.c_str());
    return t->second;
  };
  std::map<std::pair<std::string, std::string>, int> mix;  // gate/up and down formats -> layers, for the log
  uint32_t slot_bytes = 0;
  for (size_t li = 0; li < layers.size(); ++li) {
    // each matrix's format from its planes' shapes: codes [E, out, in * bits / 32], scales [E, out, in / gs]
    Fmt f[3];
    for (int mi = 0; mi < 3; ++mi) {
      const std::string base = "model.layers." + std::to_string(layers[li]) + ".mlp.switch_mlp." + kMat[mi] + "_proj";
      const StTensor& W = tensor(base + ".weight");
      const StTensor& S = tensor(base + ".scales");
      const uint32_t in = mi == 2 ? sh.I : sh.D;
      if (W.shape.size() != 3 || S.shape.size() != 3 || W.shape[2] <= 0 || S.shape[2] <= 0 ||
          (uint64_t(W.shape[2]) * 32) % in || in % uint64_t(S.shape[2]))
        fatal("a4 store: %s's planes do not fit %u columns", base.c_str(), in);
      f[mi].bits = uint32_t(uint64_t(W.shape[2]) * 32 / in);
      f[mi].gs = uint32_t(in / uint64_t(S.shape[2]));
      Fmt c;
      if (config_fmt(base, &c)) {
        f[mi].q20 = c.q20;
        if (c != f[mi])
          fatal("a4 store: %s is %s by its shapes, %s by config.json", base.c_str(), fmt_name(f[mi]).c_str(),
                fmt_name(c).c_str());
      }
      if (f[mi].q20 && where.count(base + ".biases")) fatal("a4 store: Q2_0 matrix %s has a bias plane", base.c_str());
    }
    if (f[0] != f[1])
      fatal("a4 store: layer %d's gate (%s) and up (%s) differ in format", layers[li], fmt_name(f[0]).c_str(),
            fmt_name(f[1]).c_str());
    if (!formats_ok(sh, f[0], f[2], &why)) fatal("a4 store: layer %d: %s", layers[li], why.c_str());
    m.gu[li] = f[0];
    m.dn[li] = f[2];
    m.lay[li] = slot_layout(sh, f[0], f[2]);
    slot_bytes = std::max(slot_bytes, m.lay[li].bytes);
    ++mix[{fmt_name(f[0]), fmt_name(f[2])}];
    for (int mi = 0; mi < 3; ++mi)
      for (int pi = 0; pi < (f[mi].q20 ? 2 : 3); ++pi) {
        const std::string n = "model.layers." + std::to_string(layers[li]) + ".mlp.switch_mlp." + kMat[mi] + "_proj." +
                              kPart[pi];
        const StTensor& T = tensor(n);
        const uint32_t out = mi == 2 ? sh.D : sh.I, in = mi == 2 ? sh.I : sh.D;
        const Fmt& fm = f[mi];
        const std::vector<int64_t> want = {int64_t(sh.E), int64_t(out), int64_t(pi == 0 ? in * fm.bits / 32 : in / fm.gs)};
        const uint64_t row_bytes = pi == 0 ? uint64_t(in) * fm.bits / 8 : uint64_t(in) / fm.gs * 2;
        const char* dt = pi == 0 ? "U32" : fm.q20 ? "F16" : "BF16";
        if (T.dtype != dt || T.shape != want || T.len != uint64_t(sh.E) * out * row_bytes)
          fatal("a4 store: %s is %s [%s], not %s [%lld, %lld, %lld]", n.c_str(), T.dtype.c_str(),
                [&] {
                  std::string s;
                  for (int64_t v : T.shape) s += (s.empty() ? "" : ", ") + std::to_string(v);
                  return s;
                }().c_str(),
                dt, (long long)want[0], (long long)want[1], (long long)want[2]);
        const std::string path = dir + "/" + where.at(n);
        auto fi = fidx.find(path);
        if (fi == fidx.end()) {
          fi = fidx.emplace(path, int(files.size())).first;
          files.push_back(path);
        }
        q20::SlotPlanes& lp = src[li].planes;
        const uint32_t active = lp.n++;
        lp.at[active] = m.lay[li].off[mi * 3 + pi];
        lp.len[active] = m.lay[li].len[mi * 3 + pi];
        q20::PlaneSrc& p = src[li].p[active];
        p.file = fi->second;
        p.off = T.off;
        p.stride = T.len / sh.E;
        p.len = uint32_t(T.len / sh.E);
      }
    q20::SlotPlanes& lp = src[li].planes;  // the layer's planes in its own layout
    lp.bytes = m.lay[li].bytes;
  }
  // every slot holds the largest layer's expert
  q20::SlotPlanes planes = src[0].planes;
  for (const q20::LayerSrc& l : src)
    if (l.planes.bytes == slot_bytes) planes = l.planes;
  for (q20::LayerSrc& l : src) l.planes.bytes = slot_bytes;
  std::string mix_s;
  for (const auto& [k, n] : mix) {
    mix_s += (mix_s.empty() ? "" : ", ") + std::to_string(n) + " x " + k.first +
             (k.first == k.second ? std::string() : " (down " + k.second + ")");
  }
  // the arena: the caller's bytes, NI_EXPERT_STORE=<GB>|all, else 42% of the RAM (~20 GB on a 48 GB Mac: room for
  // the resident weights, ~8 GB for Hy4, the KV caches and the rest) or what the GPU's working set leaves after what
  // is allocated so far and 4 GB, whichever is less; never below the cache's 3 E slots
  const size_t all = layers.size() * sh.E * size_t(slot_bytes), least = size_t(3) * sh.E * size_t(slot_bytes);
  size_t bytes = arena_bytes;
  const char* why_size = "the caller's";
  if (!bytes) {
    size_t pin = 0;
    if (q20::store_pin_bytes(getenv("NI_EXPERT_STORE"), all, pin)) {
      bytes = pin;
      why_size = "NI_EXPERT_STORE";
    } else {
      Gpu& g = Gpu::get();
      const size_t rws = g.recommended_working_set(), used = g.allocated_bytes(), keep = size_t(4) << 30;
      const size_t share = size_t(0.42 * double(ram_bytes()));
      const size_t left = rws > used + keep ? rws - used - keep : 0;
      bytes = std::min(share, left);
      why_size = "default (42% of the RAM, at most the working set less what is allocated and 4 GB)";
    }
  }
  bytes = std::min(all, std::max(bytes, least));
  q20::ExpertStore::Options o;
  o.bytes = bytes;
  if (const char* r = getenv("NI_EXPERT_READERS"); r && *r) o.cache.readers = atoi(r);
  if (const char* io = getenv("NI_EXPERT_IO"); io && !strcmp(io, "direct")) o.cache.direct = true;
  if (const char* fl = getenv("NI_EXPERT_FILL"); fl && !strcmp(fl, "0")) o.cache.fill = false;
  if (const char* sv = getenv("NI_SCAN_VICTIM"); sv && !strcmp(sv, "0")) o.cache.scan_victim = false;
  logi("a4 store: %zu layers of %u experts (%s), slots of %u pages, arena %.2f GB (%s) of %.2f GB, %zu files",
       layers.size(), sh.E, mix_s.c_str(), slot_bytes / kPage, bytes / 1e9, why_size, all / 1e9, files.size());
  m.st = std::make_unique<q20::ExpertStore>(files, src, sh.E, planes, o);
}

Store::~Store() = default;

void Store::begin(Encoder& e, int layer, const Buffer& ids, uint32_t n, const Buffer* pred, uint32_t npred, bool scan) {
  Impl& m = *impl_;
  const int s = m.at(layer);
  const int next = s + 1 < int(m.layers.size()) ? s + 1 : -1;
  // a prefill chunk of 64+ rows reads every absent expert of the next layer ahead (it routes most of them anyway);
  // decode and verify rows read the predicted ids ahead
  const bool all = scan && next >= 0 && n >= 64 * m.sh.k;
  const bool use_pred = !all && pred && next >= 0;
  m.st->begin(e, s, ids, n, use_pred ? pred : nullptr, use_pred ? npred : 0, all || use_pred ? next : -1, all, scan,
              /*warm=*/!scan);
}

void Store::end(Encoder& e) { impl_->st->end(e); }

Experts Store::experts(int layer) const {
  const Impl& m = *impl_;
  Experts x;
  x.arena = true;
  x.a0 = m.st->arena0();
  x.a1 = m.st->arena1();
  const int s = m.at(layer);
  x.slots = m.st->table(s);
  for (int i = 0; i < 9; ++i) x.off[i] = m.lay[size_t(s)].off[i];
  x.gu = m.gu[size_t(s)];
  x.dn = m.dn[size_t(s)];
  return x;
}

std::vector<std::string> Store::kernel_names() const {
  std::set<std::string> all;
  for (size_t i = 0; i < impl_->layers.size(); ++i)
    for (const std::string& n : a4::kernel_names(impl_->gu[i], impl_->dn[i])) all.insert(n);
  return {all.begin(), all.end()};
}

std::vector<Buffer> Store::buffers() const { return impl_->st->buffers(); }
std::string Store::describe() const { return impl_->st->describe(); }
size_t Store::arena_bytes() const { return impl_->st->arena_bytes(); }
bool Store::complete() const { return impl_->st->complete(); }

Store::Counts Store::counts() const {
  const q20::StoreStats st = impl_->st->stats();
  Counts c;
  c.handshakes = st.handshakes, c.lookups = st.lookups, c.misses = st.misses, c.prefetches = st.prefetches;
  c.evictions = st.evictions, c.mb_read = st.bytes / 1e6, c.stall_ms = st.stall_ms;
  return c;
}

}  // namespace ni::a4
