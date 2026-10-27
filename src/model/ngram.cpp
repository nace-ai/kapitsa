#include "model/ngram.h"
#include "gguf/gguf.h"
#include "gguf/quant.h"
#include "loader/safetensors.h"
#include "util/bf16.h"
#include "util/log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <thread>

namespace ni {

namespace {

constexpr uint64_t kGamma = 0x9E3779B97F4A7C15ull;

uint64_t splitmix64(uint64_t v) {
  v += kGamma;
  v = (v ^ (v >> 30)) * 0xBF58476D1CE4E5B9ull;
  v = (v ^ (v >> 27)) * 0x94D049BB133111EBull;
  return v ^ (v >> 31);
}

bool is_prime(int64_t v) {
  if (v < 2) return false;
  if (v % 2 == 0) return v == 2;
  for (int64_t d = 3; d * d <= v; d += 2)
    if (v % d == 0) return false;
  return true;
}

// FP8 E4M3 (finite, no infinities; 0x7f / 0xff are NaN) -> float
float f8e4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? std::ldexp(float(m), -9) : std::ldexp(1.0f + float(m) / 8.0f, e - 7);
  if (e == 15 && m == 7) v = NAN;
  return s ? -v : v;
}

}  // namespace

NgramHash NgramHash::make(int vocab, int seed, int ple_index, int64_t base, int per_ngram, int ngram, int eos) {
  NgramHash h;
  if (ngram < 2 || ngram > 8) fatal("n-gram size %d unsupported", ngram);
  h.ngram = ngram;
  h.per_ngram = per_ngram;
  h.heads = (ngram - 1) * per_ngram;
  h.eos = eos;
  // multipliers: odd, below (2^63 - 1) / vocab so that token * multiplier never overflows
  const uint64_t mmax = uint64_t(INT64_MAX) / uint64_t(std::max(vocab, 1));
  const uint64_t half = std::max<uint64_t>(1, mmax / 2);
  const uint64_t bseed = uint64_t(int64_t(seed) + 10007ll * ple_index);
  for (int i = 0; i < ngram; ++i) h.mult[i] = int64_t(2 * (splitmix64(bseed + kGamma * uint64_t(i + 1)) % half) + 1);
  // head h's modulus: the (ple_index * heads + h + 1)-th prime after base - 1
  int64_t p = base - 1;
  for (int i = 0; i < ple_index * h.heads; ++i)
    for (++p; !is_prime(p); ++p) {
    }
  for (int i = 0; i < h.heads; ++i) {
    for (++p; !is_prime(p); ++p) {
    }
    h.mod.push_back(p);
    h.off.push_back(h.rows);
    h.rows += p;
  }
  return h;
}

void NgramHash::ids(const int* tok, int t, int64_t* out) const {
  // tok_k(t): the token k back, or eos before the sequence or when an eos lies in [t - k, t - 1] (an eos at t - k
  // itself gives eos either way, so only (t - k, t - 1] is tested; the cut carries over to longer shifts)
  int64_t sh[8];
  sh[0] = tok[t];
  bool cut = false;
  for (int k = 1; k < ngram; ++k) {
    cut = cut || t - k < 0 || (k > 1 && tok[t - k + 1] == eos);
    sh[k] = cut ? eos : tok[t - k];
  }
  int64_t x = sh[0] * mult[0];
  for (int m = 2; m <= ngram; ++m) {
    x ^= sh[m - 1] * mult[m - 1];
    for (int j = 0; j < per_ngram; ++j) {
      const int hd = (m - 2) * per_ngram + j;
      out[hd] = x % mod[size_t(hd)] + off[size_t(hd)];
    }
  }
}

NgramTable::~NgramTable() {
  for (auto& [p, n] : maps_) munmap(p, n);
}

// clean file pages: the page cache keeps what is used, outside the process footprint
std::pair<const uint8_t*, size_t> NgramTable::map_file(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) fatal("n-gram table: cannot open %s", path.c_str());
  struct stat st;
  fstat(fd, &st);
  void* p = mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
  ::close(fd);
  if (p == MAP_FAILED) fatal("n-gram table: mapping %s failed", path.c_str());
  madvise(p, size_t(st.st_size), MADV_RANDOM);
  maps_.push_back({p, size_t(st.st_size)});
  mapped_ += size_t(st.st_size);
  return {static_cast<const uint8_t*>(p), size_t(st.st_size)};
}

void NgramTable::open(const std::vector<Part>& parts, Fmt fmt, int dim, int bits, int gs, float scale) {
  fmt_ = fmt;
  dim_ = dim;
  bits_ = bits;
  gs_ = gs;
  scale_ = scale;
  // headers of the files the parts name, then one read-only shared mapping per file
  std::map<std::string, std::pair<const uint8_t*, size_t>> files;
  auto mapped = [&](const std::string& path) -> const uint8_t* {
    auto it = files.find(path);
    if (it == files.end()) it = files.emplace(path, map_file(path)).first;
    return it->second.first;
  };
  // tensor offsets from each file's header
  auto entry = [&](const std::string& path, const std::string& name, size_t& off, std::vector<int64_t>& shape) {
    const uint8_t* base = mapped(path);
    uint64_t n = 0;
    memcpy(&n, base, 8);
    const std::string h(reinterpret_cast<const char*>(base) + 8, size_t(n));
    // a minimal scan of the header: "name":{"dtype":"..","shape":[..],"data_offsets":[a,b]}
    const std::string key = "\"" + name + "\"";
    const size_t at = h.find(key);
    if (at == std::string::npos) fatal("n-gram table: %s has no tensor %s", path.c_str(), name.c_str());
    const size_t sp = h.find("\"shape\"", at), dp = h.find("\"data_offsets\"", at);
    if (sp == std::string::npos || dp == std::string::npos) fatal("n-gram table: bad header in %s", path.c_str());
    shape.clear();
    size_t i = h.find('[', sp) + 1;
    while (h[i] != ']') {
      shape.push_back(std::stoll(h.substr(i)));
      i = h.find_first_of(",]", i);
      if (h[i] == ',') ++i;
    }
    const size_t j = h.find('[', dp) + 1;
    off = 8 + size_t(n) + size_t(std::stoull(h.substr(j)));
  };
  parts_.clear();
  rows_ = 0;
  if (fmt == Fmt::Affine && (bits_ == 0 || gs_ == 0) && !parts.empty()) {  // from the first part's shapes
    size_t off = 0;
    std::vector<int64_t> ws, ss;
    entry(parts[0].file, parts[0].w, off, ws);
    entry(parts[0].file, parts[0].s, off, ss);
    if (ws.size() != 2 || ss.size() != 2 || ss[1] <= 0) fatal("n-gram table: unexpected shapes");
    bits_ = int(ws[1] * 32 / dim);
    gs_ = int(dim / ss[1]);
  }
  if (fmt == Fmt::Affine && (bits_ < 2 || bits_ > 8 || gs_ <= 0 || dim % gs_ || (dim * bits_) % 32))
    fatal("n-gram table: %d-bit group %d codes of %d values per row unsupported", bits_, gs_, dim);
  row_bytes_ = fmt == Fmt::BF16 ? size_t(dim) * 2 : fmt == Fmt::F8E4M3 ? size_t(dim) : size_t(dim) * size_t(bits_) / 8;
  group_row_bytes_ = fmt == Fmt::Affine ? size_t(dim / gs_) * 2 : 0;
  for (const Part& p : parts) {
    Map m;
    size_t off = 0;
    std::vector<int64_t> shape;
    entry(p.file, p.w, off, shape);
    m.w = mapped(p.file) + off;
    if (shape.size() != 2) fatal("n-gram table: %s is not 2-D", p.w.c_str());
    m.rows = shape[0];
    const size_t want = fmt == Fmt::Affine ? row_bytes_ / 4 : size_t(dim);
    if (size_t(shape[1]) != want) fatal("n-gram table: %s has %lld columns, expected %zu", p.w.c_str(), (long long)shape[1], want);
    if (fmt == Fmt::Affine) {
      entry(p.file, p.s, off, shape);
      m.s = mapped(p.file) + off;
      entry(p.file, p.b, off, shape);
      m.b = mapped(p.file) + off;
    }
    m.first = rows_;
    rows_ += m.rows;
    parts_.push_back(m);
  }
}

void NgramTable::open_gguf(const std::string& file, const std::string& name, int dim) {
  uint64_t off = 0, rows = 0;
  {
    std::unique_ptr<gguf::File> g;
    try {
      g = std::make_unique<gguf::File>(file);
    } catch (const gguf::Error& e) {
      fatal("n-gram table: %s", e.what());
    }
    const gguf::Tensor* t = g->find(name);
    if (!t) fatal("n-gram table: %s has no tensor %s", file.c_str(), name.c_str());
    uint32_t be = 0, bb = 0;
    if (t->ne.size() != 2 || t->ne[0] != uint64_t(dim))
      fatal("n-gram table: %s is not [rows, %d] (ne[0] = %llu)", name.c_str(), dim, (unsigned long long)t->ne[0]);
    if (!gguf::can_dequantize(t->type) || !gguf::type_block(t->type, &be, &bb) || dim % int(be))
      fatal("n-gram table: %s is %s, which is not read here in rows of %d values", name.c_str(),
            gguf::type_name(t->type).c_str(), dim);
    gtype_ = t->type;
    row_bytes_ = size_t(dim / int(be)) * bb;
    off = t->offset;
    rows = t->ne[1];
  }
  fmt_ = Fmt::Gguf;
  dim_ = dim;
  group_row_bytes_ = 0;
  const auto [base, size] = map_file(file);
  if (off + rows * row_bytes_ > size) fatal("n-gram table: %s ends before its tensor %s", file.c_str(), name.c_str());
  Map m;
  m.w = base + off;
  m.rows = int64_t(rows);
  parts_ = {m};
  rows_ = int64_t(rows);
}

bool NgramTable::has_tensor(const std::string& file, const std::string& name) {
  const int fd = ::open(file.c_str(), O_RDONLY);
  if (fd < 0) return false;
  uint64_t n = 0;
  bool found = false;
  if (pread(fd, &n, 8, 0) == 8 && n < (512u << 20)) {
    std::string h(size_t(n), '\0');
    if (pread(fd, h.data(), size_t(n), 8) == ssize_t(n)) found = h.find("\"" + name + "\"") != std::string::npos;
  }
  ::close(fd);
  return found;
}

std::string NgramTable::describe() const {
  char b[160];
  snprintf(b, sizeof b, "%lld x %d, %s, %.1f GB mapped", (long long)rows_, dim_,
           fmt_ == Fmt::BF16     ? "bf16"
           : fmt_ == Fmt::F8E4M3 ? "fp8 e4m3"
           : fmt_ == Fmt::Gguf   ? ("GGUF " + gguf::type_name(gtype_)).c_str()
                                 : (std::to_string(bits_) + "-bit group " + std::to_string(gs_)).c_str(),
           mapped_ / 1e9);
  return b;
}

const NgramTable::Map& NgramTable::part_of(int64_t r) const {
  if (r < 0 || r >= rows_) fatal("n-gram row %lld out of range", (long long)r);
  // parts are consecutive: find the one holding r (few parts; usually equal sizes)
  size_t pi = parts_.size() > 1 && parts_[0].rows > 0 ? std::min(parts_.size() - 1, size_t(r / parts_[0].rows)) : 0;
  while (pi > 0 && parts_[pi].first > r) --pi;
  while (pi + 1 < parts_.size() && parts_[pi + 1].first <= r) ++pi;
  return parts_[pi];
}

void NgramTable::prefetch(int64_t r) const {
  const Map& m = part_of(r);
  const int64_t lr = r - m.first;
  static const uintptr_t pg = uintptr_t(getpagesize());
  auto will = [&](const uint8_t* p, size_t n) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~(pg - 1), e = reinterpret_cast<uintptr_t>(p) + n;
    madvise(reinterpret_cast<void*>(a), size_t(e - a), MADV_WILLNEED);
  };
  will(m.w + size_t(lr) * row_bytes_, row_bytes_);
  if (fmt_ == Fmt::Affine) {
    will(m.s + size_t(lr) * group_row_bytes_, group_row_bytes_);
    will(m.b + size_t(lr) * group_row_bytes_, group_row_bytes_);
  }
}

void NgramTable::row(int64_t r, float* out) const {
  const Map& m = part_of(r);
  const int64_t lr = r - m.first;
  const uint8_t* w = m.w + size_t(lr) * row_bytes_;
  switch (fmt_) {
    case Fmt::BF16: {
      const uint16_t* v = reinterpret_cast<const uint16_t*>(w);
      for (int i = 0; i < dim_; ++i) out[i] = bf16_to_f32(v[i]);
      break;
    }
    case Fmt::F8E4M3: {
      static float lut[256];
      static const bool init = [] {
        for (int i = 0; i < 256; ++i) lut[i] = f8e4m3(uint8_t(i));
        return true;
      }();
      (void)init;
      for (int i = 0; i < dim_; ++i) out[i] = lut[w[i]] * scale_;
      break;
    }
    case Fmt::Affine: {
      const uint16_t* sc = reinterpret_cast<const uint16_t*>(m.s + size_t(lr) * group_row_bytes_);
      const uint16_t* bi = reinterpret_cast<const uint16_t*>(m.b + size_t(lr) * group_row_bytes_);
      const uint32_t* q = reinterpret_cast<const uint32_t*>(w);
      const uint32_t mask = (1u << bits_) - 1;
      for (int i = 0; i < dim_; ++i) {
        const uint64_t bit = uint64_t(i) * uint64_t(bits_);
        uint64_t word = q[bit / 32];
        if (bit % 32 + uint64_t(bits_) > 32) word |= uint64_t(q[bit / 32 + 1]) << 32;
        const uint32_t c = uint32_t(word >> (bit % 32)) & mask;
        out[i] = bf16_to_f32(sc[i / gs_]) * float(c) + bf16_to_f32(bi[i / gs_]);
      }
      break;
    }
    case Fmt::Gguf:
      gguf::dequantize(gtype_, w, out, size_t(dim_));
      break;
  }
}

void NgramTable::gather(const int64_t* ids, int T, int heads, float* out) const {
  const size_t n = size_t(T) * size_t(heads);
  static const bool pre = !(getenv("NI_PLE_PREFETCH") && !strcmp(getenv("NI_PLE_PREFETCH"), "0"));
  auto work = [&](size_t a, size_t b) {
    for (size_t i = a; pre && i < b; ++i) prefetch(ids[i]);
    for (size_t i = a; i < b; ++i) row(ids[i], out + i * size_t(dim_));
  };
  // Each worker first asks for all its rows' pages (madvise WILLNEED), then decodes: reads of rows not in the page
  // cache overlap instead of faulting one after another, and resident pages are mapped in one call each rather than
  // by faults. Decode (16 rows on the GPU's path: the next pass waits for them), real pack, M5 Max: 1.35-1.48 ms ->
  // 0.04-0.27 ms per token, 76 -> 87 tok/s (NI_PLE_PREFETCH=0: off). Large gathers (prefill chunks) also take threads.
  const unsigned nt = n >= 256 ? std::max(1u, std::min(8u, std::thread::hardware_concurrency())) : 1u;
  if (nt == 1) {
    work(0, n);
    return;
  }
  std::vector<std::thread> th;
  const size_t per = (n + nt - 1) / nt;
  for (unsigned t = 0; t < nt; ++t)
    if (t * per < n) th.emplace_back(work, t * per, std::min(n, (t + 1) * per));
  for (auto& t : th) t.join();
}

}  // namespace ni
