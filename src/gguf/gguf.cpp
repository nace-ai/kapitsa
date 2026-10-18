// GGUF header parsing (see gguf.h). The format: "GGUF", version, tensor count, metadata count, then the metadata
// (key, type, value), the tensor table (name, dims, type, offset), padding to the alignment, and the tensor data.
#include "gguf/gguf.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <regex>

namespace ni::gguf {

using json = nlohmann::json;

std::string type_name(uint32_t t) {
  static const char* names[] = {"F32",  "F16",    "Q4_0",  "Q4_1",   "type 4", "type 5", "Q5_0",   "Q5_1",
                                "Q8_0", "Q8_1",   "Q2_K",  "Q3_K",   "Q4_K",   "Q5_K",   "Q6_K",   "Q8_K",
                                "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS",
                                "I8",   "I16",    "I32",   "I64",    "F64",    "IQ1_M",  "BF16",   "type 31",
                                "type 32", "type 33", "TQ1_0", "TQ2_0", "type 36", "type 37", "type 38", "MXFP4",
                                "NVFP4"};
  if (t < sizeof names / sizeof names[0]) return names[t];
  return "type " + std::to_string(t);
}

bool type_block(uint32_t t, uint32_t* elements, uint32_t* bytes) {
  uint32_t e = 0, b = 0;
  switch (t) {
    case kF32: e = 1, b = 4; break;
    case kF16: case kBF16: e = 1, b = 2; break;
    case kF64: e = 1, b = 8; break;
    case kI8: e = 1, b = 1; break;
    case kI16: e = 1, b = 2; break;
    case kI32: e = 1, b = 4; break;
    case kI64: e = 1, b = 8; break;
    case kQ4_0: e = 32, b = 18; break;
    case kQ4_1: e = 32, b = 20; break;
    case kQ5_0: e = 32, b = 22; break;
    case kQ5_1: e = 32, b = 24; break;
    case kQ8_0: e = 32, b = 34; break;
    case kQ8_1: e = 32, b = 36; break;
    case kQ2_K: e = 256, b = 84; break;
    case kQ3_K: e = 256, b = 110; break;
    case kQ4_K: e = 256, b = 144; break;
    case kQ5_K: e = 256, b = 176; break;
    case kQ6_K: e = 256, b = 210; break;
    case kQ8_K: e = 256, b = 292; break;
    case kIQ2_XXS: e = 256, b = 66; break;
    case kIQ2_XS: e = 256, b = 74; break;
    case kIQ2_S: e = 256, b = 82; break;
    case kIQ3_XXS: e = 256, b = 98; break;
    case kIQ3_S: e = 256, b = 110; break;
    case kIQ1_S: e = 256, b = 50; break;
    case kIQ1_M: e = 256, b = 56; break;
    case kIQ4_NL: e = 32, b = 18; break;
    case kIQ4_XS: e = 256, b = 136; break;
    case kTQ1_0: e = 256, b = 54; break;
    case kTQ2_0: e = 256, b = 66; break;
    case kMXFP4: e = 32, b = 17; break;
    case kNVFP4: e = 64, b = 36; break;
    default: return false;
  }
  if (elements) *elements = e;
  if (bytes) *bytes = b;
  return true;
}

uint64_t Tensor::elements() const {
  uint64_t n = 1;
  for (uint64_t d : ne) n *= d;
  return n;
}

std::vector<std::string> split_parts(const std::string& path) {
  static const std::regex re(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)", std::regex::icase);
  std::smatch m;
  if (!std::regex_match(path, m, re)) return {path};
  const int n = std::stoi(m[3].str());
  if (n < 1 || n > 9999) return {path};
  std::vector<std::string> v;
  for (int i = 1; i <= n; ++i) {
    char b[32];
    snprintf(b, sizeof b, "-%05d-of-%05d", i, n);
    v.push_back(m[1].str() + b + path.substr(path.size() - 5));  // keeps the extension's case
  }
  return v;
}

namespace {

// Sequential reads of a file's header through a buffer, with limits against corrupt or hostile files.
class Reader {
 public:
  Reader(int fd, uint64_t size, const std::string& path) : fd_(fd), size_(size), path_(path) {}
  uint64_t pos() const { return pos_; }
  void raw(void* dst, size_t n) {
    uint8_t* d = static_cast<uint8_t*>(dst);
    while (n) {
      if (at_ == len_) fill(n);
      const size_t k = std::min(n, len_ - at_);
      memcpy(d, buf_.data() + at_, k);
      d += k, at_ += k, pos_ += k, n -= k;
    }
  }
  template <class T>
  T get() {
    T v;
    raw(&v, sizeof v);
    return v;
  }
  std::string str() {
    const uint64_t n = get<uint64_t>();
    if (n > (64ull << 20)) bad("a metadata string longer than 64 MB");
    std::string s(n, '\0');
    if (n) raw(s.data(), n);
    return s;
  }
  [[noreturn]] void bad(const std::string& what) const {
    throw Error(path_ + " is not a valid GGUF file (" + what + " at byte " + std::to_string(pos_) + ").");
  }

 private:
  void fill(size_t) {
    if (pos_ >= size_) bad("unexpected end of file");
    const size_t n = 4u << 20;
    buf_.resize(n);
    const ssize_t r = pread(fd_, buf_.data(), n, off_t(pos_));
    if (r <= 0) bad("unexpected end of file");
    at_ = 0, len_ = size_t(r);
  }
  int fd_;
  uint64_t size_, pos_ = 0;
  std::string path_;
  std::vector<uint8_t> buf_;
  size_t at_ = 0, len_ = 0;
};

enum : uint32_t { kU8 = 0, kI8v, kU16, kI16v, kU32, kI32v, kF32v, kBool, kStr, kArr, kU64, kI64v, kF64v };

json read_value(Reader& r, uint32_t type, int depth) {
  switch (type) {
    case kU8: return r.get<uint8_t>();
    case kI8v: return r.get<int8_t>();
    case kU16: return r.get<uint16_t>();
    case kI16v: return r.get<int16_t>();
    case kU32: return r.get<uint32_t>();
    case kI32v: return r.get<int32_t>();
    case kF32v: return r.get<float>();
    case kBool: return r.get<uint8_t>() != 0;
    case kStr: return r.str();
    case kU64: return r.get<uint64_t>();
    case kI64v: return r.get<int64_t>();
    case kF64v: return r.get<double>();
    case kArr: {
      if (depth > 2) r.bad("nested arrays");
      const uint32_t et = r.get<uint32_t>();
      const uint64_t n = r.get<uint64_t>();
      if (n > (16ull << 20)) r.bad("an array of more than 16M items");
      json a = json::array();
      a.get_ref<json::array_t&>().reserve(size_t(n));
      for (uint64_t i = 0; i < n; ++i) a.push_back(read_value(r, et, depth + 1));
      return a;
    }
    default: r.bad("unknown metadata type " + std::to_string(type));
  }
}

}  // namespace

File::File(const std::string& path) {
  paths_ = split_parts(path);
  for (const auto& p : paths_) {
    const int fd = open(p.c_str(), O_RDONLY);
    if (fd < 0) {
      for (int f : fds_) close(f);
      fds_.clear();
      if (paths_.size() > 1 && p != path)
        throw Error("Can't open " + p + ", part of the split file " + path + " (" + strerror(errno) +
                    "). All parts must be in the same folder.");
      throw Error("Can't open " + p + ": " + strerror(errno) + ".");
    }
    fcntl(fd, F_NOCACHE, 1);  // tensors are read once, in large pieces: no second copy in the page cache
    fds_.push_back(fd);
  }
  try {
    for (size_t i = 0; i < paths_.size(); ++i) parse(int(i), i == 0);
    for (size_t i = 0; i < tensors_.size(); ++i)
      if (!index_.emplace(tensors_[i].name, i).second) throw Error(path + " lists tensor " + tensors_[i].name + " twice.");
  } catch (...) {
    for (int f : fds_) close(f);
    fds_.clear();
    throw;
  }
  if (paths_.size() > 1) {
    const int64_t n = num("split.count", int64_t(paths_.size()));
    if (n != int64_t(paths_.size())) throw Error(path + ": the split count in its header (" + std::to_string(n) +
                                                 ") differs from its file name.");
    const int64_t nt = num("split.tensors.count", int64_t(tensors_.size()));
    if (nt != int64_t(tensors_.size()))
      throw Error(path + ": the parts hold " + std::to_string(tensors_.size()) + " tensors, the header announces " +
                  std::to_string(nt) + ". Is a part incomplete?");
  }
}

File::~File() {
  for (int fd : fds_) close(fd);
}

void File::parse(int part, bool first) {
  const std::string& path = paths_[size_t(part)];
  struct stat st;
  if (fstat(fds_[size_t(part)], &st) != 0) throw Error("Can't read " + path + ".");
  const uint64_t size = uint64_t(st.st_size);
  file_bytes_ += size;
  Reader r(fds_[size_t(part)], size, path);
  char magic[4];
  r.raw(magic, 4);
  if (memcmp(magic, "GGUF", 4) != 0) throw Error(path + " is not a GGUF file.");
  const uint32_t version = r.get<uint32_t>();
  if (version == 0x03000000 || version == 0x02000000)
    throw Error(path + " is a big-endian GGUF file; only little-endian files run on a Mac.");
  if (version != 2 && version != 3)
    throw Error(path + " is GGUF version " + std::to_string(version) + "; Kapitsa reads versions 2 and 3.");
  if (first) version_ = version;
  const uint64_t nt = r.get<uint64_t>(), nk = r.get<uint64_t>();
  if (nt > (1u << 20) || nk > (1u << 20)) r.bad("too many tensors or metadata keys");
  json kv = json::object();
  for (uint64_t i = 0; i < nk; ++i) {
    std::string key = r.str();
    const uint32_t type = r.get<uint32_t>();
    kv[key] = read_value(r, type, 0);
  }
  uint64_t align = 32;
  if (kv.contains("general.alignment") && kv["general.alignment"].is_number_unsigned()) align = kv["general.alignment"];
  if (align == 0 || align > (1u << 20) || (align & (align - 1))) r.bad("an invalid alignment");
  struct Pending {
    Tensor t;
    uint64_t rel;
  };
  std::vector<Pending> ts;
  ts.reserve(size_t(nt));
  for (uint64_t i = 0; i < nt; ++i) {
    Pending p;
    p.t.name = r.str();
    const uint32_t nd = r.get<uint32_t>();
    if (nd == 0 || nd > 4) r.bad("tensor " + p.t.name + " with " + std::to_string(nd) + " dimensions");
    for (uint32_t d = 0; d < nd; ++d) p.t.ne.push_back(r.get<uint64_t>());
    p.t.type = r.get<uint32_t>();
    p.rel = r.get<uint64_t>();
    p.t.part = part;
    ts.push_back(std::move(p));
  }
  const uint64_t data0 = (r.pos() + align - 1) / align * align;
  for (auto& p : ts) {
    Tensor& t = p.t;
    uint32_t be = 0, bb = 0;
    uint64_t n = 1;
    for (uint64_t d : t.ne) {
      if (d == 0 || d > (1ull << 40) || n > (1ull << 42) / d) r.bad("tensor " + t.name + " with invalid dimensions");
      n *= d;
    }
    if (type_block(t.type, &be, &bb)) {
      if (t.ne[0] % be) r.bad("tensor " + t.name + " whose rows are not whole blocks");
      t.bytes = n / be * bb;
    } else {
      t.bytes = 0;  // unknown type: sized as nothing; refused by whoever reads it
    }
    t.offset = data0 + p.rel;
    if (t.offset < data0 || t.offset + t.bytes > size)
      throw Error(path + ": tensor " + t.name + " lies beyond the end of the file. Is the download complete?");
    tensors_.push_back(std::move(t));
  }
  if (first) {
    meta_ = std::move(kv);
  } else {  // a later part: only its split keys matter
    for (auto& [k, v] : kv.items())
      if (!meta_.contains(k)) meta_[k] = v;
  }
}

std::string File::str(const std::string& key, const std::string& def) const {
  auto it = meta_.find(key);
  return it != meta_.end() && it->is_string() ? it->get<std::string>() : def;
}

int64_t File::num(const std::string& key, int64_t def) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) return def;
  if (it->is_number_integer()) return it->get<int64_t>();
  if (it->is_boolean()) return it->get<bool>() ? 1 : 0;
  return def;
}

double File::real(const std::string& key, double def) const {
  auto it = meta_.find(key);
  return it != meta_.end() && it->is_number() ? it->get<double>() : def;
}

bool File::flag(const std::string& key, bool def) const {
  auto it = meta_.find(key);
  if (it == meta_.end()) return def;
  if (it->is_boolean()) return it->get<bool>();
  if (it->is_number_integer()) return it->get<int64_t>() != 0;
  return def;
}

const Tensor* File::find(const std::string& name) const {
  auto it = index_.find(name);
  return it == index_.end() ? nullptr : &tensors_[it->second];
}

void File::read(const Tensor& t, uint64_t off, void* dst, size_t n) const {
  if (off + n > t.bytes) throw Error("internal: read past the end of tensor " + t.name);
  uint8_t* d = static_cast<uint8_t*>(dst);
  size_t done = 0;
  while (done < n) {
    const ssize_t r = pread(fds_[size_t(t.part)], d + done, n - done, off_t(t.offset + off + done));
    if (r <= 0) throw Error("Can't read " + paths_[size_t(t.part)] + " (" + (r < 0 ? strerror(errno) : "end of file") + ").");
    done += size_t(r);
  }
}

}  // namespace ni::gguf
