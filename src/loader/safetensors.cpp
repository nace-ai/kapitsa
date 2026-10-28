#include "util/load_threads.h"
#include "loader/safetensors.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

namespace ni {

size_t dtype_size(DType d) {
  switch (d) {
    case DType::F32: case DType::U32: case DType::I32: return 4;
    case DType::F16: case DType::BF16: case DType::U16: return 2;
    case DType::U8: case DType::I8: case DType::F8E4M3: return 1;
    case DType::I64: return 8;
    default: return 0;
  }
}

// The message for a file cut short or not a safetensors file at all
[[noreturn]] static void incomplete(const std::string& path, const char* why, long long size) {
  fatal("%s is incomplete or not a safetensors file (%s, %lld bytes): copy or download it again", path.c_str(), why,
        size);
}

// A file's header, checked: a copy cut short or a placeholder (zeros) stops with the file named rather than a JSON
// exception. *size: the file's bytes (a file of headers alone serves the plans; its tensors are checked when read).
static nlohmann::json read_header(int fd, const std::string& path, uint64_t& hlen, uint64_t* size) {
  struct stat st {};
  if (fstat(fd, &st) != 0) fatal("stat %s failed", path.c_str());
  *size = uint64_t(st.st_size);
  hlen = 0;
  if (pread(fd, &hlen, 8, 0) != 8) incomplete(path, "shorter than a header", st.st_size);
  if (hlen < 2 || hlen > (uint64_t(1) << 30) || 8 + hlen > uint64_t(st.st_size))
    incomplete(path, "no header", st.st_size);
  std::string hdr(hlen, '\0');
  if (pread(fd, hdr.data(), hlen, 8) != ssize_t(hlen)) fatal("read %s header failed", path.c_str());
  nlohmann::json j = nlohmann::json::parse(hdr, nullptr, false);
  if (!j.is_object()) incomplete(path, "unreadable header", st.st_size);
  for (auto& [name, v] : j.items()) {
    if (name == "__metadata__") continue;
    const auto& o = v.is_object() ? v.value("data_offsets", nlohmann::json()) : nlohmann::json();
    if (!o.is_array() || o.size() != 2 || !o[0].is_number_unsigned() || !o[1].is_number_unsigned() ||
        o[1].get<uint64_t>() < o[0].get<uint64_t>() || !v.contains("dtype") || !v["dtype"].is_string() ||
        !v.contains("shape") || !v["shape"].is_array())
      incomplete(path, "bad tensor entry", st.st_size);
  }
  return j;
}

static DType parse_dtype(const std::string& s) {
  if (s == "F32") return DType::F32;
  if (s == "F16") return DType::F16;
  if (s == "BF16") return DType::BF16;
  if (s == "U32") return DType::U32;
  if (s == "I32") return DType::I32;
  if (s == "U8") return DType::U8;
  if (s == "U16") return DType::U16;
  if (s == "I8") return DType::I8;
  if (s == "I64") return DType::I64;
  if (s == "F8_E4M3") return DType::F8E4M3;
  return DType::Unknown;
}

namespace {
struct ReadJob {
  int fd;
  size_t file_off;
  uint8_t* dst;
  size_t len;
};
} // namespace

void WeightStore::open_dir(const std::string& dir, const Filter& keep, const Filter& file_filter, const Filter& defer) {
  std::unique_ptr<DIR, decltype(&closedir)> d(opendir(dir.c_str()), closedir);
  if (!d) fatal("cannot open model dir %s", dir.c_str());
  std::vector<std::string> files;
  while (dirent* e = readdir(d.get())) {
    std::string n = e->d_name;
    if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors" && (!file_filter || file_filter(n)))
      files.push_back(dir + "/" + n);
  }
  d.reset();
  std::sort(files.begin(), files.end());
  if (files.empty() && !file_filter) fatal("no .safetensors in %s", dir.c_str());

  constexpr size_t kAlign = 256, kChunk = 32u << 20;
  std::vector<ReadJob> jobs;
  std::vector<Fd> fds;
  for (auto& path : files) {
    Fd opened(open(path.c_str(), O_RDONLY));
    int fd = opened.get();
    if (fd < 0) fatal("open %s failed", path.c_str());
    fcntl(fd, F_NOCACHE, 1); // stream once; don't keep a second copy in the page cache
    uint64_t hlen = 0, fsize = 0;
    const auto j = read_header(fd, path, hlen, &fsize);
    size_t data0 = 8 + hlen;

    struct Item {
      std::string name;
      Tensor t;
      size_t src, len, dst;
    };
    std::vector<Item> items;
    size_t total = 0;
    for (auto& [name, v] : j.items()) {
      if (name == "__metadata__") continue;
      if (keep && !keep(name)) continue;
      Item it;
      it.name = name;
      it.t.dtype = parse_dtype(v["dtype"].get<std::string>());
      for (auto& s : v["shape"]) it.t.shape.push_back(s.get<int64_t>());
      size_t b = v["data_offsets"][0].get<size_t>(), e = v["data_offsets"][1].get<size_t>();
      it.src = data0 + b;
      it.len = e - b;
      if (defer && defer(name)) {  // listed without a buffer; read_deferred() reads it
        deferred_[name] = Place{path, it.src, it.len};
        tensors_[name] = it.t;
        continue;
      }
      it.dst = total;
      total += (it.len + kAlign - 1) / kAlign * kAlign;
      items.push_back(std::move(it));
    }
    if (items.empty()) {
      continue;
    }
    for (const Item& it : items)
      if (it.src + it.len > fsize) incomplete(path, "its tensors run past its end", (long long)fsize);
    // Read in file order for sequential I/O.
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.src < b.src; });
    std::string base = path.substr(path.find_last_of('/') + 1);
    Buffer fb = Gpu::get().alloc(total, base.c_str(), /*tracked=*/false);
    files_.push_back(fb);
    total_ += total;
    fds.push_back(std::move(opened));
    for (auto& it : items) {
      it.t.buf = fb.view(it.dst, it.len);
      for (size_t o = 0; o < it.len; o += kChunk)
        jobs.push_back({fd, it.src + o, fb.as<uint8_t>() + it.dst + o, std::min(kChunk, it.len - o)});
      tensors_[it.name] = it.t;
    }
  }

  double t0 = now_ms();
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  unsigned nth = std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
  LoadThreads th;
  for (unsigned i = 0; i < nth; ++i)
    th.emplace_back([&] {
      for (size_t k; (k = next.fetch_add(1)) < jobs.size();) {
        const ReadJob& jb = jobs[k];
        size_t done = 0;
        while (done < jb.len) {
          ssize_t r = pread(jb.fd, jb.dst + done, jb.len - done, off_t(jb.file_off + done));
          if (r <= 0) {
            failed = true;
            return;
          }
          done += size_t(r);
        }
      }
    });
  th.join();
  fds.clear();
  if (failed) fatal("reading weights from %s failed", dir.c_str());
  double ms = now_ms() - t0;
  logi("read %.2f GB of weights in %.0f ms (%.1f GB/s)", total_ / 1e9, ms, total_ / 1e6 / ms);
}

void WeightStore::read_deferred(const std::string& name, void* dst) const {
  auto it = deferred_.find(name);
  if (it == deferred_.end()) fatal("%s is not a deferred tensor", name.c_str());
  const Place& p = it->second;
  Fd opened(open(p.path.c_str(), O_RDONLY));
  int fd = opened.get();
  if (fd < 0) fatal("open %s failed", p.path.c_str());
  fcntl(fd, F_NOCACHE, 1);
  // parallel reads of 32 MB pieces
  constexpr size_t kChunk = 32u << 20;
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  const size_t n = (p.len + kChunk - 1) / kChunk;
  const unsigned nth = std::min<unsigned>(8, std::max<size_t>(1, n));
  LoadThreads th;
  for (unsigned i = 0; i < nth; ++i)
    th.emplace_back([&] {
      for (size_t k; (k = next.fetch_add(1)) < n;) {
        const size_t o = k * kChunk, len = std::min(kChunk, p.len - o);
        size_t done = 0;
        while (done < len) {
          ssize_t r = pread(fd, static_cast<uint8_t*>(dst) + o + done, len - done, off_t(p.off + o + done));
          if (r <= 0) {
            failed = true;
            return;
          }
          done += size_t(r);
        }
      }
    });
  th.join();
  if (failed) fatal("reading %s failed", name.c_str());
}

bool WeightStore::deferred_place(const std::string& name, std::string* path, size_t* off, size_t* len) const {
  auto it = deferred_.find(name);
  if (it == deferred_.end()) return false;
  if (path) *path = it->second.path;
  if (off) *off = it->second.off;
  if (len) *len = it->second.len;
  return true;
}

const Tensor* WeightStore::find(const std::string& name) const {
  auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& WeightStore::get(const std::string& name) const {
  auto* t = find(name);
  if (!t) fatal("missing tensor %s", name.c_str());
  return *t;
}

SafetensorsIndex::SafetensorsIndex(const std::string& dir, const std::function<bool(const std::string&)>& skip)
    : dir_(dir) {
  std::unique_ptr<DIR, decltype(&closedir)> d(opendir(dir.c_str()), closedir);
  if (!d) fatal("cannot open model dir %s", dir.c_str());
  while (dirent* e = readdir(d.get())) {
    std::string n = e->d_name;
    if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors") paths_.push_back(dir + "/" + n);
  }
  d.reset();
  std::sort(paths_.begin(), paths_.end());
  if (paths_.empty()) fatal("no .safetensors in %s", dir.c_str());
  for (size_t fi = 0; fi < paths_.size(); ++fi) {
    const std::string& path = paths_[fi];
    Fd opened(open(path.c_str(), O_RDONLY));
    const int fd = opened.get();
    if (fd < 0) fatal("open %s failed", path.c_str());
    fcntl(fd, F_NOCACHE, 1);
    fds_.push_back(std::move(opened));
    uint64_t hlen = 0, fsize = 0;
    const auto j = read_header(fd, path, hlen, &fsize);
    for (auto& [name, v] : j.items()) {
      if (name == "__metadata__" || (skip && skip(name))) continue;
      Entry t;
      t.file = int(fi);
      t.dtype = parse_dtype(v["dtype"].get<std::string>());
      for (auto& x : v["shape"]) t.shape.push_back(x.get<int64_t>());
      const size_t b = v["data_offsets"][0].get<size_t>(), e = v["data_offsets"][1].get<size_t>();
      t.off = 8 + hlen + b;
      t.len = e - b;
      t_[name] = std::move(t);
    }
  }
}

SafetensorsIndex::~SafetensorsIndex() = default;

const SafetensorsIndex::Entry* SafetensorsIndex::find(const std::string& name) const {
  auto it = t_.find(name);
  return it == t_.end() ? nullptr : &it->second;
}

const SafetensorsIndex::Entry& SafetensorsIndex::get(const std::string& name) const {
  const Entry* t = find(name);
  if (!t) fatal("%s: tensor %s missing", dir_.c_str(), name.c_str());
  return *t;
}

void SafetensorsIndex::read(const Entry& t, size_t off, void* dst, size_t n) const {
  if (off + n > t.len) fatal("%s: read past a tensor's end", dir_.c_str());
  size_t done = 0;
  while (done < n) {
    const ssize_t r = pread(fds_[size_t(t.file)].get(), static_cast<uint8_t*>(dst) + done, n - done, off_t(t.off + off + done));
    if (r <= 0) fatal("reading %s failed", paths_[size_t(t.file)].c_str());
    done += size_t(r);
  }
}

WeightStore::~WeightStore() {
  for (auto& b : files_) Gpu::get().release(b);
}

} // namespace ni
