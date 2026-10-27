// Safetensors loading into aligned GPU buffers.
// Tensors in safetensors files sit at arbitrary byte offsets (HF/MLX writers do not pad the
// header), so they cannot be mapped onto the GPU directly: vector loads need 16-byte alignment.
// We read each kept tensor with parallel uncached preads into one shared MTLBuffer per file,
// every tensor 256-byte aligned. Tensors rejected by the filter (e.g. the vision tower) are skipped.
#pragma once
#include "metal/gpu.h"

#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace ni {

enum class DType { F32, F16, BF16, U32, I32, U8, I8, I64, F8E4M3, U16, Unknown };
size_t dtype_size(DType d);

struct Tensor {
  Buffer buf;
  DType dtype = DType::Unknown;
  std::vector<int64_t> shape;
  int64_t numel() const {
    int64_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
  explicit operator bool() const { return bool(buf); }
};

class WeightStore {
 public:
  using Filter = std::function<bool(const std::string&)>;
  // Loads every *.safetensors in `dir` (follows HF snapshot symlinks).
  // `files` limits which *.safetensors are read (by file name); `keep` filters tensors. Tensors `defer` accepts are
  // not read: they are listed with dtype and shape but no buffer, and read_deferred() fetches their bytes later (a
  // model converting them at load then never holds both forms).
  void open_dir(const std::string& dir, const Filter& keep = nullptr, const Filter& files = nullptr,
                const Filter& defer = nullptr);
  bool deferred(const std::string& name) const { return deferred_.count(name) != 0; }
  // the deferred tensor's bytes into dst (its byte size: dtype size x numel)
  void read_deferred(const std::string& name, void* dst) const;
  // where a deferred tensor's bytes are: its file, absolute offset and byte count (false if not deferred)
  bool deferred_place(const std::string& name, std::string* path, size_t* off, size_t* len) const;
  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const Tensor& get(const std::string& name) const;
  const Tensor* find(const std::string& name) const;
  const std::map<std::string, Tensor>& all() const { return tensors_; }
  std::vector<Buffer> buffers() const { return files_; }
  size_t total_bytes() const { return total_; }
  ~WeightStore();

 private:
  struct Place {
    std::string path;
    size_t off = 0, len = 0;
  };
  std::map<std::string, Place> deferred_;
  std::map<std::string, Tensor> tensors_;
  std::vector<Buffer> files_;
  size_t total_ = 0;
};

// The tensor directory of a folder of safetensors files (headers only), for loaders that place and convert tensors
// themselves: any byte range of a tensor can be read from any thread (uncached preads).
class SafetensorsIndex {
 public:
  struct Entry {
    int file = -1;
    size_t off = 0, len = 0;  // absolute file offset, bytes
    DType dtype = DType::Unknown;
    std::vector<int64_t> shape;
    int64_t numel() const {
      int64_t n = 1;
      for (auto v : shape) n *= v;
      return n;
    }
  };
  // every *.safetensors in dir (follows symlinks); `skip` leaves tensors out of the directory (a vision tower)
  explicit SafetensorsIndex(const std::string& dir, const std::function<bool(const std::string&)>& skip = nullptr);
  ~SafetensorsIndex();
  SafetensorsIndex(const SafetensorsIndex&) = delete;
  SafetensorsIndex& operator=(const SafetensorsIndex&) = delete;
  const Entry* find(const std::string& name) const;
  const Entry& get(const std::string& name) const;  // fatal when missing
  // bytes [off, off + n) of the tensor into dst; fatal on a read error
  void read(const Entry& t, size_t off, void* dst, size_t n) const;
  const std::unordered_map<std::string, Entry>& all() const { return t_; }
  const std::string& dir() const { return dir_; }
  const std::string& path(const Entry& t) const { return paths_[size_t(t.file)]; }

 private:
  std::string dir_;
  std::vector<int> fds_;
  std::vector<std::string> paths_;
  std::unordered_map<std::string, Entry> t_;
};

} // namespace ni
