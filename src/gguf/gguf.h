// GGUF files (llama.cpp's model format): the header of a file, or of every part of a split file (name-00001-of-0000N
// .gguf), and reads of tensor bytes. Little-endian GGUF v2 and v3. Nothing is mapped or read beyond the header until
// read() is called, so a 20 GB file costs only its header in memory.
#pragma once
#include <nlohmann/json.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ni::gguf {

// A file that cannot be used, with the reason in one or two sentences.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ggml tensor types (the numbering of ggml.h's ggml_type, which GGUF stores)
enum : uint32_t {
  kF32 = 0, kF16 = 1, kQ4_0 = 2, kQ4_1 = 3, kQ5_0 = 6, kQ5_1 = 7, kQ8_0 = 8, kQ8_1 = 9, kQ2_K = 10, kQ3_K = 11,
  kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kQ8_K = 15, kIQ2_XXS = 16, kIQ2_XS = 17, kIQ3_XXS = 18, kIQ1_S = 19,
  kIQ4_NL = 20, kIQ3_S = 21, kIQ2_S = 22, kIQ4_XS = 23, kI8 = 24, kI16 = 25, kI32 = 26, kI64 = 27, kF64 = 28,
  kIQ1_M = 29, kBF16 = 30, kTQ1_0 = 34, kTQ2_0 = 35, kMXFP4 = 39, kNVFP4 = 40,
};

std::string type_name(uint32_t type);  // "Q4_K"; "type 77" for one this code does not know
// Elements and bytes per block of a type; false when the type is unknown here.
bool type_block(uint32_t type, uint32_t* elements, uint32_t* bytes);

struct Tensor {
  std::string name;
  std::vector<uint64_t> ne;  // ggml order: ne[0] is the length of a row (the innermost dimension)
  uint32_t type = 0;
  int part = 0;              // the file of a split model it is in
  uint64_t offset = 0;       // absolute byte offset in that file
  uint64_t bytes = 0;
  uint64_t elements() const;
  uint64_t rows() const { return ne.empty() || ne[0] == 0 ? 0 : elements() / ne[0]; }
  uint64_t row_bytes() const { return rows() ? bytes / rows() : 0; }
};

class File {
 public:
  // Opens a GGUF file; a part of a split file opens every part (they must all be there). Throws Error.
  explicit File(const std::string& path);
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  // Metadata: keys as stored, values as JSON (integers, floats, booleans, strings, arrays).
  const nlohmann::json& meta() const { return meta_; }
  bool has(const std::string& key) const { return meta_.contains(key); }
  std::string str(const std::string& key, const std::string& def = "") const;
  int64_t num(const std::string& key, int64_t def) const;  // integer metadata (also accepts a bool)
  double real(const std::string& key, double def) const;    // float or integer metadata
  bool flag(const std::string& key, bool def) const;

  const std::vector<Tensor>& tensors() const { return tensors_; }
  const Tensor* find(const std::string& name) const;
  // Bytes [off, off + n) of a tensor's data.
  void read(const Tensor& t, uint64_t off, void* dst, size_t n) const;

  const std::vector<std::string>& parts() const { return paths_; }  // the files, in order
  uint64_t file_bytes() const { return file_bytes_; }                // all parts together
  uint32_t version() const { return version_; }

 private:
  void parse(int part, bool first);
  std::vector<std::string> paths_;
  std::vector<int> fds_;
  nlohmann::json meta_ = nlohmann::json::object();
  std::vector<Tensor> tensors_;
  std::unordered_map<std::string, size_t> index_;
  uint64_t file_bytes_ = 0;
  uint32_t version_ = 0;
};

// The parts of a split file in order, given any of them ("x-00002-of-00003.gguf" -> the three); {path} otherwise.
std::vector<std::string> split_parts(const std::string& path);

}  // namespace ni::gguf
