// Qwen3.8-Flash-Next's per-layer n-gram embedding (PLE): each token's hashed 2-gram and 3-gram ids, which depend on
// the token ids alone, and the row table (51 B parameters), mapped from its file and read on the CPU: it never
// enters GPU memory. A pass gathers its rows (16 per token, one per hash head) into a small GPU buffer.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ni {

// The hash of the reference implementation (transformers' Qwen4ExpTextNGramEmbedding): token t's n-gram of length
// m takes the tokens t, t-1, .., t-m+1, where a token before the sequence or across an end-of-sequence token is
// replaced by that token (n-grams never span documents); ids = (xor_j tok_j * mult_j) mod P_h + offset_h, with
// P_h the h-th prime after `base` and mult_j from splitmix64 of the seed.
struct NgramHash {
  int ngram = 3;           // longest n-gram (2-grams .. ngram-grams)
  int per_ngram = 8;       // hash heads per n-gram length
  int heads = 16;          // (ngram - 1) * per_ngram
  int eos = 0;
  int64_t mult[8] = {};
  std::vector<int64_t> mod, off;  // per head
  int64_t rows = 0;               // total rows (before padding)
  static NgramHash make(int vocab, int seed, int ple_index, int64_t base, int per_ngram, int ngram, int eos);
  // ids of position t of `tok` (heads values; t's history is tok[t - ngram + 1 .. t], positions < 0 count as eos)
  void ids(const int* tok, int t, int64_t* out) const;
};

// The row table: [rows, dim] in one of the formats below, possibly split into consecutive parts (files or tensors),
// read through mmap. Rows decode to fp32.
class NgramTable {
 public:
  enum class Fmt { BF16, F8E4M3, Affine, Gguf };
  struct Part {
    std::string file;         // safetensors file
    std::string w, s, b;      // tensor names: codes (or bf16 / fp8 values), scales, biases (Affine)
  };
  ~NgramTable();
  // fmt Affine: MLX affine `bits`-bit codes with bf16 scale and bias per group of `gs` (0: from the shapes);
  // F8E4M3: one fp32 scale for the whole table (`scale`); the parts' rows concatenate to the table
  void open(const std::vector<Part>& parts, Fmt fmt, int dim, int bits = 0, int gs = 0, float scale = 1.0f);
  // fmt Gguf: one tensor of a GGUF file, ne = {dim, rows}, in a block type gguf::dequantize reads whose blocks tile a
  // row (ISTA-DASLab's Flash-Next GGUFs: per_layer_token_embd.weight in IQ4_NL), read in place as llama.cpp decodes it
  void open_gguf(const std::string& file, const std::string& name, int dim);
  // whether a safetensors file's header lists the tensor
  static bool has_tensor(const std::string& file, const std::string& name);
  bool ok() const { return !parts_.empty(); }
  int dim() const { return dim_; }
  int64_t rows() const { return rows_; }
  size_t file_bytes() const { return mapped_; }
  std::string describe() const;
  // out[dim] = row r
  void row(int64_t r, float* out) const;
  // asks the kernel to start reading row r's pages (codes, scales, biases) if they are not resident
  void prefetch(int64_t r) const;
  // out[t, h * dim .. (h + 1) * dim) = row ids[t * heads + h], T rows (threads for large T)
  void gather(const int64_t* ids, int T, int heads, float* out) const;

 private:
  struct Map {
    const uint8_t* w = nullptr;
    const uint8_t* s = nullptr;
    const uint8_t* b = nullptr;
    int64_t rows = 0, first = 0;
  };
  const Map& part_of(int64_t r) const;
  std::pair<const uint8_t*, size_t> map_file(const std::string& path);  // a read-only shared mapping of the file
  std::vector<Map> parts_;
  std::vector<std::pair<void*, size_t>> maps_;
  Fmt fmt_ = Fmt::BF16;
  int dim_ = 0, bits_ = 0, gs_ = 0;
  uint32_t gtype_ = 0;  // Gguf: the ggml type
  float scale_ = 1.0f;
  int64_t rows_ = 0;
  size_t mapped_ = 0;
  size_t row_bytes_ = 0, group_row_bytes_ = 0;
};

}  // namespace ni
