#pragma once

#include "metal/gpu.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

// Row slices for the dense tensor_ops prefill GEMMs (qmt / qmtg) of Drex Max v1's noncausal 64-layer, 5120-wide,
// 17408-FFN shape: a GEMM of more than `cap` rows runs as dispatches of at most `cap` rows over views of the same
// input and output buffers. Each row's products and K order are unchanged, so the outputs are bitwise the same; what
// changes is how many rows' tiles one dispatch puts in flight against the same weights. Measured on the M5 Pro
// (#308): a Drex Max request about 12% faster at 4K and 16K, logits bitwise identical. NI_GEMM_SLICES=0 turns it off.
namespace ni::gemm_slices {

inline constexpr size_t cap = 2048;

// enabled: NI_GEMM_SLICES (default on). The shape check keeps it to the model it was measured on.
inline bool eligible(bool enabled, bool causal, bool moe, int layers, int hidden, int dense_I, int rows) {
  return enabled && !causal && !moe && layers == 64 && hidden == 5120 && dense_I == 17408 && rows > int(cap);
}

class Plan {
 public:
  struct Slice {
    size_t row_start, rows;
    Buffer input, output;
  };

  Plan(size_t rows, size_t K, size_t N, const Buffer& input, const Buffer& output, size_t output_element_bytes)
      : rows_(rows), input_(input), output_(output) {
    if (!K || !N || (output_element_bytes != 2 && output_element_bytes != 4))
      throw std::invalid_argument("invalid GEMM slice dimensions or output type");
    input_stride_ = multiply(K, 2);
    output_stride_ = multiply(N, output_element_bytes);
    if (input_stride_ % 16) throw std::invalid_argument("GEMM slice input row alignment");
    validate(input_, multiply(rows_, input_stride_), 16);
    validate(output_, multiply(rows_, output_stride_), output_element_bytes);
  }

  size_t count() const { return rows_ / cap + (rows_ % cap != 0); }

  Slice slice(size_t index) const {
    if (index >= count()) throw std::out_of_range("GEMM slice index");
    const size_t start = index * cap, rows = std::min(cap, rows_ - start);
    // Construction proves all subranges fit, including Buffer::view's inherited offset addition.
    return {start, rows, input_.view(start * input_stride_, rows * input_stride_),
            output_.view(start * output_stride_, rows * output_stride_)};
  }

 private:
  static size_t multiply(size_t a, size_t b) {
    if (b && a > std::numeric_limits<size_t>::max() / b) throw std::overflow_error("GEMM slice byte size");
    return a * b;
  }

  static void validate(const Buffer& buffer, size_t active_bytes, size_t alignment) {
    if (buffer.offset > std::numeric_limits<size_t>::max() - buffer.size)
      throw std::overflow_error("GEMM slice parent offset");
    if (buffer.offset % alignment || reinterpret_cast<uintptr_t>(buffer.base) % alignment)
      throw std::invalid_argument("GEMM slice element alignment");
    if (active_bytes > buffer.size) throw std::out_of_range("GEMM slice active span exceeds buffer");
  }

  size_t rows_, input_stride_ = 0, output_stride_ = 0;
  Buffer input_, output_;
};

} // namespace ni::gemm_slices
