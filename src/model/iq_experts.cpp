#include "model/iq_experts.h"

#include "util/log.h"

#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace ni {

WeightLock::WeightLock() : on_(!(getenv("NI_MLOCK") && !strcmp(getenv("NI_MLOCK"), "0"))) {}

void WeightLock::wire(const Buffer& b) {
  if (!on_) return;
  if (mlock(b.ptr(), b.size)) {
    logi("could not lock weights in memory (%s) past %.2f GB: the rest stays pageable", strerror(errno), locked_ / 1e9);
    on_ = false;
    return;
  }
  locked_ += b.size;
}

Buffer iq_read_matrix(const WeightStore& ws, const std::string& name, uint32_t type, size_t rows, size_t K,
                      gguf::IqLayout* lay, std::vector<uint8_t>& raw) {
  const size_t rb = K / 256 * gguf::iq_block_bytes(type);
  *lay = gguf::iq_layout(type, rows, K);
  raw.resize(rows * rb);
  ws.read_deferred(name, raw.data());
  Buffer b = Gpu::get().alloc(lay->bytes, "iq_experts", /*tracked=*/false);
  const unsigned nth = std::max(2u, std::min(8u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  const size_t per = (rows + nth - 1) / nth;
  for (unsigned i = 0; i < nth; ++i) {
    const size_t r0 = std::min(rows, i * per), r1 = std::min(rows, r0 + per);
    if (r0 < r1) th.emplace_back([&, r0, r1] { gguf::iq_repack(type, raw.data() + r0 * rb, rows, K, r0, r1, b.as<uint8_t>()); });
  }
  for (auto& t : th) t.join();
  return b;
}

}  // namespace ni
