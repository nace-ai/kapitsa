#include "model/model.h"

#include <algorithm>

namespace ni {

void enc_logit_mask(Encoder& e, Seq& s, const Buffer& logits, int rows, int vocab, int ld, int mask_row0) {
  if (s.mask_rows <= 0 || !s.mask) return;
  rows = std::min(rows, s.mask_rows - mask_row0);
  if (rows <= 0) return;
  if (s.mask_event) e.wait_event(s.mask_event, s.mask_value);
  static const Pipeline pso = Gpu::get().pipeline("logit_mask_rows");
  const struct {
    uint32_t V, W, ld, mrow0;
  } p{uint32_t(vocab), uint32_t(s.mask_words), uint32_t(ld), uint32_t(mask_row0)};
  e.use(pso);
  e.buf(0, logits);
  e.buf(1, s.mask);
  e.val(2, p);
  e.dispatch((uint32_t(s.mask_words) + 255) / 256, uint32_t(rows), 1, 256);
}

}  // namespace ni
