// Flint's dense Llama (Nanbeige4.3) as a DFlash2 target (model/dflash2.h): its config in the Qwen35Config fields the
// drafter reads, its embedding and LM head as QLinear, and the captures of its layer outputs.
#pragma once
#include "model/dflash2.h"
#include "model/flint.h"
#include "util/log.h"

namespace ni {

// The fields of a dense Llama's config the drafter checks (hidden, layers, vocab), from its folder alone
inline Qwen35Config flint_dfl_config(const FlintConfig& c) {
  Qwen35Config q;
  q.hidden = c.hidden;
  q.layers = c.layers;
  q.vocab = c.vocab;
  q.eos = c.eos;
  return q;
}

class FlintDflTarget : public DflTarget {
 public:
  explicit FlintDflTarget(Flint& f) : f_(f) {
    if (!f.config().llama) fatal("DFlash2 on Flint: a dense Llama only (Nanbeige4.3)");
    cfg_ = flint_dfl_config(f.config());
    emb_ = ql(f.embed_mat());
    head_ = ql(f.head_mat());
  }
  const Qwen35Config& config() const override { return cfg_; }
  std::string model_dir() const override { return f_.model_dir(); }
  const QLinear& embed_table() const override { return emb_; }
  const QLinear& lm_head() const override { return head_; }
  void set_capture(const CaptureHook* h) override { f_.set_capture(h); }

 private:
  static QLinear ql(const FlintMat& m) {
    QLinear q;
    q.w = m.w, q.s = m.s, q.b = m.b, q.N = m.N, q.K = m.K, q.bits = m.bits, q.gs = m.gs;
    return q;
  }
  Flint& f_;
  Qwen35Config cfg_;
  QLinear emb_, head_;
};

}  // namespace ni
