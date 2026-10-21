#include "model/nh_drafter.h"
#include "model/kflash_budget.h"
#include "model/dspark.h"
#include "model/lilicorr.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>

namespace ni {

namespace {

// the language model's hidden size: llm_config (Nemotron Omni releases), text_config, or the whole config.json
int target_hidden(const std::string& model_dir) {
  std::ifstream f(model_dir + "/config.json");
  if (!f) return 0;
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(f);
  } catch (...) {
    return 0;
  }
  const nlohmann::json* t = &j;
  for (const char* k : {"llm_config", "text_config"})
    if (j.is_object() && j.contains(k) && j[k].is_object()) {
      t = &j[k];
      break;
    }
  return t->is_object() ? t->value("hidden_size", 0) : 0;
}

}  // namespace

std::string NhDrafter::find(const std::string& model_dir) {
  const std::string l = LiliCorr::find(model_dir);
  if (!l.empty()) return l;
  const int hidden = target_hidden(model_dir);
  for (const char* sub : {"/dflash", "/dspark"})
    if (DSpark::is_dspark(model_dir + sub, hidden)) return model_dir + sub;
  return "";
}

std::string NhDrafter::kind(const std::string& dir) {
  if (DSpark::is_dspark(dir, target_hidden(dir))) return "dspark";
  return dir.empty() ? "" : "lilicorr";
}

size_t NhDrafter::plan_bytes(const std::string& dir, size_t* per_slot) {
  return kind(dir) == "dspark" ? DSpark::plan_bytes(dir, per_slot) : LiliCorr::plan_bytes(dir, per_slot);
}

std::unique_ptr<NhDrafter> NhDrafter::open(NemotronH& target, const std::string& dir) {
  if (kind(dir) == "dspark") return std::make_unique<DSpark>(target, dir);
  return std::make_unique<LiliCorr>(target, dir);
}

void NhDrafter::encode_draft_chained(Encoder&, Seq&, LiliSeq&, int) { fatal("%s: no chained drafting", name()); }

void NhDrafter::reset(LiliSeq& d) { d.ctx = d.hi = 0; }

void NhDrafter::rewind(LiliSeq& d, int pos) { d.ctx = pos; }

void NhDrafter::stage_init(LiliSeq& d) {
  if (d.stage) return;
  const NhCapture& v = verify_hook();
  d.stage = Gpu::get().alloc(size_t(kB) * v.ld * 4, "nh_draft_stage");
  d.staged = 0;
  d.stage_hook.feat = d.stage;
  d.stage_hook.ld = v.ld;
  d.stage_hook.slot = v.slot;
  d.stage_hook.pos_base = 0;
  d.stage_hook.rows = kB;
}

void NhDrafter::encode_commit_path(Encoder& e, LiliSeq& d, int m, const Buffer& path) {
  if (m <= 0) return;
  const NhCapture& v = verify_hook();
  if (!kfeat_) kfeat_ = Gpu::get().alloc(size_t(kB) * v.ld * 4, "nh_kf_feat");
  const uint32_t ld = v.ld;
  e.use(NemotronH::pipeline("nh_kf_gather"));
  e.buf(0, v.feat);
  e.buf(1, kfeat_);
  e.buf(2, path);
  e.val(3, ld);
  e.dispatch(uint32_t(m), (ld + 255) / 256, 1, 256);
  enc_commit_rows(e, d, kfeat_, m, d.ctx);
  d.ctx += m;
  d.hi = std::max(d.hi, d.ctx);
}

void NhDrafter::enc_kf_build(Encoder& e, NemotronH& tg, Seq& s, int P, const Buffer& cand, const Buffer& tlq, int K,
                             const Buffer& kfp) {
  const int S = kB - 1;
  if (kf_K_ != K) {  // the lattice's tables: tokens per children-row (rewritten per pass), constant successors
    Gpu& g = Gpu::get();
    if (kf_tok_) g.release(kf_tok_);
    if (kf_next_) g.release(kf_next_);
    kf_tok_ = g.alloc(size_t(S) * K * K * 4, "nh_kf_tok");
    kf_next_ = g.alloc(size_t(S) * K * K * 4, "nh_kf_next");
    const std::vector<int32_t> nx = kf::lattice_next(S, K);
    memcpy(kf_next_.ptr(), nx.data(), nx.size() * 4);
    kf_K_ = K;
  }
  const struct {
    uint32_t S, K;
  } lp{uint32_t(S), uint32_t(K)};
  e.use(NemotronH::pipeline("kf_lattice_tokens"));
  e.buf(0, cand);
  e.buf(1, kf_tok_);
  e.bytes(2, &lp, sizeof lp);
  e.dispatch((uint32_t(S * K * K) + 255) / 256, 1, 1, 256);
  const struct {
    uint32_t K, P, root, pad;
  } bp{uint32_t(K), uint32_t(P), 0u, 0u};
  e.use(NemotronH::pipeline("kf_build"));
  e.buf(0, tlq);
  e.buf(1, kf_tok_);
  e.buf(2, kf_next_);
  e.buf(3, kfp);
  e.buf(4, tg.tree_buf());
  e.buf(5, s.tok);
  e.buf(6, tg.live_rows());
  e.bytes(7, &bp, sizeof bp);
  e.dispatch(1, 1, 1, 32);
}

void NhDrafter::encode_commit_staged(Encoder& e, LiliSeq& d) {
  if (d.staged <= 0) return;
  enc_commit_rows(e, d, d.stage, d.staged, d.ctx);
  d.ctx += d.staged;
  d.hi = std::max(d.hi, d.ctx);
  d.staged = 0;
}

}  // namespace ni
