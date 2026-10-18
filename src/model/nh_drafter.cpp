#include "model/nh_drafter.h"
#include "model/dspark.h"
#include "model/lilicorr.h"

#include <nlohmann/json.hpp>

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

void NhDrafter::reset(LiliSeq& d) { d.ctx = d.hi = 0; }

void NhDrafter::rewind(LiliSeq& d, int pos) { d.ctx = pos; }

}  // namespace ni
