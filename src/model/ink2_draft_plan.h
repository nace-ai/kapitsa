#pragma once
#include "model/ink2_draft_layout.h"
#include "loader/safetensors.h"

namespace ni::ink2draft {
struct DraftPlan {
  uint32_t vocab, padded;
  float eps, theta;
  WeightStore::FilePlan weights;
  SharedLayout layout;
};
DraftPlan read_plan(const std::string& path);
}  // namespace ni::ink2draft
