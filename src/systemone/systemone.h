// Drex systemone on Kapitsa: the POST /v1/systemone request encoder and the pointer head.
//
// A Drex decision model is its backbone's hidden states read by a small pointer head: every question becomes one row
// [<|fim_middle|> instructions (<|box_start|> option <|box_end|>)* <|fim_suffix|>] that continues a shared
// [<|fim_prefix|> state] row, and the head scores each option's closing token against the row's decide token:
//   logits = (k(h_option) @ q(h_decide)) / sqrt(dp) / temperature,  probs = softmax(logits).
// No token is generated.
//
// The encoder and the answer formatting are ported from the reference (kev/api.py + kev/model.py, Apache-2.0, and the
// llama.cpp fork's tools/server/systemone.cpp, Apache-2.0) so that Kapitsa's replies carry the same numbers as the
// reference server for the same weights; see docs/systemone.md.
#pragma once

#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ni::systemone {

using ojson = nlohmann::ordered_json;

// A request the protocol rejects (400), as opposed to a failure of the model's own scoring (500).
struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- limits (SYSTEMONE_* environment variables as in the reference fork) ----
constexpr int kMaxOptions = 255;
constexpr int kRecommendedContext = 16384;  // the serving context the model is calibrated for
constexpr int kPositionLimit = 131072;

struct Limits {
  int max_state = kRecommendedContext;
  int max_branch = kRecommendedContext;
  int max_packed = kRecommendedContext;  // the whole encoded request's token count
};
// SYSTEMONE_CONTEXT sets every cap, SYSTEMONE_MAX_STATE / _BRANCH / _PACKED override one.
Limits limits_from_env();

// A question's readout: which row of its branch holds the decide token and which hold each option's closing token.
struct Group {
  int decide = -1;               // row index within the branch
  std::vector<int> options;      // row indices within the branch
};

// What the request asks the answer formatter to report for one question.
struct QuestionMeta {
  std::string id;
  std::string type;              // choice | noul | score
  std::vector<std::string> keys; // probability keys, in option order (choice names, false/true, "0".."n-1")
  std::vector<std::string> legend;  // score: the rendered level descriptions
};

// One question's row: the tokens to decode at positions n_state, n_state + 1, ... and its readout rows.
struct Branch {
  std::vector<int32_t> tokens;
  Group group;
};

struct Encoded {
  std::vector<int32_t> state;    // the shared state row, positions 0 .. n_state-1
  std::vector<Branch> branches;  // one per question, in request order
  std::vector<QuestionMeta> meta;
  int n_state() const { return (int) state.size(); }
  size_t max_branch() const;
  size_t total_tokens() const;   // state + every branch, the request's input tokens
};

// Encodes one {"state", "questions"} object. Throws std::runtime_error when it is invalid or exceeds `limits`.
Encoded encode(const Tokenizer& tok, const ojson& body, const Limits& limits);

// The answers object of the reference server, from one logit vector per question (its options, in option order).
// Throws std::runtime_error when the logits do not match the questions.
ojson format_answers(const std::vector<std::vector<float>>& logits, const std::vector<QuestionMeta>& meta);

// Tokens of the Python json.dumps of the answers (the reference server's billing-style output_tokens).
int output_tokens(const Tokenizer& tok, const ojson& answers);

// ---- the pointer head (pointer.bin) ----
// Written from a Drex GGUF's pointer.q.* / pointer.k.* / pointer.temperature.* tensors by tools/drex_pointer.py.
// Layout, little endian: magic "KPS1", u32 version = 1, u32 hidden, u32 dim, then f32 q_w[dim][hidden], q_b[dim],
// k_w[dim][hidden], k_b[dim], temperature.
class PointerHead {
 public:
  // Throws std::runtime_error (bad magic/version/size, non-finite weights, a non-positive temperature).
  static PointerHead load(const std::string& path);
  // Finds pointer.bin next to the model's weights: <dir>/pointer.bin, else <dir>/.kapitsa/pointer.bin.
  static std::string find(const std::string& model_dir);
  bool ok() const { return dim_ > 0; }
  int hidden() const { return hidden_; }
  int dim() const { return dim_; }
  float temperature() const { return temperature_; }
  // One question's logits: `h_decide` and each of the `n` option rows are post-final-norm hidden rows (hidden floats).
  void logits(const float* h_decide, const std::vector<const float*>& option_rows, std::vector<float>* out) const;

 private:
  int hidden_ = 0, dim_ = 0;
  float temperature_ = 1.0f;
  std::vector<float> q_w_, q_b_, k_w_, k_b_;
};

// Post-final-norm row of a Qwen3.5 backbone's residual row: x * rsqrt(mean(x^2) + eps) * w, as the model's own
// rows_to_half kernel computes it before its LM head. `x` holds `hidden` floats, `w` the final norm weights.
void final_norm_row(const float* x, const float* w, int hidden, float eps, std::vector<float>* out);

}  // namespace ni::systemone
