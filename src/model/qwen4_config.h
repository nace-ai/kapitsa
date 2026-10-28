// Qwen 4-family config.json files (a qwen4* model_type) against what Kapitsa's Qwen3.8-Flash-Next path runs
// (model/qwen4_exp.h: model_type qwen4_exp, Qwen's preview of the Qwen 4 architecture). Read from the config alone,
// without the weights or the GPU, so the Hub check refuses before a download what the loader would refuse after it,
// and a Qwen 4 release that changes a setting this path reads, or adds one it does not, says which.
#pragma once
#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace ni {

// config.json's model_type, or its text_config's, starts with qwen4 (qwen4_exp among them)
bool qwen4_family(const nlohmann::json& config);

// What in a Qwen 4-family config the qwen4_exp path does not run, one line each; empty: it runs. A model_type other
// than qwen4_exp / qwen4_exp_text is such a line, so a new Qwen 4 architecture runs only once its differences have
// been read.
std::vector<std::string> qwen4_unsupported(const nlohmann::json& config);

// What in a Qwen 4-family config the qwen4_exp path does not apply, one line each: RoPE scaling (YaRN; Sushi's
// Flash-Next pack asks for it, and runs, as every Flash-Next folder does, with RoPE unscaled) and the text_config
// settings it does not know (not Flash-Next's, Transformers' Qwen4ExpTextConfig's or a converter's: a new part of the
// architecture, or nothing). A qwen4_exp model loads with them and says so.
std::vector<std::string> qwen4_unapplied(const nlohmann::json& config);

// The refusal of a Qwen 4-family model whose qwen4_unsupported is not empty, for the engine and the Hub check: what
// it is (`what`: a folder or a repo), each line, then its qwen4_unapplied. "" when it runs.
std::string qwen4_refusal(const nlohmann::json& config, const std::string& what);

}  // namespace ni
