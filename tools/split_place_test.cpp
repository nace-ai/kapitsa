// CPU test of the split's placement (src/cluster/place.h): no GPU, no model. Two equal Macs take half the layers each;
// a Mac with twice the bandwidth takes more of them; memory bounds a run; the head holds its own tensors and the
// n-gram block's layers; with more members than layers, or too little memory, nothing fits. plan_split on a Flash-Next
// Q2_0 pack of headers alone (written to a temporary folder): every layer in memory where the Macs hold it, else with
// may_stream each Mac's dense weights and a store, the rest of its experts streamed.
//   split_place_test
#include "cluster/place.h"
#include "model/qwen4_exp.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace ni;
using namespace ni::cluster;
using json = nlohmann::json;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                      \
  do {                                                     \
    if (!(c)) {                                            \
      ++g_fail;                                            \
      fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__);                        \
      fprintf(stderr, "\n");                               \
    }                                                      \
  } while (0)

// 48 layers of 0.8 GB, a decode step reading 60 MB of each (Flash-Next's Q2_0 pack, roughly); the head's tensors 1 GB
std::vector<LayerCost> layers48() { return std::vector<LayerCost>(48, LayerCost{0.8e9, 60e6}); }
const LayerCost kHead{1e9, 300e6};

// A safetensors file of headers alone (the planner reads nothing else): each tensor `bytes` long, one after another
void write_headers(const std::string& path, const std::vector<std::pair<std::string, size_t>>& tensors) {
  json h = json::object();
  size_t at = 0;
  for (const auto& [name, bytes] : tensors) {
    h[name] = {{"dtype", "U8"}, {"shape", {bytes}}, {"data_offsets", {at, at + bytes}}};
    at += bytes;
  }
  const std::string t = h.dump();
  const uint64_t n = t.size();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&n), 8);
  f.write(t.data(), std::streamsize(n));
}

// A Flash-Next Q2_0 pack of 12 layers (every 4th full attention): per layer 1.5 MB of dense weights and 40 MB of
// routed experts, the head's 6 MB (the LM head; the embedding apart), the MTP head's 4 MB
std::string write_pack() {
  char tmpl[] = "/tmp/split_place_test.XXXXXX";
  const std::string dir = mkdtemp(tmpl);
  json t = json::parse(R"({"model_type": "qwen4_exp_text", "hidden_size": 512, "num_hidden_layers": 12,
    "vocab_size": 2048, "full_attention_interval": 4, "linear_num_key_heads": 1, "linear_num_value_heads": 3,
    "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "num_attention_heads": 2,
    "num_key_value_heads": 1, "head_dim": 256, "rope_parameters": {"rope_theta": 10000000,
    "partial_rotary_factor": 0.25, "rope_type": "default", "mrope_section": [11, 11, 10], "mrope_interleaved": true},
    "rms_norm_eps": 1e-06, "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 64,
    "shared_expert_intermediate_size": 64, "norm_topk_prob": true, "hc_count": 4, "hc_lowrank": 64,
    "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4,
    "output_gate_type": "sigmoid", "ple_layer_ids": [2], "ngram_size": 3, "heads_per_ngram": 4,
    "ple_conv_kernel_size": 4, "ngram_vocab_size_base": 1000, "eos_token_id": 2047, "mtp_num_hidden_layers": 1,
    "tie_word_embeddings": false})");
  t["layer_types"] = json::array();
  for (int i = 0; i < 12; ++i) t["layer_types"].push_back(i % 4 == 3 ? "full_attention" : "linear_attention");
  json root = {{"architectures", {"Qwen4ExpForConditionalGeneration"}}, {"model_type", "qwen4_exp"}, {"text_config", t},
               {"quantization_config", {{"group_size", 64}, {"bits", 4}}}};
  for (const char* p : {"gate_proj", "up_proj", "down_proj"})
    root["quantization_config"][std::string("language_model.model.layers.0.mlp.switch_mlp.") + p] = {
        {"group_size", 64}, {"bits", 2}, {"mode", "q2_0"}};
  std::ofstream(dir + "/config.json") << root.dump();
  std::vector<std::pair<std::string, size_t>> trunk = {{"language_model.lm_head.weight", 6000000},
                                                       {"language_model.model.embed_tokens.weight", 2000000}};
  for (int i = 0; i < 12; ++i) {
    const std::string l = "language_model.model.layers." + std::to_string(i) + ".";
    trunk.push_back({l + "mlp.shared_expert.gate_proj.weight", 1500000});
    trunk.push_back({l + "mlp.switch_mlp.gate_proj.weight", 40000000});
  }
  write_headers(dir + "/model.safetensors", trunk);
  write_headers(dir + "/mtp.safetensors", {{"mtp.layers.0.mlp.gate_proj.weight", 4000000}});
  return dir;
}

}  // namespace

int main() {
  {  // two equal Macs: the head's tensors make its run a little shorter
    const Placement p = place(layers48(), kHead, 2, {{"a", 30e9, 273}, {"b", 30e9, 273}});
    CHECK(p.fits, "two 30 GB Macs hold 39.4 GB");
    CHECK(p.lo.size() == 3 && p.lo[0] == 0 && p.lo[2] == 48, "runs cover the layers");
    CHECK(p.lo[1] >= 23 && p.lo[1] <= 25, "an even split, not %d", p.lo[1]);
    CHECK(p.held[0] <= 30e9 && p.held[1] <= 30e9, "within memory");
    // the time: every byte read once, at 273 GB/s
    const double want = (48 * 60e6 + 300e6) / 273e6;
    CHECK(std::fabs(p.ms - want) < 1e-6, "%.4f ms, want %.4f", p.ms, want);
    CHECK(std::fabs(whole_ms(layers48(), kHead, 273) - want) < 1e-6, "whole_ms");
  }
  {  // the faster Mac takes as much as its memory allows
    const Placement p = place(layers48(), kHead, 2, {{"a", 30e9, 273}, {"fast", 30e9, 546}});
    CHECK(p.fits && p.lo[1] < 24, "the head keeps the minimum the fast Mac's memory leaves it: %d", p.lo[1]);
    CHECK(p.held[1] <= 30e9, "within the fast Mac's memory");
  }
  {  // memory bounds the runs: a 12 GB Mac holds at most 15 layers
    const Placement p = place(layers48(), kHead, 2, {{"head", 40e9, 273}, {"small", 12e9, 546}});
    CHECK(p.fits && 48 - p.lo[1] <= 15, "the small Mac holds %d layers", 48 - p.lo[1]);
  }
  {  // the head keeps the n-gram block's layers even when the stage is far faster
    const Placement p = place(layers48(), kHead, 2, {{"head", 40e9, 100}, {"fast", 40e9, 800}});
    CHECK(p.fits && p.lo[1] >= 2, "the head holds layers 1-2, not %d", p.lo[1]);
  }
  {  // three Macs, each at least one layer, in ring order
    const Placement p = place(layers48(), kHead, 2, {{"a", 14e9, 273}, {"b", 14e9, 273}, {"c", 14e9, 273}});
    CHECK(p.fits && p.lo.size() == 4 && p.lo[0] == 0 && p.lo[1] < p.lo[2] && p.lo[2] < p.lo[3] && p.lo[3] == 48,
          "three runs");
  }
  {  // too little memory, too many members
    CHECK(!place(layers48(), kHead, 2, {{"a", 15e9, 273}, {"b", 15e9, 273}}).fits,
          "two 15 GB Macs do not hold 39.4 GB");
    const std::vector<LayerCost> two(2, LayerCost{1, 1});
    CHECK(!place(two, kHead, 1, {{"a", 9e9, 1}, {"b", 9e9, 1}, {"c", 9e9, 1}}).fits, "three Macs, two layers");
  }
  {  // unknown bandwidth: the slowest known member's
    const Placement p = place(layers48(), kHead, 2, {{"a", 30e9, 0}, {"b", 30e9, 273}});
    CHECK(p.fits && p.lo[1] >= 23 && p.lo[1] <= 25, "an unknown Mac counts as the known one: %d", p.lo[1]);
  }
  {  // plan_split on a pack's headers
    const std::string dir = write_pack();
    Qwen4Exp::LayerBytes hb;
    const auto lb = Qwen4Exp::layer_bytes(dir, &hb);
    CHECK(lb.size() == 12 && lb[0].weights == 41.5e6 && lb[0].experts == 40e6 && lb[0].linear && !lb[3].linear,
          "layer_bytes: a layer's weights, its experts, its kind");
    CHECK(hb.weights == 6e6 + 2e6 + 4e6 && hb.experts == 0, "layer_bytes: the head's own tensors, the MTP head's: %.0f",
          hb.weights);
    setenv("NI_NO_MTP", "1", 1);
    Qwen4Exp::layer_bytes(dir, &hb);
    unsetenv("NI_NO_MTP");
    CHECK(hb.weights == 8e6, "layer_bytes: NI_NO_MTP leaves the MTP head out: %.0f", hb.weights);
    // the least memory two equal Macs need, every layer in memory, and with experts streamed
    auto least = [&](bool stream) {
      double lo = 0, hi = 8e9;
      for (int i = 0; i < 40; ++i) {
        const double m = (lo + hi) / 2;
        (plan_split(dir, {{"a", m, 100}, {"b", m, 100}}, 2048, stream).fits ? hi : lo) = m;
      }
      return hi;
    };
    const double in_mem = least(false), streamed = least(true);
    // streamed, a Mac of the two holds about 6 layers' experts (240 MB) less, and a store of 3 layers' more
    const double saved = 6 * 40e6 - 3 * Qwen4Exp::store_layer_bytes(dir);
    CHECK(std::fabs(in_mem - streamed - saved) < 45e6, "streaming needs less memory a Mac: %.0f MB against %.0f in memory "
          "(%.0f MB less expected)", streamed / 1e6, in_mem / 1e6, saved / 1e6);
    const double mid = (in_mem + streamed) / 2;
    const Placement s = plan_split(dir, {{"a", mid, 100}, {"b", mid, 100}}, 2048, true);
    CHECK(s.fits && s.streams, "between the two: the Macs stream their experts");
    CHECK(!plan_split(dir, {{"a", mid, 100}, {"b", mid, 100}}, 2048, false).fits, "and without may_stream nothing fits");
    const Placement r = plan_split(dir, {{"a", in_mem * 1.05, 100}, {"b", in_mem * 1.05, 100}}, 2048, true);
    CHECK(r.fits && !r.streams, "where the Macs hold it, every layer in memory");
    CHECK(r.lo[1] >= 2, "the head keeps the n-gram block's layers: %d", r.lo[1]);
    std::remove((dir + "/config.json").c_str());
    std::remove((dir + "/model.safetensors").c_str());
    std::remove((dir + "/mtp.safetensors").c_str());
    std::remove(dir.c_str());
  }
  printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
  return g_fail ? 1 : 0;
}
