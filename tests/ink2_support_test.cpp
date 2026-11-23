#include "model/ink2_draft.h"
#include "model/ink2_draft_layout.h"
#include "loader/safetensors.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed: %s:%d: %s\n", __FILE__, __LINE__, #c); std::abort(); } } while (false)
using namespace ni;
using namespace ni::ink2draft;
using json = nlohmann::json;
template<class F> void rejects(F f) {
  bool threw = false;
  try { f(); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
}
int main() {
  CHECK(empty_seq_bytes() == 8397824);
  CHECK(SharedLayout(200058).total() == 143863208);
  CHECK(growth(0, 1, SIZE_MAX).allocation_bytes == 16777216);
  CHECK(growth(4096, 4096, SIZE_MAX).allocation_bytes == 0);
  CHECK(growth(4096, 4097, SIZE_MAX).allocation_bytes == 33554432);
  CHECK(growth(0, 5000, SIZE_MAX).cap == 5000);
  CHECK(growth(5000, 5001, SIZE_MAX).cap == 10000);
  rejects([] { (void)growth(4096, 4097, 16777215); });
  rejects([] { (void)growth(UINT32_MAX / 2 + 1, UINT32_MAX, SIZE_MAX); });
  rejects([] { (void)append_end(INT32_MAX, 1); });
  rejects([] { (void)append_end(0, 0); });
  rejects([] { (void)add(SIZE_MAX, 1); });
  rejects([] { (void)mul(SIZE_MAX, 2); });
  CHECK(safetensors_packed_end(safetensors_packed_end(0, 1), 257) == 768);
  rejects([] { (void)safetensors_packed_end(0, SIZE_MAX); });
  rejects([] { (void)safetensors_packed_end(SIZE_MAX - 255, 256); });
  char pattern[] = "/tmp/ink2-support-XXXXXX";
  const char* made = mkdtemp(pattern);
  CHECK(made);
  const std::filesystem::path dir(made);
  struct Cleanup { std::filesystem::path dir; ~Cleanup() { std::filesystem::remove_all(dir); } } cleanup{dir};
  CHECK(Ink2Draft::memory_plan(dir.string()).shared_bytes == 0);
  const auto file = dir / "mtp.safetensors";
  std::filesystem::create_symlink("missing-mtp-target", file);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  std::filesystem::remove(file);
  auto write = [&](const json& h, size_t data_bytes) {
    std::string bytes = h.dump();
    bytes.append((8 - bytes.size() % 8) % 8, ' ');
    const uint64_t len = bytes.size();
    std::ofstream f(file, std::ios::binary | std::ios::trunc);
    f.write(reinterpret_cast<const char*>(&len), sizeof len);
    f << bytes;
    if (data_bytes) { f.seekp(std::streamoff(8 + len + data_bytes - 1)); f.put('\0'); }
    f.close();
  };
  json header{{"extra", {{"dtype", "U8"}, {"shape", {3}}, {"data_offsets", {0, 3}}}}};
  write(header, 3);
  CHECK(WeightStore::plan_file(file.string()).bytes == 256);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  header["extra"]["data_offsets"] = {0, 4};
  write(header, 4);
  rejects([&] { (void)WeightStore::plan_file(file.string()); });
  header["extra"]["data_offsets"] = {0, 3};
  write(header, 2);
  rejects([&] { (void)WeightStore::plan_file(file.string()); });
  header["extra"]["dtype"] = "BAD";
  write(header, 3);
  rejects([&] { (void)WeightStore::plan_file(file.string()); });
  json config{{"hidden_size", H}, {"vocab_size", 16}, {"unpadded_vocab_size", 8},
    {"mtp_config", {{"hidden_size", H}, {"num_hidden_layers", 3}, {"num_attention_heads", HQ},
      {"num_key_value_heads", HKV}, {"head_dim", D}, {"intermediate_size", I}, {"markov_rank", R},
      {"block_size", WIDTH}, {"markov_head_type", "rnn_hidden"}, {"q_bias", false}, {"o_bias", false},
      {"local_layer_ids", {0, 2}}, {"sliding_window_size", 1024}, {"target_layer_ids", {5, 23, 41, 48}},
      {"rms_norm_eps", 1e-6}, {"rope_theta", 10000.0}}}};
  auto write_config = [&](const json& c) { std::ofstream(dir / "config.json") << c.dump(); };
  write_config(config);
  header = json::object();
  const int64_t padded = 16;
  auto tensor = [&](const std::string& name, std::vector<int64_t> shape) {
    header[name] = {{"dtype", "BF16"}, {"shape", shape}};
  };
  tensor("markov_head.hidden_out.weight", {H, R});
  tensor("markov_head.hidden_out.bias", {H});
  tensor("fuse_weight", {4, 4});
  tensor("norm.weight", {H});
  tensor("mask_embed", {H});
  tensor("markov_head.markov_w1.weight", {padded, R});
  tensor("markov_head.dflare_norm.weight", {H});
  tensor("markov_head.rnn_proj.bias", {3 * R});
  tensor("markov_head.rnn_proj.weight", {3 * R, 2 * R + 2 * H});
  for (int i = 0; i < 3; ++i) {
    const std::string s = "layers." + std::to_string(i) + ".";
    tensor(s + "dflare_norm.weight", {H});
    tensor(s + "self_attn.k_proj_ctx.weight", {HKV * D, H});
    tensor(s + "self_attn.v_proj_ctx.weight", {HKV * D, H});
    tensor(s + "self_attn.k_norm_ctx.weight", {D});
    tensor(s + "self_attn.q_proj.weight", {HQ * D, H});
    tensor(s + "self_attn.k_proj.weight", {HKV * D, H});
    tensor(s + "self_attn.v_proj.weight", {HKV * D, H});
    tensor(s + "self_attn.q_norm.weight", {D});
    tensor(s + "self_attn.k_norm.weight", {D});
    tensor(s + "self_attn.o_proj.weight", {H, HQ * D});
    tensor(s + "input_layernorm.weight", {H});
    tensor(s + "post_attention_layernorm.weight", {H});
    tensor(s + "mlp.gate_proj.weight", {I, H});
    tensor(s + "mlp.up_proj.weight", {I, H});
    tensor(s + "mlp.down_proj.weight", {H, I});
  }
  header["unused.extra"] = {{"dtype", "F32"}, {"shape", {3}}};
  CHECK(header.size() == 55);
  auto write_packed = [&](json h) {
    size_t data = 0, packed = 0;
    for (auto& [name, t] : h.items()) {
      size_t count = 1;
      for (const auto& dim : t["shape"]) count *= dim.get<size_t>();
      const size_t bytes = count * (t["dtype"] == "F32" ? 4 : 2);
      t["data_offsets"] = {data, data + bytes};
      data += bytes;
      packed += (bytes + 255) / 256 * 256;
    }
    write(h, data);
    return packed;
  };
  const size_t packed = write_packed(header);
  const auto plan = Ink2Draft::memory_plan(dir.string());
  CHECK(plan.shared_bytes == packed + 546 * 8 + 34630756 + 8);
  CHECK(plan.empty_seq_bytes == 8397824);
  CHECK(WeightStore::plan_file(file.string()).bytes == packed);
  auto bad = header;
  bad.erase("norm.weight");
  write_packed(bad);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  bad = header;
  bad["norm.weight"]["dtype"] = "F32";
  write_packed(bad);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  bad = header;
  bad["markov_head.markov_w1.weight"]["shape"] = {8, R};
  write_packed(bad);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  write_packed(header);
  auto unsupported = config;
  unsupported["mtp_config"]["num_hidden_layers"] = 4;
  write_config(unsupported);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  unsupported = config;
  unsupported["unpadded_vocab_size"] = 4096u * topk_chunk(1) + 1;
  unsupported["vocab_size"] = unsupported["unpadded_vocab_size"];
  write_config(unsupported);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  unsupported = config;
  unsupported["unpadded_vocab_size"] = -1;
  write_config(unsupported);
  rejects([&] { (void)Ink2Draft::memory_plan(dir.string()); });
  std::puts("ink2_support_test: ok (CPU only)");
}
