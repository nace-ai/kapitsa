// Parity check for thinkingmachines/Inkling-2-Mini against tests/fixtures/ink2_golden.json.
#include "metal/gpu.h"
#include "model/inkling2.h"
#include "model/sampler.h"
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>

using json = nlohmann::json;

#ifndef INK2_GOLDEN_PATH
#define INK2_GOLDEN_PATH "tests/fixtures/ink2_golden.json"
#endif

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: ink2_check <model_dir> [golden_json]\n");
    return 2;
  }
  const std::string model_dir = argv[1];
  std::string golden_path;
  if (argc > 2) {
    golden_path = argv[2];
  } else {
    golden_path = std::ifstream(INK2_GOLDEN_PATH).good() ? INK2_GOLDEN_PATH : "tests/fixtures/ink2_golden.json";
  }

  std::ifstream gf(golden_path);
  if (!gf) {
    fprintf(stderr, "cannot open %s\n", golden_path.c_str());
    return 1;
  }
  const json golden = json::parse(gf);

  ni::Tokenizer tok;
  tok.load(model_dir + "/tokenizer.json");
  ni::Inkling2 model(model_dir, 512);
  ni::Gpu& gpu = ni::Gpu::get();
  const int V = model.vocab();

  ni::SamplingParams greedy;
  greedy.temperature = 0.0f;
  greedy.top_p = 1.0f;
  greedy.top_k = 1;
  greedy.seed = 0;

  int fails = 0;
  for (const auto& p : golden.at("prompts")) {
    const std::string name = p.at("name").get<std::string>();
    const std::string text = p.at("text").get<std::string>();
    const std::vector<int> want_ids = p.at("ids").get<std::vector<int>>();
    const std::vector<int> got_ids = tok.encode(text);
    if (got_ids != want_ids) {
      printf("FAIL %s: tokenizer mismatch (%zu vs %zu)\n", name.c_str(), got_ids.size(), want_ids.size());
      ++fails;
      continue;
    }

    auto seq = model.new_seq(512, 512);
    int* tok_ptr = seq->tokens();
    for (size_t i = 0; i < got_ids.size(); ++i) tok_ptr[i] = got_ids[i];

    const int n_prompt = int(got_ids.size());
    {
      auto e = gpu.begin();
      model.encode_prefill(*e, *seq, 0, n_prompt, greedy, 0, true);
      e->commit_and_wait();
    }

    const float* logits = model.logits();
    const std::vector<int> top10_ids = p.at("top10_ids").get<std::vector<int>>();
    const std::vector<float> top10_vals = p.at("top10_vals").get<std::vector<float>>();
    const std::vector<int> fp16_top10_ids = p.at("fp16_kv_top10_ids").get<std::vector<int>>();
    const std::vector<float> fp16_top10_vals = p.at("fp16_kv_top10_vals").get<std::vector<float>>();
    const std::vector<int> want_rollout = p.at("greedy_24").get<std::vector<int>>();

    int top1_id = 0;
    float top1_val = logits[0];
    for (int v = 1; v < V; ++v) {
      if (logits[v] > top1_val) {
        top1_val = logits[v];
        top1_id = v;
      }
    }
    float max_top10_diff_fp32 = 0.0f;
    for (size_t k = 0; k < top10_ids.size(); ++k) {
      max_top10_diff_fp32 = std::max(max_top10_diff_fp32, std::fabs(logits[top10_ids[k]] - top10_vals[k]));
    }
    float max_top10_diff_fp16 = 0.0f;
    for (size_t k = 0; k < fp16_top10_ids.size(); ++k) {
      max_top10_diff_fp16 = std::max(max_top10_diff_fp16, std::fabs(logits[fp16_top10_ids[k]] - fp16_top10_vals[k]));
    }

    std::vector<int> rollout;
    rollout.push_back(tok_ptr[n_prompt]);
    int pos = n_prompt;
    for (int step = 1; step < 24; ++step) {
      auto e = gpu.begin();
      model.encode_step(*e, *seq, pos, greedy, uint32_t(step), true);
      e->commit_and_wait();
      ++pos;
      rollout.push_back(tok_ptr[pos]);
    }

    int match_n = 0;
    for (size_t i = 0; i < want_rollout.size() && i < rollout.size(); ++i) {
      if (rollout[i] == want_rollout[i]) ++match_n;
      else break;
    }
    const bool ok = (top1_id == top10_ids[0]) && (max_top10_diff_fp16 < 0.01f) && (rollout == want_rollout);
    printf("%s %s: top1=%d (want %d) top1_logit=%.4f (fp16 %.4f, fp32 %.4f) fp16_top10_diff=%.5f fp32_top10_diff=%.5f rollout=%d/24\n",
           ok ? "PASS" : "FAIL", name.c_str(), top1_id, top10_ids[0], top1_val, fp16_top10_vals[0], top10_vals[0],
           max_top10_diff_fp16, max_top10_diff_fp32, match_n);
    if (!ok) {
      ++fails;
      printf("  got  rollout:");
      for (int id : rollout) printf(" %d", id);
      printf("\n  want rollout:");
      for (int id : want_rollout) printf(" %d", id);
      printf("\n");
      if (p.contains("dumps")) {
        auto dseq = model.new_seq(512, 512);
        for (size_t i = 0; i < got_ids.size(); ++i) dseq->tokens()[i] = got_ids[i];
        const int H = model.config().hidden;
        ni::Buffer diag = gpu.alloc(8 * size_t(H) * 4, "ink2_diag");
        {
          auto e = gpu.begin();
          model.encode_prefill_diag(*e, *dseq, 0, n_prompt, diag);
          e->commit_and_wait();
        }
        const char* keys[8] = {"after_embed", "v0_out", "v5_out", "v17_out", "v18_out", "v30_out", "v44_out", "final_norm"};
        for (int sidx = 0; sidx < 8; ++sidx) {
          const float* v = diag.as<float>() + size_t(sidx) * size_t(H);
          double sum = 0.0, sq = 0.0;
          for (int c = 0; c < H; ++c) {
            sum += v[c];
            sq += double(v[c]) * double(v[c]);
          }
          const auto& w = p["dumps"].at(keys[sidx]);
          printf("  diag %-12s got(sum=%.4f l2=%.4f [%.5f,%.5f,%.5f,%.5f]) want(sum=%.4f l2=%.4f [%.5f,%.5f,%.5f,%.5f])\n",
                 keys[sidx], sum, std::sqrt(sq), v[0], v[1], v[2], v[3],
                 w["sum"].get<double>(), w["l2"].get<double>(),
                 w["first8"][0].get<double>(), w["first8"][1].get<double>(),
                 w["first8"][2].get<double>(), w["first8"][3].get<double>());
        }
        gpu.release(diag);
      }
    }
  }
  return fails ? 1 : 0;
}
