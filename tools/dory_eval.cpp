// Dory checks against reference runs (a reference folder holds <name>.json: {"prompt_ids": [...], "gen": [...]} and
// its logits; scratchpad scripts write them from NVIDIA's model code in PyTorch):
//   dory_eval <dir> --ref R [--out O] [--prompts a,b] [--loops N] [--kv last-loop|per-loop] [--chunk C]
//             [--greedy N] [--tf 0|1]
// For each reference: teacher forcing (the prompt in prefill chunks of C rows, then the reference's tokens one decode
// step each, as the reference ran them): O/<name>.kap.npy, fp32 [1 + steps][vocab] next-token logits; and greedy
// decoding of N tokens from the prompt: O/<name>.greedy.json (the tokens, prefill and decode times).
#include "metal/gpu.h"
#include "model/dory.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ni {
extern const char* const kDoryKernelSource;
}
using namespace ni;

namespace {

void write_npy(const std::string& path, const std::vector<float>& data, size_t rows, size_t cols) {
  std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", " +
                    std::to_string(cols) + "), }";
  const size_t base = 10;
  size_t total = base + hdr.size() + 1;
  const size_t pad = (64 - total % 64) % 64;
  hdr += std::string(pad, ' ') + "\n";
  std::ofstream f(path, std::ios::binary);
  const char magic[] = "\x93NUMPY\x01\x00";
  f.write(magic, 8);
  const uint16_t hl = uint16_t(hdr.size());
  f.write(reinterpret_cast<const char*>(&hl), 2);
  f.write(hdr.data(), std::streamsize(hdr.size()));
  f.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size() * 4));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: dory_eval <dir> --ref R [--out O] [--prompts a,b] [--loops N] [--kv last-loop|per-loop] "
                    "[--chunk C] [--greedy N] [--tf 0|1]\n");
    return 1;
  }
  std::string model = argv[1], ref, out, only;
  int loops = 0, chunk = 2048, greedy = 0, tf = 1;
  LoopKv kv = LoopKv::last_loop;
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--ref") ref = next();
    else if (a == "--out") out = next();
    else if (a == "--prompts") only = "," + next() + ",";
    else if (a == "--loops") loops = std::stoi(next());
    else if (a == "--kv") {
      if (!parse_loop_kv(next(), &kv)) fatal("--kv: last-loop or per-loop");
    } else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--greedy") greedy = std::stoi(next());
    else if (a == "--tf") tf = std::stoi(next());
    else fatal("unknown argument %s", a.c_str());
  }
  if (getenv("DORY_KINFO")) {  // the Dory kernels' static threadgroup memory and thread limits (no model needed)
    Gpu::get().compile_extra("Dory", kDoryKernelSource);
    for (const char* k : {"dory_prep", "dory_attn_partial", "dory_attn_reduce", "dory_attn_prefill", "dory_gate_norm",
                          "dory_swiglu_norm", "dory_slerp", "dory_loop_start", "dory_kv_store"}) {
      const Pipeline p = Gpu::get().pipeline(k);
      printf("%-20s tg_mem %6u B  max threads %4u\n", k, p.tg_mem, p.max_threads);
    }
    return 0;
  }
  if (ref.empty()) fatal("--ref is required");
  if (out.empty()) out = ref + "/kap";
  mkdir(out.c_str(), 0755);
  std::vector<std::string> names;
  if (DIR* d = opendir(ref.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 5 && n.compare(n.size() - 5, 5, ".json") == 0) {
        n.resize(n.size() - 5);
        if (only.empty() || only.find("," + n + ",") != std::string::npos) names.push_back(n);
      }
    }
    closedir(d);
  }
  std::sort(names.begin(), names.end());
  if (names.empty()) fatal("no references in %s", ref.c_str());

  Dory m(model, chunk, loops, kv);
  const int V = m.vocab();
  Gpu& g = Gpu::get();
  for (const std::string& name : names) {
    std::ifstream f(ref + "/" + name + ".json");
    const nlohmann::json j = nlohmann::json::parse(f);
    const std::vector<int> prompt = j["prompt_ids"].get<std::vector<int>>();
    const std::vector<int> gen = j["gen"].get<std::vector<int>>();
    const int n = int(prompt.size());
    const int steps = int(gen.size());
    auto s = m.new_seq(n + std::max(steps, greedy) + 64);
    s->loops = loops;
    SamplingParams sp;  // greedy
    auto prefill = [&](double* ms) {
      memcpy(s->tokens(), prompt.data(), prompt.size() * 4);
      const double t0 = now_ms();
      for (int p0 = 0; p0 < n; p0 += m.prefill_chunk()) {
        const int T = std::min(m.prefill_chunk(), n - p0);
        auto e = g.begin();
        m.encode_prefill(*e, *s, p0, T, sp, 0, p0 + T == n);
        e->commit_and_wait();
      }
      if (ms) *ms = now_ms() - t0;
    };
    if (tf) {
      std::vector<float> L;
      L.reserve(size_t(steps) * size_t(V));
      prefill(nullptr);
      L.insert(L.end(), m.logits(), m.logits() + V);
      for (int i = 0; i + 1 < steps; ++i) {
        s->tokens()[n + i] = gen[size_t(i)];
        auto e = g.begin();
        m.encode_step(*e, *s, n + i, sp, uint32_t(n + i), true);
        e->commit_and_wait();
        L.insert(L.end(), m.logits(), m.logits() + V);
      }
      write_npy(out + "/" + name + ".kap.npy", L, L.size() / size_t(V), size_t(V));
      fprintf(stderr, "%s: teacher-forced %d positions\n", name.c_str(), int(L.size() / size_t(V)));
    }
    if (greedy > 0) {
      double pf_ms = 0;
      prefill(&pf_ms);
      if (Encoder::profiling()) Encoder::profile_report(1);  // NI_PROFILE=1: the prefill's kernels, then decode's
      std::vector<int> toks{s->tokens()[n]};
      const double t1 = now_ms();
      for (int i = 0; i + 1 < greedy; ++i) {
        const int t = s->tokens()[n + i];
        if (std::find(m.eos_ids().begin(), m.eos_ids().end(), t) != m.eos_ids().end()) break;
        auto e = g.begin();
        m.encode_step(*e, *s, n + i, sp, uint32_t(n + i), true);
        e->commit_and_wait();
        toks.push_back(s->tokens()[n + i + 1]);
      }
      const double dec_ms = now_ms() - t1;
      if (Encoder::profiling() && toks.size() > 1) Encoder::profile_report(int(toks.size()) - 1);
      nlohmann::json o{{"name", name}, {"gen", toks}, {"prefill_ms", pf_ms}, {"decode_ms", dec_ms},
                       {"loops", m.loops_of(*s)}, {"kv", loop_kv_name(m.kv_mode())}};
      std::ofstream(out + "/" + name + ".greedy.json") << o.dump();
      int same = 0;
      while (same < int(std::min(toks.size(), gen.size())) && toks[size_t(same)] == gen[size_t(same)]) ++same;
      fprintf(stderr, "%s: greedy %zu tokens (prefill %d in %.0f ms, decode %.1f tok/s), equal to the reference for %d of %zu\n",
              name.c_str(), toks.size(), n, pf_ms, toks.size() > 1 ? (toks.size() - 1) / (dec_ms / 1000) : 0.0, same,
              gen.size());
    }
  }
  return 0;
}
