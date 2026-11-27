// Private qualification driver: real Flint generation with populated-cache growth.
// Each pass completes before reserve. This synchronous driver is not the server's pipelined throughput.
#include "bench/bench.h"
#include "model/flint.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace ni;
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream corpus_file("bench/corpus.txt");
  std::stringstream ss; ss << corpus_file.rdbuf();
  if (!corpus_file || ss.str().empty()) return 2;
  Tokenizer tok; tok.load(std::string(argv[1]) + "/tokenizer.json");
  const auto corpus = tok.encode(ss.str());
  if (corpus.empty()) return 2;
  bench::GpuLock lock;
  if (!bench::GpuLock::held_by_caller() && !lock.acquire()) return 1;
  Gpu::AllocationScope scope;
  Gpu& g = Gpu::get();
  const int chunk = 512, steps = 64;
  Flint m(argv[1], chunk);
  const SamplingParams greedy;
  for (int prompt : {1008, 4064}) {
    const auto prefix = std::string(argv[2]) + "/p" + std::to_string(prompt);
    std::ofstream inputs(prefix + ".input.i32", std::ios::binary);
    std::ofstream tokens(prefix + ".tokens.i32", std::ios::binary);
    std::ofstream logits(prefix + ".logits.f32", std::ios::binary);
    if (!inputs || !tokens || !logits) return 2;
    std::vector<int> ids; while (int(ids.size()) < prompt) ids.insert(ids.end(),corpus.begin(),corpus.end());
    ids.resize(size_t(prompt)); inputs.write(reinterpret_cast<const char*>(ids.data()), prompt * 4);
    auto s = m.new_seq(8192, 0); std::copy(ids.begin(), ids.end(), s->tokens());
    int growths=0, populated_growths=0; double growth_ms=0, prefill_ms=0, decode_ms=0;
    auto reserve = [&](int need, int populated) {
      const int old = s->cap; const double t = Encoder::host_ms();
      if (!m.reserve(*s, need)) fatal("reserve refused");
      const double elapsed = Encoder::host_ms() - t;
      if (s->cap != old) {
        ++growths; if (populated) ++populated_growths; growth_ms += elapsed;
        printf("{\"prompt\":%d,\"growth\":true,\"old\":%d,\"capacity\":%d,\"populated\":%d,\"ms\":%.6f}\n", prompt, old, s->cap, populated, elapsed);
        fflush(stdout);
      }
    };
    auto capture = [&](int pos) {
      for (int v=0;v<m.vocab();++v) if (!std::isfinite(m.logits()[v])) fatal("non-finite logit");
      logits.write(reinterpret_cast<const char*>(m.logits()), size_t(m.vocab())*4);
      tokens.write(reinterpret_cast<const char*>(s->tokens()+pos),4);
      if (!logits || !tokens) fatal("output write failed");
    };
    for (int start=0;start<prompt;start+=chunk) {
      const int n=std::min(chunk,prompt-start); const double t=Encoder::host_ms();
      reserve(start+n,start);
      auto e=m.grouped_passes()?g.begin_grouped():g.begin();
      m.encode_prefill(*e,*s,start,n,greedy,0,start+n==prompt);e->commit_and_wait();
      prefill_ms+=Encoder::host_ms()-t;
    }
    capture(prompt);
    for (int i=0;i<steps;++i) {
      const int pos=prompt+i; const double t=Encoder::host_ms(); reserve(pos+1,pos);
      auto e=m.grouped_passes()?g.begin_grouped():g.begin();
      m.encode_step(*e,*s,pos,greedy,uint32_t(pos),true); e->commit_and_wait();
      decode_ms+=Encoder::host_ms()-t; capture(pos+1);
    }
    if (!populated_growths) fatal("no populated-cache growth");
    printf("{\"prompt\":%d,\"generated_tokens\":%d,\"vocab\":%d,\"growths\":%d,\"populated_growths\":%d,\"growth_ms\":%.6f,\"prefill_ms\":%.6f,\"decode_ms\":%.6f,\"all_logits_finite\":true}\n",prompt,steps+1,m.vocab(),growths,populated_growths,growth_ms,prefill_ms,decode_ms);fflush(stdout);
    m.drop_kv(*s); g.release(s->tok);
  }
  return 0;
}
