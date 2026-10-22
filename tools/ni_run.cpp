// Runs a prompt (token ids) through the engine and generates tokens.
//   ni_run <model_dir> <ids: comma-separated> [n_gen=32] [temp=0] [--dump-logits file]
//          [--spec D] [--dflash <drafter dir> | auto | none] [--ignore-eos]
// Plain decoding unless --spec (MTP) or --dflash is given (the server defaults to DFlash2 instead).
// --dflash: DFlash2 speculative decoding: drafting pass, 8-row
// target verify with captures, on-GPU acceptance (greedy, or exact speculative sampling at temp > 0),
// ring commit of the kept rows; "auto" = DFlash2::find. NI_DFLASH=<dir>|auto does the same, NI_DFLASH=0 /
// --dflash none is off. The prompt rows are captured for the drafter in whichever prefill runs (chunks,
// --prefix/--split, --plan, --tbt, NI_ANE split or wavefront, a rerun after a failed split).
// --ignore-eos: keep generating past end-of-sequence tokens (benchmarks, equality tests).
// --ref <file with an "ids:" line> (with --dflash, T = 0): at the first token that differs from that
// reference (plain greedy), prints the target's logits of both tokens three ways (SPEC §12 G2): the
// DFlash verify row that produced it, plain token-by-token decoding, one multi-row prefill of the prefix.
// NI_CYCLE_TRACE=1 (DFlash2, double-buffered cycles): the mean host encode time of a verify, the GPU times of the
// drafting half and the verify, the GPU idle times between them, and the host wake-up latency after a verify.
// Prints generated ids, prompt/decode speeds; optionally dumps the logits of the first
// generated position (raw float32) for comparison against a reference implementation.
#include "metal/gpu.h"
#include "ane/ffn_split.h"
#include "model/dflash2.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <mach/mach.h>

#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

using namespace ni;

static double footprint_mb() {
  task_vm_info_data_t vi;
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&vi, &n) != KERN_SUCCESS) return -1;
  return vi.phys_footprint / 1e6;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: ni_run <model_dir> <ids> [n_gen] [temp] [--dump-logits f]\n");
    return 1;
  }
  std::string dir = argv[1];
  std::vector<int> ids;
  {
    std::string arg = argv[2];
    if (!arg.empty() && arg[0] == '@') {  // @file with comma-separated ids
      std::ifstream f(arg.substr(1));
      arg.assign(std::istreambuf_iterator<char>(f), {});
    }
    std::stringstream ss(arg);
    std::string t;
    while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
  }
  int n_gen = argc > 3 ? atoi(argv[3]) : 32;
  float temp = argc > 4 ? float(atof(argv[4])) : 0.0f;
  const char* dump = nullptr;
  const char* dump_all = nullptr;  // logits of the prompt's last position and of every decode step
  std::vector<int> ref;  // --ref: reference output ids
  bool tbt = false, ignore_eos = false;
  int spec = 0;
  int pf_reps = 1;  // --pf-reps R: time the prefill R times (state reset in between), report each
  // --prefix P: prefill ids[0, P) once untimed and checkpoint it; the timed prefill then restores that
  // checkpoint and runs only ids[P, n) (a cached-turn suffix). --split B (P < B < n): a turn boundary
  // inside the suffix, checkpointed like the server does (split pass, or in-pass with NI_SHORTPF);
  // afterwards the in-pass checkpoint is compared with a separate prefill of ids[P, B).
  int prefix = 0, split = 0;
  int pf_gap = 0;  // --pf-gap MS: idle the GPU for MS ms before each timed prefill rep (server turn gaps)
  // --plan FILE: the prompt state built by a sequence of server-like passes (a multi-turn conversation as
  // the server ran it) instead of one prefill. Lines: "tokens <ids file>" loads the token buffer used by
  // the following passes; "pass A B [S]" prefills [A, B) in prefill_chunk chunks with a checkpoint at S
  // (inside the pass where the model can, as the server does); "restore S" restores that checkpoint.
  // The last pass must end at the prompt length; decoding then continues from the given ids.
  const char* plan = nullptr;
  // --kv-grow: the KV cache starts empty and grows before each prefill chunk and decode step / cycle (the
  // engine's path; decode then runs one step at a time); outputs must equal the fully allocated cache's
  bool kv_grow = false;
  for (int i = 1; i < argc; ++i) {
    if (i + 1 < argc && !strcmp(argv[i], "--dump-logits")) dump = argv[i + 1];
    if (i + 1 < argc && !strcmp(argv[i], "--dump-all-logits")) dump_all = argv[i + 1];
    if (i + 1 < argc && !strcmp(argv[i], "--ref")) {
      std::ifstream f(argv[i + 1]);
      std::string all((std::istreambuf_iterator<char>(f)), {}), line;
      const size_t at = all.find("ids:");
      if (at == std::string::npos) fatal("--ref %s: no ids: line", argv[i + 1]);
      std::stringstream ss(all.substr(at + 4, all.find('\n', at) - at - 4));
      int t;
      while (ss >> t) ref.push_back(t);
    }
    if (!strcmp(argv[i], "--tbt")) tbt = true;
    if (!strcmp(argv[i], "--ignore-eos")) ignore_eos = true;
    if (i + 1 < argc && !strcmp(argv[i], "--spec")) spec = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--pf-reps")) pf_reps = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--prefix")) prefix = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--split")) split = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--pf-gap")) pf_gap = atoi(argv[i + 1]);
    if (i + 1 < argc && !strcmp(argv[i], "--plan")) plan = argv[i + 1];
    if (!strcmp(argv[i], "--kv-grow")) kv_grow = true;
  }
  std::string dflash_dir = getenv("NI_DFLASH") ? getenv("NI_DFLASH") : "";
  for (int i = 1; i + 1 < argc; ++i)
    if (!strcmp(argv[i], "--dflash")) dflash_dir = argv[i + 1];
  if (dflash_dir == "0" || dflash_dir == "none" || dflash_dir == "off") dflash_dir.clear();
  if (dflash_dir == "1" || dflash_dir == "on") dflash_dir = "auto";

  Gpu& g = Gpu::get();
  std::unique_ptr<Qwen35> mp = load_qwen(dir, getenv("NI_CHUNK") ? atoi(getenv("NI_CHUNK")) : 2048);
  Qwen35& m = *mp;
  if (kv_grow && !dflash_dir.empty()) fatal("--kv-grow covers the plain, MTP and teacher-forced paths, not --dflash");
  auto seq = m.new_seq(std::max(8192, int(ids.size()) + n_gen + 64), kv_grow ? 0 : -1);
  auto grow = [&](int need) {
    if (kv_grow && !m.reserve(*seq, need)) fatal("--kv-grow: cannot grow the KV cache to %d positions", need);
  };
  Seq& s = *seq;
  SamplingParams sp;
  sp.temperature = temp;
  sp.top_p = 0.95f;
  sp.top_k = 20;
  sp.seed = getenv("NI_SEED") && *getenv("NI_SEED") ? uint32_t(strtoul(getenv("NI_SEED"), nullptr, 10)) : 42u;  // sampled runs

  int n = int(ids.size());
  for (int i = 0; i < n; ++i) s.tokens()[i] = ids[i];

  // Opt-in DFlash2 block drafter; the prompt rows are captured for the drafter's context during the
  // prefill below.
  std::unique_ptr<DFlash2> dfl;
  std::unique_ptr<DFlashSeq> dseq;
  if (dflash_dir == "auto") {
    dflash_dir = DFlash2::find(dir, m.config());
    if (dflash_dir.empty()) fatal("--dflash auto: no DFlash2 drafter found for %s", dir.c_str());
  }
  if (!dflash_dir.empty()) {
    dfl = std::make_unique<DFlash2>(m, dflash_dir);
    DFlash2::prewarm_pipelines();
    dseq = dfl->new_seq();
    dfl->begin_prefill(int(ids.size()));
  }

  int last_chunk = 0;
  double t0 = now_ms();
  if (Encoder::profiling()) fprintf(stderr, "(profiling: prompt timings not meaningful)\n");
  if (plan) {
    if (spec) { fprintf(stderr, "--plan does not run the MTP history (use it without --spec)\n"); return 1; }
    auto read_ids = [](const std::string& path) {
      std::ifstream f(path);
      std::string a((std::istreambuf_iterator<char>(f)), {}), t;
      std::stringstream ss(a);
      std::vector<int> v;
      while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
      return v;
    };
    std::ifstream pf(plan);
    std::string line;
    std::vector<std::pair<int, std::unique_ptr<Checkpoint>>> cks;
    const int C = m.prefill_chunk();
    int end = -1;
    while (std::getline(pf, line)) {
      std::stringstream ls(line);
      std::string op;
      ls >> op;
      if (op.empty() || op[0] == '#') continue;
      if (op == "tokens") {
        std::string f;
        ls >> f;
        std::vector<int> v = read_ids(f);
        if (int(v.size()) + 1 >= s.limit) { fprintf(stderr, "plan: %s too long\n", f.c_str()); return 1; }
        std::copy(v.begin(), v.end(), s.tokens());
      } else if (op == "restore") {
        int at = -1;
        ls >> at;
        Checkpoint* c = nullptr;
        for (auto& k : cks)
          if (k.first == at) c = k.second.get();
        if (!c) { fprintf(stderr, "plan: no checkpoint at %d\n", at); return 1; }
        auto e = g.begin();
        m.restore(*e, s, *c);
        e->commit_and_wait();
      } else if (op == "pass") {
        int a = -1, b = -1, at = -1;
        ls >> a >> b;
        if (!(ls >> at)) at = -1;
        Checkpoint* ck = nullptr;
        if (at > a && at <= b) {
          for (auto& k : cks)
            if (k.first == at) ck = k.second.get();
          if (!ck) {
            cks.emplace_back(at, std::make_unique<Checkpoint>(m.new_checkpoint()));
            ck = cks.back().second.get();
          }
        }
        int p = a;
        auto e = g.begin();
        while (p < b) {
          int T = std::min(C, b - p);
          Checkpoint* snap = nullptr;
          if (ck && at > p && at < p + T) {
            if (m.can_snapshot() && !m.ane_splits(T)) snap = ck;
            else T = at - p;
          }
          m.encode_prefill(*e, s, p, T, sp, 0, b == n && p + T == b, snap, snap ? at - p : 0);
          last_chunk = p;
          p += T;
          if (ck && p == at) m.save(*e, s, *ck, at);
          e->commit_and_wait();
          if (p < b) e = g.begin();
        }
        end = b;
        fprintf(stderr, "plan: pass [%d, %d)%s\n", a, b, ck ? (" checkpoint at " + std::to_string(at)).c_str() : "");
      } else {
        fprintf(stderr, "plan: unknown op %s\n", op.c_str());
        return 1;
      }
    }
    if (end != n) { fprintf(stderr, "plan: the last pass ends at %d, not at the prompt length %d\n", end, n); return 1; }
    std::copy(ids.begin(), ids.end(), s.tokens());  // tokens[n] holds the sampled first token
  } else if (tbt && dump_all) {
    // teacher-forced: every prompt position's logits through the decode path (one step per command
    // buffer; the sampled token is replaced by the prompt's next token), then decoding continues
    FILE* f = fopen(dump_all, "wb");
    for (int q = 0; q < n; ++q) {
      grow(q + 2);
      auto e = g.begin();
      m.encode_step(*e, s, q, sp, 0, true);
      e->commit_and_wait();
      fwrite(m.logits(), 4, m.config().vocab, f);  // row q predicts token q + 1
      if (q + 1 < n) s.tokens()[q + 1] = ids[q + 1];
    }
    fclose(f);
    dump_all = nullptr;  // the decode loop below appends nothing
  } else if (tbt) {
    // token by token through the decode path (reference for the batched prefill)
    for (int p = 0; p < n; p += 32) {
      grow(std::min(n, p + 32) + 1);
      auto e = g.begin();
      for (int q = p; q < std::min(n, p + 32); ++q) m.encode_step(*e, s, q, sp, 0, q == n - 1);
      e->commit_and_wait();
    }
  } else {
    const int C = m.prefill_chunk();
    Checkpoint ck0 = m.new_checkpoint(), ck1 = m.new_checkpoint();
    std::unique_ptr<Checkpoint> ane_ck;
    if (prefix > 0) {
      if (prefix >= n) { fprintf(stderr, "--prefix must be < prompt length\n"); return 1; }
      for (int p = 0; p < prefix; p += C) {
        int T = std::min(C, prefix - p);
        grow(p + T + 1);
        auto e = g.begin();
        m.encode_prefill(*e, s, p, T, sp, 0, false);
        if (spec && m.has_mtp()) m.encode_mtp_prefill(*e, s, p, T, 0);
        if (p + T == prefix) m.save(*e, s, ck0, prefix);
        e->commit_and_wait();
      }
      if (Encoder::profiling()) Encoder::profile_report(1);
      fprintf(stderr, "prefix %d tok prefilled; timing ids[%d, %d)\n", prefix, prefix, n);
      t0 = now_ms();
    }
    if (split && (split <= prefix || split >= n)) { fprintf(stderr, "--split must be in (prefix, n)\n"); return 1; }
    // one timed prefill of ids[prefix, n) as the server runs it (checkpoint at --split); returns GPU ms
    auto prefill_pass = [&]() {
      double gpu = 0;
      int p = prefix;
      auto e = g.begin();
      if (prefix) m.restore(*e, s, ck0);
      else m.reset(s);
      while (p < n) {
        int T = std::min(C, n - p);
        Checkpoint* snap = nullptr;
        if (split > p && split < p + T) {
          // an in-pass checkpoint and the NI_ANE split do not mix: cut such a chunk at the boundary
          if (m.can_snapshot() && !m.ane_splits(T)) snap = &ck1;
          else T = split - p;
        }
        // NI_ANE=1: a chunk that runs the GPU + ANE FFN split first saves the recurrent state, so
        // that it can run again on the GPU alone if the split fails (KV rows are simply rewritten).
        const bool ane = !snap && m.ane_splits(T);
        if (ane) {
          if (!ane_ck) ane_ck = std::make_unique<Checkpoint>(m.new_checkpoint());
          m.save(*e, s, *ane_ck, p);
        }
        grow(p + T + 1);  // nothing in flight: every chunk is waited for
        m.encode_prefill(*e, s, p, T, sp, 0, p + T == n, snap, snap ? split - p : 0);
        // MTP history rows i in [p, min(p+T, n-1)): pairs (h_i, t_{i+1})
        if (spec && m.has_mtp()) m.encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
        if (ane) {
          e->commit_and_wait();
          gpu += e->gpu_ms();
          if (!m.ane_finish_chunk()) {
            e = g.begin();
            m.restore(*e, s, *ane_ck);
            m.encode_prefill(*e, s, p, T, sp, 0, p + T == n);
            if (spec && m.has_mtp()) m.encode_mtp_prefill(*e, s, p, std::min(p + T, n - 1) - p, 0);
            e->commit_and_wait();
            gpu += e->gpu_ms();
          }
          e = g.begin();
        }
        last_chunk = p;
        p += T;
        if (p == split) m.save(*e, s, ck1, split);
        e->commit_and_wait();
        gpu += e->gpu_ms();
        if (p < n) e = g.begin();
      }
      return gpu;
    };
    for (int rep = 0; rep < pf_reps; ++rep) {
      if (pf_gap > 0) std::this_thread::sleep_for(std::chrono::milliseconds(pf_gap));
      if (rep || pf_gap > 0) t0 = now_ms();
      const double gpu = prefill_pass();
      if (pf_reps > 1)
        fprintf(stderr, "prefill rep %d: %d tok in %.2f ms (GPU %.2f ms)\n", rep, n - prefix, now_ms() - t0, gpu);
      if (pf_reps > 1 && Encoder::profiling()) Encoder::profile_report(1);
    }
    if (split && m.can_snapshot()) {
      // reference: the state after a separate prefill of ids[prefix, split)
      std::vector<std::vector<float>> conv, rec;
      for (size_t i = 0; i < ck1.rec.size(); ++i) {
        conv.emplace_back(ck1.conv[i] ? std::vector<float>(ck1.conv[i].as<float>(), ck1.conv[i].as<float>() + ck1.conv[i].size / 4) : std::vector<float>());
        rec.emplace_back(ck1.rec[i] ? std::vector<float>(ck1.rec[i].as<float>(), ck1.rec[i].as<float>() + ck1.rec[i].size / 4) : std::vector<float>());
      }
      auto e = g.begin();
      if (prefix) m.restore(*e, s, ck0);
      else m.reset(s);
      for (int p = prefix; p < split; p += C) m.encode_prefill(*e, s, p, std::min(C, split - p), sp, 0, false);
      e->commit_and_wait();
      double dc = 0, dr = 0, mr = 0;
      for (size_t i = 0; i < rec.size(); ++i) {
        if (rec[i].empty()) continue;
        for (size_t j = 0; j < conv[i].size(); ++j) dc = std::max(dc, double(std::fabs(conv[i][j] - s.conv[i].as<float>()[j])));
        for (size_t j = 0; j < rec[i].size(); ++j) {
          dr = std::max(dr, double(std::fabs(rec[i][j] - s.rec[i].as<float>()[j])));
          mr = std::max(mr, double(std::fabs(s.rec[i].as<float>()[j])));
        }
      }
      fprintf(stderr, "snapshot at %d vs separate prefill: conv max|d| %.3g, state max|d| %.3g (max|state| %.3g)\n",
              split, dc, dr, mr);
      // the reference prefill replaced the live state: run the timed pass again before decoding
      const double tk = now_ms();
      prefill_pass();
      t0 += now_ms() - tk;
    }
  }
  if (dfl) {
    dfl->end_prefill();
    // NI_DFLASH_DUMP_CAPT=<file>: the prompt captures the drafter's ring is built from (raw bf16, rows =
    // the last min(n, 2047) prompt positions, n_target_layers x hidden columns), e.g. to compare the
    // Neural Engine prefill schedules with NI_ANE=0
    if (const char* fn = getenv("NI_DFLASH_DUMP_CAPT")) {
      const int rows = std::min(n, int(DFlash2::kCtxKeep));
      const size_t nH = dfl->config().target_layers.size() * size_t(dfl->config().H);
      FILE* f = fopen(fn, "wb");
      if (!f) fatal("cannot write %s", fn);
      fwrite(dfl->prompt_captures().ptr(), 2, size_t(rows) * nH, f);
      fclose(f);
      fprintf(stderr, "dflash: prompt captures of %d rows x %zu (bf16) -> %s\n", rows, nH, fn);
    }
    auto e = g.begin();
    dfl->encode_prefill_commit(*e, *dseq);
    e->commit_and_wait();
  }
  double t_prompt = now_ms() - t0;
  // streamed experts (Qwen3.8-Flash-Next Q2_0 packs, src/model/expert_store.h): the store's turns of the prompt, then
  // (store_stats at the end) of the decode
  q20::ExpertStore* store = nullptr;
  if (auto* q4 = dynamic_cast<Qwen4Exp*>(&m)) store = q4->expert_store();
  auto store_stats = [&](const char* what) {
    if (!store || store->complete()) return;
    const q20::StoreStats st = store->stats();
    const double t = double(std::max<uint64_t>(st.handshakes, 1));
    fprintf(stderr, "expert store, %s: %llu turns, %llu experts looked up: %.1f%% in memory, %.1f%% read ahead and "
            "waited for, %.1f%% read at the turn; %llu read ahead (%llu routed), %.2f GB read; GPU stalled %.3f ms a "
            "turn (%.3f waiting for reads)\n",
            what, (unsigned long long)st.handshakes, (unsigned long long)st.lookups,
            100.0 * st.hits / std::max<double>(1, st.lookups), 100.0 * st.late / std::max<double>(1, st.lookups),
            100.0 * st.misses / std::max<double>(1, st.lookups), (unsigned long long)st.prefetches,
            (unsigned long long)st.prefetch_used, st.bytes / 1e9, st.stall_ms / t, st.read_wait_ms / t);
    store->reset_stats();
  };
  store_stats("prompt");
  FILE* fall = dump_all ? fopen(dump_all, "wb") : nullptr;
  if (fall) fwrite(m.logits(), 4, m.config().vocab, fall);
  if (m.ane_split())
    fprintf(stderr, "ane: system wired memory +%.0f MB at the program load, +%.0f MB after the first split chunk "
            "(split surfaces %.0f MB)\n", m.ane_split()->wired_at_load(), m.ane_split()->wired_after_run(),
            m.ane_split()->metal_bytes() / 1e6);
  if (m.ane_split() || env_on("NI_MEM")) {
    const auto& st = m.ane_stats();
    fprintf(stderr, "ane: %d split chunks, %d evaluations, ANE %.0f ms (%.1f ms each), reruns %d%s; memory: Metal %.0f MB, "
            "process footprint %.0f MB; thermal %s\n", st.chunks, st.evals, st.ane_ms,
            st.evals ? st.ane_ms / st.evals : 0.0, st.reruns,
            m.ane_split() && m.ane_split()->retired() ? " (split stopped)" : "", g.allocated_bytes() / 1e6,
            footprint_mb(), Gpu::thermal_state());
  }
  if (dump) {
    FILE* f = fopen(dump, "wb");
    fwrite(m.logits(), 4, m.config().vocab, f);
    fclose(f);
  }
  std::vector<int> out;
  out.push_back(s.tokens()[n]);

  if (dfl) {
    // NI_DFLASH_VERIFY=k: verify only the first k of the 7 proposals (lossless; fewer verify rows)
    const int K = dfl->verify_depth();
    const bool trace = getenv("NI_TRACE") != nullptr;
    // NI_DFLASH_CONF=<file>: per cycle "a c_0 .. c_6", c_p = softmax over the 16 candidates of the walk's
    // scores S_p at the chosen candidate (the drafter's confidence; calibration data for a dynamic depth)
    FILE* conf = getenv("NI_DFLASH_CONF") ? fopen(getenv("NI_DFLASH_CONF"), "a") : nullptr;
    // NI_DFLASH_LAT=<file>: per cycle the drafter's whole lattice and the verify's rows (binary; tools/tree_sim.py
    // reads it, layout there): anchor P, accepted a, the verify's sampled ids, candidates / scores / edges / walk, and
    // per verify row the target's top-16 logits, its log-sum-exp and its logits of the 16 candidates of that position.
    // NI_DFLASH_STEP1=1: every cycle emits only its first token (the verify's row-0 sample = plain decoding), so the
    // lattice is recorded at every position of the plain output (a tree simulator can then replay any policy).
    FILE* lat = getenv("NI_DFLASH_LAT") ? fopen(getenv("NI_DFLASH_LAT"), "wb") : nullptr;
    const bool step1 = env_on("NI_DFLASH_STEP1");
    if (lat || step1) dfl->set_tree(false);  // the recording is the walk's chain and its lattice
    if (lat) {
      const int32_t hdr[8] = {0x3154414c /* "LAT1" */, 1, n, int32_t(std::lround(sp.temperature * 1000)), K,
                              dfl->config().v1 ? 1 : 0, step1 ? 1 : 0, m.config().vocab};
      fwrite(hdr, 4, 8, lat);
    }
    double ts = now_ms();
    int P = n, m_prev = 0, cycles = 0, emitted = 0;
    int div_at = -1, div_tok[2] = {0, 0}, div_row = 0, div_P = 0;
    long verified_sum = 0;
    float div_logit[2] = {0, 0};
    std::vector<int> acc_hist(K + 1, 0), reached(K, 0), acc_pos(K, 0);
    double gpu_sum = 0, wall_sum = 0;
    std::vector<double> draft_ms, draft_ab[2];  // GPU time of the drafting command buffers (modes 2, 3)
    double gap_sum = 0;
    bool stop2 = false;
    if (Encoder::profiling()) Encoder::profile_report(1);
    // NI_DFLASH_CB: 1 = the whole cycle in one command buffer; 2 = its drafting and verify halves as two
    // command buffers committed back to back (the verify is encoded while the draft runs); 3 (default)
    // = double-buffered: the next cycle's drafting half (GPU-driven: anchor and ring rows from this
    // cycle's acceptance) is committed right behind each verify, before the host waits for it.
    // NI_TRACE / NI_DFLASH_CONF read drafter buffers after each cycle, so they run mode 2; NI_PROFILE
    // (one command buffer per dispatch) runs mode 1.
    int mode = getenv("NI_DFLASH_CB") && *getenv("NI_DFLASH_CB") ? atoi(getenv("NI_DFLASH_CB")) : 3;
    if ((trace || conf || lat || step1) && mode == 3) mode = 2;
    if (Encoder::profiling()) mode = 1;
    // NI_AB=key=value,... (V8Knobs keys): odd cycles run with those verify-kernel knobs, even cycles with the
    // defaults, so two kernel variants are timed under the same GPU conditions (GPU ms per cycle by parity;
    // the drafting half of a cycle is encoded with that cycle's knobs too)
    const V8Knobs knob_a = v8_knobs();
    const bool ab = getenv("NI_AB") && *getenv("NI_AB");
    const V8Knobs knob_b = ab ? v8_knobs_parse(getenv("NI_AB"), knob_a) : knob_a;
    std::vector<double> ab_ms[2];
    auto knobs_for = [&](int cycle) { v8_knobs() = (ab && (cycle & 1)) ? knob_b : knob_a; };
    knobs_for(0);
    std::unique_ptr<Encoder> next_draft;  // mode 3: the committed drafting half of the coming cycle
    double tr[7] = {0, 0, 0, 0, 0, 0, 0}, prev_end = 0;
    int ntr = 0;
    if (mode == 3) {
      dfl->gpu_state_reset(P);  // the GPU is idle here
      next_draft = g.begin();
      dfl->encode_cycle_draft_gpu(*next_draft, s, *dseq, P, sp);
      next_draft->commit();
    }
    while (int(out.size()) < n_gen && !stop2) {
      if (P + DFlash2::kB + 2 >= s.cap) break;
      const double tc0 = now_ms(), tc0h = Encoder::host_ms();
      double t_commit = tc0h;
      std::unique_ptr<Encoder> e1;
      auto e = g.begin();
      if (mode == 1) {
        dfl->encode_cycle(*e, s, *dseq, P, m_prev, K, sp);
        e->commit();
      } else if (mode == 2) {
        e1 = std::move(e);
        dfl->encode_cycle_draft(*e1, s, *dseq, P, m_prev, sp);
        e1->commit();
        e = g.begin();
        dfl->encode_cycle_verify(*e, s, P, K, sp);
        e->commit();
      } else {
        if (!next_draft) break;  // no room for another drafting pass
        e1 = std::move(next_draft);
        dfl->encode_cycle_verify(*e, s, P, K, sp);
        e->commit();
        t_commit = Encoder::host_ms();
        // the next anchor is at most P + 8: commit its drafting half now (it waits on this verify)
        if (P + 8 + DFlash2::kB + 2 < s.cap) {
          knobs_for(cycles + 1);
          next_draft = g.begin();
          dfl->encode_cycle_draft_gpu(*next_draft, s, *dseq, P + 8, sp);
          next_draft->commit();
        }
      }
      e->wait();
      if (mode == 3 && cycles > 0) {  // NI_CYCLE_TRACE: where a cycle's wall time goes (host clock)
        const double h0 = Encoder::host_ms();
        tr[0] += t_commit - tc0h;                                  // host: encode the verify
        tr[1] += e1->gpu_end_host_ms() - e1->gpu_start_host_ms();  // GPU: drafting half
        tr[2] += e->gpu_start_host_ms() - e1->gpu_end_host_ms();   // GPU idle: draft end -> verify start
        tr[3] += e->gpu_end_host_ms() - e->gpu_start_host_ms();    // GPU: verify
        tr[4] += h0 - e->gpu_end_host_ms();                        // host wake-up after the verify
        tr[5] += e->gpu_start_host_ms() - t_commit;                // verify commit -> its GPU start
        tr[6] += e1->gpu_start_host_ms() - prev_end;                // GPU idle: last verify end -> drafting half start
        ++ntr;
      }
      prev_end = e->gpu_end_host_ms();
      const double wall = now_ms() - tc0;
      const double gms = e->gpu_ms() + (e1 ? e1->gpu_ms() : 0.0);
      gpu_sum += gms;
      wall_sum += wall;
      if (e1 && cycles > 0) {
        draft_ms.push_back(e1->gpu_ms());
        draft_ab[cycles & 1].push_back(e1->gpu_ms());
        gap_sum += e->gpu_start_host_ms() - e1->gpu_end_host_ms();  // GPU idle between the drafting pass and the verify
      }
      if (cycles > 0) ab_ms[cycles & 1].push_back(gms);  // the first cycle includes the first drafting half
      dfl->observe(gms);  // KFlash: the budget reads the cycle's outcome and GPU time (drafting half + verify)
      const int a_cycle = dfl->accepted();
      if (lat) {
        const int V = m.config().vocab;
        const int* tkl = s.tokens();
        const int32_t head[4] = {P, a_cycle, dfl->verified(K), tkl[P]};
        fwrite(head, 4, 4, lat);
        fwrite(m.samples(), 4, 8, lat);
        fwrite(dfl->cand_ids(), 4, 7 * 16, lat);
        fwrite(dfl->cand_vals(), 4, 7 * 16, lat);
        fwrite(dfl->edges(), 4, 7 * 16 * 16, lat);
        fwrite(dfl->walk_scores(), 4, 7 * 16, lat);
        fwrite(dfl->walk_k(), 4, 7, lat);
        fwrite(dfl->qprob(), 4, 7 * 16, lat);
        for (int r = 0; r < 8; ++r) {  // the target's row r: top-16 (ids, logits), log-sum-exp, the candidates' logits
          const float* l = m.logits() + size_t(r) * V;
          float mx = l[0];
          for (int v = 1; v < V; ++v) mx = std::max(mx, l[v]);
          double z = 0;
          for (int v = 0; v < V; ++v) z += std::exp(double(l[v] - mx));
          std::vector<std::pair<float, int>> top;
          top.reserve(17);
          for (int v = 0; v < V; ++v) {
            if (top.size() == 16 && l[v] <= top.back().first) continue;
            auto it = std::find_if(top.begin(), top.end(), [&](const auto& x) { return l[v] > x.first; });
            top.insert(it, {l[v], v});
            if (top.size() > 16) top.pop_back();
          }
          int32_t ti[16];
          float tv[16], cl[16];
          for (int k = 0; k < 16; ++k) {
            ti[k] = top[k].second;
            tv[k] = top[k].first;
            cl[k] = r < 7 ? l[dfl->cand_ids()[r * 16 + k]] : 0.0f;
          }
          const float lse = float(mx + std::log(z));
          fwrite(ti, 4, 16, lat);
          fwrite(tv, 4, 16, lat);
          fwrite(&lse, 4, 1, lat);
          fwrite(cl, 4, 16, lat);
        }
      }
      // NI_DFLASH_STEP1: keep only the first token, the verify's row-0 sample (the accepted d_0 at T > 0 is a
      // speculative-sampling draw, not the plain sampler's)
      if (step1) s.tokens()[P + 1] = m.samples()[0];
      const int a = step1 ? 0 : a_cycle;
      const int* tk = s.tokens();
      if (trace)
        fprintf(stderr, "P=%d a=%d chain=%d,%d,%d,%d,%d,%d,%d next=%d wall=%.2fms gpu=%.2fms\n", P, a, tk[P + 1],
                tk[P + 2], tk[P + 3], tk[P + 4], tk[P + 5], tk[P + 6], tk[P + 7], tk[P + 1 + a], wall, gms);
      if (conf) {
        fprintf(conf, "%d", a);
        for (int p = 0; p < 7; ++p) {
          const float* S = dfl->walk_scores() + p * 16;
          float mx = S[0], z = 0;
          for (int k = 1; k < 16; ++k) mx = std::max(mx, S[k]);
          for (int k = 0; k < 16; ++k) z += std::exp(S[k] - mx);
          fprintf(conf, " %.4f", std::exp(S[dfl->walk_k()[p]] - mx) / z);
        }
        // sampled runs: the walk's q of each chosen proposal (what the T > 0 depth rule reads), and K verified
        if (sp.temperature > 0)
          for (int p = 0; p < 7; ++p) fprintf(conf, " %.4f", dfl->qprob()[p * 16 + dfl->walk_k()[p]]);
        fprintf(conf, " | %d |", dfl->verified(K));
        // per proposal, features of its whole distribution (fixed before it is sampled, so a depth rule may
        // read them at T > 0): max and sum of squares of the softmax of S_p, then of the walk's q
        for (int p = 0; p < 7; ++p) {
          const float* S = dfl->walk_scores() + p * 16;
          float mx = S[0], z = 0, cm = 0, c2 = 0, qm = 0, q2 = 0;
          for (int k = 1; k < 16; ++k) mx = std::max(mx, S[k]);
          for (int k = 0; k < 16; ++k) z += std::exp(S[k] - mx);
          for (int k = 0; k < 16; ++k) {
            const float c = std::exp(S[k] - mx) / z, q = dfl->qprob()[p * 16 + k];
            cm = std::max(cm, c); c2 += c * c; qm = std::max(qm, q); q2 += q * q;
          }
          fprintf(conf, " %.4f %.4f %.4f %.4f", cm, c2, qm, q2);
        }
        fprintf(conf, "\n");
      }
      int mm = 0;  // tokens emitted = rows committed this cycle
      for (int i = 1; i <= a + 1 && int(out.size()) < n_gen; ++i) {
        if (!ref.empty() && div_at < 0 && out.size() < ref.size() && tk[P + i] != ref[out.size()]) {
          div_at = int(out.size());
          div_tok[0] = ref[out.size()];
          div_tok[1] = tk[P + i];
          const int V = m.config().vocab;
          // verify row i-1 predicts position P+i (a tree: the row of the accepted path's (i-1)-th node)
          div_row = dfl->tree() ? s.pend_path.as<int>()[i - 1] : i - 1;
          const float* row = m.logits() + size_t(div_row) * V;
          div_logit[0] = row[div_tok[0]];
          div_logit[1] = row[div_tok[1]];
          div_P = P;
        }
        out.push_back(tk[P + i]);
        ++mm;
        if (!ignore_eos)
          for (int eos : m.config().eos)
            if (tk[P + i] == eos) stop2 = true;
        if (stop2) break;
      }
      const int kv = dfl->verified(K);  // proposals this cycle verified
      verified_sum += kv;
      for (int i = 0; i <= std::min(a, kv - 1); ++i) {
        reached[i]++;
        if (i < a) acc_pos[i]++;
      }
      m.commit_rows(s, mm);
      m_prev = mm;
      emitted += mm;
      ++cycles;
      acc_hist[a]++;
      P += mm;
      if (mode == 3) dseq->ctx = P;  // the GPU committed the kept rows to the ring
      if (mode != 3) knobs_for(cycles);
    }
    v8_knobs() = knob_a;
    if (next_draft) next_draft->wait();
    double td = now_ms() - ts;
    if (conf) fclose(conf);
    if (lat) fclose(lat);
    if (Encoder::profiling()) Encoder::profile_report(cycles);
    printf("ids:");
    for (int t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr,
            "prompt %d tok in %.0f ms (%.1f tok/s); dflash decode (verify %d of 7, %s): %zu tok in %.0f ms (%.1f tok/s), "
            "%d cycles, %.3f tok/cycle, %.2f ms GPU/cycle, %.2f ms wall/cycle (mode %d), hist",
            n, t_prompt, n / (t_prompt / 1000), K, sp.temperature > 0 ? "sampled" : "greedy", out.size() - 1, td,
            (out.size() - 1) / (td / 1000), cycles, double(emitted) / std::max(cycles, 1), gpu_sum / std::max(cycles, 1),
            td / std::max(cycles, 1), mode);
    for (int a = 0; a <= K; ++a) fprintf(stderr, " %d", acc_hist[a]);
    fprintf(stderr, "; %.2f proposals verified per cycle; acceptance by position", double(verified_sum) / std::max(cycles, 1));
    for (int i = 0; i < K; ++i) fprintf(stderr, " %.2f", reached[i] ? double(acc_pos[i]) / reached[i] : 0.0);
    fprintf(stderr, "\n");
    if (dfl->tree() && dfl->budget()) fprintf(stderr, "%s\n", dfl->budget()->summary().c_str());
    if (getenv("NI_CYCLE_TRACE") && ntr)
      fprintf(stderr, "cycle trace (ms, mean of %d): host encodes the verify %.3f | GPU idle verify->draft %.3f, draft %.3f, "
              "idle draft->verify %.3f, verify %.3f | verify commit->start %.3f, host wake-up %.3f\n", ntr, tr[0] / ntr,
              tr[6] / ntr, tr[1] / ntr, tr[2] / ntr, tr[3] / ntr, tr[5] / ntr, tr[4] / ntr);
    if (!draft_ms.empty()) {
      std::vector<double> v = draft_ms;
      std::sort(v.begin(), v.end());
      double mean = 0;
      for (double x : v) mean += x;
      fprintf(stderr, "drafting pass: GPU ms/cycle mean %.3f median %.3f p10 %.3f p90 %.3f (%zu cycles); GPU idle before "
              "the verify %.3f ms/cycle\n", mean / v.size(), v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10], v.size(),
              gap_sum / v.size());
      if (ab) {
        double mab[2] = {0, 0};
        for (int k = 0; k < 2; ++k) {
          for (double x : draft_ab[k]) mab[k] += x;
          mab[k] /= std::max<size_t>(1, draft_ab[k].size());
        }
        fprintf(stderr, "drafting pass AB: A mean %.3f | B mean %.3f | B - A %+.3f ms\n", mab[0], mab[1], mab[1] - mab[0]);
      }
    }
    if (ab) {
      double med[2], mean[2];
      for (int k = 0; k < 2; ++k) {
        std::vector<double> v = ab_ms[k];
        std::sort(v.begin(), v.end());
        med[k] = v.empty() ? 0 : v[v.size() / 2];
        mean[k] = 0;
        for (double x : v) mean[k] += x;
        mean[k] /= std::max<size_t>(1, v.size());
      }
      fprintf(stderr, "AB [%s]: GPU ms/cycle A (default) mean %.3f median %.3f (%zu cycles) | B mean %.3f median %.3f (%zu) | "
              "B - A mean %+.3f median %+.3f\n", getenv("NI_AB"), mean[0], med[0], ab_ms[0].size(), mean[1], med[1],
              ab_ms[1].size(), mean[1] - mean[0], med[1] - med[0]);
    }
    if (!ref.empty()) {
      const size_t cmpn = std::min(ref.size(), out.size());
      if (div_at < 0) {
        printf("ref: identical to the reference over %zu tokens\n", cmpn);
        return 0;
      }
      // the same position computed by plain decoding and by one prefill of the prefix (fresh sequences)
      const int V = m.config().vocab, pos = n + div_at;  // tokens[pos] is the differing token
      auto top2 = [&](const float* l, int* i0, int* i1) {
        *i0 = 0;
        for (int v = 1; v < V; ++v)
          if (l[v] > l[*i0]) *i0 = v;
        *i1 = *i0 ? 0 : 1;
        for (int v = 0; v < V; ++v)
          if (v != *i0 && l[v] > l[*i1]) *i1 = v;
      };
      SamplingParams gsp;
      float plain[2], pre[2];
      int p0, p1, q0, q1;
      {
        auto s2 = m.new_seq(pos + 64);
        for (int i = 0; i < n; ++i) s2->tokens()[i] = ids[i];
        for (int i = 0; i < div_at; ++i) s2->tokens()[n + i] = out[i];
        for (int p = 0; p < n; p += m.prefill_chunk()) {
          auto e = g.begin();
          m.encode_prefill(*e, *s2, p, std::min(m.prefill_chunk(), n - p), gsp, 0, false);
          e->commit_and_wait();
        }
        for (int q = n; q < pos; ++q) {  // feed out[0 .. div_at-1] token by token; the last step's logits
          auto e = g.begin();
          m.encode_step(*e, *s2, q, gsp, 0, q == pos - 1);
          e->commit_and_wait();
        }
        if (pos == n) {  // the first generated token: the prefill's own logits
          auto e = g.begin();
          m.reset(*s2);
          for (int p = 0; p < n; p += m.prefill_chunk()) {
            const int T = std::min(m.prefill_chunk(), n - p);
            m.encode_prefill(*e, *s2, p, T, gsp, 0, p + T == n);
          }
          e->commit_and_wait();
        }
        plain[0] = m.logits()[div_tok[0]];
        plain[1] = m.logits()[div_tok[1]];
        top2(m.logits(), &p0, &p1);
        printf("G2 plain decode: top-2 %d (%.4f) %d (%.4f), gap %.4f\n", p0, m.logits()[p0], p1, m.logits()[p1],
               m.logits()[p0] - m.logits()[p1]);
      }
      {
        auto s3 = m.new_seq(pos + 64);
        for (int i = 0; i < n; ++i) s3->tokens()[i] = ids[i];
        for (int i = 0; i < div_at; ++i) s3->tokens()[n + i] = out[i];
        for (int p = 0; p < pos; p += m.prefill_chunk()) {
          const int T = std::min(m.prefill_chunk(), pos - p);
          auto e = g.begin();
          m.encode_prefill(*e, *s3, p, T, gsp, 0, p + T == pos);
          e->commit_and_wait();
        }
        pre[0] = m.logits()[div_tok[0]];
        pre[1] = m.logits()[div_tok[1]];
        top2(m.logits(), &q0, &q1);
      }
      printf("G2 first difference at output token %d (position %d): reference %d, dflash %d\n", div_at, pos, div_tok[0],
             div_tok[1]);
      printf("G2   logits (reference token / dflash token): plain decode %.4f / %.4f (gap %+.4f); dflash verify "
             "row %d at P=%d %.4f / %.4f (gap %+.4f); one prefill of the prefix %.4f / %.4f (gap %+.4f, argmax %d)\n",
             plain[0], plain[1], plain[0] - plain[1], div_row, div_P, div_logit[0], div_logit[1],
             div_logit[0] - div_logit[1], pre[0], pre[1], pre[0] - pre[1], q0);
    }
    return 0;
  }

  if (spec && m.has_mtp()) {
    const int D = spec;
    double ts = now_ms();
    int P = n, c = 1, h_row0 = n - 1 - last_chunk;
    int cycles = 0, accepted = 0;
    if (Encoder::profiling()) Encoder::profile_report(1);  // drop prefill records
    std::vector<int> acc_hist(D + 1, 0);
    bool stop2 = false;
    double gpu_sum = 0;
    while (int(out.size()) < n_gen && !stop2) {
      double tc0 = now_ms();
      grow(P + D + 3);  // nothing in flight: every cycle is waited for
      static const bool split_t = getenv("NI_SPEC_SPLIT") != nullptr;  // timing: drafts and verify apart
      std::unique_ptr<Encoder> ed;
      auto e = g.begin();
      m.encode_mtp(*e, s, P - c, c, P - c + 1, m.hidden_rows(), h_row0, P + 1);
      for (int k = 2; k <= D; ++k)
        m.encode_mtp(*e, s, P + k - 2, 1, P + k - 1, m.mtp_post(), k == 2 ? c - 1 : 0, P + k, false);
      if (split_t) {
        e->commit_and_wait();
        ed = std::move(e);
        e = g.begin();
      }
      m.encode_verify(*e, s, P, D + 1, sp, uint32_t(P));
      e->commit_and_wait();
      gpu_sum += e->gpu_ms();
      if (split_t) {
        static double sd = 0, sv = 0;
        static int ns = 0;
        sd += ed->gpu_ms();
        sv += e->gpu_ms();
        if (++ns % 32 == 0) fprintf(stderr, "split timing: drafts %.2f ms, verify %.2f ms (mean of %d cycles)\n", sd / ns, sv / ns, ns);
      }
      const int* y = m.samples();
      int* tk = s.tokens();
      int a = 0;
      while (a < D && tk[P + 1 + a] == y[a]) ++a;
      if (getenv("NI_TRACE"))
        fprintf(stderr, "P=%d c=%d draft=%d y0=%d a=%d wall=%.1fms gpu=%.1fms\n", P, c, tk[P + 1], y[0], a,
                now_ms() - tc0, e->gpu_ms());
      tk[P + a + 1] = y[a];
      for (int i = 1; i <= a + 1 && int(out.size()) < n_gen; ++i) {
        out.push_back(tk[P + i]);
        if (!ignore_eos)
          for (int eos : m.config().eos)
            if (tk[P + i] == eos) stop2 = true;
        if (stop2) break;
      }
      m.commit_rows(s, a + 1);
      ++cycles;
      accepted += a;
      acc_hist[a]++;
      c = a + 1;
      h_row0 = 0;
      P += a + 1;
    }
    double td = now_ms() - ts;
    if (Encoder::profiling()) Encoder::profile_report(cycles);
    printf("ids:");
    for (int t : out) printf(" %d", t);
    printf("\n");
    fprintf(stderr, "prompt %d tok in %.0f ms (%.1f tok/s); spec decode D=%d: %zu tok in %.0f ms (%.1f tok/s), %d cycles, %.2f tok/cycle, accepted/drafted %.2f, %.3f ms GPU/cycle, %.3f ms wall/cycle, hist",
            n, t_prompt, n / (t_prompt / 1000), D, out.size() - 1, td, (out.size() - 1) / (td / 1000), cycles,
            double(out.size() - 1) / cycles, double(accepted) / (cycles * D), gpu_sum / std::max(cycles, 1),
            td / std::max(cycles, 1));
    for (int a = 0; a <= D; ++a) fprintf(stderr, " %d", acc_hist[a]);
    fprintf(stderr, "\n");
    store_stats("decode");
    return 0;
  }

  // Decode: keep up to 2 command buffers in flight; the GPU writes each sampled id into the
  // token buffer, which the next step's embedding reads, so the CPU never waits per token.
  double t1 = now_ms();
  std::deque<std::unique_ptr<Encoder>> inflight;
  int pos = n;
  int done = 1;
  bool stop = false;
  double gpu_total = 0;
  int gpu_n = 0;
  for (int step = 1; step < n_gen && !stop; ++step) {
    grow(pos + 2);  // under --kv-grow nothing is in flight here (steps are waited for one by one)
    auto e = g.begin();
    m.encode_step(*e, s, pos, sp, uint32_t(step), true);
    e->commit();
    inflight.push_back(std::move(e));
    ++pos;
    while (inflight.size() > (fall || kv_grow ? 0 : 2) || (step == n_gen - 1 && !inflight.empty())) {
      inflight.front()->wait();
      if (fall) fwrite(m.logits(), 4, m.config().vocab, fall);
      gpu_total += inflight.front()->gpu_ms();
      ++gpu_n;
      inflight.pop_front();
      int tok = s.tokens()[n + done];
      out.push_back(tok);
      ++done;
      if (!ignore_eos)
        for (int eos : m.config().eos)
          if (tok == eos) stop = true;
    }
  }
  while (!inflight.empty()) {
    inflight.front()->wait();
    inflight.pop_front();
    out.push_back(s.tokens()[n + done]);
    ++done;
  }
  double t_dec = now_ms() - t1;
  if (fall) fclose(fall);
  if (Encoder::profiling()) Encoder::profile_report(int(out.size()) - 1);
  if (gpu_n) fprintf(stderr, "mean GPU time per step: %.2f ms (%d steps)\n", gpu_total / gpu_n, gpu_n);
  printf("ids:");
  for (int t : out) printf(" %d", t);
  printf("\n");
  fprintf(stderr, "prompt %d tok in %.0f ms (%.1f tok/s); decode %zu tok in %.0f ms (%.1f tok/s)\n", n, t_prompt,
          n / (t_prompt / 1000), out.size() - 1, t_dec, (out.size() - 1) / (t_dec / 1000));
  store_stats("decode");
  return 0;
}
