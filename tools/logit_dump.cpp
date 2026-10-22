// Full-vocabulary logits after fixed prefixes, teacher-forced through the production decode paths: the dump that
// tools/logit_compare.py compares and gates against this Mac's noise floor (CONTRIBUTING.md, "Output check").
//
//   logit_dump <model_dir> <out_dir> [--paths step,verify,batch] [--from DIR] [--corpus FILE] [--windows N]
//              [--rows W] [--chunk C] [--batch N] [--batch-rows R] [--loops N] [--loop-kv last-loop|per-loop]
//
// A window is a cold request: its first L0 tokens are prefilled as the engine prefills a prompt (chunks of C rows,
// default the engine's 2048, with the Neural Engine split wherever the model runs it), then W rows are fed
// teacher-forced (every row the sequence's own next token, never the model's sample) and each row's logits over the
// whole vocabulary are written. The paths (each from the same prefilled state: a checkpoint restored per path):
//   step    plain decoding, one row per pass (what --spec 0 runs)
//   verify  the single-stream verify of 8 rows. Qwen: KFlash's tree verify (the default) as DFlash2's cycle encodes it,
//           the tree a chain of 8 live rows, the experts' row-independent numerics (NI_DFLASH_EXACT_MOE), the K/V
//           compaction and the next cycle's fold of the accepted path; NI_KFLASH=0 the chain verify instead (DFlash2
//           without trees). Nemotron-H: the 8-row chain verify. Dory: only with per-loop KV (--loop-kv per-loop), as
//           the engine runs it.
//   batch   --batch N windows decoding together (encode_verify_multi: one pass over every member's rows), R rows each
//           per pass: by default the engine's numbers for N requests (Engine::batch_rows: 2 at once 4 rows each, 3 or
//           4 at once 2 each; Nemotron-H 4 at once 1 each). Qwen and Nemotron-H.
//   prefill (always) the prompt's last row as the prefill samples it, one record per window: bitwise the same in two
//           dumps, the decode paths' differences are the decode passes' own (logit_compare.py gates them at their floor)
// Default windows: the 8 files of bench/corpus.txt, tokenized as they are (no chat template); window i prefills the
// first 3200 - 384 i tokens of file i (3200 .. 512: two prefill chunks past 2048, one up to it) and scores the next
// --rows W = 24 (192 rows in all). --from DIR scores the records of an earlier dump instead (Kapitsa's or another
// engine's): per sequence, its scored prefixes form the window, and the files keep DIR's lines in DIR's order.
//
// out_dir/<path>/ (the layout of Splash's teacher-forced dumps, so that either engine's dumps compare with the other's):
//   sequences.txt  one window's token ids per line (as far as its last scored prefix's next token)
//   records.txt    per scored prefix: its sequence's line (from 0) and the prefix's length L; the row predicts token L
//   logits.f32     per record, in order, the vocabulary's logits (fp32, little-endian)
//   meta.json      the model, the path and how it ran (build, Mac, macOS, chunk, Neural Engine, tree, batch, NI_*)
// Takes the GPU lock ($BENCH_GPU_LOCK, default ~/.cache/nace-gpu.lock) unless NI_LOCK_HELD=1 or NACE_GPU_LOCK_HELD=1.
#include "engine/engine.h"
#include "bench/bench.h"
#include "metal/gpu.h"
#include "model/dory.h"
#include "model/flint.h"
#include "model/kflash_budget.h"
#include "model/nemotron_h.h"
#include "model/qwen35.h"
#include "model/qwen4_exp.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

extern char** environ;

using namespace ni;

namespace {

constexpr int kVerifyRows = 8;  // DFlash2's block: the anchor and 7 drafts (Qwen35::kMaxVerify, NemotronH::kMaxRows)

struct Window {
  std::vector<int> toks;  // the sequence (sequences.txt's line)
  int L0 = 0;             // prefilled tokens
  int W = 0;              // rows fed after them: positions L0 .. L0 + W - 1 (the last scored prefix is L0 + W)
  std::vector<long> rec;  // per row i (prefix L0 + i + 1): its line in records.txt, -1 if not scored
};

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, sep))
    if (!t.empty()) v.push_back(t);
  return v;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) fatal("cannot read %s", path.c_str());
  return std::string(std::istreambuf_iterator<char>(f), {});
}

std::vector<std::vector<long>> read_int_lines(const std::string& path) {
  std::vector<std::vector<long>> out;
  std::stringstream ss(read_file(path));
  std::string line;
  while (std::getline(ss, line)) {
    std::vector<long> v;
    std::stringstream ls(line);
    long x;
    while (ls >> x) v.push_back(x);
    if (!ls.eof()) fatal("%s: not a line of integers: %.60s", path.c_str(), line.c_str());
    out.push_back(std::move(v));
  }
  return out;
}

// The default windows: file i of the corpus ("### FILE:" sections), the first 512 + 384 (n - 1 - i) tokens prefilled
// (the longest prefixes from the first files: bench/corpus.txt's last file is its shortest)
std::vector<Window> corpus_windows(const std::string& corpus, const std::string& model_dir, int n, int W) {
  const std::string text = read_file(corpus);
  std::vector<std::string> docs;
  for (size_t a = text.find("### FILE:"); a != std::string::npos;) {
    const size_t b = text.find("\n### FILE:", a + 1);
    docs.push_back(text.substr(a, b == std::string::npos ? std::string::npos : b + 1 - a));
    a = b == std::string::npos ? b : b + 1;
  }
  if (docs.empty()) fatal("%s: no \"### FILE:\" sections", corpus.c_str());
  Tokenizer tok;
  tok.load(model_dir + "/tokenizer.json");
  std::vector<Window> ws;
  for (int i = 0; i < n; ++i) {
    Window w;
    w.L0 = 512 + 384 * (n - 1 - i);
    w.W = W;
    std::vector<int> ids = tok.encode(docs[size_t(i) % docs.size()]);
    if (int(ids.size()) < w.L0 + W + 1)
      fatal("%s: section %zu has %zu tokens, window %d needs %d", corpus.c_str(), size_t(i) % docs.size(), ids.size(), i,
            w.L0 + W + 1);
    w.toks.assign(ids.begin(), ids.begin() + w.L0 + W + 1);
    ws.push_back(std::move(w));
  }
  return ws;
}

// --from: each sequence's scored prefixes are a window (prefilled up to its first scored prefix less one token). A
// directory of logit_dump's path dumps gives any of them: they hold the same sequences and records.
std::vector<Window> dump_windows(std::string dir, std::vector<std::pair<long, long>>& records) {
  if (access((dir + "/records.txt").c_str(), R_OK) != 0)
    for (const char* p : {"verify", "step", "batch"})
      if (access((dir + "/" + p + "/records.txt").c_str(), R_OK) == 0) {
        dir += std::string("/") + p;
        break;
      }
  const auto seqs = read_int_lines(dir + "/sequences.txt");
  const auto recs = read_int_lines(dir + "/records.txt");
  std::vector<Window> ws(seqs.size());
  for (size_t i = 0; i < seqs.size(); ++i) ws[i].toks.assign(seqs[i].begin(), seqs[i].end());
  std::vector<int> lo(seqs.size(), -1), hi(seqs.size(), -1);
  for (const auto& r : recs) {
    if (r.empty()) continue;
    if (r.size() != 2 || r[0] < 0 || size_t(r[0]) >= seqs.size()) fatal("%s/records.txt: bad record", dir.c_str());
    if (r[1] < 2 || size_t(r[1]) >= seqs[size_t(r[0])].size())
      fatal("%s/records.txt: prefix %ld of sequence %ld (%zu tokens): not one this tool scores (from 2 tokens, with the "
            "next token in the sequence)", dir.c_str(), r[1], r[0], seqs[size_t(r[0])].size());
    records.emplace_back(r[0], r[1]);
    int& l = lo[size_t(r[0])];
    int& h = hi[size_t(r[0])];
    l = l < 0 ? int(r[1]) : std::min(l, int(r[1]));
    h = std::max(h, int(r[1]));
  }
  if (records.empty()) fatal("%s/records.txt: no records", dir.c_str());
  for (size_t i = 0; i < ws.size(); ++i) {
    if (lo[i] < 0) continue;
    ws[i].L0 = lo[i] - 1;
    ws[i].W = hi[i] - ws[i].L0;
    ws[i].rec.assign(size_t(ws[i].W), -1);
  }
  for (size_t k = 0; k < records.size(); ++k) {
    Window& w = ws[size_t(records[k].first)];
    w.rec[size_t(records[k].second - w.L0 - 1)] = long(k);
  }
  return ws;
}

std::string model_name(std::string dir) {
  while (dir.size() > 1 && dir.back() == '/') dir.pop_back();
  return dir.substr(dir.find_last_of('/') + 1);
}

std::string now_iso() {
  char buf[32];
  const time_t t = time(nullptr);
  strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: logit_dump <model_dir> <out_dir> [--paths step,verify,batch] [--from DIR] [--corpus FILE] "
                    "[--windows N] [--rows W] [--chunk C] [--batch N] [--batch-rows R] [--loops N] "
                    "[--loop-kv last-loop|per-loop]\n");
    return 1;
  }
  const std::string dir = argv[1], out = argv[2];
  std::string from, corpus = "bench/corpus.txt", paths_arg = "step,verify,batch", loop_kv;
  int n_windows = 8, rows = 24, chunk = 2048, members = 4, member_rows = 0, loops = 0;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", a.c_str());
      return argv[++i];
    };
    if (a == "--paths") paths_arg = next();
    else if (a == "--from") from = next();
    else if (a == "--corpus") corpus = next();
    else if (a == "--windows") n_windows = std::stoi(next());
    else if (a == "--rows") rows = std::stoi(next());
    else if (a == "--chunk") chunk = std::stoi(next());
    else if (a == "--batch") members = std::stoi(next());
    else if (a == "--batch-rows") member_rows = std::stoi(next());
    else if (a == "--loops") loops = std::stoi(next());
    else if (a == "--loop-kv") loop_kv = next();
    else fatal("unknown argument %s", a.c_str());
  }
  const std::vector<std::string> paths = split(paths_arg, ',');
  for (const auto& p : paths)
    if (p != "step" && p != "verify" && p != "batch") fatal("--paths: unknown path %s (step, verify, batch)", p.c_str());
  if (n_windows < 1 || rows < 1 || chunk < 1 || members < 1 || members > 8)
    fatal("--windows, --rows, --chunk: positive; --batch: 1 to 8");

  // ---- the windows and their records ----
  std::vector<std::pair<long, long>> records;  // (sequence, prefix length), records.txt's order
  std::vector<Window> wins;
  if (!from.empty()) {
    wins = dump_windows(from, records);
  } else {
    wins = corpus_windows(corpus, dir, n_windows, rows);
    for (size_t i = 0; i < wins.size(); ++i) {
      wins[i].rec.resize(size_t(wins[i].W));
      for (int r = 0; r < wins[i].W; ++r) {
        wins[i].rec[size_t(r)] = long(records.size());
        records.emplace_back(long(i), long(wins[i].L0 + r + 1));
      }
    }
  }
  for (size_t i = 0; i < wins.size(); ++i)
    if (wins[i].W > 0 && wins[i].L0 < 1) fatal("sequence %zu: a scored prefix of one token leaves nothing to prefill", i);

  // ---- the GPU lock, then the model ----
  bench::GpuLock lock;
  if (!env_on("NI_LOCK_HELD") && !env_on("NACE_GPU_LOCK_HELD")) {
    if (!lock.acquire([](const std::string& who) {
          fprintf(stderr, "waiting for the GPU lock %s%s\n", bench::GpuLock::path().c_str(),
                  who.empty() ? "" : (", held by " + who).c_str());
        }))
      fatal("cannot take the GPU lock %s: %s", bench::GpuLock::path().c_str(), strerror(errno));
  }
  Gpu& g = Gpu::get();
  if (Qwen4Exp::is_model(dir)) fatal("Qwen3.8-Flash-Next decodes with its MTP head: no path of this tool runs it");
  std::unique_ptr<Model> model;
  Qwen35* q = nullptr;
  NemotronH* nh = nullptr;
  Dory* dory = nullptr;
  Flint* fl = nullptr;
  if (Flint::is_model(dir)) {
    auto m = std::make_unique<Flint>(dir, chunk);
    fl = m.get();
    model = std::move(m);
  } else if (Dory::is_model(dir)) {
    LoopKv kv = LoopKv::last_loop;
    if (!loop_kv.empty() && !parse_loop_kv(loop_kv, &kv)) fatal("--loop-kv: last-loop or per-loop");
    auto m = std::make_unique<Dory>(dir, chunk, loops, kv);
    dory = m.get();
    model = std::move(m);
  } else if (NemotronH::is_model(dir)) {
    auto m = std::make_unique<NemotronH>(dir, chunk);
    nh = m.get();
    model = std::move(m);
  } else {
    auto m = std::make_unique<Qwen35>(dir, chunk);
    q = m.get();
    model = std::move(m);
  }
  Model& m = *model;
  const int V = m.vocab();
  auto logits = [&]() -> const float* {
    return q ? q->logits() : nh ? nh->logits() : fl ? fl->logits() : dory->logits();
  };
  auto begin_pass = [&] { return m.grouped_passes() ? g.begin_grouped() : g.begin(); };
  const SamplingParams sp;  // greedy: no pass samples a token the next row reads

  // KFlash, as DFlash2 decides it (NI_KFLASH, NI_DFLASH_TREE; the verify must have a tree path), and the experts'
  // row-independent numerics its tree verifies run with (NI_DFLASH_EXACT_MOE)
  bool tree = false, moe_exact = true;
  if (q) {
    const char* kf = getenv("NI_KFLASH");
    const char* dt = getenv("NI_DFLASH_TREE");
    tree = dt && *dt ? atoi(dt) != 0 : !(kf && *kf) || atoi(kf) != 0;
    tree = tree && q->block_drafter_ok() && q->verify_tree_ok();
    if (const char* v = getenv("NI_DFLASH_EXACT_MOE")) moe_exact = atoi(v) != 0;
  }
  if (member_rows <= 0) member_rows = members == 2 ? 4 : members <= 4 ? (nh && members == 4 ? 1 : 2) : 1;
  if (members * member_rows > kVerifyRows) fatal("--batch %d x --batch-rows %d: more than 8 rows", members, member_rows);
  for (const auto& p : paths) {
    if (p == "batch" && dory) fatal("Dory decodes one request at a time: no batch path");
    if (p == "batch" && fl) fatal("Flint decodes one request at a time: no batch path");
    if (p == "verify" && dory && dory->kv_mode() != LoopKv::per_loop)
      fatal("Dory verifies drafts only with per-loop KV (--loop-kv per-loop); with last-loop KV it decodes step by step");
  }
  // rows a window feeds on a path: its scored rows, up to whole passes
  // Flint: NI_FLINT_VERIFY_ROWS=R (2-8) verifies R rows a pass instead of 8 (its 2-4-row passes take other expert kernels)
  const int vrows = fl && getenv("NI_FLINT_VERIFY_ROWS") ? std::clamp(atoi(getenv("NI_FLINT_VERIFY_ROWS")), 2, 8)
                                                         : kVerifyRows;
  auto path_rows = [&](const std::string& p, const Window& w) {
    const int k = p == "verify" ? vrows : p == "batch" ? member_rows : 1;
    const int W = (w.W + k - 1) / k * k;
    if (w.W > 0 && int(w.toks.size()) < w.L0 + W)
      fatal("a sequence of %zu tokens: the %s path's passes of %d rows need %d", w.toks.size(), p.c_str(), k, w.L0 + W);
    return W;
  };

  // ---- the output files ----
  mkdir(out.c_str(), 0755);
  std::string seq_txt, rec_txt;
  for (const Window& w : wins) {
    for (size_t t = 0; t < w.toks.size(); ++t) seq_txt += (t ? " " : "") + std::to_string(w.toks[t]);
    seq_txt += "\n";
  }
  for (const auto& r : records) rec_txt += std::to_string(r.first) + " " + std::to_string(r.second) + "\n";
  std::string pre_txt;  // the prefill path's records: each window's prompt
  std::vector<long> pre_rec(wins.size(), -1);
  for (size_t i = 0, k = 0; i < wins.size(); ++i)
    if (wins[i].W > 0) {
      pre_rec[i] = long(k++);
      pre_txt += std::to_string(i) + " " + std::to_string(wins[i].L0) + "\n";
    }
  struct Out {
    std::string name, dir;
    int fd = -1;
    double nll = 0;
    long hits = 0, done = 0;
  };
  std::vector<Out> outs;  // outs[0]: the prefill path
  for (const std::string& p : [&] { auto v = paths; v.insert(v.begin(), "prefill"); return v; }()) {
    Out o{p, out + "/" + p};
    const bool pre = p == "prefill";
    mkdir(o.dir.c_str(), 0755);
    std::ofstream(o.dir + "/sequences.txt", std::ios::binary) << seq_txt;
    std::ofstream(o.dir + "/records.txt", std::ios::binary) << (pre ? pre_txt : rec_txt);
    o.fd = open((o.dir + "/logits.f32").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    const off_t n = pre ? off_t(std::count_if(pre_rec.begin(), pre_rec.end(), [](long k) { return k >= 0; }))
                        : off_t(records.size());
    if (o.fd < 0 || ftruncate(o.fd, n * V * 4) != 0) fatal("cannot write %s/logits.f32", o.dir.c_str());
    outs.push_back(o);
  }
  // record k of path o: its row, and how well it predicts the sequence's token `next`
  auto put_row = [&](Out& o, long k, const float* row, int next) {
    if (pwrite(o.fd, row, size_t(V) * 4, off_t(k) * V * 4) != ssize_t(size_t(V) * 4)) fatal("write %s", o.dir.c_str());
    const float* top = std::max_element(row, row + V);
    double s = 0;
    for (int v = 0; v < V; ++v) s += std::exp(double(row[v]) - double(*top));
    o.nll += std::log(s) + double(*top) - double(row[next]);
    o.hits += int(top - row) == next;
    ++o.done;
  };
  auto put = [&](Out& o, const Window& w, int i, const float* row) {  // row i of window w (prefix L0 + i + 1)
    if (i < w.W && w.rec[size_t(i)] >= 0) put_row(o, w.rec[size_t(i)], row, w.toks[size_t(w.L0 + i + 1)]);
  };

  // ---- per group of windows: prefill each alone, then every path from the same state ----
  Buffer kf_tree, kf_live;
  if (tree) {  // the tree descriptor of a chain of 8 live rows, as kf_build writes it and kf_accept completes it
    kf_tree = g.alloc(size_t(kf::kWords) * 4, "logit_dump_tree");
    kf_live = g.alloc(16, "logit_dump_live");
    int32_t* t = kf_tree.as<int32_t>();
    memset(t, 0, size_t(kf::kWords) * 4);
    for (int r = 0; r < kVerifyRows; ++r) {
      t[kf::kPar + r] = r - 1;
      t[kf::kDep + r] = r;
      t[kf::kMask + r] = (1 << (r + 1)) - 1;
      t[kf::kPath + r] = r;
    }
    t[kf::kN] = t[kf::kNodes] = kVerifyRows;
    t[kf::kA] = kVerifyRows - 1;  // every draft accepted: the path is the chain
    kf_live.as<uint32_t>()[0] = kVerifyRows;
  }
  // one group's sequences and checkpoints, reused group after group as the engine reuses a slot (reset, prefill)
  int need = 1;  // positions a window reads on any path
  for (const Window& w : wins)
    if (w.W > 0)
      for (const auto& p : paths) need = std::max(need, w.L0 + path_rows(p, w));
  std::vector<std::unique_ptr<Seq>> seqs;
  std::vector<Checkpoint> cks;
  for (size_t j = 0; j < std::min(wins.size(), size_t(members)); ++j) {
    seqs.push_back(m.new_seq(need + 64));
    cks.push_back(m.new_checkpoint());
  }
  auto load_tokens = [&](Seq& s, const Window& w) {
    std::copy(w.toks.begin(), w.toks.begin() + long(std::min(w.toks.size(), size_t(need) + 1)), s.tokens());
  };
  const double t_start = now_ms();
  double t_prefill = 0;
  long prefilled = 0;
  for (size_t g0 = 0; g0 < wins.size(); g0 += size_t(members)) {
    std::vector<Window*> grp;
    for (size_t i = g0; i < std::min(wins.size(), g0 + size_t(members)); ++i)
      if (wins[i].W > 0) grp.push_back(&wins[i]);
    for (size_t j = 0; j < grp.size(); ++j) {
      const Window& w = *grp[j];
      Seq& s = *seqs[j];
      m.reset(s);
      load_tokens(s, w);
      const double t0 = now_ms();
      // the engine's chunking (Engine::chat takes prefill_chunk_rows too): --chunk 2112 with NI_BAL on is what the
      // engine runs, --chunk 2048 NI_BAL=0 what it ran before, and --chunk 2048 with NI_BAL on changes only the split
      const int C = m.prefill_chunk();
      const bool even = prefill_even_chunks() && !m.ane_split_on();
      for (int p0 = 0; p0 < w.L0;) {
        const int T = prefill_chunk_rows(C, p0, w.L0, even);
        auto e = g.begin();
        m.encode_prefill(*e, s, p0, T, sp, 0, p0 + T == w.L0);
        e->commit_and_wait();
        if (!m.ane_finish_chunk()) fatal("the Neural Engine prefill split failed on a chunk; rerun with NI_ANE=0");
        p0 += T;
      }
      t_prefill += now_ms() - t0;
      prefilled += w.L0;
      put_row(outs[0], pre_rec[size_t(grp[j] - wins.data())], logits(), w.toks[size_t(w.L0)]);  // the sampled row
      auto e = g.begin();
      m.save(*e, s, cks[j], w.L0);
      e->commit_and_wait();
    }
    for (Out& o : outs) {
      if (o.name == "prefill") continue;
      // the prefilled state again: recurrent state from the checkpoint (KV past L0 is rewritten before it is read),
      // no pending rows, the true tokens (a step's sample overwrites the next one)
      for (size_t j = 0; j < grp.size(); ++j) {
        Seq& s = *seqs[j];
        auto e = g.begin();
        m.restore(*e, s, cks[j]);
        e->commit_and_wait();
        s.n_pend = 0;
        s.pend_tree = false;
        load_tokens(s, *grp[j]);
      }
      if (o.name == "step") {
        for (size_t j = 0; j < grp.size(); ++j) {
          const Window& w = *grp[j];
          Seq& s = *seqs[j];
          for (int i = 0, W = path_rows(o.name, w); i < W; ++i) {
            const int pos = w.L0 + i;
            auto e = begin_pass();
            m.encode_step(*e, s, pos, sp, uint32_t(pos), true);
            e->commit_and_wait();
            put(o, w, i, logits());
            s.tokens()[pos + 1] = w.toks[size_t(pos + 1)];
          }
        }
      } else if (o.name == "verify") {
        for (size_t j = 0; j < grp.size(); ++j) {
          const Window& w = *grp[j];
          Seq& s = *seqs[j];
          if (tree && !s.pend_path) {
            s.pend_path = g.alloc(size_t(kf::kPathWords) * 4, "logit_dump_path");
            for (int r = 0; r < kVerifyRows; ++r) s.pend_path.as<int32_t>()[r] = r;  // kf_accept's path: the chain
          }
          for (int i = 0, W = path_rows(o.name, w); i < W; i += vrows) {
            const int P = w.L0 + i;
            auto e = q ? g.begin() : begin_pass();
            if (q && tree) {  // DFlash2::encode_cycle_verify's tree branch, the acceptance's writes made in advance
              q->set_verify_tree(&kf_tree);
              q->set_verify_row_limit(&kf_live);
              q->set_verify_moe_exact(moe_exact);
              q->encode_verify(*e, s, P, kVerifyRows, sp, uint32_t(P), /*v8=*/true);
              q->set_verify_moe_exact(false);
              q->set_verify_row_limit(nullptr);
              q->set_verify_tree(nullptr);
              q->encode_kv_tree_compact(*e, s, P, kf_tree);
            } else if (q) {  // its chain branch (greedy: no row limit fed by the drafter, the experts as they group)
              q->encode_verify(*e, s, P, kVerifyRows, sp, uint32_t(P), /*v8=*/true);
            } else if (nh) {
              nh->encode_verify(*e, s, P, kVerifyRows);
            } else if (fl) {
              fl->encode_verify(*e, s, P, vrows);
            } else {
              dory->encode_verify(*e, s, P, kVerifyRows);
            }
            e->commit_and_wait();
            for (int r = 0; r < vrows; ++r) put(o, w, i + r, logits() + size_t(r) * V);
            if (q) {
              q->commit_rows(s, kVerifyRows);
              s.pend_tree = tree;  // a tree's next pass folds the accepted path (pend_path), as after kf_accept
            }
            if (nh) nh->commit_rows(s, kVerifyRows);
          }
        }
      } else {  // batch: the members' rows in one pass, a member leaving the batch when its rows are done
        for (int i = 0;; i += member_rows) {
          std::vector<size_t> live;
          for (size_t j = 0; j < grp.size(); ++j)
            if (i < path_rows(o.name, *grp[j])) live.push_back(j);
          if (live.empty()) break;
          auto e = g.begin_grouped();
          if (q) {
            std::vector<Qwen35::VerifyGroup> gr;
            for (size_t j : live) gr.push_back(Qwen35::VerifyGroup{seqs[j].get(), grp[j]->L0 + i, member_rows, sp});
            q->encode_verify_multi(*e, gr);
          } else {
            std::vector<NemotronH::VerifyGroup> gr;
            for (size_t j : live) gr.push_back(NemotronH::VerifyGroup{seqs[j].get(), grp[j]->L0 + i, member_rows, sp});
            nh->encode_verify_multi(*e, gr);
          }
          e->commit_and_wait();
          for (size_t k = 0; k < live.size(); ++k) {
            const size_t j = live[k];
            for (int r = 0; r < member_rows; ++r) put(o, *grp[j], i + r, logits() + (k * member_rows + r) * size_t(V));
            if (q) q->commit_rows(*seqs[j], member_rows);
            else nh->commit_rows(*seqs[j], member_rows);
          }
        }
      }
    }
  }

  // ---- meta.json per path ----
  const bench::Mac mac = bench::this_mac();
  std::string build = KAPITSA_VERSION;
#ifdef KAPITSA_GIT
  if (*KAPITSA_GIT) build += std::string(" (") + KAPITSA_GIT + ")";
#endif
  using bench::ojson;
  ojson env = ojson::object();
  for (char** e = environ; *e; ++e)
    if (!strncmp(*e, "NI_", 3) && strncmp(*e, "NI_LOCK_HELD=", 13) && strncmp(*e, "NI_LOG=", 7)) {
      const std::string kv = *e;
      env[kv.substr(0, kv.find('='))] = kv.substr(kv.find('=') + 1);
    }
  const AneStats& ane = m.ane_stats();
  for (Out& o : outs) {
    close(o.fd);
    const long want = o.name == "prefill" ? long(std::count_if(pre_rec.begin(), pre_rec.end(), [](long k) { return k >= 0; }))
                                          : long(records.size());
    if (o.done != want) fatal("%s: %ld of %ld records written", o.dir.c_str(), o.done, want);
    ojson meta;
    meta["engine"] = "kapitsa";
    meta["build"] = build;
    meta["model"] = model_name(dir);
    meta["model_dir"] = dir;
    meta["path"] = o.name;
    if (o.name == "verify")
      meta["verify"] = q ? (tree ? "KFlash tree (a chain of 8 live rows)" : "chain of 8 rows") : "8 rows";
    if (o.name == "verify" && q && tree && q->config().moe) meta["moe_exact"] = moe_exact;
    if (o.name == "batch") meta["batch"] = {{"members", members}, {"rows", member_rows}};
    meta["prefill_chunk"] = m.prefill_chunk();
    meta["ane"] = {{"chunks", ane.chunks}, {"evals", ane.evals}, {"reruns", ane.reruns}};
    if (dory) meta["dory"] = {{"loops", dory->default_loops()}, {"kv", loop_kv_name(dory->kv_mode())}};
    meta["records"] = o.done;
    meta["vocab"] = V;
    meta["source"] = from.empty() ? corpus : from;
    meta["mac"] = bench::mac_display(mac.name, mac.year, mac.chip, mac.identifier) + ", " + bench::chip_short(mac.chip) +
                  (mac.gpu_cores ? " (" + std::to_string(mac.gpu_cores) + "-core GPU)" : "") + ", " +
                  std::to_string(mac.memory_bytes >> 30) + " GB";
    meta["macos"] = mac.macos;
    meta["date"] = now_iso();
    meta["env"] = env;
    std::ofstream(o.dir + "/meta.json") << meta.dump(2) << "\n";
    printf("%-6s %ld rows: next-token NLL %.4f, top-1 = next token %.4f -> %s\n", o.name.c_str(), o.done,
           o.done ? o.nll / double(o.done) : 0.0, o.done ? double(o.hits) / double(o.done) : 0.0, o.dir.c_str());
  }
  printf("%zu windows, %ld prefilled tokens in %.1f s; %.1f s in all\n", wins.size(), prefilled, t_prefill / 1000,
         (now_ms() - t_start) / 1000);
  return 0;
}
