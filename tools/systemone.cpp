// Drex systemone on Kapitsa: serves POST /v1/systemone (the Drex decision-model protocol) from a Qwen3.5 / Qwen3.5-MoE
// backbone whose weights are in a Kapitsa model folder, read by a Drex pointer head (pointer.bin, written by
// tools/drex_pointer.py). Nothing is generated: the head scores the backbone's post-final-norm hidden state at each
// option's closing token against the question's decide token.
//
//   systemone --model DIR [--port 8000] [--host 127.0.0.1] [--context 16384] [--name drex-v2.0]
//   systemone --model DIR --once request.json [--repeat N]     # one request from a file, print the reply
//
// One state pass fills the sequence to n_state and a checkpoint of the recurrent state is saved there; every question
// then restores that checkpoint and runs one pass of its own row (instructions, options, decide) at positions
// n_state .. So the state's attention rows and recurrent state are shared and each question's row is causal after it.
#include "bench/bench.h"
#include "engine/prefix_cache.h"
#include "metal/gpu.h"
#include "model/qwen4_exp.h"
#include "model/qwen35.h"
#include "systemone/systemone.h"
#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using namespace ni;
using ni::systemone::ojson;

namespace {

httplib::Server* g_srv = nullptr;

int g_repeat = 1;
int g_warmup = 0;
std::string g_once_path;
std::string g_encode_path;
bool g_show_tokens = false;

struct Options {
  std::string model_dir, name, host = "127.0.0.1";
  int port = 8000;
  int context = ni::systemone::kRecommendedContext;
  // The state cache's byte cap (0: off): a request that re-sends a state it sent before restores its checkpoint and
  // attention rows instead of prefilling it again. NI_SYS_CACHE=0 turns it off whatever the flag says.
  uint64_t cache_bytes = uint64_t(1024) << 20;
};

void usage() {
  std::fprintf(stderr,
               "usage: systemone --model DIR [--port 8000] [--host 127.0.0.1] [--context 16384] [--name NAME]\n"
               "                 [--cache-mb 1024]\n"
               "                 [--once request.json [--repeat N] [--warmup N]]\n"
               "  DIR: a Kapitsa model folder (safetensors + tokenizer.json) with the Drex pointer head as\n"
               "       DIR/pointer.bin or DIR/.kapitsa/pointer.bin (tools/drex_pointer.py writes it)\n"
               "  --context: the sequence's positions (a request's state plus its longest question row must fit)\n"
               "  --cache-mb: memory for states already prefilled (a follow-up question on the same state then costs a\n"
               "              restore: the reply's timings.cached_tokens says how many positions came from the cache);\n"
               "              0 serves every request cold. NI_SYS_CACHE=0 also disables it\n"
               "  --once: score one request from a file, print the reply, exit (--repeat times, --warmup first\n"
               "          runs unmeasured: the latencies reported are min / median / max ms over the measured runs)\n"
               "  --encode: only the tokenizer is loaded: print the encoded layout of a request and exit\n"
               "            (--tokens also prints the token ids), no GPU and no weights are touched\n");
}

Options parse_args(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") o.model_dir = value("--model");
    else if (a == "--port") o.port = std::atoi(value("--port").c_str());
    else if (a == "--host") o.host = value("--host");
    else if (a == "--context") o.context = std::atoi(value("--context").c_str());
    else if (a == "--name") o.name = value("--name");
    else if (a == "--cache-mb") {
      const long mb = std::atol(value("--cache-mb").c_str());
      o.cache_bytes = mb > 0 ? uint64_t(mb) << 20 : 0;
    }
    else if (a == "--once") g_once_path = value("--once");
    else if (a == "--encode") g_encode_path = value("--encode");
    else if (a == "--tokens") g_show_tokens = true;
    else if (a == "--repeat") g_repeat = std::max(1, std::atoi(value("--repeat").c_str()));
    else if (a == "--warmup") g_warmup = std::max(0, std::atoi(value("--warmup").c_str()));
    else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage();
      std::exit(2);
    }
  }
  if (o.model_dir.empty()) { usage(); std::exit(2); }
  if (o.context < 256) {
    std::fprintf(stderr, "--context must be at least 256\n");
    std::exit(2);
  }
  const char* cache_env = std::getenv("NI_SYS_CACHE");
  if (cache_env && !std::strcmp(cache_env, "0")) o.cache_bytes = 0;
  return o;
}

// One request's GPU time, split into the parts the protocol's latency is made of. cached_tokens: positions of the state
// that came from the state cache instead of a prefill.
struct Timings {
  double state_ms = 0, branch_ms = 0, head_ms = 0;
  int cached_tokens = 0;
  double get() const { return state_ms + branch_ms + head_ms; }
};

// The model's identity for cache keys: its file names and sizes plus config.json, like the disk cache's namespace.
ni::pcache::Key model_digest(const std::string& dir) {
  ni::pcache::Hasher h;
  std::error_code ec;
  std::vector<std::string> names;
  for (const auto& de : std::filesystem::directory_iterator(dir, ec))
    if (de.is_regular_file(ec)) names.push_back(de.path().filename().string());
  std::sort(names.begin(), names.end());
  for (const std::string& n : names) {
    std::error_code e;
    const auto size = std::filesystem::file_size(dir + "/" + n, e);
    h.str(n);
    if (!e) h.u64(uint64_t(size));
  }
  std::ifstream f(dir + "/config.json", std::ios::binary);
  if (f) {
    std::stringstream ss;
    ss << f.rdbuf();
    h.str(ss.str());
  }
  return h.done();
}

// A state's checkpoint and attention rows, so that a request re-sending a state (a follow-up question) restores them
// instead of prefilling the state again. The key is the model plus the state's tokens, so another model or another
// state never matches.
struct CacheEntry {
  ni::pcache::Key key;
  int n_state = 0;
  Checkpoint ck;
  std::vector<uint8_t> kv;  // every attention cache's rows [0, n_state), head by head: heads * n_state * row_bytes
  size_t bytes = 0;
};

size_t checkpoint_bytes(const Checkpoint& c) {
  size_t n = c.pconv ? c.pconv.size : 0;
  for (const Buffer& b : c.conv) n += b.size;
  for (const Buffer& b : c.rec) n += b.size;
  return n;
}

class SystemOne {
 public:
  explicit SystemOne(const Options& o) : opts_(o) {
    name_ = opts_.name;
    if (name_.empty()) {
      name_ = opts_.model_dir;
      while (!name_.empty() && name_.back() == '/') name_.pop_back();
      name_ = name_.substr(name_.find_last_of('/') + 1);
    }
    Gpu& g = Gpu::get();
    logi("loading %s", opts_.model_dir.c_str());
    model_ = load_qwen(opts_.model_dir, 2048);
    if (!model_) fatal("could not load %s", opts_.model_dir.c_str());
    if (!model_->can_snapshot()) {
      fatal("%s: this model's recurrent state is not something a checkpoint captures, and systemone answers every "
            "question by restoring the state's checkpoint; it needs a model that can snapshot (Qwen3.5/3.6/3.8)",
            opts_.model_dir.c_str());
    }
    tok_.load(opts_.model_dir + "/tokenizer.json");
    const std::string head_path = ni::systemone::PointerHead::find(opts_.model_dir);
    if (head_path.empty()) {
      fatal("%s has no pointer head: put pointer.bin there (tools/drex_pointer.py <gguf> <out>)", opts_.model_dir.c_str());
    }
    try {
      head_ = ni::systemone::PointerHead::load(head_path);
    } catch (const std::exception& e) {
      fatal("%s", e.what());
    }
    hidden_ = model_->hidden_width();
    if (head_.hidden() != hidden_) fatal("pointer head expects %d hidden values, the model has %d", head_.hidden(), hidden_);
    norm_w_ = model_->final_norm_weights();
    eps_ = model_->rms_eps();
    limits_ = ni::systemone::limits_from_env();
    seq_ = model_->new_seq(opts_.context);
    if (opts_.context < limits_.max_state + 1) {
      logi("note: --context %d is below the state/row caps (%d): a request that needs more is rejected",
           opts_.context, limits_.max_state);
    }
    // The attention rows above a request's n_state hold what the last question left there. A masked row is never read
    // for its value, but an unwritten one could hold a NaN pattern, so the caches start zeroed.
    const auto z0 = std::chrono::steady_clock::now();
    for (Buffer& b : seq_->kc)
      if (b) std::memset(b.ptr(), 0, b.size);
    for (Buffer& b : seq_->vc)
      if (b) std::memset(b.ptr(), 0, b.size);
    const double zero_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - z0).count();
    ck_ = model_->new_checkpoint();
    model_key_ = model_digest(opts_.model_dir);
    logi("%s ready: context %d, hidden %d, pointer dim %d, KV caches zeroed in %.0f ms", name_.c_str(), opts_.context,
         hidden_, head_.dim(), zero_ms);
    if (opts_.cache_bytes) {
      logi("state cache: up to %.0f MiB (a re-sent state restores instead of prefilling; NI_SYS_CACHE=0 disables)",
           double(opts_.cache_bytes) / (1024.0 * 1024.0));
    } else {
      logi("state cache: off");
    }
  }

  // Scores one {"state", "questions"} object, returning its answers object.
  ojson score(const ojson& body, Timings* tm) {
    Gpu& g = Gpu::get();
    const ni::systemone::Encoded enc = ni::systemone::encode(tok_, body, limits_);
    const int n_state = enc.n_state();
    const int row = n_state + (int) enc.max_branch();
    if (n_state > seq_->limit) {
      throw ni::systemone::BadRequest("state is " + std::to_string(n_state) + " tokens and does not fit the " +
                                      std::to_string(seq_->limit) + "-token context (raise --context)");
    }
    if (row > seq_->limit) {
      throw ni::systemone::BadRequest("a question row is " + std::to_string(row) +
                                      " tokens with the state and does not fit the " + std::to_string(seq_->limit) +
                                      "-token context (raise --context)");
    }
    std::copy(enc.state.begin(), enc.state.end(), seq_->tokens());
    ni::SamplingParams sp;  // nothing is sampled: these passes only advance the state
    const int chunk = model_->prefill_chunk();
    const auto t0 = std::chrono::steady_clock::now();

    // ---- the state, either restored from the cache or prefilled once with its recurrent checkpoint at n_state ----
    const CacheEntry* cached = opts_.cache_bytes ? cache_find(enc.state) : nullptr;
    const Checkpoint* ck = cached ? &cached->ck : &ck_;
    if (cached) {
      tm->cached_tokens += n_state;
      auto e = g.begin();
      model_->restore(*e, *seq_, cached->ck);
      e->commit_and_wait();
      rows_put(n_state, cached->kv);
    } else {
      // A state prefilled on top of the recurrent state the previous request of this process left behind answers from
      // a diluted distribution (the first request of a process is clean and every later one is not), so the state is
      // prefilled from an empty one. A cache hit does not reset: restore() writes the whole recurrent state back from
      // the checkpoint, and the KV rows above n_state stay masked by position, so the KV caches need no reset.
      model_->reset(*seq_);
      {
        auto e = g.begin();
        for (int pos = 0; pos < n_state;) {
          const int T = std::min(chunk, n_state - pos);
          model_->encode_prefill(*e, *seq_, pos, T, sp, 0, false);
          pos += T;
        }
        model_->save(*e, *seq_, ck_, n_state);
        e->commit_and_wait();
      }
      if (opts_.cache_bytes) cache_store(enc.state, n_state);
    }
    tm->state_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    // ---- every question's row: restore, prefill, read the rows the head scores ----
    std::vector<std::vector<float>> logits;
    logits.reserve(enc.branches.size());
    std::vector<float> decide_row, norm_row;
    std::vector<std::vector<float>> option_rows;
    std::vector<const float*> ptrs;
    const auto t1 = std::chrono::steady_clock::now();
    for (size_t q = 0; q < enc.branches.size(); ++q) {
      const ni::systemone::Branch& b = enc.branches[q];
      const int T = (int) b.tokens.size();
      std::copy(b.tokens.begin(), b.tokens.end(), seq_->tokens() + n_state);
      decide_row.clear();
      option_rows.assign(b.group.options.size(), {});
      {
        auto e = g.begin();
        model_->restore(*e, *seq_, *ck);
        for (int off = 0; off < T;) {
          const int rows = std::min(chunk, T - off);
          model_->encode_prefill(*e, *seq_, n_state + off, rows, sp, 0, false);
          e->commit_and_wait();
          const float* x = model_->hidden_rows().as<float>();
          auto take = [&](int index, std::vector<float>* out) {
            if (index < off || index >= off + rows) return;
            ni::systemone::final_norm_row(x + size_t(index - off) * hidden_, norm_w_.data(), hidden_, eps_, out);
          };
          if (b.group.decide >= 0) take(b.group.decide, &decide_row);
          for (size_t o = 0; o < b.group.options.size(); ++o) take(b.group.options[o], &option_rows[o]);
          off += rows;
        }
      }
      if ((int) decide_row.size() != hidden_) throw std::runtime_error("the decide row of a question was not read");
      for (const auto& r : option_rows)
        if ((int) r.size() != hidden_) throw std::runtime_error("an option row of a question was not read");
      ptrs.clear();
      for (const auto& r : option_rows) ptrs.push_back(r.data());
      std::vector<float> group;
      head_.logits(decide_row.data(), ptrs, &group);
      logits.push_back(std::move(group));
      if (q + 1 == enc.branches.size()) {
        tm->branch_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
      }
    }
    const auto t2 = std::chrono::steady_clock::now();
    ojson out = ni::systemone::format_answers(logits, enc.meta);
    tm->head_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t2).count();
    scored_input_tokens_ = (int) enc.total_tokens();
    input_tokens_ = scored_input_tokens_;
    return out;
  }

  // One API request: {"state", "questions"} or {"requests": [...]}, answered in the reference server's envelope.
  ojson handle(const ojson& body) {
    const auto started = std::chrono::steady_clock::now();
    Timings tm;
    ojson answers = ojson::object();
    int output_tokens = 0;
    if (body.is_object() && body.contains("requests")) {
      const ojson& batch = body.at("requests");
      if (!batch.is_array()) throw ni::systemone::BadRequest("requests must be a list");
      if (batch.empty()) throw ni::systemone::BadRequest("requests must not be empty");
      if (batch.size() > 1024) throw ni::systemone::BadRequest("at most 1024 requests per call");
      if (body.contains("state") || body.contains("questions")) {
        throw ni::systemone::BadRequest("send either state and questions or a requests list, not both");
      }
      input_tokens_ = 0;
      for (const ojson& one : batch) {
        const ojson a = score(one, &tm);
        input_tokens_ += scored_input_tokens_;
        for (const auto& [qid, value] : a.items()) answers[qid] = value;
        output_tokens += ni::systemone::output_tokens(tok_, a);
      }
    } else {
      answers = score(body, &tm);
      output_tokens = ni::systemone::output_tokens(tok_, answers);
    }
    const auto ms = [](double v) { return std::round(v * 100.0) / 100.0; };
    ojson reply = ojson::object();
    reply["model"] = name_;
    reply["answers"] = std::move(answers);
    reply["usage"] = ojson{{"input_tokens", input_tokens_},
                             {"output_tokens", output_tokens},
                             {"input_tokens_details", ojson{{"cached_tokens", tm.cached_tokens}}}};
    reply["latency_ms"] = ms(tm.get());
    reply["timings"] = ojson{{"state_ms", ms(tm.state_ms)},
                             {"branch_ms", ms(tm.branch_ms)},
                             {"head_ms", ms(tm.head_ms)},
                             {"cached_tokens", tm.cached_tokens},
                             {"total_ms",
                              ms(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count())}};
    return reply;
  }

  const std::string& name() const { return name_; }
  std::mutex& mutex() { return mu_; }

  // The cache's bookkeeping, for the reply's timings and the log line at exit.
  size_t cache_cost() const { return cache_bytes_used_; }
  int cache_hits() const { return cache_hits_; }
  int cache_misses() const { return cache_misses_; }
  size_t cache_states() const { return cache_.size(); }

 private:
  // ---- the state cache: keyed by the model plus the state's tokens, LRU by bytes ----
  ni::pcache::Key cache_key(const std::vector<int32_t>& tokens) const {
    ni::pcache::Hasher h;
    h.key(model_key_);
    h.u64(uint64_t(tokens.size()));
    h.i32s(tokens.data(), tokens.size());
    return h.done();
  }

  // The entry holding this state, moved to the front of the LRU list (nullptr: a miss).
  const CacheEntry* cache_find(const std::vector<int32_t>& tokens) {
    const ni::pcache::Key key = cache_key(tokens);
    auto it = cache_index_.find(key);
    if (it == cache_index_.end()) {
      ++cache_misses_;
      return nullptr;
    }
    cache_.splice(cache_.begin(), cache_, it->second);
    ++cache_hits_;
    return &cache_.front();
  }

  // Writes the state's checkpoint and attention rows into a new entry, evicting least-recently-used entries for room.
  void cache_store(const std::vector<int32_t>& tokens, int n_state) {
    Gpu& g = Gpu::get();
    CacheEntry entry;
    entry.key = cache_key(tokens);
    if (cache_index_.count(entry.key)) return;  // cannot happen (a hit would have returned), but never duplicate a key
    entry.n_state = n_state;
    entry.ck = model_->new_checkpoint();
    {
      auto e = g.begin();
      model_->save(*e, *seq_, entry.ck, n_state);
      e->commit_and_wait();
    }
    std::vector<KvRows> rows;
    model_->disk_rows(*seq_, false, rows);
    rows_take(rows, n_state, &entry.kv);
    entry.bytes = entry.kv.size() + checkpoint_bytes(entry.ck);
    if (entry.bytes > opts_.cache_bytes) return;  // a single state too big for the cap: serve it, do not keep it
    while (!cache_.empty() && cache_bytes_used_ + entry.bytes > opts_.cache_bytes) {
      cache_bytes_used_ -= cache_.back().bytes;
      cache_index_.erase(cache_.back().key);
      cache_.pop_back();
    }
    cache_bytes_used_ += entry.bytes;
    cache_.push_front(std::move(entry));
    cache_index_[cache_.front().key] = cache_.begin();
  }

  // rows [0, n) of every attention cache, head by head, as the cache keeps them
  void rows_take(const std::vector<KvRows>& rows, int n, std::vector<uint8_t>* out) const {
    const int cap = seq_->cap;
    out->clear();
    for (const KvRows& r : rows) out->resize(out->size() + size_t(r.heads) * size_t(n) * r.row_bytes);
    uint8_t* d = out->data();
    for (const KvRows& r : rows)
      for (int h = 0; h < r.heads; ++h) {
        std::memcpy(d, r.buf.as<uint8_t>() + size_t(h) * size_t(cap) * r.row_bytes, size_t(n) * r.row_bytes);
        d += size_t(n) * r.row_bytes;
      }
  }

  void rows_put(int n, const std::vector<uint8_t>& in) const {
    const int cap = seq_->cap;
    std::vector<KvRows> rows;
    model_->disk_rows(*seq_, false, rows);
    const uint8_t* s = in.data();
    for (const KvRows& r : rows)
      for (int h = 0; h < r.heads; ++h) {
        std::memcpy(r.buf.as<uint8_t>() + size_t(h) * size_t(cap) * r.row_bytes, s, size_t(n) * r.row_bytes);
        s += size_t(n) * r.row_bytes;
      }
  }

  Options opts_;
  std::string name_;
  std::unique_ptr<Qwen35> model_;
  Tokenizer tok_;
  ni::systemone::PointerHead head_;
  std::vector<float> norm_w_;
  float eps_ = 1e-6f;
  int hidden_ = 0;
  int input_tokens_ = 0;         // the reply's usage.input_tokens
  int scored_input_tokens_ = 0;  // the last score() call's, summed over a batch
  ni::systemone::Limits limits_;
  std::unique_ptr<Seq> seq_;
  Checkpoint ck_;
  ni::pcache::Key model_key_;
  std::list<CacheEntry> cache_;                                       // front: most recently used
  std::map<ni::pcache::Key, std::list<CacheEntry>::iterator> cache_index_;
  size_t cache_bytes_used_ = 0;                                       // what the entries take
  int cache_hits_ = 0, cache_misses_ = 0;
  std::mutex mu_;
};

ojson parse_body(const std::string& body) {
  try {
    return ojson::parse(body);
  } catch (const std::exception& e) {
    throw ni::systemone::BadRequest(std::string("request is not JSON: ") + e.what());
  }
}

void on_signal(int) {
  if (g_srv) g_srv->stop();
}

}  // namespace

// --encode: what the encoder makes of a request, without loading weights (the check that a layout is right).
int encode_only(const Options& opts) {
  Tokenizer tok;
  tok.load(opts.model_dir + "/tokenizer.json");
  std::ifstream f(g_encode_path);
  if (!f) fatal("cannot open %s", g_encode_path.c_str());
  std::stringstream ss;
  ss << f.rdbuf();
  const ojson body = parse_body(ss.str());
  const ni::systemone::Encoded enc = ni::systemone::encode(tok, body, ni::systemone::limits_from_env());
  ojson out = ojson::object();
  out["state_tokens"] = enc.n_state();
  out["total_tokens"] = (int) enc.total_tokens();
  ojson qs = ojson::object();
  for (size_t i = 0; i < enc.branches.size(); ++i) {
    const ni::systemone::Branch& b = enc.branches[i];
    const ni::systemone::QuestionMeta& m = enc.meta[i];
    ojson q = ojson::object();
    q["type"] = m.type;
    q["tokens"] = (int) b.tokens.size();
    q["decide"] = b.group.decide;
    q["options"] = b.group.options;
    q["keys"] = m.keys;
    if (!m.legend.empty()) q["legend"] = m.legend;
    if (g_show_tokens) q["ids"] = b.tokens;
    qs[m.id] = q;
  }
  out["questions"] = qs;
  std::printf("%s\n", out.dump(2).c_str());
  return 0;
}

int main(int argc, char** argv) {
  const Options opts = parse_args(argc, argv);
  if (!g_encode_path.empty()) return encode_only(opts);

  // A GPU tool takes the harness lock (src/bench/system.mm GpuLock), like every other tool here.
  ni::bench::GpuLock lock;
  const char* held = std::getenv("NI_LOCK_HELD");
  const char* held2 = std::getenv("NACE_GPU_LOCK_HELD");
  const bool lock_held = (held && std::strcmp(held, "0")) || (held2 && std::strcmp(held2, "0"));
  if (!lock_held) {
    if (!lock.acquire([](const std::string& who) {
          std::fprintf(stderr, "waiting for the GPU lock %s%s\n", ni::bench::GpuLock::path().c_str(),
                       who.empty() ? "" : (", held by " + who).c_str());
        })) {
      fatal("cannot take the GPU lock %s: %s", ni::bench::GpuLock::path().c_str(), std::strerror(errno));
    }
  }

  SystemOne server(opts);

  if (!g_once_path.empty()) {
    std::ifstream f(g_once_path);
    if (!f) fatal("cannot open %s", g_once_path.c_str());
    std::stringstream ss;
    ss << f.rdbuf();
    ojson reply;
    try {
      std::lock_guard<std::mutex> l(server.mutex());
      ojson body = parse_body(ss.str());
      for (int i = 0; i < g_warmup; ++i) server.handle(body);
      std::vector<double> ms;
      for (int i = 0; i < g_repeat; ++i) {
        reply = server.handle(body);
        ms.push_back(reply.value("latency_ms", 0.0));
      }
      std::printf("%s\n", reply.dump(2).c_str());
      if (g_repeat > 1 || g_warmup > 0) {
        std::sort(ms.begin(), ms.end());
        std::fprintf(stderr, "%d measured runs (%d unmeasured): latency ms min %.1f  median %.1f  max %.1f\n",
                     g_repeat, g_warmup, ms.front(), ms[ms.size() / 2], ms.back());
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "request failed: %s\n", e.what());
      return 1;
    }
    return 0;
  }

  httplib::Server srv;
  srv.set_payload_max_length(64u << 20);
  srv.set_read_timeout(600);
  srv.set_write_timeout(600);
  srv.set_idle_interval(0, 10000);

  auto reply_json = [](httplib::Response& res, int status, const ojson& body) {
    res.status = status;
    res.set_content(body.dump(), "application/json");
  };
  auto error_json = [&](httplib::Response& res, int status, const std::string& message) {
    reply_json(res, status, ojson{{"error", message}});
  };

  srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
    reply_json(res, 200, ojson{{"status", "ok"}, {"model", server.name()}});
  });
  srv.Post("/v1/systemone", [&](const httplib::Request& req, httplib::Response& res) {
    ojson body;
    try {
      body = parse_body(req.body);
    } catch (const std::exception& e) {
      error_json(res, 400, e.what());
      return;
    }
    ojson out;
    try {
      std::lock_guard<std::mutex> l(server.mutex());
      out = server.handle(body);
    } catch (const ni::systemone::BadRequest& e) {
      error_json(res, 400, e.what());
      return;
    } catch (const std::exception& e) {
      error_json(res, 500, std::string("systemone scoring failed: ") + e.what());
      return;
    }
    reply_json(res, 200, out);
  });

  g_srv = &srv;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  logi("listening on http://%s:%d (POST /v1/systemone, GET /health)", opts.host.c_str(), opts.port);
  if (!srv.listen(opts.host, opts.port)) {
    fatal("cannot listen on %s:%d (is the port taken?)", opts.host.c_str(), opts.port);
  }
  logi("state cache: %d hits, %d misses, %.0f MiB held for %zu states", server.cache_hits(), server.cache_misses(),
       double(server.cache_cost()) / (1024.0 * 1024.0), server.cache_states());
  logi("stopped");
  return 0;
}
