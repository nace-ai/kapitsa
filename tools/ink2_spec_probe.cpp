// Standalone greedy Inkling drafter measurement. Takes the shared GPU lock before loading weights.
// Commit either replays verified inputs or keeps their prefix; both paths time the state update.
// --row-check: greedy decode steps keep every row's logits, then encode_verify_rows passes of --block rows walk the
// same tokens from the same prefill (the Engine's chunks) and every verify row must be bitwise the step row at its
// position. Exit 1 when one is not.
#include "bench/bench.h"
#include "engine/engine.h"
#include "model/ink2_draft.h"
#include "model/inkling2.h"
#include "util/log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ni;

namespace {

struct Options {
  std::string model, request, mode = "both", commit = "replay", output;
  int tokens = 0, block = 8, chunk = 512, keep = 0;
  bool row_check = false;
};

int integer(const std::string& s) {
  size_t end = 0;
  const int n = std::stoi(s, &end);
  if (end != s.size()) throw std::runtime_error("not an integer: " + s);
  return n;
}

Options options(int argc, char** argv) {
  if (argc < 3)
    throw std::runtime_error("usage: ink2_spec_probe MODEL_DIR REQUEST.json [--tokens N] "
                             "[--mode baseline|spec|both] [--commit replay|prefix] "
                             "[--block 2..8] [--chunk 1..512] [--json PATH]\n"
                             "       ink2_spec_probe MODEL_DIR REQUEST.json --row-check [--tokens N] [--block 1..8] "
                             "[--keep 1..block] [--chunk 1..4096] [--json PATH]");
  Options o;
  o.model = argv[1];
  o.request = argv[2];
  for (int i = 3; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--row-check") {
      o.row_check = true;
      continue;
    }
    if (++i == argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[i];
    if (key == "--tokens") o.tokens = integer(value);
    else if (key == "--block") o.block = integer(value);
    else if (key == "--chunk") o.chunk = integer(value);
    else if (key == "--mode") o.mode = value;
    else if (key == "--commit") o.commit = value;
    else if (key == "--json") o.output = value;
    else if (key == "--keep") o.keep = integer(value);
    else throw std::runtime_error("unknown option: " + key);
  }
  if (o.row_check) {
    // passes of `block` rows that commit their first `keep` (the rest is rolled back, as a rejected draft is)
    if (!o.keep) o.keep = o.block;
    if (o.tokens < 0 || o.block < 1 || o.block > 8 || o.keep < 1 || o.keep > o.block || o.chunk < 1 || o.chunk > 4096)
      throw std::runtime_error("invalid tokens, block, keep or chunk for --row-check");
    return o;
  }
  if (o.keep) throw std::runtime_error("--keep needs --row-check");
  if (o.tokens < 0 || o.block < 2 || o.block > 8 || o.chunk < 1 || o.chunk > 512 ||
      (o.mode != "baseline" && o.mode != "spec" && o.mode != "both") ||
      (o.commit != "replay" && o.commit != "prefix"))
    throw std::runtime_error("invalid tokens, block, chunk, mode or commit");
  return o;
}

ojson read_request(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open request: " + path);
  const ojson req = ojson::parse(in);
  if (!req.is_object() || !req.contains("messages") || !req["messages"].is_array() || req["messages"].empty())
    throw std::runtime_error("request needs a nonempty messages array");
  if (req.value("temperature", 0.0) != 0.0)
    throw std::runtime_error("this probe supports temperature=0 only");
  if ((req.contains("stop") && !req["stop"].empty()) || req.value("presence_penalty", 0.0) != 0.0 ||
      req.value("frequency_penalty", 0.0) != 0.0 || req.value("repetition_penalty", 1.0) != 1.0)
    throw std::runtime_error("stop strings and logit penalties are not supported");
  if (req.contains("logit_bias") && !req["logit_bias"].empty())
    throw std::runtime_error("logit_bias is not supported");
  return req;
}

std::string prompt_text(const ojson& req) {
  ChatRequest r;
  r.messages = req.at("messages");
  r.tools = req.value("tools", ojson::array());
  r.template_kwargs = req.value("chat_template_kwargs", ojson::object());
  if (!r.template_kwargs.is_object()) throw std::runtime_error("chat_template_kwargs must be an object");
  if (req.contains("reasoning_effort")) r.template_kwargs["reasoning_effort"] = req["reasoning_effort"];
  return render_inkling2_chat(r);
}

int argmax(const float* row, int vocab) {
  int best = 0;
  for (int t = 0; t < vocab; ++t) {
    if (!std::isfinite(row[t])) throw std::runtime_error("nonfinite target logits");
    if (row[t] > row[best]) best = t;
  }
  return best;
}

struct Run {
  std::vector<int> tokens;
  int rounds = 0, proposed = 0, accepted = 0;
  double prefill_ms = 0, decode_ms = 0, draft_ms = 0, verify_ms = 0, replay_ms = 0, commit_ms = 0;
  bool serial_replay_checked = false;
  size_t draft_sequence_bytes = 0;
  std::vector<int> accepted_per_round;
  ojson replay_flips = ojson::array();
};

Run generate(Inkling2& target, Ink2Draft* draft, Seq& seq, const std::vector<int>& prompt,
             const Options& o, const Buffer& captures, const Buffer& committed_captures, const Buffer& logits,
             const Buffer& replay_logits, bool ignore_eos) {
  Gpu& gpu = Gpu::get();
  Run result;
  target.reset(seq);
  auto draft_seq = draft ? draft->new_seq() : nullptr;
  std::copy(prompt.begin(), prompt.end(), seq.tokens());
  const int vocab = target.vocab(), hidden = target.config().hidden;
  const auto cap_rows = [&](int rows) { return captures.view(0, size_t(4) * rows * hidden * sizeof(float)); };
  const auto is_eos = [&](int id) {
    const auto& eos = target.eos_ids();
    return !ignore_eos && std::find(eos.begin(), eos.end(), id) != eos.end();
  };
  SamplingParams greedy;
  greedy.temperature = 0;
  greedy.top_p = 1;
  greedy.top_k = 1;
  double begin = now_ms();
  int pos = 0;
  while (pos < int(prompt.size()) - 1) {
    const int rows = std::min(o.chunk, int(prompt.size()) - 1 - pos);
    auto e = gpu.begin();
    if (draft) {
      const Buffer cap = cap_rows(rows);
      target.encode_features_rows(*e, seq, pos, rows, cap);
      draft->observe(*e, *draft_seq, cap, rows, pos);
    } else {
      target.encode_prefill(*e, seq, pos, rows, greedy, 0, false);
    }
    e->commit_and_wait();
    pos += rows;
  }
  {
    auto e = gpu.begin();
    if (draft) {
      const Buffer cap = cap_rows(1);
      target.encode_features_rows(*e, seq, pos, 1, cap, &replay_logits);
      draft->observe(*e, *draft_seq, cap, 1, pos);
    } else {
      target.encode_logits_rows(*e, seq, pos, 1, replay_logits);
    }
    e->commit_and_wait();
    ++pos;
  }
  result.prefill_ms = now_ms() - begin;
  begin = now_ms();
  result.tokens.push_back(argmax(replay_logits.as<float>(), vocab));
  Checkpoint checkpoint = target.new_checkpoint();
  while (int(result.tokens.size()) < o.tokens && !is_eos(result.tokens.back())) {
    seq.tokens()[pos] = result.tokens.back();
    if (!draft) {
      auto e = gpu.begin();
      target.encode_logits_rows(*e, seq, pos, 1, replay_logits);
      e->commit_and_wait();
      result.tokens.push_back(argmax(replay_logits.as<float>(), vocab));
      ++pos;
      ++result.rounds;
      continue;
    }

    const int count = std::min(o.block - 1, o.tokens - int(result.tokens.size()));
    double phase = now_ms();
    const std::vector<int> guesses = draft->propose(*draft_seq, result.tokens.back(), count);
    result.draft_ms += now_ms() - phase;
    if (int(guesses.size()) != count) throw std::runtime_error("drafter returned the wrong proposal length");
    for (int id : guesses)
      if (id < 0 || id >= vocab) throw std::runtime_error("drafter returned an out-of-vocabulary token");
    std::copy(guesses.begin(), guesses.end(), seq.tokens() + pos + 1);
    const int rows = count + 1;
    std::optional<Inkling2::VerifyTicket> ticket;
    phase = now_ms();
    {
      auto e = gpu.begin();
      const Buffer cap = cap_rows(rows);
      if (o.commit == "prefix") {
        ticket.emplace(target.encode_verify_rows(*e, seq, pos, rows, cap, logits));
      } else {
        target.save(*e, seq, checkpoint, pos);
        if (checkpoint.pos != pos) throw std::runtime_error("target checkpoint allocation failed");
        target.encode_features_rows(*e, seq, pos, rows, cap, &logits);
      }
      e->commit_and_wait();
    }
    result.verify_ms += now_ms() - phase;
    std::vector<int> predictions;
    for (int row = 0; row < rows; ++row)
      predictions.push_back(argmax(logits.as<float>() + size_t(row) * vocab, vocab));

    std::vector<int> emitted;
    int accepted = 0;
    while (accepted < count && guesses[size_t(accepted)] == predictions[size_t(accepted)]) {
      emitted.push_back(guesses[size_t(accepted)]);
      ++accepted;
      if (is_eos(emitted.back())) break;
    }
    if (int(result.tokens.size() + emitted.size()) < o.tokens &&
        (emitted.empty() || !is_eos(emitted.back())))
      emitted.push_back(predictions[size_t(accepted)]);

    // Every emitted token consumes one input. The final output remains the next unfed anchor.
    phase = now_ms();
    if (o.commit == "prefix") {
      const int k = int(emitted.size());
      const size_t prefix_bytes = size_t(k) * hidden * sizeof(float);
      const size_t source_stride = size_t(rows) * hidden * sizeof(float);
      for (size_t tap = 0; tap < 4; ++tap)
        std::memcpy(committed_captures.base + committed_captures.offset + tap * prefix_bytes,
                    captures.base + captures.offset + tap * source_stride, prefix_bytes);
      const Buffer cap = committed_captures.view(0, 4 * prefix_bytes);
      auto e = gpu.begin();
      target.commit_verify_prefix(*e, seq, std::move(*ticket), k);
      draft->observe(*e, *draft_seq, cap, k, pos);
      e->commit_and_wait();
      result.commit_ms += now_ms() - phase;
    } else {
      {
        auto e = gpu.begin();
        target.restore(*e, seq, checkpoint);
        e->commit_and_wait();
      }
      for (int row = 0; row < int(emitted.size()); ++row) {
        auto e = gpu.begin();
        const Buffer cap = cap_rows(1);
        target.encode_features_rows(*e, seq, pos + row, 1, cap, &replay_logits);
        draft->observe(*e, *draft_seq, cap, 1, pos + row);
        e->commit_and_wait();
        result.serial_replay_checked = true;
        const int serial = argmax(replay_logits.as<float>(), vocab);
        if (serial != predictions[size_t(row)])
          result.replay_flips.push_back({{"position", pos + row}, {"wide", predictions[size_t(row)]}, {"serial", serial}});
      }
      result.replay_ms += now_ms() - phase;
    }
    ++result.rounds;
    result.proposed += count;
    result.accepted += accepted;
    result.accepted_per_round.push_back(accepted);
    if (!result.replay_flips.empty()) break;
    result.tokens.insert(result.tokens.end(), emitted.begin(), emitted.end());
    pos += int(emitted.size());
    if (draft_seq->position() != pos) throw std::runtime_error("drafter and target positions diverged");
    fprintf(stderr, "KFlash: round=%d proposed=%d accepted=%d generated=%zu\n",
            result.rounds, count, accepted, result.tokens.size());
  }
  result.decode_ms = now_ms() - begin;
  if (draft_seq) result.draft_sequence_bytes = draft_seq->bytes();
  return result;
}

ojson summary(const Run& r, const Tokenizer& tok, bool validation_available) {
  const int decoded = std::max(0, int(r.tokens.size()) - 1);
  return {{"valid", validation_available ? ojson(r.replay_flips.empty()) : ojson(nullptr)},
          {"tokens", r.tokens}, {"text", utf8_valid(tok.decode(r.tokens))},
          {"generated_tokens", r.tokens.size()}, {"decode_tokens_excluding_first", decoded},
          {"prefill_ms", r.prefill_ms}, {"decode_ms", r.decode_ms},
          {"decode_tok_s", r.decode_ms > 0 ? 1000.0 * decoded / r.decode_ms : 0.0},
          {"rounds", r.rounds}, {"proposed", r.proposed}, {"accepted", r.accepted},
          {"acceptance_fraction", r.proposed ? double(r.accepted) / r.proposed : 0.0},
          {"accepted_per_round", r.accepted_per_round}, {"draft_ms", r.draft_ms},
          {"verify_ms", r.verify_ms}, {"replay_ms", r.replay_ms}, {"commit_ms", r.commit_ms},
          {"serial_replay_checked", r.serial_replay_checked},
          {"wide_vs_serial_top1_flips", r.serial_replay_checked ? r.replay_flips : ojson(nullptr)}};
}

struct Top2 {
  int id1 = -1, id2 = -1, nonfinite = 0;
  float v1 = 0, v2 = 0;
};

// The two largest finite logits; the lower id wins a tie, as argmax() does
Top2 top2(const float* row, int vocab) {
  Top2 t;
  for (int i = 0; i < vocab; ++i) {
    const float v = row[i];
    if (!std::isfinite(v)) {
      ++t.nonfinite;
    } else if (t.id1 < 0 || v > t.v1) {
      t.id2 = t.id1, t.v2 = t.v1, t.id1 = i, t.v1 = v;
    } else if (t.id2 < 0 || v > t.v2) {
      t.id2 = i, t.v2 = v;
    }
  }
  return t;
}

ojson top2_json(const Top2& t) {
  return {{"top1_id", t.id1}, {"top1", t.v1}, {"top2_id", t.id2}, {"top2", t.v2}, {"gap", t.v1 - t.v2},
          {"nonfinite", t.nonfinite}};
}

// The Engine's prefill of a cold request: its chunks (prefill_chunk_rows), the last one sampling the first token
int prefill_engine(Inkling2& target, Seq& seq, const std::vector<int>& prompt, int chunk, const SamplingParams& sp,
                   std::vector<int>* chunks) {
  Gpu& gpu = Gpu::get();
  target.reset(seq);
  std::copy(prompt.begin(), prompt.end(), seq.tokens());
  const int n = int(prompt.size());
  for (int p = 0; p < n;) {
    const int rows = prefill_chunk_rows(chunk, p, n, prefill_even_chunks());
    auto e = gpu.begin();
    target.encode_prefill(*e, seq, p, rows, sp, 0, p + rows == n);
    e->commit_and_wait();
    if (chunks) chunks->push_back(rows);
    p += rows;
  }
  return seq.tokens()[n];
}

struct RowDiff {
  int position, pass_start, row, rows, differing = 0, pick = -1, expected = -1;
  bool equal = false;
  float max_abs = 0;
  Top2 step, verify;
};

ojson row_json(const RowDiff& d) {
  return {{"position", d.position}, {"pass_start", d.pass_start}, {"row_in_pass", d.row}, {"rows_in_pass", d.rows},
          {"bitwise_equal", d.equal}, {"max_abs_diff", d.max_abs}, {"differing_logits", d.differing},
          {"step", top2_json(d.step)}, {"verify", top2_json(d.verify)}, {"verify_pick", d.pick},
          {"plain_next_token", d.expected}};
}

// Plain greedy decoding keeps every step row's logits; encode_verify_rows passes then walk the same tokens on a fresh
// prefill, as the Engine's verify does (a grouped encoder, encode_greedy_rows, commit_verify_prefix), and each verify
// row is compared bitwise with the step row at its position. Returns the exit code: 0 when every row is equal.
int row_check(Inkling2& target, Seq& seq, const std::vector<int>& prompt, const Options& o, const Buffer& captures,
              const Buffer& logits, const Buffer& picks, bool ignore_eos, const Tokenizer& tok, ojson& out) {
  Gpu& gpu = Gpu::get();
  const int vocab = target.vocab(), n = int(prompt.size()), hidden = target.config().hidden;
  const size_t vocab_bytes = size_t(vocab) * sizeof(float);
  SamplingParams greedy;
  greedy.temperature = 0;
  greedy.top_p = 1;
  greedy.top_k = 1;
  const auto is_eos = [&](int id) {
    const auto& eos = target.eos_ids();
    return !ignore_eos && std::find(eos.begin(), eos.end(), id) != eos.end();
  };

  // Plain decoding: encode_step, the Engine's decode step; step row i is position n + i with input tokens[i]
  double begin = now_ms();
  std::vector<int> chunks;
  std::vector<int> tokens{prefill_engine(target, seq, prompt, o.chunk, greedy, &chunks)};
  std::vector<float> steps;
  std::vector<Top2> step_top;
  steps.reserve(size_t(std::max(0, o.tokens - 1)) * vocab);
  ojson step_pick_flips = ojson::array();
  for (int pos = n; int(tokens.size()) < o.tokens && !is_eos(tokens.back()); ++pos) {
    seq.tokens()[pos] = tokens.back();
    auto e = gpu.begin();
    target.encode_step(*e, seq, pos, greedy, uint32_t(pos), true);
    e->commit_and_wait();
    const float* row = target.logits();
    steps.insert(steps.end(), row, row + vocab);
    step_top.push_back(top2(row, vocab));
    const int next = seq.tokens()[pos + 1];
    if (next != step_top.back().id1)
      step_pick_flips.push_back({{"position", pos}, {"gpu_pick", next}, {"host_argmax", step_top.back().id1}});
    tokens.push_back(next);
  }
  const double plain_ms = now_ms() - begin;

  // The verify walk over the same inputs: passes of o.block rows, the first o.keep committed
  begin = now_ms();
  const int first = prefill_engine(target, seq, prompt, o.chunk, greedy, nullptr);
  const int end = n + int(tokens.size()) - 1;  // positions that have a step row
  std::copy(tokens.begin(), tokens.end(), seq.tokens() + n);
  std::vector<RowDiff> diffs;
  int passes = 0;
  for (int P = n; P < end;) {
    const int rows = std::min(o.block, end - P), keep = std::min(o.keep, rows);
    const Buffer cap = captures.view(0, size_t(4) * rows * hidden * sizeof(float));
    const Buffer lg = logits.view(0, size_t(rows) * vocab_bytes);
    std::optional<Inkling2::VerifyTicket> ticket;
    {
      auto e = gpu.begin_grouped();
      ticket.emplace(target.encode_verify_rows(*e, seq, P, rows, cap, lg));
      target.encode_greedy_rows(*e, seq, lg, rows, picks);
      e->commit_and_wait();
    }
    for (int r = 0; r < rows; ++r) {
      const int i = P + r - n;
      const float* a = steps.data() + size_t(i) * vocab;
      const float* b = logits.as<float>() + size_t(r) * vocab;
      RowDiff d{P + r, P, r, rows};
      d.equal = std::memcmp(a, b, vocab_bytes) == 0;
      for (int t = 0; !d.equal && t < vocab; ++t) {
        if (std::memcmp(a + t, b + t, sizeof(float)) == 0) continue;
        ++d.differing;
        const float diff = std::fabs(a[t] - b[t]);
        d.max_abs = std::isnan(diff) ? std::numeric_limits<float>::infinity() : std::max(d.max_abs, diff);
      }
      d.step = step_top[size_t(i)];
      d.verify = d.equal ? d.step : top2(b, vocab);
      d.pick = picks.as<int>()[r];
      d.expected = tokens[size_t(i) + 1];
      diffs.push_back(d);
    }
    {
      auto e = gpu.begin_grouped();
      target.commit_verify_prefix(*e, seq, std::move(*ticket), keep);
      e->commit_and_wait();
    }
    ++passes;
    P += keep;
  }
  const double verify_ms = now_ms() - begin;

  int equal = 0;
  ojson flips = ojson::array(), pick_flips = ojson::array(), all = ojson::array();
  std::vector<const RowDiff*> unequal;
  for (const RowDiff& d : diffs) {
    equal += d.equal;
    if (!d.equal) unequal.push_back(&d);
    if (d.verify.id1 != d.step.id1) flips.push_back(row_json(d));
    if (d.pick != d.expected) pick_flips.push_back(row_json(d));
    all.push_back({d.position, d.row, d.rows, d.equal, d.max_abs, d.differing});
  }
  std::stable_sort(unequal.begin(), unequal.end(), [](const RowDiff* x, const RowDiff* y) { return x->max_abs > y->max_abs; });
  ojson worst = ojson::array();
  for (size_t k = 0; k < unequal.size() && k < 10; ++k) worst.push_back(row_json(*unequal[k]));
  ojson min_gap = nullptr;
  for (size_t i = 0; i < step_top.size(); ++i) {
    const float gap = step_top[i].v1 - step_top[i].v2;
    if (min_gap.is_null() || gap < min_gap["gap"].get<float>()) {
      min_gap = top2_json(step_top[i]);
      min_gap["position"] = n + int(i);
    }
  }
  const char* exact = getenv("NI_INK2_VERIFY_EXACT");
  const char* profile = getenv("KAPITSA_PROFILE");
  const bool valid = first == tokens[0] && equal == int(diffs.size()) && pick_flips.empty();
  out["row_check"] = {
      {"block", o.block}, {"keep", o.keep}, {"prefill_chunks", chunks},
      {"KAPITSA_PROFILE", profile ? ojson(profile) : ojson(nullptr)},
      {"NI_INK2_VERIFY_EXACT", exact ? ojson(exact) : ojson(nullptr)},
      {"plain_tokens", tokens}, {"plain_text", utf8_valid(tok.decode(tokens))}, {"plain_ms", plain_ms},
      {"verify_ms", verify_ms}, {"verify_passes", passes},
      {"first_token_same", first == tokens[0]},
      {"rows_compared", diffs.size()}, {"rows_bitwise_equal", equal},
      {"positions_compared", std::max(0, end - n)},
      {"top1_differs", flips}, {"verify_pick_differs_from_plain", pick_flips},
      {"step_gpu_pick_differs_from_host_argmax", step_pick_flips},
      {"worst_rows", worst}, {"min_step_top1_top2_gap", min_gap},
      {"rows", all}, {"rows_columns", {"position", "row_in_pass", "rows_in_pass", "bitwise_equal", "max_abs_diff",
                                       "differing_logits"}}};
  out["valid"] = valid;
  fprintf(stderr, "ink2_spec_probe: row check %d of %zu verify rows bitwise equal, %zu top-1 flips, %s\n", equal,
          diffs.size(), flips.size(), valid ? "PASS" : "FAIL");
  return valid ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options o = options(argc, argv);
    const ojson request = read_request(o.request);
    if (!o.tokens) o.tokens = request.value("max_tokens", 256);
    if (o.tokens < 1) throw std::runtime_error("tokens must be positive");
    Tokenizer tok;
    tok.load(o.model + "/tokenizer.json");
    const std::vector<int> prompt = tok.encode(prompt_text(request));
    if (prompt.empty() || prompt.size() + size_t(o.tokens) + 8 > size_t(std::numeric_limits<int>::max()))
      throw std::runtime_error("empty prompt or context size overflows int");
    bench::GpuLock lock;
    if (!bench::GpuLock::held_by_caller() && !lock.acquire()) throw std::runtime_error("cannot acquire GPU lock");
    Gpu& gpu = Gpu::get();
    Inkling2 target(o.model, std::max(8, o.chunk));
    const int limit = int(prompt.size()) + o.tokens + 8;
    auto seq = target.new_seq(limit, limit);
    if (seq->cap < limit || !target.reserve(*seq, limit)) throw std::runtime_error("cannot reserve complete context");
    const int capture_rows = o.row_check ? 8 : std::max(8, o.chunk);
    const Buffer captures = gpu.alloc(size_t(4) * capture_rows * target.config().hidden * sizeof(float), "probe_features");
    const Buffer logits = gpu.alloc(size_t(8) * target.vocab() * sizeof(float), "probe_verify_logits");
    const Buffer replay_logits = gpu.alloc(size_t(target.vocab()) * sizeof(float), "probe_replay_logits");
    gpu.make_resident({captures, logits, replay_logits});
    Buffer committed_captures;
    if (o.commit == "prefix" && o.mode != "baseline") {
      committed_captures = gpu.alloc(size_t(4) * 8 * target.config().hidden * sizeof(float), "probe_committed_features");
      gpu.make_resident({committed_captures});
    }
    const bool ignore_eos = request.value("ignore_eos", false);
    if (o.row_check) {
      const Buffer picks = gpu.alloc(size_t(8) * sizeof(int), "probe_verify_picks");
      gpu.make_resident({picks});
      ojson out = {{"tool", "ink2_spec_probe"}, {"mode", "row-check"}, {"model", o.model}, {"request", o.request},
                   {"device", gpu.name()}, {"prompt_tokens", prompt.size()}, {"max_tokens", o.tokens},
                   {"prefill_chunk", o.chunk}, {"sampling", "greedy"}, {"ignore_eos", ignore_eos}};
      const int code = row_check(target, *seq, prompt, o, captures, logits, picks, ignore_eos, tok, out);
      const std::string text = out.dump(2) + "\n";
      if (!o.output.empty()) {
        std::ofstream file(o.output);
        if (!file || !(file << text)) throw std::runtime_error("cannot write JSON: " + o.output);
      }
      fputs(text.c_str(), stdout);
      return code;
    }
    ojson out = {{"tool", "ink2_spec_probe"}, {"model", o.model}, {"request", o.request},
                 {"device", gpu.name()}, {"mode", o.mode}, {"commit_mode", o.commit}, {"prompt_tokens", prompt.size()},
                 {"max_tokens", o.tokens}, {"block", o.block}, {"prefill_chunk", o.chunk},
                 {"sampling", "greedy"}, {"ignore_eos", ignore_eos}, {"interleaved_benchmark", false}};
    Run baseline, spec;
    bool valid = true, validation_available = true;
    if (o.mode != "spec") {
      baseline = generate(target, nullptr, *seq, prompt, o, captures, committed_captures, logits, replay_logits, ignore_eos);
      out["baseline"] = summary(baseline, tok, true);
    }
    if (o.mode != "baseline") {
      Ink2Draft draft(o.model, target.draft_embedding(), target.draft_embed_norm(), target.draft_unembed());
      spec = generate(target, &draft, *seq, prompt, o, captures, committed_captures, logits, replay_logits, ignore_eos);
      validation_available = o.commit == "replay";
      out["spec"] = summary(spec, tok, validation_available);
      out["draft_shared_bytes"] = draft.bytes();
      out["draft_sequence_bytes"] = spec.draft_sequence_bytes;
      out["draft_bytes"] = draft.bytes() + spec.draft_sequence_bytes;
      valid = validation_available && spec.replay_flips.empty();
    }
    if (o.mode == "both") {
      const bool equal = baseline.tokens == spec.tokens;
      out["tokens_equal"] = equal;
      validation_available = true;
      valid = equal && (o.commit == "prefix" || spec.replay_flips.empty());
      out["spec"]["valid"] = valid;
      size_t first = 0;
      while (first < std::min(baseline.tokens.size(), spec.tokens.size()) && baseline.tokens[first] == spec.tokens[first]) ++first;
      out["common_prefix_tokens"] = first;
    }
    out["serial_replay_included_in_decode_time"] = spec.serial_replay_checked;
    out["valid"] = validation_available ? ojson(valid) : ojson(nullptr);
    const std::string text = out.dump(2) + "\n";
    if (!o.output.empty()) {
      std::ofstream file(o.output);
      if (!file || !(file << text)) throw std::runtime_error("cannot write JSON: " + o.output);
    }
    fputs(text.c_str(), stdout);
    return valid ? 0 : 1;
  } catch (const std::exception& e) {
    fprintf(stderr, "ink2_spec_probe: %s\n", e.what());
    return 2;
  }
}
