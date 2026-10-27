// CPU test of src/splash/protocol.{h,cpp} against bytes Splash 1.4.0's own protocol.py produced and accepted
// (tools/splash_protocol_golden.h, regenerate with tools/splash_protocol_golden.py). No model, no GPU, no child process.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "splash/protocol.h"
#include "splash_protocol_golden.h"

using namespace ni::splash;

static int failures = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } \
  } while (0)

static std::vector<uint8_t> unhex(const char* h) {
  std::vector<uint8_t> v;
  for (size_t i = 0; h[i] && h[i + 1]; i += 2) {
    auto nib = [](char c) { return c <= '9' ? c - '0' : c - 'a' + 10; };
    v.push_back(uint8_t(nib(h[i]) << 4 | nib(h[i + 1])));
  }
  return v;
}
static std::vector<uint8_t> golden(const GoldenBytes* t, size_t n, const char* name) {
  for (size_t i = 0; i < n; i++)
    if (!std::strcmp(t[i].name, name)) return unhex(t[i].hex);
  std::printf("FAIL no golden %s\n", name);
  failures++;
  return {};
}
#define CLIENT(n) golden(kGoldenClient, sizeof kGoldenClient / sizeof *kGoldenClient, n)
#define EVENT(n) golden(kGoldenEvents, sizeof kGoldenEvents / sizeof *kGoldenEvents, n)

static Request base_request() {
  Request r;
  r.request_id = 7;
  r.absolute_deadline_unix_micros = 1'700'000'000'000'000ull;
  r.remaining_deadline_micros = 30'000'000;
  r.max_output_tokens = 256;
  r.prompt = {1, 2, 3, 248045, 5};
  return r;
}

static Event parse_one(const std::vector<uint8_t>& frame) {
  FrameParser p;
  p.feed(frame.data(), frame.size());
  Event e;
  CHECK(p.next(e));
  CHECK(p.buffered() == 0);
  return e;
}

static void test_client_frames() {
  CHECK(encode_request(base_request()) == CLIENT("request_greedy"));

  Request s = base_request();
  s.request_id = 0x1122334455667788ull;
  s.priority = Priority::kForeground;
  s.max_output_tokens = 1u << 20;
  s.sampling = {0.7f, 0.95f, 40, 0.5f, -0.25f, 1.1f, 0.05f};
  s.seed = ~0ull - 1;
  s.generation_prompt_tokens = 3;
  s.flags = kIgnoreEndOfSequence | kReturnProgress;
  s.cache_salt_lo = 0x0102030405060708ull;
  s.cache_salt_hi = 0xAABBCCDDull;
  CHECK(encode_request(s) == CLIENT("request_sampled"));

  Request t = base_request();
  t.request_id = 9;
  t.generation_prompt_tokens = 2;
  t.thinking = ThinkingBudget{512, {11, 248046}, 248047};
  CHECK(encode_request(t) == CLIENT("request_thinking"));

  CHECK(encode_cancel(7) == CLIENT("cancel"));
  CHECK(encode_status_request(0xFEEDFACE) == CLIENT("status_request"));
  CHECK(encode_release_request(5) == CLIENT("release_request"));

  // A refreshed deadline is now + remaining, and nothing else moves.
  auto f = CLIENT("request_greedy");
  auto before = f;
  refresh_request_deadline(f, 2'000'000'000'000'000ull);
  CHECK(f != before);
  uint64_t abs = 0;
  for (int i = 0; i < 8; i++) abs |= uint64_t(f[kFrameHeaderBytes + 10 + i]) << (8 * i);
  CHECK(abs == 2'000'000'000'000'000ull + 30'000'000ull);
  f = before;
  for (size_t i = 0; i < f.size(); i++)
    if (i < kFrameHeaderBytes + 10 || i >= kFrameHeaderBytes + 18) CHECK(f[i] == before[i]);
  auto cancel = CLIENT("cancel");
  auto cancel0 = cancel;
  refresh_request_deadline(cancel, 5);
  CHECK(cancel == cancel0);
}

static void expect_request_issue(Request r, FailureClass fc, IssueCode code, const char* what) {
  try {
    encode_request(r);
    std::printf("FAIL %s: encoded\n", what);
    failures++;
  } catch (const ProtocolError& e) {
    if (e.issue.failure != fc || e.issue.code != code) {
      std::printf("FAIL %s: %s\n", what, e.what());
      failures++;
    }
  }
}

static void test_request_validation() {
  using F = FailureClass;
  using C = IssueCode;
  Request r;
  r = base_request(); r.absolute_deadline_unix_micros = 0;
  expect_request_issue(r, F::kRequestError, C::kInvalidDeadline, "zero deadline");
  r = base_request(); r.max_output_tokens = 0;
  expect_request_issue(r, F::kRequestError, C::kLimitExceeded, "no output");
  r = base_request(); r.max_output_tokens = (1u << 20) + 1;
  expect_request_issue(r, F::kRequestError, C::kLimitExceeded, "output limit");
  r = base_request(); r.prompt.clear();
  expect_request_issue(r, F::kRequestError, C::kLimitExceeded, "empty prompt");
  r = base_request(); r.generation_prompt_tokens = 5;
  expect_request_issue(r, F::kRequestError, C::kInvalidCount, "generation prompt eats the prompt");
  r = base_request(); r.flags = 1u << 5;
  expect_request_issue(r, F::kRequestError, C::kInvalidEnumValue, "unknown flag");
  r = base_request(); r.sampling.top_p = 0.0f;
  expect_request_issue(r, F::kRequestError, C::kInvalidSampling, "top_p 0");
  r = base_request(); r.sampling.temperature = 1e-40f;  // subnormal: Metal flushes it to zero
  expect_request_issue(r, F::kRequestError, C::kInvalidSampling, "subnormal temperature");
  r = base_request(); r.sampling.temperature = -1.0f;
  expect_request_issue(r, F::kRequestError, C::kInvalidSampling, "negative temperature");
  r = base_request(); r.sampling.presence_penalty = 2.5f;
  expect_request_issue(r, F::kRequestError, C::kInvalidSampling, "presence penalty");
  r = base_request(); r.sampling.repetition_penalty = 0.0f;
  expect_request_issue(r, F::kRequestError, C::kInvalidSampling, "repetition penalty");
  r = base_request(); r.sampling.min_p = 1.5f;
  expect_request_issue(r, F::kRequestError, C::kInvalidSampling, "min_p");
  r = base_request(); r.thinking = ThinkingBudget{10, {}, 0};
  expect_request_issue(r, F::kRequestError, C::kInvalidCount, "thinking without an end");
  r = base_request(); r.thinking = ThinkingBudget{10, {5, 6, 5}, 0};
  expect_request_issue(r, F::kRequestError, C::kInvalidCount, "thinking end repeats its last token");
  r = base_request(); r.thinking = ThinkingBudget{10, {5, 6}, 5};
  expect_request_issue(r, F::kRequestError, C::kInvalidCount, "also_ends inside the end");
  r = base_request(); r.thinking = ThinkingBudget{0xFFFFFFFFu, {5}, 0};
  expect_request_issue(r, F::kRequestError, C::kInvalidCount, "thinking budget is the no-budget marker");
  // An id of zero cannot name a request: the stream fails instead, whatever else is right or wrong with the request
  // (protocol.py looks at the id first)
  r = base_request(); r.request_id = 0;
  expect_request_issue(r, F::kProtocolFatal, C::kInvalidRequestId, "no request id");
  r = base_request(); r.request_id = 0; r.max_output_tokens = 0;
  expect_request_issue(r, F::kProtocolFatal, C::kInvalidRequestId, "no request id, no output either: the id is first");
  try { encode_cancel(0); CHECK(false); } catch (const ProtocolError& pe) { CHECK(pe.issue.code == C::kInvalidRequestId && pe.issue.failure == F::kProtocolFatal); }
  CHECK(encode_cancel(1).size() == 24);
}

static void test_events() {
  auto ready = std::get<Ready>(parse_one(EVENT("ready")));
  CHECK(ready.max_concurrent_requests == 4 && ready.max_context_tokens == 262144 && ready.vision);

  auto start = std::get<Start>(parse_one(EVENT("start")));
  CHECK(start.request_id == 7 && start.lane == 2 && start.matched_prompt_tokens == 1024);

  auto prog = std::get<PromptProgress>(parse_one(EVENT("progress")));
  CHECK(prog.request_id == 7 && prog.processed_tokens == 512 && prog.elapsed_micros == 123456);

  auto toks = std::get<Tokens>(parse_one(EVENT("tokens")));
  CHECK(toks.request_id == 7 && toks.sequence_offset == 40);
  CHECK((toks.tokens == std::vector<uint32_t>{10, 248045, 4000000000u}));

  auto done = std::get<Done>(parse_one(EVENT("done")));
  CHECK(done.request_id == 7 && done.reason == FinishReason::kLength && done.bounded);
  CHECK(done.prompt_tokens == 1000 && done.completion_tokens == 256);
  CHECK(done.prefill_micros == 5'000'000 && done.decode_micros == 9'000'000 && done.wall_micros == 14'500'000);
  CHECK(done.deduplicated_tokens == 12 && done.drafted_tokens == 100 && done.accepted_draft_tokens == 60 && done.draft_cycles == 40);
  CHECK(done.option_logits.empty());

  auto can = std::get<Done>(parse_one(EVENT("done_cancelled")));
  CHECK(can.reason == FinishReason::kCancelled && !can.bounded && can.completion_tokens == 3);

  auto er = std::get<Error>(parse_one(EVENT("error_request")));
  CHECK(er.failure == FailureClass::kRequestError && er.request_id == 7 && er.retryable && er.retry_after_seconds == 3);
  CHECK(er.code == "busy" && er.message == "later");

  auto ef = std::get<Error>(parse_one(EVENT("error_fatal")));
  CHECK(ef.failure == FailureClass::kProtocolFatal && ef.request_id == 0 && !ef.retryable && ef.code == "bad" && ef.message == "no");

  auto st = std::get<StatusJson>(parse_one(EVENT("status")));
  CHECK(st.correlation_id == 99 && st.json == "{\"ok\":true}");

  auto mk = std::get<MaskRequest>(parse_one(EVENT("mask_request")));
  CHECK(mk.request_id == 7 && mk.mask_request_id == 3 && mk.words_per_mask == 8 && (mk.simulation_tokens == std::vector<uint32_t>{1, 2}));
}

static void test_bad_events() {
  for (const auto& b : kGoldenBad) {
    auto frame = unhex(b.hex);
    FrameParser p;
    p.feed(frame.data(), frame.size());
    Event e;
    try {
      bool got = p.next(e);
      std::printf("FAIL %s: %s\n", b.name, got ? "decoded" : "waited for more bytes");
      failures++;
    } catch (const ProtocolError& pe) {
      if (std::string(issue_code_name(pe.issue.code)) != b.code) {
        std::printf("FAIL %s: got %s want %s\n", b.name, issue_code_name(pe.issue.code), b.code);
        failures++;
      }
      CHECK(p.failed());
      bool refused = false;
      try { p.feed(frame.data(), 1); } catch (const std::logic_error&) { refused = true; }
      CHECK(refused);
    }
  }
}

// What an event says, enough to tell one from another: its kind and its fields
static std::string key(const Event& e) {
  std::string k = std::to_string(e.index());
  auto add = [&](uint64_t v) { k += "," + std::to_string(v); };
  if (auto* v = std::get_if<Ready>(&e)) add(v->max_concurrent_requests), add(v->max_context_tokens), add(v->vision);
  else if (auto* v = std::get_if<Start>(&e)) add(v->request_id), add(v->lane), add(v->matched_prompt_tokens);
  else if (auto* v = std::get_if<PromptProgress>(&e)) add(v->request_id), add(v->processed_tokens), add(v->elapsed_micros);
  else if (auto* v = std::get_if<Tokens>(&e)) { add(v->request_id); add(v->sequence_offset); for (auto t : v->tokens) add(t); }
  else if (auto* v = std::get_if<Done>(&e)) add(v->request_id), add(uint64_t(v->reason)), add(v->prompt_tokens), add(v->completion_tokens), add(v->prefill_micros), add(v->decode_micros), add(v->wall_micros);
  else if (auto* v = std::get_if<Error>(&e)) { add(v->request_id); k += v->code + "|" + v->message; }
  else if (auto* v = std::get_if<StatusJson>(&e)) { add(v->correlation_id); k += v->json; }
  return k;
}

static void test_incremental() {
  // Every event back to back, fed one byte at a time and then in odd chunks, comes out in order and unchanged.
  std::vector<uint8_t> all;
  const char* names[] = {"ready", "start", "progress", "tokens", "tokens", "done", "error_request", "status"};
  size_t want = 0;
  for (auto n : names) { auto f = EVENT(n); all.insert(all.end(), f.begin(), f.end()); want++; }
  for (size_t chunk : {size_t(1), size_t(3), size_t(17), all.size()}) {
    FrameParser p;
    size_t got = 0;
    Event e;
    for (size_t i = 0; i < all.size(); i += chunk) {
      p.feed(all.data() + i, std::min(chunk, all.size() - i));
      while (p.next(e)) {
        CHECK(got < want && key(e) == key(parse_one(EVENT(names[got]))));
        got++;
      }
    }
    CHECK(got == want);
    CHECK(p.buffered() == 0);
  }
  // A header that promises more than its type allows fails before the payload arrives.
  uint8_t huge[16] = {'S', 'P', 'L', 'H', 11, 0, 0x02, 0x01, 0xFF, 0xFF, 0xFF, 0x7F, 0, 0, 0, 0};  // tokens, 2 GiB
  FrameParser p;
  p.feed(huge, sizeof huge);
  Event e;
  try { p.next(e); CHECK(false); } catch (const ProtocolError& pe) { CHECK(pe.issue.code == IssueCode::kFrameTooLarge); }
}

static void test_truncated() {
  const auto frame = EVENT("done");
  // EOF between frames is clean; EOF inside a header or a payload is protocol_fatal:truncated_frame, once
  for (size_t cut : {size_t(0), size_t(5), size_t(15), size_t(16), frame.size() - 1, frame.size()}) {
    FrameParser p;
    p.feed(frame.data(), cut);
    Event e;
    const bool whole = p.next(e);
    auto issue = p.finish();
    if (cut == 0 || cut == frame.size()) {
      CHECK(!issue && whole == (cut == frame.size()));
    } else {
      CHECK(!whole && issue && issue->failure == FailureClass::kProtocolFatal && issue->code == IssueCode::kTruncatedFrame);
      CHECK(p.failed());
      bool refused = false;
      try { p.finish(); } catch (const std::logic_error&) { refused = true; }
      CHECK(refused);
    }
  }
  FrameParser p;
  p.feed(frame.data(), 7);
  CHECK(p.finish()->message == "stream ended after 7 of 16 frame-header bytes");
  FrameParser q;
  q.feed(frame.data(), frame.size() - 3);
  Event e;
  CHECK(!q.next(e));
  auto m = q.finish()->message;
  CHECK(m.find("payload bytes") != std::string::npos);
}

int main() {
  test_client_frames();
  test_request_validation();
  test_events();
  test_bad_events();
  test_incremental();
  test_truncated();
  if (failures) { std::printf("%d failure(s)\n", failures); return 1; }
  std::printf("splash_protocol_test: ok\n");
  return 0;
}
