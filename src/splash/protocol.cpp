#include "splash/protocol.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace ni::splash {
namespace {

// ---- little-endian writers and readers (the wire is little-endian; Apple silicon is too, but say so) ----

struct Writer {
  std::vector<uint8_t>& b;
  void u8(uint8_t v) { b.push_back(v); }
  void u16(uint16_t v) { for (int i = 0; i < 2; i++) b.push_back(uint8_t(v >> (8 * i))); }
  void u32(uint32_t v) { for (int i = 0; i < 4; i++) b.push_back(uint8_t(v >> (8 * i))); }
  void u64(uint64_t v) { for (int i = 0; i < 8; i++) b.push_back(uint8_t(v >> (8 * i))); }
  void f32(float v) { uint32_t u; std::memcpy(&u, &v, 4); u32(u); }
};

struct Reader {
  const uint8_t* p;
  size_t n, off = 0;
  uint8_t u8() { return p[off++]; }
  uint32_t u32() { uint32_t v = 0; for (int i = 0; i < 4; i++) v |= uint32_t(p[off + i]) << (8 * i); off += 4; return v; }
  uint64_t u64() { uint64_t v = 0; for (int i = 0; i < 8; i++) v |= uint64_t(p[off + i]) << (8 * i); off += 8; return v; }
  float f32() { uint32_t u = u32(); float v; std::memcpy(&v, &u, 4); return v; }
};

[[noreturn]] void fail(FailureClass f, IssueCode c, std::string m, uint64_t rid = 0) {
  throw ProtocolError(Issue{f, c, rid, std::move(m)});
}
[[noreturn]] void fatal(IssueCode c, std::string m, uint64_t rid = 0) { fail(FailureClass::kProtocolFatal, c, std::move(m), rid); }
// A client frame's failure fails the request it names; one that names none fails the stream.
[[noreturn]] void fail_request(IssueCode c, std::string m, uint64_t rid) {
  fail(rid ? FailureClass::kRequestError : FailureClass::kProtocolFatal, c, std::move(m), rid);
}

constexpr size_t kStartBytes = 16, kReadyBytes = 9, kPromptProgressBytes = 20, kTokensHead = 16, kMaskRequestHead = 24,
                 kDoneHead = 62, kErrorHead = 22, kStatusHead = 8, kCancelBytes = 8;
static_assert(kStartBytes == 8 + 4 + 4 && kReadyBytes == 4 + 4 + 1 && kPromptProgressBytes == 8 + 4 + 8);

// protocol.py's _payload_bounds, for the event types (the engine's frames).
std::pair<uint64_t, uint64_t> event_bounds(FrameType t) {
  uint64_t lo = 0, hi = 0;
  switch (t) {
    case FrameType::kReady: lo = hi = kReadyBytes; break;
    case FrameType::kPromptProgress: lo = hi = kPromptProgressBytes; break;
    case FrameType::kStart: lo = hi = kStartBytes; break;
    case FrameType::kTokens: lo = kTokensHead; hi = kTokensHead + uint64_t(kMaxTokenBatch) * 4; break;
    case FrameType::kMaskRequest: lo = kMaskRequestHead; hi = kMaskRequestHead + uint64_t(kMaxSimulationTokens) * 4; break;
    case FrameType::kDone: lo = kDoneHead; hi = kDoneHead + uint64_t(kMaxScoreTokens) * 4; break;
    case FrameType::kError: lo = kErrorHead; hi = kErrorHead + uint64_t(kMaxErrorStringBytes) * 2; break;
    case FrameType::kStatusJson: lo = kStatusHead; hi = kStatusHead + kMaxStatusJsonBytes; break;
    default: break;
  }
  return {lo, std::min<uint64_t>(hi, kMaxFramePayloadBytes)};
}

void check_payload_length(FrameType t, uint64_t bytes) {
  auto [lo, hi] = event_bounds(t);
  if (bytes > hi) fatal(IssueCode::kFrameTooLarge, "payload length " + std::to_string(bytes) + " exceeds its safe limit " + std::to_string(hi));
  if (bytes < lo) fatal(IssueCode::kInvalidPayloadLength, "payload length " + std::to_string(bytes) + " is below its required minimum " + std::to_string(lo));
}

void header(Writer& w, FrameType t, uint64_t payload_bytes) {
  const char magic[4] = {'S', 'P', 'L', 'H'};
  for (char c : magic) w.u8(uint8_t(c));
  w.u16(kProtocolVersion);
  w.u16(uint16_t(t));
  w.u64(payload_bytes);
}

bool is_event(uint16_t t) {
  switch (FrameType(t)) {
    case FrameType::kReady: case FrameType::kStart: case FrameType::kTokens: case FrameType::kMaskRequest:
    case FrameType::kDone: case FrameType::kError: case FrameType::kStatusJson: case FrameType::kPromptProgress:
      return true;
    default: return false;
  }
}

constexpr float kFloat32Min = 1.17549435e-38f;  // 0x1p-126

}  // namespace

const char* issue_code_name(IssueCode c) {
  switch (c) {
    case IssueCode::kNone: return "none";
    case IssueCode::kBadMagic: return "bad_magic";
    case IssueCode::kUnsupportedVersion: return "unsupported_version";
    case IssueCode::kUnknownFrameType: return "unknown_frame_type";
    case IssueCode::kFrameTooLarge: return "frame_too_large";
    case IssueCode::kInvalidPayloadLength: return "invalid_payload_length";
    case IssueCode::kTruncatedFrame: return "truncated_frame";
    case IssueCode::kInvalidRequestId: return "invalid_request_id";
    case IssueCode::kInvalidEnumValue: return "invalid_enum_value";
    case IssueCode::kInvalidDeadline: return "invalid_deadline";
    case IssueCode::kInvalidSampling: return "invalid_sampling";
    case IssueCode::kInvalidCount: return "invalid_count";
    case IssueCode::kInvalidConstraint: return "invalid_constraint";
    case IssueCode::kInvalidErrorClassification: return "invalid_error_classification";
    case IssueCode::kLimitExceeded: return "limit_exceeded";
    case IssueCode::kIntegerOverflow: return "integer_overflow";
    case IssueCode::kAllocationFailure: return "allocation_failure";
  }
  return "unknown";
}

const char* failure_class_name(FailureClass c) {
  switch (c) {
    case FailureClass::kRequestError: return "request_error";
    case FailureClass::kEngineUnhealthy: return "engine_unhealthy";
    case FailureClass::kProtocolFatal: return "protocol_fatal";
  }
  return "unknown";
}

std::string Issue::describe() const {
  std::string s = std::string(failure_class_name(failure)) + ":" + issue_code_name(code);
  if (request_id) s += " request=" + std::to_string(request_id);
  if (!message.empty()) s += ": " + message;
  return s;
}

// ---- client frames ----

std::vector<uint8_t> encode_request(const Request& r) {
  const uint64_t rid = r.request_id;
  // first, as protocol.py's: an id of zero names no request, so the stream fails
  if (!rid) fatal(IssueCode::kInvalidRequestId, "request id must be non-zero");
  if (r.priority > Priority::kBackground) fail_request(IssueCode::kInvalidEnumValue, "priority is not defined by native protocol", rid);
  if (r.flags & ~uint32_t(kIgnoreEndOfSequence | kReturnProgress))
    fail_request(IssueCode::kInvalidEnumValue, "request flags are not defined by native protocol", rid);
  if (!r.absolute_deadline_unix_micros || !r.remaining_deadline_micros)
    fail_request(IssueCode::kInvalidDeadline, "absolute and remaining deadlines must be non-zero", rid);
  if (!r.max_output_tokens || r.max_output_tokens > kMaxLogicalOutputTokens)
    fail_request(IssueCode::kLimitExceeded, "logical max output token count exceeds its limit", rid);
  if (r.prompt.empty() || r.prompt.size() > kMaxPromptTokens)
    fail_request(IssueCode::kLimitExceeded, "prompt token count exceeds its limit", rid);
  if (r.generation_prompt_tokens >= r.prompt.size())
    fail_request(IssueCode::kInvalidCount, "generation prompt must leave a prompt token", rid);

  const Sampling& s = r.sampling;
  if (!(s.temperature == 0.0f || (kFloat32Min <= s.temperature && s.temperature < std::numeric_limits<float>::infinity())) ||
      !std::isfinite(s.top_p) || !(0.0f < s.top_p && s.top_p <= 1.0f) || !(0.0f <= s.min_p && s.min_p <= 1.0f))
    fail_request(IssueCode::kInvalidSampling, "sampling requires temperature 0 or at least FLT_MIN, top_p in (0,1] and min_p in [0,1]", rid);
  if (!(std::fabs(s.presence_penalty) <= 2.0f) || !(std::fabs(s.frequency_penalty) <= 2.0f) || !std::isfinite(s.repetition_penalty) ||
      s.repetition_penalty <= 0.0f)
    fail_request(IssueCode::kInvalidSampling, "sampling requires presence and frequency penalties in [-2,2] and a positive repetition penalty", rid);

  uint32_t budget = kNoThinkingBudget, end_count = 0, also_ends = 0;
  if (r.thinking) {
    const ThinkingBudget& t = *r.thinking;
    if (t.tokens >= kNoThinkingBudget)
      fail_request(IssueCode::kInvalidCount, "thinking budget must be an integer in [0, 4294967294]", rid);
    if (t.end.empty() || t.end.size() > kMaxThinkingEndTokens)
      fail_request(IssueCode::kInvalidCount, "thinking end token count must be in [1, " + std::to_string(kMaxThinkingEndTokens) + "]", rid);
    // Thinking ends where its end does, and nowhere inside it.
    const auto head_end = t.end.end() - 1;
    if (std::find(t.end.begin(), head_end, t.end.back()) != head_end || std::find(t.end.begin(), head_end, t.also_ends) != head_end)
      fail_request(IssueCode::kInvalidCount, "only the last thinking end token may end thinking", rid);
    budget = t.tokens;
    end_count = uint32_t(t.end.size());
    also_ends = t.also_ends;
  }

  const uint64_t payload = kRequestFixedBytes + uint64_t(r.prompt.size()) * 4 + uint64_t(end_count) * 4;
  if (payload > kMaxFramePayloadBytes) fail_request(IssueCode::kFrameTooLarge, "request payload exceeds its safe limit", rid);

  std::vector<uint8_t> out;
  out.reserve(kFrameHeaderBytes + payload);
  Writer w{out};
  header(w, FrameType::kRequest, payload);
  w.u64(rid);
  w.u8(uint8_t(r.priority));
  w.u8(uint8_t(Constraint::kNone));
  w.u64(r.absolute_deadline_unix_micros);
  w.u64(r.remaining_deadline_micros);
  w.u32(r.max_output_tokens);
  w.u32(uint32_t(r.prompt.size()));
  w.u32(0);  // image spans
  w.f32(s.temperature);
  w.f32(s.top_p);
  w.u32(s.top_k);
  w.f32(s.presence_penalty);
  w.f32(s.frequency_penalty);
  w.f32(s.repetition_penalty);
  w.f32(s.min_p);
  w.u64(r.seed);
  w.u32(0);  // score tokens
  w.u32(r.generation_prompt_tokens);
  w.u32(r.flags);
  w.u64(r.cache_salt_lo);
  w.u64(r.cache_salt_hi);
  w.u32(budget);
  w.u32(end_count);
  w.u32(also_ends);
  for (uint32_t t : r.prompt) w.u32(t);
  if (r.thinking)
    for (uint32_t t : r.thinking->end) w.u32(t);
  return out;
}

static std::vector<uint8_t> one_u64(FrameType t, uint64_t v, const char* what) {
  if (!v) fail_request(IssueCode::kInvalidRequestId, std::string(what) + " must be non-zero", 0);
  std::vector<uint8_t> out;
  Writer w{out};
  header(w, t, kCancelBytes);
  w.u64(v);
  return out;
}

std::vector<uint8_t> encode_cancel(uint64_t request_id) { return one_u64(FrameType::kCancel, request_id, "cancel request id"); }

// Status and release requests carry a correlation id of the caller's; zero is allowed there.
std::vector<uint8_t> encode_status_request(uint64_t correlation_id) {
  std::vector<uint8_t> out;
  Writer w{out};
  header(w, FrameType::kStatusRequest, kCancelBytes);
  w.u64(correlation_id);
  return out;
}
std::vector<uint8_t> encode_release_request(uint64_t correlation_id) {
  std::vector<uint8_t> out;
  Writer w{out};
  header(w, FrameType::kReleaseRequest, kCancelBytes);
  w.u64(correlation_id);
  return out;
}

void refresh_request_deadline(std::vector<uint8_t>& frame, uint64_t now) {
  if (frame.size() < kFrameHeaderBytes + kRequestFixedBytes) return;
  uint16_t type = uint16_t(frame[6] | frame[7] << 8);
  if (type != uint16_t(FrameType::kRequest)) return;
  // head: request_id u64, priority u8, constraint u8, absolute u64, remaining u64
  const size_t abs_off = kFrameHeaderBytes + 8 + 1 + 1, rem_off = abs_off + 8;
  Reader rd{frame.data(), frame.size(), rem_off};
  uint64_t remaining = rd.u64();
  uint64_t absolute = remaining > std::numeric_limits<uint64_t>::max() - now ? std::numeric_limits<uint64_t>::max() : now + remaining;
  for (int i = 0; i < 8; i++) frame[abs_off + i] = uint8_t(absolute >> (8 * i));
}

// ---- events ----

Event decode_event(FrameType type, const uint8_t* p, size_t n) {
  if (!is_event(uint16_t(type))) fatal(IssueCode::kUnknownFrameType, "frame type is not a native protocol event");
  check_payload_length(type, n);
  Reader r{p, n};
  switch (type) {
    case FrameType::kReady: {
      Ready e;
      e.max_concurrent_requests = r.u32();
      e.max_context_tokens = r.u32();
      uint8_t vision = r.u8();
      if (vision > 1) fatal(IssueCode::kInvalidEnumValue, "ready vision must be a boolean");
      if (!e.max_concurrent_requests || !e.max_context_tokens) fatal(IssueCode::kInvalidCount, "ready event capacities must be non-zero");
      e.vision = vision;
      return e;
    }
    case FrameType::kStart: {
      Start e;
      e.request_id = r.u64();
      e.lane = r.u32();
      e.matched_prompt_tokens = r.u32();
      if (!e.request_id) fatal(IssueCode::kInvalidRequestId, "start event request id must be non-zero");
      return e;
    }
    case FrameType::kPromptProgress: {
      PromptProgress e;
      e.request_id = r.u64();
      e.processed_tokens = r.u32();
      e.elapsed_micros = r.u64();
      if (!e.request_id || e.processed_tokens > kMaxPromptTokens) fatal(IssueCode::kInvalidCount, "prompt progress id or token count is invalid", e.request_id);
      return e;
    }
    case FrameType::kTokens: {
      Tokens e;
      e.request_id = r.u64();
      e.sequence_offset = r.u32();
      uint32_t count = r.u32();
      if (n - kTokensHead != uint64_t(count) * 4) fatal(IssueCode::kInvalidPayloadLength, "token count does not match the binary event payload", e.request_id);
      if (!e.request_id) fatal(IssueCode::kInvalidRequestId, "tokens event request id must be non-zero");
      if (!count) fatal(IssueCode::kLimitExceeded, "tokens event carries no token", e.request_id);
      if (uint64_t(e.sequence_offset) + count > 0xFFFFFFFFull) fatal(IssueCode::kIntegerOverflow, "tokens event sequence range overflows uint32", e.request_id);
      e.tokens.resize(count);
      for (auto& t : e.tokens) t = r.u32();
      return e;
    }
    case FrameType::kMaskRequest: {
      MaskRequest e;
      e.request_id = r.u64();
      e.mask_request_id = r.u64();
      e.words_per_mask = r.u32();
      uint32_t count = r.u32();
      if (n - kMaskRequestHead != uint64_t(count) * 4) fatal(IssueCode::kInvalidPayloadLength, "simulation token count does not match the binary event payload", e.request_id);
      if (!e.request_id || !e.mask_request_id) fatal(IssueCode::kInvalidRequestId, "mask request ids must be non-zero", e.request_id);
      if (!e.words_per_mask) fatal(IssueCode::kInvalidCount, "mask request dimensions are invalid", e.request_id);
      if (uint64_t(e.words_per_mask) * (uint64_t(count) + 1) > kMaxMaskWords) fatal(IssueCode::kLimitExceeded, "mask request output would exceed the mask limit", e.request_id);
      e.simulation_tokens.resize(count);
      for (auto& t : e.simulation_tokens) t = r.u32();
      return e;
    }
    case FrameType::kDone: {
      Done e;
      e.request_id = r.u64();
      uint8_t reason = r.u8(), bounded = r.u8();
      e.prompt_tokens = r.u32();
      e.completion_tokens = r.u32();
      e.prefill_micros = r.u64();
      e.decode_micros = r.u64();
      e.wall_micros = r.u64();
      e.deduplicated_tokens = r.u32();
      e.drafted_tokens = r.u32();
      e.accepted_draft_tokens = r.u32();
      e.draft_cycles = r.u32();
      uint32_t logit_count = r.u32();
      if (n - kDoneHead != uint64_t(logit_count) * 4) fatal(IssueCode::kInvalidPayloadLength, "option logit count does not match the binary event payload", e.request_id);
      e.option_logits.resize(logit_count);
      for (auto& v : e.option_logits) v = r.f32();
      if (reason > uint8_t(FinishReason::kCancelled)) fatal(IssueCode::kInvalidEnumValue, "done finish reason is invalid", e.request_id);
      e.reason = FinishReason(reason);
      if (!e.request_id) fatal(IssueCode::kInvalidRequestId, "done event request id must be non-zero");
      if (bounded > 1 || (bounded && e.reason != FinishReason::kLength)) fatal(IssueCode::kInvalidEnumValue, "only a length-finished done event is bounded", e.request_id);
      e.bounded = bounded;
      // Each cycle verified at least one drafted token and accepted at most all.
      if (e.accepted_draft_tokens > e.drafted_tokens || e.draft_cycles > e.drafted_tokens || (e.drafted_tokens && !e.draft_cycles))
        fatal(IssueCode::kInvalidCount, "done speculative counts are inconsistent", e.request_id);
      if (!e.option_logits.empty()) {
        bool finite = std::all_of(e.option_logits.begin(), e.option_logits.end(), [](float v) { return std::isfinite(v); });
        if (e.option_logits.size() < 2 || !finite || e.reason != FinishReason::kStop || e.completion_tokens || e.decode_micros || e.drafted_tokens)
          fatal(IssueCode::kInvalidCount, "scored done events need 2..255 finite logits, stop, and no completion or decode activity", e.request_id);
      }
      return e;
    }
    case FrameType::kError: {
      Error e;
      uint8_t failure = r.u8(), retryable = r.u8();
      e.retry_after_seconds = r.u32();
      e.request_id = r.u64();
      uint32_t code_bytes = r.u32(), message_bytes = r.u32();
      if (retryable > 1 || uint64_t(code_bytes) + message_bytes != n - kErrorHead)
        fatal(IssueCode::kInvalidPayloadLength, "error string lengths do not match the frame payload");
      if (failure < 1 || failure > 3) fatal(IssueCode::kInvalidErrorClassification, "error event classification is invalid", e.request_id);
      e.failure = FailureClass(failure);
      if ((e.failure == FailureClass::kRequestError) != (e.request_id != 0))
        fatal(IssueCode::kInvalidErrorClassification, "only request errors may carry a non-zero request id", e.request_id);
      if (!code_bytes || code_bytes > kMaxErrorStringBytes || message_bytes > kMaxErrorStringBytes)
        fatal(IssueCode::kLimitExceeded, "error code or message exceeds its safe limit", e.request_id);
      if (e.retry_after_seconds && !retryable) fatal(IssueCode::kInvalidErrorClassification, "only a retryable error says when to retry", e.request_id);
      e.retryable = retryable;
      e.code.assign(reinterpret_cast<const char*>(p + kErrorHead), code_bytes);
      e.message.assign(reinterpret_cast<const char*>(p + kErrorHead + code_bytes), message_bytes);
      return e;
    }
    case FrameType::kStatusJson: {
      StatusJson e;
      e.correlation_id = r.u64();
      if (n == kStatusHead) fatal(IssueCode::kLimitExceeded, "status JSON is empty");
      e.json.assign(reinterpret_cast<const char*>(p + kStatusHead), n - kStatusHead);
      return e;
    }
    default: break;
  }
  fatal(IssueCode::kUnknownFrameType, "frame type is not a native protocol event");
}

// ---- parser ----

void FrameParser::feed(const uint8_t* data, size_t n) {
  if (failed_) throw std::logic_error("frame parser used after it failed");
  if (off_ && off_ == buf_.size()) { buf_.clear(); off_ = 0; }
  else if (off_ > (1u << 20) && off_ * 2 > buf_.size()) { buf_.erase(buf_.begin(), buf_.begin() + off_); off_ = 0; }
  buf_.insert(buf_.end(), data, data + n);
}

bool FrameParser::next(Event& out) {
  if (failed_) throw std::logic_error("frame parser used after it failed");
  if (buf_.size() - off_ < kFrameHeaderBytes) return false;
  const uint8_t* h = buf_.data() + off_;
  try {
    if (std::memcmp(h, "SPLH", 4) != 0) fatal(IssueCode::kBadMagic, "frame magic is not SPLH");
    uint16_t version = uint16_t(h[4] | h[5] << 8), type = uint16_t(h[6] | h[7] << 8);
    uint64_t payload = 0;
    for (int i = 0; i < 8; i++) payload |= uint64_t(h[8 + i]) << (8 * i);
    if (version != kProtocolVersion) fatal(IssueCode::kUnsupportedVersion, "unsupported native protocol version");
    if (!is_event(type)) fatal(IssueCode::kUnknownFrameType, "frame type is not a native protocol event");
    check_payload_length(FrameType(type), payload);
    if (buf_.size() - off_ - kFrameHeaderBytes < payload) return false;  // the rest has not arrived
    out = decode_event(FrameType(type), h + kFrameHeaderBytes, size_t(payload));
    off_ += kFrameHeaderBytes + size_t(payload);
    return true;
  } catch (const ProtocolError&) {
    failed_ = true;
    throw;
  }
}

std::optional<Issue> FrameParser::finish() {
  if (failed_) throw std::logic_error("frame parser used after it failed");
  const size_t held = buf_.size() - off_;
  if (!held) return std::nullopt;
  failed_ = true;
  std::string msg;
  if (held < kFrameHeaderBytes) {
    msg = "stream ended after " + std::to_string(held) + " of " + std::to_string(kFrameHeaderBytes) + " frame-header bytes";
  } else {
    const uint8_t* h = buf_.data() + off_;
    uint64_t payload = 0;
    for (int i = 0; i < 8; i++) payload |= uint64_t(h[8 + i]) << (8 * i);
    char type[16];
    std::snprintf(type, sizeof type, "0x%04x", unsigned(h[6] | h[7] << 8));
    msg = "stream ended after " + std::to_string(held - kFrameHeaderBytes) + " of " + std::to_string(payload) + " " + type +
          " payload bytes";
  }
  return Issue{FailureClass::kProtocolFatal, IssueCode::kTruncatedFrame, 0, std::move(msg)};
}

}  // namespace ni::splash
