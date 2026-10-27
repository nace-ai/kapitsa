// Splash's native engine protocol (version 11), the client side: encodes the frames a server sends to the `splash
// serve-native` child and decodes and validates the events it sends back. A port of splash-1.4.0/server/protocol.py
// (Apache-2.0), which `runtime/engine/Protocol.*` mirrors on the engine's side; docs/splash-backend.md says what is left out
// (image spans, score requests and token-mask constraints: a request here is text-only generation, unconstrained).
//
// Little-endian throughout. A frame is a 16-byte header (magic "SPLH", u16 version, u16 type, u64 payload bytes) and the
// payload. Nothing here does I/O or takes a lock: splash/runtime.h owns the child process.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace ni::splash {

constexpr uint16_t kProtocolVersion = 11;
constexpr size_t kFrameHeaderBytes = 16;
constexpr size_t kRequestFixedBytes = 114;
constexpr uint64_t kMaxFramePayloadBytes = 256ull * 1024 * 1024;
constexpr uint32_t kMaxPromptTokens = 1u << 20;
constexpr uint32_t kMaxLogicalOutputTokens = 1u << 20;
constexpr uint32_t kMaxTokenBatch = 4096;
constexpr uint32_t kMaxSimulationTokens = 32;
constexpr uint32_t kMaxMaskWords = 1u << 20;
constexpr uint32_t kMaxErrorStringBytes = 1u << 20;
constexpr uint32_t kMaxStatusJsonBytes = 32u << 20;
constexpr uint32_t kMaxThinkingEndTokens = 64;
constexpr uint32_t kMaxScoreTokens = 255;
constexpr uint32_t kNoThinkingBudget = 0xFFFFFFFFu;

enum class FrameType : uint16_t {
  kRequest = 0x0001,
  kCancel = 0x0002,
  kMaskResponse = 0x0003,
  kStatusRequest = 0x0004,
  kReleaseRequest = 0x0005,
  kReady = 0x0100,
  kStart = 0x0101,
  kTokens = 0x0102,
  kMaskRequest = 0x0103,
  kDone = 0x0104,
  kError = 0x0105,
  kStatusJson = 0x0107,
  kPromptProgress = 0x0108,
};

enum class FailureClass : uint8_t { kRequestError = 1, kEngineUnhealthy = 2, kProtocolFatal = 3 };

// The values are runtime/engine/Protocol.hpp's.
enum class IssueCode : uint8_t {
  kNone = 0,
  kBadMagic = 1,
  kUnsupportedVersion = 2,
  kUnknownFrameType = 3,
  kFrameTooLarge = 4,
  kInvalidPayloadLength = 5,
  kTruncatedFrame = 6,
  kInvalidRequestId = 7,
  kInvalidEnumValue = 8,
  kInvalidDeadline = 9,
  kInvalidSampling = 10,
  kInvalidCount = 11,
  kInvalidConstraint = 12,
  kInvalidErrorClassification = 13,
  kLimitExceeded = 14,
  kIntegerOverflow = 15,
  kAllocationFailure = 16,
};

const char* issue_code_name(IssueCode c);
const char* failure_class_name(FailureClass c);

// What a frame that breaks the protocol says. An event with a request id the engine named fails that request; the
// stream is untrusted after a kProtocolFatal one (the runtime kills the child).
struct Issue {
  FailureClass failure = FailureClass::kProtocolFatal;
  IssueCode code = IssueCode::kNone;
  uint64_t request_id = 0;
  std::string message;
  std::string describe() const;
};

class ProtocolError : public std::runtime_error {
 public:
  explicit ProtocolError(Issue i) : std::runtime_error(i.describe()), issue(std::move(i)) {}
  Issue issue;
};

enum class Priority : uint8_t { kForeground = 0, kNormal = 1, kBackground = 2 };
enum class Constraint : uint8_t { kNone = 0, kTokenMask = 1 };
enum RequestFlag : uint32_t { kIgnoreEndOfSequence = 1u << 0, kReturnProgress = 1u << 1 };

// Greedy, with nothing changing the logits, as protocol.py's defaults.
struct Sampling {
  float temperature = 0.0f, top_p = 1.0f;
  uint32_t top_k = 0;  // 0, or one past the vocabulary: every token
  float presence_penalty = 0.0f, frequency_penalty = 0.0f, repetition_penalty = 1.0f, min_p = 0.0f;
};

// Thinking ends at `tokens` reasoning tokens, where the engine then selects `end` in order (the last closes the block), or
// where the request selects that close or `also_ends` first (a tool call's opening that follows thinking directly).
struct ThinkingBudget {
  uint32_t tokens = 0;
  std::vector<uint32_t> end;
  uint32_t also_ends = 0;
};

struct Request {
  uint64_t request_id = 0;
  Priority priority = Priority::kNormal;
  uint64_t absolute_deadline_unix_micros = 0;
  uint64_t remaining_deadline_micros = 0;
  uint32_t max_output_tokens = 0;  // the logical limit: 1 .. kMaxLogicalOutputTokens
  std::vector<uint32_t> prompt;
  Sampling sampling;
  uint64_t seed = 0;
  // Trailing prompt tokens of the chat template's generation prompt (zero when unknown); leaves at least one token.
  uint32_t generation_prompt_tokens = 0;
  uint32_t flags = 0;
  uint64_t cache_salt_lo = 0, cache_salt_hi = 0;  // the request's cache_salt hashed to 128 bits; zero without one
  std::optional<ThinkingBudget> thinking;
};

// Encoding checks what protocol.py's does, and throws ProtocolError (a kRequestError naming the request when its id is
// non-zero, else a kProtocolFatal) for what the engine would refuse.
std::vector<uint8_t> encode_request(const Request& r);
std::vector<uint8_t> encode_cancel(uint64_t request_id);
std::vector<uint8_t> encode_status_request(uint64_t correlation_id);
std::vector<uint8_t> encode_release_request(uint64_t correlation_id);
// Re-stamps a REQUEST frame's absolute deadline as now + its remaining budget (a request re-sent after a relaunch);
// any other frame comes back as it is.
void refresh_request_deadline(std::vector<uint8_t>& frame, uint64_t now_unix_micros);

struct Ready {
  uint32_t max_concurrent_requests = 0, max_context_tokens = 0;
  bool vision = false;
};
struct Start {
  uint64_t request_id = 0;
  uint32_t lane = 0, matched_prompt_tokens = 0;  // the cached prefix the request starts from
};
struct PromptProgress {
  uint64_t request_id = 0;
  uint32_t processed_tokens = 0;
  uint64_t elapsed_micros = 0;
};
struct Tokens {
  uint64_t request_id = 0;
  uint32_t sequence_offset = 0;
  std::vector<uint32_t> tokens;
};
// Not expected without a token-mask constraint: the runtime treats one as a protocol failure of the request.
struct MaskRequest {
  uint64_t request_id = 0, mask_request_id = 0;
  uint32_t words_per_mask = 0;
  std::vector<uint32_t> simulation_tokens;
};
enum class FinishReason : uint8_t { kStop = 0, kLength = 1, kCancelled = 2 };
struct Done {
  uint64_t request_id = 0;
  FinishReason reason = FinishReason::kStop;
  uint32_t prompt_tokens = 0, completion_tokens = 0;
  uint64_t prefill_micros = 0, decode_micros = 0, wall_micros = 0;
  std::vector<float> option_logits;  // score requests only: empty here
  uint32_t deduplicated_tokens = 0, drafted_tokens = 0, accepted_draft_tokens = 0, draft_cycles = 0;
  bool bounded = false;  // a kLength end short of the request's logical maximum, at a bound of the engine's own
};
struct Error {
  FailureClass failure = FailureClass::kProtocolFatal;
  uint64_t request_id = 0;
  bool retryable = false;
  std::string code, message;
  uint32_t retry_after_seconds = 0;
};
struct StatusJson {
  uint64_t correlation_id = 0;
  std::string json;
};

using Event = std::variant<Ready, Start, PromptProgress, Tokens, MaskRequest, Done, Error, StatusJson>;

// Decodes and strictly validates one event whose header was checked (type and payload bytes).
Event decode_event(FrameType type, const uint8_t* payload, size_t bytes);

// Incremental, one-frame-at-a-time parser of the child's stdout. feed() takes whatever a read returned; next() yields the
// events in order and returns false when the bytes held do not make another frame. A header that breaks the protocol, or
// an event that fails validation, throws ProtocolError, after which the parser refuses further input.
class FrameParser {
 public:
  void feed(const uint8_t* data, size_t n);
  bool next(Event& out);
  size_t buffered() const { return buf_.size() - off_; }
  bool failed() const { return failed_; }
  // The stream ended (EOF): bytes of a frame left over are protocol_fatal:truncated_frame (as protocol.py's finish()),
  // returned and the parser refuses further input; nothing left is nullopt.
  std::optional<Issue> finish();

 private:
  std::vector<uint8_t> buf_;
  size_t off_ = 0;
  bool failed_ = false;
};

}  // namespace ni::splash
