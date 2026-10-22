// bench/harness/bench.py's measurements in C++, request for request (bench.h). Each function names the Python one it
// ports; where they could differ, the comment says how they are kept the same.
#include "bench/bench.h"

#include <CommonCrypto/CommonDigest.h>
#include <httplib/httplib.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>

namespace ni {
extern const char* const kBenchCorpus;  // bench/harness/corpus/bench_corpus.txt (CMakeLists.txt embeds it)
}

namespace ni::bench {

// ---------------------------------------------------------------- the corpus and the prompts

const char* const kCorpusSha256 = "58a7756fd94fc48a7b7306d330b2c44663bdf22faa943f82430966c2878081b2";
// the corpus before its extension: its first 144,500 characters, so a result of it has the same prompts
const char* const kCorpusSha256V1 = "1a9bb103b3f6d39ee5b9f5bf7207e304c5ce92788c273685f597508c7d2bde28";

const std::string& corpus() {
  static const std::string c = kBenchCorpus;
  return c;
}

std::string sha256_hex(const std::string& data) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.data(), CC_LONG(data.size()), d);
  char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", d[i]);
  return hex;
}

namespace {

// UTF-8: the byte after the character at i (a stray byte counts as one character, as it never occurs in the corpus)
size_t next_char(const std::string& s, size_t i) {
  const unsigned char c = uint8_t(s[i]);
  const size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
  return std::min(s.size(), i + n);
}

uint32_t decode_char(const std::string& s, size_t i, size_t* next) {
  *next = next_char(s, i);
  const unsigned char c = uint8_t(s[i]);
  const size_t n = *next - i;
  if (n == 1) return c;
  uint32_t cp = c & (n == 2 ? 0x1F : n == 3 ? 0x0F : 0x07);
  for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (uint8_t(s[i + k]) & 0x3F);
  return cp;
}

}  // namespace

// Python's len() of the str: its code points
static int chars(const std::string& s) {
  int n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

std::string build_context(int target_tokens, int offset_chars) {
  // the byte offset of every character, once (the corpus is 650,000 characters in 653,464 bytes)
  static const std::vector<size_t> at = [] {
    std::vector<size_t> v;
    const std::string& c = corpus();
    for (size_t i = 0; i < c.size(); i = next_char(c, i)) v.push_back(i);
    v.push_back(c.size());
    return v;
  }();
  const long want = long(target_tokens * 4.25);  // int(target_tokens * CHARS_PER_TOKEN)
  const long n = long(at.size()) - 1;
  if (offset_chars < 0 || offset_chars + want > n)
    throw std::runtime_error("the corpus has " + std::to_string(n) + " characters; " + std::to_string(target_tokens) +
                             " tokens at offset " + std::to_string(offset_chars) + " need " +
                             std::to_string(offset_chars + want));
  return corpus().substr(at[size_t(offset_chars)], at[size_t(offset_chars + want)] - at[size_t(offset_chars)]);
}

const char* const kDecodeUser =
    "Write a complete, well-commented Python module that implements an LRU cache with TTL expiry, "
    "a thread-safe wrapper, and a pytest test suite. Write at least 600 lines. Do not stop early.";
const char* const kDecodeSys = "You are a helpful coding assistant. Answer with code only.";
const char* const kFollowups[3] = {"Now name one function defined in that file.",
                                   "Name one class defined in that file, or say there is none.",
                                   "Which modules does that file import?"};
const char* const kSys =
    "You are Nace, a local coding agent. You help the user with their repository. "
    "Be precise. Use tools when you need to read files or run commands.";
const char* const kToolResult =
    "import sys\n\ndef main():\n    print('hello from nace')\n    return 0\n\nif __name__ == '__main__':\n    sys.exit(main())\n";

ojson tools() {
  // TOOLS, key for key in bench.py's order: the chat template renders them as JSON, so the order is part of the prompt
  ojson read_file = {{"type", "function"},
                     {"function",
                      {{"name", "read_file"},
                       {"description", "Read a UTF-8 text file from the user's workspace and return its contents."},
                       {"parameters",
                        {{"type", "object"},
                         {"properties",
                          {{"path", {{"type", "string"}, {"description", "Workspace-relative path"}}},
                           {"start_line", {{"type", "integer"}, {"description", "Optional first line (1-based)"}}}}},
                         {"required", {"path"}}}}}}};
  ojson run_shell = {{"type", "function"},
                     {"function",
                      {{"name", "run_shell"},
                       {"description", "Run a shell command in the workspace and return stdout/stderr."},
                       {"parameters",
                        {{"type", "object"},
                         {"properties", {{"command", {{"type", "string"}}}, {"timeout_s", {{"type", "integer"}}}}},
                         {"required", {"command"}}}}}}};
  return ojson::array({read_file, run_shell});
}

ojson prefill_messages(const std::string& ctx, const std::string& nonce, const std::string& question) {
  // the nonce sits right after "<|im_start|>system\n": no engine can reuse more than ~5 tokens
  return ojson::array({{{"role", "system"}, {"content", "[" + nonce + "] " + kSys + "\n\n# Repository context\n" + ctx}},
                       {{"role", "user"}, {"content", question}}});
}

ojson decode_messages(const std::string& run_nonce) {
  return ojson::array({{{"role", "system"}, {"content", kDecodeSys}},
                       {{"role", "user"}, {"content", "[run " + run_nonce + "] " + kDecodeUser}}});
}

ojson tools_messages() {
  return ojson::array({{{"role", "system"}, {"content", kSys}},
                       {{"role", "user"}, {"content", "Open src/main.py and tell me what the entry point does."}}});
}

std::map<int, int> reference_prompt_tokens(const std::string& family) {
  // bench.py REF_PROMPT_TOKENS (Qwen3.6 / Qwen3.8 chat templates, thinking off; Qwen3.5-4B comes within 1%), and
  // configs/bolt.json's and configs/flint.json's --ref-prompt-tokens for Nemotron 3.5 Bolt and Recursive's Flint
  if (family == "qwen3.5") return {{4000, 4133}, {16000, 16286}};
  if (family == "nemotron-h") return {{4000, 4183}, {16000, 16514}};
  if (family == "flint") return {{4000, 3874}, {16000, 15066}};
  return {};
}

std::string Nonces::run() {
  if (fixed) return fixed_run;
  // random.random(): 53 random bits / 2^53
  const double x = double(rng() >> 11) * (1.0 / 9007199254740992.0);
  char b[32];
  snprintf(b, sizeof b, "%.6f", x);
  return b;
}

std::string Nonces::hex(int bits) {
  const uint64_t r = fixed ? fixed_bits : rng();
  const uint64_t v = bits >= 64 ? r : r & ((uint64_t(1) << bits) - 1);
  char b[32];
  snprintf(b, sizeof b, "%0*llx", bits / 4, (unsigned long long)v);
  return b;
}

// ---------------------------------------------------------------- request parameters

ojson sampling(const Settings& s, bool greedy) {
  // every sampling field explicit, so no engine falls back to a hidden default; greedy: T=0 plus top_k 1 / top_p 1
  ojson o = {{"temperature", s.temperature}, {"top_p", s.top_p}, {"top_k", s.top_k}, {"min_p", s.min_p},
             {"presence_penalty", s.presence_penalty}, {"frequency_penalty", s.frequency_penalty},
             {"repeat_penalty", s.repeat_penalty}};
  if (greedy) o["temperature"] = 0.0, o["top_p"] = 1.0, o["top_k"] = 1;
  return o;
}

ojson no_think() { return {{"chat_template_kwargs", {{"enable_thinking", false}}}, {"reasoning_effort", "none"}}; }

ojson think_on(const Settings& s) {
  ojson d = ojson::object();
  if (!s.reasoning_effort.empty()) d["reasoning_effort"] = s.reasoning_effort;
  d["chat_template_kwargs"] = {{"enable_thinking", true}};
  return d;
}

ojson body_of(const Settings& s, const ojson& msgs, int max_tokens, bool greedy, bool think, const ojson& extra) {
  ojson b = {{"model", s.model}, {"messages", msgs}, {"max_tokens", max_tokens}};
  // named first: a range-for over items() of a temporary would see it destroyed before the loop runs
  const ojson samp = sampling(s, greedy), thinking = think ? think_on(s) : no_think();
  for (const auto& [k, v] : samp.items()) b[k] = v;
  for (const auto& [k, v] : thinking.items()) b[k] = v;
  for (const auto& [k, v] : extra.items()) b[k] = v;
  return b;
}

// ---------------------------------------------------------------- HTTP

namespace {

using clock = std::chrono::steady_clock;
double since(clock::time_point t0) { return std::chrono::duration<double>(clock::now() - t0).count(); }

std::string strip(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && isspace(uint8_t(s[a]))) ++a;
  while (b > a && isspace(uint8_t(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

std::string errname(httplib::Error e) { return httplib::to_string(e); }

// Python's truth value of a JSON value
bool truthy(const ojson& v) {
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number()) return v.get<double>() != 0;
  if (v.is_string()) return !v.get<std::string>().empty();
  return !v.empty();
}

}  // namespace

// bench.py post() + stream_chat(): one request on a new connection, the body serialized after t0 as Python does
Stream stream_chat(const Settings& s, const ojson& body_in, const std::string& session, int timeout_s) {
  ojson body = body_in;
  body["stream"] = true;
  if (!body.contains("stream_options")) body["stream_options"] = {{"include_usage", true}};
  Stream out;
  std::map<int, ToolCallAcc> tcs;
  std::string pending, error_body;
  int status = 0;
  bool done = false;
  double first = -1, last = -1;
  const auto t0 = clock::now();
  httplib::Client cli(s.host, s.port);
  cli.set_connection_timeout(30, 0);
  cli.set_read_timeout(timeout_s, 0);
  cli.set_write_timeout(60, 0);
  httplib::Request req;
  req.method = "POST";
  req.path = "/v1/chat/completions";
  req.headers = {{"Content-Type", "application/json"}, {"Authorization", "Bearer local"}};
  if (!session.empty()) req.headers.emplace("X-Session-Id", session);  // engines like MTPLX key a session bank on it
  req.body = body.dump();
  req.response_handler = [&](const httplib::Response& r) {
    status = r.status;
    return true;
  };
  req.content_receiver = [&](const char* data, size_t n, uint64_t, uint64_t) {
    if (status != 200) {
      error_body.append(data, n);
      return true;
    }
    pending.append(data, n);
    size_t nl;
    while (!done && (nl = pending.find('\n')) != std::string::npos) {
      const std::string line = strip(pending.substr(0, nl));
      pending.erase(0, nl + 1);
      if (line.rfind("data:", 0) != 0) continue;
      const std::string data_s = strip(line.substr(5));
      if (data_s == "[DONE]") {
        done = true;
        break;
      }
      ojson ev = ojson::parse(data_s, nullptr, false);
      if (ev.is_discarded() || !ev.is_object()) continue;
      if (ev.contains("usage") && !ev["usage"].is_null() && !ev["usage"].empty()) out.usage = ev["usage"];
      if (ev.contains("timings") && !ev["timings"].is_null() && !ev["timings"].empty()) out.timings = ev["timings"];
      if (!ev.contains("choices") || !ev["choices"].is_array()) continue;
      for (const auto& ch : ev["choices"]) {
        if (!ch.is_object()) continue;
        const ojson d = ch.contains("delta") && ch["delta"].is_object() ? ch["delta"] : ojson::object();
        if (ch.contains("finish_reason") && ch["finish_reason"].is_string() && !ch["finish_reason"].get<std::string>().empty())
          out.finish = ch["finish_reason"];
        bool got = false;
        int nch = 0;
        if (d.contains("content") && d["content"].is_string() && !d["content"].get<std::string>().empty()) {
          const std::string c = d["content"];
          out.content += c, nch += chars(c), got = true;
        }
        std::string rc;
        if (d.contains("reasoning_content") && d["reasoning_content"].is_string()) rc = d["reasoning_content"];
        if (rc.empty() && d.contains("reasoning") && d["reasoning"].is_string()) rc = d["reasoning"];
        if (!rc.empty()) out.reasoning += rc, nch += chars(rc), got = true;
        if (d.contains("tool_calls") && d["tool_calls"].is_array()) {
          for (const auto& tc : d["tool_calls"]) {
            got = true;
            const int i = tc.contains("index") && tc["index"].is_number_integer() ? tc["index"].get<int>() : 0;
            ToolCallAcc& e = tcs[i];
            if (tc.contains("id") && tc["id"].is_string() && !tc["id"].get<std::string>().empty()) e.id = tc["id"], e.has_id = true;
            const ojson fn = tc.contains("function") && tc["function"].is_object() ? tc["function"] : ojson::object();
            if (fn.contains("name") && fn["name"].is_string() && !fn["name"].get<std::string>().empty()) {
              const std::string nm = fn["name"];
              e.name += nm, nch += chars(nm);
            }
            if (fn.contains("arguments") && truthy(fn["arguments"])) {  // Python: if fn.get("arguments")
              const std::string a = fn["arguments"].is_string() ? fn["arguments"].get<std::string>() : fn["arguments"].dump();
              e.arguments += a, nch += chars(a);
            }
          }
        }
        if (got) {
          ++out.n_chunks;
          const double now = since(t0);
          if (first < 0) first = now;
          last = now;
          out.ev_t.push_back(now);
          out.ev_c.push_back(nch);
        }
      }
    }
    return true;
  };
  httplib::Response res;
  httplib::Error err = httplib::Error::Success;
  const bool ok = cli.send(req, res, err);
  out.total = since(t0);
  if (status != 0 && status != 200)
    throw std::runtime_error("HTTP " + std::to_string(status) + ": " + error_body.substr(0, 500));
  if (!ok) throw std::runtime_error("request failed: " + errname(err));
  out.ttft = first;
  out.gen_time = first >= 0 && last >= 0 ? last - first : -1;
  for (auto& [i, e] : tcs) out.tool_calls.push_back(e);
  return out;
}

ojson post_json(const Settings& s, const ojson& body, int timeout_s) {
  httplib::Client cli(s.host, s.port);
  cli.set_connection_timeout(30, 0);
  cli.set_read_timeout(timeout_s, 0);
  cli.set_write_timeout(60, 0);
  httplib::Headers h = {{"Authorization", "Bearer local"}};
  auto r = cli.Post("/v1/chat/completions", h, body.dump(), "application/json");
  if (!r) throw std::runtime_error("request failed: " + errname(r.error()));
  if (r->status != 200) throw std::runtime_error("HTTP " + std::to_string(r->status) + ": " + r->body.substr(0, 500));
  return ojson::parse(r->body);
}

// ---------------------------------------------------------------- checks

// thinking_leak, or nothing for a model that always reasons (Settings::always_reasons)
static std::vector<std::string> leaks(const Settings& s, const Stream& r) {
  return s.always_reasons ? std::vector<std::string>{} : thinking_leak(r);
}

std::vector<std::string> thinking_leak(const Stream& r) {
  std::vector<std::string> out;
  if (!strip(r.reasoning).empty()) out.push_back("reasoning text returned (" + std::to_string(chars(r.reasoning)) + " chars)");
  std::string c = r.content;
  size_t a = 0;
  while (a < c.size() && isspace(uint8_t(c[a]))) ++a;
  if (c.compare(a, 7, "<think>") == 0) out.push_back("content starts with <think>");
  else if (c.find("</think>") != std::string::npos) out.push_back("content contains </think>");
  return out;
}

namespace {

// Python's \w and \s for re.findall(r"\w+|[^\w\s]", text): letters, digits (Unicode categories L and N) and "_"; the
// Unicode white space characters
#include "tokenizer/unicode_tables.inc"

template <size_t N>
bool in_ranges(const uint32_t (&t)[N][2], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < t[mid][0]) hi = mid;
    else if (cp > t[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

bool is_word(uint32_t c) {
  if (c < 0x80) return isalnum(int(c)) || c == '_';
  return in_ranges(kLetter, c) || in_ranges(kNumber, c);
}

bool is_space(uint32_t c) {
  if (c < 0x80) return c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F);  // str.isspace
  return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
         c == 0x202F || c == 0x205F || c == 0x3000;
}

}  // namespace

std::optional<double> distinct4(const std::string& text) {
  std::vector<std::string> w;
  for (size_t i = 0; i < text.size();) {
    size_t next;
    const uint32_t c = decode_char(text, i, &next);
    if (is_word(c)) {
      size_t j = next;
      while (j < text.size()) {
        size_t k;
        if (!is_word(decode_char(text, j, &k))) break;
        j = k;
      }
      w.push_back(text.substr(i, j - i));
      i = j;
    } else {
      if (!is_space(c)) w.push_back(text.substr(i, next - i));
      i = next;
    }
  }
  if (w.size() < 8) return std::nullopt;
  std::set<std::string> grams;
  const size_t n = w.size() - 3;
  for (size_t i = 0; i < n; ++i) grams.insert(w[i] + '\x1f' + w[i + 1] + '\x1f' + w[i + 2] + '\x1f' + w[i + 3]);
  return py_round(double(grams.size()) / double(n), 4);
}

std::optional<int> cached_of(const ojson& usage, const ojson& timings) {
  std::optional<int> best;
  auto take = [&](const ojson& v) {
    if (v.is_number()) best = std::max(best.value_or(INT32_MIN), int(v.get<double>()));
  };
  if (usage.is_object() && usage.contains("prompt_tokens_details") && usage["prompt_tokens_details"].is_object() &&
      usage["prompt_tokens_details"].contains("cached_tokens"))
    take(usage["prompt_tokens_details"]["cached_tokens"]);
  if (timings.is_object() && timings.contains("cache_n")) take(timings["cache_n"]);
  return best;
}

double py_round(double x, int digits) {
  // printf rounds the exact binary value, ties to even, as Python's round() does
  char b[64];
  snprintf(b, sizeof b, "%.*f", digits, x);
  return strtod(b, nullptr);
}

double median(std::vector<double> v) {
  if (v.empty()) return NAN;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

Rate decode_rates(const std::vector<double>& t, const std::vector<int>& c, int ct, int head, int tail) {
  // bench.py decode_rates: the middle window of the stream. Each event's token count is exact when the engine sends
  // one token per event; otherwise tokens are assigned to events in proportion to their characters.
  Rate r;
  r.info = {{"events", t.size()}};
  if (t.size() < 2 || ct <= 0) return r;
  if (t.back() > t.front()) r.e2e = (ct - 1) / (t.back() - t.front());
  std::vector<double> cum;
  std::string method;
  if (int(t.size()) == ct) {
    for (int k = 1; k <= ct; ++k) cum.push_back(k);
    method = "events";
  } else {
    long tot = 0;
    for (int x : c) tot += x;
    if (tot <= 0) return r;
    long acc = 0;
    for (int x : c) {
      acc += x;
      cum.push_back(double(acc * long(ct)) / double(tot));  // acc * ct / tot: exact integer product, then one rounding
    }
    method = "chars";
  }
  int i = -1, j = -1;
  for (size_t k = 0; k < t.size(); ++k)
    if (cum[k] >= head) {
      i = int(k);
      break;
    }
  for (size_t k = 0; k < t.size(); ++k)
    if (cum[k] <= ct - tail) j = int(k);
  r.info["rate_method"] = method;
  r.info["tokens_per_event"] = py_round(double(ct) / double(t.size()), 3);
  if (i < 0 || j < 0 || j <= i || t[size_t(j)] <= t[size_t(i)]) return r;
  // Python's round() of the events method's ints stays an int
  if (method == "events") r.info["window_tokens"] = long(cum[size_t(j)] - cum[size_t(i)]);
  else r.info["window_tokens"] = py_round(cum[size_t(j)] - cum[size_t(i)], 1);
  r.info["window_s"] = py_round(t[size_t(j)] - t[size_t(i)], 4);
  r.mid = (cum[size_t(j)] - cum[size_t(i)]) / (t[size_t(j)] - t[size_t(i)]);
  return r;
}

// ---------------------------------------------------------------- the measurements

namespace {

ojson opt(const std::optional<double>& v) { return v ? ojson(*v) : ojson(nullptr); }
ojson secs_or_null(double v) { return v >= 0 ? ojson(v) : ojson(nullptr); }

std::string what(const std::exception& e) {
  std::string s = std::string("request failed: ") + e.what();
  return s.substr(0, 300);
}

int int_or(const ojson& o, const char* k, int def = -1) {
  return o.is_object() && o.contains(k) && o[k].is_number() ? int(o[k].get<double>()) : def;
}

}  // namespace

ojson t_decode(const Settings& s, bool greedy, Nonces& n) {
  const ojson body = body_of(s, decode_messages(n.run()), s.decode_tokens, greedy);
  ojson res = {{"greedy", greedy}, {"invalid", ojson::array()}, {"flags", ojson::array()}};
  Stream r;
  try {
    r = stream_chat(s, body);
  } catch (const std::exception& e) {
    res["invalid"].push_back(what(e));
    return res;
  }
  const int ct = int_or(r.usage, "completion_tokens", 0);
  const Rate rate = decode_rates(r.ev_t, r.ev_c, ct, s.window_head, s.window_tail);
  const std::optional<double> d4 = distinct4(!r.content.empty() ? r.content : r.reasoning);
  res["completion_tokens"] = ct ? ojson(ct) : ojson(nullptr);
  res["chunks"] = r.n_chunks;
  res["ttft_s"] = secs_or_null(r.ttft);
  res["decode_tps"] = opt(rate.mid);
  res["decode_tps_e2e"] = opt(rate.e2e);
  res["e2e_wall_tps"] = ct ? ojson(ct / r.total) : ojson(nullptr);
  res["finish"] = r.finish.empty() ? ojson(nullptr) : ojson(r.finish);
  res["usage_reported"] = ct > 0;
  res["rate"] = rate.info;
  res["distinct4"] = opt(d4);
  for (const auto& why : leaks(s, r)) res["invalid"].push_back(why);
  if (!ct) res["invalid"].push_back("usage.completion_tokens missing (no fallback to event counts)");
  if (!rate.mid && ct) res["invalid"].push_back("too few streamed events for a middle-window rate");
  if (r.finish != "length") res["flags"].push_back("finish_reason=" + (r.finish.empty() ? std::string("None") : r.finish) +
                                                   " (expected length)");
  if (ct && ct != s.decode_tokens)
    res["flags"].push_back("completion_tokens=" + std::to_string(ct) + " != " + std::to_string(s.decode_tokens));
  if (d4 && *d4 < s.rep_threshold) {
    char b[96];
    snprintf(b, sizeof b, "repetition loop (distinct-4-gram %g)", *d4);
    res["flags"].push_back(b);
  }
  if (r.timings.is_object() && r.timings.contains("draft_n") && r.timings["draft_n"].is_number() &&
      r.timings["draft_n"].get<double>() > 0)
    res["draft_acceptance"] = py_round(r.timings.value("draft_n_accepted", 0.0) / r.timings["draft_n"].get<double>(), 4);
  return res;
}

ojson t_prefill(const Settings& s, int size, Nonces& n, const Progress& progress, const std::function<bool()>& stopped) {
  const std::string ctx = build_context(size);
  const std::string nonce = "session " + n.hex(64);
  const ojson msgs = prefill_messages(ctx, nonce, kPrefillQuestion);
  // max_tokens high enough that the answer ends naturally (a turn cut by max_tokens is not reusable as a cached
  // prefix in some engines; real agent turns end with stop or tool calls)
  const ojson body = body_of(s, msgs, s.always_reasons ? 1024 : 256);
  const std::string sid = "bench-" + n.hex(48);
  ojson res = {{"target", size}, {"invalid", ojson::array()}, {"flags", ojson::array()}};
  Stream r;
  try {
    r = stream_chat(s, body, sid);
  } catch (const std::exception& e) {
    res["invalid"].push_back(what(e));
    return res;
  }
  const int pt = int_or(r.usage, "prompt_tokens", 0);
  const auto ref = s.ref_prompt_tokens.find(size);
  const int exp = ref == s.ref_prompt_tokens.end() ? 0 : ref->second;
  const std::optional<int> cold_cached = cached_of(r.usage, r.timings);
  res["prompt_tokens"] = pt ? ojson(pt) : ojson(nullptr);
  res["expected_prompt_tokens"] = exp ? ojson(exp) : ojson(nullptr);
  res["ttft_s"] = secs_or_null(r.ttft);
  res["prefill_tps"] = pt && r.ttft > 0 ? ojson(pt / r.ttft) : ojson(nullptr);
  res["cold_cached_tokens"] = cold_cached ? ojson(*cold_cached) : ojson(nullptr);
  res["finish"] = r.finish.empty() ? ojson(nullptr) : ojson(r.finish);
  res["reasoning_chars"] = chars(r.reasoning);
  for (const auto& why : leaks(s, r)) res["invalid"].push_back(why);
  if (!pt) {
    res["invalid"].push_back("usage.prompt_tokens missing");
  } else if (exp) {
    if (std::abs(pt - exp) > kPromptTol * exp)
      res["invalid"].push_back("prompt_tokens " + std::to_string(pt) + " outside ±1% of " + std::to_string(exp) +
                               " (truncated or template drift)");
  } else {
    res["flags"].push_back("no reference prompt_tokens for this corpus/size (unchecked)");
  }
  if (cold_cached && *cold_cached > kColdCacheMax)
    res["invalid"].push_back("cold request reported " + std::to_string(*cold_cached) + " cached tokens");
  // the agent pattern: the same prefix, the previous turns and a new question should hit the prefix cache
  ojson cached = {{"samples", ojson::array()}, {"invalid", ojson::array()}, {"flags", ojson::array()}};
  if (strip(r.content).empty()) {
    cached["invalid"].push_back("turn 1 returned no content; not substituting text, cached test skipped");
    res["cached"] = cached;
    return res;
  }
  if (r.finish != "stop")
    cached["flags"].push_back("turn 1 finish_reason=" + (r.finish.empty() ? std::string("None") : r.finish) +
                              " (engines may not cache a 'length' turn)");
  ojson hist = msgs;
  hist.push_back({{"role", "assistant"}, {"content", r.content}});
  int prev_pt = pt;
  for (int k = 0; k < std::min(s.cached_samples, 3); ++k) {
    if (stopped && stopped()) break;
    if (progress) progress("follow-up " + std::to_string(k + 1));
    // the tool's run or the user's think time between agent turns
    std::this_thread::sleep_for(std::chrono::milliseconds(long(s.turn_gap * 1000)));
    ojson msgs2 = hist;
    msgs2.push_back({{"role", "user"}, {"content", kFollowups[k]}});
    Stream r2;
    try {
      r2 = stream_chat(s, body_of(s, msgs2, s.always_reasons ? 1024 : 256), sid);
    } catch (const std::exception& e) {
      cached["invalid"].push_back(("follow-up request failed: " + std::string(e.what())).substr(0, 300));
      break;
    }
    const std::optional<int> c2 = cached_of(r2.usage, r2.timings);
    const int pt2 = int_or(r2.usage, "prompt_tokens", 0);
    cached["samples"].push_back({{"ttft_s", secs_or_null(r2.ttft)},
                                 {"prompt_tokens", pt2 ? ojson(pt2) : ojson(nullptr)},
                                 {"cached_tokens", c2 ? ojson(*c2) : ojson(nullptr)},
                                 {"finish", r2.finish.empty() ? ojson(nullptr) : ojson(r2.finish)}});
    for (const auto& why : leaks(s, r2)) cached["invalid"].push_back("follow-up: " + why);
    if (c2 && prev_pt && *c2 < 0.5 * prev_pt)
      cached["flags"].push_back("follow-up reused only " + std::to_string(*c2) + " of " + std::to_string(prev_pt) +
                                " prefix tokens");
    if (strip(r2.content).empty()) {
      cached["flags"].push_back("follow-up returned no content; stopping the cached series");
      break;
    }
    hist = msgs2;
    hist.push_back({{"role", "assistant"}, {"content", r2.content}});
    if (pt2) prev_pt = pt2;
  }
  std::vector<double> tt;
  for (const auto& x : cached["samples"])
    if (x["ttft_s"].is_number() && x["ttft_s"].get<double>() > 0) tt.push_back(x["ttft_s"]);
  cached["median_ttft_s"] = tt.empty() ? ojson(nullptr) : ojson(py_round(median(tt), 4));
  res["cached"] = cached;
  res["turn1_finish"] = res["finish"];
  return res;
}

namespace {

// bench.py _check_tc: a read_file call whose path names main.py
bool check_tc(const std::vector<ToolCallAcc>& tcs) {
  bool ok = false;
  for (const auto& tc : tcs) {
    const ojson parsed = tc.arguments.empty() ? ojson::object() : ojson::parse(tc.arguments, nullptr, false);
    if (parsed.is_discarded()) continue;
    if (tc.name == "read_file" && parsed.is_object() && parsed.contains("path") && parsed["path"].is_string() &&
        parsed["path"].get<std::string>().find("main.py") != std::string::npos)
      ok = true;
  }
  return ok;
}

ojson tool_calls_json(const std::vector<ToolCallAcc>& tcs) {
  ojson a = ojson::array();
  for (const auto& t : tcs)
    a.push_back({{"id", t.has_id ? ojson(t.id) : ojson(nullptr)}, {"name", t.name}, {"arguments", t.arguments}});
  return a;
}

std::string lower(std::string s) {
  for (auto& ch : s) ch = char(tolower(uint8_t(ch)));
  return s;
}

}  // namespace

ojson t_tools(const Settings& s, bool think) {
  const ojson msgs = tools_messages();
  const ojson body = body_of(s, msgs, think || s.always_reasons ? 4096 : 512, false, think,
                             {{"tools", tools()}, {"tool_choice", "auto"}});
  ojson out = {{"thinking", think}};
  Stream r;
  try {
    r = stream_chat(s, body);
  } catch (const std::exception& e) {
    out["stream"] = {{"ok", false}, {"error", std::string(e.what()).substr(0, 300)}, {"tool_calls", ojson::array()}};
    out["nonstream"] = {{"ok", false}, {"error", "skipped"}};
    return out;
  }
  const bool leaked = r.content.find("<tool_call") != std::string::npos || r.content.find("<function") != std::string::npos;
  out["stream"] = {{"ok", check_tc(r.tool_calls)},
                   {"tool_calls", tool_calls_json(r.tool_calls)},
                   {"finish", r.finish.empty() ? ojson(nullptr) : ojson(r.finish)},
                   {"ttft_s", secs_or_null(r.ttft)},
                   {"total_s", r.total},
                   {"content_leak", leaked},
                   {"reasoning_chars", chars(r.reasoning)}};
  if (!think && !leaks(s, r).empty()) out["stream"]["thinking_leak"] = thinking_leak(r);
  // the same request, not streamed
  try {
    ojson b2 = body;
    b2["stream"] = false;
    const auto t0 = clock::now();
    const ojson resp = post_json(s, b2);
    const ojson& m = resp.at("choices").at(0).at("message");
    std::vector<ToolCallAcc> tcs;
    if (m.contains("tool_calls") && m["tool_calls"].is_array())
      for (const auto& t : m["tool_calls"]) {
        ToolCallAcc a;
        a.name = t.at("function").at("name");
        const ojson& args = t["function"]["arguments"];
        a.arguments = args.is_string() ? args.get<std::string>() : args.dump();
        if (t.contains("id") && t["id"].is_string()) a.id = t["id"], a.has_id = true;
        tcs.push_back(a);
      }
    const ojson& fin = resp["choices"][0].contains("finish_reason") ? resp["choices"][0]["finish_reason"] : ojson(nullptr);
    out["nonstream"] = {{"ok", check_tc(tcs)}, {"tool_calls", tool_calls_json(tcs)}, {"finish", fin}, {"total_s", since(t0)}};
  } catch (const std::exception& e) {
    out["nonstream"] = {{"ok", false}, {"error", std::string(e.what()).substr(0, 300)}};
  }
  // the round trip: the tool's result back, a final answer expected (and no failure on role=tool)
  if (!r.tool_calls.empty()) {
    const ToolCallAcc& tc = r.tool_calls[0];
    const std::string call_id = tc.has_id && !tc.id.empty() ? tc.id : "call_0";
    ojson msgs2 = msgs;
    msgs2.push_back({{"role", "assistant"},
                     {"content", ""},
                     {"tool_calls", ojson::array({{{"id", call_id},
                                                   {"type", "function"},
                                                   {"function", {{"name", tc.name}, {"arguments", tc.arguments.empty() ? "{}" : tc.arguments}}}}})}});
    msgs2.push_back({{"role", "tool"}, {"tool_call_id", call_id}, {"content", kToolResult}});
    try {
      ojson b3 = body;
      b3["messages"] = msgs2;
      const Stream r2 = stream_chat(s, b3);
      const std::string low = lower(r2.content);
      const bool ok = !strip(low).empty() && (low.find("hello") != std::string::npos || low.find("print") != std::string::npos ||
                                              low.find("main") != std::string::npos);
      out["round_trip"] = {{"ok", ok},
                           {"finish", r2.finish.empty() ? ojson(nullptr) : ojson(r2.finish)},
                           {"more_tool_calls", r2.tool_calls.size()}};
    } catch (const std::exception& e) {
      out["round_trip"] = {{"ok", false}, {"error", std::string(e.what()).substr(0, 300)}};
    }
  }
  return out;
}

bool tools_pass(const ojson& t) {
  if (!t.is_array() || t.empty()) return false;
  for (const auto& x : t) {
    const bool rt = x.contains("round_trip") && x["round_trip"].value("ok", false);
    if (!(x["stream"].value("ok", false) && x["nonstream"].value("ok", false) && rt)) return false;
  }
  return true;
}

ojson summary(const ojson& runs, const char* key) {
  std::vector<double> v, e;
  std::set<std::string> invalid, flags;
  int valid = 0;
  for (const auto& x : runs) {
    for (const auto& r : x["invalid"]) invalid.insert(r.get<std::string>());
    for (const auto& f : x["flags"]) flags.insert(f.get<std::string>());
    if (!x["invalid"].empty() || !x.contains(key) || !x[key].is_number() || x[key].get<double>() == 0) continue;
    ++valid;
    v.push_back(x[key]);
    if (x.contains("decode_tps_e2e") && x["decode_tps_e2e"].is_number() && x["decode_tps_e2e"].get<double>() != 0)
      e.push_back(x["decode_tps_e2e"]);
  }
  std::sort(v.begin(), v.end());
  ojson s = {{"runs", runs.size()}, {"valid_runs", valid}};
  if (!v.empty()) {
    s["median_tps"] = py_round(median(v), 2);
    s["min_tps"] = py_round(v.front(), 2);
    s["max_tps"] = py_round(v.back(), 2);
    if (!e.empty()) s["median_tps_e2e"] = py_round(median(e), 2);
  }
  s["invalid"] = invalid;
  s["flags"] = flags;
  return s;
}

ojson warmup(const Settings& s, Nonces& n, const Progress& progress) {
  // compare_engines.py's warm-up (bench.warmup, not adaptive): a ~2.9K-token prompt with a 64-token answer, then a
  // tool call, with the benchmark's own sampling and thinking settings, before anything is timed
  ojson steps = ojson::array();
  auto step = [&](const std::string& name, const std::function<Stream()>& fn) {
    if (progress) progress(name);
    const auto t0 = clock::now();
    ojson rec;
    try {
      const Stream r = fn();
      rec = {{"name", name}, {"ok", true}, {"s", py_round(since(t0), 2)},
             {"prompt_tokens", int_or(r.usage, "prompt_tokens") >= 0 ? ojson(int_or(r.usage, "prompt_tokens")) : ojson(nullptr)},
             {"completion_tokens", int_or(r.usage, "completion_tokens") >= 0 ? ojson(int_or(r.usage, "completion_tokens")) : ojson(nullptr)},
             {"finish", r.finish.empty() ? ojson(nullptr) : ojson(r.finish)},
             {"thinking_leak", thinking_leak(r)}};
    } catch (const std::exception& e) {
      rec = {{"name", name}, {"ok", false}, {"s", py_round(since(t0), 2)}, {"error", std::string(e.what()).substr(0, 300)}};
    }
    steps.push_back(rec);
  };
  const std::string ctx = build_context(kWarmTokens, kWarmOffsetChars);
  const ojson msgs = prefill_messages(ctx, "warmup " + n.hex(64), kWarmQuestion);
  step("prefill~2.9k+decode64", [&] { return stream_chat(s, body_of(s, msgs, 64)); });
  step("tools", [&] { return stream_chat(s, body_of(s, tools_messages(), 512, false, false, {{"tools", tools()}, {"tool_choice", "auto"}})); });
  bool all = true;
  for (const auto& x : steps) all = all && x["ok"].get<bool>();
  return {{"adaptive", false}, {"steps", steps}, {"ok", all}};
}

}  // namespace ni::bench
