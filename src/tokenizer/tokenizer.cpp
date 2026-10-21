#include "tokenizer/tokenizer.h"
#include "util/log.h"

#include <nlohmann/json.hpp>

#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <fstream>

namespace ni {

#include "tokenizer/unicode_tables.inc"
#include "tokenizer/unicode_tables_cased.inc"

namespace {

template <size_t N>
bool in_table(const uint32_t (&t)[N][2], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (cp < t[mid][0]) hi = mid;
    else if (cp > t[mid][1]) lo = mid + 1;
    else return true;
  }
  return false;
}

inline bool is_L(uint32_t c) { return c < 0x80 ? ((c | 32) - 'a' < 26) : in_table(kLetter, c); }
inline bool is_M(uint32_t c) { return c >= 0x300 && in_table(kMark, c); }
inline bool is_N(uint32_t c) { return c < 0x80 ? (c - '0' < 10) : in_table(kNumber, c); }
inline bool is_space(uint32_t c) {  // Unicode White_Space
  return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
         c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}
inline bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

struct CP {
  uint32_t c;
  uint32_t off;  // byte offset in the UTF-8 string
};

std::vector<CP> to_cps(const std::string& s) {
  std::vector<CP> out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    unsigned char b = s[i];
    uint32_t c;
    int n;
    if (b < 0x80) { c = b; n = 1; }
    else if ((b >> 5) == 6 && i + 1 < s.size()) { c = b & 0x1F; n = 2; }
    else if ((b >> 4) == 14 && i + 2 < s.size()) { c = b & 0x0F; n = 3; }
    else if ((b >> 3) == 30 && i + 3 < s.size()) { c = b & 0x07; n = 4; }
    else { c = 0xFFFD; n = 1; }  // invalid byte: treat as one opaque code point
    for (int k = 1; k < n; ++k) c = (c << 6) | (s[i + k] & 0x3F);
    out.push_back({c, uint32_t(i)});
    i += n;
  }
  return out;
}

// Qwen3.5 pre-tokenizer:
// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
void pretokenize(const std::string& s, std::vector<std::string>& out) {
  std::vector<CP> cps = to_cps(s);
  const size_t n = cps.size();
  auto cp = [&](size_t i) -> uint32_t { return i < n ? cps[i].c : 0xFFFFFFFF; };
  auto LM = [&](size_t i) { return i < n && (is_L(cps[i].c) || is_M(cps[i].c)); };
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    uint32_t c = cps[i].c;
    // 1. contractions
    if (c == '\'' && i + 1 < n) {
      uint32_t a = cp(i + 1) | 32, b = i + 2 < n ? (cp(i + 2) | 32) : 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
      else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
    }
    // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    if (j == i) {
      size_t k = i;
      if (!is_nl(c) && !is_L(c) && !is_N(c) && LM(i + 1)) k = i + 1;
      if (LM(k)) {
        while (LM(k)) ++k;
        j = k;
      }
    }
    // 3. \p{N}
    if (j == i && is_N(c)) j = i + 1;
    // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
    if (j == i) {
      size_t k = i;
      auto punct = [&](size_t q) {
        if (q >= n) return false;
        uint32_t x = cps[q].c;
        return !is_space(x) && !is_L(x) && !is_M(x) && !is_N(x);
      };
      if (c == ' ' && punct(i + 1)) k = i + 1;
      if (punct(k)) {
        while (punct(k)) ++k;
        while (k < n && is_nl(cps[k].c)) ++k;
        j = k;
      }
    }
    // 5-7. whitespace
    if (j == i && is_space(c)) {
      size_t k = i;
      while (k < n && is_space(cps[k].c)) ++k;  // maximal whitespace run [i, k)
      size_t last_nl = n;
      for (size_t q = i; q < k; ++q)
        if (is_nl(cps[q].c)) last_nl = q;
      if (last_nl != n) {
        j = last_nl + 1;                         // \s*[\r\n]+
      } else if (k < n && k - i > 1) {
        j = k - 1;                               // \s+(?!\S): leave one space for the next word
      } else {
        j = k;                                   // \s+
      }
    }
    if (j == i) j = i + 1;  // unreachable for valid input; never stall
    uint32_t b0 = cps[i].off, b1 = j < n ? cps[j].off : uint32_t(s.size());
    out.emplace_back(s.substr(b0, b1 - b0));
    i = j;
  }
}

inline bool is_Lut(uint32_t c) { return c < 0x80 ? (c - 'A' < 26) : in_table(kUpperTitle, c); }   // Lu, Lt
inline bool is_Ll(uint32_t c) { return c < 0x80 ? (c - 'a' < 26) : in_table(kLowerCase, c); }      // Ll
inline bool is_Lmo(uint32_t c) { return c >= 0x80 && in_table(kModOther, c); }                    // Lm, Lo

// Tekken pre-tokenizer (Mistral's tokenizer.json, Nemotron; transformers applies the same pattern with
// fix_mistral_regex):
// [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
// |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
// |\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Each alternative is matched as the regex engine does (first alternative that matches, greedy with
// backtracking): A* B+ backtracks A* until a B follows.
void pretokenize_tekken(const std::string& s, std::vector<std::string>& out) {
  std::vector<CP> cps = to_cps(s);
  const size_t n = cps.size();
  auto at = [&](size_t i) { return cps[i].c; };
  auto P = [&](size_t i) { return i < n && !is_nl(at(i)) && !is_L(at(i)) && !is_N(at(i)); };
  auto A = [&](size_t i) { return i < n && (is_Lut(at(i)) || is_Lmo(at(i)) || is_M(at(i))); };
  auto B = [&](size_t i) { return i < n && (is_Ll(at(i)) || is_Lmo(at(i)) || is_M(at(i))); };
  auto Q = [&](size_t i) { return i < n && !is_space(at(i)) && !is_L(at(i)) && !is_N(at(i)); };
  size_t i = 0;
  while (i < n) {
    size_t j = i;
    // 1. P? A* B+
    for (int pre = P(i) ? 1 : 0; pre >= 0 && j == i; --pre) {
      const size_t j0 = i + size_t(pre);
      size_t a_end = j0;
      while (A(a_end)) ++a_end;
      for (size_t k = a_end + 1; k-- > j0;) {
        if (!B(k)) continue;
        size_t e = k;
        while (B(e)) ++e;
        j = e;
        break;
      }
    }
    // 2. P? A+ B*
    for (int pre = P(i) ? 1 : 0; pre >= 0 && j == i; --pre) {
      const size_t j0 = i + size_t(pre);
      size_t a_end = j0;
      while (A(a_end)) ++a_end;
      if (a_end == j0) continue;
      while (B(a_end)) ++a_end;
      j = a_end;
    }
    // 3. \p{N}
    if (j == i && is_N(at(i))) j = i + 1;
    // 4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
    if (j == i) {
      size_t k = at(i) == ' ' && Q(i + 1) ? i + 1 : i;
      if (Q(k)) {
        while (Q(k)) ++k;
        while (k < n && (is_nl(at(k)) || at(k) == '/')) ++k;
        j = k;
      }
    }
    // 5-7. whitespace, as in the Qwen split
    if (j == i && is_space(at(i))) {
      size_t k = i;
      while (k < n && is_space(at(k))) ++k;
      size_t last_nl = n;
      for (size_t q = i; q < k; ++q)
        if (is_nl(at(q))) last_nl = q;
      if (last_nl != n) j = last_nl + 1;
      else if (k < n && k - i > 1) j = k - 1;
      else j = k;
    }
    if (j == i) j = i + 1;
    const uint32_t b0 = cps[i].off, b1 = j < n ? cps[j].off : uint32_t(s.size());
    out.emplace_back(s.substr(b0, b1 - b0));
    i = j;
  }
}

std::string nfc(const std::string& s) {
  bool ascii = std::all_of(s.begin(), s.end(), [](char c) { return (unsigned char)c < 0x80; });
  if (ascii) return s;
  CFStringRef in = CFStringCreateWithBytes(nullptr, (const UInt8*)s.data(), CFIndex(s.size()), kCFStringEncodingUTF8, false);
  if (!in) return s;
  CFMutableStringRef m = CFStringCreateMutableCopy(nullptr, 0, in);
  CFStringNormalize(m, kCFStringNormalizationFormC);
  CFIndex len = CFStringGetLength(m), need = 0;
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, nullptr, 0, &need);
  std::string out(size_t(need), '\0');
  CFStringGetBytes(m, CFRangeMake(0, len), kCFStringEncodingUTF8, 0, false, (UInt8*)out.data(), need, nullptr);
  CFRelease(m);
  CFRelease(in);
  return out;
}

// GPT-2 byte <-> unicode mapping
void byte_unicode(uint32_t (&b2u)[256]) {
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    b2u[b] = keep ? uint32_t(b) : uint32_t(256 + n++);
  }
}

void put_utf8(std::string& s, uint32_t c) {
  if (c < 0x80) s += char(c);
  else if (c < 0x800) { s += char(0xC0 | (c >> 6)); s += char(0x80 | (c & 0x3F)); }
  else if (c < 0x10000) { s += char(0xE0 | (c >> 12)); s += char(0x80 | ((c >> 6) & 0x3F)); s += char(0x80 | (c & 0x3F)); }
  else { s += char(0xF0 | (c >> 18)); s += char(0x80 | ((c >> 12) & 0x3F)); s += char(0x80 | ((c >> 6) & 0x3F)); s += char(0x80 | (c & 0x3F)); }
}

} // namespace

void Tokenizer::load(const std::string& path) {
  double t0 = now_ms();
  std::ifstream f(path);
  if (!f) fatal("cannot read %s", path.c_str());
  nlohmann::json j = nlohmann::json::parse(f);
  auto& model = j["model"];
  if (model["type"] != "BPE") fatal("only BPE tokenizers are supported");
  // Tekken (Mistral, Nemotron 3.5): its Split regex, no normalizer, whole words looked up first (ignore_merges).
  // Anything else gets Qwen's split with NFC.
  static const char* kTekken =
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|"
      "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|"
      "\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  const auto& pt = j["pre_tokenizer"];
  if (pt.is_object() && pt.value("type", "") == "Sequence" && pt["pretokenizers"].is_array())
    for (auto& p : pt["pretokenizers"])
      if (p.value("type", "") == "Split" && p.contains("pattern") && p["pattern"].contains("Regex") &&
          p["pattern"]["Regex"] == kTekken) {
        tekken_ = true;
        nfc_ = !j["normalizer"].is_null();
        ignore_merges_ = model.value("ignore_merges", false);
      }
  uint32_t b2u[256];
  byte_unicode(b2u);
  std::unordered_map<uint32_t, uint8_t> u2b;
  for (int b = 0; b < 256; ++b) u2b[b2u[b]] = uint8_t(b);

  int max_id = 0;
  for (auto& [tok, id] : model["vocab"].items()) max_id = std::max(max_id, id.get<int>());
  for (auto& a : j["added_tokens"]) max_id = std::max(max_id, a["id"].get<int>());
  id_to_bytes_.assign(max_id + 1, "");
  special_.assign(max_id + 1, false);
  added_id_.assign(max_id + 1, false);
  for (auto& [tok, idj] : model["vocab"].items()) {
    int id = idj.get<int>();
    vocab_[tok] = id;
    std::string raw;
    for (auto& c : to_cps(tok)) {
      auto it = u2b.find(c.c);
      raw += it != u2b.end() ? char(it->second) : '?';
    }
    id_to_bytes_[id] = raw;
  }
  for (int b = 0; b < 256; ++b) {
    std::string u;
    put_utf8(u, b2u[b]);
    auto it = vocab_.find(u);
    byte_id_[b] = it == vocab_.end() ? -1 : it->second;
  }
  int rank = 0;
  for (auto& m : model["merges"]) {
    std::string a, b;
    if (m.is_array()) {
      a = m[0];
      b = m[1];
    } else {
      std::string s = m;
      size_t sp = s.find(' ');
      a = s.substr(0, sp);
      b = s.substr(sp + 1);
    }
    auto ia = vocab_.find(a), ib = vocab_.find(b), im = vocab_.find(a + b);
    if (ia != vocab_.end() && ib != vocab_.end() && im != vocab_.end())
      merges_[(uint64_t(uint32_t(ia->second)) << 32) | uint32_t(ib->second)] = {rank, im->second};
    ++rank;
  }
  for (auto& a : j["added_tokens"]) {
    int id = a["id"];
    std::string content = a["content"];
    added_.push_back({content, id});
    id_to_bytes_[id] = content;
    special_[id] = a.value("special", false);
    added_id_[id] = true;
  }
  std::sort(added_.begin(), added_.end(), [](auto& x, auto& y) { return x.first.size() > y.first.size(); });
  logi("tokenizer: %zu tokens, %zu merges, %zu added (%.0f ms)", vocab_.size(), merges_.size(), added_.size(),
       now_ms() - t0);
}

void Tokenizer::bpe(const std::string& word, std::vector<int>& out) const {
  {
    std::lock_guard<std::mutex> lk(cache_mu_);
    auto it = cache_.find(word);
    if (it != cache_.end()) {
      out.insert(out.end(), it->second.begin(), it->second.end());
      return;
    }
  }
  if (ignore_merges_) {
    std::string u;
    for (unsigned char b : word) {
      static const auto b2u = [] {
        std::array<uint32_t, 256> t{};
        uint32_t raw[256];
        byte_unicode(raw);
        for (int i = 0; i < 256; ++i) t[i] = raw[i];
        return t;
      }();
      put_utf8(u, b2u[b]);
    }
    auto it = vocab_.find(u);
    if (it != vocab_.end()) {
      out.push_back(it->second);
      return;
    }
  }
  std::vector<int> sym;
  sym.reserve(word.size());
  for (unsigned char b : word) sym.push_back(byte_id_[b]);
  while (sym.size() > 1) {
    int best_rank = INT32_MAX, best_i = -1, best_id = -1;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = merges_.find((uint64_t(uint32_t(sym[i])) << 32) | uint32_t(sym[i + 1]));
      if (it != merges_.end() && it->second.first < best_rank) {
        best_rank = it->second.first;
        best_i = int(i);
        best_id = it->second.second;
      }
    }
    if (best_i < 0) break;
    // merge every occurrence of this pair left to right
    std::vector<int> nxt;
    nxt.reserve(sym.size());
    int a = sym[best_i], b = sym[best_i + 1];
    for (size_t i = 0; i < sym.size(); ++i) {
      if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
        nxt.push_back(best_id);
        ++i;
      } else {
        nxt.push_back(sym[i]);
      }
    }
    sym.swap(nxt);
  }
  out.insert(out.end(), sym.begin(), sym.end());
  if (word.size() <= 64) {
    std::lock_guard<std::mutex> lk(cache_mu_);
    if (cache_.size() > 200000) cache_.clear();
    cache_.emplace(word, std::move(sym));
  }
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int>& out) const {
  if (text.empty()) return;
  std::vector<std::string> words;
  if (tekken_) pretokenize_tekken(nfc_ ? nfc(text) : text, words);
  else pretokenize(nfc_ ? nfc(text) : text, words);
  for (auto& w : words) bpe(w, out);
}

std::vector<int> Tokenizer::encode(const std::string& text, size_t split_at, int* split_index) const {
  std::vector<int> out;
  size_t start = 0, i = 0;
  while (i < text.size()) {
    bool hit = false;
    if (text[i] == '<' || text[i] == '[') {
      for (auto& [content, id] : added_) {
        if (text.compare(i, content.size(), content) == 0) {
          encode_plain(text.substr(start, i - start), out);
          if (split_index && i == split_at) *split_index = int(out.size());
          out.push_back(id);
          i += content.size();
          start = i;
          hit = true;
          break;
        }
      }
    }
    if (!hit) ++i;
  }
  encode_plain(text.substr(start), out);
  return out;
}

const std::string& Tokenizer::token_bytes(int id) const {
  static const std::string empty;
  return id >= 0 && id < int(id_to_bytes_.size()) ? id_to_bytes_[id] : empty;
}

bool Tokenizer::is_special(int id) const { return id >= 0 && id < int(special_.size()) && special_[id]; }
bool Tokenizer::is_added(int id) const { return id >= 0 && id < int(added_id_.size()) && added_id_[id]; }

int Tokenizer::find(const std::string& token) const {
  for (auto& [c, id] : added_)
    if (c == token) return id;
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
  std::string s;
  for (int id : ids) s += token_bytes(id);
  return s;
}

std::string StreamDecoder::push(int id) {
  pending_ += tok_.token_bytes(id);
  // emit the longest prefix that ends on a UTF-8 boundary
  size_t n = pending_.size(), cut = n;
  for (size_t back = 1; back <= 4 && back <= n; ++back) {
    unsigned char c = pending_[n - back];
    if ((c & 0xC0) == 0x80) continue;  // continuation byte
    int need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (int(back) < need) cut = n - back;  // incomplete sequence at the end
    break;
  }
  std::string out = pending_.substr(0, cut);
  pending_.erase(0, cut);
  return utf8_valid(out);
}

std::string StreamDecoder::flush() {
  std::string out;
  out.swap(pending_);
  return utf8_valid(out);
}

std::string utf8_valid(const std::string& s) {
  const size_t n = s.size();
  size_t i = 0;
  std::string out;
  bool fixed = false;
  while (i < n) {
    const unsigned char c = s[i];
    if (c < 0x80) {
      if (fixed) out += char(c);
      ++i;
      continue;
    }
    // continuation bytes after the lead byte, and the allowed range of the first one (no overlongs, surrogates
    // or code points past U+10FFFF)
    int need = 0;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) need = 1;
    else if (c >= 0xE0 && c <= 0xEF) need = 2, lo = c == 0xE0 ? 0xA0 : 0x80, hi = c == 0xED ? 0x9F : 0xBF;
    else if (c >= 0xF0 && c <= 0xF4) need = 3, lo = c == 0xF0 ? 0x90 : 0x80, hi = c == 0xF4 ? 0x8F : 0xBF;
    size_t j = i + 1;
    int k = 0;
    for (; k < need && j < n; ++k, ++j) {
      const unsigned char d = s[j];
      if (d < (k ? 0x80 : lo) || d > (k ? 0xBF : hi)) break;
    }
    if (need > 0 && k == need) {
      if (fixed) out.append(s, i, j - i);
    } else {  // s[i, j) is the maximal subpart: the lead byte and the continuation bytes that fit it
      if (!fixed) out.assign(s, 0, i), fixed = true;
      out += "\xEF\xBF\xBD";
    }
    i = j;
  }
  return fixed ? out : s;
}

} // namespace ni
