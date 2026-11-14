// Tokenizer check, CPU only, no model folder: Flint's pre-tokenizer split (pretokenize_flint in
// src/tokenizer/tokenizer.cpp) against the pieces Hugging Face tokenizers 0.23.3 makes of the same texts with Flint's
// tokenizer.json (its added tokens split off first, then its pre_tokenizer). The test loads a tokenizer.json with
// Flint's pre_tokenizer, the 256 byte tokens and every run of adjacent pieces as a token, and ignore_merges: with the
// right split each piece is one token, and a piece cut in two or two pieces taken as one give other ids (HF gives one
// token per piece on the same tokenizer.json). load() must take the split only for exactly Flint's Sequence.
//   tokenizer_test   exit status 1 if any check fails
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
  if (ok && name.rfind("LFM2", 0) == 0) printf("ok %s\n", name.c_str());  // the autoresearch metric's checks
  ok ? ++passes : ++fails;
}

struct Case {
  const char* what;
  const char* text;
  std::vector<std::string> pieces;  // as HF splits the text; an added token is a piece of its own
};

const Case kCases[] = {
    {"words, a contraction, punctuation", "Hello world, it's a test.",
     {"Hello", " world", ",", " it's", " a", " test", "."}},
    {"contractions in any case, after the letters (U+017F folds to s)",
     "it's IT'S It'S they'RE we've YOU'LL I'm she'd don't x'\u017F",
     {"it's", " IT'S", " It'S", " they'RE", " we've", " YOU'LL", " I'm", " she'd", " don't", " x'\u017F"}},
    {"no contraction: U+2019, a quote before letters, after spaces or digits",
     "IT\u2019S o'clock rock'n'roll 'tis ''s ABC's 123's",
     {"IT", "\u2019S", " o", "'clock", " rock", "'n", "'roll", " '", "tis", " ''", "s", " ABC's", " ", "123", "'s"}},
    {"digits in threes; a piece's trailing spaces stay whole", "a1b22c333d4444e55555f 1234567 x  123",
     {"a", "1", "b", "22", "c", "333", "d", "444", "4", "e", "555", "55", "f", " ", "123", "456", "7", " x", "  ",
      "123"}},
    {"numbers other than ASCII digits (No, Nd, Nl)", "\u00BD\u00BD\u00BD\u00BD \u0661\u0662\u0663\u0664 \u216B",
     {"\u00BD\u00BD\u00BD", "\u00BD", " ", "\u0661\u0662\u0663", "\u0664", " ", "\u216B"}},
    {"CJK and kana runs, split again by the word split",
     "\u6771\u4EAC\u30BF\u30EF\u30FC\u306F333\u30E1\u30FC\u30C8\u30EB\u3002\u30AB\u30BF\u30AB\u30CA\u30FB\u30C6"
     "\u30B9\u30C8 \u65E5\u672C\u8A9E\u306Etest \u4E2D\u6587English",
     {"\u6771\u4EAC\u30BF\u30EF\u30FC\u306F", "333", "\u30E1\u30FC\u30C8\u30EB", "\u3002", "\u30AB\u30BF\u30AB\u30CA",
      "\u30FB\u30C6\u30B9\u30C8", " ", "\u65E5\u672C\u8A9E\u306E", "test", " ", "\u4E2D\u6587", "English"}},
    {"the ends of the CJK and kana ranges", "\u9FA5\u9FA6\u3400 \u30A0\u309B\u3099\u30FF\u3040x",
     {"\u9FA5", "\u9FA6\u3400", " ", "\u30A0\u309B", "\u3099\u30FF", "\u3040", "x"}},
    {"emoji: a ZWJ is a piece of its own; a variation selector joins a symbol that starts a piece",
     "\U0001F468\u200D\U0001F469\u200D\U0001F467 \u2705\uFE0F ok\u2705\uFE0F \U0001F44D\U0001F3FD!",
     {"\U0001F468", "\u200D", "\U0001F469", "\u200D", "\U0001F467", " \u2705", "\uFE0F", " ok", "\u2705\uFE0F",
      " \U0001F44D\U0001F3FD!"}},
    {"accents, combining marks, title case, case changes",
     "caf\u00E9 cafe\u0301 \u0301abc \u01C5emal HTTPServer getHTTPResponse McDONALD's",
     {"caf\u00E9", " cafe\u0301", " \u0301abc", " \u01C5emal", " HTTPServer", " get", "HTTPResponse", " Mc",
      "DONALD's"}},
    {"whitespace other than spaces, control and format characters",
     "a\u3000\u3000b a\302\205b x\u00A0\u00A0y \034 \001\002abc a\u200Bb \u2028z",
     {"a", "\u3000", "\u3000b", " a", "\302\205b", " x", "\u00A0", "\u00A0y", " ", "\034", " ", "\001", "\002abc",
      " a", "\u200Bb", " ", "\u2028z"}},
    {"CR LF, tabs, spaces at the end", "line1\r\nline2\r\n\r\n  x\ttab\t\tend   ",
     {"line", "1", "\r\n", "line", "2", "\r\n\r\n", " ", " x", "\ttab", "\t", "\tend", "   "}},
    {"code", "def f(x):\n    return x**2  # comment\n\n\n\tpass\n",
     {"def", " f", "(x", "):\n", "   ", " return", " x", "**", "2", " ", " #", " comment", "\n\n\n", "\tpass", "\n"}},
    {"punctuation and symbol runs",
     "!!! ... ?!?! ---> (((a))) \"q\" #include <vector> \u00A9 50\u00B0C \u20AC100 a/b",
     {"!!!", " ...", " ?!?!", " --->", " (((", "a", ")))", " \"", "q", "\"", " #", "include", " <", "vector", ">",
      " \u00A9", " ", "50", "\u00B0C", " \u20AC", "100", " a", "/b"}},
    {"added tokens", "<think>\nhi\n</think>\n\n<tool_call>{\"a\": 1}</tool_call><|user|> x",
     {"<think>", "\n", "hi", "\n", "</think>", "\n\n", "<tool_call>", "{\"", "a", "\":", " ", "1", "}",
      "</tool_call>", "<|user|>", " x"}},
    {"only spaces", "   ",
     {"   "}},
    {"only newlines and a space", "\n \n",
     {"\n \n"}},
    {"a lone quote", "'",
     {"'"}},
};

const std::vector<std::string> kAdded = {"<think>", "</think>", "<tool_call>", "</tool_call>", "<|user|>"};

// LFM2's pre-tokenizer (Liquid, LFM3; one Isolated split): the pieces HF tokenizers 0.22.1 makes of the same
// texts with the regex (pretokenize_lfm2). Differs from Qwen3.5's where its letters leave the combining marks
// out, its digits go in threes and its contractions stand alone; from Flint's where it has no CJK split and its
// symbol run takes a ZWJ or variation selector with the emoji.
const Case kLfm2Cases[] = {
    {"LFM2: words, a contraction of its own, punctuation", "Hello world, it's a test.",
     {"Hello", " world", ",", " it", "'s", " a", " test", "."}},
    {"LFM2: contractions in any case, split off (U+017F folds to s)",
     "it's IT'S It'S they'RE we've YOU'LL I'm she'd don't x'\u017F",
     {"it", "'s", " IT", "'S", " It", "'S", " they", "'RE", " we", "'ve", " YOU", "'LL", " I", "'m", " she", "'d",
      " don", "'t", " x", "'\u017F"}},
    {"LFM2: a space takes the quote with it before a non-contraction ('tis)",
     "IT\u2019S o'clock rock'n'roll 'tis ''s ABC's 123's",
     {"IT", "\u2019S", " o", "'clock", " rock", "'n", "'roll", " '", "tis", " ''", "s", " ABC", "'s", " ", "123",
      "'s"}},
    {"LFM2: digits in threes, a run's leftover space stays whole",
     "a1b22c333d4444e55555f 1234567 x  123",
     {"a", "1", "b", "22", "c", "333", "d", "444", "4", "e", "555", "55", "f", " ", "123", "456", "7", " x", " ",
      " ", "123"}},
    {"LFM2: numbers other than ASCII digits (No, Nd, Nl)", "\u00BD\u00BD\u00BD\u00BD \u0661\u0662\u0663\u0664 \u216B",
     {"\u00BD\u00BD\u00BD", "\u00BD", " ", "\u0661\u0662\u0663", "\u0664", " ", "\u216B"}},
    {"LFM2: CJK and kana are letter runs; a middle dot starts the next word with its mark-less letters",
     "\u6771\u4EAC\u30BF\u30EF\u30FC\u306F333\u30E1\u30FC\u30C8\u30EB\u3067\u3059\u3002\u30AB\u30BF\u30AB\u30CA"
     "\u30FB\u30C6\u30B9\u30C8 \u65E5\u672C\u8A9E\u306Etest \u4E2D\u6587English",
     {"\u6771\u4EAC\u30BF\u30EF\u30FC\u306F", "333", "\u30E1\u30FC\u30C8\u30EB\u3067\u3059", "\u3002\u30AB\u30BF"
      "\u30AB\u30CA", "\u30FB\u30C6\u30B9\u30C8", " \u65E5\u672C\u8A9E\u306Etest", " \u4E2D\u6587English"}},
    {"LFM2: a ZWJ or variation selector stays in the symbol run",
     "\U0001F468\u200D\U0001F469\u200D\U0001F467 \u2705\uFE0F ok\u2705\uFE0F \U0001F44D\U0001F3FD!",
     {"\U0001F468\u200D\U0001F469\u200D\U0001F467", " \u2705\uFE0F", " ok", "\u2705\uFE0F",
      " \U0001F44D\U0001F3FD!"}},
    {"LFM2: a combining mark is a symbol piece where a letter run ends",
     "caf\u00E9 cafe\u0301 \u0301abc \u01C5emal HTTPServer McDONALD's",
     {"caf\u00E9", " cafe", "\u0301", " \u0301", "abc", " \u01C5emal", " HTTPServer", " McDONALD", "'s"}},
    {"LFM2: a mark between letters starts a piece with the letters after it",
     "cafe\u0301x x\u0301 y e\u0301\u0301f a\u0301\u0301",
     {"cafe", "\u0301x", " x", "\u0301", " y", " e", "\u0301\u0301", "f", " a", "\u0301\u0301"}},
    {"LFM2: whitespace other than spaces, control and format characters",
     "a\u3000\u3000b a\u0085b x\u00A0\u00A0y \034 \001\002abc a\u200Bb \u2028z",
     {"a", "\u3000", "\u3000b", " a", "\u0085b", " x", "\u00A0", "\u00A0y", " \034", " \001\002", "abc", " a",
      "\u200Bb", " ", "\u2028z"}},
    {"LFM2: CR LF, tabs, spaces at the end", "line1\r\nline2\r\n\r\n  x\ttab\t\tend   ",
     {"line", "1", "\r\n", "line", "2", "\r\n\r\n", " ", " x", "\ttab", "\t", "\tend", "   "}},
    {"LFM2: code", "def f(x):\n    return x**2  # comment\n\n\n\tpass\n",
     {"def", " f", "(x", "):\n", "   ", " return", " x", "**", "2", " ", " #", " comment", "\n\n\n", "\tpass",
      "\n"}},
    {"LFM2: punctuation and symbol runs",
     "!!! ... ?!?! ---> (((a))) \"q\" #include <vector> \u00A9 50\u00B0C \u20AC100 a/b",
     {"!!!", " ...", " ?!?!", " --->", " (((", "a", ")))", " \"", "q", "\"", " #", "include", " <", "vector", ">",
      " \u00A9", " ", "50", "\u00B0C", " \u20AC", "100", " a", "/b"}},
    {"LFM2: only spaces", "   ", {"   "}},
    {"LFM2: only newlines and a space", "\n \n", {"\n \n"}},
    {"LFM2: a lone quote", "'", {"'"}},
};

bool is_added(const std::string& p) { return std::find(kAdded.begin(), kAdded.end(), p) != kAdded.end(); }

// A token as tokenizer.json writes it: its bytes through GPT-2's byte-to-unicode mapping
std::string byte_level(const std::string& s) {
  static const std::vector<uint32_t> map = [] {
    std::vector<uint32_t> m(256);
    for (int b = 0, n = 0; b < 256; ++b) {
      const bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
      m[b] = keep ? uint32_t(b) : uint32_t(256 + n++);
    }
    return m;
  }();
  std::string out;
  for (unsigned char b : s) {
    const uint32_t c = map[b];  // < 0x800
    if (c < 0x80) out += char(c);
    else out += {char(0xC0 | (c >> 6)), char(0x80 | (c & 0x3F))};
  }
  return out;
}

json flint_pre_tokenizer() {
  auto split = [](const char* re) {
    return json{{"type", "Split"}, {"pattern", {{"Regex", re}}}, {"behavior", "Isolated"}, {"invert", false}};
  };
  const char* words =
      R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
      R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|)"
      R"( ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  const char* cjk = "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+";  // CJK ideographs, hiragana, katakana
  const json bytes = {
      {"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}};
  return {{"type", "Sequence"},
          {"pretokenizers", json::array({split(R"(\p{N}{1,3})"), split(cjk), split(words), bytes})}};
}

json lfm2_pre_tokenizer() {
  auto split = [](const char* re) {
    return json{{"type", "Split"}, {"pattern", {{"Regex", re}}}, {"behavior", "Isolated"}, {"invert", false}};
  };
  const json bytes = {
      {"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}};
  return {{"type", "Sequence"},
          {"pretokenizers", json::array({split(ni::Tokenizer::kLfm2Split), bytes})}};
}

// `j` loaded as a tokenizer.json (from a temporary file)
void load(ni::Tokenizer& tok, const json& j) {
  const char* tmp = getenv("TMPDIR");
  std::string path = std::string(tmp && *tmp ? tmp : "/tmp") + "/kapitsa_tokenizer_test_XXXXXX";
  const int fd = mkstemp(path.data());
  if (fd < 0) {
    perror("mkstemp");
    exit(1);
  }
  close(fd);
  std::ofstream(path) << j.dump();
  tok.load(path);
  unlink(path.c_str());
}

std::string ids_str(const std::vector<int>& ids) {
  std::string s;
  for (int id : ids) s += (s.empty() ? "" : ",") + std::to_string(id);
  return s;
}

// Independent array-based reference: choose a pair, then replace all its
// nonoverlapping occurrences before consulting the ranks again.
std::vector<int> reference_bpe(std::string_view text, const json& vocab, const json& merges) {
  std::unordered_map<uint64_t, std::pair<int, int>> ranks;
  for (int rank = 0; rank < int(merges.size()); ++rank) {
    const std::string a = merges[rank][0], b = merges[rank][1];
    const int ai = vocab[a], bi = vocab[b], id = vocab[a + b];
    ranks[(uint64_t(uint32_t(ai)) << 32) | uint32_t(bi)] = {rank, id};
  }
  std::vector<int> ids;
  for (char c : text) ids.push_back(vocab[std::string(1, c)]);
  for (;;) {
    int best = INT32_MAX, left = -1, right = -1, merged = -1;
    for (size_t i = 1; i < ids.size(); ++i) {
      auto it = ranks.find((uint64_t(uint32_t(ids[i - 1])) << 32) | uint32_t(ids[i]));
      if (it != ranks.end() && it->second.first < best) {
        best = it->second.first;
        left = ids[i - 1]; right = ids[i]; merged = it->second.second;
      }
    }
    if (left < 0) return ids;
    std::vector<int> next;
    for (size_t i = 0; i < ids.size(); ++i) {
      if (i + 1 < ids.size() && ids[i] == left && ids[i + 1] == right) {
        next.push_back(merged);
        ++i;
      } else next.push_back(ids[i]);
    }
    ids.swap(next);
  }
}

void check_bpe_merges() {
  std::mt19937 rng(72519);
  json vocab = json::object(), merges = json::array();
  for (int b = 0; b < 256; ++b) vocab[byte_level(std::string(1, char(b)))] = b;
  std::vector<std::string> parts = {"a", "b", "c", "d"};
  for (int tries = 0; merges.size() < 240 && tries < 100000; ++tries) {
    std::string a = parts[rng() % parts.size()], b = parts[rng() % parts.size()];
    if (a.size() + b.size() > 12 || vocab.contains(a + b)) continue;
    vocab[a + b] = int(vocab.size());
    merges.push_back(json::array({a, b}));
    parts.push_back(a + b);
  }
  // Forward and reversed dependency ranks exercise newly created lower-ranked
  // edges, stale edges, overlapping pairs, and several equal-ranked occurrences.
  for (int order = 0; order < 3; ++order) {
    if (order == 1) std::reverse(merges.begin(), merges.end());
    if (order == 2)
      for (size_t i = merges.size(); i > 1; --i) std::swap(merges[i - 1], merges[rng() % i]);
    json j = {{"added_tokens", json::array()}, {"normalizer", nullptr}, {"pre_tokenizer", nullptr},
              {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}}};
    ni::Tokenizer tok;
    load(tok, j);
    for (int n : {0, 1, 63, 64, 65, 127, 128, 256, 1024, 8192}) {
      for (int pattern = 0; pattern < 4; ++pattern) {
        std::string text;
        for (int i = 0; i < n; ++i)
          text += pattern == 0 ? 'a' : pattern == 1 ? "aab"[i % 3] : "abcd"[rng() % 4];
        auto want = reference_bpe(text, vocab, merges);
        const auto name = "BPE order " + std::to_string(order) + " size " + std::to_string(n) +
                          " pattern " + std::to_string(pattern);
        check(tok.encode(text) == want && tok.encode(text) == want, name);
      }
    }
  }
}

// The pre-tokenizer of lmstudio-community's and mlx-community's Qwen3.8-27B packs lost \p{M} from both letter classes
// (their tokenizer.json: `[^\r\n\p{L}\p{N}]?\p{L}+`, ` ?[^\s\p{L}\p{N}]+[\r\n]*`). Hugging Face tokenizers split Thai and
// Devanagari at every combining mark with it, and the original Qwen3.8-27B file does not (#443). Kapitsa gives a
// Sequence it does not know Qwen's own split, marks included, so loading the lmstudio file must give the original's ids.
// A vocabulary of the 256 bytes plus, for each piece Qwen's split makes, a chain of merges to the whole piece: one cut
// in the wrong place leaves two tokens.
void check_lmstudio_pretokenizer() {
  struct Text {
    const char* what;
    const char* text;
    std::vector<std::string> pieces;  // as Qwen/Qwen3.8-27B's tokenizer.json splits it
  };
  const Text texts[] = {
      {"Thai: vowel and tone marks stay in the word", "\u0E2A\u0E27\u0E31\u0E2A\u0E14\u0E35\u0E04\u0E23\u0E31\u0E1A \u0E20\u0E32\u0E29\u0E32\u0E44\u0E17\u0E22",
       {"\u0E2A\u0E27\u0E31\u0E2A\u0E14\u0E35\u0E04\u0E23\u0E31\u0E1A", " \u0E20\u0E32\u0E29\u0E32\u0E44\u0E17\u0E22"}},
      {"Devanagari: virama and vowel signs stay in the word", "\u0928\u092E\u0938\u094D\u0924\u0947 \u0926\u0941\u0928\u093F\u092F\u093E",
       {"\u0928\u092E\u0938\u094D\u0924\u0947", " \u0926\u0941\u0928\u093F\u092F\u093E"}},
      {"Latin: unchanged", "Hello world, it's 2026.", {"Hello", " world", ",", " it", "'s", " ", "2", "0", "2", "6", "."}},
  };
  const json lmstudio_pre = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array({{{"type", "Split"},
                     {"pattern", {{"Regex", R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
                     {"behavior", "Isolated"},
                     {"invert", false}},
                    {{"type", "ByteLevel"}, {"add_prefix_space", false}, {"trim_offsets", true}, {"use_regex", false}}})}};
  json vocab = json::object(), merges = json::array();
  for (int b = 0; b < 256; ++b) vocab[byte_level(std::string(1, char(b)))] = b;
  // merge `parts` (strings of raw bytes) one after the other, from the left: (p0 p1), (p0p1 p2), ...
  auto chain = [&](const std::vector<std::string>& parts) {
    std::string head = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
      const std::string left = byte_level(head), right = byte_level(parts[i]);
      head += parts[i];
      if (vocab.contains(byte_level(head))) continue;
      vocab[byte_level(head)] = int(vocab.size());
      merges.push_back(json::array({left, right}));
    }
  };
  auto characters = [](const std::string& s) {  // UTF-8 characters
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
      const unsigned char c = s[i];
      const size_t n = c < 0x80 ? 1 : c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
      out.push_back(s.substr(i, n));
      i += n;
    }
    return out;
  };
  for (const Text& t : texts)  // every character first (lower ranks), then each piece from its characters
    for (const std::string& piece : t.pieces)
      for (const std::string& ch : characters(piece)) {
        std::vector<std::string> bytes;
        for (char c : ch) bytes.push_back(std::string(1, c));
        chain(bytes);
      }
  for (const Text& t : texts)
    for (const std::string& piece : t.pieces) chain(characters(piece));
  json j = {{"added_tokens", json::array()}, {"normalizer", {{"type", "NFC"}}}, {"pre_tokenizer", lmstudio_pre},
            {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}}};
  ni::Tokenizer tok;
  load(tok, j);
  for (const Text& t : texts) {
    std::vector<int> want;
    for (const std::string& piece : t.pieces) want.push_back(tok.find(byte_level(piece)));
    const std::vector<int> got = tok.encode(t.text);
    check(got == want, std::string("lmstudio's Qwen3.8-27B pre-tokenizer gives the original's ids: ") + t.what,
          "ids " + ids_str(got) + ", want " + ids_str(want));
  }
}

}  // namespace

int main() {
  // the vocabulary: the 256 bytes, then every run of adjacent pieces (between added tokens) of every case
  json vocab = json::object();
  auto add = [&](const std::string& s) {
    const std::string t = byte_level(s);
    if (!vocab.contains(t)) vocab[t] = int(vocab.size());
  };
  for (int b = 0; b < 256; ++b) add(std::string(1, char(b)));
  for (const Case& c : kCases)
    for (size_t i = 0; i < c.pieces.size(); ++i) {
      std::string run;
      for (size_t k = i; k < c.pieces.size() && !is_added(c.pieces[k]); ++k) add(run += c.pieces[k]);
    }
  json added = json::array();
  for (const auto& a : kAdded)
    added.push_back({{"id", int(vocab.size() + added.size())}, {"content", a}, {"special", false}});
  json j = {{"added_tokens", added}, {"normalizer", nullptr}, {"pre_tokenizer", flint_pre_tokenizer()},
            {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", json::array()}, {"ignore_merges", true}}}};
  ni::Tokenizer tok;
  load(tok, j);
  auto want = [&](const Case& c) {
    std::vector<int> ids;
    for (const auto& p : c.pieces) ids.push_back(tok.find(is_added(p) ? p : byte_level(p)));
    return ids;
  };
  for (const Case& c : kCases) {
    const std::vector<int> got = tok.encode(c.text), w = want(c);
    check(got == w, std::string("Flint split: ") + c.what, "ids " + ids_str(got) + ", want " + ids_str(w));
  }

  // one step different (ByteLevel with its regex): not Flint's split, the default one (digits one by one)
  j["pre_tokenizer"]["pretokenizers"][3]["use_regex"] = true;
  ni::Tokenizer other;
  load(other, j);
  const Case& digits = kCases[3];  // digits in threes
  check(other.encode(digits.text) != want(digits), "a Sequence other than Flint's keeps the default split");

  check_bpe_merges();
  check_lmstudio_pretokenizer();

  // LFM2's split: the same synthetic vocabulary trick, HF tokenizers' pieces as the goldens
  json lvocab = json::object();
  auto ladd = [&](const std::string& s) {
    const std::string t = byte_level(s);
    if (!lvocab.contains(t)) lvocab[t] = int(lvocab.size());
  };
  for (int b = 0; b < 256; ++b) ladd(std::string(1, char(b)));
  for (const Case& c : kLfm2Cases)
    for (size_t i = 0; i < c.pieces.size(); ++i) {
      std::string run;
      for (size_t k = i; k < c.pieces.size() && !is_added(c.pieces[k]); ++k) ladd(run += c.pieces[k]);
    }
  json lj = {{"added_tokens", added}, {"normalizer", nullptr}, {"pre_tokenizer", lfm2_pre_tokenizer()},
             {"model", {{"type", "BPE"}, {"vocab", lvocab}, {"merges", json::array()}, {"ignore_merges", true}}}};
  ni::Tokenizer ltok;
  load(ltok, lj);
  auto lwant = [&](const Case& c) {
    std::vector<int> ids;
    for (const auto& p : c.pieces) ids.push_back(ltok.find(is_added(p) ? p : byte_level(p)));
    return ids;
  };
  for (const Case& c : kLfm2Cases) {
    const std::vector<int> got = ltok.encode(c.text), w = lwant(c);
    check(got == w, c.what, "ids " + ids_str(got) + ", want " + ids_str(w));
  }
  // one step different: not LFM2's split either, the default one
  json l2 = lj;
  l2["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] =
      "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|"
      "\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
  ni::Tokenizer lother;
  load(lother, l2);
  check(lother.encode(kLfm2Cases[3].text) != lwant(kLfm2Cases[3]),
        "a Sequence other than LFM2's keeps the default split");

  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
