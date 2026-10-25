// Tokenizer check, CPU only, no model folder: Flint's pre-tokenizer split (pretokenize_flint in
// src/tokenizer/tokenizer.cpp) against the pieces Hugging Face tokenizers 0.23.3 makes of the same texts with Flint's
// tokenizer.json (its added tokens split off first, then its pre_tokenizer). The test loads a tokenizer.json with
// Flint's pre_tokenizer, the 256 byte tokens and every run of adjacent pieces as a token, and ignore_merges: with the
// right split each piece is one token, and a piece cut in two or two pieces taken as one give other ids (HF gives one
// token per piece on the same tokenizer.json). load() must take the split only for exactly Flint's Sequence.
// Then SentencePiece-style BPE (Llama's tokenizer.json, Nanbeige4.3's: Metaspace, byte fallback, the decoder's leading
// space stripped) on a small synthetic vocabulary: the ids are those Hugging Face tokenizers 0.23.3 gives for the same
// tokenizer.json (prepend_scheme first and split false as Nanbeige's; always with split, as older Llama exports).
//   tokenizer_test   exit status 1 if any check fails
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

namespace {

int fails = 0, passes = 0;

void check(bool ok, const std::string& name, const std::string& detail = "") {
  printf("%s %s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok || detail.empty() ? "" : ("\n     " + detail).c_str());
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

// A SentencePiece-style tokenizer.json: <unk> <s> </s>, the 256 byte tokens, a few pieces and merges (U+2581 for a space)
json spm_tokenizer(bool split) {
  std::vector<std::string> pieces = {"<unk>", "<s>", "</s>"};
  for (int b = 0; b < 256; ++b) {
    char h[8];
    snprintf(h, sizeof h, "<0x%02X>", b);
    pieces.push_back(h);
  }
  for (const char* p : {"\u2581", "a", "b", "c", "d", "e", "h", "l", "o", "r", "w", "\u4f60", "\u597d", "\u2581h", "el", "lo",
                        "\u2581hel", "\u2581hello", "\u2581w", "or", "\u2581wor", "\u2581world", "\u2581worl", "aa", "\u2581a",
                        "\u2581\u2581", "ab", "\u4f60\u597d"})
    pieces.push_back(p);
  json vocab = json::object();
  for (size_t i = 0; i < pieces.size(); ++i) vocab[pieces[i]] = int(i);
  json merges = json::array();  // (pairs one by one: a braced list of string pairs would make an object)
  for (const auto& [a, b] : std::vector<std::pair<const char*, const char*>>{
           {"\u2581", "h"}, {"e", "l"}, {"l", "o"}, {"\u2581h", "el"}, {"\u2581hel", "lo"}, {"\u2581", "w"}, {"o", "r"},
           {"\u2581w", "or"}, {"\u2581wor", "l"}, {"a", "a"}, {"\u2581", "a"}, {"\u2581", "\u2581"}, {"a", "b"},
           {"\u4f60", "\u597d"}})
    merges.push_back(json::array({a, b}));
  const int n = int(pieces.size());
  const json added = {{{"id", n}, {"content", "<|im_start|>"}, {"special", true}, {"normalized", false}},
                      {{"id", n + 1}, {"content", "<think>"}, {"special", false}, {"normalized", true}}};
  const json decoder = {{"type", "Sequence"},
                        {"decoders", {{{"type", "Replace"}, {"pattern", {{"String", "\u2581"}}}, {"content", " "}},
                                      {{"type", "ByteFallback"}},
                                      {{"type", "Fuse"}},
                                      {{"type", "Strip"}, {"content", " "}, {"start", 1}, {"stop", 0}}}}};
  return {{"added_tokens", added},
          {"normalizer", nullptr},
          {"pre_tokenizer",
           {{"type", "Metaspace"}, {"replacement", "\u2581"}, {"prepend_scheme", split ? "always" : "first"}, {"split", split}}},
          {"decoder", decoder},
          {"model",
           {{"type", "BPE"}, {"unk_token", "<unk>"}, {"continuing_subword_prefix", nullptr}, {"end_of_word_suffix", nullptr},
            {"fuse_unk", true}, {"byte_fallback", true}, {"ignore_merges", false}, {"vocab", vocab}, {"merges", merges}}}};
}

// a text as a check's name shows it (newlines and tabs escaped)
std::string shown(const std::string& t) {
  std::string s;
  for (char c : t) s += c == '\n' ? "\\n" : c == '\t' ? "\\t" : std::string(1, c);
  return "\"" + s + "\"";
}

struct SpmCase {
  const char* text;
  std::vector<int> ids;  // Hugging Face tokenizers 0.23.3's on the same tokenizer.json
  const char* decoded;
};

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

  // SentencePiece-style BPE: the whole text between added tokens one word (split false), U+2581 prepended at the text's
  // start only (not after an added token, not before a U+2581 or a space already there), the lowest-ranked pair merged
  // first and the leftmost on a tie, characters outside the vocabulary as their <0xXX> bytes; decode drops the one
  // leading space
  const SpmCase spm[] = {
      {"hello world", {276, 281, 263}, "hello world"},
      {"<|im_start|>hello\n<think>\n", {287, 265, 273, 274, 13, 288, 13}, "<|im_start|>hello\n<think>\n"},
      {"  aa a", {284, 282, 283}, " aa a"},
      {"aaa", {259, 282, 260}, "aaa"},
      {"\u4f60\u597d\U0001F642!", {259, 286, 243, 162, 156, 133, 36}, "\u4f60\u597d\U0001F642!"},
      {" hello", {276}, "hello"},
      {"ab\taab", {283, 261, 12, 282, 261}, "ab\taab"},
      {"hel lo  w", {275, 259, 274, 259, 277}, "hel lo  w"},
      {"\u2581hello", {276}, "hello"},
  };
  ni::Tokenizer sp;
  load(sp, spm_tokenizer(false));
  for (const SpmCase& c : spm) {
    const std::vector<int> got = sp.encode(c.text);
    check(got == c.ids, "SentencePiece BPE: " + shown(c.text), "ids " + ids_str(got) + ", want " + ids_str(c.ids));
    check(sp.decode(c.ids) == c.decoded, "SentencePiece decode: " + shown(c.text), sp.decode(c.ids));
  }
  check(sp.token_bytes(259) == " " && sp.token_bytes(13) == "\n" && sp.token_bytes(276) == " hello",
        "SentencePiece token bytes: U+2581 as a space, <0x0A> as a newline");
  const SpmCase spm_split[] = {
      {"hello world", {276, 281, 263}, ""},
      {"<|im_start|>hello\n<think> aa", {287, 276, 13, 288, 259, 282}, ""},
      {"  aa a", {259, 259, 282, 283}, ""},
  };
  ni::Tokenizer sps;
  load(sps, spm_tokenizer(true));
  for (const SpmCase& c : spm_split) {
    const std::vector<int> got = sps.encode(c.text);
    check(got == c.ids, "SentencePiece BPE, split, always prepended: " + shown(c.text),
          "ids " + ids_str(got) + ", want " + ids_str(c.ids));
  }

  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
