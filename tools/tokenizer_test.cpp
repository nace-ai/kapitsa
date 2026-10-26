// Tokenizer check, CPU only, no model folder: Flint's pre-tokenizer split (pretokenize_flint in
// src/tokenizer/tokenizer.cpp) against the pieces Hugging Face tokenizers 0.23.3 makes of the same texts with Flint's
// tokenizer.json (its added tokens split off first, then its pre_tokenizer). The test loads a tokenizer.json with
// Flint's pre_tokenizer, the 256 byte tokens and every run of adjacent pieces as a token, and ignore_merges: with the
// right split each piece is one token, and a piece cut in two or two pieces taken as one give other ids (HF gives one
// token per piece on the same tokenizer.json). load() must take the split only for exactly Flint's Sequence.
// Then the Metaspace BPE with byte fallback (Nanbeige 4.3's SentencePiece-style file) on a small tokenizer.json,
// against the tokens HF tokenizers 0.23.3 gives for the same file and texts.
//   tokenizer_test   exit status 1 if any check fails
//   tokenizer_test path/to/nanbeige/tokenizer.json   also Nanbeige 4.3's own file against HF's ids and decode in
//                    tests/data/nanbeige43_tokenizer.json (skipped, with a message, for a file of another sha256)
#include "engine/output_parser.h"
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <CommonCrypto/CommonDigest.h>
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

// Metaspace with byte fallback: ▁ for a space and in front of the input's first text, raw UTF-8 tokens, a character
// outside the vocabulary as its <0xXX> tokens, merges by rank. Each case's tokens are HF's on the same tokenizer.json.
const Case kMetaspaceCases[] = {
    {"▁ in front of the input only, not after an added token", "hi<think>hi hi", {"▁hi", "<think>", "hi", "▁hi"}},
    {"a run of spaces", "a  b", {"▁a", "▁▁", "b"}},
    {"a leading space is the ▁, none added; the lower rank merges first", "  a", {"▁", "▁a"}},
    {"rank order: b c before a b", "abc", {"▁a", "bc"}},
    {"equal ranks merge leftmost first", "<think>aaa", {"<think>", "aa", "a"}},
    {"a ▁ in the text is a space", "<think>\u2581a", {"<think>", "▁a"}},
    {"a trailing space", "hi ", {"▁hi", "▁"}},
    {"byte fallback for an emoji and a letter outside the vocabulary", "a\U0001F600\u00E9x",
     {"▁a", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>", "<0xC3>", "<0xA9>", "<0x78>"}},
    {"empty", "", {}},
};

void metaspace() {
  std::vector<std::string> toks = {"<unk>", "<s>", "</s>"};
  for (int b = 0; b < 256; ++b) {
    char name[8];
    snprintf(name, sizeof name, "<0x%02X>", b);
    toks.push_back(name);
  }
  for (const char* t : {"\u2581", "a", "b", "c", "h", "i"}) toks.push_back(t);
  const std::vector<std::pair<std::string, std::string>> merges = {
      {"b", "c"}, {"a", "b"}, {"\u2581", "a"}, {"h", "i"}, {"\u2581", "hi"}, {"\u2581", "\u2581"}, {"a", "a"},
      {"ab", "c"}};
  json vocab = json::object(), m = json::array();
  for (const auto& [a, b] : merges) {
    if (std::find(toks.begin(), toks.end(), a + b) == toks.end()) toks.push_back(a + b);
    m.push_back({a, b});
  }
  for (size_t i = 0; i < toks.size(); ++i) vocab[toks[i]] = int(i);
  json added = json::array();
  for (int i = 0; i < 3; ++i) added.push_back({{"id", i}, {"content", toks[i]}, {"special", true}});
  added.push_back({{"id", int(toks.size())}, {"content", "<think>"}, {"special", false}});
  const json j = {
      {"added_tokens", added}, {"normalizer", nullptr},
      {"pre_tokenizer",
       {{"type", "Metaspace"}, {"replacement", "\u2581"}, {"prepend_scheme", "first"}, {"split", false}}},
      {"model", {{"type", "BPE"}, {"unk_token", "<unk>"}, {"fuse_unk", true}, {"byte_fallback", true},
                 {"vocab", vocab}, {"merges", m}}}};
  ni::Tokenizer tok;
  load(tok, j);
  for (const Case& c : kMetaspaceCases) {
    std::vector<int> want;
    for (const auto& p : c.pieces) want.push_back(tok.find(p));
    const std::vector<int> got = tok.encode(c.text);
    check(got == want, std::string("Metaspace: ") + c.what, "ids " + ids_str(got) + ", want " + ids_str(want));
    // HF's decode: a ▁ is a space, and the text loses one leading space (the ▁ put in front, or the text's own when
    // none was); a StreamDecoder that starts the text the same, one that starts within a text keeps every space
    std::string raw = c.text;
    for (size_t k; (k = raw.find("\u2581")) != std::string::npos;) raw.replace(k, 3, " ");
    if (!raw.empty() && raw[0] != ' ' && raw[0] != '<') raw.insert(0, " ");
    const std::string back = !raw.empty() && raw[0] == ' ' ? raw.substr(1) : raw;
    check(tok.decode(got) == back, std::string("Metaspace decode: ") + c.what,
          "\"" + tok.decode(got) + "\", want \"" + back + "\"");
    ni::StreamDecoder starts(tok, true), within(tok);
    std::string s, w;
    for (int id : got) s += starts.push(id), w += within.push(id);
    s += starts.flush(), w += within.flush();
    check(s == back && w == raw, std::string("Metaspace StreamDecoder: ") + c.what,
          "\"" + s + "\" and \"" + w + "\", want \"" + back + "\" and \"" + raw + "\"");
  }
  // a reply: its own leading space goes, as in HF's decode of the reply; after a tag that opens it, a space stays
  // (HF's decode starts with the tag, so it strips nothing)
  const int think = tok.find("<think>");
  auto reply = [&](const std::vector<int>& ids) {
    ni::OutputParser p(tok, think, -1, -1, -1, false, false, ni::ojson::object());
    ni::ChatDelta all;
    for (int id : ids) {
      const ni::ChatDelta d = p.push(id);
      all.reasoning += d.reasoning, all.content += d.content;
    }
    const ni::ChatDelta d = p.finish();
    return all.reasoning + "|" + all.content + d.reasoning + d.content;
  };
  const std::string r1 = reply(tok.encode(" hi hi")), r2 = reply(tok.encode("<think> hi"));
  check(r1 == "|hi hi" && r2 == " hi|", "Metaspace reply: the leading space goes, not one after <think>",
        "\"" + r1 + "\" and \"" + r2 + "\"");
}

std::string sha256_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  const std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.data(), CC_LONG(data.size()), d);
  char hex[2 * CC_SHA256_DIGEST_LENGTH + 1];
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) snprintf(hex + 2 * i, 3, "%02x", d[i]);
  return hex;
}

// Nanbeige 4.3's own tokenizer.json against tests/data/nanbeige43_tokenizer.json (tests/nanbeige43_tokenizer_fixture.py
// writes it): HF's ids and decode of each text, against encode, decode and a StreamDecoder fed one token at a time.
void nanbeige43(const std::string& path) {
  const std::string src = __FILE__;
  const std::string fixture = src.substr(0, src.rfind('/')) + "/../tests/data/nanbeige43_tokenizer.json";
  std::ifstream f(fixture);
  if (!f) return check(false, "Nanbeige 4.3 fixture", "cannot read " + fixture);
  const json j = json::parse(f);
  const std::string sha = sha256_file(path), want_sha = j["tokenizer_sha256"];
  if (sha.empty()) return check(false, "Nanbeige 4.3 tokenizer.json", "cannot read " + path);
  if (sha != want_sha) {
    printf("SKIP Nanbeige 4.3 fixture: %s has sha256 %s, the fixture is of %s\n", path.c_str(), sha.c_str(),
           want_sha.c_str());
    return;
  }
  ni::Tokenizer tok;
  tok.load(path);
  auto q = [](const std::string& s) { return json(s).dump(-1, ' ', false, json::error_handler_t::replace); };
  auto add = [](std::string& to, const std::string& s) { to += (to.empty() ? "" : "\n     ") + s; };
  int n = 0;
  std::string enc_bad, dec_bad, stream_bad;
  for (const auto& c : j["cases"]) {
    const std::string text = c["text"], back = c["decode"];
    const std::vector<int> ids = c["ids"], got = tok.encode(text);
    ni::StreamDecoder sd(tok, true);
    std::string s;
    for (int id : ids) s += sd.push(id);
    s += sd.flush();
    const std::string at = "case " + std::to_string(n++) + " " + q(text);
    if (got != ids) add(enc_bad, at + ": ids " + ids_str(got) + ", want " + ids_str(ids));
    if (tok.decode(ids) != back) add(dec_bad, at + ": " + q(tok.decode(ids)) + ", want " + q(back));
    if (s != back) add(stream_bad, at + ": " + q(s) + ", want " + q(back));
  }
  const std::string version = j["tokenizers_version"];
  const std::string of = " on " + std::to_string(n) + " texts (HF tokenizers " + version + ")";
  check(enc_bad.empty(), "Nanbeige 4.3 encode gives HF's ids" + of, enc_bad);
  check(dec_bad.empty(), "Nanbeige 4.3 decode gives HF's text" + of, dec_bad);
  check(stream_bad.empty(), "Nanbeige 4.3 StreamDecoder, one token at a time, gives HF's text" + of, stream_bad);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && argv[1][0] == '-')) {
    fprintf(stderr, "usage: tokenizer_test [path/to/nanbeige/tokenizer.json]\n");
    return 2;
  }
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

  // byte-level: every space is a token's own, none is stripped (a StreamDecoder that starts the text keeps it too)
  const std::vector<int> spaced = tok.encode(" world, it's");
  ni::StreamDecoder starts(tok, true);
  std::string s;
  for (int id : spaced) s += starts.push(id);
  s += starts.flush();
  check(tok.decode(spaced) == " world, it's" && s == " world, it's", "byte-level decode keeps a leading space",
        "\"" + tok.decode(spaced) + "\" and \"" + s + "\"");

  metaspace();
  if (argc == 2) nanbeige43(argv[1]);

  printf("%d passed, %d failed\n", passes, fails);
  return fails ? 1 : 0;
}
