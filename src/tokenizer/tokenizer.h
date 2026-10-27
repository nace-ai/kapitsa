// Byte-level BPE tokenizer for HF tokenizer.json files: GPT-2 byte-to-unicode mapping, ranked merges, added
// tokens; Qwen2/3/3.5's pre-tokenizer split with NFC normalization, the Tekken split (Mistral, Nemotron 3.5: no
// normalization, cased letter runs, single digits) with whole-word vocabulary hits first (ignore_merges), or Flint's
// (Recursive: no normalization, digits in threes, CJK and kana runs, cased letter runs with their contractions).
#pragma once
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ni {

class Tokenizer {
 public:
  void load(const std::string& tokenizer_json);
  // Added tokens (e.g. <|im_start|>) in `text` are always recognized, as HF tokenizers do.
  // With `split_index`: when an added token starts exactly at byte `split_at`, *split_index is set
  // to the number of ids before it, which equals encode(text.substr(0, split_at)).size() with those
  // ids as a prefix (text between added tokens is encoded independently); else left unchanged.
  std::vector<int> encode(const std::string& text, size_t split_at = std::string::npos,
                          int* split_index = nullptr) const;
  std::string decode(const std::vector<int>& ids) const;
  // Raw bytes of one token (may be a partial UTF-8 sequence).
  const std::string& token_bytes(int id) const;
  int find(const std::string& token) const; // -1 if absent
  int vocab_size() const { return int(id_to_bytes_.size()); }
  bool is_special(int id) const;
  // An added token (tokenizer.json's added_tokens: <|im_end|>, <think>, <tool_call> ...): matched whole, never text
  bool is_added(int id) const;
  // Flint's pre-tokenizer: the regexes of its tokenizer.json's three Isolated splits (before a ByteLevel without its
  // regex), which select Split::Flint (the GGUF importer writes them)
  static const char* const kFlintSplit[3];

 private:
  void bpe(const std::string& word, std::vector<int>& out) const;
  void encode_plain(const std::string& text, std::vector<int>& out) const;

  std::unordered_map<std::string, int> vocab_;        // byte-level string -> id
  std::vector<std::string> id_to_bytes_;              // id -> raw bytes
  std::unordered_map<uint64_t, std::pair<int, int>> merges_; // (a,b) -> (rank, merged id)
  std::vector<std::pair<std::string, int>> added_;    // content -> id, longest first
  std::vector<bool> special_;
  std::vector<bool> added_id_;
  int byte_id_[256];
  enum class Split { Qwen, Tekken, Flint };  // the pre-tokenizer split, from tokenizer.json's pre_tokenizer
  Split split_ = Split::Qwen;
  bool nfc_ = true;             // NFC-normalize the text first (Qwen; Tekken and Flint have no normalizer)
  bool ignore_merges_ = false;  // a word found whole in the vocabulary is one token (HF BPE ignore_merges)
  mutable std::unordered_map<std::string, std::vector<int>> cache_;
  mutable std::mutex cache_mu_;
};

// `s` with every ill-formed UTF-8 subsequence replaced by U+FFFD (one per maximal subpart, as Python's
// decode(errors="replace")): JSON can carry nothing else.
std::string utf8_valid(const std::string& s);

// Incremental UTF-8-safe detokenizer: feeds token ids, emits only complete, well-formed UTF-8 text. A sequence the
// generation cut off (flush) or the model wrote ill-formed comes out as U+FFFD.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& t) : tok_(t) {}
  std::string push(int id);
  std::string flush();

 private:
  const Tokenizer& tok_;
  std::string pending_;
};

} // namespace ni
