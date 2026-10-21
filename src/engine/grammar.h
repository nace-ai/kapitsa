// Structured output: the grammars a reply can be held to (a JSON Schema, any JSON object, Qwen's XML tool calls with
// their arguments following each tool's schema) and the token masks that keep generation inside one. No dependency
// beyond nlohmann::json: a JSON-Schema subset compiles into a graph of value nodes, a pushdown matcher walks it byte by
// byte (and by the tool-call tokens), and the vocabulary's bytes, in a trie, are walked through the matcher's state to
// give the tokens that may come next.
//
// Speed: a position inside free text (a JSON string, a raw tool argument, the text before a call) allows nearly every
// token. There the vocabulary is not walked: for the lexical state the position is in, the tokens that stay inside the
// text are precomputed once as a set (TokenTable's cache), and only the few that leave it (a closing quote, the end of
// a tool argument) are walked through the state after it. Everywhere else the walk prunes at the first byte that cannot
// follow.
#pragma once
#include <nlohmann/json.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace ni {

class Tokenizer;
using ojson = nlohmann::ordered_json;

// A schema, format or tool choice the server cannot hold a reply to (the client's to fix: HTTP 400)
struct GrammarError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// What a request asks of its reply after the reasoning (the API translations fill it in)
struct OutputSpec {
  enum class Format : uint8_t { kText, kJsonObject, kJsonSchema };
  Format format = Format::kText;
  ojson schema;          // kJsonSchema
  bool strict = false;   // kJsonSchema: objects take no keys their schema does not declare unless it says they may
  enum class Tools : uint8_t { kAuto, kRequired, kNamed };
  Tools tools = Tools::kAuto;
  std::string tool;      // kNamed: the tool's name as the prompt declares it
  bool parallel = true;  // a reply may make more than one call
  std::vector<std::string> strict_tools;  // tools whose arguments follow their schema whatever the choice
  std::vector<std::string> allowed;       // the tools a call may name (Responses' allowed_tools); empty: every one
};

class Grammar;
// The grammar a reply must follow for `spec`, with the request's tools (ChatRequest::tools: OpenAI functions as the
// prompt declares them), or null when nothing constrains it (plain text, tool_choice auto without strict tools).
// tool_choice required or named: the reply is one call or more (one without `parallel`), each to an offered tool (the
// named one), with arguments following its parameters' schema. Auto with strict tools: text, or calls; a strict tool's
// arguments follow its schema, another's are free. A format with tools and tool_choice auto: the JSON, or calls.
// Throws GrammarError.
std::shared_ptr<const Grammar> output_grammar(const OutputSpec& spec, const ojson& tools);
// A JSON value following `schema` (strict: as OutputSpec::strict); tests and tools
std::shared_ptr<const Grammar> json_grammar(const ojson& schema, bool strict = false);
// One line on what a grammar holds (logs, tests)
std::string grammar_summary(const Grammar& g);
// Whether the grammar has tool calls (the model must have <tool_call> / </tool_call> tokens)
bool grammar_calls(const Grammar& g);

// One position in a grammar: the parses the reply so far leaves open (several only under anyOf and the like). Plain
// data: a copy is a fork, as each verify row of a draft takes one.
struct GrammarState {
  struct Frame {
    uint32_t node = 0;  // grammar node (a tool's index for a call)
    uint8_t kind = 0, st = 0;
    uint16_t n = 0;     // whitespace run, digit counts, the property being written
    uint32_t a = 0;     // trie node, literal offset, lexical state, item count, property index
    uint64_t x = 0;     // properties seen (an object's first 64), a number's magnitude, a key's name trie node
    bool operator==(const Frame& o) const {
      return node == o.node && kind == o.kind && st == o.st && n == o.n && a == o.a && x == o.x;
    }
  };
  std::vector<std::vector<Frame>> threads;
  bool dead() const { return threads.empty(); }
};

// The vocabulary as the matcher reads it: the text tokens' bytes in a trie, the added tokens as atoms (<tool_call> and
// </tool_call> are the calls' delimiters, the end-of-turn ids end a reply, any other is never allowed), and the cache
// of free-text token sets. Built once per tokenizer (~0.1 s for Qwen's 248K tokens); thread-safe.
class TokenTable {
 public:
  // model_vocab: the logits' width (mask bits; tokens past the tokenizer's are never allowed); eos: the ids that end a
  // reply; tc_open / tc_close: the tool-call tokens (-1: none)
  TokenTable(const Tokenizer& tok, int model_vocab, const std::vector<int>& eos, int tc_open, int tc_close);
  ~TokenTable();
  int vocab() const { return vocab_; }
  int words() const { return (vocab_ + 31) / 32; }  // uint32 words per mask row
  int tool_open() const { return tc_open_; }
  int tool_close() const { return tc_close_; }
  const std::vector<int>& eos() const { return eos_; }
  size_t trie_nodes() const { return byte_.size(); }
  // Builds the cache entries most replies need first (a JSON string's content, a raw tool argument): ~20 ms, once
  void warm() const;

  struct Bulk;  // the tokens that stay inside free text from one lexical state, and those that leave it

 private:
  friend class GrammarMatcher;
  const Bulk& bulk(uint32_t node, int lexer, int sigma) const;
  int vocab_ = 0;
  int ntok_ = 0;                    // tokenizer ids
  std::vector<std::string> bytes_;  // text tokens' bytes (empty for atoms)
  std::vector<int16_t> sym_;        // -1 text, else the atom's symbol
  std::vector<int> eos_;
  int tc_open_ = -1, tc_close_ = -1;
  // the trie, in preorder: node 0 is the root; node i's subtree is [i, end_[i])
  std::vector<uint8_t> byte_, depth_;
  std::vector<int32_t> tok_;
  std::vector<uint32_t> end_;
  int max_depth_ = 0;
  mutable std::mutex mu_;
  mutable std::unordered_map<uint64_t, std::unique_ptr<Bulk>> bulk_;
};

// A grammar applied to a vocabulary. States are plain values the caller keeps (one per reply, plus copies for drafts).
// Thread-safe (masks use thread-local scratch).
class GrammarMatcher {
 public:
  // table may be null for byte-level use (advance_text, tests)
  GrammarMatcher(std::shared_ptr<const Grammar> g, std::shared_ptr<const TokenTable> table);
  GrammarState start() const;
  // Feeds a token: false (the state is then dead) when the grammar does not allow it. An end-of-turn id leaves the
  // state as it is, allowed where the reply may end.
  bool advance(GrammarState& s, int token) const;
  // Byte-level (tests): text, and the tool-call delimiters
  bool advance_text(GrammarState& s, const std::string& text) const;
  bool advance_tool_open(GrammarState& s) const;
  bool advance_tool_close(GrammarState& s) const;
  // The reply may end here (an end-of-turn token is allowed)
  bool accepting(const GrammarState& s) const;
  // The tokens allowed next: bit t of words[t / 32] (TokenTable::words() words; the rest of the vocabulary 0)
  void fill_mask(const GrammarState& s, uint32_t* words) const;
  const Grammar& grammar() const { return *g_; }
  const TokenTable* table() const { return t_.get(); }

 private:
  void walk_thread(const std::vector<GrammarState::Frame>& th, uint32_t* words) const;
  void apply_bulk(uint32_t node, const std::vector<GrammarState::Frame>& th, int lexer, int sigma, uint32_t* words,
                  int64_t lo, int64_t hi) const;
  std::shared_ptr<const Grammar> g_;
  std::shared_ptr<const TokenTable> t_;
};

}  // namespace ni
