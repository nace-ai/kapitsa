#pragma once
// The reasoning this server wrote, by the assistant turn that carried it.
//
// Many clients send an assistant turn back without its reasoning (no reasoning_content field, or another name for it).
// The template then renders an empty think block where the live state holds the real one, the prompt diverges there
// and everything after the last checkpoint is prefilled again. restore() puts the remembered reasoning back into such
// turns (it never replaces reasoning a client sent), which is the prompt the template is written for inside a tool
// loop.
//
// A turn is found by the conversation before it plus its own content and tool calls: a digest of every message ahead
// of it (role, text, tool calls by name and arguments with object keys in sorted order; not reasoning, not tool-call
// ids), then the turn's trimmed content and its calls. Two sessions that end a step with the same call (or both say
// "Done.") have different histories and so keep their own reasoning; a client that re-serializes the arguments, drops
// the ids or the reasoning still matches. The arguments come back as the model wrote them too (their key order is
// part of the rendered text). A client that rewrites earlier messages (trims old tool results) gets no reasoning put
// back: no match is the old behaviour. NI_REASONING_MEMORY=0 turns it off.
#include "engine/grammar.h"
#include <cstdint>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ni {

class ReasoningMemory {
 public:
  using Calls = std::vector<std::pair<std::string, ojson>>;  // name, arguments (an object, or a string that is not JSON)

  static bool enabled() {
    const char* v = getenv("NI_REASONING_MEMORY");
    return !(v && v[0] == '0');
  }

  // a finished turn of the conversation `before` (the request's messages): its reasoning, content and calls (arguments
  // as JSON object text)
  void remember(const ojson& before, const std::string& content,
                const std::vector<std::pair<std::string, std::string>>& calls, const std::string& reasoning) {
    if (!enabled() || reasoning.empty() || !before.is_array()) return;
    Entry e;
    e.reasoning = reasoning;
    Calls c;
    for (auto& [name, args] : calls) {
      c.emplace_back(name, parsed(ojson(args)));
      if (too_deep(c.back().second)) return;
      e.args.push_back(c.back().second);
    }
    const std::string text = trimmed(ojson(content));
    if (text.empty() && c.empty()) return;
    Hash h;
    for (const ojson& m : before) h.message(m);
    if (h.skip) return;
    e.key = h.turn(text, c);
    if (e.key.empty()) return;
    e.bytes = e.key.size() + reasoning.size();
    for (auto& a : e.args) e.bytes += a.dump().size();
    std::lock_guard<std::mutex> lk(mu_);
    if (auto it = idx_.find(e.key); it != idx_.end()) {
      bytes_ -= it->second->bytes;
      lru_.erase(it->second);
      idx_.erase(it);
    }
    bytes_ += e.bytes;
    lru_.push_front(std::move(e));
    idx_[lru_.front().key] = lru_.begin();
    while (!lru_.empty() && (lru_.size() > kMaxEntries || bytes_ > kMaxBytes)) {
      bytes_ -= lru_.back().bytes;
      idx_.erase(lru_.back().key);
      lru_.pop_back();
    }
  }

  // the messages' assistant turns that carry no reasoning, given the reasoning (and arguments) remembered for the same
  // turn after the same conversation; how many
  int restore(ojson& messages) {
    if (!enabled() || !messages.is_array()) return 0;
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (lru_.empty()) return 0;
    }
    int restored = 0;
    Hash h;
    for (auto& m : messages) {
      if (!m.is_object()) continue;
      const bool assistant = m.contains("role") && m["role"] == "assistant";
      const bool has_reasoning = m.contains("reasoning_content") && m["reasoning_content"].is_string() &&
                                 !m["reasoning_content"].get<std::string>().empty();  // the client sent it
      if (assistant && !has_reasoning && !h.skip) {
        Calls calls;
        bool deep = false;
        if (m.contains("tool_calls") && m["tool_calls"].is_array())
          for (auto& tc : m["tool_calls"])
            if (tc.is_object() && tc.contains("function") && tc["function"].is_object() && tc["function"].contains("name") &&
                tc["function"]["name"].is_string()) {
              calls.emplace_back(tc["function"]["name"].get<std::string>(),
                                 parsed(tc["function"].contains("arguments") ? tc["function"]["arguments"] : ojson::object()));
              deep = deep || too_deep(calls.back().second);
            }
        const std::string text = trimmed(m.contains("content") ? m["content"] : ojson(""));
        if (!deep && !(text.empty() && calls.empty())) {
          const std::string key = h.turn(text, calls);
          std::string reasoning;
          std::vector<ojson> args;
          {
            std::lock_guard<std::mutex> lk(mu_);
            if (auto it = idx_.find(key); it != idx_.end()) {
              lru_.splice(lru_.begin(), lru_, it->second);
              reasoning = lru_.front().reasoning;
              args = lru_.front().args;
            }
          }
          if (!reasoning.empty()) {
            m["reasoning_content"] = reasoning;
            if (m.contains("tool_calls") && m["tool_calls"].is_array() && m["tool_calls"].size() == args.size())
              for (size_t i = 0; i < args.size(); ++i) {
                if (!m["tool_calls"][i].is_object() || !m["tool_calls"][i].contains("function") ||
                    !m["tool_calls"][i]["function"].is_object())
                  continue;
                ojson& a = m["tool_calls"][i]["function"]["arguments"];
                a = a.is_string() ? ojson(args[i].dump()) : args[i];
              }
            ++restored;
          }
        }
      }
      h.message(m);
    }
    return restored;
  }

  size_t size() {
    std::lock_guard<std::mutex> lk(mu_);
    return lru_.size();
  }

  static constexpr size_t kMaxEntries = 512, kMaxBytes = size_t(16) << 20;

 private:
  static constexpr int kMaxDepth = 64;  // arguments nested deeper are never matched (canon recurses over the depth)

  static std::string trimmed(const ojson& c) {
    std::string s;
    if (c.is_string()) s = c.get<std::string>();
    else if (c.is_array())
      for (auto& p : c)
        if (p.is_object() && p.contains("text") && p["text"].is_string()) s += p["text"].get<std::string>();
    const size_t a = s.find_first_not_of(" \n\r\t"), b = s.find_last_not_of(" \n\r\t");
    return a == std::string::npos ? "" : s.substr(a, b + 1 - a);
  }
  static ojson parsed(const ojson& a) {
    if (!a.is_string()) return a;
    try {
      return ojson::parse(a.get<std::string>());
    } catch (...) {
      return a;
    }
  }
  static bool too_deep(const ojson& j, int d = 0) {
    if (d > kMaxDepth) return true;
    if (j.is_object() || j.is_array())
      for (auto& v : j)
        if (too_deep(v, d + 1)) return true;
    return false;
  }
  static std::string canon(const ojson& j) {  // only for values too_deep() accepted
    if (j.is_object()) {
      std::map<std::string, std::string> kv;
      for (auto it = j.begin(); it != j.end(); ++it) kv[it.key()] = canon(it.value());
      std::string o = "{";
      for (auto& [k, v] : kv) o += ojson(k).dump() + ":" + v + ",";
      return o + "}";
    }
    if (j.is_array()) {
      std::string o = "[";
      for (auto& v : j) o += canon(v) + ",";
      return o + "]";
    }
    return j.dump();
  }

  // 128 bits (two FNV-1a runs), each field written with its length, over the conversation so far
  struct Hash {
    uint64_t a = 1469598103934665603ull, b = 0x9e3779b97f4a7c15ull;
    bool skip = false;  // a message nested too deep to digest: nothing after it is matched
    void bytes(const std::string& s) {
      uint64_t n = s.size();
      for (int i = 0; i < 8; ++i) put(uint8_t(n >> (8 * i)));
      for (unsigned char ch : s) put(ch);
    }
    void put(uint8_t v) {
      a = (a ^ v) * 1099511628211ull;
      b = (b ^ (v + 0x6d)) * 0x100000001b3ull + (b >> 29);
    }
    void message(const ojson& m) {
      if (!m.is_object()) return;
      bytes(m.contains("role") && m["role"].is_string() ? m["role"].get<std::string>() : "");
      bytes(trimmed(m.contains("content") ? m["content"] : ojson("")));
      if (m.contains("tool_calls") && m["tool_calls"].is_array())
        for (auto& tc : m["tool_calls"]) {
          if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object()) continue;
          const ojson args = parsed(tc["function"].contains("arguments") ? tc["function"]["arguments"] : ojson::object());
          if (too_deep(args)) {
            skip = true;
            return;
          }
          bytes(tc["function"].contains("name") && tc["function"]["name"].is_string() ? tc["function"]["name"].get<std::string>() : "");
          bytes(canon(args));
        }
      put(0xff);  // message boundary
    }
    // the key of a turn after the messages digested so far
    std::string turn(const std::string& content, const Calls& calls) const {
      Hash t = *this;
      t.bytes("turn");
      t.bytes(content);
      for (auto& [name, args] : calls) {
        t.bytes(name);
        t.bytes(canon(args));
      }
      char buf[40];
      snprintf(buf, sizeof buf, "%016llx%016llx", (unsigned long long)t.a, (unsigned long long)t.b);
      return buf;
    }
  };

  struct Entry {
    std::string key, reasoning;
    std::vector<ojson> args;  // each call's arguments as the model wrote them
    size_t bytes = 0;
  };

  std::mutex mu_;
  std::list<Entry> lru_;  // most recently used first
  std::unordered_map<std::string, std::list<Entry>::iterator> idx_;
  size_t bytes_ = 0;
};

}  // namespace ni
