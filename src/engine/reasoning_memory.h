#pragma once
// The reasoning this server wrote, by the assistant turn that carried it (the turn's content and tool calls).
//
// Many clients send an assistant turn back without its reasoning (no reasoning_content field, or another name for it).
// The template then renders an empty think block where the live state holds the real one, the prompt diverges there
// and everything after the last checkpoint is prefilled again. restore() puts the remembered reasoning back into such
// turns (it never replaces reasoning a client sent), which is the prompt the template is written for inside a tool
// loop. A turn is found by its trimmed content and its tool calls (name and arguments, object keys in sorted order,
// so a client that re-serializes the arguments still matches); the arguments come back as the model wrote them too
// (their key order is part of the rendered text). NI_REASONING_MEMORY=0 turns it off.
#include "engine/grammar.h"
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
    static const bool on = [] {
      const char* v = getenv("NI_REASONING_MEMORY");
      return !(v && v[0] == '0');
    }();
    return on;
  }

  // a finished turn: its reasoning, its content and its calls (arguments as JSON object text)
  void remember(const std::string& content, const std::vector<std::pair<std::string, std::string>>& calls,
                const std::string& reasoning) {
    if (!enabled() || reasoning.empty()) return;
    Entry e;
    e.reasoning = reasoning;
    Calls c;
    for (auto& [name, args] : calls) {
      c.emplace_back(name, parsed(ojson(args)));
      e.args.push_back(c.back().second);
    }
    const std::string text = trimmed(ojson(content));
    if (text.empty() && c.empty()) return;
    e.key = turn_key(text, c);
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

  // the messages' assistant turns that carry no reasoning, given the remembered reasoning (and arguments) of the same
  // turn; how many
  int restore(ojson& messages) {
    if (!enabled() || !messages.is_array()) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    if (lru_.empty()) return 0;
    int restored = 0;
    for (auto& m : messages) {
      if (!m.is_object() || !m.contains("role") || m["role"] != "assistant") continue;
      if (m.contains("reasoning_content") && m["reasoning_content"].is_string() &&
          !m["reasoning_content"].get<std::string>().empty())
        continue;  // the client sent it
      Calls calls;
      if (m.contains("tool_calls") && m["tool_calls"].is_array())
        for (auto& tc : m["tool_calls"])
          if (tc.is_object() && tc.contains("function") && tc["function"].is_object() && tc["function"].contains("name") &&
              tc["function"]["name"].is_string())
            calls.emplace_back(tc["function"]["name"].get<std::string>(),
                               parsed(tc["function"].contains("arguments") ? tc["function"]["arguments"] : ojson::object()));
      const std::string text = trimmed(m.contains("content") ? m["content"] : ojson(""));
      if (text.empty() && calls.empty()) continue;
      auto it = idx_.find(turn_key(text, calls));
      if (it == idx_.end()) continue;
      lru_.splice(lru_.begin(), lru_, it->second);
      const Entry& e = lru_.front();
      m["reasoning_content"] = e.reasoning;
      if (m.contains("tool_calls") && m["tool_calls"].is_array() && m["tool_calls"].size() == e.args.size())
        for (size_t i = 0; i < e.args.size(); ++i) {
          ojson& a = m["tool_calls"][i]["function"]["arguments"];
          a = a.is_string() ? ojson(e.args[i].dump()) : e.args[i];
        }
      ++restored;
    }
    return restored;
  }

 private:
  static constexpr size_t kMaxEntries = 512, kMaxBytes = size_t(16) << 20;

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
  static std::string canon(const ojson& j) {
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
  static std::string turn_key(const std::string& content, const Calls& calls) {
    std::string k = content;
    for (auto& [name, args] : calls) k += "\x01" + name + "\x02" + canon(args);
    return k;
  }

  std::mutex mu_;
  struct Entry {
    std::string key, reasoning;
    std::vector<ojson> args;  // each call's arguments as the model wrote them
    size_t bytes = 0;
  };
  std::list<Entry> lru_;  // most recently used first
  std::unordered_map<std::string, std::list<Entry>::iterator> idx_;
  size_t bytes_ = 0;
};

}  // namespace ni
